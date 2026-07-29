#!/usr/bin/env python3
"""Body-fitted circular surface-inlet Hex8 meshes for the bearing film."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Sequence

import gmsh
import meshio
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "out" / ".matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshing import structured_hex_no_port as base
from meshing import structured_hex_surface_inlet as staircase
from meshing.gmsh_brep_preflight import (
    atomic_replace_directory,
    make_staging_directory,
    relative_error,
    require,
    sha256_file,
)
from meshing.layered_prism_central_feed import symmetric_gap_coordinates
from meshing.structured_hex_surface_inlet import InletSpec, load_inlet_spec
from interchange.cgns_compat import sanitize_gmsh_cgns
from interchange import fluent_legacy_msh


SI_PER_MM = base.SI_PER_MM
PHYSICAL_IDS = dict(staircase.PHYSICAL_IDS)
SURFACE_ENTITIES = dict(staircase.SURFACE_ENTITIES)
VOLUME_ENTITY = staircase.VOLUME_ENTITY
PATCH_NAMES = tuple(SURFACE_ENTITIES)
GeometryMode = Literal["inscribed", "equal-area"]
TopologyName = Literal["tensor-warp", "ogrid"]

CELL_FIELD_NAMES = (
    "block_id",
    "master_quad_index",
    "gap_index",
    "theta_deg",
    "axial_coordinate_mm",
    "gap_um",
    "nominal_geometry",
    "research_variant",
)
QUALITY_FIELD_NAMES = (
    "signed_volume_m3",
    "gauss_volume_m3",
    "gauss_min_det",
    "aspect_ratio",
    "max_nonorthogonality_deg",
    "max_skewness",
    "min_face_pyramid_m3",
    "cell_volume_m3",
    "minSICN",
    "minDetJac",
)


class BodyFittedError(RuntimeError):
    """An expected body-fitted construction, validation, or export failure."""


class BodyFittedRunError(BodyFittedError):
    """A failed atomic run with its serializable failure report."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _readonly(array: np.ndarray, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=dtype)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class MasterMesh:
    """Generic body-fitted bore-surface topology in millimetres."""

    points_mm: np.ndarray
    node_tags: np.ndarray
    quads: np.ndarray
    quad_tags: np.ndarray
    pressure_mask: np.ndarray
    block_id: np.ndarray
    rim_node_tags: np.ndarray
    control_loop_node_tags: np.ndarray
    fixed_node_tags: np.ndarray
    unchanged_node_tags: np.ndarray
    unchanged_points_mm: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for name, dtype in (
            ("points_mm", np.float64),
            ("node_tags", np.uint64),
            ("quads", np.uint64),
            ("quad_tags", np.uint64),
            ("pressure_mask", np.bool_),
            ("block_id", np.int32),
            ("rim_node_tags", np.uint64),
            ("control_loop_node_tags", np.uint64),
            ("fixed_node_tags", np.uint64),
            ("unchanged_node_tags", np.uint64),
            ("unchanged_points_mm", np.float64),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))


@dataclass(frozen=True)
class BodyFittedMesh:
    """Generic swept Hex8 mesh with topology-derived Quad4 boundaries."""

    points_m: np.ndarray
    node_tags: np.ndarray
    hexes: np.ndarray
    cell_tags: np.ndarray
    boundary_quads: dict[str, np.ndarray]
    cell_centres_m: np.ndarray
    cell_fields: dict[str, np.ndarray]
    cell_metrics: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boundary_quads",
            MappingProxyType(
                {
                    name: _readonly(values, np.uint64)
                    for name, values in self.boundary_quads.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "cell_fields",
            MappingProxyType(
                {name: _readonly(values) for name, values in self.cell_fields.items()}
            ),
        )
        object.__setattr__(
            self,
            "cell_metrics",
            MappingProxyType(
                {
                    name: _readonly(values, np.float64)
                    for name, values in self.cell_metrics.items()
                }
            ),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for name, dtype in (
            ("points_m", np.float64),
            ("node_tags", np.uint64),
            ("hexes", np.uint64),
            ("cell_tags", np.uint64),
            ("cell_centres_m", np.float64),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))


@dataclass(frozen=True)
class BodyFittedCaseInputs:
    params: Path
    outdir: Path
    case_name: str
    topology: TopologyName
    geometry_mode: GeometryMode
    q: int
    inner_layers: int = 2
    outer_layers: int = 4
    n_theta: int = 256
    n_axial: int = 96
    n_gap: int = 12
    gap_inflation_ratio: float = 1.0
    quality_optimized_ogrid: bool = False
    control_radius_factor: float = 1.4
    control_square_blend: float = 0.0
    central_corner_radius_factor: float = 0.9
    smoothing_iterations: int = 12
    smoothing_damping: float = 0.25
    smoothing_fixed_nodes: Literal[
        "all-interfaces", "background-and-rim"
    ] = "all-interfaces"
    minimum_fluent_orthogonal_quality: float | None = None
    openfoam: Literal["auto", "required", "skip"] = "skip"
    ansys: Literal["auto", "required", "skip"] = "required"
    context_step: Path | None = None


def _effective_radius(radius_mm: float, rim_segments: int, mode: GeometryMode) -> float:
    if rim_segments < 8 or rim_segments % 4:
        raise BodyFittedError("rim_segments must be a multiple of four and at least eight")
    if not math.isfinite(radius_mm) or radius_mm <= 0.0:
        raise BodyFittedError("the inlet radius must be finite and positive")
    if mode == "inscribed":
        return radius_mm
    if mode == "equal-area":
        return radius_mm * math.sqrt(
            2.0 * math.pi / (rim_segments * math.sin(2.0 * math.pi / rim_segments))
        )
    raise BodyFittedError(f"unknown geometry mode {mode!r}")


def analytic_rim_nodes(
    params: base.BearingParams,
    inlet: InletSpec,
    rim_segments: int,
    geometry_mode: GeometryMode = "inscribed",
    *,
    start_angle_rad: float = 0.0,
    clockwise: bool = False,
) -> np.ndarray:
    """Return exact drilling-cylinder/conical-bore intersection nodes in mm."""
    radius = _effective_radius(inlet.radius_mm, rim_segments, geometry_mode)
    direction = -1.0 if clockwise else 1.0
    alpha = start_angle_rad + direction * (
        2.0 * math.pi * np.arange(rim_segments, dtype=np.float64) / rim_segments
    )
    x = radius * np.cos(alpha)
    z = inlet.axial_position_mm + radius * np.sin(alpha)
    bore = np.asarray(params.bore_radius_mm(z), dtype=np.float64)
    radicand = bore**2 - x**2
    if np.any(radicand <= 0.0):
        raise BodyFittedError("the analytic inlet rim lies outside the conical bore")
    return np.column_stack((x, np.sqrt(radicand), z))


def rim_geometry_diagnostics(
    inlet: InletSpec, rim_segments: int, geometry_mode: GeometryMode
) -> dict[str, Any]:
    radius = _effective_radius(inlet.radius_mm, rim_segments, geometry_mode)
    polygon_area = (
        0.5 * rim_segments * radius**2 * math.sin(2.0 * math.pi / rim_segments)
    )
    nominal_area = math.pi * inlet.radius_mm**2
    return {
        "geometry_mode": geometry_mode,
        "rim_segments": rim_segments,
        "effective_radius_mm": radius,
        "nominal_radius_mm": inlet.radius_mm,
        "radial_bias_mm": radius - inlet.radius_mm,
        "radial_bias_relative": radius / inlet.radius_mm - 1.0,
        "polygon_area_mm2": polygon_area,
        "nominal_circle_area_mm2": nominal_area,
        "area_correction_mm2": polygon_area
        - 0.5
        * rim_segments
        * inlet.radius_mm**2
        * math.sin(2.0 * math.pi / rim_segments),
        "polygon_area_relative_error": relative_error(polygon_area, nominal_area),
        "polygon_perimeter_mm": 2.0
        * rim_segments
        * radius
        * math.sin(math.pi / rim_segments),
        "chord_sagitta_mm": radius * (1.0 - math.cos(math.pi / rim_segments)),
        "rim_node_geometry": "analytic drilling-cylinder/conical-bore intersection",
        "quad4_rim_edge_geometry": "straight circular chords",
        "nominal_geometry": geometry_mode == "inscribed",
        "research_variant": geometry_mode == "equal-area",
        "production_handoff_default": geometry_mode == "inscribed",
    }


def _base_master(
    params: base.BearingParams, n_theta: int, n_axial: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_theta < 16 or n_theta % 2 or n_axial < 4:
        raise BodyFittedError("the master grid requires even n_theta>=16 and n_axial>=4")
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=np.float64) / n_theta
    z = params.length_mm * np.arange(n_axial + 1, dtype=np.float64) / n_axial
    radius = np.asarray(params.bore_radius_mm(z), dtype=np.float64)
    grid = np.empty((n_axial + 1, n_theta, 3), dtype=np.float64)
    grid[..., 0] = radius[:, None] * np.sin(theta)[None, :]
    grid[..., 1] = -radius[:, None] * np.cos(theta)[None, :]
    grid[..., 2] = z[:, None]
    points = np.ascontiguousarray(grid.reshape(-1, 3))
    tags = np.arange(1, len(points) + 1, dtype=np.uint64)

    j, k = np.meshgrid(
        np.arange(n_theta, dtype=np.int64),
        np.arange(n_axial, dtype=np.int64),
        indexing="ij",
    )

    def tag(j_value: np.ndarray, k_value: np.ndarray) -> np.ndarray:
        return 1 + (k_value * n_theta + j_value)

    quads = np.column_stack(
        (
            tag(j, k).ravel(),
            tag((j + 1) % n_theta, k).ravel(),
            tag((j + 1) % n_theta, k + 1).ravel(),
            tag(j, k + 1).ravel(),
        )
    ).astype(np.uint64)
    logical_jk = np.column_stack((j.ravel(), k.ravel()))
    return points, tags, quads, logical_jk, grid


def _centred_patch_bounds(
    params: base.BearingParams,
    inlet: InletSpec,
    q: int,
    n_theta: int,
    n_axial: int,
    *,
    support_factor: int,
) -> tuple[int, int, int, int]:
    if q < 4 or q % 2:
        raise BodyFittedError("q must be an even integer >=4")
    theta_centre = n_theta // 2
    axial_position = inlet.axial_position_mm * n_axial / params.length_mm
    axial_centre = int(round(axial_position))
    if not math.isclose(axial_position, axial_centre, abs_tol=1.0e-12):
        raise BodyFittedError("the inlet centre must coincide with a master-grid axial node")
    half = support_factor * q // 2
    if theta_centre - half < 1 or theta_centre + half >= n_theta - 1:
        raise BodyFittedError("the body-fitted support would cross the periodic seam")
    if axial_centre - half < 1 or axial_centre + half > n_axial - 1:
        raise BodyFittedError("the body-fitted support would touch an axial end")
    return theta_centre, axial_centre, theta_centre - q // 2, axial_centre - q // 2


def _concentric_square_to_disk(u: float, v: float) -> tuple[float, float]:
    if u == 0.0 and v == 0.0:
        return 0.0, 0.0
    if abs(u) > abs(v):
        radius = u
        angle = (math.pi / 4.0) * (v / u)
    else:
        radius = v
        angle = math.pi / 2.0 - (math.pi / 4.0) * (u / v)
    return radius * math.cos(angle), radius * math.sin(angle)


def _rect_perimeter_tags(
    grid_tags: np.ndarray, j0: int, k0: int, q: int
) -> np.ndarray:
    j1, k1 = j0 + q, k0 + q
    values = [
        *(grid_tags[k0, j0:j1].tolist()),
        *(grid_tags[k0:k1, j1].tolist()),
        *(grid_tags[k1, j1:j0:-1].tolist()),
        *(grid_tags[k1:k0:-1, j0].tolist()),
    ]
    return np.asarray(values, dtype=np.uint64)


def _harmonic_tensor_support(
    base_grid: np.ndarray,
    mapped_points: np.ndarray,
    theta_centre: int,
    axial_centre: int,
    q: int,
) -> np.ndarray:
    """Blend the mapped q-square to the unchanged 2q-square boundary."""
    support_j0, support_k0 = theta_centre - q, axial_centre - q
    size = 2 * q + 1
    coordinates = np.empty((size, size, 2), dtype=np.float64)
    for local_k in range(size):
        for local_j in range(size):
            point = base_grid[support_k0 + local_k, support_j0 + local_j]
            coordinates[local_k, local_j] = point[(0, 2),]

    fixed = np.zeros((size, size), dtype=bool)
    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    patch_start, patch_end = q // 2, 3 * q // 2
    fixed[patch_start : patch_end + 1, patch_start : patch_end + 1] = True
    coordinates[
        patch_start : patch_end + 1, patch_start : patch_end + 1
    ] = mapped_points

    unknown = np.argwhere(~fixed)
    row_for = {tuple(index): row for row, index in enumerate(unknown.tolist())}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    rhs = np.zeros((len(unknown), 2), dtype=np.float64)
    for row, (k, j) in enumerate(unknown):
        rows.append(row)
        columns.append(row)
        data.append(4.0)
        for dk, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (int(k + dk), int(j + dj))
            if fixed[neighbour]:
                rhs[row] += coordinates[neighbour]
            else:
                rows.append(row)
                columns.append(row_for[neighbour])
                data.append(-1.0)
    matrix = csr_matrix((data, (rows, columns)), shape=(len(unknown), len(unknown)))
    solution = spsolve(matrix, rhs)
    coordinates[unknown[:, 0], unknown[:, 1]] = solution
    return coordinates


def _project_bore_point(
    params: base.BearingParams, x_mm: float, z_mm: float
) -> np.ndarray:
    radius = float(params.bore_radius_mm(z_mm))
    radicand = radius**2 - x_mm**2
    if radicand <= 0.0:
        raise BodyFittedError("body-fitted node lies outside the conical bore")
    return np.asarray((x_mm, math.sqrt(radicand), z_mm), dtype=np.float64)


def build_tensor_warp_master(
    params: base.BearingParams,
    inlet: InletSpec,
    q: int,
    *,
    geometry_mode: GeometryMode = "inscribed",
    n_theta: int = 256,
    n_axial: int = 96,
) -> MasterMesh:
    """Build the same-connectivity concentric-map tensor experiment."""
    points, tags, quads, logical, grid = _base_master(params, n_theta, n_axial)
    theta_centre, axial_centre, patch_j0, patch_k0 = _centred_patch_bounds(
        params, inlet, q, n_theta, n_axial, support_factor=2
    )
    radius = _effective_radius(inlet.radius_mm, 4 * q, geometry_mode)
    mapped = np.empty((q + 1, q + 1, 2), dtype=np.float64)
    for local_k in range(q + 1):
        for local_j in range(q + 1):
            u = -(local_j - q / 2.0) / (q / 2.0)
            v = (local_k - q / 2.0) / (q / 2.0)
            disk_x, disk_z = _concentric_square_to_disk(u, v)
            mapped[local_k, local_j] = (
                radius * disk_x,
                inlet.axial_position_mm + radius * disk_z,
            )
    support = _harmonic_tensor_support(
        grid, mapped, theta_centre, axial_centre, q
    )
    support_j0, support_k0 = theta_centre - q, axial_centre - q
    changed = points.copy()
    for local_k in range(1, 2 * q):
        for local_j in range(1, 2 * q):
            x_mm, z_mm = support[local_k, local_j]
            index = (support_k0 + local_k) * n_theta + support_j0 + local_j
            changed[index] = _project_bore_point(params, float(x_mm), float(z_mm))

    j = logical[:, 0]
    k = logical[:, 1]
    pressure = (
        (j >= patch_j0)
        & (j < patch_j0 + q)
        & (k >= patch_k0)
        & (k < patch_k0 + q)
    )
    in_support = (
        (j >= support_j0)
        & (j < support_j0 + 2 * q)
        & (k >= support_k0)
        & (k < support_k0 + 2 * q)
    )
    block_id = np.zeros(len(quads), dtype=np.int32)
    block_id[in_support & ~pressure] = 5
    pressure_rows = np.flatnonzero(pressure)
    for row in pressure_rows:
        u = (j[row] + 0.5 - theta_centre) / (q / 2.0)
        v = (k[row] + 0.5 - axial_centre) / (q / 2.0)
        if abs(u) >= abs(v):
            block_id[row] = 1 if u < 0.0 else 3
        else:
            block_id[row] = 2 if v > 0.0 else 4

    grid_tags = tags.reshape(n_axial + 1, n_theta)
    rim_tags = _rect_perimeter_tags(grid_tags, patch_j0, patch_k0, q)
    support_boundary = _rect_perimeter_tags(
        grid_tags, support_j0, support_k0, 2 * q
    )
    unchanged_mask = np.asarray(
        [
            not (
                support_j0 < index % n_theta < support_j0 + 2 * q
                and support_k0 < index // n_theta < support_k0 + 2 * q
            )
            for index in range(len(tags))
        ],
        dtype=bool,
    )
    all_fixed = np.unique(
        np.concatenate((tags[unchanged_mask], rim_tags, support_boundary))
    )
    metadata = {
        "topology": "tensor-warp",
        "geometry_mode": geometry_mode,
        "n_theta": n_theta,
        "n_axial": n_axial,
        "q": q,
        "rim_segments": 4 * q,
        "support_size_cells": [2 * q, 2 * q],
        "pressure_quad_count": q**2,
        "expected_master_quad_count": n_theta * n_axial,
        "four_concentric_map_distortion_regions": True,
        "block_names": {
            "0": "unchanged_background",
            "1": "disk_sector_right",
            "2": "disk_sector_top",
            "3": "disk_sector_left",
            "4": "disk_sector_bottom",
            "5": "harmonic_support",
        },
        **rim_geometry_diagnostics(inlet, 4 * q, geometry_mode),
    }
    return MasterMesh(
        points_mm=changed,
        node_tags=tags,
        quads=quads,
        quad_tags=np.arange(1, len(quads) + 1, dtype=np.uint64),
        pressure_mask=pressure,
        block_id=block_id,
        rim_node_tags=rim_tags,
        control_loop_node_tags=np.empty(0, dtype=np.uint64),
        fixed_node_tags=all_fixed,
        unchanged_node_tags=tags[unchanged_mask],
        unchanged_points_mm=points[unchanged_mask],
        metadata=metadata,
    )


def build_ogrid_master(
    params: base.BearingParams,
    inlet: InletSpec,
    rim_segments: int,
    inner_layers: int,
    outer_layers: int,
    *,
    geometry_mode: GeometryMode = "inscribed",
    n_theta: int = 256,
    n_axial: int = 96,
    quality_optimized: bool = False,
    control_radius_factor: float = 1.4,
    control_square_blend: float = 0.0,
    central_corner_radius_factor: float = 0.9,
) -> MasterMesh:
    """Build a conformal central-square plus eight-sector O-grid insert."""
    if inner_layers < 1 or outer_layers < 1:
        raise BodyFittedError("inner_layers and outer_layers must be positive")
    if quality_optimized and inner_layers != 1:
        raise BodyFittedError(
            "the quality-optimized O-grid requires one inner ring"
        )
    if quality_optimized and (
        not math.isfinite(control_radius_factor)
        or control_radius_factor <= 1.0
        or not math.isfinite(control_square_blend)
        or not 0.0 <= control_square_blend <= 1.0
        or not math.isfinite(central_corner_radius_factor)
        or not 0.0 < central_corner_radius_factor < 1.0
    ):
        raise BodyFittedError(
            "quality-optimized O-grid factors require control_radius_factor>1, "
            "0<=control_square_blend<=1, and 0<central_corner_radius_factor<1"
        )
    if rim_segments % 4:
        raise BodyFittedError("rim_segments must be divisible by four")
    q = rim_segments // 4
    points, tags, quads, logical, grid = _base_master(params, n_theta, n_axial)
    theta_centre, axial_centre, patch_j0, patch_k0 = _centred_patch_bounds(
        params,
        inlet,
        q,
        n_theta,
        n_axial,
        support_factor=2 if quality_optimized else 1,
    )
    patch_j1, patch_k1 = patch_j0 + q, patch_k0 + q
    removed = (
        (logical[:, 0] >= patch_j0)
        & (logical[:, 0] < patch_j1)
        & (logical[:, 1] >= patch_k0)
        & (logical[:, 1] < patch_k1)
    )
    background_quads = quads[~removed]
    grid_tags = tags.reshape(n_axial + 1, n_theta)
    control_tags = _rect_perimeter_tags(
        grid_tags, patch_j0, patch_k0, q
    )
    radius = _effective_radius(inlet.radius_mm, rim_segments, geometry_mode)
    if quality_optimized:
        mapped = np.empty((q + 1, q + 1, 2), dtype=np.float64)
        for local_k in range(q + 1):
            for local_j in range(q + 1):
                u = -(local_j - q / 2.0) / (q / 2.0)
                v = (local_k - q / 2.0) / (q / 2.0)
                disk_x, disk_z = _concentric_square_to_disk(u, v)
                circular = np.asarray(
                    (
                        control_radius_factor * radius * disk_x,
                        inlet.axial_position_mm
                        + control_radius_factor * radius * disk_z,
                    )
                )
                background = grid[
                    patch_k0 + local_k, patch_j0 + local_j
                ][(0, 2),]
                mapped[local_k, local_j] = (
                    (1.0 - control_square_blend) * circular
                    + control_square_blend * background
                )
        support = _harmonic_tensor_support(
            grid, mapped, theta_centre, axial_centre, q
        )
        support_j0, support_k0 = theta_centre - q, axial_centre - q
        points = points.copy()
        for local_k in range(1, 2 * q):
            for local_j in range(1, 2 * q):
                x_mm, z_mm = support[local_k, local_j]
                index = (
                    (support_k0 + local_k) * n_theta
                    + support_j0
                    + local_j
                )
                points[index] = _project_bore_point(
                    params, float(x_mm), float(z_mm)
                )

    original_interior = np.asarray(
        [
            grid_tags[k, j]
            for k in range(patch_k0 + 1, patch_k1)
            for j in range(patch_j0 + 1, patch_j1)
        ],
        dtype=np.uint64,
    )
    retained_mask = ~np.isin(tags, original_interior)
    node_points = {
        int(tag): point.copy()
        for tag, point in zip(tags[retained_mask], points[retained_mask])
    }
    next_tag = int(tags[-1]) + 1

    def add_node(point: np.ndarray) -> int:
        nonlocal next_tag
        tag = next_tag
        next_tag += 1
        node_points[tag] = np.asarray(point, dtype=np.float64)
        return tag

    corner_radius_factor = (
        central_corner_radius_factor if quality_optimized else 0.5
    )
    half_side = corner_radius_factor * radius / math.sqrt(2.0)
    central_tags = np.empty((q + 1, q + 1), dtype=np.uint64)
    for k_local in range(q + 1):
        v = 2.0 * k_local / q - 1.0
        z_mm = inlet.axial_position_mm + half_side * (
            math.tan(math.pi * v / 4.0) if quality_optimized else v
        )
        for j_local in range(q + 1):
            u = 1.0 - 2.0 * j_local / q
            x_mm = half_side * (
                math.tan(math.pi * u / 4.0) if quality_optimized else u
            )
            central_tags[k_local, j_local] = add_node(
                _project_bore_point(params, x_mm, z_mm)
            )
    central_perimeter = _rect_perimeter_tags(central_tags, 0, 0, q)
    central_corner_tags = np.asarray(
        (
            central_tags[0, 0],
            central_tags[0, -1],
            central_tags[-1, -1],
            central_tags[-1, 0],
        ),
        dtype=np.uint64,
    )

    rim_points = analytic_rim_nodes(
        params,
        inlet,
        rim_segments,
        geometry_mode,
        start_angle_rad=-math.pi / 4.0,
        clockwise=True,
    )
    rim_tags = np.asarray([add_node(point) for point in rim_points], dtype=np.uint64)
    control_points = np.asarray(
        [node_points[int(tag)] for tag in control_tags], dtype=np.float64
    )
    central_points = np.asarray(
        [node_points[int(tag)] for tag in central_perimeter], dtype=np.float64
    )

    inner_rings = [central_perimeter]
    for layer in range(1, inner_layers):
        fraction = layer / inner_layers
        ring = []
        for inner, outer in zip(central_points, rim_points):
            x_mm, z_mm = (
                (1.0 - fraction) * inner[(0, 2),]
                + fraction * outer[(0, 2),]
            )
            ring.append(add_node(_project_bore_point(params, x_mm, z_mm)))
        inner_rings.append(np.asarray(ring, dtype=np.uint64))
    inner_rings.append(rim_tags)

    outer_rings = [rim_tags]
    for layer in range(1, outer_layers):
        fraction = layer / outer_layers
        ring = []
        for inner, outer in zip(rim_points, control_points):
            x_mm, z_mm = (
                (1.0 - fraction) * inner[(0, 2),]
                + fraction * outer[(0, 2),]
            )
            ring.append(add_node(_project_bore_point(params, x_mm, z_mm)))
        outer_rings.append(np.asarray(ring, dtype=np.uint64))
    outer_rings.append(control_tags)

    central_quads = []
    for k_local in range(q):
        for j_local in range(q):
            central_quads.append(
                (
                    central_tags[k_local, j_local],
                    central_tags[k_local, j_local + 1],
                    central_tags[k_local + 1, j_local + 1],
                    central_tags[k_local + 1, j_local],
                )
            )

    def ring_quads(rings: list[np.ndarray]) -> tuple[list[tuple[int, ...]], list[int]]:
        result: list[tuple[int, ...]] = []
        blocks: list[int] = []
        for layer in range(len(rings) - 1):
            inner, outer = rings[layer], rings[layer + 1]
            for index in range(rim_segments):
                following = (index + 1) % rim_segments
                result.append(
                    (
                        int(inner[index]),
                        int(outer[index]),
                        int(outer[following]),
                        int(inner[following]),
                    )
                )
                blocks.append(index // q)
        return result, blocks

    inner_quads, inner_blocks = ring_quads(inner_rings)
    outer_quads, outer_blocks = ring_quads(outer_rings)
    inserted = np.asarray(
        [*central_quads, *inner_quads, *outer_quads], dtype=np.uint64
    )
    all_quads = np.concatenate((background_quads, inserted))
    pressure = np.concatenate(
        (
            np.zeros(len(background_quads), dtype=bool),
            np.ones(len(central_quads) + len(inner_quads), dtype=bool),
            np.zeros(len(outer_quads), dtype=bool),
        )
    )
    block_id = np.concatenate(
        (
            np.full(len(background_quads), 9, dtype=np.int32),
            np.zeros(len(central_quads), dtype=np.int32),
            1 + np.asarray(inner_blocks, dtype=np.int32),
            5 + np.asarray(outer_blocks, dtype=np.int32),
        )
    )
    sorted_tags = np.asarray(sorted(node_points), dtype=np.uint64)
    all_points = np.asarray(
        [node_points[int(tag)] for tag in sorted_tags], dtype=np.float64
    )
    if quality_optimized:
        logical_j = np.arange(len(tags), dtype=np.int64) % n_theta
        logical_k = np.arange(len(tags), dtype=np.int64) // n_theta
        support_interior = (
            (logical_j > support_j0)
            & (logical_j < support_j0 + 2 * q)
            & (logical_k > support_k0)
            & (logical_k < support_k0 + 2 * q)
        )
        unchanged_mask = retained_mask & ~support_interior
    else:
        unchanged_mask = retained_mask
    original_tags = tags[unchanged_mask]
    original_points = points[unchanged_mask]
    expected = n_theta * n_axial + 4 * q * (inner_layers + outer_layers)
    if len(all_quads) != expected:
        raise BodyFittedError(
            f"O-grid arithmetic produced {len(all_quads)} quads, expected {expected}"
        )
    metadata = {
        "topology": "ogrid",
        "geometry_mode": geometry_mode,
        "n_theta": n_theta,
        "n_axial": n_axial,
        "q": q,
        "rim_segments": rim_segments,
        "inner_layers": inner_layers,
        "outer_layers": outer_layers,
        "removed_background_quads": q**2,
        "inserted_quads": len(inserted),
        "pressure_quad_count": q**2 + 4 * q * inner_layers,
        "expected_master_quad_count": expected,
        "central_square_corner_radius_mm": corner_radius_factor * radius,
        "central_square_corner_node_tags": [
            int(tag) for tag in central_corner_tags
        ],
        "initialization": (
            "radial-spoke central square, concentric rings, harmonic far-field support"
            if quality_optimized
            else "piecewise transfinite linear/Coons ring interpolation"
        ),
        "quality_optimized": quality_optimized,
        "support_size_cells": (
            [2 * q, 2 * q] if quality_optimized else [q, q]
        ),
        "control_loop_radius_factor": (
            control_radius_factor if quality_optimized else None
        ),
        "control_loop_square_blend": (
            control_square_blend if quality_optimized else None
        ),
        "central_square_initial_corner_radius_factor": (
            central_corner_radius_factor if quality_optimized else 0.5
        ),
        "same_total_count_claim": "retired",
        "exact_budget_transition_templates": "out_of_scope",
        "block_names": {
            "0": "central_square",
            "1": "inner_bottom",
            "2": "inner_left",
            "3": "inner_top",
            "4": "inner_right",
            "5": "outer_bottom",
            "6": "outer_left",
            "7": "outer_top",
            "8": "outer_right",
            "9": "unchanged_background",
        },
        **rim_geometry_diagnostics(inlet, rim_segments, geometry_mode),
    }
    return MasterMesh(
        points_mm=all_points,
        node_tags=sorted_tags,
        quads=all_quads,
        quad_tags=np.arange(1, len(all_quads) + 1, dtype=np.uint64),
        pressure_mask=pressure,
        block_id=block_id,
        rim_node_tags=rim_tags,
        control_loop_node_tags=control_tags,
        fixed_node_tags=np.unique(
            np.concatenate(
                (original_tags, rim_tags, control_tags, central_corner_tags)
            )
        ),
        unchanged_node_tags=original_tags,
        unchanged_points_mm=original_points,
        metadata=metadata,
    )


def _indices_for_tags(node_tags: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(node_tags)
    positions = np.searchsorted(node_tags[order], values)
    if np.any(positions >= len(order)) or not np.array_equal(
        node_tags[order[positions]], values
    ):
        raise BodyFittedError("connectivity references an unknown node tag")
    return order[positions]


def quad_edge_census(quads: np.ndarray) -> dict[str, np.ndarray]:
    """Return generic undirected Quad4 edge ownership."""
    local_edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)))
    occurrences = quads[:, local_edges].reshape(-1, 2)
    keys = np.sort(occurrences, axis=1)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    owners = np.repeat(np.arange(len(quads), dtype=np.int64), 4)
    first = owners[order[starts]]
    second = np.full(len(unique), -1, dtype=np.int64)
    internal = counts == 2
    second[internal] = owners[order[starts[internal] + 1]]
    return {
        "edges": unique,
        "arity": np.full(len(unique), 2, dtype=np.uint8),
        "counts": counts,
        "owner": first,
        "neighbour": second,
        "inverse": inverse,
    }


def _master_quad_quality(
    master: MasterMesh, params: base.BearingParams
) -> dict[str, np.ndarray]:
    indices = _indices_for_tags(master.node_tags, master.quads)
    vertices = master.points_mm[indices]
    centres, area_vectors = base._quad_geometry(vertices)
    radius = np.asarray(params.bore_radius_mm(centres[:, 2]))
    analytic = np.column_stack(
        (centres[:, 0], centres[:, 1], radius * params.cone_slope)
    )
    signed_area = np.einsum("ij,ij->i", area_vectors, analytic)
    signed_area /= np.linalg.norm(analytic, axis=1)
    corner_quality = np.empty((len(vertices), 4), dtype=np.float64)
    for corner in range(4):
        following = vertices[:, (corner + 1) % 4] - vertices[:, corner]
        previous = vertices[:, (corner - 1) % 4] - vertices[:, corner]
        normal = np.column_stack(
            (
                vertices[:, corner, 0],
                vertices[:, corner, 1],
                np.asarray(params.bore_radius_mm(vertices[:, corner, 2]))
                * params.cone_slope,
            )
        )
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        cross = np.cross(following, previous)
        denominator = np.linalg.norm(following, axis=1) * np.linalg.norm(
            previous, axis=1
        )
        corner_quality[:, corner] = np.einsum("ij,ij->i", cross, normal) / denominator
    return {
        "signed_area_mm2": signed_area,
        "scaled_jacobian": corner_quality.min(axis=1),
        "area_mm2": np.linalg.norm(area_vectors, axis=1),
    }


def _component_count(cell_count: int, owner: np.ndarray, neighbour: np.ndarray) -> int:
    parent = np.arange(cell_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first, second in zip(owner, neighbour):
        first_root, second_root = find(int(first)), find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root
    return len({find(index) for index in range(cell_count)})


def _loop_edges(tags: np.ndarray) -> np.ndarray:
    return np.sort(
        np.column_stack((tags, np.roll(tags, -1))).astype(np.uint64), axis=1
    )


def validate_master_mesh(
    master: MasterMesh,
    params: base.BearingParams,
    inlet: InletSpec,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate generic master connectivity, interfaces, and analytic geometry."""
    records = [] if records is None else records
    expected = int(master.metadata["expected_master_quad_count"])
    require(records, "master.counts.quads", len(master.quads) == expected, len(master.quads), expected)
    require(
        records,
        "master.counts.pressure_quads",
        int(master.pressure_mask.sum()) == int(master.metadata["pressure_quad_count"]),
        int(master.pressure_mask.sum()),
        int(master.metadata["pressure_quad_count"]),
    )
    require(
        records,
        "master.data.generic_shapes",
        master.points_mm.shape == (len(master.node_tags), 3)
        and master.quads.shape == (expected, 4)
        and master.pressure_mask.shape == (expected,)
        and master.block_id.shape == (expected,),
        {
            "points": master.points_mm.shape,
            "quads": master.quads.shape,
            "pressure": master.pressure_mask.shape,
            "block_id": master.block_id.shape,
        },
        "generic N x 3 nodes and M x 4 quads",
    )
    require(
        records,
        "master.data.unique_tags_and_quads",
        len(np.unique(master.node_tags)) == len(master.node_tags)
        and len(np.unique(np.sort(master.quads, axis=1), axis=0)) == len(master.quads),
        {"node_tags": len(np.unique(master.node_tags)), "quads": len(np.unique(np.sort(master.quads, axis=1), axis=0))},
        {"node_tags": len(master.node_tags), "quads": len(master.quads)},
    )
    quality = _master_quad_quality(master, params)
    require(
        records,
        "master.quality.positive_signed_area",
        bool(np.all(quality["signed_area_mm2"] > 0.0)),
        float(quality["signed_area_mm2"].min()),
        ">0",
    )
    require(
        records,
        "master.quality.positive_scaled_jacobian",
        bool(np.all(quality["scaled_jacobian"] > 0.0)),
        float(quality["scaled_jacobian"].min()),
        ">0",
    )
    census = quad_edge_census(master.quads)
    require(
        records,
        "master.topology.edge_owners",
        bool(np.all((census["counts"] == 1) | (census["counts"] == 2))),
        {
            str(value): int(np.count_nonzero(census["counts"] == value))
            for value in np.unique(census["counts"])
        },
        "one or two owners",
    )
    internal = census["counts"] == 2
    components = _component_count(
        len(master.quads),
        census["owner"][internal],
        census["neighbour"][internal],
    )
    require(records, "master.topology.one_connected_surface", components == 1, components, 1)

    edge_lookup = {
        tuple(edge): index for index, edge in enumerate(census["edges"].tolist())
    }
    rim_edges = _loop_edges(master.rim_node_tags)
    rim_indices = np.asarray(
        [edge_lookup.get(tuple(edge), -1) for edge in rim_edges], dtype=np.int64
    )
    require(
        records,
        "master.rim.closed_degree_two",
        len(np.unique(master.rim_node_tags)) == len(master.rim_node_tags)
        and np.all(rim_indices >= 0),
        {
            "segments": len(rim_edges),
            "vertices": len(np.unique(master.rim_node_tags)),
            "missing_edges": int(np.count_nonzero(rim_indices < 0)),
        },
        {"segments": len(master.rim_node_tags), "degree": 2},
    )
    rim_owner_partition = []
    for edge_index in rim_indices:
        owners = (
            int(census["owner"][edge_index]),
            int(census["neighbour"][edge_index]),
        )
        rim_owner_partition.append(
            sorted(bool(master.pressure_mask[owner]) for owner in owners)
        )
    require(
        records,
        "master.rim.two_owners_pressure_stationary",
        bool(np.all(census["counts"][rim_indices] == 2))
        and all(value == [False, True] for value in rim_owner_partition),
        {
            "two_owner_edges": int(np.count_nonzero(census["counts"][rim_indices] == 2)),
            "partitioned_edges": sum(
                value == [False, True] for value in rim_owner_partition
            ),
        },
        len(rim_indices),
    )
    if len(master.control_loop_node_tags):
        control_indices = np.asarray(
            [
                edge_lookup.get(tuple(edge), -1)
                for edge in _loop_edges(master.control_loop_node_tags)
            ],
            dtype=np.int64,
        )
        require(
            records,
            "master.control_loop.two_owner_interface",
            np.all(control_indices >= 0)
            and np.all(census["counts"][control_indices] == 2),
            {
                "edges": len(control_indices),
                "two_owner": int(
                    np.count_nonzero(
                        census["counts"][control_indices[control_indices >= 0]] == 2
                    )
                ),
            },
            len(control_indices),
        )

    boundary_edges = census["edges"][census["counts"] == 1]
    boundary_points = master.points_mm[
        _indices_for_tags(master.node_tags, boundary_edges)
    ]
    at_z0 = np.all(np.abs(boundary_points[:, :, 2]) <= 1.0e-12, axis=1)
    at_zl = np.all(
        np.abs(boundary_points[:, :, 2] - params.length_mm) <= 1.0e-12,
        axis=1,
    )
    require(
        records,
        "master.topology.only_axial_boundary_edges",
        bool(np.all(at_z0 | at_zl)),
        {
            "boundary_edges": len(boundary_edges),
            "non_axial": int(np.count_nonzero(~(at_z0 | at_zl))),
        },
        "all at z=0 or z=L",
    )

    rim_points = master.points_mm[
        _indices_for_tags(master.node_tags, master.rim_node_tags)
    ]
    effective_radius = float(master.metadata["effective_radius_mm"])
    circle_residual = np.hypot(
        rim_points[:, 0], rim_points[:, 2] - inlet.axial_position_mm
    ) - effective_radius
    bore_residual = np.hypot(rim_points[:, 0], rim_points[:, 1]) - np.asarray(
        params.bore_radius_mm(rim_points[:, 2])
    )
    analytic_residual = max(
        float(np.abs(circle_residual).max(initial=0.0)),
        float(np.abs(bore_residual).max(initial=0.0)),
    )
    require(
        records,
        "master.rim.analytic_residual_mm",
        analytic_residual <= 1.0e-10,
        analytic_residual,
        1.0e-10,
    )
    if master.metadata["topology"] == "ogrid":
        corner_tags = np.asarray(
            master.metadata["central_square_corner_node_tags"],
            dtype=np.uint64,
        )
        corner_points = master.points_mm[
            _indices_for_tags(master.node_tags, corner_tags)
        ]
        target_radius = float(
            master.metadata["central_square_corner_radius_mm"]
        )
        corner_residual = np.abs(
            np.hypot(
                corner_points[:, 0],
                corner_points[:, 2] - inlet.axial_position_mm,
            )
            - target_radius
        )
        require(
            records,
            "master.ogrid.central_square_corner_radius_mm",
            float(corner_residual.max()) <= 1.0e-10,
            float(corner_residual.max()),
            1.0e-10,
        )
    unchanged = master.points_mm[
        _indices_for_tags(master.node_tags, master.unchanged_node_tags)
    ]
    require(
        records,
        "master.far_field.bitwise_unchanged",
        np.array_equal(unchanged, master.unchanged_points_mm),
        int(np.count_nonzero(unchanged != master.unchanged_points_mm)),
        0,
    )
    return {
        "topology": master.metadata["topology"],
        "counts": {
            "points": len(master.points_mm),
            "quads": len(master.quads),
            "pressure_quads": int(master.pressure_mask.sum()),
            "stationary_quads": int((~master.pressure_mask).sum()),
            "rim_segments": len(master.rim_node_tags),
        },
        "quality": {
            "minimum_signed_area_mm2": float(quality["signed_area_mm2"].min()),
            "minimum_scaled_jacobian": float(quality["scaled_jacobian"].min()),
            "maximum_area_mm2": float(quality["area_mm2"].max()),
        },
        "connected_regions": components,
        "analytic_rim_residual_mm": analytic_residual,
        "far_field_node_count": len(master.unchanged_node_tags),
        "records": records,
    }


def smooth_master_mesh(
    master: MasterMesh,
    params: base.BearingParams,
    *,
    iterations: int = 12,
    damping: float = 0.25,
) -> MasterMesh:
    """Apply deterministic damped Laplacian smoothing when quality does not fall."""
    if iterations < 0 or not 0.0 < damping <= 1.0:
        raise BodyFittedError("smoothing requires iterations>=0 and 0<damping<=1")
    census = quad_edge_census(master.quads)
    adjacency: dict[int, set[int]] = {
        int(tag): set() for tag in master.node_tags
    }
    for first, second in census["edges"]:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    fixed = set(map(int, master.fixed_node_tags))
    movable = np.asarray(
        [tag for tag in master.node_tags if int(tag) not in fixed], dtype=np.uint64
    )
    points = master.points_mm.copy()
    tag_to_index = {int(tag): index for index, tag in enumerate(master.node_tags)}
    previous_minimum = float(
        _master_quad_quality(master, params)["scaled_jacobian"].min()
    )
    accepted = 0
    for _iteration in range(iterations):
        candidate = points.copy()
        for tag in movable:
            index = tag_to_index[int(tag)]
            neighbours = np.asarray(
                [tag_to_index[value] for value in adjacency[int(tag)]],
                dtype=np.int64,
            )
            target_xz = points[neighbours][:, (0, 2)].mean(axis=0)
            xz = (1.0 - damping) * points[index, (0, 2)] + damping * target_xz
            candidate[index] = _project_bore_point(
                params, float(xz[0]), float(xz[1])
            )
        proposed = replace(
            master,
            points_mm=candidate,
            metadata=dict(master.metadata),
        )
        quality = _master_quad_quality(proposed, params)
        minimum = float(quality["scaled_jacobian"].min())
        if (
            np.all(quality["signed_area_mm2"] > 0.0)
            and minimum + 1.0e-14 >= previous_minimum
            and np.array_equal(
                candidate[
                    _indices_for_tags(master.node_tags, master.fixed_node_tags)
                ],
                points[
                    _indices_for_tags(master.node_tags, master.fixed_node_tags)
                ],
            )
        ):
            points = candidate
            previous_minimum = minimum
            accepted += 1
        else:
            break
    return replace(
        master,
        points_mm=points,
        metadata=dict(master.metadata)
        | {
            "smoothing": {
                "method": "damped Laplacian",
                "requested_iterations": iterations,
                "accepted_iterations": accepted,
                "damping": damping,
                "minimum_scaled_jacobian_after": previous_minimum,
            }
        },
    )


def _refresh_ogrid_corner_radius(
    master: MasterMesh, inlet: InletSpec
) -> MasterMesh:
    if master.metadata["topology"] != "ogrid":
        return master
    corner_tags = np.asarray(
        master.metadata["central_square_corner_node_tags"],
        dtype=np.uint64,
    )
    corner_points = master.points_mm[
        _indices_for_tags(master.node_tags, corner_tags)
    ]
    corner_radii = np.hypot(
        corner_points[:, 0],
        corner_points[:, 2] - inlet.axial_position_mm,
    )
    if np.ptp(corner_radii) > 1.0e-10:
        raise BodyFittedError("smoothed central corners lost fourfold symmetry")
    return replace(
        master,
        metadata=dict(master.metadata)
        | {"central_square_corner_radius_mm": float(corner_radii.mean())},
    )


def hex_face_census(hexes: np.ndarray) -> dict[str, np.ndarray]:
    """Return generic Hex8 face ownership while retaining face arity."""
    occurrences = hexes[:, base.HEX_FACES].reshape(-1, 4)
    keys = np.sort(occurrences, axis=1)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    occurrence_owner = np.repeat(np.arange(len(hexes), dtype=np.int64), 6)
    first_occurrence = order[starts]
    owner = occurrence_owner[first_occurrence]
    neighbour = np.full(len(unique), -1, dtype=np.int64)
    internal = counts == 2
    neighbour[internal] = occurrence_owner[order[starts[internal] + 1]]
    return {
        "faces": unique,
        "oriented_faces": occurrences[first_occurrence],
        "arity": np.full(len(unique), 4, dtype=np.uint8),
        "counts": counts,
        "owner": owner,
        "neighbour": neighbour,
        "inverse": inverse,
    }


def _generic_hex_metrics(
    points_m: np.ndarray,
    node_tags: np.ndarray,
    hexes: np.ndarray,
    *,
    chunk_size: int = 40_000,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = len(hexes)
    volumes = np.empty(count, dtype=np.float64)
    gauss_volumes = np.empty(count, dtype=np.float64)
    gauss_minimum = np.empty(count, dtype=np.float64)
    aspect = np.empty(count, dtype=np.float64)
    minimum_pyramid = np.empty(count, dtype=np.float64)
    centres = np.empty((count, 3), dtype=np.float64)
    hex_indices = _indices_for_tags(node_tags, hexes)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        cell_points = points_m[hex_indices[start:stop]]
        face_centres = np.empty((stop - start, 6, 3), dtype=np.float64)
        face_areas = np.empty_like(face_centres)
        for face_index, local_face in enumerate(base.HEX_FACES):
            face_centres[:, face_index], face_areas[:, face_index] = (
                base._quad_geometry(cell_points[:, local_face])
            )
        estimate = face_centres.mean(axis=1)
        pyramid_three = np.einsum(
            "mfc,mfc->mf", face_areas, face_centres - estimate[:, None, :]
        )
        volume = pyramid_three.sum(axis=1) / 3.0
        centre = (
            pyramid_three[:, :, None]
            * (0.75 * face_centres + 0.25 * estimate[:, None, :])
        ).sum(axis=1) / pyramid_three.sum(axis=1)[:, None]
        pyramids = (
            np.einsum(
                "mfc,mfc->mf", face_areas, face_centres - centre[:, None, :]
            )
            / 3.0
        )
        component_area = np.abs(face_areas).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            aspect_area = component_area.max(axis=1) / component_area.min(axis=1)
            aspect_volume = component_area.sum(axis=1) / (
                6.0 * np.power(volume, 2.0 / 3.0)
            )
        gauss_volume, gauss_det = base._gauss_hex_volume(cell_points)
        volumes[start:stop] = volume
        gauss_volumes[start:stop] = gauss_volume
        gauss_minimum[start:stop] = gauss_det
        aspect[start:stop] = np.maximum(aspect_area, aspect_volume)
        minimum_pyramid[start:stop] = pyramids.min(axis=1)
        centres[start:stop] = centre

    nonorthogonality = np.zeros(count, dtype=np.float64)
    skewness = np.zeros(count, dtype=np.float64)
    census = hex_face_census(hexes)
    for start in range(0, len(census["faces"]), chunk_size):
        stop = min(start + chunk_size, len(census["faces"]))
        face_tags = census["oriented_faces"][start:stop]
        vertices = points_m[_indices_for_tags(node_tags, face_tags)]
        face_centre, face_area = base._quad_geometry(vertices)
        area_norm = np.linalg.norm(face_area, axis=1)
        owner = census["owner"][start:stop]
        neighbour = census["neighbour"][start:stop]
        internal = neighbour >= 0

        direction = np.empty_like(face_centre)
        direction[internal] = (
            centres[neighbour[internal]] - centres[owner[internal]]
        )
        normal = face_area / area_norm[:, None]
        owner_displacement = face_centre - centres[owner]
        boundary = ~internal
        if np.any(boundary):
            normal_distance = np.einsum(
                "ij,ij->i", normal[boundary], owner_displacement[boundary]
            )
            direction[boundary] = (
                normal[boundary] * normal_distance[:, None]
            )

        dot = np.abs(np.einsum("ij,ij->i", face_area, direction))
        denominator = area_norm * np.linalg.norm(direction, axis=1)
        cosine = np.divide(
            dot,
            denominator,
            out=np.zeros_like(dot),
            where=denominator > 0.0,
        )
        angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
        np.maximum.at(nonorthogonality, owner, angle)
        np.maximum.at(nonorthogonality, neighbour[internal], angle[internal])

        area_dot_direction = np.einsum("ij,ij->i", face_area, direction)
        area_dot_displacement = np.einsum(
            "ij,ij->i", face_area, owner_displacement
        )
        intersection = centres[owner] + (
            area_dot_displacement / area_dot_direction
        )[:, None] * direction
        skew_vector = face_centre - intersection
        skew_norm = np.linalg.norm(skew_vector, axis=1)
        skew_unit = np.divide(
            skew_vector,
            skew_norm[:, None],
            out=np.zeros_like(skew_vector),
            where=skew_norm[:, None] > 0.0,
        )
        vertex_projection = np.abs(
            np.einsum(
                "mc,mfc->mf",
                skew_unit,
                vertices - face_centre[:, None, :],
            )
        ).max(axis=1)
        scale = np.where(internal, 0.2, 0.4) * np.linalg.norm(
            direction, axis=1
        )
        skew = np.divide(
            skew_norm,
            np.maximum(scale, vertex_projection),
            out=np.zeros_like(skew_norm),
            where=np.maximum(scale, vertex_projection) > 0.0,
        )
        np.maximum.at(skewness, owner, skew)
        np.maximum.at(skewness, neighbour[internal], skew[internal])

    return (
        centres,
        {
            "signed_volume_m3": volumes,
            "gauss_volume_m3": gauss_volumes,
            "gauss_min_det": gauss_minimum,
            "aspect_ratio": aspect,
            "max_nonorthogonality_deg": nonorthogonality,
            "max_skewness": skewness,
            "min_face_pyramid_m3": minimum_pyramid,
        },
        census,
    )


def sweep_master_mesh(
    master: MasterMesh,
    params: base.BearingParams,
    *,
    n_gap: int = 12,
    gap_inflation_ratio: float = 1.0,
) -> BodyFittedMesh:
    """Sweep every generic master node through its same-z radial clearance."""
    if n_gap < 1:
        raise BodyFittedError("n_gap must be positive")
    if not math.isfinite(gap_inflation_ratio) or gap_inflation_ratio < 1.0:
        raise BodyFittedError("gap_inflation_ratio must be finite and >=1")
    if gap_inflation_ratio > 1.0 and n_gap < 3:
        raise BodyFittedError("symmetric inflation requires n_gap>=3")
    xi = (
        np.arange(n_gap + 1, dtype=np.float64) / n_gap
        if gap_inflation_ratio == 1.0
        else symmetric_gap_coordinates(n_gap, gap_inflation_ratio)
    )
    master_points = master.points_mm
    theta = np.mod(
        np.arctan2(master_points[:, 0], -master_points[:, 1]),
        2.0 * math.pi,
    )
    z_mm = master_points[:, 2]
    journal_radius = np.asarray(params.journal_radius_mm(z_mm), dtype=np.float64)
    q_ray = params.ex_mm * np.sin(theta) - params.ey_mm * np.cos(theta)
    radicand = (
        journal_radius**2
        - params.ex_mm**2
        - params.ey_mm**2
        + q_ray**2
    )
    if np.any(radicand <= 0.0):
        raise BodyFittedError("journal-ray intersection has a nonpositive radicand")
    rho_journal = q_ray + np.sqrt(radicand)
    rho_bore = np.hypot(master_points[:, 0], master_points[:, 1])
    gap = rho_bore - rho_journal
    if np.any(rho_journal <= 0.0) or np.any(gap <= 0.0):
        raise BodyFittedError("the sweep requires rho_bore > rho_journal > 0")
    rho = rho_journal[:, None] + xi[None, :] * gap[:, None]
    swept = np.empty((len(master_points), n_gap + 1, 3), dtype=np.float64)
    swept[..., 0] = rho * np.sin(theta)[:, None]
    swept[..., 1] = -rho * np.cos(theta)[:, None]
    swept[..., 2] = z_mm[:, None]
    points_m = np.ascontiguousarray(swept.reshape(-1, 3) * SI_PER_MM)
    node_tags = np.arange(1, len(points_m) + 1, dtype=np.uint64)

    master_indices = _indices_for_tags(master.node_tags, master.quads)
    master_index, gap_index = np.meshgrid(
        np.arange(len(master.quads), dtype=np.int64),
        np.arange(n_gap, dtype=np.int64),
        indexing="ij",
    )
    master_index = master_index.ravel()
    gap_index = gap_index.ravel()
    inner = np.column_stack(
        [
            1
            + master_indices[master_index, corner] * (n_gap + 1)
            + gap_index
            for corner in range(4)
        ]
    )
    outer = inner + 1
    hexes = np.column_stack((inner, outer)).astype(np.uint64)
    cell_tags = np.arange(1, len(hexes) + 1, dtype=np.uint64)
    centres, metrics, census = _generic_hex_metrics(
        points_m, node_tags, hexes
    )

    external = census["counts"] == 1
    external_faces = census["oriented_faces"][external]
    external_owner = census["owner"][external]
    layers = (external_faces.astype(np.int64) - 1) % (n_gap + 1)
    face_points = points_m[
        _indices_for_tags(node_tags, external_faces)
    ]
    at_journal = np.all(layers == 0, axis=1)
    at_bore = np.all(layers == n_gap, axis=1)
    at_z0 = np.all(np.abs(face_points[:, :, 2]) <= 1.0e-14, axis=1)
    at_zl = np.all(
        np.abs(face_points[:, :, 2] - params.length_mm * SI_PER_MM)
        <= 1.0e-14,
        axis=1,
    )
    owner_master = master_index[external_owner]
    pressure = at_bore & master.pressure_mask[owner_master]
    stationary = at_bore & ~master.pressure_mask[owner_master]
    classifications = np.column_stack(
        (at_journal, stationary, at_z0, at_zl, pressure)
    )
    if not np.all(classifications.sum(axis=1) == 1):
        raise BodyFittedError("topology-derived external faces do not classify exactly once")
    boundary_quads = {
        "journal_wall": external_faces[at_journal],
        "stationary_wall": external_faces[stationary],
        "axial_end_z0": external_faces[at_z0],
        "axial_end_zL": external_faces[at_zl],
        "pressure_feed": external_faces[pressure],
    }

    cell_theta = np.mod(
        np.arctan2(centres[:, 0], -centres[:, 1]), 2.0 * math.pi
    )
    cell_z_mm = centres[:, 2] / SI_PER_MM
    cell_rj = np.asarray(params.journal_radius_mm(cell_z_mm))
    cell_q = (
        params.ex_mm * np.sin(cell_theta)
        - params.ey_mm * np.cos(cell_theta)
    )
    cell_rho_j = cell_q + np.sqrt(
        cell_rj**2
        - params.ex_mm**2
        - params.ey_mm**2
        + cell_q**2
    )
    nominal = bool(master.metadata["nominal_geometry"])
    fields = {
        "block_id": master.block_id[master_index].astype(np.int32),
        "master_quad_index": master_index.astype(np.uint32),
        "gap_index": gap_index.astype(np.uint16),
        "theta_deg": np.degrees(cell_theta),
        "axial_coordinate_mm": cell_z_mm,
        "gap_um": (
            np.asarray(params.bore_radius_mm(cell_z_mm)) - cell_rho_j
        )
        * 1_000.0,
        "nominal_geometry": np.full(len(hexes), nominal, dtype=np.uint8),
        "research_variant": np.full(len(hexes), not nominal, dtype=np.uint8),
    }
    fractions = np.diff(xi)
    metadata = dict(master.metadata) | {
        "coordinate_unit": "m",
        "source_parameter_unit": "mm",
        "scale_to_m_applied_once": SI_PER_MM,
        "n_gap": n_gap,
        "gap_layer_coordinates": xi.tolist(),
        "gap_inflation_ratio_target": gap_inflation_ratio,
        "gap_inflation_ratio_achieved": float(fractions.max() / fractions.min()),
        "contains_feed_volume": False,
        "surface_pressure_inlet": True,
        "master_point_count": len(master.points_mm),
        "master_quad_count": len(master.quads),
    }
    return BodyFittedMesh(
        points_m=points_m,
        node_tags=node_tags,
        hexes=hexes,
        cell_tags=cell_tags,
        boundary_quads=boundary_quads,
        cell_centres_m=centres,
        cell_fields=fields,
        cell_metrics=metrics,
        metadata=metadata,
    )


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values
    return values[
        np.lexsort(
            tuple(
                values[:, column]
                for column in reversed(range(values.shape[1]))
            )
        )
    ]


def _boundary_volume(mesh: BodyFittedMesh) -> float:
    total = 0.0
    for quads in mesh.boundary_quads.values():
        vertices = mesh.points_m[_indices_for_tags(mesh.node_tags, quads)]
        centres, area_vectors = base._quad_geometry(vertices)
        total += float(np.einsum("ij,ij->i", area_vectors, centres).sum())
    return total / 3.0


def _boundary_orientation(
    mesh: BodyFittedMesh, census: dict[str, np.ndarray]
) -> dict[str, float]:
    external = census["counts"] == 1
    owner_by_key = {
        tuple(face): int(owner)
        for face, owner in zip(
            census["faces"][external], census["owner"][external]
        )
    }
    result: dict[str, float] = {}
    for name, quads in mesh.boundary_quads.items():
        vertices = mesh.points_m[_indices_for_tags(mesh.node_tags, quads)]
        centres, areas = base._quad_geometry(vertices)
        owners = np.asarray(
            [owner_by_key[tuple(sorted(map(int, face)))] for face in quads],
            dtype=np.int64,
        )
        projection = np.einsum(
            "ij,ij->i", areas, centres - mesh.cell_centres_m[owners]
        )
        result[name] = float(projection.min())
    return result


def validate_body_fitted_mesh(
    mesh: BodyFittedMesh,
    master: MasterMesh,
    params: base.BearingParams,
    inlet: InletSpec,
    records: list[dict[str, Any]] | None = None,
    *,
    require_gmsh_metrics: bool = False,
) -> dict[str, Any]:
    """Apply topology, analytic, conservation, and quality acceptance gates."""
    records = [] if records is None else records
    expected_hexes = len(master.quads) * int(mesh.metadata["n_gap"])
    require(records, "body.counts.Hex8", mesh.hexes.shape == (expected_hexes, 8), mesh.hexes.shape, (expected_hexes, 8))
    require(
        records,
        "body.data.generic_field_semantics",
        set(mesh.cell_fields) == set(CELL_FIELD_NAMES)
        and all(len(values) == expected_hexes for values in mesh.cell_fields.values()),
        sorted(mesh.cell_fields),
        sorted(CELL_FIELD_NAMES),
    )
    require(
        records,
        "body.data.finite",
        np.isfinite(mesh.points_m).all()
        and np.isfinite(mesh.cell_centres_m).all()
        and all(np.isfinite(values).all() for values in mesh.cell_metrics.values()),
        "finite",
        "finite",
    )
    require(
        records,
        "body.data.unique_nodes_and_cells",
        len(np.unique(mesh.points_m, axis=0)) == len(mesh.points_m)
        and len(np.unique(np.sort(mesh.hexes, axis=1), axis=0)) == len(mesh.hexes),
        {
            "points": len(np.unique(mesh.points_m, axis=0)),
            "cells": len(np.unique(np.sort(mesh.hexes, axis=1), axis=0)),
        },
        {"points": len(mesh.points_m), "cells": len(mesh.hexes)},
    )

    census = hex_face_census(mesh.hexes)
    require(
        records,
        "body.topology.face_owners",
        bool(np.all((census["counts"] == 1) | (census["counts"] == 2))),
        {
            str(value): int(np.count_nonzero(census["counts"] == value))
            for value in np.unique(census["counts"])
        },
        "one or two owners",
    )
    external_keys = census["faces"][census["counts"] == 1]
    patch_faces = np.concatenate(
        [mesh.boundary_quads[name] for name in PATCH_NAMES]
    )
    patch_keys = np.sort(patch_faces, axis=1)
    unique_patch, patch_counts = np.unique(
        patch_keys, axis=0, return_counts=True
    )
    require(
        records,
        "body.topology.boundaries_disjoint",
        len(unique_patch) == len(patch_keys) and np.all(patch_counts == 1),
        {"faces": len(patch_keys), "unique": len(unique_patch)},
        "all boundary faces unique",
    )
    require(
        records,
        "body.topology.boundaries_exactly_cover_external_faces",
        np.array_equal(_sorted_rows(unique_patch), _sorted_rows(external_keys)),
        len(unique_patch),
        len(external_keys),
    )
    internal = census["counts"] == 2
    regions = _component_count(
        len(mesh.hexes),
        census["owner"][internal],
        census["neighbour"][internal],
    )
    require(records, "body.topology.one_connected_fluid_region", regions == 1, regions, 1)
    require(
        records,
        "body.topology.no_seam_default_or_internal_inlet_patch",
        set(mesh.boundary_quads) == set(PATCH_NAMES)
        and not any(
            token in name.lower()
            for name in mesh.boundary_quads
            for token in ("seam", "default", "internal", "mouth", "cap")
        ),
        sorted(mesh.boundary_quads),
        sorted(PATCH_NAMES),
    )

    n_gap = int(mesh.metadata["n_gap"])
    grid = mesh.points_m.reshape(len(master.points_mm), n_gap + 1, 3)
    journal = grid[:, 0] / SI_PER_MM
    bore = grid[:, -1] / SI_PER_MM
    journal_residual = np.hypot(
        journal[:, 0] - params.ex_mm,
        journal[:, 1] - params.ey_mm,
    ) - np.asarray(params.journal_radius_mm(journal[:, 2]))
    bore_residual = np.hypot(bore[:, 0], bore[:, 1]) - np.asarray(
        params.bore_radius_mm(bore[:, 2])
    )
    require(
        records,
        "body.analytic.general_journal_cone_residual_mm",
        float(np.abs(journal_residual).max()) <= 1.0e-10,
        float(np.abs(journal_residual).max()),
        1.0e-10,
    )
    require(
        records,
        "body.analytic.bore_cone_residual_mm",
        float(np.abs(bore_residual).max()) <= 1.0e-10,
        float(np.abs(bore_residual).max()),
        1.0e-10,
    )
    require(
        records,
        "body.analytic.continuous_journal_beneath_inlet",
        len(mesh.boundary_quads["journal_wall"]) == len(master.quads)
        and len(mesh.boundary_quads["pressure_feed"])
        == int(master.pressure_mask.sum()),
        {
            "journal_quads": len(mesh.boundary_quads["journal_wall"]),
            "pressure_quads": len(mesh.boundary_quads["pressure_feed"]),
        },
        {
            "journal_quads": len(master.quads),
            "pressure_quads": int(master.pressure_mask.sum()),
        },
    )

    signed = mesh.cell_metrics["signed_volume_m3"]
    gauss = mesh.cell_metrics["gauss_volume_m3"]
    gauss_det = mesh.cell_metrics["gauss_min_det"]
    pyramids = mesh.cell_metrics["min_face_pyramid_m3"]
    require(records, "body.quality.positive_volume", bool(np.all(signed > 0.0)), float(signed.min()), ">0")
    require(records, "body.quality.positive_gauss_volume", bool(np.all(gauss > 0.0)), float(gauss.min()), ">0")
    require(records, "body.quality.positive_gauss_determinant", bool(np.all(gauss_det > 0.0)), float(gauss_det.min()), ">0")
    require(records, "body.quality.positive_face_pyramids", bool(np.all(pyramids > 0.0)), float(pyramids.min()), ">0")
    require(
        records,
        "body.quality.cell_and_gauss_volume_agree",
        relative_error(float(signed.sum()), float(gauss.sum())) <= 1.0e-9,
        relative_error(float(signed.sum()), float(gauss.sum())),
        1.0e-9,
    )
    maximum_nonorthogonality = float(
        mesh.cell_metrics["max_nonorthogonality_deg"].max()
    )
    maximum_skewness = float(mesh.cell_metrics["max_skewness"].max())
    require(
        records,
        "body.quality.maximum_nonorthogonality",
        maximum_nonorthogonality <= 45.0,
        maximum_nonorthogonality,
        "<=45 deg",
    )
    require(
        records,
        "body.quality.maximum_skewness",
        maximum_skewness <= 4.0,
        maximum_skewness,
        "<=4",
    )
    if require_gmsh_metrics:
        require(
            records,
            "body.gmsh.metrics_present",
            {"minSICN", "minDetJac", "cell_volume_m3"}.issubset(mesh.cell_metrics),
            sorted(set(mesh.cell_metrics) & {"minSICN", "minDetJac", "cell_volume_m3"}),
            ["cell_volume_m3", "minDetJac", "minSICN"],
        )
        require(
            records,
            "body.gmsh.positive_minSICN",
            bool(np.all(mesh.cell_metrics["minSICN"] > 0.0)),
            float(mesh.cell_metrics["minSICN"].min()),
            ">0",
        )
        require(
            records,
            "body.gmsh.positive_minDetJac",
            bool(np.all(mesh.cell_metrics["minDetJac"] > 0.0)),
            float(mesh.cell_metrics["minDetJac"].min()),
            ">0",
        )

    mesh_volume = float(signed.sum())
    boundary_volume = _boundary_volume(mesh)
    boundary_error = relative_error(boundary_volume, mesh_volume)
    continuous_error = relative_error(mesh_volume, params.exact_volume_m3)
    require(
        records,
        "body.volume.cell_sum_boundary_agreement",
        boundary_error <= 1.0e-9,
        boundary_error,
        1.0e-9,
    )
    require(
        records,
        "body.volume.continuous_annulus_error",
        continuous_error <= 5.0e-4,
        continuous_error,
        5.0e-4,
    )
    orientations = _boundary_orientation(mesh, census)
    require(
        records,
        "body.boundary.all_outward_from_owner",
        all(value > 0.0 for value in orientations.values()),
        orientations,
        "all >0",
    )
    return {
        "counts": {
            "points": len(mesh.points_m),
            "Hex8": len(mesh.hexes),
            "boundary_Quad4": {
                name: len(mesh.boundary_quads[name]) for name in PATCH_NAMES
            },
        },
        "topology": {
            "connected_regions": regions,
            "unique_faces": len(census["faces"]),
            "internal_faces": int(np.count_nonzero(internal)),
            "external_faces": len(external_keys),
        },
        "analytic": {
            "maximum_journal_residual_mm": float(
                np.abs(journal_residual).max()
            ),
            "maximum_bore_residual_mm": float(np.abs(bore_residual).max()),
        },
        "volume": {
            "cell_sum_m3": mesh_volume,
            "boundary_integral_m3": boundary_volume,
            "cell_boundary_relative_error": boundary_error,
            "continuous_annulus_m3": params.exact_volume_m3,
            "continuous_relative_error": continuous_error,
        },
        "quality": {
            "minimum_signed_volume_m3": float(signed.min()),
            "minimum_gauss_determinant": float(gauss_det.min()),
            "minimum_face_pyramid_m3": float(pyramids.min()),
            "maximum_nonorthogonality_deg": maximum_nonorthogonality,
            "maximum_skewness": maximum_skewness,
            "minimum_minSICN": (
                float(mesh.cell_metrics["minSICN"].min())
                if "minSICN" in mesh.cell_metrics
                else None
            ),
            "minimum_minDetJac": (
                float(mesh.cell_metrics["minDetJac"].min())
                if "minDetJac" in mesh.cell_metrics
                else None
            ),
        },
        "boundary_orientation": orientations,
        "records": records,
    }


def _add_gmsh_metrics(
    mesh: BodyFittedMesh, records: list[dict[str, Any]]
) -> BodyFittedMesh:
    tags = mesh.cell_tags.astype(np.int64, copy=False)
    metrics = dict(mesh.cell_metrics)
    for field, gmsh_name in (
        ("minSICN", "minSICN"),
        ("minDetJac", "minDetJac"),
        ("cell_volume_m3", "volume"),
    ):
        values = np.asarray(
            gmsh.model.mesh.getElementQualities(tags, gmsh_name),
            dtype=np.float64,
        )
        require(
            records,
            f"gmsh.quality.{field}.finite",
            len(values) == len(mesh.hexes) and np.isfinite(values).all(),
            {"count": len(values), "minimum": float(values.min(initial=math.inf))},
            "one finite value per Hex8",
        )
        metrics[field] = values
    require(
        records,
        "gmsh.quality.volume_matches_canonical",
        relative_error(
            float(metrics["cell_volume_m3"].sum()),
            float(metrics["signed_volume_m3"].sum()),
        )
        <= 1.0e-9,
        relative_error(
            float(metrics["cell_volume_m3"].sum()),
            float(metrics["signed_volume_m3"].sum()),
        ),
        1.0e-9,
    )
    return replace(mesh, cell_metrics=metrics)


def _physical_groups() -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for dimension, physical_id in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(int(dimension), int(physical_id))
        groups[name] = {
            "dimension": int(dimension),
            "physical_id": int(physical_id),
            "entities": [
                int(tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(
                    int(dimension), int(physical_id)
                )
            ],
        }
    return groups


def _expected_group_ids() -> dict[str, tuple[int, int]]:
    return {
        **{name: (2, PHYSICAL_IDS[name]) for name in PATCH_NAMES},
        "fluid": (3, PHYSICAL_IDS["fluid"]),
    }


def _group_elements(
    element_type: int, entities: Sequence[int], arity: int
) -> tuple[np.ndarray, np.ndarray]:
    tags: list[np.ndarray] = []
    nodes: list[np.ndarray] = []
    for entity in entities:
        raw_tags, raw_nodes = gmsh.model.mesh.getElementsByType(
            element_type, int(entity)
        )
        if len(raw_tags):
            tags.append(np.asarray(raw_tags, dtype=np.int64))
            nodes.append(np.asarray(raw_nodes, dtype=np.int64).reshape(-1, arity))
    return (
        np.concatenate(tags) if tags else np.empty(0, dtype=np.int64),
        np.concatenate(nodes) if nodes else np.empty((0, arity), dtype=np.int64),
    )


def _signature_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return False
    return np.array_equal(
        _sorted_rows(np.sort(actual, axis=1)),
        _sorted_rows(np.sort(expected, axis=1)),
    )


def _round_trip_node_mapping(
    mesh: BodyFittedMesh,
    records: list[dict[str, Any]],
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    raw_tags, raw_coordinates, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(raw_tags, dtype=np.int64)
    points = np.asarray(raw_coordinates, dtype=np.float64).reshape(-1, 3)
    read_order = base._coordinate_order(points)
    source_order = base._coordinate_order(mesh.points_m)
    error = float(
        np.abs(points[read_order] - mesh.points_m[source_order]).max(initial=0.0)
    )
    require(
        records,
        f"{prefix}.coordinates",
        len(points) == len(mesh.points_m) and error <= 1.0e-14,
        {"points": len(points), "maximum_error_m": error},
        {"points": len(mesh.points_m), "maximum_error_m": 1.0e-14},
    )
    mapping = np.zeros(int(tags.max(initial=0)) + 1, dtype=np.uint64)
    mapping[tags[read_order]] = mesh.node_tags[source_order]
    return points, mapping, error


def _audit_mesh_round_trip(
    path: Path,
    mesh: BodyFittedMesh,
    records: list[dict[str, Any]],
    format_name: str,
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.open(str(path))
    prefix = f"round_trip.{format_name}"
    groups = _physical_groups()
    actual_ids = {
        name: (value["dimension"], value["physical_id"])
        for name, value in groups.items()
    }
    expected_ids = _expected_group_ids()
    if format_name == "CGNS":
        require(
            records,
            f"{prefix}.physical_names_and_dimensions",
            {
                name: dimension for name, (dimension, _identifier) in actual_ids.items()
            }
            == {
                name: dimension
                for name, (dimension, _identifier) in expected_ids.items()
            },
            actual_ids,
            {
                name: {"dimension": dimension, "canonical_id": identifier}
                for name, (dimension, identifier) in expected_ids.items()
            },
        )
        records.append(
            {
                "name": f"{prefix}.numeric_physical_ids",
                "status": "SKIPPED",
                "actual": actual_ids,
                "expected": (
                    "CGNS families retain exact names/memberships; canonical IDs "
                    "remain in MSH and physical_groups.json"
                ),
                "tolerance": None,
                "mandatory": False,
            }
        )
    else:
        require(
            records,
            f"{prefix}.physical_names_and_ids",
            actual_ids == expected_ids,
            actual_ids,
            expected_ids,
        )
    require(
        records,
        f"{prefix}.no_forbidden_patch",
        not any(
            token in name.lower()
            for name in groups
            for token in ("default", "seam", "mouth", "cap", "internal")
        ),
        sorted(groups),
        "no default/seam/mouth/cap/internal group",
    )
    points, mapping, coordinate_error = _round_trip_node_mapping(
        mesh, records, prefix
    )
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    hex_tags, read_hexes = _group_elements(
        hex_type, groups["fluid"]["entities"], 8
    )
    mapped_hexes = mapping[read_hexes]
    require(
        records,
        f"{prefix}.Hex8_signatures",
        _signature_equal(mapped_hexes, mesh.hexes),
        mapped_hexes.shape,
        mesh.hexes.shape,
    )

    canonical_census = hex_face_census(mesh.hexes)
    external = canonical_census["counts"] == 1
    owner_by_face = {
        tuple(face): int(owner)
        for face, owner in zip(
            canonical_census["faces"][external],
            canonical_census["owner"][external],
        )
    }
    patch_counts: dict[str, int] = {}
    outward: dict[str, float] = {}
    for name in PATCH_NAMES:
        _tags, read_quads = _group_elements(
            quad_type, groups[name]["entities"], 4
        )
        mapped = mapping[read_quads]
        require(
            records,
            f"{prefix}.{name}.Quad4_signatures",
            _signature_equal(mapped, mesh.boundary_quads[name]),
            mapped.shape,
            mesh.boundary_quads[name].shape,
        )
        vertices = mesh.points_m[_indices_for_tags(mesh.node_tags, mapped)]
        centres, areas = base._quad_geometry(vertices)
        owners = np.asarray(
            [
                owner_by_face[tuple(sorted(map(int, face)))]
                for face in mapped
            ],
            dtype=np.int64,
        )
        projection = np.einsum(
            "ij,ij->i", areas, centres - mesh.cell_centres_m[owners]
        )
        require(
            records,
            f"{prefix}.{name}.outward",
            bool(np.all(projection > 0.0)),
            float(projection.min()),
            ">0",
        )
        patch_counts[name] = len(mapped)
        outward[name] = float(projection.min())

    types, tags_by_type, _ = gmsh.model.mesh.getElements()
    type_counts = {
        int(kind): len(tags)
        for kind, tags in zip(types, tags_by_type)
    }
    expected_counts = {
        quad_type: sum(len(values) for values in mesh.boundary_quads.values()),
        hex_type: len(mesh.hexes),
    }
    require(
        records,
        f"{prefix}.only_Quad4_and_Hex8",
        type_counts == expected_counts,
        type_counts,
        expected_counts,
    )
    min_det = np.asarray(
        gmsh.model.mesh.getElementQualities(hex_tags, "minDetJac"),
        dtype=np.float64,
    )
    min_sicn = np.asarray(
        gmsh.model.mesh.getElementQualities(hex_tags, "minSICN"),
        dtype=np.float64,
    )
    require(
        records,
        f"{prefix}.positive_minDetJac_and_minSICN",
        np.all(min_det > 0.0) and np.all(min_sicn > 0.0),
        {
            "minDetJac": float(min_det.min()),
            "minSICN": float(min_sicn.min()),
        },
        "both >0",
    )
    return {
        "format": format_name,
        "path": path.name,
        "sha256": sha256_file(path),
        "points": len(points),
        "Hex8": len(mapped_hexes),
        "patch_counts": patch_counts,
        "physical_groups": groups,
        "coordinate_max_error_m": coordinate_error,
        "minimum_minDetJac": float(min_det.min()),
        "minimum_minSICN": float(min_sicn.min()),
        "minimum_outward_projection_m3": outward,
    }


def _write_vtu(
    mesh: BodyFittedMesh, case_dir: Path
) -> tuple[Path, Path]:
    volume_path = case_dir / "volume_hex.vtu"
    boundary_path = case_dir / "boundary_quads.vtu"
    volume_data = {
        **{name: [np.asarray(values)] for name, values in mesh.cell_fields.items()},
        **{name: [np.asarray(values)] for name, values in mesh.cell_metrics.items()},
        "cell_tag": [mesh.cell_tags],
        "physical_id": [
            np.full(len(mesh.hexes), PHYSICAL_IDS["fluid"], dtype=np.int32)
        ],
    }
    meshio.write(
        volume_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("hexahedron", mesh.hexes.astype(np.int64) - 1)],
            cell_data=volume_data,
            field_data={
                "fluid": np.asarray([PHYSICAL_IDS["fluid"], 3], dtype=np.int32)
            },
        ),
        file_format="vtu",
        binary=True,
    )
    boundary = np.concatenate(
        [mesh.boundary_quads[name] for name in PATCH_NAMES]
    )
    patch_ids = np.concatenate(
        [
            np.full(
                len(mesh.boundary_quads[name]),
                PHYSICAL_IDS[name],
                dtype=np.int32,
            )
            for name in PATCH_NAMES
        ]
    )
    meshio.write(
        boundary_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("quad", boundary.astype(np.int64) - 1)],
            cell_data={"patch_id": [patch_ids]},
            field_data={
                name: np.asarray([PHYSICAL_IDS[name], 2], dtype=np.int32)
                for name in PATCH_NAMES
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return volume_path, boundary_path


def _validate_vtu(
    mesh: BodyFittedMesh,
    volume_path: Path,
    boundary_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    volume = meshio.read(volume_path)
    boundary = meshio.read(boundary_path)
    expected_boundary = (
        np.concatenate([mesh.boundary_quads[name] for name in PATCH_NAMES])
        .astype(np.int64)
        - 1
    )
    require(
        records,
        "vtu.volume.Hex8_coordinates_connectivity",
        np.array_equal(volume.points, mesh.points_m)
        and np.array_equal(
            volume.cells_dict.get("hexahedron"),
            mesh.hexes.astype(np.int64) - 1,
        ),
        {
            "points": len(volume.points),
            "cell_types": sorted(volume.cells_dict),
        },
        {"points": len(mesh.points_m), "cell_types": ["hexahedron"]},
    )
    require(
        records,
        "vtu.boundary.Quad4_coordinates_connectivity",
        np.array_equal(boundary.points, mesh.points_m)
        and np.array_equal(boundary.cells_dict.get("quad"), expected_boundary),
        {
            "points": len(boundary.points),
            "cell_types": sorted(boundary.cells_dict),
        },
        {"points": len(mesh.points_m), "cell_types": ["quad"]},
    )
    expected_fields = {
        **mesh.cell_fields,
        **mesh.cell_metrics,
        "cell_tag": mesh.cell_tags,
        "physical_id": np.full(
            len(mesh.hexes), PHYSICAL_IDS["fluid"], dtype=np.int32
        ),
    }
    wrong = {
        name: len(
            volume.cell_data_dict.get(name, {}).get("hexahedron", ())
        )
        for name, expected in expected_fields.items()
        if not np.array_equal(
            volume.cell_data_dict.get(name, {}).get("hexahedron"),
            expected,
        )
    }
    require(records, "vtu.volume.fields_exact", not wrong, wrong, [])
    return {
        "volume_sha256": sha256_file(volume_path),
        "boundary_sha256": sha256_file(boundary_path),
        "Hex8": len(mesh.hexes),
        "Quad4": len(expected_boundary),
    }


def _write_npz(mesh: BodyFittedMesh, path: Path) -> None:
    arrays: dict[str, Any] = {
        "points_m": mesh.points_m,
        "node_tags": mesh.node_tags,
        "hexes": mesh.hexes,
        "cell_tags": mesh.cell_tags,
        "cell_centres_m": mesh.cell_centres_m,
        "metadata_json": np.asarray(
            json.dumps(dict(mesh.metadata), sort_keys=True)
        ),
    }
    arrays.update(
        {
            f"boundary_{name}": values
            for name, values in mesh.boundary_quads.items()
        }
    )
    arrays.update(
        {f"field_{name}": values for name, values in mesh.cell_fields.items()}
    )
    arrays.update(
        {f"metric_{name}": values for name, values in mesh.cell_metrics.items()}
    )
    np.savez_compressed(path, **arrays)


def _validate_npz(
    mesh: BodyFittedMesh, path: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        comparisons = {
            "points_m": np.array_equal(archive["points_m"], mesh.points_m),
            "node_tags": np.array_equal(archive["node_tags"], mesh.node_tags),
            "hexes": np.array_equal(archive["hexes"], mesh.hexes),
            "cell_tags": np.array_equal(archive["cell_tags"], mesh.cell_tags),
            "cell_centres_m": np.array_equal(
                archive["cell_centres_m"], mesh.cell_centres_m
            ),
            "metadata": json.loads(str(archive["metadata_json"]))
            == dict(mesh.metadata),
            **{
                f"boundary_{name}": np.array_equal(
                    archive[f"boundary_{name}"], values
                )
                for name, values in mesh.boundary_quads.items()
            },
            **{
                f"field_{name}": np.array_equal(
                    archive[f"field_{name}"], values
                )
                for name, values in mesh.cell_fields.items()
            },
            **{
                f"metric_{name}": np.array_equal(
                    archive[f"metric_{name}"], values
                )
                for name, values in mesh.cell_metrics.items()
            },
        }
    require(
        records,
        "npz.exact_round_trip",
        all(comparisons.values()),
        [name for name, passed in comparisons.items() if not passed],
        [],
    )
    return {"sha256": sha256_file(path), "comparisons": comparisons}


def _write_surface_quality(
    master: MasterMesh,
    params: base.BearingParams,
    path: Path,
) -> dict[str, Any]:
    quality = _master_quad_quality(master, params)
    indices = _indices_for_tags(master.node_tags, master.quads)
    centres = master.points_mm[indices].mean(axis=1)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "master_quad_index",
                "block_id",
                "pressure_feed",
                "x_mm",
                "z_mm",
                "area_mm2",
                "signed_area_mm2",
                "scaled_jacobian",
            )
        )
        for index in range(len(master.quads)):
            writer.writerow(
                (
                    index,
                    int(master.block_id[index]),
                    int(master.pressure_mask[index]),
                    centres[index, 0],
                    centres[index, 2],
                    quality["area_mm2"][index],
                    quality["signed_area_mm2"][index],
                    quality["scaled_jacobian"][index],
                )
            )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "minimum_scaled_jacobian": float(quality["scaled_jacobian"].min()),
    }


def _fluent_oq_projection(
    mesh: BodyFittedMesh, master: MasterMesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    values = mesh.cell_metrics.get("fluent_orthogonal_quality")
    if values is None:
        return None
    n_gap = int(mesh.metadata["n_gap"])
    expected_master = np.repeat(
        np.arange(len(master.quads), dtype=np.uint32), n_gap
    )
    expected_gap = np.tile(np.arange(n_gap, dtype=np.uint16), len(master.quads))
    if not (
        np.array_equal(mesh.cell_fields["master_quad_index"], expected_master)
        and np.array_equal(mesh.cell_fields["gap_index"], expected_gap)
    ):
        raise BodyFittedError(
            "Fluent OQ projection requires master-major gap-layer ordering"
        )
    by_master_and_gap = np.asarray(values).reshape(len(master.quads), n_gap)
    return (
        by_master_and_gap,
        by_master_and_gap.min(axis=1),
        by_master_and_gap.argmin(axis=1).astype(np.uint16),
    )


def _write_visualizations(
    mesh: BodyFittedMesh, master: MasterMesh, case_dir: Path
) -> dict[str, Any]:
    viz = case_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    master_path = viz / "master_surface.vtu"
    master_connectivity = _indices_for_tags(master.node_tags, master.quads)
    projection = _fluent_oq_projection(mesh, master)
    master_cell_data: dict[str, list[np.ndarray]] = {
        "block_id": [master.block_id],
        "pressure_feed": [master.pressure_mask.astype(np.uint8)],
        "diagnostic_only": [
            np.ones(len(master.quads), dtype=np.uint8)
        ],
        "solver_eligible": [
            np.zeros(len(master.quads), dtype=np.uint8)
        ],
    }
    if projection is not None:
        _, surface_oq, worst_gap = projection
        master_cell_data.update(
            {
                "fluent_orthogonal_quality_min_through_gap": [surface_oq],
                "fluent_orthogonal_quality_worst_gap_index": [worst_gap],
            }
        )
    meshio.write(
        master_path,
        meshio.Mesh(
            points=master.points_mm * SI_PER_MM,
            cells=[("quad", master_connectivity)],
            cell_data=master_cell_data,
        ),
        file_format="vtu",
        binary=True,
    )
    inlet_path = viz / "pressure_feed_only.vtu"
    inlet_quads = mesh.boundary_quads["pressure_feed"]
    used, connectivity = np.unique(inlet_quads.ravel(), return_inverse=True)
    meshio.write(
        inlet_path,
        meshio.Mesh(
            points=mesh.points_m[used.astype(np.int64) - 1],
            cells=[("quad", connectivity.reshape(-1, 4))],
            cell_data={
                "patch_id": [
                    np.full(
                        len(inlet_quads),
                        PHYSICAL_IDS["pressure_feed"],
                        dtype=np.int32,
                    )
                ],
                "diagnostic_only": [
                    np.ones(len(inlet_quads), dtype=np.uint8)
                ],
                "solver_eligible": [
                    np.zeros(len(inlet_quads), dtype=np.uint8)
                ],
            },
        ),
        file_format="vtu",
        binary=True,
    )
    wrapped = np.abs(
        (mesh.cell_fields["theta_deg"] + 180.0) % 360.0 - 180.0
    )
    keep = wrapped >= 30.0
    quality_name = (
        "fluent_orthogonal_quality"
        if "fluent_orthogonal_quality" in mesh.cell_metrics
        else (
            "minSICN"
            if "minSICN" in mesh.cell_metrics
            else "gauss_min_det"
        )
    )
    cutaway_path = viz / "cutaway_exact.vtu"
    meshio.write(
        cutaway_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[
                (
                    "hexahedron",
                    mesh.hexes[keep].astype(np.int64) - 1,
                )
            ],
            cell_data={
                "block_id": [mesh.cell_fields["block_id"][keep]],
                "gap_index": [mesh.cell_fields["gap_index"][keep]],
                quality_name: [mesh.cell_metrics[quality_name][keep]],
                "diagnostic_only": [
                    np.ones(np.count_nonzero(keep), dtype=np.uint8)
                ],
                "solver_eligible": [
                    np.zeros(np.count_nonzero(keep), dtype=np.uint8)
                ],
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return {
        "master_surface": str(master_path.relative_to(case_dir)),
        "pressure_feed": str(inlet_path.relative_to(case_dir)),
        "cutaway": str(cutaway_path.relative_to(case_dir)),
        "cutaway_retained_Hex8": int(np.count_nonzero(keep)),
        "quality_field": quality_name,
        "surface_oq_projection": (
            None
            if projection is None
            else {
                "field": "fluent_orthogonal_quality_min_through_gap",
                "meaning": (
                    "minimum full-3D cell OQ across all physical gap layers"
                ),
                "minimum": float(projection[1].min()),
            }
        ),
        "solve_eligible": False,
        "coordinates_distorted": False,
    }


def _write_diagnostic_gmsh_master(
    master: MasterMesh, case_dir: Path
) -> dict[str, Any]:
    """Write a surface-only Gmsh display mesh with no CFD physical groups."""
    path = case_dir / "viz" / "DIAGNOSTIC_ONLY_master_surface.msh"
    connectivity = _indices_for_tags(master.node_tags, master.quads)
    meshio.write(
        path,
        meshio.Mesh(
            points=master.points_mm * SI_PER_MM,
            cells=[("quad", connectivity)],
            cell_data={
                "gmsh:physical": [
                    np.zeros(len(master.quads), dtype=np.int32)
                ],
                "gmsh:geometrical": [
                    np.ones(len(master.quads), dtype=np.int32)
                ],
                "block_id": [master.block_id],
                "pressure_feed": [master.pressure_mask.astype(np.uint8)],
                "diagnostic_only": [
                    np.ones(len(master.quads), dtype=np.uint8)
                ],
                "solver_eligible": [
                    np.zeros(len(master.quads), dtype=np.uint8)
                ],
            },
        ),
        file_format="gmsh22",
        binary=True,
    )
    reopened = meshio.read(path)
    if (
        set(reopened.cells_dict) != {"quad"}
        or len(reopened.cells_dict["quad"]) != len(master.quads)
    ):
        raise BodyFittedError("diagnostic Gmsh master-surface round trip failed")
    return {
        "path": str(path.relative_to(case_dir)),
        "Quad4": len(master.quads),
        "solve_eligible": False,
        "physical_groups": False,
        "sha256": sha256_file(path),
    }


def _write_fluent_oq_overview(
    mesh: BodyFittedMesh,
    master: MasterMesh,
    case_dir: Path,
) -> dict[str, Any] | None:
    projection = _fluent_oq_projection(mesh, master)
    if projection is None:
        return None
    by_master_and_gap, surface_oq, _ = projection
    threshold = float(mesh.metadata["minimum_fluent_orthogonal_quality"])
    values = mesh.cell_metrics["fluent_orthogonal_quality"]
    centres = master.points_mm[
        _indices_for_tags(master.node_tags, master.quads)
    ].mean(axis=1)
    theta_deg = np.degrees(
        np.mod(np.arctan2(centres[:, 0], -centres[:, 1]), 2.0 * math.pi)
    )
    colour_norm = matplotlib.colors.Normalize(
        vmin=min(threshold, float(values.min())),
        vmax=1.0,
    )
    colour_map = "viridis"
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0))

    surface_axis = figure.add_subplot(grid[:, 0], projection="3d")
    artist = surface_axis.scatter(
        centres[:, 0],
        centres[:, 1],
        centres[:, 2],
        c=surface_oq,
        s=2.0,
        cmap=colour_map,
        norm=colour_norm,
        linewidths=0,
        rasterized=True,
    )
    surface_axis.set_xlabel("x [mm]")
    surface_axis.set_ylabel("y [mm]")
    surface_axis.set_zlabel("z [mm]")
    surface_axis.set_title("Physical conical film surface")
    surface_axis.view_init(elev=24, azim=-58)
    surface_axis.set_box_aspect((1.0, 1.0, 1.35))

    unwrapped_axis = figure.add_subplot(grid[0, 1])
    unwrapped_axis.scatter(
        theta_deg,
        centres[:, 2],
        c=surface_oq,
        s=3.0,
        cmap=colour_map,
        norm=colour_norm,
        linewidths=0,
        rasterized=True,
    )
    unwrapped_axis.set_xlim(0.0, 360.0)
    unwrapped_axis.set_xlabel("circumferential angle [deg]")
    unwrapped_axis.set_ylabel("axial coordinate [mm]")
    unwrapped_axis.set_title("Unwrapped surface projection")
    unwrapped_axis.grid(alpha=0.2)

    layer_axis = figure.add_subplot(grid[1, 1])
    layers = np.arange(by_master_and_gap.shape[1])
    layer_axis.plot(
        layers,
        by_master_and_gap.min(axis=0),
        marker="o",
        label="minimum",
    )
    layer_axis.plot(
        layers,
        by_master_and_gap.mean(axis=0),
        marker=".",
        label="mean",
    )
    layer_axis.axhline(
        threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"acceptance {threshold:g}",
    )
    layer_axis.set_xticks(layers)
    layer_axis.set_xlabel("physical Hex8 gap-layer index")
    layer_axis.set_ylabel("Fluent-equivalent OQ")
    layer_axis.set_title(
        f"All {len(layers)} thickness layers (indices 0–{len(layers) - 1})"
    )
    layer_axis.grid(alpha=0.25)
    layer_axis.legend()

    colour_axis = figure.add_axes((0.915, 0.18, 0.018, 0.64))
    figure.colorbar(
        artist,
        cax=colour_axis,
        label="standard Fluent-equivalent Orthogonal Quality",
    )
    minimum_index = int(np.argmin(values))
    figure.suptitle(
        "Full 3D Orthogonal Quality — conservative surface projection",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        (
            f"Each surface point is the minimum over all "
            f"{by_master_and_gap.shape[1]} real 3D gap cells; this is not a "
            f"2D mesh metric. Global min={values[minimum_index]:.6f}, "
            f"mean={values.mean():.6f}, cells below {threshold:g}="
            f"{np.count_nonzero(values < threshold):,}."
        ),
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.05, right=0.88, top=0.92, bottom=0.08, wspace=0.22, hspace=0.25
    )

    viz = case_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    png = viz / "full_3d_fluent_oq_overview.png"
    pdf = viz / "full_3d_fluent_oq_overview.pdf"
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)

    worst_master = int(mesh.cell_fields["master_quad_index"][minimum_index])
    summary = {
        "paths_relative_to": "case root",
        "metric": "standard Fluent-equivalent Orthogonal Quality",
        "projection": "minimum across all physical 3D gap cells",
        "gap_layers": by_master_and_gap.shape[1],
        "minimum": float(values[minimum_index]),
        "mean": float(values.mean()),
        "threshold": threshold,
        "cells_below_threshold": int(np.count_nonzero(values < threshold)),
        "minimum_by_gap_layer": by_master_and_gap.min(axis=0).tolist(),
        "circular_ogrid_minimum": (
            float(values[mesh.cell_fields["block_id"] != 9].min())
            if master.metadata["topology"] == "ogrid"
            else None
        ),
        "feed_column_minimum": float(
            by_master_and_gap[master.pressure_mask].min()
        ),
        "worst_cell_tag": int(mesh.cell_tags[minimum_index]),
        "worst_master_quad_index": worst_master,
        "worst_gap_layer": int(mesh.cell_fields["gap_index"][minimum_index]),
        "worst_cell_centre_m": mesh.cell_centres_m[minimum_index].tolist(),
        "outputs": {
            "png": str(png.relative_to(case_dir)),
            "pdf": str(pdf.relative_to(case_dir)),
            "surface_vtu": "viz/master_surface.vtu",
            "volume_vtu": "volume_hex.vtu",
        },
    }
    summary_path = viz / "full_3d_fluent_oq_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary | {"summary": str(summary_path.relative_to(case_dir))}


def _write_plots(
    mesh: BodyFittedMesh,
    master: MasterMesh,
    params: base.BearingParams,
    inlet: InletSpec,
    case_dir: Path,
) -> dict[str, Any]:
    plots = case_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    indices = _indices_for_tags(master.node_tags, master.quads)
    vertices = master.points_mm[indices]
    quality = _master_quad_quality(master, params)
    footprint = plots / "footprint.png"
    figure, axis = plt.subplots(figsize=(7, 6))
    for quad in vertices[master.pressure_mask]:
        closed = np.vstack((quad[:, (0, 2)], quad[0, (0, 2)]))
        axis.plot(closed[:, 0], closed[:, 1], color="tab:blue", linewidth=0.5)
    angle = np.linspace(0.0, 2.0 * math.pi, 361)
    radius = float(master.metadata["effective_radius_mm"])
    axis.plot(
        radius * np.cos(angle),
        inlet.axial_position_mm + radius * np.sin(angle),
        color="black",
        linewidth=1.2,
        label="analytic rim",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title(f"{master.metadata['topology']} pressure footprint")
    axis.legend()
    figure.tight_layout()
    figure.savefig(footprint, dpi=180)
    plt.close(figure)

    quality_path = plots / "master_quality.png"
    centres = vertices.mean(axis=1)
    figure, axis = plt.subplots(figsize=(7, 6))
    artist = axis.scatter(
        centres[:, 0],
        centres[:, 2],
        c=quality["scaled_jacobian"],
        s=5,
        cmap="viridis",
    )
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title("Master minimum corner scaled Jacobian")
    figure.colorbar(artist, ax=axis)
    figure.tight_layout()
    figure.savefig(quality_path, dpi=180)
    plt.close(figure)

    background_block = 0 if master.metadata["topology"] == "tensor-warp" else 9
    local = master.block_id != background_block
    local_vertices = vertices[local][:, :, (0, 2)]
    local_pressure = master.pressure_mask[local]
    local_mesh = plots / "local_master_mesh.png"
    figure, axis = plt.subplots(figsize=(7, 6))
    for quad, pressure in zip(local_vertices, local_pressure):
        closed = np.vstack((quad, quad[0]))
        axis.fill(
            quad[:, 0],
            quad[:, 1],
            color="tab:blue" if pressure else "0.92",
            alpha=0.45 if pressure else 0.2,
        )
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color="tab:blue" if pressure else "0.45",
            linewidth=0.55,
        )
    rim_points = master.points_mm[
        _indices_for_tags(master.node_tags, master.rim_node_tags)
    ][:, (0, 2)]
    axis.plot(
        *np.vstack((rim_points, rim_points[0])).T,
        color="black",
        linewidth=1.8,
        label="analytic rim nodes + chord edges",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title(f"{master.metadata['topology']} local master mesh")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(local_mesh, dpi=180)
    plt.close(figure)
    result = {
        "footprint": str(footprint.relative_to(case_dir)),
        "quality": str(quality_path.relative_to(case_dir)),
        "local_master_mesh": str(local_mesh.relative_to(case_dir)),
    }
    fluent_oq = _write_fluent_oq_overview(mesh, master, case_dir)
    if fluent_oq is not None:
        result["fluent_orthogonal_quality"] = fluent_oq
    return result


def _copy_context_step(
    params_path: Path, requested: Path | None, case_dir: Path
) -> dict[str, Any]:
    candidates = [
        requested,
        params_path.parent / "context_assembly.step",
        params_path.parent.with_name(
            f"{params_path.parent.name}.rejected-step"
        )
        / "context_assembly.step",
        (
            params_path.parents[1]
            / "ansys_surface_inlet_default"
            / "VISUAL_CONTEXT_ONLY_context_assembly.step"
            if params_path.parent.name == "strict_default"
            else None
        ),
    ]
    source = next(
        (path for path in candidates if path is not None and path.is_file()),
        None,
    )
    warning = (
        "VISUAL CONTEXT ONLY: not solver geometry; the circular inlet is a "
        "volume-mesh boundary partition and STEP carries no CFD physical groups."
    )
    if source is None:
        path = case_dir / "VISUAL_CONTEXT_WARNING.txt"
        path.write_text(
            warning + "\nNo matching context STEP was supplied or found.\n",
            encoding="utf-8",
        )
        return {"status": "NOT_AVAILABLE", "warning": warning, "notice": path.name}
    target = case_dir / "VISUAL_CONTEXT_ONLY_context_assembly.step"
    shutil.copy2(source, target)
    return {
        "status": "COPIED",
        "source": str(source),
        "path": target.name,
        "sha256": sha256_file(target),
        "warning": warning,
    }


def _file_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "validation_report.json"}
    }


def export_body_fitted_case(
    mesh: BodyFittedMesh,
    master: MasterMesh,
    params: base.BearingParams,
    inlet: InletSpec,
    case_dir: Path,
    *,
    params_path: Path | None = None,
    openfoam: Literal["auto", "required", "skip"] = "skip",
    ansys: Literal["auto", "required", "skip"] = "required",
    minimum_fluent_orthogonal_quality: float | None = None,
    context_step: Path | None = None,
    records: list[dict[str, Any]] | None = None,
) -> tuple[BodyFittedMesh, dict[str, Any]]:
    """Write and re-open every solver-neutral body-fitted artifact."""
    records = [] if records is None else records
    case_dir.mkdir(parents=True, exist_ok=True)
    initialized = False
    logger_started = False
    gmsh_log: list[str] = []
    round_trips: list[dict[str, Any]] = []
    try:
        gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
        initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()
        logger_started = True
        discrete = staircase.add_discrete_model(
            mesh,
            records,
            f"body_fitted_{master.metadata['topology']}",
        )
        groups = _physical_groups()
        mesh = _add_gmsh_metrics(mesh, records)
        body_validation = validate_body_fitted_mesh(
            mesh,
            master,
            params,
            inlet,
            records,
            require_gmsh_metrics=True,
        )
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        msh41 = case_dir / "structured_hex.msh"
        msh22 = case_dir / "structured_hex_openfoam.msh"
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.write(str(msh41))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(msh22))
        build_info = gmsh.option.getString("General.BuildInfo")
        cgns_available = "Cgns" in build_info
        if ansys == "required" and not cgns_available:
            raise BodyFittedError("--ansys required but Gmsh has no CGNS writer")
        cgns = case_dir / "bearing_body_fitted_hex.cgns"
        if ansys != "skip" and cgns_available:
            gmsh.write(str(cgns))
            sanitize_gmsh_cgns(
                cgns, expected_surface_boundaries=len(PATCH_NAMES)
            )
        round_trips.append(
            _audit_mesh_round_trip(msh41, mesh, records, "GMSH_4_1")
        )
        round_trips.append(
            _audit_mesh_round_trip(msh22, mesh, records, "GMSH_2_2_ASCII")
        )
        if cgns.is_file():
            round_trips.append(
                _audit_mesh_round_trip(cgns, mesh, records, "CGNS")
            )
        elif ansys == "auto":
            records.append(
                {
                    "name": "ansys.cgns",
                    "status": "SKIPPED",
                    "actual": "CGNS writer unavailable",
                    "expected": "optional CGNS export",
                    "tolerance": None,
                    "mandatory": False,
                }
            )
    finally:
        if logger_started:
            gmsh_log = [str(line) for line in gmsh.logger.get()]
            gmsh.logger.stop()
        if initialized:
            gmsh.finalize()

    (case_dir / "gmsh_body_fitted.log").write_text(
        "\n".join(gmsh_log) + "\n", encoding="utf-8"
    )
    npz_path = case_dir / "mesh_arrays.npz"
    _write_npz(mesh, npz_path)
    npz = _validate_npz(mesh, npz_path, records)
    fluent_result: dict[str, Any] = {
        "status": "NOT_REQUESTED",
        "minimum_orthogonal_quality": None,
    }
    if minimum_fluent_orthogonal_quality is not None:
        fluent_model = fluent_legacy_msh.build_fluent_legacy_mesh(npz_path)
        fluent_oq = fluent_legacy_msh.orthogonal_quality(fluent_model)
        mesh = replace(
            mesh,
            cell_metrics=dict(mesh.cell_metrics)
            | {"fluent_orthogonal_quality": fluent_oq},
            metadata=dict(mesh.metadata)
            | {
                "minimum_fluent_orthogonal_quality": (
                    minimum_fluent_orthogonal_quality
                )
            },
        )
        fluent_dir = case_dir / "fluent"
        fluent_dir.mkdir(parents=True, exist_ok=True)
        case_name = str(master.metadata["case_name"])
        fluent_path = fluent_dir / f"{case_name}.msh"
        independent_audit_path = (
            fluent_dir / "independent_centroid_oq_audit.json"
        )
        independent_audit = (
            fluent_legacy_msh.independent_centroid_orthogonal_quality_audit(
                fluent_model,
                minimum_fluent_orthogonal_quality,
            )
        )
        if not independent_audit["passed"]:
            raise BodyFittedError(
                "independent reconstructed-centroid Orthogonal Quality "
                f"{independent_audit['minimum_orthogonal_quality']:.12g} is "
                f"below required {minimum_fluent_orthogonal_quality:.12g}"
            )
        del fluent_model
        fluent_result = fluent_legacy_msh.write_fluent_legacy_mesh(
            npz_path,
            fluent_path,
            minimum_fluent_orthogonal_quality,
        )
        fluent_report_path = fluent_dir / "fluent_native_validation.json"
        independent_audit_path.write_text(
            json.dumps(independent_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fluent_result.update(
            {
                "source_npz": str(npz_path.relative_to(case_dir)),
                "output": str(fluent_path.relative_to(case_dir)),
                "path": str(fluent_path.relative_to(case_dir)),
                "validation_report": str(
                    fluent_report_path.relative_to(case_dir)
                ),
                "independent_centroid_audit": str(
                    independent_audit_path.relative_to(case_dir)
                ),
            }
        )
        fluent_report_path.write_text(
            json.dumps(fluent_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    volume_vtu, boundary_vtu = _write_vtu(mesh, case_dir)
    vtu = _validate_vtu(mesh, volume_vtu, boundary_vtu, records)
    surface_quality = _write_surface_quality(
        master, params, case_dir / "surface_quality.csv"
    )
    visualizations = _write_visualizations(mesh, master, case_dir)
    plots = _write_plots(mesh, master, params, inlet, case_dir)
    source_params = params.source if params_path is None else params_path
    context = _copy_context_step(source_params, context_step, case_dir)
    openfoam_result = staircase.audit_openfoam(
        openfoam, case_dir, mesh, records, case_dir
    )

    physical = {
        "coordinate_unit": "m",
        "volume": {
            "fluid": {
                "physical_id": PHYSICAL_IDS["fluid"],
                "entity_tag": VOLUME_ENTITY,
                "Hex8": len(mesh.hexes),
            }
        },
        "boundaries": {
            name: {
                "physical_id": PHYSICAL_IDS[name],
                "entity_tag": SURFACE_ENTITIES[name],
                "Quad4": len(mesh.boundary_quads[name]),
            }
            for name in PATCH_NAMES
        },
        "contains_feed_tube": False,
        "internal_rim_patch": False,
    }
    (case_dir / "physical_groups.json").write_text(
        json.dumps(physical, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (case_dir / "zones.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "dimension", "element_type", "count", "role"))
        writer.writerow(("fluid", 3, "HEXA_8", len(mesh.hexes), "fluid volume"))
        roles = {
            "journal_wall": "moving wall",
            "stationary_wall": "stationary wall",
            "axial_end_z0": "axial outlet candidate",
            "axial_end_zL": "axial outlet candidate",
            "pressure_feed": "pressure inlet",
        }
        for name in PATCH_NAMES:
            writer.writerow(
                (name, 2, "QUAD_4", len(mesh.boundary_quads[name]), roles[name])
            )
    cgns_name = "bearing_body_fitted_hex.cgns"
    cgns_ready = (case_dir / cgns_name).is_file()
    native_fluent_path = fluent_result.get("path")
    native_instruction = (
        f"Preferred import: read {native_fluent_path} directly in Fluent.\n"
        if native_fluent_path
        else ""
    )
    cgns_instruction = (
        f"Import {cgns_name} through File > Import > CGNS > Mesh.\n"
        if cgns_ready
        else (
            f"CGNS was not generated because the ANSYS mode was {ansys!r}; "
            "rerun with --ansys required before Fluent import.\n"
        )
    )
    ansys_instructions = (
        "ANSYS FLUENT BODY-FITTED MESH HANDOFF\n\n"
        f"Static status: STATICALLY_VALIDATED_NOT_IMPORTED\n"
        + native_instruction
        + cgns_instruction
        + "Run Mesh Check; confirm one fluid Hex8 zone and exactly the five named "
        "Quad4 boundary zones in zones.csv.\n"
        "pressure_feed is the zero-length circular surface inlet on the bore.\n"
        "There is no feed tube, recess, mouth cap, journal intrusion, or internal "
        "rim patch.\n"
        "The optional STEP file is visual context only and must not be used as "
        "solver geometry.\n"
        "A successful Gmsh CGNS round trip is not a live Fluent import pass.\n"
    )
    (case_dir / "ANSYS_IMPORT.txt").write_text(
        ansys_instructions,
        encoding="utf-8",
    )
    report = {
        "overall": "PASS",
        "readiness": "STATICALLY_VALIDATED_NOT_IMPORTED",
        "master_validation": validate_master_mesh(
            master, params, inlet, []
        ),
        "body_validation": body_validation,
        "gmsh": {
            "discrete_registration": discrete,
            "physical_groups": groups,
            "round_trips": round_trips,
            "log": "gmsh_body_fitted.log",
        },
        "vtu_round_trip": vtu,
        "npz_round_trip": npz,
        "surface_quality": surface_quality,
        "visualizations": visualizations,
        "plots": plots,
        "context_step": context,
        "openfoam": openfoam_result,
        "fluent": fluent_result,
        "ansys": {
            "mode": ansys,
            "cgns_status": "WRITTEN" if cgns_ready else "NOT_WRITTEN",
            "path": cgns_name if cgns_ready else None,
        },
        "validation_records": records,
    }
    manifest = {
        "schema_version": 1,
        "overall": "PASS",
        "readiness": report["readiness"],
        "case_name": case_dir.name,
        "topology": master.metadata["topology"],
        "geometry_mode": master.metadata["geometry_mode"],
        "nominal_geometry": master.metadata["nominal_geometry"],
        "research_variant": master.metadata["research_variant"],
        "effective_radius_mm": master.metadata["effective_radius_mm"],
        "radial_bias_mm": master.metadata["radial_bias_mm"],
        "area_correction_mm2": master.metadata["area_correction_mm2"],
        "contains_only_Hex8_volume_cells": True,
        "contains_only_Quad4_boundary_faces": True,
        "contains_feed_tube": False,
        "contains_pressure_feed_patch": True,
        "canonical_arrays": "mesh_arrays.npz",
        "coordinate_unit": "m",
        "scale_to_m_applied_exactly_once": SI_PER_MM,
        "params": asdict(params) | {"source": str(params.source)},
        "inlet": asdict(inlet),
        "master_metadata": dict(master.metadata),
        "counts": body_validation["counts"],
        "physical_groups": physical,
        "context_step": context,
        "fluent": fluent_result,
        "ansys": report["ansys"],
    }
    report["files"] = _file_inventory(case_dir)
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (case_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return mesh, report


def _validate_case_inputs(inputs: BodyFittedCaseInputs) -> None:
    if not inputs.case_name or any(
        token in inputs.case_name for token in ("/", "\\", "..")
    ):
        raise BodyFittedError("case_name must be a nonempty path-safe label")
    if inputs.topology not in ("tensor-warp", "ogrid"):
        raise BodyFittedError(f"unknown topology {inputs.topology!r}")
    if inputs.geometry_mode not in ("inscribed", "equal-area"):
        raise BodyFittedError(f"unknown geometry mode {inputs.geometry_mode!r}")
    if inputs.n_gap < 1:
        raise BodyFittedError("n_gap must be positive")
    if inputs.smoothing_iterations < 0:
        raise BodyFittedError("smoothing_iterations must be nonnegative")
    if not 0.0 < inputs.smoothing_damping <= 1.0:
        raise BodyFittedError("smoothing_damping must be in (0, 1]")
    if inputs.smoothing_fixed_nodes not in (
        "all-interfaces",
        "background-and-rim",
    ):
        raise BodyFittedError(
            "smoothing_fixed_nodes must be all-interfaces or background-and-rim"
        )
    if inputs.quality_optimized_ogrid and inputs.topology != "ogrid":
        raise BodyFittedError(
            "quality_optimized_ogrid is only valid for the ogrid topology"
        )
    if (
        inputs.smoothing_fixed_nodes == "background-and-rim"
        and not inputs.quality_optimized_ogrid
    ):
        raise BodyFittedError(
            "background-and-rim smoothing is only valid for a "
            "quality-optimized O-grid"
        )
    if (
        inputs.minimum_fluent_orthogonal_quality is not None
        and not 0.0 < inputs.minimum_fluent_orthogonal_quality <= 1.0
    ):
        raise BodyFittedError(
            "minimum_fluent_orthogonal_quality must be in (0, 1]"
        )
    if inputs.openfoam not in ("auto", "required", "skip"):
        raise BodyFittedError("openfoam must be auto, required, or skip")
    if inputs.ansys not in ("auto", "required", "skip"):
        raise BodyFittedError("ansys must be auto, required, or skip")


def _publish_failure(
    outdir: Path,
    report: dict[str, Any],
    *,
    master: MasterMesh | None = None,
    mesh: BodyFittedMesh | None = None,
    params: base.BearingParams | None = None,
    inlet: InletSpec | None = None,
    params_path: Path | None = None,
    context_step: Path | None = None,
) -> None:
    stage = make_staging_directory(outdir)
    try:
        if (
            master is not None
            and mesh is not None
            and params is not None
            and inlet is not None
            and params_path is not None
        ):
            try:
                warning = (
                    "REJECTED BY A MANDATORY QUALITY GATE. VISUAL ONLY. "
                    "DO NOT SOLVE, CONVERT, OR IMPORT AS A CFD MESH."
                )
                (stage / "VISUAL_ONLY_DO_NOT_SOLVE.txt").write_text(
                    warning
                    + "\nThe VTU and surface-only MSH files exist only so the "
                    "rejected topology can be inspected.\n",
                    encoding="utf-8",
                )
                preview = {
                    "warning": warning,
                    "solve_eligible": False,
                    "visualizations": _write_visualizations(
                        mesh, master, stage
                    ),
                    "plots": _write_plots(
                        mesh, master, params, inlet, stage
                    ),
                    "surface_quality": _write_surface_quality(
                        master, params, stage / "surface_quality.csv"
                    ),
                    "gmsh_surface_preview": _write_diagnostic_gmsh_master(
                        master, stage
                    ),
                    "freecad_context": _copy_context_step(
                        params_path, context_step, stage
                    ),
                }
                report["visual_only_preview"] = preview
            except Exception as preview_error:
                report["visual_only_preview"] = {
                    "solve_eligible": False,
                    "preview_error": {
                        "type": type(preview_error).__name__,
                        "message": str(preview_error),
                    },
                }
        report["files"] = _file_inventory(stage)
        (stage / "failure_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        atomic_replace_directory(stage, outdir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run_body_fitted_case(inputs: BodyFittedCaseInputs) -> dict[str, Any]:
    """Build, validate, atomically export, and publish one named case."""
    inputs = replace(
        inputs,
        params=inputs.params.resolve(),
        outdir=inputs.outdir.resolve(),
        context_step=(
            None if inputs.context_step is None else inputs.context_step.resolve()
        ),
    )
    serialized_inputs = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(inputs).items()
    }
    stage: Path | None = None
    master: MasterMesh | None = None
    mesh: BodyFittedMesh | None = None
    master_validation: dict[str, Any] | None = None
    master_records: list[dict[str, Any]] = []
    body_records: list[dict[str, Any]] = []
    try:
        _validate_case_inputs(inputs)
        if not inputs.params.is_file():
            raise BodyFittedError(f"params file not found: {inputs.params}")
        if inputs.openfoam == "required" and not (
            shutil.which("gmshToFoam") and shutil.which("checkMesh")
        ):
            raise BodyFittedError(
                "--openfoam required but gmshToFoam/checkMesh are unavailable"
            )
        params = base.load_params(inputs.params)
        inlet = load_inlet_spec(inputs.params)
        if inputs.topology == "tensor-warp":
            master = build_tensor_warp_master(
                params,
                inlet,
                inputs.q,
                geometry_mode=inputs.geometry_mode,
                n_theta=inputs.n_theta,
                n_axial=inputs.n_axial,
            )
        else:
            master = build_ogrid_master(
                params,
                inlet,
                4 * inputs.q,
                inputs.inner_layers,
                inputs.outer_layers,
                geometry_mode=inputs.geometry_mode,
                n_theta=inputs.n_theta,
                n_axial=inputs.n_axial,
                quality_optimized=inputs.quality_optimized_ogrid,
                control_radius_factor=inputs.control_radius_factor,
                control_square_blend=inputs.control_square_blend,
                central_corner_radius_factor=(
                    inputs.central_corner_radius_factor
                ),
            )
        if inputs.smoothing_fixed_nodes == "background-and-rim":
            master = replace(
                master,
                fixed_node_tags=np.unique(
                    np.concatenate(
                        (
                            master.unchanged_node_tags,
                            master.rim_node_tags,
                        )
                    )
                ),
            )
        master = smooth_master_mesh(
            master,
            params,
            iterations=inputs.smoothing_iterations,
            damping=inputs.smoothing_damping,
        )
        master = _refresh_ogrid_corner_radius(master, inlet)
        master = replace(
            master,
            metadata=dict(master.metadata) | {"case_name": inputs.case_name},
        )
        master_validation = validate_master_mesh(
            master, params, inlet, master_records
        )
        mesh = sweep_master_mesh(
            master,
            params,
            n_gap=inputs.n_gap,
            gap_inflation_ratio=inputs.gap_inflation_ratio,
        )
        validate_body_fitted_mesh(
            mesh, master, params, inlet, body_records
        )
        stage = make_staging_directory(inputs.outdir)
        mesh, report = export_body_fitted_case(
            mesh,
            master,
            params,
            inlet,
            stage,
            params_path=inputs.params,
            openfoam=inputs.openfoam,
            ansys=inputs.ansys,
            minimum_fluent_orthogonal_quality=(
                inputs.minimum_fluent_orthogonal_quality
            ),
            context_step=inputs.context_step,
            records=[*master_records, *body_records],
        )
        report["inputs"] = serialized_inputs
        report["case_name"] = inputs.case_name
        report["master_validation"] = master_validation
        report["files"] = _file_inventory(stage)
        (stage / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path = stage / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["case_name"] = inputs.case_name
        manifest["inputs"] = serialized_inputs
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        atomic_replace_directory(stage, inputs.outdir)
        stage = None
        return report
    except BaseException as error:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if not isinstance(error, Exception):
            raise
        failure = {
            "overall": "FAIL",
            "case_name": inputs.case_name,
            "inputs": serialized_inputs,
            "solve_eligible_outputs_published": False,
            "validation_records": [*master_records, *body_records],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        if master_validation is not None:
            failure["master_validation"] = master_validation
        if mesh is not None:
            signed = mesh.cell_metrics["signed_volume_m3"]
            failure["body_validation"] = {
                "counts": {
                    "points": len(mesh.points_m),
                    "Hex8": len(mesh.hexes),
                    "boundary_Quad4": {
                        name: len(mesh.boundary_quads[name])
                        for name in PATCH_NAMES
                    },
                },
                "volume": {
                    "cell_sum_m3": float(signed.sum()),
                    "continuous_annulus_m3": params.exact_volume_m3,
                    "continuous_relative_error": relative_error(
                        float(signed.sum()), params.exact_volume_m3
                    ),
                },
                "quality": {
                    "minimum_signed_volume_m3": float(signed.min()),
                    "minimum_gauss_determinant": float(
                        mesh.cell_metrics["gauss_min_det"].min()
                    ),
                    "minimum_face_pyramid_m3": float(
                        mesh.cell_metrics["min_face_pyramid_m3"].min()
                    ),
                    "maximum_nonorthogonality_deg": float(
                        mesh.cell_metrics["max_nonorthogonality_deg"].max()
                    ),
                    "maximum_skewness": float(
                        mesh.cell_metrics["max_skewness"].max()
                    ),
                    "minimum_minSICN": None,
                    "minimum_minDetJac": None,
                },
            }
        rejected_by_gate = any(
            record.get("status") == "FAIL"
            for record in (*master_records, *body_records)
        )
        _publish_failure(
            inputs.outdir,
            failure,
            master=master if rejected_by_gate else None,
            mesh=mesh if rejected_by_gate else None,
            params=params if rejected_by_gate and mesh is not None else None,
            inlet=inlet if rejected_by_gate and mesh is not None else None,
            params_path=inputs.params,
            context_step=inputs.context_step,
        )
        raise BodyFittedRunError(str(error), failure) from error
