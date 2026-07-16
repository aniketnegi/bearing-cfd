#!/usr/bin/env python3
"""Analytic full-360 structured Hex8 mesh for the no-port bearing film."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Sequence

import gmsh
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "out" / ".matplotlib-cache"))
import matplotlib
import meshio
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshing.gmsh_brep_preflight import (
    atomic_replace_directory,
    make_staging_directory,
    relative_error,
    require,
    sha256_file,
)


SI_PER_MM = 1.0e-3
PHYSICAL_IDS = {
    "journal_wall": 101,
    "stationary_wall": 102,
    "axial_end_z0": 103,
    "axial_end_zL": 104,
    "fluid": 201,
}
RING_PHYSICAL_IDS = {
    "journal_z0_ring": 301,
    "journal_zL_ring": 302,
    "bushing_z0_ring": 303,
    "bushing_zL_ring": 304,
}
CURVE_ENTITIES = {
    "journal_z0_ring": 11,
    "journal_zL_ring": 12,
    "bushing_z0_ring": 13,
    "bushing_zL_ring": 14,
}
SURFACE_ENTITIES = {
    "journal_wall": 21,
    "stationary_wall": 22,
    "axial_end_z0": 23,
    "axial_end_zL": 24,
}
VOLUME_ENTITY = 31
HEX_FACES = np.asarray(
    [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
    ],
    dtype=np.uint8,
)
VIEW_NAMES = (
    "gap_um",
    "layer_index",
    "theta_deg",
    "axial_index",
    "cell_volume_m3",
    "aspect_ratio",
    "minSICN",
    "minDetJac",
    "max_nonorthogonality_deg",
    "max_skewness",
    "min_face_pyramid_m3",
)


class StructuredMeshError(RuntimeError):
    """An expected structured-mesh construction or validation failure."""


class StructuredRunError(StructuredMeshError):
    """A failed run with a serializable report."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class BearingParams:
    source: Path
    source_sha256: str
    length_mm: float
    mean_radius_mm: float
    semicone_angle_deg: float
    radial_clearance_mm: float
    eccentricity_ratio: float
    eccentricity_mm: float
    ex_mm: float
    ey_mm: float
    cone_slope: float

    def journal_radius_mm(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        return self.mean_radius_mm + (self.length_mm / 2.0 - z_mm) * self.cone_slope

    def bore_radius_mm(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        return self.journal_radius_mm(z_mm) + self.radial_clearance_mm

    @property
    def exact_volume_m3(self) -> float:
        return (
            math.pi
            * self.length_mm
            * (
                2.0 * self.mean_radius_mm * self.radial_clearance_mm
                + self.radial_clearance_mm**2
            )
            * SI_PER_MM**3
        )

    @property
    def h_min_mm(self) -> float:
        return self.radial_clearance_mm - self.eccentricity_mm

    @property
    def h_max_mm(self) -> float:
        return self.radial_clearance_mm + self.eccentricity_mm


@dataclass(frozen=True)
class StructuredInputs:
    params: Path = Path("out/strict_default/params.json")
    outdir: Path = Path("out/structured_no_port_default")
    n_theta: int = 256
    n_axial: int = 96
    gap_levels: tuple[int, ...] = (4, 8, 12)
    preview_ngap: int = 8
    gui: bool = False
    gui_mode: Literal["exact", "cutaway", "quality", "exaggerated"] = "cutaway"
    display_gap_scale: float = 100.0
    quality_view: str = "minSICN"
    openfoam: Literal["auto", "required", "skip"] = "auto"


@dataclass(frozen=True)
class MeshData:
    points_m: np.ndarray
    hexes: np.ndarray
    boundary_quads: dict[str, np.ndarray]
    logical_cell_indices: np.ndarray
    cell_tags: np.ndarray
    node_tags: np.ndarray
    cell_centres_m: np.ndarray
    cell_metrics: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_quads", MappingProxyType(dict(self.boundary_quads)))
        object.__setattr__(self, "cell_metrics", MappingProxyType(dict(self.cell_metrics)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for array in (
            self.points_m,
            self.hexes,
            self.logical_cell_indices,
            self.cell_tags,
            self.node_tags,
            self.cell_centres_m,
            *self.boundary_quads.values(),
            *self.cell_metrics.values(),
        ):
            array.setflags(write=False)


def load_params(path: Path) -> BearingParams:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        resolved = raw["resolved_parameters"]
        params = BearingParams(
            source=path.resolve(),
            source_sha256=sha256_file(path),
            length_mm=float(resolved["length"]),
            mean_radius_mm=float(resolved["mean_radius"]),
            semicone_angle_deg=float(resolved["semicone_angle_deg"]),
            radial_clearance_mm=float(resolved["radial_clearance"]),
            eccentricity_ratio=float(resolved["eccentricity_ratio"]),
            eccentricity_mm=float(resolved["eccentricity"]),
            ex_mm=float(resolved["ex"]),
            ey_mm=float(resolved["ey"]),
            cone_slope=float(resolved["cone_slope"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StructuredMeshError(f"invalid params.json {path}: {error}") from error
    validate_params(params)
    return params


def validate_params(params: BearingParams) -> None:
    values = (
        params.length_mm,
        params.mean_radius_mm,
        params.semicone_angle_deg,
        params.radial_clearance_mm,
        params.eccentricity_mm,
        params.ex_mm,
        params.ey_mm,
        params.cone_slope,
    )
    if not all(math.isfinite(value) for value in values):
        raise StructuredMeshError("all geometry parameters must be finite")
    if params.length_mm <= 0.0 or params.mean_radius_mm <= 0.0:
        raise StructuredMeshError("length and mean radius must be positive")
    if params.radial_clearance_mm <= 0.0:
        raise StructuredMeshError("radial clearance must be positive")
    if not 0.0 <= params.eccentricity_ratio < 1.0:
        raise StructuredMeshError("eccentricity ratio must satisfy 0 <= epsilon < 1")
    if not 0.0 <= params.eccentricity_mm < params.radial_clearance_mm:
        raise StructuredMeshError("eccentricity must satisfy 0 <= e < c")
    expected_eccentricity = params.eccentricity_ratio * params.radial_clearance_mm
    if not math.isclose(
        params.eccentricity_mm, expected_eccentricity, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise StructuredMeshError(
            f"derived eccentricity {params.eccentricity_mm} does not equal epsilon*c "
            f"({expected_eccentricity})"
        )
    expected_slope = math.tan(math.radians(params.semicone_angle_deg))
    if not math.isclose(
        params.cone_slope, expected_slope, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise StructuredMeshError(
            f"derived cone slope {params.cone_slope} does not equal tan(gamma) "
            f"({expected_slope})"
        )
    if abs(params.ex_mm) > 1.0e-12 or abs(params.ey_mm + params.eccentricity_mm) > 1.0e-12:
        raise StructuredMeshError(
            "Stage 2B requires the journal centre at (0,-e); params ex/ey do not match"
        )
    radii = np.asarray(params.journal_radius_mm(np.asarray([0.0, params.length_mm])))
    if np.any(radii <= 0.0):
        raise StructuredMeshError(f"journal radius is nonpositive at an axial end: {radii}")


def validate_inputs(inputs: StructuredInputs) -> None:
    if inputs.n_theta < 4:
        raise StructuredMeshError("n-theta must be at least 4")
    if inputs.n_axial < 1:
        raise StructuredMeshError("n-axial must be positive")
    if not inputs.gap_levels or any(level < 1 for level in inputs.gap_levels):
        raise StructuredMeshError("gap-levels must contain positive integers")
    if len(set(inputs.gap_levels)) != len(inputs.gap_levels):
        raise StructuredMeshError("gap-levels must be unique")
    if inputs.preview_ngap not in inputs.gap_levels:
        raise StructuredMeshError("preview-ngap must be one of gap-levels")
    if not math.isfinite(inputs.display_gap_scale) or inputs.display_gap_scale <= 1.0:
        raise StructuredMeshError("display-gap-scale must be finite and greater than 1")
    if inputs.quality_view not in VIEW_NAMES:
        raise StructuredMeshError(f"quality-view must be one of {VIEW_NAMES}")


def node_tag(j: np.ndarray | int, k: np.ndarray | int, i: np.ndarray | int, n_theta: int, n_gap: int) -> np.ndarray:
    return 1 + np.asarray(i, dtype=np.uint64) + np.uint64(n_gap + 1) * (
        np.asarray(j, dtype=np.uint64) + np.uint64(n_theta) * np.asarray(k, dtype=np.uint64)
    )


def _quad_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    average = points.mean(axis=1)
    following = np.roll(points, -1, axis=1)
    triangles_twice = np.cross(following - points, average[:, None, :] - points)
    summed = triangles_twice.sum(axis=1)
    summed_norm = np.linalg.norm(summed, axis=1)
    if np.any(summed_norm <= 0.0):
        raise StructuredMeshError("degenerate quadrilateral face")
    normal = summed / summed_norm[:, None]
    weights = np.einsum("mfc,mc->mf", triangles_twice, normal)
    weight_sum = weights.sum(axis=1)
    if np.any(weight_sum <= 0.0):
        raise StructuredMeshError("inconsistent quadrilateral winding")
    centre = (
        weights[:, :, None]
        * (points + following + average[:, None, :])
    ).sum(axis=1) / (3.0 * weight_sum[:, None])
    return centre, 0.5 * summed


def _gauss_hex_volume(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signs = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    gauss = 1.0 / math.sqrt(3.0)
    volume = np.zeros(len(points), dtype=np.float64)
    minimum = np.full(len(points), np.inf, dtype=np.float64)
    for r in (-gauss, gauss):
        for s in (-gauss, gauss):
            for t in (-gauss, gauss):
                coordinates = (r, s, t)
                derivatives = np.empty((8, 3), dtype=np.float64)
                for axis in range(3):
                    other = [index for index in range(3) if index != axis]
                    derivatives[:, axis] = (
                        0.125
                        * signs[:, axis]
                        * (1.0 + coordinates[other[0]] * signs[:, other[0]])
                        * (1.0 + coordinates[other[1]] * signs[:, other[1]])
                    )
                jacobian = np.einsum("mnc,na->mca", points, derivatives)
                determinant = np.linalg.det(jacobian)
                volume += determinant
                minimum = np.minimum(minimum, determinant)
    return volume, minimum


def compute_custom_metrics(
    points_m: np.ndarray,
    hexes: np.ndarray,
    logical: np.ndarray,
    n_theta: int,
    n_axial: int,
    n_gap: int,
    chunk_size: int = 50_000,
) -> dict[str, np.ndarray]:
    count = len(hexes)
    volumes = np.empty(count, dtype=np.float64)
    gauss_volumes = np.empty(count, dtype=np.float64)
    gauss_min_det = np.empty(count, dtype=np.float64)
    aspect = np.empty(count, dtype=np.float64)
    min_pyramid = np.empty(count, dtype=np.float64)
    centres = np.empty((count, 3), dtype=np.float64)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        cell_points = points_m[hexes[start:stop].astype(np.int64) - 1]
        face_centres = np.empty((stop - start, 6, 3), dtype=np.float64)
        face_areas = np.empty_like(face_centres)
        for face_index, local_face in enumerate(HEX_FACES):
            face_centres[:, face_index], face_areas[:, face_index] = _quad_geometry(
                cell_points[:, local_face]
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
        pyramids = np.einsum(
            "mfc,mfc->mf", face_areas, face_centres - centre[:, None, :]
        ) / 3.0
        component_area = np.abs(face_areas).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            first_aspect = component_area.max(axis=1) / component_area.min(axis=1)
            second_aspect = component_area.sum(axis=1) / (
                6.0 * np.power(volume, 2.0 / 3.0)
            )
        gauss_volume, gauss_det = _gauss_hex_volume(cell_points)
        volumes[start:stop] = volume
        gauss_volumes[start:stop] = gauss_volume
        gauss_min_det[start:stop] = gauss_det
        aspect[start:stop] = np.maximum(first_aspect, second_aspect)
        min_pyramid[start:stop] = pyramids.min(axis=1)
        centres[start:stop] = centre

    max_nonorthogonality = np.zeros(count, dtype=np.float64)
    max_skewness = np.zeros(count, dtype=np.float64)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        cell_points = points_m[hexes[start:stop].astype(np.int64) - 1]
        cell_centres = centres[start:stop]
        ids = np.arange(start, stop, dtype=np.int64)
        j = logical[start:stop, 0].astype(np.int64)
        k = logical[start:stop, 1].astype(np.int64)
        i = logical[start:stop, 2].astype(np.int64)
        for face_index, local_face in enumerate(HEX_FACES):
            vertices = cell_points[:, local_face]
            face_centre, face_area = _quad_geometry(vertices)
            if face_index == 0:
                internal = i > 0
                neighbour = ids - 1
            elif face_index == 1:
                internal = i < n_gap - 1
                neighbour = ids + 1
            elif face_index == 2:
                internal = np.ones(len(ids), dtype=bool)
                neighbour = i + n_gap * (k + n_axial * ((j - 1) % n_theta))
            elif face_index == 3:
                internal = np.ones(len(ids), dtype=bool)
                neighbour = i + n_gap * (k + n_axial * ((j + 1) % n_theta))
            elif face_index == 4:
                internal = k > 0
                neighbour = ids - n_gap
            else:
                internal = k < n_axial - 1
                neighbour = ids + n_gap

            vector = face_centre - cell_centres
            if np.any(internal):
                vector[internal] = centres[neighbour[internal]] - cell_centres[internal]
                dot = np.einsum(
                    "mc,mc->m", face_area[internal], vector[internal]
                )
                denominator = np.linalg.norm(face_area[internal], axis=1) * np.linalg.norm(
                    vector[internal], axis=1
                )
                cosine = np.clip(dot / denominator, -1.0, 1.0)
                max_nonorthogonality[start:stop][internal] = np.maximum(
                    max_nonorthogonality[start:stop][internal],
                    np.degrees(np.arccos(cosine)),
                )

            displacement = face_centre - cell_centres
            area_norm = np.linalg.norm(face_area, axis=1)
            normal = face_area / area_norm[:, None]
            boundary = ~internal
            if np.any(boundary):
                normal_distance = np.einsum(
                    "mc,mc->m", normal[boundary], displacement[boundary]
                )
                vector[boundary] = normal[boundary] * normal_distance[:, None]
            area_dot_vector = np.einsum("mc,mc->m", face_area, vector)
            area_dot_displacement = np.einsum(
                "mc,mc->m", face_area, displacement
            )
            skew_vector = displacement - (
                area_dot_displacement / area_dot_vector
            )[:, None] * vector
            skew_norm = np.linalg.norm(skew_vector, axis=1)
            skew_unit = np.zeros_like(skew_vector)
            nonzero = skew_norm > 0.0
            skew_unit[nonzero] = skew_vector[nonzero] / skew_norm[nonzero, None]
            vertex_projection = np.abs(
                np.einsum(
                    "mc,mfc->mf", skew_unit, vertices - face_centre[:, None, :]
                )
            ).max(axis=1)
            base = np.where(internal, 0.2, 0.4) * np.linalg.norm(vector, axis=1)
            denominator = np.maximum(base, vertex_projection)
            skew = np.divide(
                skew_norm,
                denominator,
                out=np.zeros_like(skew_norm),
                where=denominator > 0.0,
            )
            max_skewness[start:stop] = np.maximum(max_skewness[start:stop], skew)

    return {
        "signed_volume_m3": volumes,
        "gauss_volume_m3": gauss_volumes,
        "gauss_min_det": gauss_min_det,
        "aspect_ratio": aspect,
        "max_nonorthogonality_deg": max_nonorthogonality,
        "max_skewness": max_skewness,
        "min_face_pyramid_m3": min_pyramid,
        "cell_centre_m": centres,
    }


def generate_mesh(
    params: BearingParams,
    n_theta: int,
    n_axial: int,
    n_gap: int,
    gap_scale: float = 1.0,
    solve_eligible: bool = True,
) -> MeshData:
    if n_theta < 4 or n_axial < 1 or n_gap < 1:
        raise StructuredMeshError("mesh counts require n_theta>=4, n_axial>=1, n_gap>=1")
    theta = 2.0 * math.pi * np.arange(n_theta, dtype=np.float64) / n_theta
    z_mm = params.length_mm * np.arange(n_axial + 1, dtype=np.float64) / n_axial
    xi = np.arange(n_gap + 1, dtype=np.float64) / n_gap
    journal_radius = np.asarray(params.journal_radius_mm(z_mm), dtype=np.float64)
    radicand = journal_radius[:, None] ** 2 - (
        params.eccentricity_mm * np.sin(theta)[None, :]
    ) ** 2
    if np.any(radicand <= 0.0):
        raise StructuredMeshError(f"journal-ray radicand is nonpositive: {radicand.min()}")
    rho_j = params.eccentricity_mm * np.cos(theta)[None, :] + np.sqrt(radicand)
    rho_b = np.asarray(params.bore_radius_mm(z_mm), dtype=np.float64)[:, None]
    gap = rho_b - rho_j
    if np.any(rho_j <= 0.0) or np.any(gap <= 0.0):
        raise StructuredMeshError(
            f"requires rho_b > rho_j > 0; min rho_j={rho_j.min()}, min gap={gap.min()}"
        )
    rho = rho_j[:, :, None] + gap_scale * xi[None, None, :] * gap[:, :, None]
    points_grid = np.empty((n_axial + 1, n_theta, n_gap + 1, 3), dtype=np.float64)
    points_grid[..., 0] = rho * np.sin(theta)[None, :, None] * SI_PER_MM
    points_grid[..., 1] = -rho * np.cos(theta)[None, :, None] * SI_PER_MM
    points_grid[..., 2] = z_mm[:, None, None] * SI_PER_MM
    points = np.ascontiguousarray(points_grid.reshape(-1, 3), dtype=np.float64)
    node_tags = np.arange(1, len(points) + 1, dtype=np.uint64)

    j, k, i = np.meshgrid(
        np.arange(n_theta, dtype=np.uint64),
        np.arange(n_axial, dtype=np.uint64),
        np.arange(n_gap, dtype=np.uint64),
        indexing="ij",
    )
    jp = (j + 1) % n_theta
    hexes = np.column_stack(
        [
            node_tag(j, k, i, n_theta, n_gap).ravel(),
            node_tag(jp, k, i, n_theta, n_gap).ravel(),
            node_tag(jp, k + 1, i, n_theta, n_gap).ravel(),
            node_tag(j, k + 1, i, n_theta, n_gap).ravel(),
            node_tag(j, k, i + 1, n_theta, n_gap).ravel(),
            node_tag(jp, k, i + 1, n_theta, n_gap).ravel(),
            node_tag(jp, k + 1, i + 1, n_theta, n_gap).ravel(),
            node_tag(j, k + 1, i + 1, n_theta, n_gap).ravel(),
        ]
    ).astype(np.uint64, copy=False)
    logical = np.column_stack([j.ravel(), k.ravel(), i.ravel()]).astype(
        np.uint32, copy=False
    )
    cell_tags = np.arange(1, len(hexes) + 1, dtype=np.uint64)

    bj, bk = np.meshgrid(
        np.arange(n_theta, dtype=np.uint64),
        np.arange(n_axial, dtype=np.uint64),
        indexing="ij",
    )
    bjp = (bj + 1) % n_theta
    journal = np.column_stack(
        [
            node_tag(bj, bk, 0, n_theta, n_gap).ravel(),
            node_tag(bj, bk + 1, 0, n_theta, n_gap).ravel(),
            node_tag(bjp, bk + 1, 0, n_theta, n_gap).ravel(),
            node_tag(bjp, bk, 0, n_theta, n_gap).ravel(),
        ]
    )
    stationary = np.column_stack(
        [
            node_tag(bj, bk, n_gap, n_theta, n_gap).ravel(),
            node_tag(bjp, bk, n_gap, n_theta, n_gap).ravel(),
            node_tag(bjp, bk + 1, n_gap, n_theta, n_gap).ravel(),
            node_tag(bj, bk + 1, n_gap, n_theta, n_gap).ravel(),
        ]
    )
    ej, ei = np.meshgrid(
        np.arange(n_theta, dtype=np.uint64),
        np.arange(n_gap, dtype=np.uint64),
        indexing="ij",
    )
    ejp = (ej + 1) % n_theta
    z0 = np.column_stack(
        [
            node_tag(ej, 0, ei, n_theta, n_gap).ravel(),
            node_tag(ejp, 0, ei, n_theta, n_gap).ravel(),
            node_tag(ejp, 0, ei + 1, n_theta, n_gap).ravel(),
            node_tag(ej, 0, ei + 1, n_theta, n_gap).ravel(),
        ]
    )
    z_l = np.column_stack(
        [
            node_tag(ej, n_axial, ei, n_theta, n_gap).ravel(),
            node_tag(ej, n_axial, ei + 1, n_theta, n_gap).ravel(),
            node_tag(ejp, n_axial, ei + 1, n_theta, n_gap).ravel(),
            node_tag(ejp, n_axial, ei, n_theta, n_gap).ravel(),
        ]
    )
    boundary_quads = {
        "journal_wall": journal.astype(np.uint64, copy=False),
        "stationary_wall": stationary.astype(np.uint64, copy=False),
        "axial_end_z0": z0.astype(np.uint64, copy=False),
        "axial_end_zL": z_l.astype(np.uint64, copy=False),
    }

    metrics = compute_custom_metrics(points, hexes, logical, n_theta, n_axial, n_gap)
    cell_centres = metrics.pop("cell_centre_m")
    theta_centre = 2.0 * math.pi * (logical[:, 0].astype(np.float64) + 0.5) / n_theta
    z_centre = params.length_mm * (logical[:, 1].astype(np.float64) + 0.5) / n_axial
    rj_centre = np.asarray(params.journal_radius_mm(z_centre))
    rho_j_centre = params.eccentricity_mm * np.cos(theta_centre) + np.sqrt(
        rj_centre**2 - (params.eccentricity_mm * np.sin(theta_centre)) ** 2
    )
    gap_centre = np.asarray(params.bore_radius_mm(z_centre)) - rho_j_centre
    metrics.update(
        {
            "gap_um": gap_centre * 1_000.0,
            "layer_index": logical[:, 2].astype(np.float64),
            "theta_deg": np.degrees(theta_centre),
            "axial_index": logical[:, 1].astype(np.float64),
        }
    )
    metadata = {
        "coordinate_unit": "m",
        "source_parameter_unit": "mm",
        "scale_to_m_applied_once": SI_PER_MM,
        "n_theta": n_theta,
        "n_axial": n_axial,
        "n_gap": n_gap,
        "gap_scale": gap_scale,
        "solve_eligible": solve_eligible,
        "distorted_geometry": gap_scale != 1.0,
        "full_360_degrees": True,
        "duplicate_theta_endpoint": False,
        "params_sha256": params.source_sha256,
    }
    return MeshData(
        points_m=points,
        hexes=np.ascontiguousarray(hexes),
        boundary_quads=boundary_quads,
        logical_cell_indices=np.ascontiguousarray(logical),
        cell_tags=cell_tags,
        node_tags=node_tags,
        cell_centres_m=np.ascontiguousarray(cell_centres),
        cell_metrics={name: np.ascontiguousarray(values) for name, values in metrics.items()},
        metadata=metadata,
    )


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    order = np.lexsort(tuple(values[:, column] for column in reversed(range(values.shape[1]))))
    return values[order]


def _cell_index(
    j: np.ndarray, k: np.ndarray, i: np.ndarray, n_axial: int, n_gap: int
) -> np.ndarray:
    return i + n_gap * (k + n_axial * j)


def expected_neighbour_pairs(n_theta: int, n_axial: int, n_gap: int) -> np.ndarray:
    tj, tk, ti = np.meshgrid(
        np.arange(n_theta, dtype=np.int64),
        np.arange(n_axial, dtype=np.int64),
        np.arange(n_gap, dtype=np.int64),
        indexing="ij",
    )
    pairs = [
        np.column_stack(
            [
                _cell_index(tj, tk, ti, n_axial, n_gap).ravel(),
                _cell_index((tj + 1) % n_theta, tk, ti, n_axial, n_gap).ravel(),
            ]
        )
    ]
    if n_axial > 1:
        aj, ak, ai = np.meshgrid(
            np.arange(n_theta, dtype=np.int64),
            np.arange(n_axial - 1, dtype=np.int64),
            np.arange(n_gap, dtype=np.int64),
            indexing="ij",
        )
        pairs.append(
            np.column_stack(
                [
                    _cell_index(aj, ak, ai, n_axial, n_gap).ravel(),
                    _cell_index(aj, ak + 1, ai, n_axial, n_gap).ravel(),
                ]
            )
        )
    if n_gap > 1:
        gj, gk, gi = np.meshgrid(
            np.arange(n_theta, dtype=np.int64),
            np.arange(n_axial, dtype=np.int64),
            np.arange(n_gap - 1, dtype=np.int64),
            indexing="ij",
        )
        pairs.append(
            np.column_stack(
                [
                    _cell_index(gj, gk, gi, n_axial, n_gap).ravel(),
                    _cell_index(gj, gk, gi + 1, n_axial, n_gap).ravel(),
                ]
            )
        )
    result = np.concatenate(pairs).astype(np.int64, copy=False)
    result.sort(axis=1)
    return _sorted_rows(result)


def validate_topology(mesh: MeshData, records: list[dict[str, Any]]) -> dict[str, Any]:
    j_count = int(mesh.metadata["n_theta"])
    k_count = int(mesh.metadata["n_axial"])
    g_count = int(mesh.metadata["n_gap"])
    cell_count = len(mesh.hexes)
    faces = mesh.hexes[:, HEX_FACES].reshape(-1, 4)
    face_keys = np.sort(faces, axis=1)
    unique_faces, inverse, face_counts = np.unique(
        face_keys, axis=0, return_inverse=True, return_counts=True
    )
    expected_boundary = 2 * j_count * k_count + 2 * j_count * g_count
    expected_internal = (
        j_count * k_count * g_count
        + j_count * (k_count - 1) * g_count
        + j_count * k_count * (g_count - 1)
    )
    require(
        records,
        "topology.face_adjacency_counts",
        np.all((face_counts == 1) | (face_counts == 2)),
        {str(value): int(np.count_nonzero(face_counts == value)) for value in np.unique(face_counts)},
        "only 1 or 2 adjacent cells",
    )
    require(
        records,
        "topology.boundary_face_count",
        int(np.count_nonzero(face_counts == 1)) == expected_boundary,
        int(np.count_nonzero(face_counts == 1)),
        expected_boundary,
    )
    require(
        records,
        "topology.internal_face_count",
        int(np.count_nonzero(face_counts == 2)) == expected_internal,
        int(np.count_nonzero(face_counts == 2)),
        expected_internal,
    )

    patch_faces = np.concatenate(list(mesh.boundary_quads.values()))
    patch_keys = np.sort(patch_faces, axis=1)
    unique_patch_keys, patch_counts = np.unique(patch_keys, axis=0, return_counts=True)
    require(
        records,
        "topology.boundary_patches_disjoint",
        len(unique_patch_keys) == len(patch_keys) and np.all(patch_counts == 1),
        {"total": len(patch_keys), "unique": len(unique_patch_keys)},
        "all boundary quads unique",
    )
    require(
        records,
        "topology.complete_external_boundary_coverage",
        np.array_equal(unique_patch_keys, unique_faces[face_counts == 1]),
        len(unique_patch_keys),
        int(np.count_nonzero(face_counts == 1)),
    )

    starts = np.concatenate(([0], np.cumsum(face_counts[:-1], dtype=np.int64)))
    order = np.argsort(inverse, kind="stable")
    sorted_cells = np.repeat(np.arange(cell_count, dtype=np.int64), 6)[order]
    internal_unique = np.flatnonzero(face_counts == 2)
    actual_pairs = np.column_stack(
        [sorted_cells[starts[internal_unique]], sorted_cells[starts[internal_unique] + 1]]
    )
    actual_pairs.sort(axis=1)
    actual_pairs = _sorted_rows(actual_pairs)
    expected_pairs = expected_neighbour_pairs(j_count, k_count, g_count)
    require(
        records,
        "topology.exact_tensor_product_neighbours",
        np.array_equal(actual_pairs, expected_pairs),
        len(actual_pairs),
        len(expected_pairs),
    )
    require(
        records,
        "topology.one_connected_volume_region",
        np.array_equal(actual_pairs, expected_pairs) and cell_count > 0,
        1,
        1,
    )

    first_cells = np.arange(k_count * g_count, dtype=np.int64)
    last_cells = (j_count - 1) * k_count * g_count + first_cells
    seam_last = np.sort(mesh.hexes[last_cells][:, HEX_FACES[3]], axis=1)
    seam_first = np.sort(mesh.hexes[first_cells][:, HEX_FACES[2]], axis=1)
    require(
        records,
        "topology.periodic_last_to_first_connectivity",
        np.array_equal(seam_last, seam_first),
        len(seam_last),
        k_count * g_count,
    )
    seam_keys = {tuple(row) for row in seam_last.tolist()}
    boundary_key_set = {tuple(row) for row in unique_patch_keys.tolist()}
    require(
        records,
        "topology.no_circumferential_seam_patch",
        seam_keys.isdisjoint(boundary_key_set),
        len(seam_keys & boundary_key_set),
        0,
    )
    return {
        "unique_face_count": len(unique_faces),
        "boundary_face_count": expected_boundary,
        "internal_face_count": expected_internal,
        "connected_regions": 1,
        "periodic_seam_face_count": len(seam_keys),
    }


def validate_boundary_orientation(
    mesh: MeshData, params: BearingParams, records: list[dict[str, Any]]
) -> dict[str, Any]:
    j_count = int(mesh.metadata["n_theta"])
    k_count = int(mesh.metadata["n_axial"])
    g_count = int(mesh.metadata["n_gap"])
    centres = mesh.cell_centres_m
    owner_indices = {
        "journal_wall": np.asarray(
            [j * k_count * g_count + k * g_count for j in range(j_count) for k in range(k_count)]
        ),
        "stationary_wall": np.asarray(
            [
                j * k_count * g_count + k * g_count + g_count - 1
                for j in range(j_count)
                for k in range(k_count)
            ]
        ),
        "axial_end_z0": np.asarray(
            [j * k_count * g_count + i for j in range(j_count) for i in range(g_count)]
        ),
        "axial_end_zL": np.asarray(
            [
                j * k_count * g_count + (k_count - 1) * g_count + i
                for j in range(j_count)
                for i in range(g_count)
            ]
        ),
    }
    diagnostics: dict[str, Any] = {}
    e_m = params.eccentricity_mm * SI_PER_MM
    for name, quads in mesh.boundary_quads.items():
        face_centre, face_area = _quad_geometry(mesh.points_m[quads.astype(np.int64) - 1])
        owner_centre = centres[owner_indices[name]]
        generic_dot = np.einsum("mc,mc->m", face_area, face_centre - owner_centre)
        if name == "journal_wall":
            rj_m = np.asarray(params.journal_radius_mm(face_centre[:, 2] / SI_PER_MM)) * SI_PER_MM
            analytic_normal = np.column_stack(
                [
                    -face_centre[:, 0],
                    -(face_centre[:, 1] + e_m),
                    -rj_m * params.cone_slope,
                ]
            )
        elif name == "stationary_wall":
            rb_m = np.asarray(params.bore_radius_mm(face_centre[:, 2] / SI_PER_MM)) * SI_PER_MM
            analytic_normal = np.column_stack(
                [face_centre[:, 0], face_centre[:, 1], rb_m * params.cone_slope]
            )
        elif name == "axial_end_z0":
            analytic_normal = np.broadcast_to((0.0, 0.0, -1.0), face_centre.shape)
        else:
            analytic_normal = np.broadcast_to((0.0, 0.0, 1.0), face_centre.shape)
        analytic_dot = np.einsum("mc,mc->m", face_area, analytic_normal)
        require(
            records,
            f"boundary.{name}.outward_from_owner",
            bool(np.all(generic_dot > 0.0)),
            float(generic_dot.min()),
            "> 0",
        )
        require(
            records,
            f"boundary.{name}.analytic_outward_direction",
            bool(np.all(analytic_dot > 0.0)),
            float(analytic_dot.min()),
            "> 0",
        )
        diagnostics[name] = {
            "quad_count": len(quads),
            "minimum_owner_dot": float(generic_dot.min()),
            "minimum_analytic_dot": float(analytic_dot.min()),
        }
    return diagnostics


def validate_analytic_mesh(
    mesh: MeshData, params: BearingParams, records: list[dict[str, Any]]
) -> dict[str, Any]:
    j_count = int(mesh.metadata["n_theta"])
    k_count = int(mesh.metadata["n_axial"])
    g_count = int(mesh.metadata["n_gap"])
    expected_points = j_count * (k_count + 1) * (g_count + 1)
    expected_hexes = j_count * k_count * g_count
    expected_patch_counts = {
        "journal_wall": j_count * k_count,
        "stationary_wall": j_count * k_count,
        "axial_end_z0": j_count * g_count,
        "axial_end_zL": j_count * g_count,
    }
    require(records, "counts.points", len(mesh.points_m) == expected_points, len(mesh.points_m), expected_points)
    require(records, "counts.hexes", len(mesh.hexes) == expected_hexes, len(mesh.hexes), expected_hexes)
    for name, expected in expected_patch_counts.items():
        require(records, f"counts.{name}_quads", len(mesh.boundary_quads[name]) == expected, len(mesh.boundary_quads[name]), expected)
    require(
        records,
        "data.canonical_dtypes",
        mesh.points_m.dtype == np.float64
        and mesh.hexes.dtype == np.uint64
        and mesh.logical_cell_indices.dtype == np.uint32
        and mesh.cell_tags.dtype == np.uint64
        and mesh.node_tags.dtype == np.uint64
        and mesh.cell_centres_m.dtype == np.float64
        and all(values.dtype == np.uint64 for values in mesh.boundary_quads.values())
        and all(values.dtype == np.float64 for values in mesh.cell_metrics.values()),
        {
            "points": str(mesh.points_m.dtype),
            "hexes": str(mesh.hexes.dtype),
            "logical": str(mesh.logical_cell_indices.dtype),
            "cell_tags": str(mesh.cell_tags.dtype),
            "node_tags": str(mesh.node_tags.dtype),
            "cell_centres": str(mesh.cell_centres_m.dtype),
            "boundary_quads": {
                name: str(values.dtype) for name, values in mesh.boundary_quads.items()
            },
            "cell_metrics": {
                name: str(values.dtype) for name, values in mesh.cell_metrics.items()
            },
        },
        "canonical float64/uint64/uint32 arrays",
    )
    actual_shapes = {
        "points": mesh.points_m.shape,
        "hexes": mesh.hexes.shape,
        "logical": mesh.logical_cell_indices.shape,
        "cell_tags": mesh.cell_tags.shape,
        "node_tags": mesh.node_tags.shape,
        "cell_centres": mesh.cell_centres_m.shape,
        "boundary_quads": {
            name: values.shape for name, values in mesh.boundary_quads.items()
        },
        "cell_metrics": {
            name: values.shape for name, values in mesh.cell_metrics.items()
        },
    }
    canonical_shapes = (
        mesh.points_m.shape == (expected_points, 3)
        and mesh.hexes.shape == (expected_hexes, 8)
        and mesh.logical_cell_indices.shape == (expected_hexes, 3)
        and mesh.cell_tags.shape == (expected_hexes,)
        and mesh.node_tags.shape == (expected_points,)
        and mesh.cell_centres_m.shape == (expected_hexes, 3)
        and all(
            mesh.boundary_quads[name].shape == (expected_patch_counts[name], 4)
            for name in expected_patch_counts
        )
        and all(values.shape == (expected_hexes,) for values in mesh.cell_metrics.values())
    )
    require(
        records,
        "data.canonical_shapes",
        canonical_shapes,
        actual_shapes,
        "points[N,3], hexes[M,8], centres[M,3], quads[F,4], metrics[M]",
    )
    all_finite = (
        np.isfinite(mesh.points_m).all()
        and np.isfinite(mesh.cell_centres_m).all()
        and all(np.isfinite(values).all() for values in mesh.cell_metrics.values())
    )
    require(
        records,
        "data.no_nan_or_inf",
        all_finite,
        "finite" if all_finite else "nonfinite canonical floating-point data",
        "all finite",
    )
    unique_point_count = len(np.unique(mesh.points_m, axis=0))
    require(
        records,
        "data.no_duplicate_nodes",
        unique_point_count == len(mesh.points_m),
        unique_point_count,
        len(mesh.points_m),
    )
    sorted_hexes = np.sort(mesh.hexes, axis=1)
    unique_hex_count = len(np.unique(sorted_hexes, axis=0))
    require(
        records,
        "data.no_duplicate_hexes",
        unique_hex_count == len(mesh.hexes),
        unique_hex_count,
        len(mesh.hexes),
    )
    require(
        records,
        "data.distinct_nodes_per_hex",
        bool(np.all(np.diff(sorted_hexes, axis=1) > 0)),
        "distinct" if np.all(np.diff(sorted_hexes, axis=1) > 0) else "duplicate node in hex",
        "8 distinct nodes",
    )

    grid = mesh.points_m.reshape(k_count + 1, j_count, g_count + 1, 3)
    z_mm = grid[:, 0, 0, 2] / SI_PER_MM
    rj_m = np.asarray(params.journal_radius_mm(z_mm))[:, None] * SI_PER_MM
    rb_m = np.asarray(params.bore_radius_mm(z_mm))[:, None] * SI_PER_MM
    journal_residual = np.sqrt(
        grid[:, :, 0, 0] ** 2
        + (grid[:, :, 0, 1] + params.eccentricity_mm * SI_PER_MM) ** 2
    ) - rj_m
    bushing_residual = np.sqrt(
        grid[:, :, -1, 0] ** 2 + grid[:, :, -1, 1] ** 2
    ) - rb_m
    geometry_tolerance = 1.0e-12
    require(
        records,
        "analytic.journal_cone_residual",
        float(np.abs(journal_residual).max()) <= geometry_tolerance,
        float(np.abs(journal_residual).max()),
        geometry_tolerance,
    )
    require(
        records,
        "analytic.stationary_cone_residual",
        float(np.abs(bushing_residual).max()) <= geometry_tolerance,
        float(np.abs(bushing_residual).max()),
        geometry_tolerance,
    )
    rho = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2)
    theta = 2.0 * math.pi * np.arange(j_count, dtype=np.float64) / j_count
    rj_grid_mm = np.asarray(params.journal_radius_mm(z_mm))[:, None]
    expected_journal_rho_m = (
        params.eccentricity_mm * np.cos(theta)[None, :]
        + np.sqrt(
            rj_grid_mm**2
            - (params.eccentricity_mm * np.sin(theta)[None, :]) ** 2
        )
    ) * SI_PER_MM
    ray_residual = rho[:, :, 0] - expected_journal_rho_m
    require(
        records,
        "analytic.bearing_ray_journal_intersection",
        float(np.abs(ray_residual).max()) <= geometry_tolerance,
        float(np.abs(ray_residual).max()),
        geometry_tolerance,
    )
    require(
        records,
        "analytic.positive_monotonic_xi_layers",
        float(rho.min()) > 0.0 and bool(np.all(np.diff(rho, axis=2) > 0.0)),
        {"minimum_rho_m": float(rho.min()), "minimum_increment_m": float(np.diff(rho, axis=2).min())},
        "rho>0 and strictly increasing through gap",
    )
    xi = np.arange(g_count + 1, dtype=np.float64) / g_count
    expected_rho = rho[:, :, :1] + xi[None, None, :] * (rho[:, :, -1:] - rho[:, :, :1])
    require(
        records,
        "analytic.exact_linear_xi_layers",
        float(np.abs(rho - expected_rho).max()) <= geometry_tolerance,
        float(np.abs(rho - expected_rho).max()),
        geometry_tolerance,
    )
    sampled_gap_mm = (rho[:, :, -1] - rho[:, :, 0]) / SI_PER_MM
    cardinal_indices = (
        (0, j_count // 4, j_count // 2, 3 * j_count // 4)
        if j_count % 4 == 0
        else (0,)
    )
    cardinal_errors: dict[str, float] = {}
    for theta_index in cardinal_indices:
        theta_value = theta[theta_index]
        expected_rho_j = params.eccentricity_mm * math.cos(theta_value) + np.sqrt(
            np.asarray(params.journal_radius_mm(z_mm)) ** 2
            - (params.eccentricity_mm * math.sin(theta_value)) ** 2
        )
        expected_gap = np.asarray(params.bore_radius_mm(z_mm)) - expected_rho_j
        cardinal_errors[f"{math.degrees(theta_value):.12g}"] = float(
            np.abs(sampled_gap_mm[:, theta_index] - expected_gap).max()
        )
    require(
        records,
        "analytic.cardinal_gap_samples",
        max(cardinal_errors.values()) <= 1.0e-12,
        cardinal_errors,
        "<= 1e-12 mm",
        1.0e-12,
    )
    require(
        records,
        "analytic.minimum_gap",
        float(np.abs(sampled_gap_mm[:, 0] - params.h_min_mm).max()) <= 1.0e-12,
        float(sampled_gap_mm[:, 0].mean()),
        params.radial_clearance_mm - params.eccentricity_mm,
        1.0e-12,
    )
    if j_count % 2 == 0:
        require(
            records,
            "analytic.maximum_gap",
            float(
                np.abs(sampled_gap_mm[:, j_count // 2] - params.h_max_mm).max()
            )
            <= 1.0e-12,
            float(sampled_gap_mm[:, j_count // 2].mean()),
            params.radial_clearance_mm + params.eccentricity_mm,
            1.0e-12,
        )

    signed = mesh.cell_metrics["signed_volume_m3"]
    gauss_volume = mesh.cell_metrics["gauss_volume_m3"]
    require(records, "quality.positive_signed_hex_volume", bool(np.all(signed > 0.0)), float(signed.min()), "> 0")
    require(records, "quality.positive_gauss_hex_volume", bool(np.all(gauss_volume > 0.0)), float(gauss_volume.min()), "> 0")
    require(
        records,
        "quality.face_and_gauss_volume_agree",
        relative_error(float(signed.sum()), float(gauss_volume.sum())) <= 1.0e-10,
        relative_error(float(signed.sum()), float(gauss_volume.sum())),
        1.0e-10,
    )
    minimum_pyramid = mesh.cell_metrics["min_face_pyramid_m3"]
    require(records, "quality.positive_face_pyramids", bool(np.all(minimum_pyramid > 0.0)), float(minimum_pyramid.min()), "> 0")
    nonorth = mesh.cell_metrics["max_nonorthogonality_deg"]
    skewness = mesh.cell_metrics["max_skewness"]
    require(records, "quality.maximum_nonorthogonality", float(nonorth.max()) <= 45.0, float(nonorth.max()), "<=45 deg")
    require(records, "quality.maximum_skewness", float(skewness.max()) <= 4.0, float(skewness.max()), "<=4")
    topology = validate_topology(mesh, records)
    boundary = validate_boundary_orientation(mesh, params, records)
    mesh_volume = float(signed.sum())
    return {
        "counts": {
            "points": expected_points,
            "hexes": expected_hexes,
            "boundary_quads": expected_patch_counts,
        },
        "radial_gap_mm": {"minimum": params.h_min_mm, "maximum": params.h_max_mm},
        "maximum_journal_residual_m": float(np.abs(journal_residual).max()),
        "maximum_stationary_residual_m": float(np.abs(bushing_residual).max()),
        "mesh_volume_m3": mesh_volume,
        "exact_continuous_volume_m3": params.exact_volume_m3,
        "faceted_volume_relative_error": relative_error(mesh_volume, params.exact_volume_m3),
        "quality": {
            "minimum_signed_volume_m3": float(signed.min()),
            "minimum_gauss_volume_m3": float(gauss_volume.min()),
            "minimum_face_pyramid_m3": float(minimum_pyramid.min()),
            "maximum_nonorthogonality_deg": float(nonorth.max()),
            "maximum_skewness": float(skewness.max()),
            "aspect_ratio_min": float(mesh.cell_metrics["aspect_ratio"].min()),
            "aspect_ratio_max": float(mesh.cell_metrics["aspect_ratio"].max()),
        },
        "topology": topology,
        "boundary_orientation": boundary,
    }


def ring_lines(mesh: MeshData) -> dict[str, np.ndarray]:
    j_count = int(mesh.metadata["n_theta"])
    k_count = int(mesh.metadata["n_axial"])
    g_count = int(mesh.metadata["n_gap"])
    j = np.arange(j_count, dtype=np.uint64)
    jp = (j + 1) % j_count
    return {
        "journal_z0_ring": np.column_stack(
            [node_tag(j, 0, 0, j_count, g_count), node_tag(jp, 0, 0, j_count, g_count)]
        ),
        "journal_zL_ring": np.column_stack(
            [
                node_tag(j, k_count, 0, j_count, g_count),
                node_tag(jp, k_count, 0, j_count, g_count),
            ]
        ),
        "bushing_z0_ring": np.column_stack(
            [
                node_tag(j, 0, g_count, j_count, g_count),
                node_tag(jp, 0, g_count, j_count, g_count),
            ]
        ),
        "bushing_zL_ring": np.column_stack(
            [
                node_tag(j, k_count, g_count, j_count, g_count),
                node_tag(jp, k_count, g_count, j_count, g_count),
            ]
        ),
    }


def add_discrete_model(
    mesh: MeshData,
    records: list[dict[str, Any]],
    model_name: str,
    volume_physical_name: str = "fluid",
    volume_physical_id: int = PHYSICAL_IDS["fluid"],
) -> dict[str, Any]:
    gmsh.model.add(model_name)
    for tag in CURVE_ENTITIES.values():
        gmsh.model.addDiscreteEntity(1, tag)
    gmsh.model.addDiscreteEntity(
        2,
        SURFACE_ENTITIES["journal_wall"],
        [-CURVE_ENTITIES["journal_z0_ring"], CURVE_ENTITIES["journal_zL_ring"]],
    )
    gmsh.model.addDiscreteEntity(
        2,
        SURFACE_ENTITIES["stationary_wall"],
        [CURVE_ENTITIES["bushing_z0_ring"], -CURVE_ENTITIES["bushing_zL_ring"]],
    )
    gmsh.model.addDiscreteEntity(
        2,
        SURFACE_ENTITIES["axial_end_z0"],
        [CURVE_ENTITIES["journal_z0_ring"], -CURVE_ENTITIES["bushing_z0_ring"]],
    )
    gmsh.model.addDiscreteEntity(
        2,
        SURFACE_ENTITIES["axial_end_zL"],
        [-CURVE_ENTITIES["journal_zL_ring"], CURVE_ENTITIES["bushing_zL_ring"]],
    )
    gmsh.model.addDiscreteEntity(3, VOLUME_ENTITY, list(SURFACE_ENTITIES.values()))
    for name, tag in CURVE_ENTITIES.items():
        gmsh.model.setEntityName(1, tag, name)
    for name, tag in SURFACE_ENTITIES.items():
        gmsh.model.setEntityName(2, tag, name)
    gmsh.model.setEntityName(3, VOLUME_ENTITY, volume_physical_name)

    line_type = int(gmsh.model.mesh.getElementType("Line", 1))
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    require(records, "gmsh.element_type.Line2", line_type == 1, line_type, 1)
    require(records, "gmsh.element_type.Quadrangle4", quad_type == 3, quad_type, 3)
    require(records, "gmsh.element_type.Hexahedron8", hex_type == 5, hex_type, 5)

    gmsh.model.mesh.addNodes(3, VOLUME_ENTITY, mesh.node_tags, mesh.points_m.ravel())
    next_element_tag = int(mesh.cell_tags[-1]) + 1
    quad_element_tags: dict[str, np.ndarray] = {}
    for name, quads in mesh.boundary_quads.items():
        tags = np.arange(next_element_tag, next_element_tag + len(quads), dtype=np.uint64)
        next_element_tag += len(quads)
        quad_element_tags[name] = tags
        gmsh.model.mesh.addElementsByType(
            SURFACE_ENTITIES[name], quad_type, tags, quads.ravel()
        )
    line_element_tags: dict[str, np.ndarray] = {}
    lines = ring_lines(mesh)
    for name, connectivity in lines.items():
        tags = np.arange(
            next_element_tag, next_element_tag + len(connectivity), dtype=np.uint64
        )
        next_element_tag += len(connectivity)
        line_element_tags[name] = tags
        gmsh.model.mesh.addElementsByType(
            CURVE_ENTITIES[name], line_type, tags, connectivity.ravel()
        )
    gmsh.model.mesh.addElementsByType(
        VOLUME_ENTITY, hex_type, mesh.cell_tags, mesh.hexes.ravel()
    )
    gmsh.model.mesh.reclassifyNodes()

    for name, physical_id in PHYSICAL_IDS.items():
        if name == "fluid":
            continue
        created = gmsh.model.addPhysicalGroup(
            2, [SURFACE_ENTITIES[name]], tag=physical_id, name=name
        )
        require(records, f"physical.{name}.id", created == physical_id, created, physical_id)
    created_volume = gmsh.model.addPhysicalGroup(
        3, [VOLUME_ENTITY], tag=volume_physical_id, name=volume_physical_name
    )
    require(
        records,
        f"physical.{volume_physical_name}.id",
        created_volume == volume_physical_id,
        created_volume,
        volume_physical_id,
    )
    for name, physical_id in RING_PHYSICAL_IDS.items():
        created = gmsh.model.addPhysicalGroup(
            1, [CURVE_ENTITIES[name]], tag=physical_id, name=name
        )
        require(records, f"physical.{name}.id", created == physical_id, created, physical_id)

    return {
        "element_types": {"Line2": line_type, "Quad4": quad_type, "Hex8": hex_type},
        "quad_elements": {
            name: {
                "count": len(tags),
                "first_tag": int(tags[0]),
                "last_tag": int(tags[-1]),
            }
            for name, tags in quad_element_tags.items()
        },
        "line_elements": {
            name: {
                "count": len(tags),
                "first_tag": int(tags[0]),
                "last_tag": int(tags[-1]),
            }
            for name, tags in line_element_tags.items()
        },
        "volume_physical_name": volume_physical_name,
        "volume_physical_id": volume_physical_id,
    }


def expected_physical_groups(
    volume_name: str = "fluid", volume_id: int = PHYSICAL_IDS["fluid"]
) -> dict[tuple[int, int], dict[str, Any]]:
    groups = {
        (2, PHYSICAL_IDS[name]): {"name": name, "entities": [SURFACE_ENTITIES[name]]}
        for name in SURFACE_ENTITIES
    }
    groups[(3, volume_id)] = {"name": volume_name, "entities": [VOLUME_ENTITY]}
    groups.update(
        {
            (1, RING_PHYSICAL_IDS[name]): {
                "name": name,
                "entities": [CURVE_ENTITIES[name]],
            }
            for name in CURVE_ENTITIES
        }
    )
    return groups


def validate_physical_groups(
    records: list[dict[str, Any]],
    prefix: str,
    volume_name: str = "fluid",
    volume_id: int = PHYSICAL_IDS["fluid"],
) -> dict[str, Any]:
    expected = expected_physical_groups(volume_name, volume_id)
    actual: dict[tuple[int, int], dict[str, Any]] = {}
    for dimension, physical_id in gmsh.model.getPhysicalGroups():
        key = (int(dimension), int(physical_id))
        actual[key] = {
            "name": gmsh.model.getPhysicalName(*key),
            "entities": sorted(
                int(tag) for tag in gmsh.model.getEntitiesForPhysicalGroup(*key)
            ),
        }
    actual_serial = {
        f"{dimension}:{physical_id}": data
        for (dimension, physical_id), data in actual.items()
    }
    expected_serial = {
        f"{dimension}:{physical_id}": data
        for (dimension, physical_id), data in expected.items()
    }
    require(
        records,
        f"{prefix}.physical_groups_exact",
        actual == expected,
        actual_serial,
        expected_serial,
    )
    surface_memberships = [
        entity
        for (dimension, physical_id), data in actual.items()
        if dimension == 2 and physical_id in PHYSICAL_IDS.values()
        for entity in data["entities"]
    ]
    require(
        records,
        f"{prefix}.boundary_groups_disjoint_complete",
        len(surface_memberships) == len(set(surface_memberships)) == 4
        and set(surface_memberships) == set(SURFACE_ENTITIES.values()),
        surface_memberships,
        sorted(SURFACE_ENTITIES.values()),
    )
    require(
        records,
        f"{prefix}.no_pressure_feed_group",
        all(data["name"] != "pressure_feed" for data in actual.values()),
        sorted(data["name"] for data in actual.values()),
        "pressure_feed absent",
    )
    return actual_serial


def add_gmsh_quality_metrics(
    mesh: MeshData, records: list[dict[str, Any]]
) -> MeshData:
    cell_tags = mesh.cell_tags.astype(np.int64, copy=False)
    min_sicn = np.asarray(
        gmsh.model.mesh.getElementQualities(cell_tags, "minSICN"), dtype=np.float64
    )
    min_det_jac = np.asarray(
        gmsh.model.mesh.getElementQualities(cell_tags, "minDetJac"), dtype=np.float64
    )
    gmsh_volume = np.asarray(
        gmsh.model.mesh.getElementQualities(cell_tags, "volume"), dtype=np.float64
    )
    require(
        records,
        "gmsh.quality.minSICN_finite",
        len(min_sicn) == len(mesh.hexes) and np.isfinite(min_sicn).all(),
        {"count": len(min_sicn), "minimum": float(min_sicn.min())},
        "finite value per Hex8",
    )
    require(
        records,
        "gmsh.quality.minDetJac_positive",
        len(min_det_jac) == len(mesh.hexes)
        and np.isfinite(min_det_jac).all()
        and np.all(min_det_jac > 0.0),
        {"count": len(min_det_jac), "minimum": float(min_det_jac.min())},
        "finite and > 0",
    )
    require(
        records,
        "gmsh.quality.volume_positive",
        len(gmsh_volume) == len(mesh.hexes)
        and np.isfinite(gmsh_volume).all()
        and np.all(gmsh_volume > 0.0),
        {"count": len(gmsh_volume), "minimum": float(gmsh_volume.min())},
        "finite and > 0",
    )
    require(
        records,
        "gmsh.quality.volume_vs_canonical",
        relative_error(
            float(gmsh_volume.sum()), float(mesh.cell_metrics["signed_volume_m3"].sum())
        )
        <= 1.0e-10,
        relative_error(
            float(gmsh_volume.sum()), float(mesh.cell_metrics["signed_volume_m3"].sum())
        ),
        1.0e-10,
    )
    metrics = dict(mesh.cell_metrics)
    metrics.update(
        {
            "cell_volume_m3": gmsh_volume,
            "minSICN": min_sicn,
            "minDetJac": min_det_jac,
        }
    )
    return replace(mesh, cell_metrics=metrics)


def write_quality_views(mesh: MeshData, views_dir: Path) -> dict[str, dict[str, Any]]:
    views_dir.mkdir(parents=True, exist_ok=True)
    model_name = gmsh.model.getCurrent()
    outputs: dict[str, dict[str, Any]] = {}
    for name in VIEW_NAMES:
        tag = gmsh.view.add(name)
        gmsh.view.addModelData(
            tag,
            0,
            model_name,
            "ElementData",
            mesh.cell_tags,
            mesh.cell_metrics[name].reshape(-1, 1),
            numComponents=1,
        )
        path = views_dir / f"{name}.pos"
        gmsh.view.write(tag, str(path))
        gmsh.view.remove(tag)
        outputs[name] = {"path": str(path.name), "bytes": path.stat().st_size}
    return outputs


def write_gmsh_files(case_dir: Path) -> tuple[Path, Path]:
    visual = case_dir / "structured_hex.msh"
    openfoam = case_dir / "structured_hex_openfoam.msh"
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(visual))
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(openfoam))
    return visual, openfoam


def _coordinate_node_mapping(
    read_tags: np.ndarray,
    read_points: np.ndarray,
    mesh: MeshData,
    records: list[dict[str, Any]],
    prefix: str,
) -> np.ndarray:
    read_order = _coordinate_order(read_points)
    expected_order = _coordinate_order(mesh.points_m)
    maximum_error = float(
        np.abs(read_points[read_order] - mesh.points_m[expected_order]).max(initial=0.0)
    )
    require(
        records,
        f"{prefix}.point_count_coordinates",
        len(read_points) == len(mesh.points_m) and maximum_error <= 1.0e-14,
        {"count": len(read_points), "maximum_coordinate_error_m": maximum_error},
        {"count": len(mesh.points_m), "maximum_coordinate_error_m": 1.0e-14},
    )
    mapping = np.zeros(int(read_tags.max(initial=0)) + 1, dtype=np.uint64)
    mapping[read_tags[read_order].astype(np.int64)] = mesh.node_tags[expected_order]
    return mapping


def _coordinate_order(points: np.ndarray, quantum_m: float = 1.0e-11) -> np.ndarray:
    """Order coordinates robustly across ASCII rounding of analytic symmetries."""
    quantized = np.rint(points / quantum_m).astype(np.int64)
    if len(np.unique(quantized, axis=0)) != len(points):
        raise StructuredMeshError(
            f"coordinate matching quantum {quantum_m:.1e} m merged distinct nodes"
        )
    return np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))


def _compare_oriented_connectivity(
    actual: np.ndarray,
    expected: np.ndarray,
    records: list[dict[str, Any]],
    name: str,
) -> None:
    actual_keys = np.sort(actual, axis=1)
    expected_keys = np.sort(expected, axis=1)
    actual_order = np.lexsort(tuple(actual_keys[:, index] for index in reversed(range(actual.shape[1]))))
    expected_order = np.lexsort(tuple(expected_keys[:, index] for index in reversed(range(expected.shape[1]))))
    keys_match = np.array_equal(actual_keys[actual_order], expected_keys[expected_order])
    orientation_match = keys_match and np.array_equal(
        actual[actual_order], expected[expected_order]
    )
    require(
        records,
        name,
        len(actual) == len(expected) and orientation_match,
        {"count": len(actual), "keys_match": keys_match, "orientation_match": orientation_match},
        {"count": len(expected), "keys_match": True, "orientation_match": True},
    )


def validate_gmsh_round_trip(
    path: Path,
    mesh: MeshData,
    records: list[dict[str, Any]],
    prefix: str,
    volume_name: str = "fluid",
    volume_id: int = PHYSICAL_IDS["fluid"],
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.open(str(path))
    physical = validate_physical_groups(records, prefix, volume_name, volume_id)
    line_type = int(gmsh.model.mesh.getElementType("Line", 1))
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    read_tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes()
    read_tags = np.asarray(read_tags_raw, dtype=np.int64)
    read_points = np.asarray(coordinates_raw, dtype=np.float64).reshape(-1, 3)
    mapping = _coordinate_node_mapping(read_tags, read_points, mesh, records, prefix)

    hex_tags_raw, hex_nodes_raw = gmsh.model.mesh.getElementsByType(
        hex_type, VOLUME_ENTITY
    )
    hex_tags = np.asarray(hex_tags_raw, dtype=np.int64)
    hex_nodes = np.asarray(hex_nodes_raw, dtype=np.int64).reshape(-1, 8)
    mapped_hexes = mapping[hex_nodes]
    _compare_oriented_connectivity(
        mapped_hexes, mesh.hexes, records, f"{prefix}.Hex8_connectivity"
    )
    rings = ring_lines(mesh)
    for name, expected in mesh.boundary_quads.items():
        _tags, nodes = gmsh.model.mesh.getElementsByType(
            quad_type, SURFACE_ENTITIES[name]
        )
        mapped = mapping[np.asarray(nodes, dtype=np.int64).reshape(-1, 4)]
        _compare_oriented_connectivity(
            mapped, expected, records, f"{prefix}.{name}_Quad4_connectivity"
        )
    for name, expected in rings.items():
        _tags, nodes = gmsh.model.mesh.getElementsByType(
            line_type, CURVE_ENTITIES[name]
        )
        mapped = mapping[np.asarray(nodes, dtype=np.int64).reshape(-1, 2)]
        _compare_oriented_connectivity(
            mapped, expected, records, f"{prefix}.{name}_Line2_connectivity"
        )

    element_types, tags_by_type, _nodes_by_type = gmsh.model.mesh.getElements()
    type_counts = {
        int(element_type): len(tags)
        for element_type, tags in zip(element_types, tags_by_type)
    }
    expected_type_counts = {
        line_type: sum(len(lines) for lines in rings.values()),
        quad_type: sum(len(quads) for quads in mesh.boundary_quads.values()),
        hex_type: len(mesh.hexes),
    }
    require(
        records,
        f"{prefix}.no_unexpected_elements",
        type_counts == expected_type_counts,
        type_counts,
        expected_type_counts,
    )
    bbox = (
        *read_points.min(axis=0).tolist(),
        *read_points.max(axis=0).tolist(),
    )
    expected_bbox = (
        *mesh.points_m.min(axis=0).tolist(),
        *mesh.points_m.max(axis=0).tolist(),
    )
    bbox_error = max(abs(a - b) for a, b in zip(bbox, expected_bbox))
    require(
        records,
        f"{prefix}.SI_bounding_box",
        bbox_error <= 1.0e-14 and max(abs(value) for value in bbox) < 1.0,
        {"bbox_m": bbox, "maximum_error_m": bbox_error},
        expected_bbox,
        1.0e-14,
    )
    min_det = np.asarray(
        gmsh.model.mesh.getElementQualities(hex_tags, "minDetJac"), dtype=np.float64
    )
    volumes = np.asarray(
        gmsh.model.mesh.getElementQualities(hex_tags, "volume"), dtype=np.float64
    )
    require(
        records,
        f"{prefix}.positive_minDetJac",
        np.isfinite(min_det).all() and np.all(min_det > 0.0),
        float(min_det.min()),
        "> 0",
    )
    require(
        records,
        f"{prefix}.positive_volume",
        np.isfinite(volumes).all() and np.all(volumes > 0.0),
        float(volumes.min()),
        "> 0",
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "point_count": len(read_points),
        "element_type_counts": type_counts,
        "physical_groups": physical,
        "bbox_m": bbox,
        "minimum_minDetJac": float(min_det.min()),
        "minimum_volume_m3": float(volumes.min()),
    }


def write_vtu_files(mesh: MeshData, case_dir: Path) -> tuple[Path, Path]:
    volume_path = case_dir / "volume_hex.vtu"
    boundary_path = case_dir / "boundary_quads.vtu"
    volume_cell_data = {
        name: [np.asarray(mesh.cell_metrics[name])]
        for name in VIEW_NAMES
    }
    volume_cell_data.update(
        {
            "theta_index": [mesh.logical_cell_indices[:, 0]],
            "axial_index_integer": [mesh.logical_cell_indices[:, 1]],
            "gap_index": [mesh.logical_cell_indices[:, 2]],
            "cell_tag": [mesh.cell_tags],
        }
    )
    meshio.write(
        volume_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("hexahedron", mesh.hexes.astype(np.int64) - 1)],
            cell_data=volume_cell_data,
            field_data={"fluid": np.asarray([PHYSICAL_IDS["fluid"], 3])},
        ),
        file_format="vtu",
        binary=True,
    )
    patch_order = tuple(SURFACE_ENTITIES)
    boundary_cells = np.concatenate([mesh.boundary_quads[name] for name in patch_order])
    patch_ids = np.concatenate(
        [
            np.full(len(mesh.boundary_quads[name]), PHYSICAL_IDS[name], dtype=np.int32)
            for name in patch_order
        ]
    )
    meshio.write(
        boundary_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("quad", boundary_cells.astype(np.int64) - 1)],
            cell_data={"patch_id": [patch_ids]},
            field_data={
                name: np.asarray([PHYSICAL_IDS[name], 2], dtype=np.int32)
                for name in patch_order
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return volume_path, boundary_path


def validate_vtu_round_trip(
    mesh: MeshData,
    volume_path: Path,
    boundary_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    volume = meshio.read(volume_path)
    boundary = meshio.read(boundary_path)
    volume_cells = volume.cells_dict.get("hexahedron")
    boundary_cells = boundary.cells_dict.get("quad")
    expected_boundary = np.concatenate(list(mesh.boundary_quads.values())).astype(np.int64) - 1
    require(
        records,
        "vtu.volume.coordinates_connectivity",
        np.array_equal(volume.points, mesh.points_m)
        and volume_cells is not None
        and np.array_equal(volume_cells, mesh.hexes.astype(np.int64) - 1),
        {
            "point_count": len(volume.points),
            "cell_type": [cell.type for cell in volume.cells],
            "hex_count": 0 if volume_cells is None else len(volume_cells),
        },
        {"point_count": len(mesh.points_m), "cell_type": ["hexahedron"], "hex_count": len(mesh.hexes)},
    )
    expected_cell_data = {
        **{name: mesh.cell_metrics[name] for name in VIEW_NAMES},
        "theta_index": mesh.logical_cell_indices[:, 0],
        "axial_index_integer": mesh.logical_cell_indices[:, 1],
        "gap_index": mesh.logical_cell_indices[:, 2],
        "cell_tag": mesh.cell_tags,
    }
    missing_or_wrong = {
        name: {
            "length": len(volume.cell_data_dict.get(name, {}).get("hexahedron", [])),
            "values_equal": bool(
                np.array_equal(
                    volume.cell_data_dict.get(name, {}).get("hexahedron", np.asarray([])),
                    expected,
                )
            ),
        }
        for name, expected in expected_cell_data.items()
        if not np.array_equal(
            volume.cell_data_dict.get(name, {}).get("hexahedron", np.asarray([])),
            expected,
        )
    }
    require(
        records,
        "vtu.volume.cell_data_lengths",
        not missing_or_wrong,
        missing_or_wrong,
        f"exact canonical values for all {len(expected_cell_data)} volume fields",
    )
    expected_patch_ids = np.concatenate(
        [
            np.full(len(mesh.boundary_quads[name]), PHYSICAL_IDS[name], dtype=np.int32)
            for name in SURFACE_ENTITIES
        ]
    )
    read_patch_ids = boundary.cell_data_dict.get("patch_id", {}).get("quad")
    require(
        records,
        "vtu.boundary.coordinates_connectivity_patch_ids",
        np.array_equal(boundary.points, mesh.points_m)
        and boundary_cells is not None
        and np.array_equal(boundary_cells, expected_boundary)
        and read_patch_ids is not None
        and np.array_equal(read_patch_ids, expected_patch_ids),
        {
            "point_count": len(boundary.points),
            "cell_type": [cell.type for cell in boundary.cells],
            "quad_count": 0 if boundary_cells is None else len(boundary_cells),
            "patch_ids": [] if read_patch_ids is None else sorted(set(read_patch_ids.tolist())),
        },
        {
            "point_count": len(mesh.points_m),
            "cell_type": ["quad"],
            "quad_count": len(expected_boundary),
            "patch_ids": sorted(PHYSICAL_IDS[name] for name in SURFACE_ENTITIES),
        },
    )
    return {
        "volume_vtu_sha256": sha256_file(volume_path),
        "boundary_vtu_sha256": sha256_file(boundary_path),
        "point_count": len(mesh.points_m),
        "hex_count": len(mesh.hexes),
        "quad_count": len(expected_boundary),
        "patch_ids": sorted(PHYSICAL_IDS[name] for name in SURFACE_ENTITIES),
    }


def write_mesh_arrays(mesh: MeshData, path: Path) -> None:
    arrays: dict[str, Any] = {
        "points_m": mesh.points_m,
        "hexes": mesh.hexes,
        "logical_cell_indices": mesh.logical_cell_indices,
        "cell_tags": mesh.cell_tags,
        "node_tags": mesh.node_tags,
        "cell_centres_m": mesh.cell_centres_m,
        "metadata_json": np.asarray(json.dumps(dict(mesh.metadata), sort_keys=True)),
    }
    arrays.update({f"boundary_{name}": values for name, values in mesh.boundary_quads.items()})
    arrays.update({f"metric_{name}": values for name, values in mesh.cell_metrics.items()})
    np.savez_compressed(path, **arrays)


def validate_mesh_arrays_round_trip(
    mesh: MeshData, path: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        comparisons = {
            "points_m": np.array_equal(archive["points_m"], mesh.points_m),
            "hexes": np.array_equal(archive["hexes"], mesh.hexes),
            "logical_cell_indices": np.array_equal(
                archive["logical_cell_indices"], mesh.logical_cell_indices
            ),
            "cell_tags": np.array_equal(archive["cell_tags"], mesh.cell_tags),
            "node_tags": np.array_equal(archive["node_tags"], mesh.node_tags),
            "cell_centres_m": np.array_equal(
                archive["cell_centres_m"], mesh.cell_centres_m
            ),
            "metadata": json.loads(str(archive["metadata_json"])) == mesh.metadata,
        }
        comparisons.update(
            {
                f"boundary_{name}": np.array_equal(archive[f"boundary_{name}"], values)
                for name, values in mesh.boundary_quads.items()
            }
        )
        comparisons.update(
            {
                f"metric_{name}": np.array_equal(archive[f"metric_{name}"], values)
                for name, values in mesh.cell_metrics.items()
            }
        )
    require(
        records,
        "npz.exact_array_round_trip",
        all(comparisons.values()),
        {name: passed for name, passed in comparisons.items() if not passed},
        "exact equality for every canonical array",
    )
    return {"sha256": sha256_file(path), "comparisons": comparisons}


def write_exaggerated_vtu(mesh: MeshData, path: Path, gap_scale: float) -> None:
    cell_count = len(mesh.hexes)
    meshio.write(
        path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("hexahedron", mesh.hexes.astype(np.int64) - 1)],
            cell_data={
                "solve_eligible": [np.zeros(cell_count, dtype=np.uint8)],
                "distorted_geometry": [np.ones(cell_count, dtype=np.uint8)],
                "gap_scale": [np.full(cell_count, gap_scale, dtype=np.float64)],
                "VISUALIZATION_ONLY_DO_NOT_SOLVE": [np.ones(cell_count, dtype=np.uint8)],
            },
            field_data={
                "VISUALIZATION_ONLY_DO_NOT_SOLVE": np.asarray([999, 3], dtype=np.int32)
            },
        ),
        file_format="vtu",
        binary=True,
    )


def write_visualization_models(
    true_mesh: MeshData,
    params: BearingParams,
    case_dir: Path,
    display_gap_scale: float,
    quality_view: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    viz_dir = case_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    scale_label = f"{display_gap_scale:g}"
    exaggerated = generate_mesh(
        params,
        int(true_mesh.metadata["n_theta"]),
        int(true_mesh.metadata["n_axial"]),
        int(true_mesh.metadata["n_gap"]),
        gap_scale=display_gap_scale,
        solve_eligible=False,
    )
    gmsh.clear()
    add_discrete_model(
        exaggerated,
        records,
        "gap_exaggerated_visualization_only",
        volume_physical_name="VISUALIZATION_ONLY_DO_NOT_SOLVE",
        volume_physical_id=999,
    )
    exaggerated = add_gmsh_quality_metrics(exaggerated, records)
    exaggerated_msh = viz_dir / f"gap_x{scale_label}_VISUALIZATION_ONLY_DO_NOT_SOLVE.msh"
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(exaggerated_msh))
    exaggerated_round_trip = validate_gmsh_round_trip(
        exaggerated_msh,
        exaggerated,
        records,
        "viz.exaggerated_round_trip",
        volume_name="VISUALIZATION_ONLY_DO_NOT_SOLVE",
        volume_id=999,
    )
    exaggerated_vtu = viz_dir / f"gap_x{scale_label}_VISUALIZATION_ONLY_DO_NOT_SOLVE.vtu"
    write_exaggerated_vtu(exaggerated, exaggerated_vtu, display_gap_scale)
    read_viz = meshio.read(exaggerated_vtu)
    viz_cell_data = read_viz.cell_data_dict
    viz_marked = (
        np.all(viz_cell_data["solve_eligible"]["hexahedron"] == 0)
        and np.all(viz_cell_data["distorted_geometry"]["hexahedron"] == 1)
        and np.allclose(
            viz_cell_data["gap_scale"]["hexahedron"], display_gap_scale
        )
        and np.all(
            viz_cell_data["VISUALIZATION_ONLY_DO_NOT_SOLVE"]["hexahedron"] == 1
        )
    )
    require(
        records,
        "viz.exaggerated_vtu_permanently_marked",
        bool(viz_marked),
        bool(viz_marked),
        True,
    )

    j_count = int(true_mesh.metadata["n_theta"])
    omitted_theta_cells = max(1, int(round(j_count / 6.0)))
    keep = true_mesh.logical_cell_indices[:, 0] >= omitted_theta_cells
    kept_hexes = true_mesh.hexes[keep]
    kept_tags = true_mesh.cell_tags[keep]
    used_node_tags = np.unique(kept_hexes)
    gmsh.clear()
    gmsh.model.add("exact_cutaway_visualization_only")
    gmsh.model.addDiscreteEntity(3, VOLUME_ENTITY)
    gmsh.model.mesh.addNodes(
        3,
        VOLUME_ENTITY,
        used_node_tags,
        true_mesh.points_m[used_node_tags.astype(np.int64) - 1].ravel(),
    )
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    gmsh.model.mesh.addElementsByType(
        VOLUME_ENTITY, hex_type, kept_tags, kept_hexes.ravel()
    )
    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.addPhysicalGroup(
        3, [VOLUME_ENTITY], tag=998, name="CUTAWAY_VISUALIZATION_ONLY_DO_NOT_SOLVE"
    )
    cutaway_msh = viz_dir / "cutaway_exact.msh"
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(cutaway_msh))
    view_tag = gmsh.view.add(f"cutaway_{quality_view}")
    gmsh.view.addModelData(
        view_tag,
        0,
        gmsh.model.getCurrent(),
        "ElementData",
        kept_tags,
        true_mesh.cell_metrics[quality_view][keep].reshape(-1, 1),
        numComponents=1,
    )
    cutaway_view = viz_dir / f"cutaway_{quality_view}.pos"
    gmsh.view.write(view_tag, str(cutaway_view))
    gmsh.view.remove(view_tag)
    actual_omitted_angle = 360.0 * omitted_theta_cells / j_count
    gmsh.clear()
    gmsh.open(str(cutaway_msh))
    read_node_tags_raw, read_coordinates_raw, _ = gmsh.model.mesh.getNodes()
    read_node_tags = np.asarray(read_node_tags_raw, dtype=np.int64)
    read_coordinates = np.asarray(read_coordinates_raw, dtype=np.float64).reshape(-1, 3)
    expected_coordinates = true_mesh.points_m[used_node_tags.astype(np.int64) - 1]
    read_order = _coordinate_order(read_coordinates)
    expected_order = _coordinate_order(expected_coordinates)
    coordinate_error = float(
        np.abs(read_coordinates[read_order] - expected_coordinates[expected_order]).max(
            initial=0.0
        )
    )
    node_mapping = np.zeros(int(read_node_tags.max()) + 1, dtype=np.uint64)
    node_mapping[read_node_tags[read_order]] = used_node_tags[expected_order]
    _read_cell_tags, read_hex_nodes = gmsh.model.mesh.getElementsByType(
        hex_type, VOLUME_ENTITY
    )
    mapped_cutaway_hexes = node_mapping[
        np.asarray(read_hex_nodes, dtype=np.int64).reshape(-1, 8)
    ]
    read_groups = {
        (int(dim), int(tag)): {
            "name": gmsh.model.getPhysicalName(int(dim), int(tag)),
            "entities": sorted(
                int(entity)
                for entity in gmsh.model.getEntitiesForPhysicalGroup(int(dim), int(tag))
            ),
        }
        for dim, tag in gmsh.model.getPhysicalGroups()
    }
    cutaway_valid = (
        len(read_coordinates) == len(expected_coordinates)
        and coordinate_error <= 1.0e-14
        and read_groups
        == {
            (3, 998): {
                "name": "CUTAWAY_VISUALIZATION_ONLY_DO_NOT_SOLVE",
                "entities": [VOLUME_ENTITY],
            }
        }
    )
    require(
        records,
        "viz.cutaway_disk_round_trip",
        cutaway_valid,
        {
            "node_count": len(read_coordinates),
            "maximum_coordinate_error_m": coordinate_error,
            "physical_groups": {
                f"{dim}:{tag}": value for (dim, tag), value in read_groups.items()
            },
        },
        {
            "node_count": len(expected_coordinates),
            "maximum_coordinate_error_m": 1.0e-14,
            "physical_name": "CUTAWAY_VISUALIZATION_ONLY_DO_NOT_SOLVE",
        },
    )
    _compare_oriented_connectivity(
        mapped_cutaway_hexes,
        kept_hexes,
        records,
        "viz.cutaway_Hex8_connectivity",
    )
    require(
        records,
        "viz.cutaway_is_incomplete_solver_domain",
        len(kept_hexes) < len(true_mesh.hexes),
        {"kept_hexes": len(kept_hexes), "omitted_wedge_deg": actual_omitted_angle},
        "cell-aligned approximately 60 degree wedge omitted",
    )
    manifest = {
        "schema_version": 1,
        "cutaway": {
            "path": cutaway_msh.name,
            "quality_view_path": cutaway_view.name,
            "solve_eligible": False,
            "subset": True,
            "distorted_geometry": False,
            "coordinate_unit": "m",
            "requested_omitted_wedge_deg": 60.0,
            "cell_aligned_omitted_wedge_deg": actual_omitted_angle,
        },
        "exaggerated": {
            "msh": exaggerated_msh.name,
            "vtu": exaggerated_vtu.name,
            "volume_physical_name": "VISUALIZATION_ONLY_DO_NOT_SOLVE",
            "solve_eligible": False,
            "distorted_geometry": True,
            "gap_scale": display_gap_scale,
            "coordinate_unit": "m",
        },
    }
    (viz_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "exaggerated_round_trip": exaggerated_round_trip,
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                exaggerated_msh,
                exaggerated_vtu,
                cutaway_msh,
                cutaway_view,
                viz_dir / "manifest.json",
            )
        },
    }


def _exact_layer_radii_mm(
    params: BearingParams,
    theta: np.ndarray,
    z_mm: np.ndarray,
    xi: np.ndarray,
    gap_scale: float = 1.0,
) -> np.ndarray:
    rj = np.asarray(params.journal_radius_mm(z_mm))[:, None]
    radicand = rj**2 - (params.eccentricity_mm * np.sin(theta)[None, :]) ** 2
    rho_j = params.eccentricity_mm * np.cos(theta)[None, :] + np.sqrt(radicand)
    rho_b = np.asarray(params.bore_radius_mm(z_mm))[:, None]
    return rho_j[:, :, None] + gap_scale * xi[None, None, :] * (
        rho_b - rho_j
    )[:, :, None]


def write_diagnostic_images(
    mesh: MeshData,
    params: BearingParams,
    case_dir: Path,
    display_gap_scale: float,
) -> dict[str, dict[str, Any]]:
    images = case_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    j_count = int(mesh.metadata["n_theta"])
    k_count = int(mesh.metadata["n_axial"])
    g_count = int(mesh.metadata["n_gap"])
    theta = 2.0 * math.pi * np.arange(j_count, dtype=np.float64) / j_count
    theta_closed = np.append(theta, 2.0 * math.pi)
    xi = np.arange(g_count + 1, dtype=np.float64) / g_count
    z_mid = np.asarray([params.length_mm / 2.0])

    def midspan(path: Path, scale: float, title: str) -> None:
        rho = _exact_layer_radii_mm(params, theta, z_mid, xi, scale)[0]
        rho = np.vstack([rho, rho[0]])
        fig, axis = plt.subplots(figsize=(8, 8))
        for layer in range(g_count + 1):
            axis.plot(
                rho[:, layer] * np.sin(theta_closed),
                -rho[:, layer] * np.cos(theta_closed),
                linewidth=0.8,
                label=f"xi={layer}/{g_count}" if layer in (0, g_count) else None,
            )
        axis.set_aspect("equal")
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
        axis.set_title(title)
        axis.text(
            0.02,
            0.02,
            f"z={params.length_mm / 2:g} mm\n"
            f"true h_min={params.h_min_mm * 1000:.3f} µm\n"
            f"true h_max={params.h_max_mm * 1000:.3f} µm",
            transform=axis.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85},
        )
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    true_midspan = images / "midspan_true_scale.png"
    midspan(true_midspan, 1.0, "Midspan structured layers — TRUE SCALE")
    exaggerated_midspan = images / "midspan_gap_x100_VISUALIZATION_ONLY.png"
    midspan(
        exaggerated_midspan,
        display_gap_scale,
        f"VISUALIZATION ONLY — GAP ×{display_gap_scale:g}\nDO NOT SOLVE",
    )

    def gap_zoom(path: Path, gap_mm: float, label: str, theta_deg: float) -> None:
        offsets_um = np.arange(g_count + 1, dtype=np.float64) * gap_mm * 1000.0 / g_count
        fig, axis = plt.subplots(figsize=(9, 3.6))
        axis.hlines(0.0, 0.0, gap_mm * 1000.0, color="0.5", linewidth=1)
        axis.scatter(offsets_um, np.zeros_like(offsets_um), c=np.arange(g_count + 1), cmap="viridis", zorder=3)
        for layer, value in enumerate(offsets_um):
            axis.annotate(
                f"i={layer}\n{value:.3f} µm",
                (value, 0.0),
                xytext=(0, 12 if layer % 2 == 0 else -34),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                arrowprops={"arrowstyle": "-", "lw": 0.5},
            )
        axis.set_xlim(-0.03 * gap_mm * 1000.0, 1.03 * gap_mm * 1000.0)
        axis.set_ylim(-0.45, 0.45)
        axis.set_yticks([])
        axis.set_xlabel("Distance from journal wall along bearing-centred ray [µm]")
        axis.set_title(f"{label} at θ={theta_deg:g}° — every through-gap layer")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    minimum_zoom = images / "minimum_gap_layer_zoom.png"
    maximum_zoom = images / "maximum_gap_layer_zoom.png"
    gap_zoom(minimum_zoom, params.h_min_mm, "Minimum radial gap", 0.0)
    gap_zoom(maximum_zoom, params.h_max_mm, "Maximum radial gap", 180.0)

    map_theta = np.linspace(0.0, 2.0 * math.pi, 361)
    map_z = np.linspace(0.0, params.length_mm, 121)
    radii = _exact_layer_radii_mm(
        params, map_theta, map_z, np.asarray([0.0, 1.0])
    )
    gap_um = (radii[:, :, 1] - radii[:, :, 0]) * 1000.0
    unwrapped = images / "unwrapped_gap_map.png"
    fig, axis = plt.subplots(figsize=(10, 4.8))
    plot = axis.pcolormesh(np.degrees(map_theta), map_z, gap_um, shading="auto")
    fig.colorbar(plot, ax=axis, label="Exact same-z radial gap [µm]")
    axis.set_xlabel("θ from −Y [deg]")
    axis.set_ylabel("Axial z [mm]")
    axis.set_title("Full-360 exact radial-gap map — geometry diagnostic")
    fig.tight_layout()
    fig.savefig(unwrapped, dpi=180)
    plt.close(fig)

    quality = images / "quality_histograms.png"
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fields = (
        ("minSICN", "minSICN [1]", False),
        ("minDetJac", "minDetJac [m³]", True),
        ("max_nonorthogonality_deg", "Max non-orthogonality [deg]", False),
        ("max_skewness", "Max skewness [1]", False),
        ("aspect_ratio", "Aspect ratio [1]", True),
        ("cell_volume_m3", "Cell volume [m³]", True),
    )
    for axis, (field, label, logarithmic) in zip(axes.ravel(), fields):
        values = mesh.cell_metrics[field]
        axis.hist(values, bins=60)
        if logarithmic:
            axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Cell count")
        axis.set_title(f"min={values.min():.4g}, max={values.max():.4g}")
    fig.suptitle("Structured Hex8 quality distributions — true-scale solver mesh")
    fig.tight_layout()
    fig.savefig(quality, dpi=180)
    plt.close(fig)

    axial = images / "axial_cutaway.png"
    z_lines = np.linspace(0.0, params.length_mm, k_count + 1)
    theta_meridians = np.asarray([0.0, math.pi])
    layer_radii = _exact_layer_radii_mm(params, theta_meridians, z_lines, xi)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for meridian, (axis, title) in enumerate(
        zip(axes, ("Minimum-gap meridian θ=0°", "Maximum-gap meridian θ=180°"))
    ):
        for layer in range(g_count + 1):
            axis.plot(layer_radii[:, meridian, layer], z_lines, linewidth=0.8)
        for z_value in z_lines[:: max(1, k_count // 16)]:
            index = int(round(z_value / params.length_mm * k_count))
            axis.plot(
                [layer_radii[index, meridian, 0], layer_radii[index, meridian, -1]],
                [z_value, z_value],
                color="0.75",
                linewidth=0.5,
            )
        axis.set_xlabel("Bearing-centred radial coordinate ρ [mm]")
        axis.set_title(title)
    axes[0].set_ylabel("Axial z [mm]")
    fig.suptitle("Exact axial cutaway of structured through-gap layers")
    fig.tight_layout()
    fig.savefig(axial, dpi=180)
    plt.close(fig)

    paths = (
        true_midspan,
        exaggerated_midspan,
        minimum_zoom,
        maximum_zoom,
        unwrapped,
        quality,
        axial,
    )
    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    }


def _run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _openfoam_boundary_patches(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*\{(.*?)^\s*\}", text
    )
    patches: dict[str, dict[str, str]] = {}
    for name, body in blocks:
        if name == "FoamFile":
            continue
        patch_type = re.search(r"(?m)^\s*type\s+([^;\s]+)\s*;", body)
        patches[name] = {
            "type": patch_type.group(1) if patch_type else "",
        }
    return patches


def audit_openfoam(
    mode: str,
    case_dir: Path,
    msh_path: Path,
    mesh: MeshData,
    records: list[dict[str, Any]],
    published_case_dir: Path | None = None,
) -> dict[str, Any]:
    gmsh_to_foam = shutil.which("gmshToFoam")
    check_mesh = shutil.which("checkMesh")
    if mode == "skip" or (mode == "auto" and not (gmsh_to_foam and check_mesh)):
        reason = "disabled by --openfoam skip" if mode == "skip" else "gmshToFoam/checkMesh unavailable"
        records.append(
            {
                "name": "openfoam.audit",
                "status": "SKIPPED",
                "actual": reason,
                "expected": "optional audited conversion",
                "tolerance": None,
                "mandatory": False,
            }
        )
        return {
            "status": "SKIPPED",
            "reason": reason,
            "gmshToFoam": gmsh_to_foam,
            "checkMesh": check_mesh,
        }
    require(
        records,
        "openfoam.executables_available",
        bool(gmsh_to_foam and check_mesh),
        {"gmshToFoam": gmsh_to_foam, "checkMesh": check_mesh},
        "both executables available",
    )
    assert gmsh_to_foam is not None and check_mesh is not None
    foam_case = case_dir / "openfoam_case"
    (foam_case / "constant").mkdir(parents=True, exist_ok=True)
    (foam_case / "system").mkdir(parents=True, exist_ok=True)
    (foam_case / "system" / "controlDict").write_text(
        """FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
runTimeModifiable false;
""",
        encoding="utf-8",
    )
    converter = _run_command(
        [gmsh_to_foam, "-case", str(foam_case), str(msh_path.resolve())]
    )
    converter_text = converter["stdout"] + "\n" + converter["stderr"]
    unhandled_types = re.findall(
        r"Unhandled element\s+(\d+)", converter_text, flags=re.IGNORECASE
    )
    expected_line_records = 4 * int(mesh.metadata["n_theta"])
    require(
        records,
        "openfoam.only_expected_ignored_Line2_elements",
        not unhandled_types
        or (
            len(unhandled_types) == expected_line_records
            and set(unhandled_types) == {"1"}
        ),
        {"count": len(unhandled_types), "types": sorted(set(unhandled_types))},
        {"count": f"0 or {expected_line_records}", "types": ["1"]},
    )
    rejected_conversion = (
        "inverting hex",
        "undefined faces",
        "could not match gmsh face",
        "foam fatal",
        "foam exiting",
        "segmentation fault",
    )
    require(
        records,
        "openfoam.gmshToFoam",
        converter["returncode"] == 0
        and not any(token in converter_text.lower() for token in rejected_conversion),
        {
            "returncode": converter["returncode"],
            "rejected_messages": [
                token for token in rejected_conversion if token in converter_text.lower()
            ],
        },
        {"returncode": 0, "rejected_messages": []},
    )
    checker = _run_command(
        [check_mesh, "-case", str(foam_case), "-allTopology", "-allGeometry"]
    )
    checker_text = checker["stdout"] + "\n" + checker["stderr"]
    require(
        records,
        "openfoam.checkMesh",
        checker["returncode"] == 0
        and re.search(r"(?m)^\s*Mesh OK\.\s*$", checker_text) is not None
        and re.search(r"Failed\s+[1-9]\d*\s+mesh checks", checker_text) is None
        and "foam fatal" not in checker_text.lower()
        and "foam exiting" not in checker_text.lower(),
        {"returncode": checker["returncode"], "mesh_ok": "Mesh OK." in checker_text},
        {"returncode": 0, "mesh_ok": True},
    )
    cell_match = re.search(r"(?m)^\s*cells:\s*(\d+)\s*$", checker_text)
    region_match = re.search(r"Number of regions:\s*(\d+)", checker_text)
    bbox_match = re.search(
        r"Overall domain bounding box\s*\(([^)]+)\)\s*\(([^)]+)\)", checker_text
    )
    volume_match = re.search(r"Total volume\s*=\s*([+\-0-9.eE]+)", checker_text)
    parsed_cells = int(cell_match.group(1)) if cell_match else None
    parsed_regions = int(region_match.group(1)) if region_match else None
    parsed_bbox = None
    if bbox_match:
        parsed_bbox = tuple(
            float(value)
            for group in bbox_match.groups()
            for value in group.split()
        )
    parsed_volume = float(volume_match.group(1)) if volume_match else None
    expected_bbox = (
        *mesh.points_m.min(axis=0).tolist(),
        *mesh.points_m.max(axis=0).tolist(),
    )
    expected_volume = float(mesh.cell_metrics["cell_volume_m3"].sum())
    require(records, "openfoam.cell_count", parsed_cells == len(mesh.hexes), parsed_cells, len(mesh.hexes))
    require(records, "openfoam.connected_regions", parsed_regions == 1, parsed_regions, 1)
    require(
        records,
        "openfoam.bounding_box",
        parsed_bbox is not None
        and max(abs(a - b) for a, b in zip(parsed_bbox, expected_bbox)) <= 1.0e-9,
        parsed_bbox,
        expected_bbox,
        1.0e-9,
    )
    require(
        records,
        "openfoam.total_volume",
        parsed_volume is not None and relative_error(parsed_volume, expected_volume) <= 1.0e-6,
        parsed_volume,
        expected_volume,
        1.0e-6,
    )
    poly_mesh = foam_case / "constant" / "polyMesh"
    required_files = ("points", "faces", "owner", "neighbour", "boundary")
    missing = [
        name
        for name in required_files
        if not (poly_mesh / name).is_file() or (poly_mesh / name).stat().st_size == 0
    ]
    require(records, "openfoam.polyMesh_files", not missing, missing, [])
    patches = _openfoam_boundary_patches(poly_mesh / "boundary")
    patch_names = sorted(patches)
    expected_patches = sorted(SURFACE_ENTITIES)
    require(
        records,
        "openfoam.patch_names",
        patch_names == expected_patches,
        patch_names,
        expected_patches,
    )
    patch_types = {name: data["type"] for name, data in patches.items()}
    allowed_patch_types = {"patch", "wall"}
    forbidden_patch_tokens = ("cyclic", "symmetry", "seam", "default")
    require(
        records,
        "openfoam.patch_types",
        set(patch_types) == set(expected_patches)
        and all(patch_type in allowed_patch_types for patch_type in patch_types.values())
        and not any(
            token in (name + " " + patch_type).lower()
            for name, patch_type in patch_types.items()
            for token in forbidden_patch_tokens
        ),
        patch_types,
        {name: "patch or wall; never cyclic/symmetry/seam/default" for name in expected_patches},
    )
    foam_version = shutil.which("foamVersion")
    versions = {
        "foamVersion": _run_command([foam_version]) if foam_version else None,
        "gmshToFoam_help": _run_command([gmsh_to_foam, "-help"]),
        "checkMesh_help": _run_command([check_mesh, "-help"]),
    }
    published_foam_case = (
        published_case_dir / "openfoam_case"
        if published_case_dir is not None
        else foam_case
    )
    published_msh = (
        published_case_dir / "structured_hex_openfoam.msh"
        if published_case_dir is not None
        else msh_path
    )
    return {
        "status": "PASS",
        "executables": {"gmshToFoam": gmsh_to_foam, "checkMesh": check_mesh},
        "versions": versions,
        "conversion": converter,
        "checkMesh": checker,
        "parsed": {
            "cells": parsed_cells,
            "regions": parsed_regions,
            "bbox_m": parsed_bbox,
            "volume_m3": parsed_volume,
            "patches": patches,
        },
        "case": "openfoam_case",
        "published_case": str(published_foam_case),
        "published_commands": {
            "gmshToFoam": [
                gmsh_to_foam,
                "-case",
                str(published_foam_case),
                str(published_msh),
            ],
            "checkMesh": [
                check_mesh,
                "-case",
                str(published_foam_case),
                "-allTopology",
                "-allGeometry",
            ],
        },
    }


def metric_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _recursive_file_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "mesh_report.json"}
    }


def generate_gap_case(
    params: BearingParams,
    inputs: StructuredInputs,
    n_gap: int,
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    mesh = generate_mesh(params, inputs.n_theta, inputs.n_axial, n_gap)
    analytic = validate_analytic_mesh(mesh, params, records)
    gmsh_lines: list[str] = []
    initialized = False
    logger_started = False
    try:
        gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
        initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()
        logger_started = True
        discrete = add_discrete_model(mesh, records, f"structured_no_port_nGap_{n_gap:02d}")
        physical = validate_physical_groups(records, "gmsh_model")
        mesh = add_gmsh_quality_metrics(mesh, records)
        require(
            records,
            "gmsh.quality.minimum_face_pyramid_positive",
            bool(np.all(mesh.cell_metrics["min_face_pyramid_m3"] > 0.0)),
            float(mesh.cell_metrics["min_face_pyramid_m3"].min()),
            "> 0",
        )
        views = write_quality_views(mesh, case_dir / "views")
        msh41, msh22 = write_gmsh_files(case_dir)
        round_trips = {
            "gmsh_4_1_binary": validate_gmsh_round_trip(
                msh41, mesh, records, "round_trip.msh41"
            ),
            "gmsh_2_2_ascii": validate_gmsh_round_trip(
                msh22, mesh, records, "round_trip.msh22"
            ),
        }
        gmsh.clear()
        add_discrete_model(mesh, records, f"structured_no_port_nGap_{n_gap:02d}_vtu_source")
        volume_vtu, boundary_vtu = write_vtu_files(mesh, case_dir)
        vtu_round_trip = validate_vtu_round_trip(
            mesh, volume_vtu, boundary_vtu, records
        )
        npz_path = case_dir / "mesh_arrays.npz"
        write_mesh_arrays(mesh, npz_path)
        npz_round_trip = validate_mesh_arrays_round_trip(mesh, npz_path, records)
        visualization = write_visualization_models(
            mesh,
            params,
            case_dir,
            inputs.display_gap_scale,
            inputs.quality_view,
            records,
        )
    finally:
        if logger_started:
            gmsh_lines = [str(line) for line in gmsh.logger.get()]
            gmsh.logger.stop()
        if initialized:
            gmsh.finalize()

    published_case_dir = inputs.outdir / f"nGap_{n_gap:02d}"
    images = write_diagnostic_images(
        mesh, params, case_dir, inputs.display_gap_scale
    )
    openfoam = audit_openfoam(
        inputs.openfoam,
        case_dir,
        case_dir / "structured_hex_openfoam.msh",
        mesh,
        records,
        published_case_dir,
    )
    (case_dir / "gmsh_structured.log").write_text(
        "\n".join(gmsh_lines) + "\n", encoding="utf-8"
    )
    quality = {
        name: metric_statistics(mesh.cell_metrics[name])
        for name in (
            "gap_um",
            "cell_volume_m3",
            "aspect_ratio",
            "minSICN",
            "minDetJac",
            "max_nonorthogonality_deg",
            "max_skewness",
            "min_face_pyramid_m3",
        )
    }
    commands = {
        "gmsh_exact": f"uv run gmsh {published_case_dir / 'structured_hex.msh'}",
        "gmsh_cutaway": f"uv run gmsh {published_case_dir / 'viz' / 'cutaway_exact.msh'}",
        "gmsh_quality": (
            f"uv run gmsh {published_case_dir / 'viz' / 'cutaway_exact.msh'} "
            f"{published_case_dir / 'viz' / f'cutaway_{inputs.quality_view}.pos'}"
        ),
        "gmsh_exaggerated": (
            f"uv run gmsh {published_case_dir / 'viz' / f'gap_x{inputs.display_gap_scale:g}_VISUALIZATION_ONLY_DO_NOT_SOLVE.msh'}"
        ),
        "paraview_volume": f"paraview {published_case_dir / 'volume_hex.vtu'}",
    }
    manifest = {
        "schema_version": 1,
        "overall": "PASS",
        "solve_eligible": True,
        "geometry": "full-360 no-feed-port lubricant film",
        "canonical_arrays": "mesh_arrays.npz",
        "coordinate_unit": "m",
        "source_parameter_unit": "mm",
        "scale_to_m_applied_exactly_once": SI_PER_MM,
        "gmsh_generated_volume_mesh": False,
        "contains_only_hex8_volume_cells": True,
        "contains_pressure_feed": False,
        "counts": analytic["counts"],
        "physical_groups": physical,
        "quality_view_names": list(VIEW_NAMES),
        "visualization_only": visualization["manifest"],
        "commands": commands,
        "params": asdict(params) | {"source": str(params.source)},
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "overall": "PASS",
        "validation_records": records,
        "inputs": {
            "n_theta": inputs.n_theta,
            "n_axial": inputs.n_axial,
            "n_gap": n_gap,
            "display_gap_scale": inputs.display_gap_scale,
            "openfoam": inputs.openfoam,
        },
        "coordinate_units": {
            "params": "mm",
            "canonical_mesh": "m",
            "cell_volume": "m^3",
            "surface_area": "m^2",
        },
        "analytic": analytic,
        "quality": quality,
        "gmsh": {
            "discrete_registration": discrete,
            "physical_groups": physical,
            "views": views,
            "round_trips": round_trips,
            "log": "gmsh_structured.log",
        },
        "vtu_round_trip": vtu_round_trip,
        "npz_round_trip": npz_round_trip,
        "visualization": visualization,
        "images": images,
        "openfoam": openfoam,
        "commands": commands,
        "error": None,
    }
    report["files"] = _recursive_file_inventory(case_dir)
    (case_dir / "mesh_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "n_gap": n_gap,
        "cell_count": len(mesh.hexes),
        "point_count": len(mesh.points_m),
        "faceted_volume_m3": analytic["mesh_volume_m3"],
        "faceted_volume_relative_error": analytic["faceted_volume_relative_error"],
        "quality": quality,
        "openfoam": {key: value for key, value in openfoam.items() if key not in {"conversion", "checkMesh", "versions"}},
        "commands": commands,
        "validation_count": len(records),
    }


def theta_convergence(
    params: BearingParams, n_theta: int, n_axial: int
) -> list[dict[str, Any]]:
    theta_levels = sorted({max(4, n_theta // 4), max(4, n_theta // 2), n_theta})
    rows: list[dict[str, Any]] = []
    previous_error: float | None = None
    previous_theta: int | None = None
    for theta_count in theta_levels:
        probe = generate_mesh(params, theta_count, n_axial, 1)
        volume = float(probe.cell_metrics["signed_volume_m3"].sum())
        error = relative_error(volume, params.exact_volume_m3)
        rate = None
        if previous_error is not None and error > 0.0 and previous_error > 0.0:
            rate = math.log(previous_error / error) / math.log(theta_count / previous_theta)
        rows.append(
            {
                "family": "theta_convergence",
                "n_theta": theta_count,
                "n_axial": n_axial,
                "n_gap": 1,
                "cell_count": theta_count * n_axial,
                "faceted_volume_m3": volume,
                "exact_volume_m3": params.exact_volume_m3,
                "relative_error": error,
                "observed_rate": rate,
            }
        )
        previous_error = error
        previous_theta = theta_count
    return rows


def write_convergence(
    stage: Path,
    params: BearingParams,
    inputs: StructuredInputs,
    case_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    theta_rows = theta_convergence(params, inputs.n_theta, inputs.n_axial)
    gap_rows = [
        {
            "family": "gap_level_invariance",
            "n_theta": inputs.n_theta,
            "n_axial": inputs.n_axial,
            "n_gap": result["n_gap"],
            "cell_count": result["cell_count"],
            "faceted_volume_m3": result["faceted_volume_m3"],
            "exact_volume_m3": params.exact_volume_m3,
            "relative_error": result["faceted_volume_relative_error"],
            "observed_rate": None,
        }
        for result in case_results
    ]
    gap_volumes = np.asarray([row["faceted_volume_m3"] for row in gap_rows])
    gap_invariance = (
        float(np.ptp(gap_volumes) / abs(gap_volumes.mean())) if len(gap_volumes) > 1 else 0.0
    )
    theta_errors = [row["relative_error"] for row in theta_rows]
    if any(next_error > error * (1.0 + 1.0e-12) for error, next_error in zip(theta_errors, theta_errors[1:])):
        raise StructuredMeshError(f"faceted volume did not converge with n_theta: {theta_errors}")
    if gap_invariance > 1.0e-12:
        raise StructuredMeshError(
            f"faceted volume changed with gap subdivision: relative range {gap_invariance:.3e}"
        )
    rows = theta_rows + gap_rows
    with (stage / "convergence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "overall": "PASS",
        "coordinate_unit": "m",
        "exact_continuous_no_port_volume_m3": params.exact_volume_m3,
        "theta_convergence": theta_rows,
        "gap_level_invariance": gap_rows,
        "gap_level_volume_relative_range": gap_invariance,
        "theta_error_monotonic_nonincreasing": True,
    }
    (stage / "convergence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _publish_failure(outdir: Path, report: dict[str, Any]) -> None:
    stage = make_staging_directory(outdir)
    try:
        (stage / "failure_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        atomic_replace_directory(stage, outdir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def open_gui(inputs: StructuredInputs, outdir: Path) -> None:
    case_dir = outdir / f"nGap_{inputs.preview_ngap:02d}"
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    try:
        if inputs.gui_mode == "exact":
            gmsh.open(str(case_dir / "structured_hex.msh"))
        elif inputs.gui_mode == "exaggerated":
            gmsh.open(
                str(
                    case_dir
                    / "viz"
                    / f"gap_x{inputs.display_gap_scale:g}_VISUALIZATION_ONLY_DO_NOT_SOLVE.msh"
                )
            )
        else:
            gmsh.open(str(case_dir / "viz" / "cutaway_exact.msh"))
            if inputs.gui_mode == "quality":
                gmsh.merge(str(case_dir / "viz" / f"cutaway_{inputs.quality_view}.pos"))
                view_tags = list(gmsh.view.getTags())
                for view_tag in view_tags:
                    gmsh.view.option.setNumber(int(view_tag), "Visible", 1)
        for name in (
            "Mesh.SurfaceFaces",
            "Mesh.SurfaceEdges",
            "Mesh.VolumeFaces",
            "Mesh.VolumeEdges",
            "Mesh.Hexahedra",
        ):
            gmsh.option.setNumber(name, 1)
        gmsh.option.setNumber("Mesh.DrawSkinOnly", 0)
        gmsh.fltk.run()
    finally:
        gmsh.finalize()


def run_structured(inputs: StructuredInputs) -> dict[str, Any]:
    inputs = replace(inputs, params=inputs.params.resolve(), outdir=inputs.outdir.resolve())
    base_report = {
        "inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(inputs).items()
        },
        "volume_mesh_generated_by_gmsh": False,
    }
    stage: Path | None = None
    try:
        validate_inputs(inputs)
        if not inputs.params.is_file():
            raise StructuredMeshError(f"params file not found: {inputs.params}")
        if inputs.openfoam == "required" and not (
            shutil.which("gmshToFoam") and shutil.which("checkMesh")
        ):
            raise StructuredMeshError(
                "--openfoam required but gmshToFoam/checkMesh are unavailable"
            )
        params = load_params(inputs.params)
        stage = make_staging_directory(inputs.outdir)
        case_results = [
            generate_gap_case(
                params,
                inputs,
                n_gap,
                stage / f"nGap_{n_gap:02d}",
            )
            for n_gap in inputs.gap_levels
        ]
        convergence = write_convergence(stage, params, inputs, case_results)
        report = {
            **base_report,
            "overall": "PASS",
            "params": asdict(params) | {"source": str(params.source)},
            "cases": case_results,
            "convergence": convergence,
            "error": None,
        }
        (stage / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        atomic_replace_directory(stage, inputs.outdir)
        stage = None
        if inputs.gui:
            open_gui(inputs, inputs.outdir)
        return report
    except Exception as error:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        failure = {
            **base_report,
            "overall": "FAIL",
            "solve_eligible_outputs_published": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _publish_failure(inputs.outdir, failure)
        raise StructuredRunError(str(error), failure) from error


def print_report(report: dict[str, Any]) -> None:
    print("\nStructured full-360 no-port Hex8 mesh")
    print(f"{'case':<12} {'points':>12} {'hexes':>12} {'vol.err':>12} {'minSICN':>12} {'maxNonOrth':>12} {'status':>8}")
    print("-" * 88)
    for case in report.get("cases", []):
        print(
            f"nGap_{case['n_gap']:02d} {case['point_count']:12d} {case['cell_count']:12d} "
            f"{case['faceted_volume_relative_error']:12.3e} "
            f"{case['quality']['minSICN']['min']:12.4g} "
            f"{case['quality']['max_nonorthogonality_deg']['max']:12.4g} {'PASS':>8}"
        )
    print(f"Gmsh generated volume cells: {report.get('volume_mesh_generated_by_gmsh', False)}")
    print(f"OVERALL: {report.get('overall', 'FAIL')}")
    if report.get("overall") == "PASS":
        print("\nOpen commands")
        preview = next(
            case for case in report["cases"] if case["n_gap"] == report["inputs"]["preview_ngap"]
        )
        for name, command in preview["commands"].items():
            print(f"{name:18s} {command}")


def parse_args(argv: Sequence[str] | None = None) -> StructuredInputs:
    parser = argparse.ArgumentParser(
        description="Build a full-360 analytic no-port structured Hex8 bearing-film mesh."
    )
    parser.add_argument("--params", type=Path, default=StructuredInputs.params)
    parser.add_argument("--outdir", type=Path, default=StructuredInputs.outdir)
    parser.add_argument("--n-theta", type=int, default=256)
    parser.add_argument("--n-axial", type=int, default=96)
    parser.add_argument("--gap-levels", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--preview-ngap", type=int, default=8)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gui-mode",
        choices=("exact", "cutaway", "quality", "exaggerated"),
        default="cutaway",
    )
    parser.add_argument("--display-gap-scale", type=float, default=100.0)
    parser.add_argument("--quality-view", choices=VIEW_NAMES, default="minSICN")
    parser.add_argument("--openfoam", choices=("auto", "required", "skip"), default="auto")
    args = parser.parse_args(argv)
    return StructuredInputs(
        params=args.params,
        outdir=args.outdir,
        n_theta=args.n_theta,
        n_axial=args.n_axial,
        gap_levels=tuple(args.gap_levels),
        preview_ngap=args.preview_ngap,
        gui=args.gui,
        gui_mode=args.gui_mode,
        display_gap_scale=args.display_gap_scale,
        quality_view=args.quality_view,
        openfoam=args.openfoam,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_structured(parse_args(argv))
    except StructuredRunError as error:
        print_report(error.report)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
