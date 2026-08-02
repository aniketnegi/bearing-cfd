#!/usr/bin/env python3
"""Structured Hex8 bearing-film mesh with a pressure patch on the bore."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import gmsh
import meshio
import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path.cwd() / "out/conical_journal/.matplotlib-cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bearing_cfd.artifacts import make_staging_directory, publish_generation

from bearing_cfd.bearings.conical_journal.meshing.brep_preflight import (
    relative_error,
    require,
    sha256_file,
)
from bearing_cfd.bearings.conical_journal.meshing.gap_grading import symmetric_gap_coordinates
from bearing_cfd.bearings.conical_journal.meshing import no_port as base


PHYSICAL_IDS = {
    "journal_wall": 101,
    "stationary_wall": 102,
    "axial_end_z0": 103,
    "axial_end_zL": 104,
    "pressure_feed": 106,
    "fluid": 201,
}
SURFACE_ENTITIES = {
    "journal_wall": 21,
    "stationary_wall": 22,
    "axial_end_z0": 23,
    "axial_end_zL": 24,
    "pressure_feed": 26,
}
VOLUME_ENTITY = 31
PRESSURE_PATCH_NAMES = tuple(SURFACE_ENTITIES)
SI_PER_MM = base.SI_PER_MM


class SurfaceInletError(RuntimeError):
    """An expected surface-inlet mesh construction or validation failure."""


class SurfaceInletRunError(SurfaceInletError):
    """A failed run with a serializable report."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class InletSpec:
    axial_position_mm: float
    diameter_mm: float
    radius_mm: float


@dataclass(frozen=True)
class SurfaceInletInputs:
    params: Path = Path("out/conical_journal/geometry/default/params.json")
    outdir: Path = Path("out/conical_journal/meshing/surface-inlet")
    n_theta: int = 256
    n_axial: int = 96
    gap_levels: tuple[int, ...] = (4, 8, 12)
    preview_ngap: int = 8
    gap_inflation_ratio: float = 5.0
    inlet_cluster_strength: float = 0.82
    max_projected_area_relative_error: float = 0.01
    max_inlet_rim_error_mm: float = 0.16
    gui: bool = False
    gui_mode: Literal["exact", "cutaway", "inlet", "quality"] = "cutaway"
    quality_view: str = "minSICN"
    openfoam: Literal["auto", "required", "skip"] = "auto"


def load_inlet_spec(path: Path) -> InletSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        resolved = raw["resolved_parameters"]
        diameter = float(resolved["hole_diameter"])
        radius = float(resolved.get("hole_radius", diameter / 2.0))
        spec = InletSpec(
            axial_position_mm=float(resolved["hole_axial_pos"]),
            diameter_mm=diameter,
            radius_mm=radius,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SurfaceInletError(f"invalid inlet data in {path}: {error}") from error
    if (
        not all(
            math.isfinite(value)
            for value in (spec.axial_position_mm, spec.diameter_mm, spec.radius_mm)
        )
        or spec.diameter_mm <= 0.0
        or spec.radius_mm <= 0.0
        or not math.isclose(
            spec.radius_mm,
            spec.diameter_mm / 2.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    ):
        raise SurfaceInletError(f"invalid pressure-inlet dimensions: {spec}")
    return spec


def validate_inputs(
    inputs: SurfaceInletInputs, params: base.BearingParams, inlet: InletSpec
) -> None:
    if inputs.n_theta < 8 or inputs.n_theta % 2 != 0 or inputs.n_axial < 2:
        raise SurfaceInletError(
            "an even n-theta>=8 and n-axial>=2 are required"
        )
    if not inputs.gap_levels or any(level < 3 for level in inputs.gap_levels):
        raise SurfaceInletError("gap-levels must contain integers >=3 for 5:1 inflation")
    if len(set(inputs.gap_levels)) != len(inputs.gap_levels):
        raise SurfaceInletError("gap-levels must be unique")
    if inputs.preview_ngap not in inputs.gap_levels:
        raise SurfaceInletError("preview-ngap must be one of gap-levels")
    if (
        not math.isfinite(inputs.gap_inflation_ratio)
        or inputs.gap_inflation_ratio < 1.0
    ):
        raise SurfaceInletError("gap-inflation-ratio must be finite and >=1")
    if (
        not math.isfinite(inputs.inlet_cluster_strength)
        or not 0.0 <= inputs.inlet_cluster_strength < 1.0
    ):
        raise SurfaceInletError(
            "inlet-cluster-strength must be finite and satisfy 0<=value<1"
        )
    if not 0.0 < inputs.max_projected_area_relative_error < 1.0:
        raise SurfaceInletError(
            "max-projected-area-relative-error must lie strictly between 0 and 1"
        )
    if (
        not math.isfinite(inputs.max_inlet_rim_error_mm)
        or inputs.max_inlet_rim_error_mm <= 0.0
    ):
        raise SurfaceInletError("max-inlet-rim-error-mm must be finite and positive")
    if inputs.quality_view not in base.VIEW_NAMES:
        raise SurfaceInletError(f"quality-view must be one of {base.VIEW_NAMES}")
    if not inlet.radius_mm < inlet.axial_position_mm < params.length_mm - inlet.radius_mm:
        raise SurfaceInletError("the complete pressure footprint must lie inside both axial ends")


def _clustered_axis_nodes(
    start: float,
    end: float,
    centre: float,
    count: int,
    strength: float,
    power: float = 3.0,
) -> np.ndarray:
    """Return monotonic tensor-grid nodes clustered smoothly around centre."""
    left_count = min(
        count - 1,
        max(1, round(count * (centre - start) / (end - start))),
    )
    right_count = count - left_count

    def warp(distance: np.ndarray) -> np.ndarray:
        return (1.0 - strength) * distance + strength * distance**power

    left_distance = 1.0 - np.arange(left_count + 1, dtype=np.float64) / left_count
    right_distance = np.arange(1, right_count + 1, dtype=np.float64) / right_count
    left = centre - (centre - start) * warp(left_distance)
    right = centre + (end - centre) * warp(right_distance)
    nodes = np.concatenate((left, right))
    nodes[0], nodes[left_count], nodes[-1] = start, centre, end
    if len(nodes) != count + 1 or np.any(np.diff(nodes) <= 0.0):
        raise SurfaceInletError("clustered coordinate mapping is not strictly monotonic")
    return nodes


def _remapped_mesh(
    template: base.MeshData,
    params: base.BearingParams,
    inlet: InletSpec,
    inflation_ratio: float,
    cluster_strength: float,
) -> base.MeshData:
    n_theta = int(template.metadata["n_theta"])
    n_axial = int(template.metadata["n_axial"])
    n_gap = int(template.metadata["n_gap"])
    theta_edges = _clustered_axis_nodes(
        0.0, 2.0 * math.pi, math.pi, n_theta, cluster_strength
    )
    theta = theta_edges[:-1]
    z_mm = _clustered_axis_nodes(
        0.0,
        params.length_mm,
        inlet.axial_position_mm,
        n_axial,
        cluster_strength,
    )
    xi = symmetric_gap_coordinates(n_gap, inflation_ratio)
    fractions = np.diff(xi)
    journal_radius = np.asarray(params.journal_radius_mm(z_mm))[:, None]
    q = (
        params.ex_mm * np.sin(theta)[None, :]
        - params.ey_mm * np.cos(theta)[None, :]
    )
    radicand = (
        journal_radius**2
        - params.ex_mm**2
        - params.ey_mm**2
        + q**2
    )
    if np.any(radicand <= 0.0):
        raise SurfaceInletError(
            f"journal-ray radicand is nonpositive: {float(radicand.min())}"
        )
    rho_j = q + np.sqrt(radicand)
    rho_b = np.asarray(params.bore_radius_mm(z_mm))[:, None]
    gap = rho_b - rho_j
    if np.any(rho_j <= 0.0) or np.any(gap <= 0.0):
        raise SurfaceInletError("clustered grid requires rho_b > rho_j > 0")
    rho = rho_j[:, :, None] + xi[None, None, :] * gap[:, :, None]
    grid = np.empty((n_axial + 1, n_theta, n_gap + 1, 3), dtype=np.float64)
    grid[..., 0] = rho * np.sin(theta)[None, :, None] * SI_PER_MM
    grid[..., 1] = -rho * np.cos(theta)[None, :, None] * SI_PER_MM
    grid[..., 2] = z_mm[:, None, None] * SI_PER_MM
    points = np.ascontiguousarray(grid.reshape(-1, 3), dtype=np.float64)
    metrics = base.compute_custom_metrics(
        points,
        template.hexes,
        template.logical_cell_indices,
        n_theta,
        n_axial,
        n_gap,
    )
    centres = np.ascontiguousarray(metrics.pop("cell_centre_m"))
    theta_centres = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    z_centres = 0.5 * (z_mm[:-1] + z_mm[1:])
    logical = template.logical_cell_indices
    cell_theta = theta_centres[logical[:, 0]]
    cell_z = z_centres[logical[:, 1]]
    cell_rj = np.asarray(params.journal_radius_mm(cell_z))
    cell_q = params.ex_mm * np.sin(cell_theta) - params.ey_mm * np.cos(cell_theta)
    cell_rho_j = cell_q + np.sqrt(
        cell_rj**2 - params.ex_mm**2 - params.ey_mm**2 + cell_q**2
    )
    metrics.update(
        {
            "gap_um": (
                np.asarray(params.bore_radius_mm(cell_z)) - cell_rho_j
            )
            * 1_000.0,
            "layer_index": logical[:, 2].astype(np.float64),
            "theta_deg": np.degrees(cell_theta),
            "axial_index": logical[:, 1].astype(np.float64),
        }
    )
    metadata = dict(template.metadata)
    theta_widths = np.diff(theta_edges)
    axial_heights = np.diff(z_mm)
    theta_growth = np.maximum(
        theta_widths[1:] / theta_widths[:-1],
        theta_widths[:-1] / theta_widths[1:],
    )
    axial_growth = np.maximum(
        axial_heights[1:] / axial_heights[:-1],
        axial_heights[:-1] / axial_heights[1:],
    )
    metadata.update(
        {
            "theta_edge_coordinates_rad": theta_edges.tolist(),
            "z_node_coordinates_mm": z_mm.tolist(),
            "inlet_cluster_strength": cluster_strength,
            "inlet_cluster_power": 3.0,
            "theta_cell_width_ratio": float(
                theta_widths.max() / theta_widths.min()
            ),
            "axial_cell_height_ratio": float(
                axial_heights.max() / axial_heights.min()
            ),
            "theta_max_adjacent_width_growth": float(theta_growth.max()),
            "axial_max_adjacent_height_growth": float(axial_growth.max()),
            "gap_layer_coordinates": xi.tolist(),
            "gap_layer_fractions": fractions.tolist(),
            "gap_inflation_ratio_target": inflation_ratio,
            "gap_inflation_ratio_achieved": float(fractions.max() / fractions.min()),
            "surface_pressure_inlet": True,
            "contains_feed_volume": False,
        }
    )
    return replace(
        template,
        points_m=points,
        cell_centres_m=centres,
        cell_metrics={
            name: np.ascontiguousarray(values, dtype=np.float64)
            for name, values in metrics.items()
        },
        metadata=metadata,
    )


def _split_pressure_patch(
    mesh: base.MeshData, inlet: InletSpec
) -> tuple[base.MeshData, np.ndarray]:
    outer = np.asarray(mesh.boundary_quads["stationary_wall"])
    centres_m, _areas_m2 = base._quad_geometry(
        mesh.points_m[outer.astype(np.int64) - 1]
    )
    centres_mm = centres_m / SI_PER_MM
    mask = (centres_mm[:, 1] > 0.0) & (
        centres_mm[:, 0] ** 2
        + (centres_mm[:, 2] - inlet.axial_position_mm) ** 2
        <= inlet.radius_mm**2
    )
    if not np.any(mask):
        raise SurfaceInletError(
            "the mesh-aligned inlet selected no bore faces; increase n-theta/n-axial"
        )
    boundaries = {
        "journal_wall": np.asarray(mesh.boundary_quads["journal_wall"]),
        "stationary_wall": np.ascontiguousarray(outer[~mask]),
        "axial_end_z0": np.asarray(mesh.boundary_quads["axial_end_z0"]),
        "axial_end_zL": np.asarray(mesh.boundary_quads["axial_end_zL"]),
        "pressure_feed": np.ascontiguousarray(outer[mask]),
    }
    split = replace(
        mesh,
        boundary_quads=boundaries,
        metadata=dict(mesh.metadata)
        | {
            "pressure_patch_rule": (
                "whole xi=1 Quad4 faces whose +Y face centre lies inside "
                "x^2+(z-zh)^2<=rh^2"
            ),
            "pressure_patch_quad_count": int(mask.sum()),
        },
    )
    return split, outer


def build_surface_inlet_mesh(
    params: base.BearingParams,
    inlet: InletSpec,
    n_theta: int,
    n_axial: int,
    n_gap: int,
    gap_inflation_ratio: float = 5.0,
    inlet_cluster_strength: float = 0.82,
) -> tuple[base.MeshData, np.ndarray, float]:
    template = base.generate_mesh(params, n_theta, n_axial, n_gap)
    inflated = _remapped_mesh(
        template,
        params,
        inlet,
        gap_inflation_ratio,
        inlet_cluster_strength,
    )
    reference_template = base.generate_mesh(params, n_theta, n_axial, 1)
    reference = _remapped_mesh(
        reference_template,
        params,
        inlet,
        1.0,
        inlet_cluster_strength,
    )
    reference_volume = float(reference.cell_metrics["signed_volume_m3"].sum())
    split, original_outer = _split_pressure_patch(inflated, inlet)
    return split, original_outer, reference_volume


def _edge_census(quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate(
        (
            quads[:, (0, 1)],
            quads[:, (1, 2)],
            quads[:, (2, 3)],
            quads[:, (3, 0)],
        )
    )
    return np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)


def _connected_component_count(nodes: np.ndarray, edges: np.ndarray) -> int:
    adjacency: dict[int, set[int]] = {int(node): set() for node in nodes}
    for first, second in edges:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            unseen = adjacency[current] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    return components


def pressure_patch_diagnostics(
    mesh: base.MeshData, params: base.BearingParams, inlet: InletSpec
) -> dict[str, Any]:
    quads = np.asarray(mesh.boundary_quads["pressure_feed"])
    vertices = mesh.points_m[quads.astype(np.int64) - 1]
    centres, area_vectors = base._quad_geometry(vertices)
    areas = np.linalg.norm(area_vectors, axis=1)
    projected_weights = area_vectors[:, 1]
    if np.any(projected_weights <= 0.0):
        raise SurfaceInletError("pressure patch is not consistently oriented toward +Y")
    unique_edges, edge_counts = _edge_census(quads)
    rim_edges = unique_edges[edge_counts == 1]
    rim_nodes, rim_degrees = np.unique(rim_edges, return_counts=True)
    rim_points_mm = mesh.points_m[rim_nodes.astype(np.int64) - 1] / SI_PER_MM
    rim_radius_mm = np.hypot(
        rim_points_mm[:, 0], rim_points_mm[:, 2] - inlet.axial_position_mm
    )
    projected_area_mm2 = float(projected_weights.sum() / SI_PER_MM**2)
    surface_area_mm2 = float(areas.sum() / SI_PER_MM**2)
    exact_disk_area_mm2 = math.pi * inlet.radius_mm**2
    weighted_centre = np.average(centres, axis=0, weights=areas) / SI_PER_MM
    projected_centre = np.average(
        centres[:, (0, 2)] / SI_PER_MM, axis=0, weights=projected_weights
    )
    target_centre = np.asarray(
        [0.0, float(params.bore_radius_mm(inlet.axial_position_mm)), inlet.axial_position_mm]
    )
    net_normal = area_vectors.sum(axis=0)
    net_normal /= np.linalg.norm(net_normal)
    theta_edges = np.asarray(
        mesh.metadata["theta_edge_coordinates_rad"], dtype=np.float64
    )
    z_nodes = np.asarray(
        mesh.metadata["z_node_coordinates_mm"], dtype=np.float64
    )
    feed_theta_node = int(np.argmin(np.abs(theta_edges - math.pi)))
    feed_z_node = int(np.argmin(np.abs(z_nodes - inlet.axial_position_mm)))
    theta_widths = np.diff(theta_edges)
    axial_heights = np.diff(z_nodes)
    adjacent_theta_widths = theta_widths[
        [feed_theta_node - 1, feed_theta_node]
    ]
    adjacent_axial_heights = axial_heights[
        [feed_z_node - 1, feed_z_node]
    ]
    bore_radius_at_feed = float(
        params.bore_radius_mm(inlet.axial_position_mm)
    )
    return {
        "quad_count": len(quads),
        "surface_area_mm2": surface_area_mm2,
        "projected_xz_area_mm2": projected_area_mm2,
        "analytic_circle_area_mm2": exact_disk_area_mm2,
        "projected_area_relative_error": relative_error(
            projected_area_mm2, exact_disk_area_mm2
        ),
        "surface_area_vs_flat_circle_relative_difference": (
            surface_area_mm2 - exact_disk_area_mm2
        )
        / exact_disk_area_mm2,
        "projected_equivalent_diameter_mm": 2.0
        * math.sqrt(projected_area_mm2 / math.pi),
        "surface_area_weighted_centroid_mm": weighted_centre.tolist(),
        "nominal_bore_centre_mm": target_centre.tolist(),
        "centroid_offset_from_nominal_mm": float(
            np.linalg.norm(weighted_centre - target_centre)
        ),
        "projected_centroid_xz_mm": projected_centre.tolist(),
        "net_outward_unit_normal": net_normal.tolist(),
        "minimum_face_normal_dot_plus_y": float(
            (area_vectors[:, 1] / areas).min()
        ),
        "rim_edge_count": len(rim_edges),
        "rim_vertex_count": len(rim_nodes),
        "rim_vertex_degrees": {
            str(int(value)): int(np.count_nonzero(rim_degrees == value))
            for value in np.unique(rim_degrees)
        },
        "rim_connected_components": _connected_component_count(rim_nodes, rim_edges),
        "rim_radius_mm": {
            "minimum": float(rim_radius_mm.min()),
            "maximum": float(rim_radius_mm.max()),
            "maximum_absolute_error": float(
                np.abs(rim_radius_mm - inlet.radius_mm).max()
            ),
        },
        "mesh_resolution_at_feed": {
            "adjacent_theta_cell_widths_deg": np.degrees(
                adjacent_theta_widths
            ).tolist(),
            "maximum_adjacent_bore_arc_width_mm": float(
                adjacent_theta_widths.max() * bore_radius_at_feed
            ),
            "adjacent_axial_cell_heights_mm": (
                adjacent_axial_heights.tolist()
            ),
            "maximum_adjacent_axial_cell_height_mm": float(
                adjacent_axial_heights.max()
            ),
            "global_theta_width_min_deg": float(
                math.degrees(theta_widths.min())
            ),
            "global_theta_width_max_deg": float(
                math.degrees(theta_widths.max())
            ),
            "global_axial_height_min_mm": float(axial_heights.min()),
            "global_axial_height_max_mm": float(axial_heights.max()),
            "theta_width_ratio": float(
                theta_widths.max() / theta_widths.min()
            ),
            "axial_height_ratio": float(
                axial_heights.max() / axial_heights.min()
            ),
            "maximum_adjacent_theta_width_growth": mesh.metadata[
                "theta_max_adjacent_width_growth"
            ],
            "maximum_adjacent_axial_height_growth": mesh.metadata[
                "axial_max_adjacent_height_growth"
            ],
        },
        "representation": (
            "cluster-refined mesh-aligned union of complete bore Quad4 faces; "
            "not an exact circular cut"
        ),
    }


def _boundary_owner_indices(mesh: base.MeshData, name: str) -> np.ndarray:
    quads = np.asarray(mesh.boundary_quads[name])
    n_theta = int(mesh.metadata["n_theta"])
    n_axial = int(mesh.metadata["n_axial"])
    n_gap = int(mesh.metadata["n_gap"])
    first = quads[:, 0].astype(np.uint64) - 1
    layer = (first % np.uint64(n_gap + 1)).astype(np.int64)
    plane_index = (first // np.uint64(n_gap + 1)).astype(np.int64)
    theta_index = plane_index % n_theta
    axial_node_index = plane_index // n_theta
    if name == "journal_wall":
        axial_cell_index = axial_node_index
        gap_index = np.zeros(len(quads), dtype=np.int64)
    elif name in {"stationary_wall", "pressure_feed"}:
        axial_cell_index = axial_node_index
        gap_index = np.full(len(quads), n_gap - 1, dtype=np.int64)
    elif name == "axial_end_z0":
        axial_cell_index = np.zeros(len(quads), dtype=np.int64)
        gap_index = layer
    elif name == "axial_end_zL":
        axial_cell_index = np.full(len(quads), n_axial - 1, dtype=np.int64)
        gap_index = layer
    else:
        raise SurfaceInletError(f"unknown boundary group {name}")
    return gap_index + n_gap * (axial_cell_index + n_axial * theta_index)


def _validate_boundary_orientations(
    mesh: base.MeshData,
    params: base.BearingParams,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    eccentricity_m = params.eccentricity_mm * SI_PER_MM
    for name, quads in mesh.boundary_quads.items():
        centres, area_vectors = base._quad_geometry(
            mesh.points_m[np.asarray(quads, dtype=np.int64) - 1]
        )
        owner_centres = mesh.cell_centres_m[_boundary_owner_indices(mesh, name)]
        owner_dot = np.einsum(
            "mc,mc->m", area_vectors, centres - owner_centres
        )
        if name == "journal_wall":
            radius_m = (
                np.asarray(params.journal_radius_mm(centres[:, 2] / SI_PER_MM))
                * SI_PER_MM
            )
            analytic = np.column_stack(
                (
                    -centres[:, 0],
                    -(centres[:, 1] + eccentricity_m),
                    -radius_m * params.cone_slope,
                )
            )
        elif name in {"stationary_wall", "pressure_feed"}:
            radius_m = (
                np.asarray(params.bore_radius_mm(centres[:, 2] / SI_PER_MM))
                * SI_PER_MM
            )
            analytic = np.column_stack(
                (centres[:, 0], centres[:, 1], radius_m * params.cone_slope)
            )
        elif name == "axial_end_z0":
            analytic = np.broadcast_to((0.0, 0.0, -1.0), centres.shape)
        else:
            analytic = np.broadcast_to((0.0, 0.0, 1.0), centres.shape)
        analytic_dot = np.einsum("mc,mc->m", area_vectors, analytic)
        require(
            records,
            f"boundary.{name}.outward_from_owner",
            bool(np.all(owner_dot > 0.0)),
            float(owner_dot.min()),
            ">0",
        )
        require(
            records,
            f"boundary.{name}.analytic_outward",
            bool(np.all(analytic_dot > 0.0)),
            float(analytic_dot.min()),
            ">0",
        )
        diagnostics[name] = {
            "quad_count": len(quads),
            "minimum_owner_dot": float(owner_dot.min()),
            "minimum_analytic_dot": float(analytic_dot.min()),
        }
    return diagnostics


def validate_surface_inlet_mesh(
    mesh: base.MeshData,
    original_outer: np.ndarray,
    uniform_volume_m3: float,
    params: base.BearingParams,
    inlet: InletSpec,
    max_projected_area_relative_error: float,
    records: list[dict[str, Any]],
    max_inlet_rim_error_mm: float = 0.16,
) -> dict[str, Any]:
    n_theta = int(mesh.metadata["n_theta"])
    n_axial = int(mesh.metadata["n_axial"])
    n_gap = int(mesh.metadata["n_gap"])
    expected_points = n_theta * (n_axial + 1) * (n_gap + 1)
    expected_hexes = n_theta * n_axial * n_gap
    require(records, "counts.points", len(mesh.points_m) == expected_points, len(mesh.points_m), expected_points)
    require(records, "counts.Hex8", len(mesh.hexes) == expected_hexes, len(mesh.hexes), expected_hexes)
    require(
        records,
        "counts.only_Hex8_volume_cells",
        mesh.hexes.shape == (expected_hexes, 8),
        mesh.hexes.shape,
        (expected_hexes, 8),
    )
    require(
        records,
        "data.finite_coordinates_and_metrics",
        np.isfinite(mesh.points_m).all()
        and np.isfinite(mesh.cell_centres_m).all()
        and all(np.isfinite(values).all() for values in mesh.cell_metrics.values()),
        "finite",
        "finite",
    )
    require(
        records,
        "data.no_duplicate_nodes",
        len(np.unique(mesh.points_m, axis=0)) == len(mesh.points_m),
        len(np.unique(mesh.points_m, axis=0)),
        len(mesh.points_m),
    )
    require(
        records,
        "data.no_duplicate_hexes",
        len(np.unique(np.sort(mesh.hexes, axis=1), axis=0)) == len(mesh.hexes),
        len(np.unique(np.sort(mesh.hexes, axis=1), axis=0)),
        len(mesh.hexes),
    )

    pressure = np.asarray(mesh.boundary_quads["pressure_feed"])
    stationary = np.asarray(mesh.boundary_quads["stationary_wall"])
    reconstructed = np.concatenate((stationary, pressure))
    reconstructed_keys = base._sorted_rows(np.sort(reconstructed, axis=1))
    original_keys = base._sorted_rows(np.sort(original_outer, axis=1))
    require(
        records,
        "patch.pressure_and_stationary_reconstruct_outer_wall",
        np.array_equal(reconstructed_keys, original_keys),
        len(reconstructed_keys),
        len(original_keys),
    )
    require(
        records,
        "patch.pressure_nonempty",
        len(pressure) > 0,
        len(pressure),
        ">0",
    )
    pressure_layers = (
        (pressure.astype(np.uint64) - 1) % np.uint64(n_gap + 1)
    )
    require(
        records,
        "patch.pressure_faces_are_only_xi1",
        bool(np.all(pressure_layers == n_gap)),
        sorted(set(pressure_layers.ravel().tolist())),
        [n_gap],
    )
    require(
        records,
        "patch.journal_continuous_and_untouched",
        len(mesh.boundary_quads["journal_wall"]) == n_theta * n_axial,
        len(mesh.boundary_quads["journal_wall"]),
        n_theta * n_axial,
    )

    patch = pressure_patch_diagnostics(mesh, params, inlet)
    require(
        records,
        "patch.projected_area_accuracy",
        patch["projected_area_relative_error"]
        <= max_projected_area_relative_error,
        patch["projected_area_relative_error"],
        max_projected_area_relative_error,
    )
    require(
        records,
        "patch.maximum_rim_radius_error_mm",
        patch["rim_radius_mm"]["maximum_absolute_error"]
        <= max_inlet_rim_error_mm,
        patch["rim_radius_mm"]["maximum_absolute_error"],
        max_inlet_rim_error_mm,
    )
    require(
        records,
        "patch.one_closed_degree_two_rim",
        patch["rim_connected_components"] == 1
        and patch["rim_vertex_degrees"] == {"2": patch["rim_vertex_count"]},
        {
            "components": patch["rim_connected_components"],
            "degrees": patch["rim_vertex_degrees"],
        },
        {"components": 1, "degrees": {"2": patch["rim_vertex_count"]}},
    )
    require(
        records,
        "patch.outward_normal_has_positive_y",
        patch["minimum_face_normal_dot_plus_y"] > 0.9
        and patch["net_outward_unit_normal"][1] > 0.9,
        {
            "minimum_face_dot_plus_y": patch["minimum_face_normal_dot_plus_y"],
            "net_normal": patch["net_outward_unit_normal"],
        },
        "both +Y components >0.9",
    )
    projected_centroid = patch["projected_centroid_xz_mm"]
    local_resolution = patch["mesh_resolution_at_feed"]
    require(
        records,
        "patch.centred_near_feed_axis",
        abs(projected_centroid[0])
        <= local_resolution["maximum_adjacent_bore_arc_width_mm"]
        and abs(projected_centroid[1] - inlet.axial_position_mm)
        <= local_resolution["maximum_adjacent_axial_cell_height_mm"],
        projected_centroid,
        [0.0, inlet.axial_position_mm],
    )

    grid = mesh.points_m.reshape(n_axial + 1, n_theta, n_gap + 1, 3)
    z_mm = grid[:, 0, 0, 2] / SI_PER_MM
    theta_edges = np.asarray(
        mesh.metadata["theta_edge_coordinates_rad"], dtype=np.float64
    )
    stored_z_nodes = np.asarray(
        mesh.metadata["z_node_coordinates_mm"], dtype=np.float64
    )
    theta_widths = np.diff(theta_edges)
    axial_heights = np.diff(stored_z_nodes)
    require(
        records,
        "clustering.theta_coordinates_monotonic_and_anchored",
        len(theta_edges) == n_theta + 1
        and np.all(theta_widths > 0.0)
        and theta_edges[0] == 0.0
        and theta_edges[-1] == 2.0 * math.pi
        and theta_edges[n_theta // 2] == math.pi,
        {
            "count": len(theta_edges),
            "first": float(theta_edges[0]),
            "feed": float(theta_edges[n_theta // 2]),
            "last": float(theta_edges[-1]),
            "minimum_width": float(theta_widths.min()),
        },
        "strictly increasing, [0,pi,2pi] anchored",
    )
    feed_z_index = int(
        np.argmin(np.abs(stored_z_nodes - inlet.axial_position_mm))
    )
    require(
        records,
        "clustering.axial_coordinates_monotonic_and_anchored",
        len(stored_z_nodes) == n_axial + 1
        and np.all(axial_heights > 0.0)
        and stored_z_nodes[0] == 0.0
        and stored_z_nodes[-1] == params.length_mm
        and stored_z_nodes[feed_z_index] == inlet.axial_position_mm
        and np.allclose(z_mm, stored_z_nodes, rtol=0.0, atol=1.0e-12),
        {
            "count": len(stored_z_nodes),
            "first": float(stored_z_nodes[0]),
            "feed": float(stored_z_nodes[feed_z_index]),
            "last": float(stored_z_nodes[-1]),
            "minimum_height": float(axial_heights.min()),
        },
        "strictly increasing, ends and zh anchored",
    )
    journal_radius_m = (
        np.asarray(params.journal_radius_mm(z_mm))[:, None] * SI_PER_MM
    )
    bore_radius_m = np.asarray(params.bore_radius_mm(z_mm))[:, None] * SI_PER_MM
    journal_residual = (
        np.sqrt(
            grid[:, :, 0, 0] ** 2
            + (grid[:, :, 0, 1] + params.eccentricity_mm * SI_PER_MM) ** 2
        )
        - journal_radius_m
    )
    bore_residual = (
        np.sqrt(grid[:, :, -1, 0] ** 2 + grid[:, :, -1, 1] ** 2)
        - bore_radius_m
    )
    require(
        records,
        "geometry.journal_cone_residual_m",
        float(np.abs(journal_residual).max()) <= 1.0e-12,
        float(np.abs(journal_residual).max()),
        1.0e-12,
    )
    require(
        records,
        "geometry.bore_cone_residual_m",
        float(np.abs(bore_residual).max()) <= 1.0e-12,
        float(np.abs(bore_residual).max()),
        1.0e-12,
    )
    rho = np.linalg.norm(grid[..., :2], axis=-1)
    gap_mm = (rho[:, :, -1] - rho[:, :, 0]) / SI_PER_MM
    require(
        records,
        "geometry.minimum_radial_gap_mm",
        float(np.abs(gap_mm[:, 0] - params.h_min_mm).max()) <= 1.0e-12,
        float(gap_mm[:, 0].mean()),
        params.h_min_mm,
        1.0e-12,
    )
    if n_theta % 2 == 0:
        require(
            records,
            "geometry.maximum_radial_gap_mm",
            float(np.abs(gap_mm[:, n_theta // 2] - params.h_max_mm).max())
            <= 1.0e-12,
            float(gap_mm[:, n_theta // 2].mean()),
            params.h_max_mm,
            1.0e-12,
        )
    expected_xi = np.asarray(mesh.metadata["gap_layer_coordinates"])
    measured_xi = (rho - rho[:, :, :1]) / (
        rho[:, :, -1:] - rho[:, :, :1]
    )
    require(
        records,
        "inflation.exact_layer_coordinates",
        float(
            np.abs(measured_xi - expected_xi[None, None, :]).max()
        )
        <= 1.0e-10,
        float(np.abs(measured_xi - expected_xi[None, None, :]).max()),
        1.0e-10,
    )
    fractions = np.diff(expected_xi)
    require(
        records,
        "inflation.symmetric_positive_5_to_1",
        np.all(fractions > 0.0)
        and np.allclose(fractions, fractions[::-1], atol=1.0e-14, rtol=0.0)
        and math.isclose(
            float(fractions.max() / fractions.min()),
            float(mesh.metadata["gap_inflation_ratio_target"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        {
            "fractions": fractions.tolist(),
            "ratio": float(fractions.max() / fractions.min()),
        },
        {
            "symmetric": True,
            "ratio": mesh.metadata["gap_inflation_ratio_target"],
        },
    )

    signed = mesh.cell_metrics["signed_volume_m3"]
    gauss = mesh.cell_metrics["gauss_volume_m3"]
    require(records, "quality.positive_signed_volume", bool(np.all(signed > 0.0)), float(signed.min()), ">0")
    require(records, "quality.positive_gauss_volume", bool(np.all(gauss > 0.0)), float(gauss.min()), ">0")
    require(
        records,
        "quality.positive_face_pyramids",
        bool(np.all(mesh.cell_metrics["min_face_pyramid_m3"] > 0.0)),
        float(mesh.cell_metrics["min_face_pyramid_m3"].min()),
        ">0",
    )
    mesh_volume = float(signed.sum())
    require(
        records,
        "volume.inflation_preserves_faceted_volume",
        relative_error(mesh_volume, uniform_volume_m3) <= 1.0e-12,
        relative_error(mesh_volume, uniform_volume_m3),
        1.0e-12,
    )
    continuous_error = relative_error(mesh_volume, params.exact_volume_m3)
    require(
        records,
        "volume.faceted_vs_exact_annulus",
        continuous_error <= 5.0e-4,
        continuous_error,
        5.0e-4,
    )
    require(
        records,
        "geometry.no_feed_column",
        float(mesh.points_m[:, 1].max())
        <= float(np.max(params.bore_radius_mm(np.asarray([0.0, params.length_mm]))))
        * SI_PER_MM
        + 1.0e-12,
        float(mesh.points_m[:, 1].max()),
        "within bore envelope",
    )
    require(
        records,
        "quality.maximum_nonorthogonality",
        float(mesh.cell_metrics["max_nonorthogonality_deg"].max()) <= 45.0,
        float(mesh.cell_metrics["max_nonorthogonality_deg"].max()),
        45.0,
    )
    require(
        records,
        "quality.maximum_skewness",
        float(mesh.cell_metrics["max_skewness"].max()) <= 4.0,
        float(mesh.cell_metrics["max_skewness"].max()),
        4.0,
    )
    topology = base.validate_topology(mesh, records)
    orientations = _validate_boundary_orientations(mesh, params, records)
    return {
        "counts": {
            "points": expected_points,
            "hexes": expected_hexes,
            "boundary_quads": {
                name: len(quads) for name, quads in mesh.boundary_quads.items()
            },
        },
        "radial_gap_mm": {"minimum": params.h_min_mm, "maximum": params.h_max_mm},
        "inflation": {
            "target_ratio": mesh.metadata["gap_inflation_ratio_target"],
            "achieved_ratio": mesh.metadata["gap_inflation_ratio_achieved"],
            "layer_coordinates": expected_xi.tolist(),
            "layer_fractions": fractions.tolist(),
        },
        "inlet_clustering": {
            "strength": mesh.metadata["inlet_cluster_strength"],
            "power": mesh.metadata["inlet_cluster_power"],
            "theta_cell_width_ratio": mesh.metadata[
                "theta_cell_width_ratio"
            ],
            "axial_cell_height_ratio": mesh.metadata[
                "axial_cell_height_ratio"
            ],
            "theta_max_adjacent_width_growth": mesh.metadata[
                "theta_max_adjacent_width_growth"
            ],
            "axial_max_adjacent_height_growth": mesh.metadata[
                "axial_max_adjacent_height_growth"
            ],
            "mesh_resolution_at_feed": patch["mesh_resolution_at_feed"],
        },
        "pressure_patch": patch,
        "mesh_volume_m3": mesh_volume,
        "exact_continuous_annulus_volume_m3": params.exact_volume_m3,
        "faceted_volume_relative_error": continuous_error,
        "topology": topology,
        "boundary_orientation": orientations,
    }


def expected_physical_groups() -> dict[tuple[int, int], dict[str, Any]]:
    groups = {
        (2, PHYSICAL_IDS[name]): {
            "name": name,
            "entities": [SURFACE_ENTITIES[name]],
        }
        for name in SURFACE_ENTITIES
    }
    groups[(3, PHYSICAL_IDS["fluid"])] = {
        "name": "fluid",
        "entities": [VOLUME_ENTITY],
    }
    return groups


def validate_physical_groups(
    records: list[dict[str, Any]], prefix: str
) -> dict[str, Any]:
    actual: dict[tuple[int, int], dict[str, Any]] = {}
    for dimension, physical_id in gmsh.model.getPhysicalGroups():
        key = (int(dimension), int(physical_id))
        actual[key] = {
            "name": gmsh.model.getPhysicalName(*key),
            "entities": sorted(
                int(tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(*key)
            ),
        }
    expected = expected_physical_groups()

    def serialize(groups: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
        return {
            f"{dimension}:{physical_id}": value
            for (dimension, physical_id), value in groups.items()
        }

    require(
        records,
        f"{prefix}.physical_groups_exact",
        actual == expected,
        serialize(actual),
        serialize(expected),
    )
    forbidden = {"feed_tube_wall", "feed_mouth", "defaultFaces", "mouth_cap"}
    require(
        records,
        f"{prefix}.no_feed_tube_or_internal_patch",
        forbidden.isdisjoint(value["name"] for value in actual.values()),
        sorted(value["name"] for value in actual.values()),
        f"none of {sorted(forbidden)}",
    )
    return serialize(actual)


def add_discrete_model(
    mesh: base.MeshData, records: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    gmsh.model.add(model_name)
    for name, tag in SURFACE_ENTITIES.items():
        gmsh.model.addDiscreteEntity(2, tag)
        gmsh.model.setEntityName(2, tag, name)
    gmsh.model.addDiscreteEntity(3, VOLUME_ENTITY, list(SURFACE_ENTITIES.values()))
    gmsh.model.setEntityName(3, VOLUME_ENTITY, "fluid")

    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    require(records, "gmsh.element_type.Quad4", quad_type == 3, quad_type, 3)
    require(records, "gmsh.element_type.Hex8", hex_type == 5, hex_type, 5)
    gmsh.model.mesh.addNodes(
        3, VOLUME_ENTITY, mesh.node_tags, mesh.points_m.ravel()
    )
    next_tag = int(mesh.cell_tags[-1]) + 1
    quad_tags: dict[str, np.ndarray] = {}
    for name, quads in mesh.boundary_quads.items():
        tags = np.arange(next_tag, next_tag + len(quads), dtype=np.uint64)
        next_tag += len(quads)
        quad_tags[name] = tags
        gmsh.model.mesh.addElementsByType(
            SURFACE_ENTITIES[name], quad_type, tags, quads.ravel()
        )
    gmsh.model.mesh.addElementsByType(
        VOLUME_ENTITY, hex_type, mesh.cell_tags, mesh.hexes.ravel()
    )
    gmsh.model.mesh.reclassifyNodes()

    for name in SURFACE_ENTITIES:
        created = gmsh.model.addPhysicalGroup(
            2,
            [SURFACE_ENTITIES[name]],
            tag=PHYSICAL_IDS[name],
            name=name,
        )
        require(
            records,
            f"physical.{name}.id",
            created == PHYSICAL_IDS[name],
            created,
            PHYSICAL_IDS[name],
        )
    created = gmsh.model.addPhysicalGroup(
        3, [VOLUME_ENTITY], tag=PHYSICAL_IDS["fluid"], name="fluid"
    )
    require(
        records,
        "physical.fluid.id",
        created == PHYSICAL_IDS["fluid"],
        created,
        PHYSICAL_IDS["fluid"],
    )
    return {
        "element_types": {"Quad4": quad_type, "Hex8": hex_type},
        "quad_elements": {
            name: {
                "count": len(tags),
                "first_tag": int(tags[0]),
                "last_tag": int(tags[-1]),
            }
            for name, tags in quad_tags.items()
        },
        "hex_count": len(mesh.hexes),
    }


def write_gmsh_files(case_dir: Path) -> tuple[Path, Path]:
    msh41 = case_dir / "structured_hex.msh"
    msh22 = case_dir / "structured_hex_openfoam.msh"
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(msh41))
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(msh22))
    return msh41, msh22


def write_quality_view(
    mesh: base.MeshData, views_dir: Path, name: str
) -> dict[str, dict[str, Any]]:
    views_dir.mkdir(parents=True, exist_ok=True)
    view_tag = gmsh.view.add(name)
    gmsh.view.addModelData(
        view_tag,
        0,
        gmsh.model.getCurrent(),
        "ElementData",
        mesh.cell_tags,
        mesh.cell_metrics[name].reshape(-1, 1),
        numComponents=1,
    )
    path = views_dir / f"{name}.pos"
    gmsh.view.write(view_tag, str(path))
    gmsh.view.remove(view_tag)
    return {name: {"path": str(path.name), "bytes": path.stat().st_size}}


def validate_gmsh_round_trip(
    path: Path,
    mesh: base.MeshData,
    records: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.open(str(path))
    physical = validate_physical_groups(records, prefix)
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags_raw, dtype=np.int64)
    points = np.asarray(coordinates_raw, dtype=np.float64).reshape(-1, 3)
    mapping = base._coordinate_node_mapping(tags, points, mesh, records, prefix)
    hex_tags_raw, hex_nodes_raw = gmsh.model.mesh.getElementsByType(
        hex_type, VOLUME_ENTITY
    )
    hex_tags = np.asarray(hex_tags_raw, dtype=np.int64)
    mapped_hexes = mapping[
        np.asarray(hex_nodes_raw, dtype=np.int64).reshape(-1, 8)
    ]
    base._compare_oriented_connectivity(
        mapped_hexes, mesh.hexes, records, f"{prefix}.Hex8_connectivity"
    )
    for name, expected in mesh.boundary_quads.items():
        _quad_tags, quad_nodes = gmsh.model.mesh.getElementsByType(
            quad_type, SURFACE_ENTITIES[name]
        )
        mapped = mapping[
            np.asarray(quad_nodes, dtype=np.int64).reshape(-1, 4)
        ]
        base._compare_oriented_connectivity(
            mapped,
            expected,
            records,
            f"{prefix}.{name}_Quad4_connectivity",
        )
    types, tags_by_type, _nodes_by_type = gmsh.model.mesh.getElements()
    counts = {
        int(element_type): len(element_tags)
        for element_type, element_tags in zip(types, tags_by_type)
    }
    expected_counts = {
        quad_type: sum(len(quads) for quads in mesh.boundary_quads.values()),
        hex_type: len(mesh.hexes),
    }
    require(
        records,
        f"{prefix}.only_Quad4_and_Hex8",
        counts == expected_counts,
        counts,
        expected_counts,
    )
    bbox = (*points.min(axis=0).tolist(), *points.max(axis=0).tolist())
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
        gmsh.model.mesh.getElementQualities(hex_tags, "minDetJac"),
        dtype=np.float64,
    )
    volumes = np.asarray(
        gmsh.model.mesh.getElementQualities(hex_tags, "volume"),
        dtype=np.float64,
    )
    require(
        records,
        f"{prefix}.positive_minDetJac",
        np.isfinite(min_det).all() and np.all(min_det > 0.0),
        float(min_det.min()),
        ">0",
    )
    require(
        records,
        f"{prefix}.positive_volume",
        np.isfinite(volumes).all() and np.all(volumes > 0.0),
        float(volumes.min()),
        ">0",
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "point_count": len(points),
        "element_type_counts": counts,
        "physical_groups": physical,
        "bbox_m": bbox,
        "minimum_minDetJac": float(min_det.min()),
        "minimum_volume_m3": float(volumes.min()),
    }


def write_vtu_files(
    mesh: base.MeshData, case_dir: Path
) -> tuple[Path, Path]:
    volume_path = case_dir / "volume_hex.vtu"
    boundary_path = case_dir / "boundary_quads.vtu"
    cell_count = len(mesh.hexes)
    cell_data = {
        name: [np.asarray(mesh.cell_metrics[name])]
        for name in base.VIEW_NAMES
    }
    cell_data.update(
        {
            "theta_index": [mesh.logical_cell_indices[:, 0]],
            "axial_index_integer": [mesh.logical_cell_indices[:, 1]],
            "gap_index": [mesh.logical_cell_indices[:, 2]],
            "cell_tag": [mesh.cell_tags],
            "physical_id": [
                np.full(cell_count, PHYSICAL_IDS["fluid"], dtype=np.int32)
            ],
            "solve_eligible": [np.ones(cell_count, dtype=np.uint8)],
            "distorted_geometry": [np.zeros(cell_count, dtype=np.uint8)],
        }
    )
    meshio.write(
        volume_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("hexahedron", mesh.hexes.astype(np.int64) - 1)],
            cell_data=cell_data,
            field_data={"fluid": np.asarray([PHYSICAL_IDS["fluid"], 3])},
        ),
        file_format="vtu",
        binary=True,
    )
    patch_order = tuple(SURFACE_ENTITIES)
    boundary = np.concatenate(
        [mesh.boundary_quads[name] for name in patch_order]
    )
    patch_ids = np.concatenate(
        [
            np.full(
                len(mesh.boundary_quads[name]),
                PHYSICAL_IDS[name],
                dtype=np.int32,
            )
            for name in patch_order
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
                for name in patch_order
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return volume_path, boundary_path


def validate_vtu_round_trip(
    mesh: base.MeshData,
    volume_path: Path,
    boundary_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    volume = meshio.read(volume_path)
    boundary = meshio.read(boundary_path)
    read_hexes = volume.cells_dict.get("hexahedron")
    read_quads = boundary.cells_dict.get("quad")
    expected_quads = (
        np.concatenate(list(mesh.boundary_quads.values())).astype(np.int64) - 1
    )
    require(
        records,
        "vtu.volume_exact_coordinates_and_Hex8",
        np.array_equal(volume.points, mesh.points_m)
        and read_hexes is not None
        and np.array_equal(read_hexes, mesh.hexes.astype(np.int64) - 1),
        {
            "points": len(volume.points),
            "cell_types": [cell.type for cell in volume.cells],
            "hexes": 0 if read_hexes is None else len(read_hexes),
        },
        {"points": len(mesh.points_m), "cell_types": ["hexahedron"], "hexes": len(mesh.hexes)},
    )
    patch_ids = np.concatenate(
        [
            np.full(
                len(mesh.boundary_quads[name]),
                PHYSICAL_IDS[name],
                dtype=np.int32,
            )
            for name in SURFACE_ENTITIES
        ]
    )
    read_patch_ids = boundary.cell_data_dict.get("patch_id", {}).get("quad")
    require(
        records,
        "vtu.boundary_exact_Quad4_and_patch_ids",
        np.array_equal(boundary.points, mesh.points_m)
        and read_quads is not None
        and np.array_equal(read_quads, expected_quads)
        and read_patch_ids is not None
        and np.array_equal(read_patch_ids, patch_ids),
        {
            "points": len(boundary.points),
            "cell_types": [cell.type for cell in boundary.cells],
            "quads": 0 if read_quads is None else len(read_quads),
            "patch_ids": []
            if read_patch_ids is None
            else sorted(set(read_patch_ids.tolist())),
        },
        {
            "points": len(mesh.points_m),
            "cell_types": ["quad"],
            "quads": len(expected_quads),
            "patch_ids": sorted(PHYSICAL_IDS[name] for name in SURFACE_ENTITIES),
        },
    )
    expected_fields: dict[str, np.ndarray] = {
        **{name: np.asarray(mesh.cell_metrics[name]) for name in base.VIEW_NAMES},
        "theta_index": mesh.logical_cell_indices[:, 0],
        "axial_index_integer": mesh.logical_cell_indices[:, 1],
        "gap_index": mesh.logical_cell_indices[:, 2],
        "cell_tag": mesh.cell_tags,
        "physical_id": np.full(
            len(mesh.hexes), PHYSICAL_IDS["fluid"], dtype=np.int32
        ),
        "solve_eligible": np.ones(len(mesh.hexes), dtype=np.uint8),
        "distorted_geometry": np.zeros(len(mesh.hexes), dtype=np.uint8),
    }
    wrong_fields = {
        name: {
            "count": 0 if actual is None else len(actual),
            "values_equal": actual is not None
            and np.array_equal(actual, expected),
        }
        for name, expected in expected_fields.items()
        if not np.array_equal(
            actual := volume.cell_data_dict.get(name, {}).get("hexahedron"),
            expected,
        )
    }
    require(
        records,
        "vtu.volume_fields_exact",
        not wrong_fields,
        wrong_fields,
        f"exact values for {len(expected_fields)} fields",
    )
    records.append(
        {
            "name": "vtu.physical_names",
            "status": "SKIPPED",
            "actual": "meshio VTU round-trip does not retain field_data names",
            "expected": (
                "exact physical_id/patch_id arrays in VTU; names in "
                "physical_groups.json and Gmsh physical groups"
            ),
            "tolerance": None,
            "mandatory": False,
        }
    )
    return {
        "volume_sha256": sha256_file(volume_path),
        "boundary_sha256": sha256_file(boundary_path),
        "points": len(mesh.points_m),
        "hexes": len(mesh.hexes),
        "quads": len(expected_quads),
        "patch_ids": sorted(PHYSICAL_IDS[name] for name in SURFACE_ENTITIES),
    }


def _write_diagnostic_msh(
    mesh: base.MeshData,
    path: Path,
    cell_mask: np.ndarray | None = None,
    pressure_only: bool = False,
) -> None:
    gmsh.clear()
    gmsh.model.add(path.stem)
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    hex_type = int(gmsh.model.mesh.getElementType("Hexahedron", 1))
    if pressure_only:
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.setEntityName(2, 1, "pressure_feed")
        quads = np.asarray(mesh.boundary_quads["pressure_feed"])
        used_tags = np.unique(quads)
        gmsh.model.mesh.addNodes(
            2,
            1,
            used_tags,
            mesh.points_m[used_tags.astype(np.int64) - 1].ravel(),
        )
        tags = np.arange(1, len(quads) + 1, dtype=np.uint64)
        gmsh.model.mesh.addElementsByType(1, quad_type, tags, quads.ravel())
        gmsh.model.addPhysicalGroup(2, [1], 106, "pressure_feed")
    else:
        selected = (
            np.ones(len(mesh.hexes), dtype=bool)
            if cell_mask is None
            else np.asarray(cell_mask, dtype=bool)
        )
        gmsh.model.addDiscreteEntity(3, 1)
        gmsh.model.setEntityName(3, 1, "DIAGNOSTIC_ONLY_DO_NOT_SOLVE")
        gmsh.model.mesh.addNodes(3, 1, mesh.node_tags, mesh.points_m.ravel())
        gmsh.model.mesh.addElementsByType(
            1,
            hex_type,
            mesh.cell_tags[selected],
            mesh.hexes[selected].ravel(),
        )
        gmsh.model.addPhysicalGroup(
            3, [1], 901, "DIAGNOSTIC_ONLY_DO_NOT_SOLVE"
        )
    gmsh.model.mesh.reclassifyNodes()
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(path))


def write_visualizations(
    mesh: base.MeshData, case_dir: Path
) -> dict[str, Any]:
    viz = case_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    inlet_msh = viz / "pressure_feed_only.msh"
    _write_diagnostic_msh(mesh, inlet_msh, pressure_only=True)
    theta_edges = np.asarray(
        mesh.metadata["theta_edge_coordinates_rad"], dtype=np.float64
    )
    theta_centres_by_index = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_centres = theta_centres_by_index[
        mesh.logical_cell_indices[:, 0]
    ]
    wrapped = np.abs((theta_centres + math.pi) % (2.0 * math.pi) - math.pi)
    keep = wrapped >= math.radians(30.0)
    cutaway_msh = viz / "cutaway_exact.msh"
    _write_diagnostic_msh(mesh, cutaway_msh, cell_mask=keep)
    cutaway_vtu = viz / "cutaway_exact.vtu"
    meshio.write(
        cutaway_vtu,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[
                (
                    "hexahedron",
                    mesh.hexes[keep].astype(np.int64) - 1,
                )
            ],
            cell_data={
                "gap_layer_index": [
                    mesh.logical_cell_indices[keep, 2].astype(np.int32)
                ],
                "gap_um": [mesh.cell_metrics["gap_um"][keep]],
                "minSICN": [mesh.cell_metrics["minSICN"][keep]],
                "solve_eligible": [np.zeros(np.count_nonzero(keep), dtype=np.uint8)],
                "diagnostic_only": [np.ones(np.count_nonzero(keep), dtype=np.uint8)],
            },
            field_data={
                "DIAGNOSTIC_ONLY_DO_NOT_SOLVE": np.asarray([901, 3])
            },
        ),
        file_format="vtu",
        binary=True,
    )
    inlet_vtu = viz / "pressure_feed_only.vtu"
    inlet_quads = np.asarray(mesh.boundary_quads["pressure_feed"])
    used_tags, compact_connectivity = np.unique(
        inlet_quads.ravel(), return_inverse=True
    )
    meshio.write(
        inlet_vtu,
        meshio.Mesh(
            points=mesh.points_m[used_tags.astype(np.int64) - 1],
            cells=[
                (
                    "quad",
                    compact_connectivity.reshape(inlet_quads.shape),
                )
            ],
            cell_data={
                "patch_id": [
                    np.full(
                        len(mesh.boundary_quads["pressure_feed"]),
                        PHYSICAL_IDS["pressure_feed"],
                        dtype=np.int32,
                    )
                ],
                "diagnostic_only": [
                    np.ones(
                        len(mesh.boundary_quads["pressure_feed"]), dtype=np.uint8
                    )
                ],
            },
            field_data={
                "pressure_feed": np.asarray([PHYSICAL_IDS["pressure_feed"], 2])
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return {
        "solve_eligible": False,
        "coordinates_distorted": False,
        "cutaway": {
            "omitted_wedge_degrees": 60.0,
            "retained_hexes": int(np.count_nonzero(keep)),
            "msh": str(cutaway_msh.relative_to(case_dir)),
            "vtu": str(cutaway_vtu.relative_to(case_dir)),
        },
        "pressure_feed": {
            "msh": str(inlet_msh.relative_to(case_dir)),
            "vtu": str(inlet_vtu.relative_to(case_dir)),
            "quad_count": len(mesh.boundary_quads["pressure_feed"]),
        },
    }


def write_footprint_plot(
    mesh: base.MeshData, inlet: InletSpec, case_dir: Path
) -> str:
    images = case_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    path = images / "pressure_feed_footprint.png"
    quads = np.asarray(mesh.boundary_quads["pressure_feed"])
    vertices = mesh.points_m[quads.astype(np.int64) - 1] / SI_PER_MM
    angle = np.linspace(0.0, 2.0 * math.pi, 512)
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.plot(
        inlet.radius_mm * np.cos(angle),
        inlet.axial_position_mm + inlet.radius_mm * np.sin(angle),
        color="black",
        linewidth=1.8,
        label=f"intended {inlet.diameter_mm:g} mm circle",
    )
    for index, quad in enumerate(vertices):
        closed = np.vstack((quad[:, (0, 2)], quad[0, (0, 2)]))
        axis.fill(
            closed[:, 0],
            closed[:, 1],
            color="#3f88c5",
            alpha=0.45,
            label="selected bore Quad4 faces" if index == 0 else None,
        )
        axis.plot(closed[:, 0], closed[:, 1], color="#174f78", linewidth=0.8)
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title("Pressure-feed footprint on stationary bore")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path.relative_to(case_dir))


def _run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def audit_openfoam(
    mode: str,
    case_dir: Path,
    mesh: base.MeshData,
    records: list[dict[str, Any]],
    published_case_dir: Path,
) -> dict[str, Any]:
    gmsh_to_foam = shutil.which("gmshToFoam")
    check_mesh = shutil.which("checkMesh")
    if mode == "skip" or (mode == "auto" and not (gmsh_to_foam and check_mesh)):
        reason = (
            "disabled by --openfoam skip"
            if mode == "skip"
            else "gmshToFoam/checkMesh unavailable"
        )
        records.append(
            {
                "name": "openfoam.audit",
                "status": "SKIPPED",
                "actual": reason,
                "expected": "optional audited mesh conversion",
                "tolerance": None,
                "mandatory": False,
            }
        )
        return {"status": "SKIPPED", "reason": reason}
    require(
        records,
        "openfoam.executables_available",
        bool(gmsh_to_foam and check_mesh),
        {"gmshToFoam": gmsh_to_foam, "checkMesh": check_mesh},
        "both executables",
    )
    assert gmsh_to_foam is not None and check_mesh is not None
    foam_case = case_dir / "openfoam_case"
    (foam_case / "constant").mkdir(parents=True)
    (foam_case / "system").mkdir(parents=True)
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
    conversion = _run_command(
        [
            gmsh_to_foam,
            "-case",
            str(foam_case),
            str((case_dir / "structured_hex_openfoam.msh").resolve()),
        ]
    )
    conversion_text = conversion["stdout"] + "\n" + conversion["stderr"]
    forbidden = (
        "inverting hex",
        "undefined faces",
        "could not match gmsh face",
        "foam fatal",
        "foam exiting",
    )
    require(
        records,
        "openfoam.gmshToFoam",
        conversion["returncode"] == 0
        and not any(token in conversion_text.lower() for token in forbidden),
        {
            "returncode": conversion["returncode"],
            "forbidden_messages": [
                token for token in forbidden if token in conversion_text.lower()
            ],
        },
        {"returncode": 0, "forbidden_messages": []},
    )
    check = _run_command(
        [check_mesh, "-case", str(foam_case), "-allTopology", "-allGeometry"]
    )
    check_text = check["stdout"] + "\n" + check["stderr"]
    require(
        records,
        "openfoam.checkMesh",
        check["returncode"] == 0
        and "Mesh OK." in check_text
        and "Failed " not in check_text,
        {"returncode": check["returncode"], "mesh_ok": "Mesh OK." in check_text},
        {"returncode": 0, "mesh_ok": True},
    )
    cell_match = re.search(r"(?m)^\s*cells:\s*(\d+)\s*$", check_text)
    region_match = re.search(r"Number of regions:\s*(\d+)", check_text)
    bbox_match = re.search(
        r"Overall domain bounding box\s*\(([^)]+)\)\s*\(([^)]+)\)",
        check_text,
    )
    volume_match = re.search(
        r"Total volume\s*=\s*([+\-0-9.eE]+)", check_text
    )
    parsed_cells = int(cell_match.group(1)) if cell_match else None
    parsed_regions = int(region_match.group(1)) if region_match else None
    parsed_bbox = (
        tuple(
            float(value)
            for group in bbox_match.groups()
            for value in group.split()
        )
        if bbox_match
        else None
    )
    parsed_volume = float(volume_match.group(1)) if volume_match else None
    expected_bbox = (
        *mesh.points_m.min(axis=0).tolist(),
        *mesh.points_m.max(axis=0).tolist(),
    )
    expected_volume = float(mesh.cell_metrics["cell_volume_m3"].sum())
    require(
        records,
        "openfoam.cell_count",
        parsed_cells == len(mesh.hexes),
        parsed_cells,
        len(mesh.hexes),
    )
    require(
        records,
        "openfoam.connected_regions",
        parsed_regions == 1,
        parsed_regions,
        1,
    )
    require(
        records,
        "openfoam.SI_bounding_box",
        parsed_bbox is not None
        and max(abs(a - b) for a, b in zip(parsed_bbox, expected_bbox))
        <= 1.0e-9,
        parsed_bbox,
        expected_bbox,
        1.0e-9,
    )
    require(
        records,
        "openfoam.total_volume",
        parsed_volume is not None
        and relative_error(parsed_volume, expected_volume) <= 1.0e-6,
        parsed_volume,
        expected_volume,
        1.0e-6,
    )
    poly_mesh = foam_case / "constant" / "polyMesh"
    required_files = ("points", "faces", "owner", "neighbour", "boundary")
    missing = [
        name
        for name in required_files
        if not (poly_mesh / name).is_file()
        or (poly_mesh / name).stat().st_size == 0
    ]
    require(records, "openfoam.polyMesh_files", not missing, missing, [])
    boundary_path = poly_mesh / "boundary"
    patches = base._openfoam_boundary_patches(boundary_path)
    require(
        records,
        "openfoam.patch_names",
        set(patches) == set(SURFACE_ENTITIES),
        sorted(patches),
        sorted(SURFACE_ENTITIES),
    )
    require(
        records,
        "openfoam.no_defaultFaces_or_seam",
        not any(
            token in name.lower()
            for name in patches
            for token in ("default", "seam", "cyclic", "symmetry")
        ),
        sorted(patches),
        "no default/seam/cyclic/symmetry patch",
    )
    return {
        "status": "PASS",
        "conversion": conversion,
        "checkMesh": check,
        "patches": patches,
        "parsed": {
            "cells": parsed_cells,
            "regions": parsed_regions,
            "bbox_m": parsed_bbox,
            "volume_m3": parsed_volume,
        },
        "published_case": str(published_case_dir / "openfoam_case"),
    }


def _file_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "mesh_report.json"}
    }


def generate_gap_case(
    params: base.BearingParams,
    inlet: InletSpec,
    inputs: SurfaceInletInputs,
    n_gap: int,
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    mesh, original_outer, uniform_volume = build_surface_inlet_mesh(
        params,
        inlet,
        inputs.n_theta,
        inputs.n_axial,
        n_gap,
        inputs.gap_inflation_ratio,
        inputs.inlet_cluster_strength,
    )
    analytic = validate_surface_inlet_mesh(
        mesh,
        original_outer,
        uniform_volume,
        params,
        inlet,
        inputs.max_projected_area_relative_error,
        records,
        inputs.max_inlet_rim_error_mm,
    )
    initialized = False
    logger_started = False
    gmsh_lines: list[str] = []
    try:
        gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
        initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()
        logger_started = True
        discrete = add_discrete_model(
            mesh, records, f"structured_surface_inlet_nGap_{n_gap:02d}"
        )
        physical = validate_physical_groups(records, "gmsh_model")
        mesh = base.add_gmsh_quality_metrics(mesh, records)
        views = write_quality_view(
            mesh, case_dir / "views", inputs.quality_view
        )
        msh41, msh22 = write_gmsh_files(case_dir)
        round_trips = {
            "gmsh_4_1_binary": validate_gmsh_round_trip(
                msh41, mesh, records, "round_trip.msh41"
            ),
            "gmsh_2_2_ascii": validate_gmsh_round_trip(
                msh22, mesh, records, "round_trip.msh22"
            ),
        }
        volume_vtu, boundary_vtu = write_vtu_files(mesh, case_dir)
        vtu_round_trip = validate_vtu_round_trip(
            mesh, volume_vtu, boundary_vtu, records
        )
        npz_path = case_dir / "mesh_arrays.npz"
        base.write_mesh_arrays(mesh, npz_path)
        npz_round_trip = base.validate_mesh_arrays_round_trip(
            mesh, npz_path, records
        )
        visualization = write_visualizations(mesh, case_dir)
    finally:
        if logger_started:
            gmsh_lines = [str(line) for line in gmsh.logger.get()]
            gmsh.logger.stop()
        if initialized:
            gmsh.finalize()

    image = write_footprint_plot(mesh, inlet, case_dir)
    published_case = inputs.outdir / f"nGap_{n_gap:02d}"
    openfoam = audit_openfoam(
        inputs.openfoam, case_dir, mesh, records, published_case
    )
    (case_dir / "gmsh_surface_inlet.log").write_text(
        "\n".join(gmsh_lines) + "\n", encoding="utf-8"
    )
    physical_groups = {
        "coordinate_unit": "m",
        "volume": {
            "fluid": {
                "physical_id": PHYSICAL_IDS["fluid"],
                "entity_tag": VOLUME_ENTITY,
            }
        },
        "boundaries": {
            name: {
                "physical_id": PHYSICAL_IDS[name],
                "entity_tag": SURFACE_ENTITIES[name],
                "quad_count": len(mesh.boundary_quads[name]),
            }
            for name in SURFACE_ENTITIES
        },
        "moving_wall": "journal_wall",
        "stationary_walls": ["stationary_wall"],
        "pressure_boundary": "pressure_feed",
        "axial_boundaries": ["axial_end_z0", "axial_end_zL"],
        "contains_feed_tube": False,
    }
    (case_dir / "physical_groups.json").write_text(
        json.dumps(physical_groups, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quality = {
        name: base.metric_statistics(mesh.cell_metrics[name])
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
        "gmsh_exact": f"uv run gmsh {published_case / 'structured_hex.msh'}",
        "gmsh_cutaway": (
            "uv run gmsh -setnumber Mesh.VolumeFaces 1 "
            f"-setnumber Mesh.VolumeEdges 1 {published_case / 'viz' / 'cutaway_exact.msh'}"
        ),
        "gmsh_pressure_patch": (
            "uv run gmsh -setnumber Mesh.SurfaceFaces 1 "
            f"-setnumber Mesh.SurfaceEdges 1 {published_case / 'viz' / 'pressure_feed_only.msh'}"
        ),
        "paraview_volume": (
            f"QT_QPA_PLATFORM=xcb paraview {published_case / 'volume_hex.vtu'}"
        ),
        "paraview_cutaway": (
            f"QT_QPA_PLATFORM=xcb paraview {published_case / 'viz' / 'cutaway_exact.vtu'}"
        ),
        "paraview_pressure_patch": (
            f"QT_QPA_PLATFORM=xcb paraview {published_case / 'viz' / 'pressure_feed_only.vtu'}"
        ),
    }
    manifest = {
        "schema_version": 1,
        "overall": "PASS",
        "solve_eligible": True,
        "geometry": "full-360 annular film with surface pressure-inlet patch",
        "reduced_physics": (
            "the feed passage is excluded; pressure_feed applies directly on the bore"
        ),
        "coordinate_unit": "m",
        "source_parameter_unit": "mm",
        "scale_to_m_applied_exactly_once": SI_PER_MM,
        "contains_only_hex8_volume_cells": True,
        "gmsh_generated_volume_mesh": False,
        "contains_feed_tube": False,
        "contains_pressure_feed_patch": True,
        "pressure_patch_representation": analytic["pressure_patch"]["representation"],
        "canonical_arrays": "mesh_arrays.npz",
        "counts": analytic["counts"],
        "physical_groups": physical,
        "commands": commands,
        "params": asdict(params) | {"source": str(params.source)},
        "inlet": asdict(inlet),
        "inlet_clustering": analytic["inlet_clustering"],
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
            "gap_inflation_ratio": inputs.gap_inflation_ratio,
            "inlet_cluster_strength": inputs.inlet_cluster_strength,
            "max_projected_area_relative_error": inputs.max_projected_area_relative_error,
            "max_inlet_rim_error_mm": inputs.max_inlet_rim_error_mm,
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
            "log": "gmsh_surface_inlet.log",
        },
        "vtu_round_trip": vtu_round_trip,
        "npz_round_trip": npz_round_trip,
        "visualization": visualization,
        "footprint_plot": image,
        "openfoam": openfoam,
        "commands": commands,
        "error": None,
    }
    report["files"] = _file_inventory(case_dir)
    (case_dir / "mesh_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "n_gap": n_gap,
        "point_count": len(mesh.points_m),
        "cell_count": len(mesh.hexes),
        "pressure_quad_count": len(mesh.boundary_quads["pressure_feed"]),
        "projected_inlet_area_error": analytic["pressure_patch"][
            "projected_area_relative_error"
        ],
        "faceted_volume_m3": analytic["mesh_volume_m3"],
        "faceted_volume_relative_error": analytic[
            "faceted_volume_relative_error"
        ],
        "quality": quality,
        "openfoam": {
            key: value
            for key, value in openfoam.items()
            if key not in {"conversion", "checkMesh"}
        },
        "commands": commands,
        "validation_count": len(records),
    }


def write_convergence_report(
    stage: Path,
    params: base.BearingParams,
    inputs: SurfaceInletInputs,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    volumes = np.asarray([case["faceted_volume_m3"] for case in cases])
    pressure_counts = {case["pressure_quad_count"] for case in cases}
    volume_relative_range = (
        float(np.ptp(volumes) / abs(volumes.mean())) if len(volumes) > 1 else 0.0
    )
    if volume_relative_range > 1.0e-12:
        raise SurfaceInletError(
            "faceted volume changed with through-gap subdivision: "
            f"{volume_relative_range:.3e}"
        )
    if len(pressure_counts) != 1:
        raise SurfaceInletError(
            f"pressure patch changed with through-gap subdivision: {pressure_counts}"
        )
    rows = [
        {
            "n_theta": inputs.n_theta,
            "n_axial": inputs.n_axial,
            "n_gap": case["n_gap"],
            "hex_count": case["cell_count"],
            "pressure_quad_count": case["pressure_quad_count"],
            "faceted_volume_m3": case["faceted_volume_m3"],
            "exact_annulus_volume_m3": params.exact_volume_m3,
            "faceted_volume_relative_error": case[
                "faceted_volume_relative_error"
            ],
            "projected_inlet_area_relative_error": case[
                "projected_inlet_area_error"
            ],
        }
        for case in cases
    ]
    import csv

    with (stage / "convergence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "overall": "PASS",
        "coordinate_unit": "m",
        "gap_level_volume_relative_range": volume_relative_range,
        "pressure_patch_invariant_across_gap_levels": True,
        "cases": rows,
    }
    (stage / "convergence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _publish_failure(outdir: Path, report: dict[str, Any]) -> None:
    stage = make_staging_directory(outdir)
    try:
        (stage / "failure_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_generation(
            stage,
            outdir,
            stage="meshing",
            operation="surface-inlet",
            status=str(report["overall"]),
            resolved_inputs=report.get("inputs", {}),
            input_units={"geometry": "mm", "mesh": "m"},
            producer_files=(Path(__file__),),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run_surface_inlet(inputs: SurfaceInletInputs) -> dict[str, Any]:
    inputs = replace(
        inputs, params=inputs.params.resolve(), outdir=inputs.outdir.resolve()
    )
    base_report = {
        "inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(inputs).items()
        },
        "volume_mesh_generated_by_gmsh": False,
    }
    stage: Path | None = None
    try:
        if not inputs.params.is_file():
            raise SurfaceInletError(f"params file not found: {inputs.params}")
        params = base.load_params(inputs.params)
        inlet = load_inlet_spec(inputs.params)
        validate_inputs(inputs, params, inlet)
        if inputs.openfoam == "required" and not (
            shutil.which("gmshToFoam") and shutil.which("checkMesh")
        ):
            raise SurfaceInletError(
                "--openfoam required but gmshToFoam/checkMesh are unavailable"
            )
        stage = make_staging_directory(inputs.outdir)
        cases = [
            generate_gap_case(
                params,
                inlet,
                inputs,
                n_gap,
                stage / f"nGap_{n_gap:02d}",
            )
            for n_gap in inputs.gap_levels
        ]
        convergence = write_convergence_report(
            stage, params, inputs, cases
        )
        report = {
            **base_report,
            "overall": "PASS",
            "params": asdict(params) | {"source": str(params.source)},
            "inlet": asdict(inlet),
            "cases": cases,
            "convergence": convergence,
            "error": None,
        }
        (stage / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_generation(
            stage,
            inputs.outdir,
            stage="meshing",
            operation="surface-inlet",
            status="PASS",
            resolved_inputs=base_report["inputs"],
            input_units={"geometry": "mm", "mesh": "m"},
            producer_files=(Path(__file__),),
            upstream_artifacts=(inputs.params,),
            tool_versions={
                "gmsh": gmsh.__version__,
                "meshio": meshio.__version__,
                "numpy": np.__version__,
            },
        )
        stage = None
    except BaseException as error:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if not isinstance(error, Exception):
            raise
        failure = {
            **base_report,
            "overall": "FAIL",
            "solve_eligible_outputs_published": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _publish_failure(inputs.outdir, failure)
        raise SurfaceInletRunError(str(error), failure) from error
    if inputs.gui:
        try:
            open_gui(inputs, inputs.outdir)
            gui_status = {"status": "PASS", "error": None}
        except Exception as error:
            gui_status = {
                "status": "WARNING",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            print(
                f"WARNING: validated mesh published, but GUI launch failed: {error}",
                file=sys.stderr,
            )
        report["gui"] = gui_status
        try:
            (inputs.outdir / "gui_status.json").write_text(
                json.dumps(gui_status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            print(
                f"WARNING: could not persist GUI status: {error}",
                file=sys.stderr,
            )
    return report


def open_gui(inputs: SurfaceInletInputs, outdir: Path) -> None:
    case_dir = outdir / f"nGap_{inputs.preview_ngap:02d}"
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    try:
        if inputs.gui_mode == "exact":
            gmsh.open(str(case_dir / "structured_hex.msh"))
        elif inputs.gui_mode == "inlet":
            gmsh.open(str(case_dir / "viz" / "pressure_feed_only.msh"))
        else:
            gmsh.open(str(case_dir / "viz" / "cutaway_exact.msh"))
            if inputs.gui_mode == "quality":
                gmsh.merge(
                    str(case_dir / "views" / f"{inputs.quality_view}.pos")
                )
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


def print_report(report: dict[str, Any]) -> None:
    print("\nStructured full-360 Hex8 film with surface pressure inlet")
    print(
        f"{'case':<10} {'points':>10} {'Hex8':>10} {'inletQ':>8} "
        f"{'area.err':>10} {'minSICN':>11} {'maxNonOrth':>12} {'status':>8}"
    )
    print("-" * 90)
    for case in report.get("cases", []):
        print(
            f"nGap_{case['n_gap']:02d} {case['point_count']:10d} "
            f"{case['cell_count']:10d} {case['pressure_quad_count']:8d} "
            f"{case['projected_inlet_area_error']:10.3%} "
            f"{case['quality']['minSICN']['min']:11.4g} "
            f"{case['quality']['max_nonorthogonality_deg']['max']:12.4g} "
            f"{'PASS':>8}"
        )
    print("Volume cells: Hex8 only")
    print("Feed tube cells: 0")
    print(f"OVERALL: {report.get('overall', 'FAIL')}")
    if report.get("overall") == "PASS":
        preview = next(
            case
            for case in report["cases"]
            if case["n_gap"] == report["inputs"]["preview_ngap"]
        )
        print("\nOpen commands")
        for name, command in preview["commands"].items():
            print(f"{name:24s} {command}")


def parse_args(argv: Sequence[str] | None = None) -> SurfaceInletInputs:
    parser = argparse.ArgumentParser(
        description=(
            "Build an analytic full-360 Hex8 bearing-film mesh with a "
            "mesh-aligned pressure patch on the stationary bore."
        )
    )
    parser.add_argument("--params", type=Path, default=SurfaceInletInputs.params)
    parser.add_argument("--outdir", type=Path, default=SurfaceInletInputs.outdir)
    parser.add_argument("--n-theta", type=int, default=256)
    parser.add_argument("--n-axial", type=int, default=96)
    parser.add_argument(
        "--gap-levels", type=int, nargs="+", default=[4, 8, 12]
    )
    parser.add_argument("--preview-ngap", type=int, default=8)
    parser.add_argument("--gap-inflation-ratio", type=float, default=5.0)
    parser.add_argument(
        "--inlet-cluster-strength", type=float, default=0.82
    )
    parser.add_argument(
        "--max-projected-area-relative-error", type=float, default=0.01
    )
    parser.add_argument("--max-inlet-rim-error-mm", type=float, default=0.16)
    parser.add_argument(
        "--gui", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--gui-mode",
        choices=("exact", "cutaway", "inlet", "quality"),
        default="cutaway",
    )
    parser.add_argument(
        "--quality-view", choices=base.VIEW_NAMES, default="minSICN"
    )
    parser.add_argument(
        "--openfoam", choices=("auto", "required", "skip"), default="auto"
    )
    args = parser.parse_args(argv)
    return SurfaceInletInputs(
        params=args.params,
        outdir=args.outdir,
        n_theta=args.n_theta,
        n_axial=args.n_axial,
        gap_levels=tuple(args.gap_levels),
        preview_ngap=args.preview_ngap,
        gap_inflation_ratio=args.gap_inflation_ratio,
        inlet_cluster_strength=args.inlet_cluster_strength,
        max_projected_area_relative_error=args.max_projected_area_relative_error,
        max_inlet_rim_error_mm=args.max_inlet_rim_error_mm,
        gui=args.gui,
        gui_mode=args.gui_mode,
        quality_view=args.quality_view,
        openfoam=args.openfoam,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_surface_inlet(parse_args(argv))
    except SurfaceInletRunError as error:
        print_report(error.report)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
