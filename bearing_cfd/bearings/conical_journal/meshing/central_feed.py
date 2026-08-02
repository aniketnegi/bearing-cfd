#!/usr/bin/env python3
"""Build a conformal layered Prism6 mesh of the ported bearing-film volume."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import stat
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Sequence

import gmsh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

from bearing_cfd.artifacts import make_staging_directory, publish_generation

from bearing_cfd.bearings.conical_journal.meshing.brep_preflight import (
    configure_occ_options,
    import_brep as gmsh_import_brep,
    inventory_surfaces,
    relative_error,
    require,
    sha256_file,
)
from bearing_cfd.bearings.conical_journal.meshing.no_port import (
    _coordinate_order,
    _openfoam_boundary_patches,
    _run_command,
)
from bearing_cfd.bearings.conical_journal.meshing.gap_grading import symmetric_gap_coordinates


SI_PER_MM = 1.0e-3
NODE_RESIDUAL_MM = 1.0e-10
BREP_NODE_TOL_MM = 1.0e-6
BREP_VOLUME_REL_TOL = 5.0e-4
BOUNDARY_VOLUME_REL_TOL = 1.0e-9
INLET_POLYGON_REL_TOL = 1.0e-10
PHYSICAL_IDS = {
    "journal_wall": 101,
    "bushing_bore_wall": 102,
    "axial_end_z0": 103,
    "axial_end_zL": 104,
    "feed_tube_wall": 105,
    "pressure_feed": 106,
    "fluid": 201,
}
SURFACE_ENTITIES = {
    "journal_wall": 21,
    "bushing_bore_wall": 22,
    "axial_end_z0": 23,
    "axial_end_zL": 24,
    "feed_tube_wall": 25,
    "pressure_feed": 26,
}
VOLUME_ENTITY = 31
PRISM_TRI_FACES = np.asarray([[0, 2, 1], [3, 4, 5]], dtype=np.uint8)
PRISM_QUAD_FACES = np.asarray(
    [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]], dtype=np.uint8
)
PRISM_EDGES = np.asarray(
    [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]],
    dtype=np.uint8,
)
CELL_FIELD_NAMES = (
    "region_id",
    "axial_zone_id",
    "gap_layer_index",
    "gap_um",
    "theta_deg",
    "minSICN",
    "minDetJac",
    "volume_m3",
    "aspect_ratio",
    "solve_eligible",
    "distorted_geometry",
)


class PortedMeshError(RuntimeError):
    """A mandatory ported-mesh construction or validation failure."""


class PortedRunError(PortedMeshError):
    """A failed run with a serializable report."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class PortedInputs:
    params: Path = Path("out/conical_journal/geometry/default/params.json")
    brep: Path = Path("out/conical_journal/geometry/default/film_unsplit.brep")
    preflight: Path = Path(
        "out/conical_journal/meshing/brep-preflight/preflight_report.json"
    )
    outdir: Path = Path("out/conical_journal/meshing/central-feed")
    n_theta: int = 256
    n_axial: int = 96
    gap_levels: tuple[int, ...] = (4, 8, 12)
    preview_ngap: int = 8
    rim_segments: int = 128
    gap_inflation_ratio: float = 5.0
    tube_layers: int = 48
    tube_grading: float = 1.0
    openfoam: Literal["auto", "required", "skip"] = "auto"
    gui: bool = False
    gui_mode: Literal["full", "cutaway", "mouth", "quality"] = "cutaway"


@dataclass(frozen=True)
class PortedParams:
    source: Path
    source_sha256: str
    brep_sha256: str
    length_mm: float
    mean_radius_mm: float
    semicone_angle_deg: float
    cone_slope: float
    radial_clearance_mm: float
    eccentricity_mm: float
    eccentricity_ratio: float
    ex_mm: float
    ey_mm: float
    hole_axial_pos_mm: float
    hole_radius_mm: float
    y_feed_end_mm: float
    z1_mm: float
    z2_mm: float
    native_volume_mm3: float
    native_bbox_mm: tuple[float, float, float, float, float, float]
    inlet_area_mm2: float
    inlet_centre_mm: tuple[float, float, float]

    def journal_radius_mm(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        return self.mean_radius_mm + (self.length_mm / 2.0 - z_mm) * self.cone_slope

    def bore_radius_mm(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        return self.journal_radius_mm(z_mm) + self.radial_clearance_mm

    def rho_j_mm(
        self, theta: np.ndarray | float, z_mm: np.ndarray | float
    ) -> np.ndarray | float:
        theta_array = np.asarray(theta)
        q = self.ex_mm * np.sin(theta_array) - self.ey_mm * np.cos(theta_array)
        radius = np.asarray(self.journal_radius_mm(z_mm))
        radicand = radius**2 - self.ex_mm**2 - self.ey_mm**2 + q**2
        if np.any(radicand <= 0.0):
            raise PortedMeshError(
                f"journal-ray radicand must be positive; minimum={float(np.min(radicand))}"
            )
        result = q + np.sqrt(radicand)
        return float(result) if result.ndim == 0 else result

    @property
    def minimum_gap_mm(self) -> float:
        return self.radial_clearance_mm - self.eccentricity_mm

    @property
    def maximum_gap_mm(self) -> float:
        return self.radial_clearance_mm + self.eccentricity_mm


@dataclass(frozen=True)
class MasterMesh:
    points_uz_mm: np.ndarray
    triangles: np.ndarray
    disk_triangle_mask: np.ndarray
    axial_zone_id: np.ndarray
    rim_nodes: np.ndarray
    rim_edges: np.ndarray
    seam_edges: np.ndarray
    z1_edges: np.ndarray
    z2_edges: np.ndarray
    z0_edges: np.ndarray
    zL_edges: np.ndarray
    centre_node: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for array in (
            self.points_uz_mm,
            self.triangles,
            self.disk_triangle_mask,
            self.axial_zone_id,
            self.rim_nodes,
            self.rim_edges,
            self.seam_edges,
            self.z1_edges,
            self.z2_edges,
            self.z0_edges,
            self.zL_edges,
        ):
            array.setflags(write=False)


@dataclass(frozen=True)
class PrismMesh:
    points_m: np.ndarray
    prisms: np.ndarray
    boundary_triangles: dict[str, np.ndarray]
    boundary_quads: dict[str, np.ndarray]
    mouth_triangles: np.ndarray
    cell_tags: np.ndarray
    node_tags: np.ndarray
    cell_fields: dict[str, np.ndarray]
    cell_centres_m: np.ndarray
    film_cell_count: int
    feed_cell_count: int
    master_triangle_count: int
    disk_triangle_count: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "boundary_triangles", MappingProxyType(dict(self.boundary_triangles))
        )
        object.__setattr__(
            self, "boundary_quads", MappingProxyType(dict(self.boundary_quads))
        )
        object.__setattr__(self, "cell_fields", MappingProxyType(dict(self.cell_fields)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for array in (
            self.points_m,
            self.prisms,
            self.mouth_triangles,
            self.cell_tags,
            self.node_tags,
            self.cell_centres_m,
            *self.boundary_triangles.values(),
            *self.boundary_quads.values(),
            *self.cell_fields.values(),
        ):
            array.setflags(write=False)


def _json_record(records: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("name") == name]
    if len(matches) != 1:
        raise PortedMeshError(f"expected one preflight record {name!r}, found {len(matches)}")
    return matches[0]


def load_contract(
    inputs: PortedInputs, records: list[dict[str, Any]]
) -> tuple[PortedParams, dict[str, Any], dict[str, Any]]:
    for label, path in (
        ("params", inputs.params),
        ("brep", inputs.brep),
        ("preflight", inputs.preflight),
    ):
        require(records, f"input.{label}.exists", path.is_file(), str(path), "readable file")
    try:
        raw = json.loads(inputs.params.read_text(encoding="utf-8"))
        preflight = json.loads(inputs.preflight.read_text(encoding="utf-8"))
        resolved = raw["resolved_parameters"]
        measurements = raw["measurements"]
        hole_radius = (
            float(resolved["hole_radius"])
            if "hole_radius" in resolved
            else float(resolved["hole_diameter"]) / 2.0
        )
        bbox = measurements["final_bounding_box_mm"]
        inlet = measurements["inlet_face"]
        params = PortedParams(
            source=inputs.params.resolve(),
            source_sha256=sha256_file(inputs.params),
            brep_sha256=str(raw["sha256"]["film_unsplit.brep"]),
            length_mm=float(resolved["length"]),
            mean_radius_mm=float(resolved["mean_radius"]),
            semicone_angle_deg=float(resolved["semicone_angle_deg"]),
            cone_slope=float(resolved["cone_slope"]),
            radial_clearance_mm=float(resolved["radial_clearance"]),
            eccentricity_mm=float(resolved["eccentricity"]),
            eccentricity_ratio=float(resolved["eccentricity_ratio"]),
            ex_mm=float(resolved["ex"]),
            ey_mm=float(resolved["ey"]),
            hole_axial_pos_mm=float(resolved["hole_axial_pos"]),
            hole_radius_mm=hole_radius,
            y_feed_end_mm=float(resolved["y_feed_end"]),
            z1_mm=float(resolved["z1"]),
            z2_mm=float(resolved["z2"]),
            native_volume_mm3=float(measurements["volumes_mm3"]["total"]),
            native_bbox_mm=tuple(float(value) for value in (*bbox["min"], *bbox["max"])),
            inlet_area_mm2=float(inlet["area_mm2"]),
            inlet_centre_mm=tuple(float(value) for value in inlet["centre_mm"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PortedMeshError(f"invalid ported-mesh input contract: {error}") from error

    require(
        records,
        "input.preflight_overall",
        preflight.get("overall") == "PASS",
        preflight.get("overall"),
        "PASS",
    )
    require(
        records,
        "input.preflight_brep_units",
        preflight.get("coordinate_units", {}).get("input_brep") == "mm",
        preflight.get("coordinate_units", {}).get("input_brep"),
        "mm",
    )
    current_brep_hash = sha256_file(inputs.brep)
    hash_record = _json_record(preflight.get("validation_records", []), "input.unsplit.sha256")
    require(
        records,
        "input.native_brep_sha256",
        current_brep_hash == params.brep_sha256
        and hash_record.get("status") == "PASS"
        and hash_record.get("actual") == current_brep_hash
        and hash_record.get("expected") == current_brep_hash,
        {
            "current": current_brep_hash,
            "params": params.brep_sha256,
            "preflight": hash_record,
        },
        "one identical validated native BREP hash",
    )
    reference = preflight.get("cad_reference", {})
    diagnostics = preflight.get("diagnostics", {}).get("unsplit", {})
    require(
        records,
        "input.preflight_native_volume",
        relative_error(float(diagnostics.get("mass", math.nan)), params.native_volume_mm3)
        <= 1.0e-10,
        diagnostics.get("mass"),
        params.native_volume_mm3,
        1.0e-10,
    )
    require(
        records,
        "input.preflight_feed_coordinate",
        abs(float(reference.get("y_feed_end", math.nan)) - params.y_feed_end_mm)
        <= NODE_RESIDUAL_MM,
        reference.get("y_feed_end"),
        params.y_feed_end_mm,
        NODE_RESIDUAL_MM,
    )
    preflight_bbox = tuple(float(value) for value in diagnostics.get("bbox", ()))
    expected_preflight_bbox = tuple(float(value) for value in reference.get("bounding_box", ()))
    require(
        records,
        "input.preflight_native_bbox",
        len(preflight_bbox) == 6
        and len(expected_preflight_bbox) == 6
        and max(abs(a - b) for a, b in zip(preflight_bbox, expected_preflight_bbox))
        <= 1.0e-6,
        preflight_bbox,
        expected_preflight_bbox,
        1.0e-6,
    )
    return params, raw, preflight


def validate_inputs(inputs: PortedInputs, params: PortedParams) -> None:
    if inputs.n_theta < 8 or inputs.n_axial < 3:
        raise PortedMeshError("n-theta>=8 and n-axial>=3 are required")
    if not inputs.gap_levels or any(level < 1 for level in inputs.gap_levels):
        raise PortedMeshError("gap-levels must contain positive integers")
    if len(set(inputs.gap_levels)) != len(inputs.gap_levels):
        raise PortedMeshError("gap-levels must be unique")
    if inputs.preview_ngap not in inputs.gap_levels:
        raise PortedMeshError("preview-ngap must be one of gap-levels")
    if inputs.rim_segments < 32 or inputs.rim_segments % 4:
        raise PortedMeshError("rim-segments must be >=32 and divisible by four")
    if (
        not math.isfinite(inputs.gap_inflation_ratio)
        or inputs.gap_inflation_ratio < 1.0
    ):
        raise PortedMeshError("gap-inflation-ratio must be finite and >=1")
    if inputs.tube_layers < 1:
        raise PortedMeshError("tube-layers must be positive")
    if not math.isfinite(inputs.tube_grading) or inputs.tube_grading <= 0.0:
        raise PortedMeshError("tube-grading must be finite and positive")
    expected_slope = math.tan(math.radians(params.semicone_angle_deg))
    if abs(params.cone_slope - expected_slope) > 1.0e-14:
        raise PortedMeshError("cone_slope is inconsistent with semicone_angle_deg")
    if abs(params.eccentricity_mm - params.eccentricity_ratio * params.radial_clearance_mm) > 1.0e-14:
        raise PortedMeshError("eccentricity is inconsistent with epsilon*c")
    if not (0.0 < params.eccentricity_ratio < 1.0):
        raise PortedMeshError(
            "concentric geometry is unsupported: eccentricity_ratio must satisfy 0<epsilon<1"
        )
    if not (0.0 < params.z1_mm < params.z2_mm < params.length_mm):
        raise PortedMeshError("z1 and z2 must lie strictly inside the bearing")
    if not (params.z1_mm < params.hole_axial_pos_mm - params.hole_radius_mm):
        raise PortedMeshError("feed disk must lie strictly inside the middle band")
    if not (params.hole_axial_pos_mm + params.hole_radius_mm < params.z2_mm):
        raise PortedMeshError("feed disk must lie strictly inside the middle band")
    if params.minimum_gap_mm <= 0.0:
        raise PortedMeshError("minimum radial clearance must be positive")


def validate_native_brep(
    inputs: PortedInputs,
    params: PortedParams,
    preflight: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    from build123d import import_brep

    shape = import_brep(inputs.brep)
    solids = shape.solids()
    require(records, "brep.one_connected_solid", len(solids) == 1, len(solids), 1)
    require(
        records,
        "brep.valid_manifold",
        len(solids) == 1 and bool(solids[0].is_valid) and bool(solids[0].is_manifold),
        {
            "valid": bool(solids[0].is_valid) if solids else False,
            "manifold": bool(solids[0].is_manifold) if solids else False,
        },
        {"valid": True, "manifold": True},
    )
    volume = float(shape.volume)
    require(
        records,
        "brep.volume_vs_params",
        relative_error(volume, params.native_volume_mm3) <= 1.0e-10,
        volume,
        params.native_volume_mm3,
        1.0e-10,
    )
    bounds = shape.bounding_box()
    bbox = (
        float(bounds.min.X),
        float(bounds.min.Y),
        float(bounds.min.Z),
        float(bounds.max.X),
        float(bounds.max.Y),
        float(bounds.max.Z),
    )
    preflight_bbox = tuple(
        float(value) for value in preflight["diagnostics"]["unsplit"]["bbox"]
    )
    preflight_bbox_error = max(abs(a - b) for a, b in zip(bbox, preflight_bbox))
    params_bbox_error = max(abs(a - b) for a, b in zip(bbox, params.native_bbox_mm))
    require(
        records,
        "brep.bbox_vs_preflight",
        preflight_bbox_error <= 1.0e-6 and params_bbox_error <= 1.0e-6,
        {
            "bbox_mm": bbox,
            "maximum_preflight_error_mm": preflight_bbox_error,
            "maximum_params_error_mm": params_bbox_error,
        },
        preflight_bbox,
        1.0e-6,
    )
    return {
        "sha256": sha256_file(inputs.brep),
        "solid_count": len(solids),
        "valid": bool(solids[0].is_valid),
        "manifold": bool(solids[0].is_manifold),
        "volume_mm3": volume,
        "bbox_mm": bbox,
    }


def rim_coordinates(params: PortedParams, count: int) -> dict[str, np.ndarray]:
    alpha = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
    x = params.hole_radius_mm * np.cos(alpha)
    z = params.hole_axial_pos_mm + params.hole_radius_mm * np.sin(alpha)
    rb = np.asarray(params.bore_radius_mm(z), dtype=np.float64)
    y = np.sqrt(rb**2 - x**2)
    theta = np.mod(np.arctan2(x, -y), 2.0 * math.pi)
    u = params.mean_radius_mm * theta
    return {"alpha": alpha, "x": x, "y": y, "z": z, "theta": theta, "u": u}


def _partition_counts(total: int, spans: Sequence[float]) -> tuple[int, ...]:
    if total < len(spans):
        raise PortedMeshError("n-axial is too small for all axial partitions")
    raw = np.asarray(spans, dtype=np.float64) / sum(spans) * total
    counts = np.floor(raw).astype(int)
    counts[counts < 1] = 1
    while int(counts.sum()) < total:
        index = int(np.argmax(raw - counts))
        counts[index] += 1
    while int(counts.sum()) > total:
        candidates = np.where(counts > 1)[0]
        if not len(candidates):
            raise PortedMeshError("cannot allocate positive axial partition counts")
        index = int(candidates[np.argmin(raw[candidates] - counts[candidates])])
        counts[index] -= 1
    return tuple(int(value) for value in counts)


def _edge_census(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    canonical = np.sort(edges, axis=1)
    unique, counts = np.unique(canonical, axis=0, return_counts=True)
    return unique, counts


def _select_edges(
    unique_edges: np.ndarray,
    points: np.ndarray,
    *,
    u: float | None = None,
    z: float | None = None,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    coordinates = points[unique_edges]
    mask = np.ones(len(unique_edges), dtype=bool)
    if u is not None:
        mask &= np.all(np.abs(coordinates[:, :, 0] - u) <= tolerance, axis=1)
    if z is not None:
        mask &= np.all(np.abs(coordinates[:, :, 1] - z) <= tolerance, axis=1)
    return unique_edges[mask]


def build_master_mesh(
    params: PortedParams,
    inputs: PortedInputs,
    output_path: Path,
    records: list[dict[str, Any]],
) -> MasterMesh:
    width = 2.0 * math.pi * params.mean_radius_mm
    rim = rim_coordinates(params, inputs.rim_segments)
    gmsh.model.add("ported_master_surface")
    geo = gmsh.model.geo
    zcuts = (0.0, params.z1_mm, params.z2_mm, params.length_mm)
    left = [geo.addPoint(0.0, z, 0.0) for z in zcuts]
    right = [geo.addPoint(width, z, 0.0) for z in zcuts]
    horizontal = [geo.addLine(left[index], right[index]) for index in range(4)]
    left_side = [geo.addLine(left[index], left[index + 1]) for index in range(3)]
    right_side = [geo.addLine(right[index], right[index + 1]) for index in range(3)]
    rim_points = [
        geo.addPoint(float(u), float(z), 0.0)
        for u, z in zip(rim["u"], rim["z"])
    ]
    centre_point = geo.addPoint(
        math.pi * params.mean_radius_mm, params.hole_axial_pos_mm, 0.0
    )
    rim_lines = [
        geo.addLine(rim_points[index], rim_points[(index + 1) % inputs.rim_segments])
        for index in range(inputs.rim_segments)
    ]
    rim_loop = geo.addCurveLoop(rim_lines)
    outer_loops = [
        geo.addCurveLoop(
            [horizontal[index], right_side[index], -horizontal[index + 1], -left_side[index]]
        )
        for index in range(3)
    ]
    surface_tags = {
        "ring_A": geo.addPlaneSurface([outer_loops[0]]),
        "hole_band": geo.addPlaneSurface([outer_loops[1], rim_loop]),
        "ring_B": geo.addPlaneSurface([outer_loops[2]]),
        "feed_disk": geo.addPlaneSurface([rim_loop]),
    }
    geo.synchronize()
    gmsh.model.mesh.embed(0, [centre_point], 2, surface_tags["feed_disk"])
    for curve in rim_lines:
        gmsh.model.mesh.setTransfiniteCurve(curve, 2)
    for curve in horizontal:
        gmsh.model.mesh.setTransfiniteCurve(curve, inputs.n_theta + 1)
    spans = (
        params.z1_mm,
        params.z2_mm - params.z1_mm,
        params.length_mm - params.z2_mm,
    )
    axial_counts = _partition_counts(inputs.n_axial, spans)
    for curve, count in zip(left_side, axial_counts):
        gmsh.model.mesh.setTransfiniteCurve(curve, count + 1)
    for curve, count in zip(right_side, axial_counts):
        gmsh.model.mesh.setTransfiniteCurve(curve, count + 1)
    translation = [
        1.0,
        0.0,
        0.0,
        width,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    gmsh.model.mesh.setPeriodic(1, right_side, left_side, translation)
    settings = {
        "Mesh.ElementOrder": 1.0,
        "Mesh.RecombineAll": 0.0,
        "Mesh.MeshSizeFromPoints": 0.0,
        "Mesh.MeshSizeFromCurvature": 0.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.0,
        "Mesh.Algorithm": 6.0,
    }
    for name, value in settings.items():
        gmsh.option.setNumber(name, value)
    far_size = min(width / inputs.n_theta, params.length_mm / inputs.n_axial)
    rim_chord = 2.0 * params.hole_radius_mm * math.sin(math.pi / inputs.rim_segments)
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", rim_lines)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", inputs.rim_segments)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", rim_chord)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", far_size)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 4.0 * params.hole_radius_mm)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    for name, tag in surface_tags.items():
        physical = gmsh.model.addPhysicalGroup(2, [tag])
        gmsh.model.setPhysicalName(2, physical, name)
    gmsh.model.mesh.generate(2)
    triangle_type = int(gmsh.model.mesh.getElementType("Triangle", 1))
    require(records, "master.element_type.Tri3", triangle_type == 2, triangle_type, 2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))

    raw_tags, raw_coordinates, _ = gmsh.model.mesh.getNodes()
    raw_tags = np.asarray(raw_tags, dtype=np.int64)
    raw_points = np.asarray(raw_coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
    tag_to_raw = {int(tag): index for index, tag in enumerate(raw_tags)}
    raw_triangles: list[np.ndarray] = []
    disk_flags: list[np.ndarray] = []
    zone_values: list[np.ndarray] = []
    for name, surface_tag in surface_tags.items():
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2, surface_tag)
        unexpected = [int(kind) for kind in element_types if int(kind) != triangle_type]
        require(records, f"master.{name}.only_Tri3", not unexpected, unexpected, [])
        count = 0
        for kind, tags, nodes in zip(element_types, element_tags, node_tags):
            if int(kind) != triangle_type:
                continue
            triangles = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
            coordinates = raw_points[
                np.vectorize(tag_to_raw.__getitem__, otypes=[np.int64])(triangles)
            ]
            area2 = (
                (coordinates[:, 1, 0] - coordinates[:, 0, 0])
                * (coordinates[:, 2, 1] - coordinates[:, 0, 1])
                - (coordinates[:, 1, 1] - coordinates[:, 0, 1])
                * (coordinates[:, 2, 0] - coordinates[:, 0, 0])
            )
            require(
                records,
                f"master.{name}.nondegenerate",
                bool(np.all(np.abs(area2) > 1.0e-14)),
                float(np.min(np.abs(area2))),
                "> 1e-14 mm2",
            )
            reverse = area2 < 0.0
            triangles[reverse, 1], triangles[reverse, 2] = (
                triangles[reverse, 2].copy(),
                triangles[reverse, 1].copy(),
            )
            raw_triangles.append(triangles)
            disk_flags.append(np.full(len(triangles), name == "feed_disk", dtype=bool))
            zone = {"ring_A": 0, "hole_band": 1, "feed_disk": 1, "ring_B": 2}[name]
            zone_values.append(np.full(len(triangles), zone, dtype=np.uint8))
            count += len(tags)
        require(records, f"master.{name}.nonempty", count > 0, count, "> 0")
    triangles_by_tag = np.concatenate(raw_triangles)
    disk_mask = np.concatenate(disk_flags)
    zones = np.concatenate(zone_values)

    periodic_map: dict[int, int] = {}
    periodic_pairs: dict[str, int] = {}
    for index, (slave_curve, master_curve) in enumerate(zip(right_side, left_side)):
        returned_master, slave_nodes, master_nodes, affine = gmsh.model.mesh.getPeriodicNodes(
            1, slave_curve, True
        )
        require(
            records,
            f"master.periodic.band_{index}.master_curve",
            int(returned_master) == master_curve,
            int(returned_master),
            master_curve,
        )
        slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
        master_nodes = np.asarray(master_nodes, dtype=np.int64)
        require(
            records,
            f"master.periodic.band_{index}.node_count",
            len(slave_nodes) == len(master_nodes) and len(slave_nodes) >= 2,
            len(slave_nodes),
            ">=2 matching nodes",
        )
        for slave, master in zip(slave_nodes, master_nodes):
            existing = periodic_map.get(int(slave))
            if existing is not None and existing != int(master):
                raise PortedMeshError("inconsistent periodic corner correspondence")
            periodic_map[int(slave)] = int(master)
        periodic_pairs[f"band_{index}"] = len(slave_nodes)
    remapped_tags = triangles_by_tag.copy()
    for slave, master in periodic_map.items():
        remapped_tags[remapped_tags == slave] = master
    used_tags = np.unique(remapped_tags)
    compact = {int(tag): index for index, tag in enumerate(used_tags)}
    triangles = np.vectorize(compact.__getitem__, otypes=[np.uint64])(remapped_tags)
    points = np.asarray([raw_points[tag_to_raw[int(tag)]] for tag in used_tags])
    points[:, 0] = np.mod(points[:, 0], width)

    def point_node(point_tag: int) -> int:
        tags, _coords, _params = gmsh.model.mesh.getNodes(0, point_tag, includeBoundary=True)
        if len(tags) != 1:
            raise PortedMeshError(f"point entity {point_tag} has {len(tags)} mesh nodes")
        tag = periodic_map.get(int(tags[0]), int(tags[0]))
        return compact[tag]

    rim_nodes = np.asarray([point_node(tag) for tag in rim_points], dtype=np.uint64)
    centre_node = point_node(centre_point)
    rim_edges = np.column_stack([rim_nodes, np.roll(rim_nodes, -1)])
    for index, curve in enumerate(rim_lines):
        _element_tags, nodes = gmsh.model.mesh.getElementsByType(1, curve)
        line_nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 2)
        require(
            records,
            f"master.rim_curve_{index}.endpoints_only",
            len(line_nodes) == 1
            and set(int(value) for value in line_nodes[0])
            == {int(gmsh.model.mesh.getNodes(0, rim_points[index], True)[0][0]),
                int(gmsh.model.mesh.getNodes(0, rim_points[(index + 1) % inputs.rim_segments], True)[0][0])},
            line_nodes.tolist(),
            "one Line2 with intended endpoints",
        )
    unique_edges, edge_counts = _edge_census(triangles)
    require(
        records,
        "master.edge_owners_one_or_two",
        bool(np.all((edge_counts == 1) | (edge_counts == 2))),
        {"minimum": int(edge_counts.min()), "maximum": int(edge_counts.max())},
        "1 or 2",
    )
    edge_count_map = {tuple(int(v) for v in edge): int(count) for edge, count in zip(unique_edges, edge_counts)}
    require(
        records,
        "master.rim_shared",
        all(edge_count_map.get(tuple(sorted(map(int, edge)))) == 2 for edge in rim_edges),
        [edge_count_map.get(tuple(sorted(map(int, edge)))) for edge in rim_edges],
        "all 2 owners",
    )
    z0_edges = _select_edges(unique_edges, points, z=0.0)
    z_l_edges = _select_edges(unique_edges, points, z=params.length_mm)
    z1_edges = _select_edges(unique_edges, points, z=params.z1_mm)
    z2_edges = _select_edges(unique_edges, points, z=params.z2_mm)
    seam_edges = _select_edges(unique_edges, points, u=0.0)
    require(
        records,
        "master.periodic_seam_internal",
        len(seam_edges) > 0
        and all(edge_count_map[tuple(map(int, edge))] == 2 for edge in seam_edges),
        {"edge_count": len(seam_edges), "owners": sorted({edge_count_map[tuple(map(int, edge))] for edge in seam_edges})},
        "nonempty and all 2 owners",
    )
    for name, edges in (("z1", z1_edges), ("z2", z2_edges)):
        require(
            records,
            f"master.{name}_partition_internal",
            len(edges) > 0
            and all(edge_count_map[tuple(map(int, edge))] == 2 for edge in edges),
            len(edges),
            "nonempty and all 2 owners",
        )
    boundary_edges = unique_edges[edge_counts == 1]
    expected_boundary = np.concatenate([z0_edges, z_l_edges])
    require(
        records,
        "master.only_axial_external_edges",
        np.array_equal(
            np.unique(boundary_edges, axis=0), np.unique(expected_boundary, axis=0)
        ),
        len(boundary_edges),
        len(expected_boundary),
    )
    exact_rim_residual = np.maximum(
        np.abs(
            np.hypot(rim["x"], rim["z"] - params.hole_axial_pos_mm)
            - params.hole_radius_mm
        ),
        np.abs(
            np.hypot(rim["x"], rim["y"])
            - np.asarray(params.bore_radius_mm(rim["z"]))
        ),
    )
    require(
        records,
        "master.rim_analytic_residual_mm",
        float(exact_rim_residual.max()) <= NODE_RESIDUAL_MM,
        float(exact_rim_residual.max()),
        NODE_RESIDUAL_MM,
    )
    disk_nodes = np.unique(triangles[disk_mask])
    require(
        records,
        "master.disk_contains_feed_centre",
        centre_node in set(int(value) for value in disk_nodes),
        centre_node,
        "embedded disk node",
    )
    sagitta = params.hole_radius_mm * (1.0 - math.cos(math.pi / inputs.rim_segments))
    polygon_area = (
        0.5
        * inputs.rim_segments
        * params.hole_radius_mm**2
        * math.sin(2.0 * math.pi / inputs.rim_segments)
    )
    polygon_circle_error = relative_error(
        polygon_area, math.pi * params.hole_radius_mm**2
    )
    if inputs.rim_segments >= 128:
        require(
            records,
            "master.rim_sagitta",
            sagitta <= 0.001,
            sagitta,
            "<=0.001 mm",
        )
        require(
            records,
            "master.polygon_circle_area_error",
            polygon_circle_error <= 5.0e-4,
            polygon_circle_error,
            5.0e-4,
        )
    return MasterMesh(
        points_uz_mm=np.ascontiguousarray(points),
        triangles=np.ascontiguousarray(triangles, dtype=np.uint64),
        disk_triangle_mask=np.ascontiguousarray(disk_mask),
        axial_zone_id=np.ascontiguousarray(zones),
        rim_nodes=np.ascontiguousarray(rim_nodes),
        rim_edges=np.ascontiguousarray(rim_edges),
        seam_edges=np.ascontiguousarray(seam_edges),
        z1_edges=np.ascontiguousarray(z1_edges),
        z2_edges=np.ascontiguousarray(z2_edges),
        z0_edges=np.ascontiguousarray(z0_edges),
        zL_edges=np.ascontiguousarray(z_l_edges),
        centre_node=centre_node,
        metadata={
            "coordinate_unit": "mm",
            "full_width_mm": width,
            "raw_node_count": len(raw_tags),
            "collapsed_node_count": len(points),
            "triangle_count": len(triangles),
            "disk_triangle_count": int(disk_mask.sum()),
            "periodic_pairs": periodic_pairs,
            "axial_partition_cells": axial_counts,
            "rim_segments": inputs.rim_segments,
            "rim_sagitta_mm": sagitta,
            "rim_polygon_area_mm2": polygon_area,
            "rim_polygon_circle_relative_error": polygon_circle_error,
            "rim_statement": (
                "rim nodes are analytic cylinder/bore intersections; first-order Tri3 edges are circular chords"
            ),
            "far_size_mm": far_size,
            "rim_chord_mm": rim_chord,
        },
    )


def _prism_gauss_metrics(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangle_points = ((1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 6.0))
    volume = np.zeros(len(points), dtype=np.float64)
    minimum = np.full(len(points), np.inf, dtype=np.float64)
    for u, v in triangle_points:
        for w in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)):
            derivatives = np.asarray(
                [
                    [-(1.0 - w) / 2.0, -(1.0 - w) / 2.0, -(1.0 - u - v) / 2.0],
                    [(1.0 - w) / 2.0, 0.0, -u / 2.0],
                    [0.0, (1.0 - w) / 2.0, -v / 2.0],
                    [-(1.0 + w) / 2.0, -(1.0 + w) / 2.0, (1.0 - u - v) / 2.0],
                    [(1.0 + w) / 2.0, 0.0, u / 2.0],
                    [0.0, (1.0 + w) / 2.0, v / 2.0],
                ],
                dtype=np.float64,
            )
            jacobian = np.einsum("mnc,na->mca", points, derivatives)
            determinant = np.linalg.det(jacobian)
            volume += determinant / 6.0
            minimum = np.minimum(minimum, determinant)
    return volume, minimum


def _prism_metrics(
    points_m: np.ndarray, prisms: np.ndarray, chunk_size: int = 100_000
) -> dict[str, np.ndarray]:
    count = len(prisms)
    volumes = np.empty(count, dtype=np.float64)
    minimum_det = np.empty(count, dtype=np.float64)
    aspect = np.empty(count, dtype=np.float64)
    centres = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        cell_points = points_m[prisms[start:stop].astype(np.int64) - 1]
        volume, determinant = _prism_gauss_metrics(cell_points)
        edge_lengths = np.linalg.norm(
            cell_points[:, PRISM_EDGES[:, 1]] - cell_points[:, PRISM_EDGES[:, 0]],
            axis=2,
        )
        volumes[start:stop] = volume
        minimum_det[start:stop] = determinant
        aspect[start:stop] = edge_lengths.max(axis=1) / edge_lengths.min(axis=1)
        centres[start:stop] = cell_points.mean(axis=1)
    return {
        "volume_m3": volumes,
        "custom_minDetJac": minimum_det,
        "aspect_ratio": aspect,
        "cell_centres_m": centres,
    }


def _film_node_tags(master_nodes: np.ndarray, layer: int, master_count: int) -> np.ndarray:
    return master_nodes.astype(np.uint64) + np.uint64(1 + layer * master_count)


def _orient_periodic_edges(edges: np.ndarray, points: np.ndarray, width: float) -> np.ndarray:
    oriented = edges.copy()
    for row in oriented:
        ua, ub = points[row, 0]
        if abs(ua - ub) > width / 2.0:
            if ua < ub:
                row[:] = row[::-1]
        elif ua > ub:
            row[:] = row[::-1]
    return oriented


def build_prism_mesh(
    master: MasterMesh,
    params: PortedParams,
    n_gap: int,
    tube_layers: int,
    tube_grading: float,
    gap_inflation_ratio: float = 5.0,
) -> PrismMesh:
    if n_gap < 1:
        raise PortedMeshError("nGap must be positive")
    master_count = len(master.points_uz_mm)
    triangle_count = len(master.triangles)
    theta = master.points_uz_mm[:, 0] / params.mean_radius_mm
    z_mm = master.points_uz_mm[:, 1]
    rho_j = np.asarray(params.rho_j_mm(theta, z_mm), dtype=np.float64)
    rho_b = np.asarray(params.bore_radius_mm(z_mm), dtype=np.float64)
    gap = rho_b - rho_j
    if np.any(rho_j <= 0.0) or np.any(gap <= 0.0):
        raise PortedMeshError(
            f"requires rho_b>rho_j>0; min rho_j={rho_j.min()}, min gap={gap.min()}"
        )
    xi = symmetric_gap_coordinates(n_gap, gap_inflation_ratio)
    gap_layer_fractions = np.diff(xi)
    rho = rho_j[None, :] + xi[:, None] * gap[None, :]
    film = np.empty((n_gap + 1, master_count, 3), dtype=np.float64)
    film[:, :, 0] = rho * np.sin(theta)[None, :]
    film[:, :, 1] = -rho * np.cos(theta)[None, :]
    film[:, :, 2] = z_mm[None, :]

    disk_triangles = master.triangles[master.disk_triangle_mask]
    disk_nodes = np.unique(disk_triangles)
    disk_lookup = np.full(master_count, -1, dtype=np.int64)
    disk_lookup[disk_nodes] = np.arange(len(disk_nodes), dtype=np.int64)
    disk_local_triangles = disk_lookup[disk_triangles]
    bore_disk = film[-1, disk_nodes]
    eta = np.linspace(0.0, 1.0, tube_layers + 1, dtype=np.float64)
    grading = eta**tube_grading
    if not (
        grading[0] == 0.0
        and grading[-1] == 1.0
        and np.all(np.diff(grading) > 0.0)
    ):
        raise PortedMeshError("tube grading must map [0,1] strictly increasingly to [0,1]")
    tube = np.empty((tube_layers + 1, len(disk_nodes), 3), dtype=np.float64)
    tube[:, :, 0] = bore_disk[None, :, 0]
    tube[:, :, 1] = bore_disk[None, :, 1] + grading[:, None] * (
        params.y_feed_end_mm - bore_disk[None, :, 1]
    )
    tube[:, :, 2] = bore_disk[None, :, 2]
    # eta=0 is the exact outer film node set; only eta>0 adds coordinates.
    points_mm = np.concatenate([film.reshape(-1, 3), tube[1:].reshape(-1, 3)])
    points_m = np.ascontiguousarray(points_mm * SI_PER_MM, dtype=np.float64)

    film_prisms = []
    for layer in range(n_gap):
        bottom = _film_node_tags(master.triangles, layer, master_count)
        top = _film_node_tags(master.triangles, layer + 1, master_count)
        film_prisms.append(np.column_stack([bottom, top]))
    film_prisms_array = np.ascontiguousarray(np.concatenate(film_prisms), dtype=np.uint64)
    film_point_count = (n_gap + 1) * master_count

    def tube_tags(layer: int, local_nodes: np.ndarray) -> np.ndarray:
        if layer == 0:
            return _film_node_tags(disk_nodes[local_nodes], n_gap, master_count)
        return (
            np.uint64(film_point_count)
            + np.uint64((layer - 1) * len(disk_nodes))
            + local_nodes.astype(np.uint64)
            + np.uint64(1)
        )

    tube_prisms = []
    for layer in range(tube_layers):
        bottom = tube_tags(layer, disk_local_triangles)
        top = tube_tags(layer + 1, disk_local_triangles)
        tube_prisms.append(np.column_stack([bottom, top]))
    tube_prisms_array = np.ascontiguousarray(np.concatenate(tube_prisms), dtype=np.uint64)
    prisms = np.ascontiguousarray(
        np.concatenate([film_prisms_array, tube_prisms_array]), dtype=np.uint64
    )

    journal = _film_node_tags(master.triangles[:, [0, 2, 1]], 0, master_count)
    outside_triangles = master.triangles[~master.disk_triangle_mask]
    bore = _film_node_tags(outside_triangles, n_gap, master_count)
    pressure = tube_tags(tube_layers, disk_local_triangles)
    mouth = _film_node_tags(disk_triangles, n_gap, master_count)

    width = float(master.metadata["full_width_mm"])
    z0_edges = _orient_periodic_edges(master.z0_edges, master.points_uz_mm, width)
    z_l_edges = _orient_periodic_edges(master.zL_edges, master.points_uz_mm, width)
    z0_quads: list[np.ndarray] = []
    z_l_quads: list[np.ndarray] = []
    for layer in range(n_gap):
        z0_inner = _film_node_tags(z0_edges, layer, master_count)
        z0_outer = _film_node_tags(z0_edges, layer + 1, master_count)
        z0_quads.append(
            np.column_stack(
                [z0_inner[:, 0], z0_inner[:, 1], z0_outer[:, 1], z0_outer[:, 0]]
            )
        )
        zl_inner = _film_node_tags(z_l_edges, layer, master_count)
        zl_outer = _film_node_tags(z_l_edges, layer + 1, master_count)
        z_l_quads.append(
            np.column_stack(
                [zl_inner[:, 0], zl_outer[:, 0], zl_outer[:, 1], zl_inner[:, 1]]
            )
        )
    rim_local = disk_lookup[master.rim_edges]
    tube_wall: list[np.ndarray] = []
    for layer in range(tube_layers):
        lower = tube_tags(layer, rim_local)
        upper = tube_tags(layer + 1, rim_local)
        tube_wall.append(
            np.column_stack(
                [lower[:, 0], upper[:, 0], upper[:, 1], lower[:, 1]]
            )
        )

    film_count = len(film_prisms_array)
    feed_count = len(tube_prisms_array)
    master_theta = np.mod(
        np.arctan2(
            np.sin(theta[master.triangles]).mean(axis=1),
            np.cos(theta[master.triangles]).mean(axis=1),
        ),
        2.0 * math.pi,
    )
    master_z = z_mm[master.triangles].mean(axis=1)
    master_gap_um = (
        np.asarray(params.bore_radius_mm(master_z))
        - np.asarray(params.rho_j_mm(master_theta, master_z))
    ) * 1_000.0
    film_zone = np.tile(master.axial_zone_id, n_gap)
    film_layer = np.repeat(np.arange(n_gap, dtype=np.int32), triangle_count)
    film_theta = np.tile(np.degrees(master_theta), n_gap)
    film_gap = np.tile(master_gap_um, n_gap)
    disk_theta = master_theta[master.disk_triangle_mask]
    fields: dict[str, np.ndarray] = {
        "region_id": np.concatenate(
            [np.zeros(film_count, dtype=np.int32), np.ones(feed_count, dtype=np.int32)]
        ),
        "axial_zone_id": np.concatenate(
            [film_zone.astype(np.int32), np.ones(feed_count, dtype=np.int32)]
        ),
        "gap_layer_index": np.concatenate(
            [film_layer, np.full(feed_count, -1, dtype=np.int32)]
        ),
        "gap_um": np.concatenate([film_gap, np.zeros(feed_count, dtype=np.float64)]),
        "theta_deg": np.concatenate(
            [film_theta, np.tile(np.degrees(disk_theta), tube_layers)]
        ),
        "solve_eligible": np.ones(len(prisms), dtype=np.int32),
        "distorted_geometry": np.zeros(len(prisms), dtype=np.int32),
    }
    custom = _prism_metrics(points_m, prisms)
    fields["volume_m3"] = custom["volume_m3"]
    fields["minDetJac"] = custom["custom_minDetJac"]
    fields["aspect_ratio"] = custom["aspect_ratio"]
    layer_distances = np.diff(tube[:, :, 1], axis=0)
    centre_local = int(disk_lookup[master.centre_node])
    if centre_local < 0:
        raise PortedMeshError("embedded feed-centre node is not in the disk node set")
    centreline_tags = np.concatenate(
        [
            _film_node_tags(np.asarray([master.centre_node]), layer, master_count)
            for layer in range(n_gap + 1)
        ]
        + [tube_tags(layer, np.asarray([centre_local])) for layer in range(1, tube_layers + 1)]
    )
    return PrismMesh(
        points_m=points_m,
        prisms=prisms,
        boundary_triangles={
            "journal_wall": np.ascontiguousarray(journal, dtype=np.uint64),
            "bushing_bore_wall": np.ascontiguousarray(bore, dtype=np.uint64),
            "pressure_feed": np.ascontiguousarray(pressure, dtype=np.uint64),
        },
        boundary_quads={
            "axial_end_z0": np.ascontiguousarray(np.concatenate(z0_quads), dtype=np.uint64),
            "axial_end_zL": np.ascontiguousarray(np.concatenate(z_l_quads), dtype=np.uint64),
            "feed_tube_wall": np.ascontiguousarray(np.concatenate(tube_wall), dtype=np.uint64),
        },
        mouth_triangles=np.ascontiguousarray(mouth, dtype=np.uint64),
        cell_tags=np.arange(1, len(prisms) + 1, dtype=np.uint64),
        node_tags=np.arange(1, len(points_m) + 1, dtype=np.uint64),
        cell_fields={name: np.ascontiguousarray(values) for name, values in fields.items()},
        cell_centres_m=np.ascontiguousarray(custom["cell_centres_m"]),
        film_cell_count=film_count,
        feed_cell_count=feed_count,
        master_triangle_count=triangle_count,
        disk_triangle_count=len(disk_triangles),
        metadata={
            "coordinate_unit": "m",
            "source_unit": "mm",
            "scale_to_m_applied_once": SI_PER_MM,
            "n_gap": n_gap,
            "gap_layer_coordinates": xi.tolist(),
            "gap_layer_fractions": gap_layer_fractions.tolist(),
            "gap_inflation_ratio_target": gap_inflation_ratio,
            "gap_inflation_ratio_achieved": float(
                gap_layer_fractions.max() / gap_layer_fractions.min()
            ),
            "tube_layers": tube_layers,
            "tube_grading_exponent": tube_grading,
            "minimum_tube_layer_mm": float(layer_distances.min()),
            "maximum_tube_layer_mm": float(layer_distances.max()),
            "master_node_count": master_count,
            "disk_node_count": len(disk_nodes),
            "film_point_count": film_point_count,
            "centreline_node_tags": centreline_tags.tolist(),
            "solve_eligible": True,
            "distorted_geometry": False,
            "full_360_degrees": True,
            "feed_mouth_is_internal": True,
        },
    )


def _triangle_boundary_volume(points: np.ndarray) -> float:
    centre = points.mean(axis=1)
    area = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) / 2.0
    return float(np.einsum("mc,mc->m", centre, area).sum() / 3.0)


def _quad_boundary_volume(points: np.ndarray) -> float:
    signs = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    total = 0.0
    for r in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)):
        for s in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)):
            shape = 0.25 * (1.0 + r * signs[:, 0]) * (1.0 + s * signs[:, 1])
            derivative_r = 0.25 * signs[:, 0] * (1.0 + s * signs[:, 1])
            derivative_s = 0.25 * signs[:, 1] * (1.0 + r * signs[:, 0])
            location = np.einsum("n,mnc->mc", shape, points)
            tangent_r = np.einsum("n,mnc->mc", derivative_r, points)
            tangent_s = np.einsum("n,mnc->mc", derivative_s, points)
            area = np.cross(tangent_r, tangent_s)
            total += float(np.einsum("mc,mc->m", location, area).sum() / 3.0)
    return total


def oriented_boundary_volume(mesh: PrismMesh) -> float:
    tri = sum(
        _triangle_boundary_volume(mesh.points_m[faces.astype(np.int64) - 1])
        for faces in mesh.boundary_triangles.values()
    )
    quad = sum(
        _quad_boundary_volume(mesh.points_m[faces.astype(np.int64) - 1])
        for faces in mesh.boundary_quads.values()
    )
    return tri + quad


def validate_external_face_orientation(
    mesh: PrismMesh,
    records: list[dict[str, Any]],
    patch_owners: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Require every stored external face normal to point away from its incident cell."""
    if patch_owners is None:
        cell_indices = np.arange(len(mesh.prisms), dtype=np.int64)
        tri_unique, tri_counts, tri_owners, _ = _face_census(
            mesh.prisms[:, PRISM_TRI_FACES].reshape(-1, 3),
            np.repeat(cell_indices, len(PRISM_TRI_FACES)),
        )
        quad_unique, quad_counts, quad_owners, _ = _face_census(
            mesh.prisms[:, PRISM_QUAD_FACES].reshape(-1, 4),
            np.repeat(cell_indices, len(PRISM_QUAD_FACES)),
        )
    result: dict[str, dict[str, float | int]] = {}
    for name, faces in (*mesh.boundary_triangles.items(), *mesh.boundary_quads.items()):
        triangle = faces.shape[1] == 3
        if patch_owners is None:
            unique, counts, owners = (
                (tri_unique, tri_counts, tri_owners)
                if triangle
                else (quad_unique, quad_counts, quad_owners)
            )
            indices = _lookup_rows(unique, faces)
            incident_cells = owners[indices]
            one_incident_cell = bool(np.all(counts[indices] == 1))
        else:
            incident_cells = patch_owners[name]
            one_incident_cell = len(incident_cells) == len(faces)
        face_points = mesh.points_m[faces.astype(np.int64) - 1]
        if triangle:
            area_vectors = np.cross(
                face_points[:, 1] - face_points[:, 0],
                face_points[:, 2] - face_points[:, 0],
            ) / 2.0
        else:
            area_vectors = (
                np.cross(
                    face_points[:, 1] - face_points[:, 0],
                    face_points[:, 2] - face_points[:, 0],
                )
                + np.cross(
                    face_points[:, 2] - face_points[:, 0],
                    face_points[:, 3] - face_points[:, 0],
                )
            ) / 2.0
        areas = np.linalg.norm(area_vectors, axis=1)
        projections = np.einsum(
            "ij,ij->i",
            area_vectors / areas[:, None],
            face_points.mean(axis=1) - mesh.cell_centres_m[incident_cells],
        )
        valid = bool(
            one_incident_cell
            and np.all(np.isfinite(areas))
            and np.all(areas > 0.0)
            and np.all(np.isfinite(projections))
            and np.all(projections > 0.0)
        )
        require(
            records,
            f"geometry.external_face_orientation.{name}",
            valid,
            {
                "face_count": len(faces),
                "locally_reversed": int(np.sum(projections <= 0.0)),
                "minimum_outward_projection_m": float(projections.min(initial=math.inf)),
            },
            "every face has one incident cell and a strictly outward stored normal",
        )
        result[name] = {
            "face_count": len(faces),
            "minimum_outward_projection_m": float(projections.min(initial=math.inf)),
            "minimum_area_m2": float(areas.min(initial=math.inf)),
        }
    return result


def _row_keys(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def _structured_rows(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    dtype = np.dtype([(f"f{index}", contiguous.dtype) for index in range(contiguous.shape[1])])
    return contiguous.view(dtype).ravel()


def _face_census(
    faces: np.ndarray, owners: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical = np.sort(faces, axis=1)
    unique, inverse, counts = np.unique(
        canonical, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts[:-1])))
    first = owners[order[offsets]]
    second = np.full(len(unique), -1, dtype=np.int64)
    internal = counts == 2
    second[internal] = owners[order[offsets[internal] + 1]]
    return unique, counts, first, second


def _lookup_rows(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    haystack_keys = _structured_rows(haystack)
    needle_keys = _structured_rows(np.sort(needles, axis=1))
    indices = np.searchsorted(haystack_keys, needle_keys)
    if np.any(indices >= len(haystack)) or not np.array_equal(
        haystack_keys[indices], needle_keys
    ):
        raise PortedMeshError("declared face is absent from the Prism6 face census")
    return indices


def _extruded_edge_quads(
    edges: np.ndarray, n_gap: int, master_count: int
) -> np.ndarray:
    quads = []
    for layer in range(n_gap):
        lower = _film_node_tags(edges, layer, master_count)
        upper = _film_node_tags(edges, layer + 1, master_count)
        quads.append(
            np.column_stack(
                [lower[:, 0], lower[:, 1], upper[:, 1], upper[:, 0]]
            )
        )
    return np.ascontiguousarray(np.concatenate(quads), dtype=np.uint64)


def _component_count(
    cell_count: int, first: np.ndarray, second: np.ndarray
) -> tuple[int, np.ndarray]:
    parent = np.arange(cell_count, dtype=np.int64)
    rank = np.zeros(cell_count, dtype=np.uint8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    for left, right in zip(first, second):
        if right < 0:
            continue
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left == root_right:
            continue
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
    roots = np.asarray([find(index) for index in range(cell_count)], dtype=np.int64)
    return len(np.unique(roots)), roots


def validate_topology(
    mesh: PrismMesh,
    master: MasterMesh,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    cell_count = len(mesh.prisms)
    prism_indices = np.arange(cell_count, dtype=np.int64)
    tri_faces = mesh.prisms[:, PRISM_TRI_FACES].reshape(-1, 3)
    quad_faces = mesh.prisms[:, PRISM_QUAD_FACES].reshape(-1, 4)
    tri_unique, tri_counts, tri_first, tri_second = _face_census(
        tri_faces, np.repeat(prism_indices, len(PRISM_TRI_FACES))
    )
    quad_unique, quad_counts, quad_first, quad_second = _face_census(
        quad_faces, np.repeat(prism_indices, len(PRISM_QUAD_FACES))
    )
    require(
        records,
        "topology.face_owner_counts",
        bool(
            np.all((tri_counts == 1) | (tri_counts == 2))
            and np.all((quad_counts == 1) | (quad_counts == 2))
        ),
        {
            "triangle": sorted(set(int(value) for value in tri_counts)),
            "quadrilateral": sorted(set(int(value) for value in quad_counts)),
        },
        [1, 2],
    )
    declared_tri = np.concatenate(list(mesh.boundary_triangles.values()))
    declared_quad = np.concatenate(list(mesh.boundary_quads.values()))
    patch_face_counts = {
        name: len(faces)
        for name, faces in (*mesh.boundary_triangles.items(), *mesh.boundary_quads.items())
    }
    total_boundary_faces = sum(patch_face_counts.values())
    census_boundary_faces = int(np.sum(tri_counts == 1) + np.sum(quad_counts == 1))
    require(
        records,
        "topology.exact_patch_face_counts",
        set(patch_face_counts) == set(SURFACE_ENTITIES)
        and total_boundary_faces == census_boundary_faces,
        {"patches": patch_face_counts, "total": total_boundary_faces},
        {"patches": sorted(SURFACE_ENTITIES), "census_total": census_boundary_faces},
    )
    declared_tri_indices = _lookup_rows(tri_unique, declared_tri)
    declared_quad_indices = _lookup_rows(quad_unique, declared_quad)
    require(
        records,
        "topology.declared_boundaries_one_owner",
        bool(
            np.all(tri_counts[declared_tri_indices] == 1)
            and np.all(quad_counts[declared_quad_indices] == 1)
        ),
        {
            "triangle_owner_counts": sorted(
                set(int(value) for value in tri_counts[declared_tri_indices])
            ),
            "quad_owner_counts": sorted(
                set(int(value) for value in quad_counts[declared_quad_indices])
            ),
        },
        [1],
    )
    require(
        records,
        "topology.boundary_groups_disjoint",
        len(np.unique(np.sort(declared_tri, axis=1), axis=0)) == len(declared_tri)
        and len(np.unique(np.sort(declared_quad, axis=1), axis=0)) == len(declared_quad),
        {"triangles": len(declared_tri), "quads": len(declared_quad)},
        "all unique within their face arity",
    )
    require(
        records,
        "topology.boundary_union_complete",
        np.array_equal(
            np.unique(np.sort(declared_tri, axis=1), axis=0), tri_unique[tri_counts == 1]
        )
        and np.array_equal(
            np.unique(np.sort(declared_quad, axis=1), axis=0), quad_unique[quad_counts == 1]
        ),
        {
            "declared_triangles": len(declared_tri),
            "census_triangles": int(np.sum(tri_counts == 1)),
            "declared_quads": len(declared_quad),
            "census_quads": int(np.sum(quad_counts == 1)),
        },
        "exact one-owner face union",
    )
    mouth_indices = _lookup_rows(tri_unique, mesh.mouth_triangles)
    mouth_regions = np.column_stack(
        [
            mesh.cell_fields["region_id"][tri_first[mouth_indices]],
            mesh.cell_fields["region_id"][tri_second[mouth_indices]],
        ]
    )
    require(
        records,
        "topology.mouth_two_incident_cells_film_and_feed",
        bool(
            np.all(tri_counts[mouth_indices] == 2)
            and np.all(np.sort(mouth_regions, axis=1) == np.asarray([0, 1]))
        ),
        {
            "mouth_triangles": len(mouth_indices),
            "incident_cell_counts": sorted(set(int(value) for value in tri_counts[mouth_indices])),
            "region_pairs": np.unique(np.sort(mouth_regions, axis=1), axis=0).tolist(),
        },
        {"incident_cells": 2, "regions": [0, 1]},
    )
    tri_boundary_keys = set(_row_keys(np.sort(declared_tri, axis=1)).tolist())
    mouth_keys = set(_row_keys(np.sort(mesh.mouth_triangles, axis=1)).tolist())
    require(
        records,
        "topology.mouth_not_boundary",
        tri_boundary_keys.isdisjoint(mouth_keys),
        len(tri_boundary_keys & mouth_keys),
        0,
    )
    expected_point_count = int(mesh.metadata["film_point_count"]) + int(
        mesh.metadata["tube_layers"]
    ) * int(mesh.metadata["disk_node_count"])
    unique_points = len(np.unique(mesh.points_m, axis=0))
    require(
        records,
        "topology.no_duplicate_or_coincident_nodes",
        len(mesh.points_m) == expected_point_count and unique_points == len(mesh.points_m),
        {"points": len(mesh.points_m), "unique": unique_points},
        expected_point_count,
    )
    internal_first = np.concatenate(
        [tri_first[tri_counts == 2], quad_first[quad_counts == 2]]
    )
    internal_second = np.concatenate(
        [tri_second[tri_counts == 2], quad_second[quad_counts == 2]]
    )
    components, roots = _component_count(cell_count, internal_first, internal_second)
    require(records, "topology.one_connected_volume", components == 1, components, 1)
    pressure_indices = _lookup_rows(
        tri_unique, mesh.boundary_triangles["pressure_feed"]
    )
    pressure_owners = tri_first[pressure_indices]
    require(
        records,
        "topology.pressure_to_film_path",
        bool(
            np.all(mesh.cell_fields["region_id"][pressure_owners] == 1)
            and np.any(mesh.cell_fields["region_id"] == 0)
            and len(np.unique(roots)) == 1
        ),
        {"pressure_owner_regions": np.unique(mesh.cell_fields["region_id"][pressure_owners]).tolist()},
        "feed owners in the same component as film",
    )
    require(
        records,
        "topology.journal_continuous_complete",
        len(mesh.boundary_triangles["journal_wall"]) == mesh.master_triangle_count,
        len(mesh.boundary_triangles["journal_wall"]),
        mesh.master_triangle_count,
    )
    require(
        records,
        "topology.bushing_excludes_only_feed_disk",
        len(mesh.boundary_triangles["bushing_bore_wall"])
        == mesh.master_triangle_count - mesh.disk_triangle_count,
        len(mesh.boundary_triangles["bushing_bore_wall"]),
        mesh.master_triangle_count - mesh.disk_triangle_count,
    )
    rim_degree: dict[int, int] = {}
    adjacency: dict[int, list[int]] = {}
    for left, right in master.rim_edges:
        left_i, right_i = int(left), int(right)
        rim_degree[left_i] = rim_degree.get(left_i, 0) + 1
        rim_degree[right_i] = rim_degree.get(right_i, 0) + 1
        adjacency.setdefault(left_i, []).append(right_i)
        adjacency.setdefault(right_i, []).append(left_i)
    visited = {int(master.rim_nodes[0])}
    stack = [int(master.rim_nodes[0])]
    while stack:
        current = stack.pop()
        for node in adjacency[current]:
            if node not in visited:
                visited.add(node)
                stack.append(node)
    require(
        records,
        "topology.rim_one_closed_degree_two_loop",
        len(rim_degree) == len(master.rim_nodes)
        and set(rim_degree.values()) == {2}
        and len(visited) == len(master.rim_nodes),
        {"vertices": len(rim_degree), "degrees": sorted(set(rim_degree.values())), "connected": len(visited)},
        {"vertices": len(master.rim_nodes), "degrees": [2], "connected": len(master.rim_nodes)},
    )
    n_gap = int(mesh.metadata["n_gap"])
    master_count = int(mesh.metadata["master_node_count"])
    for name, edges in (
        ("theta_seam", master.seam_edges),
        ("z1_partition", master.z1_edges),
        ("z2_partition", master.z2_edges),
    ):
        internal_quads = _extruded_edge_quads(edges, n_gap, master_count)
        indices = _lookup_rows(quad_unique, internal_quads)
        require(
            records,
            f"topology.{name}_internal_conformal",
            bool(np.all(quad_counts[indices] == 2)),
            sorted(set(int(value) for value in quad_counts[indices])),
            [2],
        )
    require(
        records,
        "topology.only_Prism6_Tri3_Quad4",
        mesh.prisms.shape[1] == 6
        and all(faces.shape[1] == 3 for faces in mesh.boundary_triangles.values())
        and all(faces.shape[1] == 4 for faces in mesh.boundary_quads.values()),
        {
            "volume_arity": mesh.prisms.shape[1],
            "triangle_arities": sorted({faces.shape[1] for faces in mesh.boundary_triangles.values()}),
            "quad_arities": sorted({faces.shape[1] for faces in mesh.boundary_quads.values()}),
        },
        {"volume": "Prism6", "boundary": ["Tri3", "Quad4"]},
    )
    return {
        "triangle_faces": {
            "unique": len(tri_unique),
            "boundary": int(np.sum(tri_counts == 1)),
            "internal": int(np.sum(tri_counts == 2)),
        },
        "quadrilateral_faces": {
            "unique": len(quad_unique),
            "boundary": int(np.sum(quad_counts == 1)),
            "internal": int(np.sum(quad_counts == 2)),
        },
        "connected_components": components,
        "mouth_triangles": len(mouth_indices),
        "boundary_face_counts": patch_face_counts,
        "total_boundary_faces": total_boundary_faces,
    }


def validate_geometry(
    mesh: PrismMesh,
    master: MasterMesh,
    params: PortedParams,
    inputs: PortedInputs,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    finite = np.isfinite(mesh.points_m).all() and all(
        np.isfinite(values).all() for values in mesh.cell_fields.values()
    )
    require(records, "geometry.finite_coordinates_metrics", finite, finite, True)
    points_mm = mesh.points_m / SI_PER_MM
    master_count = int(mesh.metadata["master_node_count"])
    n_gap = int(mesh.metadata["n_gap"])
    gap_layer_fractions = np.asarray(
        mesh.metadata["gap_layer_fractions"], dtype=np.float64
    )
    expected_ratio = inputs.gap_inflation_ratio if n_gap >= 3 else 1.0
    achieved_ratio = float(
        gap_layer_fractions.max() / gap_layer_fractions.min()
    )
    inflation_valid = (
        len(gap_layer_fractions) == n_gap
        and np.all(gap_layer_fractions > 0.0)
        and abs(float(gap_layer_fractions.sum()) - 1.0) <= 1.0e-14
        and np.allclose(
            gap_layer_fractions,
            gap_layer_fractions[::-1],
            rtol=0.0,
            atol=1.0e-14,
        )
        and abs(achieved_ratio - expected_ratio)
        <= 1.0e-12 * max(1.0, expected_ratio)
    )
    require(
        records,
        "geometry.symmetric_gap_inflation",
        bool(inflation_valid),
        {
            "layer_fractions": gap_layer_fractions.tolist(),
            "achieved_centre_to_wall_ratio": achieved_ratio,
        },
        {
            "positive_symmetric_fractions_sum": 1.0,
            "centre_to_wall_ratio": expected_ratio,
        },
    )
    journal = points_mm[:master_count]
    bore = points_mm[n_gap * master_count : (n_gap + 1) * master_count]
    journal_residual = np.abs(
        np.hypot(journal[:, 0] - params.ex_mm, journal[:, 1] - params.ey_mm)
        - np.asarray(params.journal_radius_mm(journal[:, 2]))
    )
    bore_residual = np.abs(
        np.hypot(bore[:, 0], bore[:, 1])
        - np.asarray(params.bore_radius_mm(bore[:, 2]))
    )
    require(
        records,
        "geometry.journal_cone_nodes",
        float(journal_residual.max()) <= NODE_RESIDUAL_MM,
        float(journal_residual.max()),
        NODE_RESIDUAL_MM,
    )
    require(
        records,
        "geometry.bore_cone_nodes",
        float(bore_residual.max()) <= NODE_RESIDUAL_MM,
        float(bore_residual.max()),
        NODE_RESIDUAL_MM,
    )
    tube_wall_nodes = np.unique(mesh.boundary_quads["feed_tube_wall"]) - 1
    tube_wall_points = points_mm[tube_wall_nodes.astype(np.int64)]
    cylinder_residual = np.abs(
        np.hypot(
            tube_wall_points[:, 0],
            tube_wall_points[:, 2] - params.hole_axial_pos_mm,
        )
        - params.hole_radius_mm
    )
    require(
        records,
        "geometry.feed_wall_cylinder_nodes",
        float(cylinder_residual.max()) <= NODE_RESIDUAL_MM,
        float(cylinder_residual.max()),
        NODE_RESIDUAL_MM,
    )
    rim_expected = rim_coordinates(params, inputs.rim_segments)
    rim_tags = _film_node_tags(master.rim_nodes, n_gap, master_count)
    rim_actual = points_mm[rim_tags.astype(np.int64) - 1]
    rim_expected_xyz = np.column_stack(
        [rim_expected["x"], rim_expected["y"], rim_expected["z"]]
    )
    rim_coordinate_error = float(np.abs(rim_actual - rim_expected_xyz).max())
    require(
        records,
        "geometry.extracted_rim_vertices_exact",
        rim_coordinate_error <= NODE_RESIDUAL_MM,
        rim_coordinate_error,
        NODE_RESIDUAL_MM,
    )
    inlet_faces = mesh.boundary_triangles["pressure_feed"]
    inlet_points = points_mm[inlet_faces.astype(np.int64) - 1]
    inlet_plane_residual = float(
        np.abs(inlet_points[:, :, 1] - params.y_feed_end_mm).max()
    )
    require(
        records,
        "geometry.inlet_plane",
        inlet_plane_residual <= NODE_RESIDUAL_MM,
        inlet_plane_residual,
        NODE_RESIDUAL_MM,
    )
    area_vectors = np.cross(
        inlet_points[:, 1] - inlet_points[:, 0],
        inlet_points[:, 2] - inlet_points[:, 0],
    ) / 2.0
    areas = np.linalg.norm(area_vectors, axis=1)
    inlet_area = float(areas.sum())
    inlet_centroid = (
        areas[:, None] * inlet_points.mean(axis=1)
    ).sum(axis=0) / inlet_area
    mean_normal = area_vectors.sum(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    require(
        records,
        "geometry.inlet_centroid",
        abs(inlet_centroid[0]) <= NODE_RESIDUAL_MM
        and abs(inlet_centroid[2] - params.hole_axial_pos_mm) <= NODE_RESIDUAL_MM,
        inlet_centroid.tolist(),
        [0.0, params.y_feed_end_mm, params.hole_axial_pos_mm],
        NODE_RESIDUAL_MM,
    )
    require(
        records,
        "geometry.inlet_outward_normal",
        float(mean_normal[1]) >= 1.0 - 1.0e-12
        and bool(np.all(area_vectors[:, 1] > 0.0)),
        mean_normal.tolist(),
        [0.0, 1.0, 0.0],
        1.0e-12,
    )
    polygon_area = float(master.metadata["rim_polygon_area_mm2"])
    inlet_polygon_error = relative_error(inlet_area, polygon_area)
    require(
        records,
        "geometry.inlet_area_vs_analytic_polygon",
        inlet_polygon_error <= INLET_POLYGON_REL_TOL,
        {"area_mm2": inlet_area, "relative_error": inlet_polygon_error},
        polygon_area,
        INLET_POLYGON_REL_TOL,
    )
    polygon_circle_error = relative_error(
        inlet_area, math.pi * params.hole_radius_mm**2
    )
    cad_inlet_area_error = relative_error(inlet_area, params.inlet_area_mm2)
    cad_inlet_centre_error = float(
        np.max(np.abs(inlet_centroid - np.asarray(params.inlet_centre_mm)))
    )
    require(
        records,
        "geometry.inlet_vs_CAD_record",
        (inputs.rim_segments < 128 or cad_inlet_area_error <= 5.0e-4)
        and cad_inlet_centre_error <= NODE_RESIDUAL_MM,
        {
            "area_relative_error": cad_inlet_area_error,
            "centroid_max_error_mm": cad_inlet_centre_error,
        },
        {
            "area_relative_error": "<=5e-4 when rim_segments>=128; diagnostic otherwise",
            "centroid_max_error_mm": NODE_RESIDUAL_MM,
        },
    )
    if inputs.rim_segments >= 128:
        require(
            records,
            "geometry.inlet_polygon_vs_circle",
            polygon_circle_error <= 5.0e-4,
            polygon_circle_error,
            5.0e-4,
        )
    if params.eccentricity_mm:
        theta_min = math.atan2(
            params.ex_mm / params.eccentricity_mm,
            -params.ey_mm / params.eccentricity_mm,
        ) % (2.0 * math.pi)
    else:
        theta_min = 0.0
    z_mid = params.length_mm / 2.0
    measured_min = float(
        params.bore_radius_mm(z_mid) - params.rho_j_mm(theta_min, z_mid)
    )
    measured_max = float(
        params.bore_radius_mm(z_mid)
        - params.rho_j_mm((theta_min + math.pi) % (2.0 * math.pi), z_mid)
    )
    require(
        records,
        "geometry.minimum_radial_gap",
        abs(measured_min - params.minimum_gap_mm) <= NODE_RESIDUAL_MM,
        measured_min,
        params.minimum_gap_mm,
        NODE_RESIDUAL_MM,
    )
    require(
        records,
        "geometry.maximum_radial_gap",
        abs(measured_max - params.maximum_gap_mm) <= NODE_RESIDUAL_MM,
        measured_max,
        params.maximum_gap_mm,
        NODE_RESIDUAL_MM,
    )
    centreline_tags = np.asarray(mesh.metadata["centreline_node_tags"], dtype=np.int64)
    centreline = points_mm[centreline_tags - 1]
    centreline_valid = (
        np.max(np.abs(centreline[:, 0])) <= NODE_RESIDUAL_MM
        and np.max(np.abs(centreline[:, 2] - params.hole_axial_pos_mm)) <= NODE_RESIDUAL_MM
        and np.all(np.diff(centreline[:, 1]) > 0.0)
        and abs(centreline[-1, 1] - params.y_feed_end_mm) <= NODE_RESIDUAL_MM
    )
    require(
        records,
        "geometry.feed_centreline_continuous_inside",
        bool(centreline_valid),
        {
            "node_count": len(centreline),
            "minimum_y_increment_mm": float(np.diff(centreline[:, 1]).min()),
            "start_mm": centreline[0].tolist(),
            "end_mm": centreline[-1].tolist(),
        },
        "continuous journal-to-inlet line with strictly increasing y",
    )
    scale_valid = (
        mesh.metadata["coordinate_unit"] == "m"
        and mesh.metadata["scale_to_m_applied_once"] == SI_PER_MM
        and abs(mesh.points_m[:, 2].min()) <= 1.0e-14
        and abs(mesh.points_m[:, 2].max() - params.length_mm * SI_PER_MM) <= 1.0e-14
        and abs(mesh.points_m[:, 1].max() - params.y_feed_end_mm * SI_PER_MM) <= 1.0e-14
        and np.max(np.abs(mesh.points_m)) < 1.0
    )
    require(records, "geometry.SI_scale_once", bool(scale_valid), dict(mesh.metadata), "mm*0.001 exactly once")
    external_orientation = validate_external_face_orientation(mesh, records)
    custom_min_det = mesh.cell_fields["minDetJac"]
    volumes = mesh.cell_fields["volume_m3"]
    require(
        records,
        "quality.custom_positive_Jacobians_volumes",
        bool(
            np.isfinite(custom_min_det).all()
            and np.isfinite(volumes).all()
            and np.all(custom_min_det > 0.0)
            and np.all(volumes > 0.0)
        ),
        {"minimum_minDetJac": float(custom_min_det.min()), "minimum_volume_m3": float(volumes.min())},
        "> 0 and finite",
    )
    cell_volume = float(volumes.sum())
    boundary_volume = oriented_boundary_volume(mesh)
    boundary_error = relative_error(boundary_volume, cell_volume)
    require(
        records,
        "geometry.cell_sum_vs_oriented_boundary_volume",
        boundary_error <= BOUNDARY_VOLUME_REL_TOL,
        {"boundary_volume_m3": boundary_volume, "relative_error": boundary_error},
        cell_volume,
        BOUNDARY_VOLUME_REL_TOL,
    )
    brep_error = relative_error(cell_volume / SI_PER_MM**3, params.native_volume_mm3)
    require(
        records,
        "geometry.linear_mesh_vs_native_BREP_volume",
        brep_error <= BREP_VOLUME_REL_TOL,
        {"mesh_volume_mm3": cell_volume / SI_PER_MM**3, "relative_error": brep_error},
        params.native_volume_mm3,
        BREP_VOLUME_REL_TOL,
    )
    return {
        "journal_node_max_residual_mm": float(journal_residual.max()),
        "bore_node_max_residual_mm": float(bore_residual.max()),
        "feed_cylinder_node_max_residual_mm": float(cylinder_residual.max()),
        "rim_vertex_max_coordinate_error_mm": rim_coordinate_error,
        "minimum_radial_gap_mm": measured_min,
        "maximum_radial_gap_mm": measured_max,
        "inlet": {
            "area_mm2": inlet_area,
            "polygon_area_mm2": polygon_area,
            "polygon_relative_error": inlet_polygon_error,
            "circle_relative_error": polygon_circle_error,
            "CAD_area_relative_error": cad_inlet_area_error,
            "CAD_centroid_max_error_mm": cad_inlet_centre_error,
            "centroid_mm": inlet_centroid.tolist(),
            "mean_normal": mean_normal.tolist(),
        },
        "external_face_orientation": external_orientation,
        "volumes": {
            "cell_sum_m3": cell_volume,
            "oriented_boundary_m3": boundary_volume,
            "boundary_relative_error": boundary_error,
            "native_brep_mm3": params.native_volume_mm3,
            "native_brep_relative_error": brep_error,
        },
    }


def add_discrete_prism_model(
    mesh: PrismMesh,
    records: list[dict[str, Any]],
    model_name: str,
    volume_name: str = "fluid",
) -> dict[str, Any]:
    gmsh.model.add(model_name)
    for tag in SURFACE_ENTITIES.values():
        gmsh.model.addDiscreteEntity(2, tag)
    gmsh.model.addDiscreteEntity(3, VOLUME_ENTITY, list(SURFACE_ENTITIES.values()))
    for name, tag in SURFACE_ENTITIES.items():
        gmsh.model.setEntityName(2, tag, name)
    gmsh.model.setEntityName(3, VOLUME_ENTITY, volume_name)
    tri_type = int(gmsh.model.mesh.getElementType("Triangle", 1))
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    prism_type = int(gmsh.model.mesh.getElementType("Prism", 1))
    require(records, "gmsh.element_type.Tri3", tri_type == 2, tri_type, 2)
    require(records, "gmsh.element_type.Quad4", quad_type == 3, quad_type, 3)
    require(records, "gmsh.element_type.Prism6", prism_type == 6, prism_type, 6)
    gmsh.model.mesh.addNodes(3, VOLUME_ENTITY, mesh.node_tags, mesh.points_m.ravel())
    next_tag = int(mesh.cell_tags[-1]) + 1
    surface_element_counts: dict[str, dict[str, int]] = {}
    for name, faces in mesh.boundary_triangles.items():
        tags = np.arange(next_tag, next_tag + len(faces), dtype=np.uint64)
        next_tag += len(faces)
        gmsh.model.mesh.addElementsByType(
            SURFACE_ENTITIES[name], tri_type, tags, faces.ravel()
        )
        surface_element_counts[name] = {"Tri3": len(faces), "Quad4": 0}
    for name, faces in mesh.boundary_quads.items():
        tags = np.arange(next_tag, next_tag + len(faces), dtype=np.uint64)
        next_tag += len(faces)
        gmsh.model.mesh.addElementsByType(
            SURFACE_ENTITIES[name], quad_type, tags, faces.ravel()
        )
        surface_element_counts[name] = {"Tri3": 0, "Quad4": len(faces)}
    gmsh.model.mesh.addElementsByType(
        VOLUME_ENTITY, prism_type, mesh.cell_tags, mesh.prisms.ravel()
    )
    gmsh.model.mesh.reclassifyNodes()
    for name, physical_id in PHYSICAL_IDS.items():
        if name == "fluid":
            continue
        created = gmsh.model.addPhysicalGroup(
            2, [SURFACE_ENTITIES[name]], tag=physical_id, name=name
        )
        require(records, f"physical.{name}.id", created == physical_id, created, physical_id)
    created = gmsh.model.addPhysicalGroup(
        3, [VOLUME_ENTITY], tag=PHYSICAL_IDS["fluid"], name=volume_name
    )
    require(
        records,
        f"physical.{volume_name}.id",
        created == PHYSICAL_IDS["fluid"],
        created,
        PHYSICAL_IDS["fluid"],
    )
    return {
        "element_types": {"Tri3": tri_type, "Quad4": quad_type, "Prism6": prism_type},
        "surface_element_counts": surface_element_counts,
        "prism_count": len(mesh.prisms),
        "gmsh_generated_3d_mesh": False,
    }


def expected_physical_groups(volume_name: str = "fluid") -> dict[tuple[int, int], dict[str, Any]]:
    groups = {
        (2, PHYSICAL_IDS[name]): {"name": name, "entities": [SURFACE_ENTITIES[name]]}
        for name in SURFACE_ENTITIES
    }
    groups[(3, PHYSICAL_IDS["fluid"])] = {
        "name": volume_name,
        "entities": [VOLUME_ENTITY],
    }
    return groups


def validate_physical_groups(
    records: list[dict[str, Any]], prefix: str, volume_name: str = "fluid"
) -> dict[str, Any]:
    expected = expected_physical_groups(volume_name)
    actual: dict[tuple[int, int], dict[str, Any]] = {}
    for dimension, physical_id in gmsh.model.getPhysicalGroups():
        key = (int(dimension), int(physical_id))
        actual[key] = {
            "name": gmsh.model.getPhysicalName(*key),
            "entities": sorted(
                int(tag) for tag in gmsh.model.getEntitiesForPhysicalGroup(*key)
            ),
        }
    require(
        records,
        f"{prefix}.physical_groups_exact",
        actual == expected,
        {f"{dim}:{tag}": value for (dim, tag), value in actual.items()},
        {f"{dim}:{tag}": value for (dim, tag), value in expected.items()},
    )
    forbidden = {"feed_mouth", "mouth_cap", "internal_feed", "defaultFaces"}
    require(
        records,
        f"{prefix}.no_forbidden_solver_patches",
        forbidden.isdisjoint(value["name"] for value in actual.values()),
        sorted(value["name"] for value in actual.values()),
        f"none of {sorted(forbidden)}",
    )
    return {f"{dim}:{tag}": value for (dim, tag), value in actual.items()}


def add_gmsh_quality(
    mesh: PrismMesh, records: list[dict[str, Any]]
) -> PrismMesh:
    tags = mesh.cell_tags.astype(np.int64, copy=False)
    min_sicn = np.asarray(
        gmsh.model.mesh.getElementQualities(tags, "minSICN"), dtype=np.float64
    )
    min_det = np.asarray(
        gmsh.model.mesh.getElementQualities(tags, "minDetJac"), dtype=np.float64
    )
    gmsh_volume = np.asarray(
        gmsh.model.mesh.getElementQualities(tags, "volume"), dtype=np.float64
    )
    require(
        records,
        "gmsh.quality.Prism6_positive",
        len(min_sicn) == len(mesh.prisms)
        and len(min_det) == len(mesh.prisms)
        and len(gmsh_volume) == len(mesh.prisms)
        and np.isfinite(min_sicn).all()
        and np.isfinite(min_det).all()
        and np.isfinite(gmsh_volume).all()
        and np.all(min_sicn > 0.0)
        and np.all(min_det > 0.0)
        and np.all(gmsh_volume > 0.0),
        {
            "minimum_minSICN": float(min_sicn.min()),
            "minimum_minDetJac": float(min_det.min()),
            "minimum_volume_m3": float(gmsh_volume.min()),
        },
        "finite and strictly positive",
    )
    canonical_volume = mesh.cell_fields["volume_m3"]
    gmsh_volume_error = relative_error(float(gmsh_volume.sum()), float(canonical_volume.sum()))
    require(
        records,
        "gmsh.quality.volume_vs_custom_Gauss",
        gmsh_volume_error <= 1.0e-10,
        gmsh_volume_error,
        1.0e-10,
    )
    fields = dict(mesh.cell_fields)
    fields.update({"minSICN": min_sicn, "minDetJac": min_det, "volume_m3": gmsh_volume})
    return replace(mesh, cell_fields=fields)


def add_element_data_views(mesh: PrismMesh) -> dict[str, int]:
    model_name = gmsh.model.getCurrent()
    views: dict[str, int] = {}
    for name in CELL_FIELD_NAMES:
        tag = gmsh.view.add(name)
        gmsh.view.addModelData(
            tag,
            0,
            model_name,
            "ElementData",
            mesh.cell_tags,
            np.asarray(mesh.cell_fields[name], dtype=np.float64).reshape(-1, 1),
            numComponents=1,
        )
        views[name] = tag
    return views


def write_gmsh_files(
    case_dir: Path, view_tags: dict[str, int]
) -> tuple[Path, Path]:
    visual = case_dir / "ported_prism.msh"
    openfoam = case_dir / "ported_prism_openfoam.msh"
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(visual))
    gmsh.option.setNumber("PostProcessing.SaveMesh", 0)
    for tag in view_tags.values():
        gmsh.view.write(tag, str(visual), append=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(openfoam))
    for tag in view_tags.values():
        gmsh.view.write(tag, str(openfoam), append=True)
    return visual, openfoam


def _coordinate_mapping(
    read_tags: np.ndarray,
    read_points: np.ndarray,
    mesh: PrismMesh,
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
        f"{prefix}.coordinates",
        len(read_points) == len(mesh.points_m) and maximum_error <= 1.0e-14,
        {"count": len(read_points), "maximum_error_m": maximum_error},
        {"count": len(mesh.points_m), "maximum_error_m": 1.0e-14},
    )
    mapping = np.zeros(int(read_tags.max(initial=0)) + 1, dtype=np.uint64)
    mapping[read_tags[read_order].astype(np.int64)] = mesh.node_tags[expected_order]
    return mapping


def _connectivity_mapping(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    expected_keys = np.sort(expected, axis=1)
    order = np.lexsort(
        tuple(expected_keys[:, index] for index in reversed(range(expected.shape[1])))
    )
    sorted_keys = expected_keys[order]
    indices = _lookup_rows(sorted_keys, actual)
    canonical_indices = order[indices]
    if not np.array_equal(actual, expected[canonical_indices]):
        raise PortedMeshError("round-trip changed oriented element connectivity")
    return canonical_indices


def _view_by_name(name: str) -> int | None:
    for tag in gmsh.view.getTags():
        if gmsh.view.option.getString(int(tag), "Name") == name:
            return int(tag)
    return None


def validate_gmsh_round_trip(
    path: Path,
    mesh: PrismMesh,
    records: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.open(str(path))
    physical = validate_physical_groups(records, prefix)
    tri_type = int(gmsh.model.mesh.getElementType("Triangle", 1))
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    prism_type = int(gmsh.model.mesh.getElementType("Prism", 1))
    read_tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes()
    read_tags = np.asarray(read_tags_raw, dtype=np.int64)
    read_points = np.asarray(coordinates_raw, dtype=np.float64).reshape(-1, 3)
    node_mapping = _coordinate_mapping(read_tags, read_points, mesh, records, prefix)
    prism_tags_raw, prism_nodes_raw = gmsh.model.mesh.getElementsByType(
        prism_type, VOLUME_ENTITY
    )
    prism_tags = np.asarray(prism_tags_raw, dtype=np.int64)
    read_prisms = node_mapping[
        np.asarray(prism_nodes_raw, dtype=np.int64).reshape(-1, 6)
    ]
    canonical_cell_indices = _connectivity_mapping(read_prisms, mesh.prisms)
    require(
        records,
        f"{prefix}.Prism6_connectivity",
        len(read_prisms) == len(mesh.prisms),
        len(read_prisms),
        len(mesh.prisms),
    )
    for name, expected in mesh.boundary_triangles.items():
        _tags, nodes = gmsh.model.mesh.getElementsByType(
            tri_type, SURFACE_ENTITIES[name]
        )
        actual = node_mapping[np.asarray(nodes, dtype=np.int64).reshape(-1, 3)]
        _connectivity_mapping(actual, expected)
        require(
            records,
            f"{prefix}.{name}_Tri3_connectivity",
            len(actual) == len(expected),
            len(actual),
            len(expected),
        )
    for name, expected in mesh.boundary_quads.items():
        _tags, nodes = gmsh.model.mesh.getElementsByType(
            quad_type, SURFACE_ENTITIES[name]
        )
        actual = node_mapping[np.asarray(nodes, dtype=np.int64).reshape(-1, 4)]
        _connectivity_mapping(actual, expected)
        require(
            records,
            f"{prefix}.{name}_Quad4_connectivity",
            len(actual) == len(expected),
            len(actual),
            len(expected),
        )
    element_types, element_tags, _nodes = gmsh.model.mesh.getElements()
    counts = {
        int(kind): len(tags) for kind, tags in zip(element_types, element_tags)
    }
    expected_counts = {
        tri_type: sum(len(value) for value in mesh.boundary_triangles.values()),
        quad_type: sum(len(value) for value in mesh.boundary_quads.values()),
        prism_type: len(mesh.prisms),
    }
    require(
        records,
        f"{prefix}.no_unexpected_elements",
        counts == expected_counts,
        counts,
        expected_counts,
    )
    region_view = _view_by_name("region_id")
    require(
        records,
        f"{prefix}.region_id_view_present",
        region_view is not None,
        region_view,
        "ElementData region_id",
    )
    assert region_view is not None
    data_type, data_tags, data, _time, components = gmsh.view.getModelData(region_view, 0)
    data_tags = np.asarray(data_tags, dtype=np.int64)
    data_values = np.asarray(data, dtype=np.float64).reshape(-1)
    tag_to_row = {int(tag): index for index, tag in enumerate(prism_tags)}
    read_rows = np.asarray([tag_to_row[int(tag)] for tag in data_tags], dtype=np.int64)
    expected_region = mesh.cell_fields["region_id"][canonical_cell_indices[read_rows]]
    require(
        records,
        f"{prefix}.region_id_round_trip",
        data_type == "ElementData"
        and components == 1
        and np.array_equal(data_values.astype(np.int32), expected_region),
        {"type": data_type, "components": components, "count": len(data_values)},
        {"type": "ElementData", "components": 1, "count": len(mesh.prisms)},
    )
    mapped_prisms = mesh.prisms[canonical_cell_indices]
    tri_faces = mapped_prisms[:, PRISM_TRI_FACES].reshape(-1, 3)
    tri_unique, tri_counts, _first, _second = _face_census(
        tri_faces, np.repeat(np.arange(len(mapped_prisms)), 2)
    )
    mouth_indices = _lookup_rows(tri_unique, mesh.mouth_triangles)
    require(
        records,
        f"{prefix}.mouth_ownership_reconstructed",
        bool(np.all(tri_counts[mouth_indices] == 2)),
        sorted(set(int(value) for value in tri_counts[mouth_indices])),
        [2],
    )
    min_det = np.asarray(
        gmsh.model.mesh.getElementQualities(prism_tags, "minDetJac"), dtype=np.float64
    )
    volumes = np.asarray(
        gmsh.model.mesh.getElementQualities(prism_tags, "volume"), dtype=np.float64
    )
    require(
        records,
        f"{prefix}.positive_quality",
        np.isfinite(min_det).all()
        and np.isfinite(volumes).all()
        and np.all(min_det > 0.0)
        and np.all(volumes > 0.0),
        {"minimum_minDetJac": float(min_det.min()), "minimum_volume_m3": float(volumes.min())},
        "finite and >0",
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "point_count": len(read_points),
        "cell_count": len(read_prisms),
        "element_type_counts": counts,
        "physical_groups": physical,
        "minimum_minDetJac": float(min_det.min()),
        "minimum_volume_m3": float(volumes.min()),
        "mouth_triangles_with_two_incident_cells": len(mouth_indices),
    }


def _boundary_arrays(mesh: PrismMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri_names = ("journal_wall", "bushing_bore_wall", "pressure_feed")
    quad_names = ("axial_end_z0", "axial_end_zL", "feed_tube_wall")
    triangles = np.concatenate([mesh.boundary_triangles[name] for name in tri_names])
    quads = np.concatenate([mesh.boundary_quads[name] for name in quad_names])
    tri_ids = np.concatenate(
        [
            np.full(len(mesh.boundary_triangles[name]), PHYSICAL_IDS[name], dtype=np.int32)
            for name in tri_names
        ]
    )
    quad_ids = np.concatenate(
        [
            np.full(len(mesh.boundary_quads[name]), PHYSICAL_IDS[name], dtype=np.int32)
            for name in quad_names
        ]
    )
    return triangles, quads, tri_ids, quad_ids


def write_vtu_files(mesh: PrismMesh, case_dir: Path) -> tuple[Path, Path]:
    volume_path = case_dir / "volume_prism.vtu"
    boundary_path = case_dir / "boundary_faces.vtu"
    meshio.write(
        volume_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[("wedge", mesh.prisms.astype(np.int64) - 1)],
            cell_data={
                name: [np.asarray(mesh.cell_fields[name])] for name in CELL_FIELD_NAMES
            },
            field_data={"fluid": np.asarray([PHYSICAL_IDS["fluid"], 3], dtype=np.int32)},
        ),
        file_format="vtu",
        binary=True,
    )
    triangles, quads, tri_ids, quad_ids = _boundary_arrays(mesh)
    meshio.write(
        boundary_path,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[
                ("triangle", triangles.astype(np.int64) - 1),
                ("quad", quads.astype(np.int64) - 1),
            ],
            cell_data={"patch_id": [tri_ids, quad_ids]},
            field_data={
                name: np.asarray([PHYSICAL_IDS[name], 2], dtype=np.int32)
                for name in SURFACE_ENTITIES
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return volume_path, boundary_path


def validate_vtu_round_trip(
    mesh: PrismMesh,
    volume_path: Path,
    boundary_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    volume = meshio.read(volume_path)
    require(
        records,
        "round_trip.VTU.volume_coordinates",
        volume.points.dtype == np.float64 and np.array_equal(volume.points, mesh.points_m),
        {"dtype": str(volume.points.dtype), "count": len(volume.points)},
        {"dtype": "float64", "count": len(mesh.points_m)},
    )
    wedge_blocks = [block.data for block in volume.cells if block.type == "wedge"]
    require(
        records,
        "round_trip.VTU.only_Prism6",
        len(wedge_blocks) == 1
        and len(volume.cells) == 1
        and np.array_equal(wedge_blocks[0], mesh.prisms.astype(np.int64) - 1),
        [(block.type, len(block.data)) for block in volume.cells],
        [("wedge", len(mesh.prisms))],
    )
    field_checks = {
        name: np.array_equal(
            np.asarray(volume.cell_data_dict[name]["wedge"]), mesh.cell_fields[name]
        )
        for name in CELL_FIELD_NAMES
    }
    require(
        records,
        "round_trip.VTU.cell_fields",
        all(field_checks.values()),
        field_checks,
        "all exact",
    )
    boundary = meshio.read(boundary_path)
    triangles, quads, tri_ids, quad_ids = _boundary_arrays(mesh)
    read_triangles = np.concatenate(
        [block.data for block in boundary.cells if block.type == "triangle"]
    )
    read_quads = np.concatenate([block.data for block in boundary.cells if block.type == "quad"])
    patch_data = boundary.cell_data_dict["patch_id"]
    require(
        records,
        "round_trip.VTU.boundary_faces_patch_ids",
        np.array_equal(read_triangles, triangles.astype(np.int64) - 1)
        and np.array_equal(read_quads, quads.astype(np.int64) - 1)
        and np.array_equal(patch_data["triangle"], tri_ids)
        and np.array_equal(patch_data["quad"], quad_ids),
        {
            "triangles": len(read_triangles),
            "quads": len(read_quads),
            "patch_ids": sorted(
                set(patch_data["triangle"].tolist() + patch_data["quad"].tolist())
            ),
        },
        {
            "triangles": len(triangles),
            "quads": len(quads),
            "patch_ids": sorted(PHYSICAL_IDS[name] for name in SURFACE_ENTITIES),
        },
    )
    return {
        "volume_sha256": sha256_file(volume_path),
        "boundary_sha256": sha256_file(boundary_path),
        "points": len(volume.points),
        "prisms": len(read_triangles) * 0 + len(wedge_blocks[0]),
        "boundary_triangles": len(read_triangles),
        "boundary_quads": len(read_quads),
    }


def write_mesh_arrays(mesh: PrismMesh, path: Path) -> None:
    arrays: dict[str, Any] = {
        "points_m": mesh.points_m,
        "prisms": mesh.prisms,
        "mouth_triangles": mesh.mouth_triangles,
        "cell_tags": mesh.cell_tags,
        "node_tags": mesh.node_tags,
        "cell_centres_m": mesh.cell_centres_m,
        "metadata_json": np.asarray(json.dumps(dict(mesh.metadata), sort_keys=True)),
    }
    arrays.update(
        {f"boundary_tri_{name}": value for name, value in mesh.boundary_triangles.items()}
    )
    arrays.update(
        {f"boundary_quad_{name}": value for name, value in mesh.boundary_quads.items()}
    )
    arrays.update({f"field_{name}": value for name, value in mesh.cell_fields.items()})
    np.savez_compressed(path, **arrays)


def validate_npz_round_trip(
    mesh: PrismMesh, path: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        checks = {
            "points_m": np.array_equal(archive["points_m"], mesh.points_m),
            "prisms": np.array_equal(archive["prisms"], mesh.prisms),
            "mouth_triangles": np.array_equal(
                archive["mouth_triangles"], mesh.mouth_triangles
            ),
            "cell_tags": np.array_equal(archive["cell_tags"], mesh.cell_tags),
            "node_tags": np.array_equal(archive["node_tags"], mesh.node_tags),
            "cell_centres_m": np.array_equal(
                archive["cell_centres_m"], mesh.cell_centres_m
            ),
            "metadata": json.loads(str(archive["metadata_json"])) == mesh.metadata,
        }
        checks.update(
            {
                f"boundary_tri_{name}": np.array_equal(
                    archive[f"boundary_tri_{name}"], value
                )
                for name, value in mesh.boundary_triangles.items()
            }
        )
        checks.update(
            {
                f"boundary_quad_{name}": np.array_equal(
                    archive[f"boundary_quad_{name}"], value
                )
                for name, value in mesh.boundary_quads.items()
            }
        )
        checks.update(
            {
                f"field_{name}": np.array_equal(archive[f"field_{name}"], value)
                for name, value in mesh.cell_fields.items()
            }
        )
    require(records, "round_trip.NPZ.exact_arrays", all(checks.values()), checks, "all exact")
    return {"sha256": sha256_file(path), "checks": checks}


def write_physical_groups(path: Path) -> dict[str, Any]:
    data = {
        "coordinate_unit": "m",
        "volume": {
            "fluid": {
                "physical_id": PHYSICAL_IDS["fluid"],
                "entity_tag": VOLUME_ENTITY,
                "moving": False,
            }
        },
        "boundaries": {
            name: {
                "physical_id": PHYSICAL_IDS[name],
                "entity_tag": SURFACE_ENTITIES[name],
                "moving_wall": name == "journal_wall",
                "stationary_wall": name in {"bushing_bore_wall", "feed_tube_wall"},
            }
            for name in SURFACE_ENTITIES
        },
        "forbidden_solver_patches_absent": [
            "feed_mouth",
            "mouth_cap",
            "internal_feed",
            "defaultFaces",
        ],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def write_surface_quality(mesh: PrismMesh, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, faces in mesh.boundary_triangles.items():
        points = mesh.points_m[faces.astype(np.int64) - 1]
        areas = np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
        ) / 2.0
        rows.append(
            {
                "boundary": name,
                "physical_id": PHYSICAL_IDS[name],
                "face_type": "Tri3",
                "face_count": len(faces),
                "area_m2": float(areas.sum()),
                "minimum_face_area_m2": float(areas.min()),
                "maximum_face_area_m2": float(areas.max()),
            }
        )
    for name, faces in mesh.boundary_quads.items():
        points = mesh.points_m[faces.astype(np.int64) - 1]
        total_areas = np.zeros(len(faces), dtype=np.float64)
        signs = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        for r in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)):
            for s in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)):
                derivative_r = 0.25 * signs[:, 0] * (1.0 + s * signs[:, 1])
                derivative_s = 0.25 * signs[:, 1] * (1.0 + r * signs[:, 0])
                tangent_r = np.einsum("n,mnc->mc", derivative_r, points)
                tangent_s = np.einsum("n,mnc->mc", derivative_s, points)
                total_areas += np.linalg.norm(np.cross(tangent_r, tangent_s), axis=1)
        rows.append(
            {
                "boundary": name,
                "physical_id": PHYSICAL_IDS[name],
                "face_type": "Quad4",
                "face_count": len(faces),
                "area_m2": float(total_areas.sum()),
                "minimum_face_area_m2": float(total_areas.min()),
                "maximum_face_area_m2": float(total_areas.max()),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _sample_indices(count: int, maximum: int = 512) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=np.int64))


def validate_nodes_against_brep(
    mesh: PrismMesh,
    master: MasterMesh,
    params: PortedParams,
    brep: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.model.add("native_brep_nodal_probe")
    configure_occ_options(records, "brep_probe", 1.0)
    volumes = gmsh_import_brep(brep)
    require(records, "brep_probe.one_volume", len(volumes) == 1, len(volumes), 1)
    surfaces = inventory_surfaces({volumes[0].tag: "fluid"})
    classified: dict[str, int] = {}
    for surface in surfaces:
        if surface.entity_type == "Cone":
            centre_xy = np.asarray(surface.properties[:2], dtype=np.float64)
            journal_distance = np.linalg.norm(
                centre_xy - np.asarray([params.ex_mm, params.ey_mm])
            )
            bore_distance = np.linalg.norm(centre_xy)
            role = "journal_wall" if journal_distance < bore_distance else "bushing_bore_wall"
        elif surface.entity_type == "Cylinder":
            role = "feed_tube_wall"
        elif surface.entity_type == "Plane":
            normal = np.asarray(surface.properties[:3], dtype=np.float64)
            if abs(normal[2]) > 0.9:
                role = "axial_end_z0" if abs(surface.centre[2]) < params.length_mm / 2.0 else "axial_end_zL"
            elif abs(normal[1]) > 0.9:
                role = "pressure_feed"
            else:
                raise PortedMeshError(f"unrecognized BREP plane surface {surface.tag}")
        else:
            raise PortedMeshError(
                f"unexpected native BREP surface type {surface.entity_type!r}"
            )
        if role in classified:
            raise PortedMeshError(f"multiple native BREP surfaces classified as {role}")
        classified[role] = surface.tag
    require(
        records,
        "brep_probe.six_surfaces_classified",
        set(classified) == set(SURFACE_ENTITIES),
        classified,
        sorted(SURFACE_ENTITIES),
    )
    diagnostics: dict[str, Any] = {}
    for name in SURFACE_ENTITIES:
        if name in mesh.boundary_triangles:
            nodes = np.unique(mesh.boundary_triangles[name]) - 1
        else:
            nodes = np.unique(mesh.boundary_quads[name]) - 1
        sampled_nodes = nodes[_sample_indices(len(nodes))].astype(np.int64)
        source_mm = mesh.points_m[sampled_nodes] / SI_PER_MM
        closest, _parameters = gmsh.model.getClosestPoint(
            2, classified[name], source_mm.ravel()
        )
        closest_array = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
        residuals = np.linalg.norm(closest_array - source_mm, axis=1)
        maximum = float(residuals.max(initial=0.0))
        require(
            records,
            f"brep_probe.{name}.node_distance_mm",
            maximum <= BREP_NODE_TOL_MM,
            {"sample_count": len(sampled_nodes), "maximum_mm": maximum},
            BREP_NODE_TOL_MM,
        )
        diagnostics[name] = {
            "surface_tag": classified[name],
            "sample_count": len(sampled_nodes),
            "maximum_node_distance_mm": maximum,
        }
    rim_nodes = _film_node_tags(
        master.rim_nodes, int(mesh.metadata["n_gap"]), int(mesh.metadata["master_node_count"])
    )
    rim_points = mesh.points_m[rim_nodes.astype(np.int64) - 1] / SI_PER_MM
    closest, _parameters = gmsh.model.getClosestPoint(
        2, classified["bushing_bore_wall"], rim_points.ravel()
    )
    rim_bore_residual = np.linalg.norm(
        np.asarray(closest, dtype=np.float64).reshape(-1, 3) - rim_points, axis=1
    )
    require(
        records,
        "brep_probe.mouth_rim_on_bore_nodes",
        float(rim_bore_residual.max()) <= BREP_NODE_TOL_MM,
        float(rim_bore_residual.max()),
        BREP_NODE_TOL_MM,
    )
    diagnostics["mouth_rim_on_bore"] = {
        "sample_count": len(rim_nodes),
        "maximum_node_distance_mm": float(rim_bore_residual.max()),
    }
    return diagnostics


def _compact_cells(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(cells)
    mapping = np.full(int(used.max(initial=-1)) + 1, -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    return points[used], mapping[cells]


def _write_feed_boundary_msh(mesh: PrismMesh, path: Path) -> None:
    pressure = mesh.boundary_triangles["pressure_feed"].astype(np.int64) - 1
    wall = mesh.boundary_quads["feed_tube_wall"].astype(np.int64) - 1
    all_cells = np.concatenate([pressure.ravel(), wall.ravel()])
    used = np.unique(all_cells)
    mapping = np.full(len(mesh.points_m), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    meshio.write(
        path,
        meshio.Mesh(
            points=mesh.points_m[used],
            cells=[("triangle", mapping[pressure]), ("quad", mapping[wall])],
            cell_data={
                "gmsh:physical": [
                    np.full(len(pressure), PHYSICAL_IDS["pressure_feed"], dtype=np.int32),
                    np.full(len(wall), PHYSICAL_IDS["feed_tube_wall"], dtype=np.int32),
                ],
                "gmsh:geometrical": [
                    np.full(len(pressure), SURFACE_ENTITIES["pressure_feed"], dtype=np.int32),
                    np.full(len(wall), SURFACE_ENTITIES["feed_tube_wall"], dtype=np.int32),
                ],
                "solve_eligible": [
                    np.zeros(len(pressure), dtype=np.float64),
                    np.zeros(len(wall), dtype=np.float64),
                ],
            },
            field_data={
                "pressure_feed": np.asarray([PHYSICAL_IDS["pressure_feed"], 2]),
                "feed_tube_wall": np.asarray([PHYSICAL_IDS["feed_tube_wall"], 2]),
            },
        ),
        file_format="gmsh22",
        binary=True,
    )


def _write_cutaway(mesh: PrismMesh, msh_path: Path, vtu_path: Path) -> dict[str, Any]:
    theta = np.radians(mesh.cell_fields["theta_deg"])
    keep = (mesh.cell_fields["region_id"] == 1) | (
        (theta >= math.pi / 6.0) & (theta <= 2.0 * math.pi - math.pi / 6.0)
    )
    prisms = mesh.prisms[keep].astype(np.int64) - 1
    used = np.unique(prisms)
    mapping = np.full(len(mesh.points_m), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    cells = mapping[prisms]
    fields = {
        name: [
            np.zeros(int(keep.sum()), dtype=np.int32)
            if name == "solve_eligible"
            else np.asarray(mesh.cell_fields[name])[keep]
        ]
        for name in CELL_FIELD_NAMES
    }
    fields["diagnostic_only"] = [np.ones(int(keep.sum()), dtype=np.int32)]
    diagnostic = meshio.Mesh(
        points=mesh.points_m[used],
        cells=[("wedge", cells)],
        cell_data=fields,
        field_data={
            "DIAGNOSTIC_CUTAWAY_DO_NOT_SOLVE": np.asarray([301, 3], dtype=np.int32)
        },
    )
    meshio.write(vtu_path, diagnostic, file_format="vtu", binary=True)
    gmsh_fields = {
        name: [np.asarray(values[0], dtype=np.float64)]
        for name, values in fields.items()
    }
    gmsh_fields["gmsh:physical"] = [np.full(len(cells), 301, dtype=np.int32)]
    gmsh_fields["gmsh:geometrical"] = [np.full(len(cells), 301, dtype=np.int32)]
    meshio.write(
        msh_path,
        meshio.Mesh(
            points=mesh.points_m[used],
            cells=[("wedge", cells)],
            cell_data=gmsh_fields,
            field_data={
                "DIAGNOSTIC_CUTAWAY_DO_NOT_SOLVE": np.asarray([301, 3], dtype=np.int32)
            },
        ),
        file_format="gmsh22",
        binary=True,
    )
    return {
        "cell_count": int(keep.sum()),
        "omitted_cell_count": int((~keep).sum()),
        "wedge_removed_around_theta_deg": [-30.0, 30.0],
        "solve_eligible": False,
        "coordinates_unchanged": True,
    }


def _write_mouth_diagnostic(mesh: PrismMesh, path: Path) -> dict[str, Any]:
    triangles = mesh.mouth_triangles.astype(np.int64) - 1
    used = np.unique(triangles)
    mapping = np.full(len(mesh.points_m), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    meshio.write(
        path,
        meshio.Mesh(
            points=mesh.points_m[used],
            cells=[("triangle", mapping[triangles])],
            cell_data={
                "solve_eligible": [np.zeros(len(triangles), dtype=np.int32)],
                "diagnostic_only": [np.ones(len(triangles), dtype=np.int32)],
                "mouth_interface": [np.ones(len(triangles), dtype=np.int32)],
                "red": [np.ones(len(triangles), dtype=np.float64)],
                "red_rgb": [
                    np.tile(np.asarray([[255, 0, 0]], dtype=np.uint8), (len(triangles), 1))
                ],
            },
            field_data={"mouth_interface_DIAGNOSTIC_ONLY": np.asarray([401, 2])},
        ),
        file_format="vtu",
        binary=True,
    )
    return {
        "triangle_count": len(triangles),
        "solve_eligible": False,
        "diagnostic_only": True,
        "default_representation": "wireframe",
    }


def _write_x0_section(mesh: PrismMesh, path: Path) -> dict[str, Any]:
    points = mesh.points_m
    cells = mesh.prisms.astype(np.int64) - 1
    cell_points = points[cells]
    tolerance = 1.0e-14
    candidate_mask = (cell_points[:, :, 0].min(axis=1) <= tolerance) & (
        cell_points[:, :, 0].max(axis=1) >= -tolerance
    )
    candidates = np.flatnonzero(candidate_mask)
    section_points: list[tuple[float, float, float]] = []
    point_lookup: dict[tuple[int, int], int] = {}
    triangles: list[tuple[int, int, int]] = []
    regions: list[int] = []
    gaps: list[int] = []
    triangle_keys: set[tuple[int, int, int]] = set()

    def section_node(point: np.ndarray) -> int:
        key = (int(round(point[1] / 1.0e-12)), int(round(point[2] / 1.0e-12)))
        if key not in point_lookup:
            point_lookup[key] = len(section_points)
            section_points.append((0.0, float(point[1]), float(point[2])))
        return point_lookup[key]

    for cell_index in candidates:
        vertices = cell_points[cell_index]
        intersections: list[np.ndarray] = []
        for left, right in PRISM_EDGES:
            a, b = vertices[left], vertices[right]
            xa, xb = a[0], b[0]
            if abs(xa) <= tolerance:
                intersections.append(a.copy())
            if abs(xb) <= tolerance:
                intersections.append(b.copy())
            if xa * xb < -tolerance**2:
                fraction = -xa / (xb - xa)
                intersections.append(a + fraction * (b - a))
        unique: dict[tuple[int, int], np.ndarray] = {}
        for point in intersections:
            key = (int(round(point[1] / 1.0e-12)), int(round(point[2] / 1.0e-12)))
            unique[key] = point
        polygon = list(unique.values())
        if len(polygon) < 3:
            continue
        centre = np.mean(polygon, axis=0)
        polygon.sort(key=lambda point: math.atan2(point[2] - centre[2], point[1] - centre[1]))
        node_ids = [section_node(point) for point in polygon]
        for index in range(1, len(node_ids) - 1):
            tri = (node_ids[0], node_ids[index], node_ids[index + 1])
            key = tuple(sorted(tri))
            if key in triangle_keys:
                continue
            triangle_keys.add(key)
            triangles.append(tri)
            regions.append(int(mesh.cell_fields["region_id"][cell_index]))
            gaps.append(int(mesh.cell_fields["gap_layer_index"][cell_index]))
    if not triangles:
        raise PortedMeshError("x=0 section produced no triangles")
    meshio.write(
        path,
        meshio.Mesh(
            points=np.asarray(section_points, dtype=np.float64),
            cells=[("triangle", np.asarray(triangles, dtype=np.int64))],
            cell_data={
                "region_id": [np.asarray(regions, dtype=np.int32)],
                "gap_layer_index": [np.asarray(gaps, dtype=np.int32)],
                "solve_eligible": [np.zeros(len(triangles), dtype=np.int32)],
                "diagnostic_only": [np.ones(len(triangles), dtype=np.int32)],
            },
        ),
        file_format="vtu",
        binary=True,
    )
    return {
        "point_count": len(section_points),
        "triangle_count": len(triangles),
        "solve_eligible": False,
    }


def _mouth_plot(
    master: MasterMesh, params: PortedParams, path: Path
) -> None:
    disk_triangles = master.triangles[master.disk_triangle_mask]
    disk_nodes = np.unique(disk_triangles)
    theta = master.points_uz_mm[disk_nodes, 0] / params.mean_radius_mm
    z = master.points_uz_mm[disk_nodes, 1]
    x = np.asarray(params.bore_radius_mm(z)) * np.sin(theta)
    lookup = np.full(len(master.points_uz_mm), -1, dtype=np.int64)
    lookup[disk_nodes] = np.arange(len(disk_nodes))
    rim = rim_coordinates(params, len(master.rim_nodes))
    alpha = np.linspace(0.0, 2.0 * math.pi, 721)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.triplot(x, z, lookup[disk_triangles], color="0.65", linewidth=0.45, label="feed-disk Tri3")
    axis.plot(
        params.hole_radius_mm * np.cos(alpha),
        params.hole_axial_pos_mm + params.hole_radius_mm * np.sin(alpha),
        "k--",
        linewidth=1.2,
        label="analytic circle",
    )
    axis.plot(
        np.r_[rim["x"], rim["x"][0]],
        np.r_[rim["z"], rim["z"][0]],
        color="tab:red",
        linewidth=1.0,
        label="N-sided chord rim",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title("Shared feed-film interface: analytic rim nodes and linear Tri3 chords")
    axis.text(
        0.02,
        0.02,
        f"sagitta = {master.metadata['rim_sagitta_mm']:.9f} mm\n"
        f"polygon/circle area error = {master.metadata['rim_polygon_circle_relative_error']:.3e}",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.85},
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _x0_feed_plot(mesh: PrismMesh, params: PortedParams, path: Path) -> None:
    points = mesh.points_m / SI_PER_MM
    centreline = points[np.asarray(mesh.metadata["centreline_node_tags"], dtype=np.int64) - 1]
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(centreline[:, 1], centreline[:, 2], "o-", markersize=3, label="layered feed centreline")
    z = np.linspace(params.hole_axial_pos_mm - params.hole_radius_mm, params.hole_axial_pos_mm + params.hole_radius_mm, 300)
    bore_y = np.asarray(params.bore_radius_mm(z))
    axis.fill_betweenx(z, bore_y, params.y_feed_end_mm, color="tab:blue", alpha=0.15, label="feed passage at x=0")
    axis.axvline(params.y_feed_end_mm, color="tab:red", label="pressure inlet")
    axis.set_xlabel("y [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title("Exact x=0 feed-axis section (true coordinates)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _quality_plot(mesh: PrismMesh, path: Path) -> None:
    fields = ("minSICN", "minDetJac", "aspect_ratio", "volume_m3")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    regions = mesh.cell_fields["region_id"]
    for axis, name in zip(axes.ravel(), fields):
        values = mesh.cell_fields[name]
        for region, label in ((0, "film"), (1, "feed")):
            subset = values[regions == region]
            axis.hist(subset, bins=60, alpha=0.55, label=label)
        axis.set_title(name)
        axis.set_yscale("log")
        axis.legend()
    fig.suptitle("Prism6 quality distributions by region")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_paraview_helpers(viz: Path) -> dict[str, Any]:
    python_path = viz / "open_in_paraview.py"
    python_path.write_text(
        '''#!/usr/bin/env python3
from pathlib import Path
import argparse

from paraview.simple import Clip, ColorBy, GetActiveViewOrCreate, OpenDataFile, Render, ResetCamera, SaveScreenshot, Show

parser = argparse.ArgumentParser()
parser.add_argument("--screenshot", type=Path)
args = parser.parse_args()
case = Path(__file__).resolve().parent.parent
mesh = OpenDataFile(str(case / "volume_prism.vtu"))
view = GetActiveViewOrCreate("RenderView")
display = Show(mesh, view)
display.Representation = "Surface With Edges"
ColorBy(display, ("CELLS", "region_id"))
clip = Clip(Input=mesh)
clip.ClipType = "Plane"
clip.ClipType.Origin = [0.0, 0.0, 0.0]
clip.ClipType.Normal = [1.0, 0.0, 0.0]
clip_display = Show(clip, view)
clip_display.Representation = "Surface With Edges"
ColorBy(clip_display, ("CELLS", "region_id"))
display.Visibility = 0
mouth = OpenDataFile(str(case / "viz" / "mouth_interface_DIAGNOSTIC_ONLY.vtu"))
mouth_display = Show(mouth, view)
mouth_display.Representation = "Wireframe"
mouth_display.Opacity = 0.35
mouth_display.DiffuseColor = [1.0, 0.0, 0.0]
mouth_display.AmbientColor = [1.0, 0.0, 0.0]
ResetCamera(view)
Render(view)
if args.screenshot:
    SaveScreenshot(str(args.screenshot.resolve()), view)
''',
        encoding="utf-8",
    )
    launcher = viz / "launch_paraview_xcb.sh"
    launcher.write_text(
        '''#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec env QT_QPA_PLATFORM=xcb paraview "$case_dir/volume_prism.vtu" "$@"
''',
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    readme = viz / "README_VISUALIZE.md"
    readme.write_text(
        '''# Inspect the exact ported Prism6 mesh

```bash
QT_QPA_PLATFORM=xcb paraview <case>/volume_prism.vtu
```

1. Select `volume_prism.vtu` in the Pipeline Browser.
2. Click the green **Apply** button.
3. Press **R** to reset the camera.
4. Choose **Surface With Edges** in Representation.
5. Color by `region_id`, `gap_layer_index`, `gap_um`, or `minSICN`.
6. Apply **Filters > Clip** with origin `x=0` and normal `(1,0,0)`.
7. Optionally apply **Shrink** to inspect individual Prism6 cells.

`feed_cutaway_exact.*`, `x0_section.vtu`, and `mouth_interface_DIAGNOSTIC_ONLY.vtu`
are unchanged-coordinate diagnostic subsets and are not solver meshes. Run
`pvpython viz/open_in_paraview.py` for an automatic x=0 clip; add
`--screenshot image.png` to save a render. The helper overlays the shared
feed-film interface as a translucent red wireframe, never as a physical cap.
''',
        encoding="utf-8",
    )
    return {
        "pvpython": shutil.which("pvpython"),
        "status": "AVAILABLE" if shutil.which("pvpython") else "SKIPPED",
        "reason": None if shutil.which("pvpython") else "pvpython unavailable",
    }


def write_visualizations(
    mesh: PrismMesh,
    master: MasterMesh,
    params: PortedParams,
    case_dir: Path,
    master_surface_path: Path,
) -> dict[str, Any]:
    viz = case_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    shutil.copy2(master_surface_path, viz / "master_surface_2d.msh")
    shutil.copy2(case_dir / "boundary_faces.vtu", viz / "full_boundary_colored.vtu")
    _write_feed_boundary_msh(mesh, viz / "feed_boundary_only.msh")
    cutaway = _write_cutaway(
        mesh, viz / "feed_cutaway_exact.msh", viz / "feed_cutaway_exact.vtu"
    )
    section = _write_x0_section(mesh, viz / "x0_section.vtu")
    mouth = _write_mouth_diagnostic(
        mesh, viz / "mouth_interface_DIAGNOSTIC_ONLY.vtu"
    )
    _mouth_plot(master, params, viz / "mouth_footprint.png")
    _x0_feed_plot(mesh, params, viz / "x0_feed_section.png")
    _quality_plot(mesh, viz / "quality_histograms.png")
    paraview = _write_paraview_helpers(viz)
    manifest = {
        "solve_eligible": False,
        "distorted_geometry": False,
        "coordinates_unchanged": True,
        "cutaway": cutaway,
        "x0_section": section,
        "mouth_interface": mouth,
        "paraview": paraview,
    }
    (viz / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_visualization_msh(
    mesh: PrismMesh,
    viz: Path,
    visualization: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.open(str(viz / "feed_cutaway_exact.msh"))
    prism_type = int(gmsh.model.mesh.getElementType("Prism", 1))
    cutaway_tags, _cutaway_nodes = gmsh.model.mesh.getElementsByType(prism_type)
    solve_view = _view_by_name("solve_eligible")
    solve_values = np.asarray([], dtype=np.float64)
    if solve_view is not None:
        _kind, _tags, data, _time, _components = gmsh.view.getModelData(solve_view, 0)
        solve_values = np.asarray(data, dtype=np.float64).reshape(-1)
    expected_cutaway = int(visualization["cutaway"]["cell_count"])
    require(
        records,
        "visualization.cutaway_MSH_reopens",
        len(cutaway_tags) == expected_cutaway
        and solve_view is not None
        and len(solve_values) == expected_cutaway
        and np.all(solve_values == 0.0),
        {
            "Prism6": len(cutaway_tags),
            "solve_eligible_count": len(solve_values),
            "solve_eligible_values": np.unique(solve_values).tolist(),
        },
        {"Prism6": expected_cutaway, "solve_eligible": [0.0]},
    )
    gmsh.clear()
    gmsh.open(str(viz / "feed_boundary_only.msh"))
    tri_type = int(gmsh.model.mesh.getElementType("Triangle", 1))
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    tri_tags, _tri_nodes = gmsh.model.mesh.getElementsByType(tri_type)
    quad_tags, _quad_nodes = gmsh.model.mesh.getElementsByType(quad_type)
    prism_tags, _prism_nodes = gmsh.model.mesh.getElementsByType(prism_type)
    require(
        records,
        "visualization.feed_boundary_MSH_reopens",
        len(tri_tags) == len(mesh.boundary_triangles["pressure_feed"])
        and len(quad_tags) == len(mesh.boundary_quads["feed_tube_wall"])
        and len(prism_tags) == 0,
        {"Tri3": len(tri_tags), "Quad4": len(quad_tags), "Prism6": len(prism_tags)},
        {
            "Tri3": len(mesh.boundary_triangles["pressure_feed"]),
            "Quad4": len(mesh.boundary_quads["feed_tube_wall"]),
            "Prism6": 0,
        },
    )
    return {
        "cutaway_prisms": len(cutaway_tags),
        "feed_boundary_triangles": len(tri_tags),
        "feed_boundary_quads": len(quad_tags),
    }


def audit_openfoam(
    mode: str,
    case_dir: Path,
    msh_path: Path,
    mesh: PrismMesh,
    records: list[dict[str, Any]],
    published_case_dir: Path,
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
                "expected": "optional audited Prism6 conversion",
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
        "both available",
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
    conversion_text = converter["stdout"] + "\n" + converter["stderr"]
    (case_dir / "openfoam_conversion.log").write_text(conversion_text, encoding="utf-8")
    rejected_conversion = (
        "unhandled element",
        "inverting",
        "undefined faces",
        "could not match gmsh face",
        "foam fatal",
        "foam exiting",
        "segmentation fault",
    )
    require(
        records,
        "openfoam.gmshToFoam_Prism6",
        converter["returncode"] == 0
        and not any(token in conversion_text.lower() for token in rejected_conversion),
        {
            "returncode": converter["returncode"],
            "rejected_messages": [
                token for token in rejected_conversion if token in conversion_text.lower()
            ],
        },
        {"returncode": 0, "rejected_messages": []},
    )
    checker = _run_command(
        [check_mesh, "-case", str(foam_case), "-allTopology", "-allGeometry"]
    )
    checker_text = checker["stdout"] + "\n" + checker["stderr"]
    (case_dir / "openfoam_checkMesh.log").write_text(checker_text, encoding="utf-8")
    rejected_check = (
        "negative volume",
        "illegal",
        "inverted",
        "foam fatal",
        "foam exiting",
    )
    require(
        records,
        "openfoam.checkMesh",
        checker["returncode"] == 0
        and re.search(r"(?m)^\s*Mesh OK\.\s*$", checker_text) is not None
        and re.search(r"Failed\s+[1-9]\d*\s+mesh checks", checker_text) is None
        and not any(token in checker_text.lower() for token in rejected_check),
        {
            "returncode": checker["returncode"],
            "mesh_ok": "Mesh OK." in checker_text,
            "rejected_messages": [
                token for token in rejected_check if token in checker_text.lower()
            ],
        },
        {"returncode": 0, "mesh_ok": True, "rejected_messages": []},
    )
    cell_match = re.search(r"(?m)^\s*cells:\s*(\d+)\s*$", checker_text)
    region_match = re.search(r"Number of regions:\s*(\d+)", checker_text)
    cells = int(cell_match.group(1)) if cell_match else None
    regions = int(region_match.group(1)) if region_match else None
    require(records, "openfoam.cell_count", cells == len(mesh.prisms), cells, len(mesh.prisms))
    require(records, "openfoam.one_region", regions == 1, regions, 1)
    boundary_path = foam_case / "constant" / "polyMesh" / "boundary"
    require(
        records,
        "openfoam.boundary_file",
        boundary_path.is_file() and boundary_path.stat().st_size > 0,
        str(boundary_path),
        "nonempty",
    )
    patches = _openfoam_boundary_patches(boundary_path)
    expected_names = sorted(SURFACE_ENTITIES)
    require(
        records,
        "openfoam.six_exact_patches_no_defaultFaces",
        sorted(patches) == expected_names
        and "defaultFaces" not in patches
        and all(data["type"] in {"patch", "wall"} for data in patches.values()),
        patches,
        expected_names,
    )
    foam_version = shutil.which("foamVersion")
    published_foam_case = published_case_dir / "openfoam_case"
    return {
        "status": "PASS",
        "executables": {"gmshToFoam": gmsh_to_foam, "checkMesh": check_mesh},
        "versions": {
            "foamVersion": _run_command([foam_version]) if foam_version else None,
            "gmshToFoam_help": _run_command([gmsh_to_foam, "-help"]),
            "checkMesh_help": _run_command([check_mesh, "-help"]),
        },
        "conversion": converter,
        "checkMesh": checker,
        "patches": patches,
        "cells": cells,
        "regions": regions,
        "published_commands": {
            "gmshToFoam": [
                gmsh_to_foam,
                "-case",
                str(published_foam_case),
                str(published_case_dir / "ported_prism_openfoam.msh"),
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


def quality_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def grouped_quality(mesh: PrismMesh) -> dict[str, Any]:
    region = mesh.cell_fields["region_id"]
    mouth_nodes = np.unique(mesh.mouth_triangles)
    touching_mouth = np.isin(mesh.prisms, mouth_nodes).any(axis=1)
    masks = {
        "all": np.ones(len(mesh.prisms), dtype=bool),
        "film": region == 0,
        "feed_tube": region == 1,
        "touching_feed_mouth": touching_mouth,
    }
    return {
        group: {
            "cell_count": int(mask.sum()),
            **{
                name: quality_statistics(mesh.cell_fields[name][mask])
                for name in ("minSICN", "minDetJac", "volume_m3", "aspect_ratio")
            },
        }
        for group, mask in masks.items()
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
    params: PortedParams,
    inputs: PortedInputs,
    master: MasterMesh,
    master_surface_path: Path,
    case_dir: Path,
    n_gap: int,
    inherited_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    records = [dict(record) for record in inherited_records]
    mesh = build_prism_mesh(
        master,
        params,
        n_gap,
        inputs.tube_layers,
        inputs.tube_grading,
        inputs.gap_inflation_ratio,
    )
    topology = validate_topology(mesh, master, records)
    geometry = validate_geometry(mesh, master, params, inputs, records)
    gmsh.clear()
    gmsh.logger.start()
    try:
        discrete = add_discrete_prism_model(
            mesh, records, f"ported_prism_nGap_{n_gap:02d}"
        )
        physical = validate_physical_groups(records, "gmsh_model")
        mesh = add_gmsh_quality(mesh, records)
        geometry["volumes"]["gmsh_element_sum_m3"] = float(
            mesh.cell_fields["volume_m3"].sum()
        )
        geometry["volumes"]["gmsh_vs_custom_relative_error"] = relative_error(
            geometry["volumes"]["gmsh_element_sum_m3"],
            geometry["volumes"]["cell_sum_m3"],
        )
        view_tags = add_element_data_views(mesh)
        msh41, msh22 = write_gmsh_files(case_dir, view_tags)
        round_trips = {
            "gmsh_4_1_binary": validate_gmsh_round_trip(
                msh41, mesh, records, "round_trip.msh41"
            ),
            "gmsh_2_2_ascii": validate_gmsh_round_trip(
                msh22, mesh, records, "round_trip.msh22"
            ),
        }
        brep_probe = validate_nodes_against_brep(
            mesh, master, params, inputs.brep, records
        )
    finally:
        gmsh_lines = [str(line) for line in gmsh.logger.get()]
        gmsh.logger.stop()
    (case_dir / "gmsh_ported.log").write_text(
        "\n".join(gmsh_lines) + "\n", encoding="utf-8"
    )
    volume_vtu, boundary_vtu = write_vtu_files(mesh, case_dir)
    vtu_round_trip = validate_vtu_round_trip(
        mesh, volume_vtu, boundary_vtu, records
    )
    npz_path = case_dir / "mesh_arrays.npz"
    write_mesh_arrays(mesh, npz_path)
    npz_round_trip = validate_npz_round_trip(mesh, npz_path, records)
    physical_json = write_physical_groups(case_dir / "physical_groups.json")
    surface_quality = write_surface_quality(mesh, case_dir / "surface_quality.csv")
    visualization = write_visualizations(
        mesh, master, params, case_dir, master_surface_path
    )
    visualization["gmsh_round_trip"] = validate_visualization_msh(
        mesh, case_dir / "viz", visualization, records
    )
    (case_dir / "viz" / "manifest.json").write_text(
        json.dumps(visualization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    published_case_dir = inputs.outdir / f"nGap_{n_gap:02d}"
    openfoam = audit_openfoam(
        inputs.openfoam,
        case_dir,
        case_dir / "ported_prism_openfoam.msh",
        mesh,
        records,
        published_case_dir,
    )
    quality = grouped_quality(mesh)
    commands = {
        "gmsh_full": f"uv run gmsh {published_case_dir / 'ported_prism.msh'}",
        "gmsh_cutaway": f"uv run gmsh {published_case_dir / 'viz' / 'feed_cutaway_exact.msh'}",
        "gmsh_feed_boundary": f"uv run gmsh {published_case_dir / 'viz' / 'feed_boundary_only.msh'}",
        "paraview_volume": f"QT_QPA_PLATFORM=xcb paraview {published_case_dir / 'volume_prism.vtu'}",
        "paraview_cutaway": f"QT_QPA_PLATFORM=xcb paraview {published_case_dir / 'viz' / 'feed_cutaway_exact.vtu'}",
        "paraview_x0": f"QT_QPA_PLATFORM=xcb paraview {published_case_dir / 'viz' / 'x0_section.vtu'}",
        "paraview_helper": f"pvpython {published_case_dir / 'viz' / 'open_in_paraview.py'}",
    }
    manifest = {
        "schema_version": 2,
        "overall": "PASS",
        "solve_eligible": True,
        "distorted_geometry": False,
        "coordinate_unit": "m",
        "source_parameter_unit": "mm",
        "scale_to_m_applied_exactly_once": SI_PER_MM,
        "full_360_degrees": True,
        "geometry": "eccentric conical film plus open central feed tube",
        "volume_cell_type": "Prism6",
        "gmsh_generated_2d_master": True,
        "gmsh_generated_3d_volume_mesh": False,
        "mouth_internal_two_incident_cell_triangles": len(mesh.mouth_triangles),
        "journal_continuous_under_feed": True,
        "physical_groups": physical_json,
        "fields": list(CELL_FIELD_NAMES),
        "field_value_maps": {
            "region_id": {"0": "film", "1": "feed"},
            "axial_zone_id": {"0": "ring_A", "1": "hole_band", "2": "ring_B"},
            "gap_layer_index": "0..nGap-1 for film; -1 for feed",
            "solve_eligible": {"1": "canonical complete solver mesh"},
            "distorted_geometry": {"0": "true coordinates"},
        },
        "counts": {
            "nodes": len(mesh.points_m),
            "film_prisms": mesh.film_cell_count,
            "feed_prisms": mesh.feed_cell_count,
            "total_prisms": len(mesh.prisms),
            "mouth_triangles": len(mesh.mouth_triangles),
            "boundary_faces": topology["boundary_face_counts"],
            "total_boundary_faces": topology["total_boundary_faces"],
        },
        "film_inflation": {
            "centre_to_wall_cell_thickness_ratio": mesh.metadata[
                "gap_inflation_ratio_achieved"
            ],
            "layer_coordinates": mesh.metadata["gap_layer_coordinates"],
            "layer_fractions": mesh.metadata["gap_layer_fractions"],
        },
        "rim": dict(master.metadata),
        "boundary_roles": {
            "moving_wall": ["journal_wall"],
            "stationary_walls": ["bushing_bore_wall", "feed_tube_wall"],
            "pressure_inlet": ["pressure_feed"],
            "axial_ends": ["axial_end_z0", "axial_end_zL"],
        },
        "diagnostic_subsets": visualization,
        "commands": commands,
        "cfd_solution_executed": False,
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
            "rim_segments": inputs.rim_segments,
            "gap_inflation_ratio": inputs.gap_inflation_ratio,
            "tube_layers": inputs.tube_layers,
            "tube_grading": inputs.tube_grading,
            "openfoam": inputs.openfoam,
        },
        "counts": manifest["counts"],
        "coordinate_units": {
            "construction": "mm",
            "solver_exports": "m",
            "volume": "m^3",
        },
        "master_surface": dict(master.metadata),
        "topology": topology,
        "geometry": geometry,
        "quality": quality,
        "native_brep_nodal_probe": brep_probe,
        "gmsh": {
            "discrete_registration": discrete,
            "physical_groups": physical,
            "round_trips": round_trips,
            "log": "gmsh_ported.log",
        },
        "vtu_round_trip": vtu_round_trip,
        "npz_round_trip": npz_round_trip,
        "surface_quality": surface_quality,
        "visualization": visualization,
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
        **manifest["counts"],
        "minimum_minSICN": quality["all"]["minSICN"]["minimum"],
        "minimum_minDetJac": quality["all"]["minDetJac"]["minimum"],
        "brep_volume_relative_error": geometry["volumes"]["native_brep_relative_error"],
        "inlet_area_relative_error": geometry["inlet"]["polygon_relative_error"],
        "mesh_volume_m3": geometry["volumes"]["cell_sum_m3"],
        "topology_status": "PASS",
        "openfoam_status": openfoam["status"],
        "overall": "PASS",
        "commands": commands,
    }


def write_convergence(
    stage: Path, params: PortedParams, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    volumes = np.asarray([result["mesh_volume_m3"] for result in results])
    relative_range = float(np.ptp(volumes) / abs(volumes.mean())) if len(volumes) > 1 else 0.0
    if relative_range > 1.0e-9:
        raise PortedMeshError(
            f"ported mesh volume changed with gap subdivision: relative range {relative_range:.3e}"
        )
    rows = [
        {
            "n_gap": result["n_gap"],
            "nodes": result["nodes"],
            "film_prisms": result["film_prisms"],
            "feed_prisms": result["feed_prisms"],
            "total_prisms": result["total_prisms"],
            "mesh_volume_m3": result["mesh_volume_m3"],
            "native_brep_volume_m3": params.native_volume_mm3 * SI_PER_MM**3,
            "brep_volume_relative_error": result["brep_volume_relative_error"],
        }
        for result in results
    ]
    with (stage / "convergence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "overall": "PASS",
        "coordinate_unit": "m",
        "gap_level_volume_relative_range": relative_range,
        "cases": rows,
    }
    (stage / "convergence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _publish_failure(
    outdir: Path, report: dict[str, Any], stage: Path | None
) -> None:
    failure_stage = make_staging_directory(outdir)
    try:
        (failure_stage / "failure_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if stage is not None and stage.exists():
            logs = failure_stage / "diagnostic_logs"
            for source in stage.rglob("*.log"):
                destination = logs / source.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        publish_generation(
            failure_stage,
            outdir,
            stage="meshing",
            operation="central-feed",
            status=str(report["overall"]),
            resolved_inputs=report.get("inputs", {}),
            input_units={"geometry": "mm", "mesh": "m"},
            producer_files=(Path(__file__),),
        )
    finally:
        if failure_stage.exists():
            shutil.rmtree(failure_stage)


def open_gui(inputs: PortedInputs) -> None:
    case = inputs.outdir / f"nGap_{inputs.preview_ngap:02d}"
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    try:
        if inputs.gui_mode == "full":
            gmsh.open(str(case / "ported_prism.msh"))
        elif inputs.gui_mode == "mouth":
            gmsh.open(str(case / "viz" / "feed_boundary_only.msh"))
        else:
            gmsh.open(str(case / "viz" / "feed_cutaway_exact.msh"))
        for view_tag in gmsh.view.getTags():
            gmsh.view.option.setNumber(int(view_tag), "Visible", 0)
        if inputs.gui_mode == "quality":
            view = _view_by_name("minSICN")
            if view is not None:
                gmsh.view.option.setNumber(view, "Visible", 1)
        for name in (
            "Mesh.SurfaceFaces",
            "Mesh.SurfaceEdges",
            "Mesh.VolumeFaces",
            "Mesh.VolumeEdges",
            "Mesh.Prisms",
        ):
            gmsh.option.setNumber(name, 1)
        gmsh.option.setNumber("Mesh.DrawSkinOnly", 0)
        gmsh.fltk.run()
    finally:
        gmsh.finalize()


def open_optional_gui(inputs: PortedInputs) -> None:
    try:
        open_gui(inputs)
    except Exception as error:
        print(
            f"WARNING: optional Gmsh GUI failed after validated mesh publication: {error}",
            file=sys.stderr,
        )


def run_ported(inputs: PortedInputs) -> dict[str, Any]:
    inputs = replace(
        inputs,
        params=inputs.params.resolve(),
        brep=inputs.brep.resolve(),
        preflight=inputs.preflight.resolve(),
        outdir=inputs.outdir.resolve(),
    )
    base_report = {
        "inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(inputs).items()
        },
        "gmsh_generated_3d_volume_mesh": False,
        "cfd_solution_executed": False,
    }
    stage: Path | None = None
    gmsh_initialized = False
    try:
        contract_records: list[dict[str, Any]] = []
        params, _raw_params, preflight = load_contract(inputs, contract_records)
        validate_inputs(inputs, params)
        if inputs.openfoam == "required" and not (
            shutil.which("gmshToFoam") and shutil.which("checkMesh")
        ):
            raise PortedMeshError(
                "--openfoam required but gmshToFoam/checkMesh are unavailable"
            )
        native_brep = validate_native_brep(
            inputs, params, preflight, contract_records
        )
        stage = make_staging_directory(inputs.outdir)
        gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        master_path = stage / "_master_surface_2d.msh"
        master = build_master_mesh(
            params, inputs, master_path, contract_records
        )
        results = [
            generate_gap_case(
                params,
                inputs,
                master,
                master_path,
                stage / f"nGap_{n_gap:02d}",
                n_gap,
                contract_records,
            )
            for n_gap in inputs.gap_levels
        ]
        master_path.unlink()
        convergence = write_convergence(stage, params, results)
        report = {
            **base_report,
            "overall": "PASS",
            "params": asdict(params) | {"source": str(params.source)},
            "native_brep": native_brep,
            "master_surface": dict(master.metadata),
            "cases": results,
            "convergence": convergence,
            "error": None,
        }
        (stage / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        gmsh.finalize()
        gmsh_initialized = False
        publish_generation(
            stage,
            inputs.outdir,
            stage="meshing",
            operation="central-feed",
            status="PASS",
            resolved_inputs=base_report["inputs"],
            input_units={"geometry": "mm", "mesh": "m"},
            producer_files=(Path(__file__),),
            upstream_artifacts=(inputs.params, inputs.brep, inputs.preflight),
            tool_versions={
                "gmsh": gmsh.__version__,
                "meshio": meshio.__version__,
                "numpy": np.__version__,
            },
        )
        stage = None
    except Exception as error:
        if gmsh_initialized:
            gmsh.finalize()
        failure = {
            **base_report,
            "overall": "FAIL",
            "solve_eligible_outputs_published": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _publish_failure(inputs.outdir, failure, stage)
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        raise PortedRunError(str(error), failure) from error
    if inputs.gui:
        open_optional_gui(inputs)
    return report


def print_report(report: dict[str, Any]) -> None:
    print("\nConformal central-feed layered Prism6 mesh")
    print(
        f"{'case':<20} {'G':>3} {'nodes':>10} {'film':>10} {'feed':>9} "
        f"{'total':>10} {'mouth':>7} {'minSICN':>10} {'minDetJac':>11} "
        f"{'BREP err':>10} {'inlet err':>10} {'topology':>9} {'OF':>8} {'overall':>8}"
    )
    print("-" * 156)
    case_name = Path(report.get("inputs", {}).get("outdir", "ported_prism")).name
    for case in report.get("cases", []):
        print(
            f"{case_name:<20} {case['n_gap']:3d} {case['nodes']:10d} "
            f"{case['film_prisms']:10d} {case['feed_prisms']:9d} "
            f"{case['total_prisms']:10d} {case['mouth_triangles']:7d} "
            f"{case['minimum_minSICN']:10.3e} {case['minimum_minDetJac']:11.3e} "
            f"{case['brep_volume_relative_error']:10.3e} "
            f"{case['inlet_area_relative_error']:10.3e} "
            f"{case['topology_status']:>9} {case['openfoam_status']:>8} {case['overall']:>8}"
        )
    print(f"Gmsh generated 3D volume mesh: {report.get('gmsh_generated_3d_volume_mesh', False)}")
    print(f"OVERALL: {report.get('overall', 'FAIL')}")
    if report.get("overall") == "PASS":
        preview = next(
            case
            for case in report["cases"]
            if case["n_gap"] == report["inputs"]["preview_ngap"]
        )
        print("\nOpen commands")
        for name, command in preview["commands"].items():
            print(f"{name:22s} {command}")


def parse_args(argv: Sequence[str] | None = None) -> PortedInputs:
    parser = argparse.ArgumentParser(
        description="Build the conformal full-360 central-feed layered Prism6 mesh."
    )
    parser.add_argument("--params", type=Path, default=PortedInputs.params)
    parser.add_argument("--brep", type=Path, default=PortedInputs.brep)
    parser.add_argument("--preflight", type=Path, default=PortedInputs.preflight)
    parser.add_argument("--outdir", type=Path, default=PortedInputs.outdir)
    parser.add_argument("--n-theta", type=int, default=256)
    parser.add_argument("--n-axial", type=int, default=96)
    parser.add_argument("--gap-levels", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--preview-ngap", type=int, default=8)
    parser.add_argument("--rim-segments", type=int, default=128)
    parser.add_argument("--gap-inflation-ratio", type=float, default=5.0)
    parser.add_argument("--tube-layers", type=int, default=48)
    parser.add_argument("--tube-grading", type=float, default=1.0)
    parser.add_argument("--openfoam", choices=("auto", "required", "skip"), default="auto")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gui-mode",
        choices=("full", "cutaway", "mouth", "quality"),
        default="cutaway",
    )
    args = parser.parse_args(argv)
    return PortedInputs(
        params=args.params,
        brep=args.brep,
        preflight=args.preflight,
        outdir=args.outdir,
        n_theta=args.n_theta,
        n_axial=args.n_axial,
        gap_levels=tuple(args.gap_levels),
        preview_ngap=args.preview_ngap,
        rim_segments=args.rim_segments,
        gap_inflation_ratio=args.gap_inflation_ratio,
        tube_layers=args.tube_layers,
        tube_grading=args.tube_grading,
        openfoam=args.openfoam,
        gui=args.gui,
        gui_mode=args.gui_mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_ported(parse_args(argv))
    except PortedRunError as error:
        print_report(error.report)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
