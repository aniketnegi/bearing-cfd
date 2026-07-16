#!/usr/bin/env python3
"""Export and statically audit the validated ported Prism6 mesh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import gmsh
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshing.gmsh_brep_preflight import atomic_replace_directory, make_staging_directory
from meshing.layered_prism_central_feed import (
    CELL_FIELD_NAMES,
    PRISM_QUAD_FACES,
    PRISM_TRI_FACES,
    PrismMesh,
    _component_count,
    _face_census,
    _lookup_rows,
    _prism_metrics,
    _structured_rows,
    add_discrete_prism_model,
    validate_external_face_orientation,
)


TRI_PATCHES = ("journal_wall", "bushing_bore_wall", "pressure_feed")
QUAD_PATCHES = ("axial_end_z0", "axial_end_zL", "feed_tube_wall")
PATCHES = TRI_PATCHES + QUAD_PATCHES
FORBIDDEN_PATCHES = {"feed_mouth", "mouth_cap", "internal_feed", "defaultFaces"}
PATCH_ROLES = {
    "journal_wall": "moving wall later",
    "bushing_bore_wall": "stationary wall",
    "axial_end_z0": "axial pressure outlet later",
    "axial_end_zL": "axial pressure outlet later",
    "feed_tube_wall": "stationary wall",
    "pressure_feed": "external pressure inlet disk",
}
EXPORT_NAMES = {
    "gmsh41": "bearing_prism_gmsh41.msh",
    "gmsh22": "bearing_prism_gmsh22_ascii.msh",
    "cgns": "bearing_prism.cgns",
}


class InterchangeError(RuntimeError):
    """A mandatory solver-neutral interchange check failed."""


@dataclass(frozen=True)
class InterchangeInputs:
    case_dir: Path
    outdir: Path
    fluent: Literal["auto", "required", "skip"] = "auto"
    gui: bool = False
    overwrite: bool = False
    coordinate_tolerance: float = 1.0e-14
    volume_relative_tolerance: float = 1.0e-12


@dataclass(frozen=True)
class CanonicalCase:
    path: Path
    mesh: PrismMesh
    manifest: dict[str, Any]
    report: dict[str, Any]


@dataclass(frozen=True)
class CanonicalAudit:
    summary: dict[str, Any]
    patch_owners: dict[str, np.ndarray]
    mouth_owners: np.ndarray


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise InterchangeError(message)


def _relative_error(actual: float, expected: float) -> float:
    scale = max(abs(actual), abs(expected), np.finfo(float).tiny)
    return abs(actual - expected) / scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InterchangeError(f"cannot read {path}: {error}") from error
    _check(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _patch_counts(mesh: PrismMesh) -> dict[str, int]:
    return {
        name: len(mesh.boundary_triangles.get(name, mesh.boundary_quads.get(name, ())))
        for name in PATCHES
    }


def load_canonical_case(case_dir: Path) -> CanonicalCase:
    case_dir = case_dir.resolve()
    required = ("mesh_arrays.npz", "manifest.json", "mesh_report.json", "physical_groups.json")
    missing = [name for name in required if not (case_dir / name).is_file()]
    _check(not missing, f"source case is missing required files: {missing}")
    manifest = _read_json(case_dir / "manifest.json")
    report = _read_json(case_dir / "mesh_report.json")
    physical = _read_json(case_dir / "physical_groups.json")
    _check(manifest.get("overall") == "PASS", "source manifest is not PASS")
    _check(report.get("overall") == "PASS", "source mesh report is not PASS")
    _check(manifest.get("coordinate_unit") == "m", "source coordinates are not declared in metres")
    _check(manifest.get("volume_cell_type") == "Prism6", "source volume type is not Prism6")
    _check(set(physical.get("boundaries", {})) == set(PATCHES), "source physical patches are not exact")
    _check(set(physical.get("volume", {})) == {"fluid"}, "source must contain one fluid physical volume")
    _check(
        FORBIDDEN_PATCHES.isdisjoint(physical.get("boundaries", {})),
        "source contains a forbidden feed-mouth/default boundary",
    )

    npz = case_dir / "mesh_arrays.npz"
    with np.load(npz, allow_pickle=False) as archive:
        names = set(archive.files)
        tri_names = {name.removeprefix("boundary_tri_") for name in names if name.startswith("boundary_tri_")}
        quad_names = {name.removeprefix("boundary_quad_") for name in names if name.startswith("boundary_quad_")}
        _check(tri_names == set(TRI_PATCHES), f"missing or extra Tri3 patches: {sorted(tri_names)}")
        _check(quad_names == set(QUAD_PATCHES), f"missing or extra Quad4 patches: {sorted(quad_names)}")
        base_arrays = {
            key: np.array(archive[key], copy=True)
            for key in ("points_m", "prisms", "mouth_triangles", "cell_tags", "node_tags", "cell_centres_m")
        }
        metadata = json.loads(str(archive["metadata_json"].item()))
        triangles = {name: np.array(archive[f"boundary_tri_{name}"], copy=True) for name in TRI_PATCHES}
        quads = {name: np.array(archive[f"boundary_quad_{name}"], copy=True) for name in QUAD_PATCHES}
        fields = {
            name: np.array(archive[f"field_{name}"], copy=True)
            for name in CELL_FIELD_NAMES
            if f"field_{name}" in names
        }
    points = base_arrays["points_m"]
    prisms = base_arrays["prisms"]
    _check(points.ndim == 2 and points.shape[1] == 3 and points.dtype == np.float64, "points_m must be float64 Nx3")
    _check(prisms.ndim == 2 and prisms.shape[1] == 6, "prisms must be an Nx6 array")
    _check(np.array_equal(base_arrays["node_tags"], np.arange(1, len(points) + 1)), "canonical node tags must be contiguous and one-based")
    _check(len(np.unique(base_arrays["cell_tags"])) == len(prisms), "canonical cell tags are not unique")
    _check(prisms.min(initial=1) >= 1 and prisms.max(initial=0) <= len(points), "Prism6 connectivity references an invalid node")
    _check("region_id" in fields and "volume_m3" in fields, "canonical region and volume fields are required")
    mesh = PrismMesh(
        points_m=points,
        prisms=prisms,
        boundary_triangles=triangles,
        boundary_quads=quads,
        mouth_triangles=base_arrays["mouth_triangles"],
        cell_tags=base_arrays["cell_tags"],
        node_tags=base_arrays["node_tags"],
        cell_fields=fields,
        cell_centres_m=base_arrays["cell_centres_m"],
        film_cell_count=int(np.sum(fields["region_id"] == 0)),
        feed_cell_count=int(np.sum(fields["region_id"] == 1)),
        master_triangle_count=len(triangles["journal_wall"]),
        disk_triangle_count=len(base_arrays["mouth_triangles"]),
        metadata=metadata,
    )
    return CanonicalCase(case_dir, mesh, manifest, report)


def _signature_mapping(actual: np.ndarray, expected: np.ndarray, label: str) -> np.ndarray:
    _check(actual.shape == expected.shape, f"{label} count or arity changed")
    expected_keys = np.sort(expected, axis=1)
    order = np.lexsort(tuple(expected_keys[:, index] for index in reversed(range(expected.shape[1]))))
    indices = _lookup_rows(expected_keys[order], actual)
    mapped = order[indices]
    _check(len(np.unique(mapped)) == len(mapped), f"{label} contains duplicates or lost members")
    return mapped


def audit_canonical(case: CanonicalCase, inputs: InterchangeInputs) -> CanonicalAudit:
    mesh = case.mesh
    counts = _patch_counts(mesh)
    _check(set(counts) == set(PATCHES) and all(counts.values()), "all six external patches must be nonempty")
    surface_rows = {row["boundary"]: int(row["face_count"]) for row in case.report.get("surface_quality", [])}
    _check(surface_rows == counts, f"source report patch counts do not match NPZ: {surface_rows} != {counts}")
    manifest_patch_counts = case.manifest.get("counts", {}).get("boundary_faces")
    if manifest_patch_counts is not None:
        _check(manifest_patch_counts == counts, "source manifest patch counts do not match NPZ")
    manifest_counts = case.manifest.get("counts", {})
    _check(int(manifest_counts.get("nodes", -1)) == len(mesh.points_m), "source point count mismatch")
    _check(int(manifest_counts.get("total_prisms", -1)) == len(mesh.prisms), "source Prism6 count mismatch")
    _check(int(manifest_counts.get("mouth_triangles", -1)) == len(mesh.mouth_triangles), "source mouth count mismatch")
    _check(len(np.unique(mesh.points_m, axis=0)) == len(mesh.points_m), "source has duplicate coordinates")
    cell_keys = _structured_rows(np.sort(mesh.prisms, axis=1))
    _check(len(np.unique(cell_keys)) == len(mesh.prisms), "source contains duplicate Prism6 cells")

    metrics = _prism_metrics(mesh.points_m, mesh.prisms)
    _check(np.all(metrics["volume_m3"] > 0.0), "source contains a nonpositive signed Prism6 volume")
    _check(np.all(metrics["custom_minDetJac"] > 0.0), "source contains a nonpositive Prism6 Jacobian")
    _check(
        np.max(np.abs(metrics["cell_centres_m"] - mesh.cell_centres_m), initial=0.0) <= inputs.coordinate_tolerance,
        "source cell centres disagree with canonical connectivity",
    )
    canonical_volume = float(mesh.cell_fields["volume_m3"].sum())
    calculated_volume = float(metrics["volume_m3"].sum())
    _check(
        _relative_error(calculated_volume, canonical_volume) <= inputs.volume_relative_tolerance,
        "source recomputed volume disagrees with the stored canonical volume",
    )
    report_volumes = case.report.get("geometry", {}).get("volumes", {})
    _check(
        _relative_error(calculated_volume, float(report_volumes.get("cell_sum_m3", math.nan)))
        <= inputs.volume_relative_tolerance,
        "source mesh report volume disagrees with NPZ",
    )
    brep_error = float(report_volumes.get("native_brep_relative_error", math.inf))
    _check(brep_error <= 5.0e-4, f"source linear-mesh/BREP volume error is outside the established envelope: {brep_error}")
    _check(mesh.metadata.get("coordinate_unit") == "m", "NPZ metadata does not declare metre coordinates")
    _check(float(mesh.metadata.get("scale_to_m_applied_once", math.nan)) == 0.001, "canonical source scale provenance is invalid")

    cell_indices = np.arange(len(mesh.prisms), dtype=np.int64)
    tri_unique, tri_incidence, tri_first, tri_second = _face_census(
        mesh.prisms[:, PRISM_TRI_FACES].reshape(-1, 3), np.repeat(cell_indices, 2)
    )
    quad_unique, quad_incidence, quad_first, quad_second = _face_census(
        mesh.prisms[:, PRISM_QUAD_FACES].reshape(-1, 4), np.repeat(cell_indices, 3)
    )
    _check(
        np.all((tri_incidence == 1) | (tri_incidence == 2))
        and np.all((quad_incidence == 1) | (quad_incidence == 2)),
        "source has orphan or non-manifold volume faces",
    )
    patch_owners: dict[str, np.ndarray] = {}
    declared_tri = np.concatenate(list(mesh.boundary_triangles.values()))
    declared_quad = np.concatenate(list(mesh.boundary_quads.values()))
    for name, faces in mesh.boundary_triangles.items():
        indices = _lookup_rows(tri_unique, faces)
        _check(np.all(tri_incidence[indices] == 1), f"{name} is not wholly external")
        patch_owners[name] = tri_first[indices]
    for name, faces in mesh.boundary_quads.items():
        indices = _lookup_rows(quad_unique, faces)
        _check(np.all(quad_incidence[indices] == 1), f"{name} is not wholly external")
        patch_owners[name] = quad_first[indices]
    _check(
        np.array_equal(np.unique(np.sort(declared_tri, axis=1), axis=0), tri_unique[tri_incidence == 1])
        and np.array_equal(np.unique(np.sort(declared_quad, axis=1), axis=0), quad_unique[quad_incidence == 1]),
        "declared patches are not the complete external-face census",
    )
    _check(
        len(np.unique(np.sort(declared_tri, axis=1), axis=0)) == len(declared_tri)
        and len(np.unique(np.sort(declared_quad, axis=1), axis=0)) == len(declared_quad),
        "source contains duplicate external faces",
    )
    external_edges = []
    for faces in mesh.boundary_triangles.values():
        external_edges.extend(faces[:, edge] for edge in ((0, 1), (1, 2), (2, 0)))
    for faces in mesh.boundary_quads.values():
        external_edges.extend(faces[:, edge] for edge in ((0, 1), (1, 2), (2, 3), (3, 0)))
    _, edge_incidence = np.unique(
        np.sort(np.concatenate(external_edges), axis=1), axis=0, return_counts=True
    )
    _check(np.all(edge_incidence == 2), "external boundary has a non-manifold edge")

    mouth_indices = _lookup_rows(tri_unique, mesh.mouth_triangles)
    _check(np.all(tri_incidence[mouth_indices] == 2), "a feed-mouth face is not internal")
    mouth_owners = np.column_stack((tri_first[mouth_indices], tri_second[mouth_indices]))
    regions = mesh.cell_fields["region_id"].astype(np.int32)
    _check(set(np.unique(regions).tolist()) == {0, 1}, "source region_id must contain film=0 and feed=1")
    _check(
        np.all(np.sort(regions[mouth_owners], axis=1) == np.asarray([0, 1])),
        "each mouth face must join one film cell and one feed cell",
    )
    mouth_keys = set(_structured_rows(np.sort(mesh.mouth_triangles, axis=1)).tolist())
    boundary_keys = set(_structured_rows(np.sort(declared_tri, axis=1)).tolist())
    _check(mouth_keys.isdisjoint(boundary_keys), "shared feed-film interface was registered as a boundary")

    internal_first = np.concatenate((tri_first[tri_incidence == 2], quad_first[quad_incidence == 2]))
    internal_second = np.concatenate((tri_second[tri_incidence == 2], quad_second[quad_incidence == 2]))
    components, roots = _component_count(len(mesh.prisms), internal_first, internal_second)
    _check(components == 1, f"source contains {components} disconnected fluid regions")
    pressure_owners = patch_owners["pressure_feed"]
    _check(np.all(regions[pressure_owners] == 1), "pressure_feed is not incident to feed cells")
    for axial in ("axial_end_z0", "axial_end_zL"):
        _check(
            np.all(roots[patch_owners[axial]] == roots[pressure_owners[0]]),
            f"no cell-graph path from pressure_feed to {axial}",
        )
    _check(
        np.all(roots[mouth_owners.ravel()] == roots[pressure_owners[0]]),
        "pressure feed is not connected through feed cells and mouth faces to film cells",
    )
    orientation_records: list[dict[str, Any]] = []
    orientation = validate_external_face_orientation(mesh, orientation_records, patch_owners)

    bbox = np.concatenate((mesh.points_m.min(axis=0), mesh.points_m.max(axis=0))).tolist()
    boundary_total = sum(counts.values())
    _check(
        boundary_total == int(np.sum(tri_incidence == 1) + np.sum(quad_incidence == 1)),
        "sum of exact patch counts differs from the external-face census",
    )
    return CanonicalAudit(
        summary={
            "format": "SOURCE_NPZ",
            "points": len(mesh.points_m),
            "prism6_cells": len(mesh.prisms),
            "patch_counts": counts,
            "total_boundary_faces": boundary_total,
            "total_faces": len(tri_unique) + len(quad_unique),
            "mouth_internal_faces": len(mesh.mouth_triangles),
            "bounding_box_m": bbox,
            "volume_m3": calculated_volume,
            "minimum_signed_volume_m3": float(metrics["volume_m3"].min()),
            "minimum_jacobian": float(metrics["custom_minDetJac"].min()),
            "connected_regions": components,
            "units": "m",
            "brep_relative_error": brep_error,
            "external_face_orientation": orientation,
            "connectivity_checks": {
                "one_connected_fluid_region": True,
                "no_duplicate_cells": True,
                "no_duplicate_external_faces": True,
                "no_orphan_or_nonmanifold_faces": True,
                "external_shell_edges_manifold": True,
                "mouth_two_incident_cells": True,
                "mouth_is_internal": True,
                "pressure_feed_through_mouth_to_film": True,
                "pressure_feed_to_axial_end_z0": True,
                "pressure_feed_to_axial_end_zL": True,
            },
            "static_status": "PASS",
        },
        patch_owners=patch_owners,
        mouth_owners=mouth_owners,
    )


def validate_group_names(names: set[str], format_name: str) -> None:
    expected = set(PATCHES) | {"fluid"}
    _check(
        names == expected,
        f"{format_name} boundary-name/membership loss: {sorted(names)}; GEOMETRY_ONLY_NOT_FLUENT_READY",
    )


def _physical_groups() -> dict[str, tuple[int, list[int]]]:
    groups: dict[str, tuple[int, list[int]]] = {}
    for dimension, physical_id in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(int(dimension), int(physical_id))
        _check(name and name not in groups, f"duplicate or unnamed physical group in reopened mesh: {name!r}")
        groups[name] = (
            int(dimension),
            [int(tag) for tag in gmsh.model.getEntitiesForPhysicalGroup(int(dimension), int(physical_id))],
        )
    return groups


def _group_elements(element_type: int, entities: Sequence[int], arity: int) -> np.ndarray:
    blocks = []
    for entity in entities:
        _tags, nodes = gmsh.model.mesh.getElementsByType(element_type, entity)
        array = np.asarray(nodes, dtype=np.int64)
        if array.size:
            blocks.append(array.reshape(-1, arity))
    return np.concatenate(blocks) if blocks else np.empty((0, arity), dtype=np.int64)


def _canonicalize_coordinates(
    mesh: PrismMesh, tolerance: float
) -> tuple[np.ndarray, np.ndarray, float]:
    tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags_raw, dtype=np.int64)
    points = np.asarray(coordinates_raw, dtype=np.float64).reshape(-1, 3)
    _check(len(points) == len(mesh.points_m), "round-trip point count changed")
    read_order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    source_order = np.lexsort((mesh.points_m[:, 2], mesh.points_m[:, 1], mesh.points_m[:, 0]))
    error = float(np.max(np.abs(points[read_order] - mesh.points_m[source_order]), initial=0.0))
    _check(error <= tolerance, f"round-trip coordinate error {error:.3e} m exceeds {tolerance:.3e} m")
    canonical_points = np.empty_like(mesh.points_m)
    canonical_points[source_order] = points[read_order]
    node_map = np.zeros(int(tags.max(initial=0)) + 1, dtype=np.int64)
    node_map[tags[read_order]] = mesh.node_tags[source_order].astype(np.int64)
    _check(np.all(node_map[tags] > 0), "round-trip node mapping is incomplete")
    return canonical_points, node_map, error


def _orientation_from_round_trip(
    points: np.ndarray,
    faces: dict[str, np.ndarray],
    owners: dict[str, np.ndarray],
    source_prisms: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, patch in faces.items():
        face_points = points[patch - 1]
        if patch.shape[1] == 3:
            normals = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0]) / 2.0
        else:
            normals = (
                np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
                + np.cross(face_points[:, 2] - face_points[:, 0], face_points[:, 3] - face_points[:, 0])
            ) / 2.0
        magnitudes = np.linalg.norm(normals, axis=1)
        owner_centres = points[source_prisms[owners[name]] - 1].mean(axis=1)
        projection = np.einsum(
            "ij,ij->i", normals / magnitudes[:, None], face_points.mean(axis=1) - owner_centres
        )
        _check(np.all(magnitudes > 0.0) and np.all(projection > 0.0), f"{name} contains a reversed or degenerate exported face")
        result[name] = float(projection.min())
    return result


def audit_round_trip(
    path: Path,
    format_name: str,
    case: CanonicalCase,
    canonical: CanonicalAudit,
    inputs: InterchangeInputs,
) -> dict[str, Any]:
    gmsh.clear()
    gmsh.open(str(path))
    groups = _physical_groups()
    validate_group_names(set(groups), format_name)
    _check(groups["fluid"][0] == 3 and len(groups["fluid"][1]) == 1, f"{format_name} must contain one 3D fluid zone")
    _check(len(gmsh.model.getEntities(3)) == 1, f"{format_name} contains more than one 3D zone")
    _check(all(groups[name][0] == 2 and len(groups[name][1]) == 1 for name in PATCHES), f"{format_name} boundary groups are not six separate 2D zones")
    _check(FORBIDDEN_PATCHES.isdisjoint(groups), f"{format_name} contains a forbidden boundary group")
    tri_type = int(gmsh.model.mesh.getElementType("Triangle", 1))
    quad_type = int(gmsh.model.mesh.getElementType("Quadrangle", 1))
    prism_type = int(gmsh.model.mesh.getElementType("Prism", 1))
    _check((tri_type, quad_type, prism_type) == (2, 3, 6), "Gmsh linear element identifiers changed")
    points, node_map, coordinate_error = _canonicalize_coordinates(case.mesh, inputs.coordinate_tolerance)
    read_prisms = node_map[_group_elements(prism_type, groups["fluid"][1], 6)]
    cell_mapping = _signature_mapping(read_prisms, case.mesh.prisms, f"{format_name} PENTA_6")
    inverse_cells = np.empty(len(cell_mapping), dtype=np.int64)
    inverse_cells[cell_mapping] = np.arange(len(cell_mapping), dtype=np.int64)
    read_faces: dict[str, np.ndarray] = {}
    read_owners: dict[str, np.ndarray] = {}
    for name in PATCHES:
        arity = 3 if name in TRI_PATCHES else 4
        kind = tri_type if arity == 3 else quad_type
        expected = case.mesh.boundary_triangles[name] if arity == 3 else case.mesh.boundary_quads[name]
        actual = node_map[_group_elements(kind, groups[name][1], arity)]
        face_mapping = _signature_mapping(actual, expected, f"{format_name} {name}")
        read_faces[name] = actual
        read_owners[name] = canonical.patch_owners[name][face_mapping]
    element_types, element_tags, _ = gmsh.model.mesh.getElements()
    actual_type_counts = {int(kind): len(tags) for kind, tags in zip(element_types, element_tags)}
    expected_type_counts = {
        tri_type: sum(len(case.mesh.boundary_triangles[name]) for name in TRI_PATCHES),
        quad_type: sum(len(case.mesh.boundary_quads[name]) for name in QUAD_PATCHES),
        prism_type: len(case.mesh.prisms),
    }
    _check(actual_type_counts == expected_type_counts, f"{format_name} contains unexpected or missing elements")

    metrics = _prism_metrics(points, read_prisms)
    _check(np.all(metrics["volume_m3"] > 0.0), f"{format_name} contains nonpositive signed volumes")
    _check(np.all(metrics["custom_minDetJac"] > 0.0), f"{format_name} contains nonpositive Jacobians")
    volume = float(metrics["volume_m3"].sum())
    _check(
        _relative_error(volume, canonical.summary["volume_m3"]) <= inputs.volume_relative_tolerance,
        f"{format_name} volume changed",
    )
    bbox = np.concatenate((points.min(axis=0), points.max(axis=0)))
    source_bbox = np.asarray(canonical.summary["bounding_box_m"])
    _check(np.max(np.abs(bbox - source_bbox), initial=0.0) <= inputs.coordinate_tolerance, f"{format_name} bounding box changed")
    orientation = _orientation_from_round_trip(points, read_faces, read_owners, case.mesh.prisms)

    for mouth, incident in zip(case.mesh.mouth_triangles, canonical.mouth_owners):
        actual_cells = read_prisms[inverse_cells[incident]]
        _check(
            all(set(map(int, mouth)).issubset(set(map(int, cell))) for cell in actual_cells),
            f"{format_name} lost two-cell connectivity at a mouth face",
        )
    return {
        "format": format_name,
        "file": path.name,
        "points": len(points),
        "prism6_cells": len(read_prisms),
        "patch_counts": {name: len(read_faces[name]) for name in PATCHES},
        "total_boundary_faces": sum(len(read_faces[name]) for name in PATCHES),
        "total_faces": canonical.summary["total_faces"],
        "mouth_internal_faces": len(case.mesh.mouth_triangles),
        "bounding_box_m": bbox.tolist(),
        "volume_m3": volume,
        "minimum_signed_volume_m3": float(metrics["volume_m3"].min()),
        "minimum_jacobian": float(metrics["custom_minDetJac"].min()),
        "connected_regions": canonical.summary["connected_regions"],
        "units": "m",
        "coordinate_max_error_m": coordinate_error,
        "external_face_minimum_outward_projection_m": orientation,
        "physical_groups": sorted(groups),
        "fluid_zone_count": 1,
        "volume_element_type": "PENTA_6",
        "boundary_element_types": ["TRI_3", "QUAD_4"],
        "solution_fields": 0,
        "static_status": "PASS",
    }


def write_clean_meshes(case: CanonicalCase, stage: Path, records: list[dict[str, Any]]) -> dict[str, Path]:
    gmsh.clear()
    add_discrete_prism_model(case.mesh, records, "bearing_prism_solver_neutral")
    _check(len(gmsh.view.getTags()) == 0, "clean interchange model unexpectedly contains post-processing views")
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    paths = {key: stage / name for key, name in EXPORT_NAMES.items()}
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(paths["gmsh41"]))
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(paths["gmsh22"]))
    build_info = gmsh.option.getString("General.BuildInfo")
    _check("Cgns" in build_info, "this official Gmsh build has no CGNS writer")
    gmsh.write(str(paths["cgns"]))
    _check(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()), "an interchange mesh was not written")
    return paths


def _write_zones(path: Path, case: CanonicalCase) -> None:
    counts = _patch_counts(case.mesh)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("name", "dimension", "element_type", "count", "role", "external")
        )
        writer.writeheader()
        writer.writerow(
            {"name": "fluid", "dimension": 3, "element_type": "PENTA_6", "count": len(case.mesh.prisms), "role": "fluid volume", "external": False}
        )
        for name in PATCHES:
            writer.writerow(
                {
                    "name": name,
                    "dimension": 2,
                    "element_type": "TRI_3" if name in TRI_PATCHES else "QUAD_4",
                    "count": counts[name],
                    "role": PATCH_ROLES[name],
                    "external": True,
                }
            )


def _readme_text(outdir: Path, fluent_passed: bool) -> str:
    native = outdir / "bearing_prism_imported.msh.h5"
    workbench_file = str(native) if fluent_passed else "NONE YET (create bearing_prism_imported.msh.h5 only after a real Fluent audit passes)"
    return f"""SOLVER-NEUTRAL PORTED PRISM MESH

Static status does not imply a Fluent import pass. Read interchange_report.json first.

Open in Gmsh:
  {outdir / EXPORT_NAMES['gmsh41']}
  {outdir / EXPORT_NAMES['gmsh22']}

Import in standalone Fluent or a Fluent Setup session:
  {outdir / EXPORT_NAMES['cgns']}

Manual Fluent import-only audit:
  1. Start Fluent in 3D, double precision; do not initialize a solution.
  2. File > Import > CGNS > Mesh... and select bearing_prism.cgns.
  3. Run Mesh > Check with maximum available verbosity.
  4. Confirm one fluid cell zone, Prism6/wedge cells, and exactly the six named boundary zones.
  5. Compare nodes, faces, cells, per-zone counts, bounding box, volume, minimum volume,
     maximum skewness, and region count with interchange_report.json.
  6. Confirm there is no boundary zone at the shared feed-film interface.
  7. Only after every check passes, save bearing_prism_imported.msh.h5.
  8. Do not initialize, iterate, or assign CFD physics in this stage.

Automated audit when PyFluent and a CGNS/VKI license are available:
  uv run python interchange/fluent_import_check.py --interchange-dir {outdir}

File for later Workbench integration:
  {workbench_file}

Gmsh .msh and Fluent .msh are different formats; renaming an extension is not conversion.
Workbench Meshing is not expected to edit this Gmsh volume mesh. Import CGNS in Fluent,
audit it, save native Fluent .msh.h5 (or an import-only .cas.h5), then use that native file.
Retain the separate BREP/CAD route if Ansys Meshing must generate a new mesh later.
"""


def _print_table(rows: Sequence[dict[str, Any]], source: Path, fluent_status: str) -> None:
    print(f"\nSource case: {source}")
    header = ["format", "points", "Prism6", *PATCHES, "mouth", "bbox [m]", "volume [m3]", "min signed V", "regions", "units", "static", "Fluent"]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for row in rows:
        bbox = ",".join(f"{value:.9g}" for value in row["bounding_box_m"])
        values = [
            row["format"],
            str(row["points"]),
            str(row["prism6_cells"]),
            *(str(row["patch_counts"][name]) for name in PATCHES),
            str(row["mouth_internal_faces"]),
            bbox,
            f"{row['volume_m3']:.16e}",
            f"{row['minimum_signed_volume_m3']:.6e}",
            str(row["connected_regions"]),
            row["units"],
            row["static_status"],
            fluent_status,
        ]
        print(" | ".join(values))


def enforce_fluent_mode(mode: str, result: dict[str, Any]) -> bool:
    if result["status"] == "NOT_RUN" and mode == "required":
        raise InterchangeError(f"--fluent required but Fluent could not run: {result['reason']}")
    _check(result["status"] != "FAIL", f"real Fluent import audit failed: {result.get('reason')}")
    return result.get("status") == "PASS" and result.get("real_import") is True


def run_interchange(inputs: InterchangeInputs) -> dict[str, Any]:
    inputs = InterchangeInputs(
        case_dir=inputs.case_dir.resolve(),
        outdir=inputs.outdir.resolve(),
        fluent=inputs.fluent,
        gui=inputs.gui,
        overwrite=inputs.overwrite,
        coordinate_tolerance=inputs.coordinate_tolerance,
        volume_relative_tolerance=inputs.volume_relative_tolerance,
    )
    _check(inputs.coordinate_tolerance > 0.0 and math.isfinite(inputs.coordinate_tolerance), "coordinate tolerance must be finite and positive")
    _check(inputs.volume_relative_tolerance > 0.0 and math.isfinite(inputs.volume_relative_tolerance), "volume tolerance must be finite and positive")
    _check(inputs.overwrite or not inputs.outdir.exists(), f"output directory exists; pass --overwrite: {inputs.outdir}")
    case = load_canonical_case(inputs.case_dir)
    canonical = audit_canonical(case, inputs)
    stage = make_staging_directory(inputs.outdir)
    gmsh_initialized = False
    try:
        gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()
        records: list[dict[str, Any]] = []
        paths = write_clean_meshes(case, stage, records)
        round_trips = [
            audit_round_trip(paths["gmsh41"], "GMSH_4_1", case, canonical, inputs),
            audit_round_trip(paths["gmsh22"], "GMSH_2_2_ASCII", case, canonical, inputs),
        ]
        try:
            round_trips.append(audit_round_trip(paths["cgns"], "CGNS", case, canonical, inputs))
        except Exception as error:
            raise InterchangeError(
                f"CGNS did not preserve the mandatory zones/connectivity; GEOMETRY_ONLY_NOT_FLUENT_READY: {error}"
            ) from error
        gmsh_log = [str(line) for line in gmsh.logger.get()]
        gmsh.logger.stop()
        gmsh.finalize()
        gmsh_initialized = False
        (stage / "gmsh_interchange.log").write_text("\n".join(gmsh_log) + "\n", encoding="utf-8")

        fluent_result: dict[str, Any]
        if inputs.fluent == "skip":
            fluent_result = {"status": "NOT_RUN", "reason": "disabled by --fluent skip", "real_import": False}
        else:
            from interchange.fluent_import_check import run_fluent_import_audit

            fluent_result = run_fluent_import_audit(
                cgns=paths["cgns"],
                canonical=canonical.summary,
                outdir=stage,
                gui=inputs.gui,
            )
        fluent_passed = enforce_fluent_mode(inputs.fluent, fluent_result)
        overall = "FLUENT_IMPORT_PASS" if fluent_passed else "STATIC_PASS_FLUENT_NOT_RUN"
        if not fluent_passed:
            print("FLUENT IMPORT NOT EXECUTED")

        hashes = {
            "source": {
                name: {"path": str(inputs.case_dir / name), "sha256": _sha256(inputs.case_dir / name)}
                for name in ("mesh_arrays.npz", "manifest.json", "mesh_report.json")
            },
            "exports": {
                path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in paths.values()
            },
        }
        _write_json(stage / "file_hashes.json", hashes)
        _write_zones(stage / "zones.csv", case)
        report = {
            "overall": overall,
            "static_status": "STATIC_PASS",
            "fluent_import_status": "PASS" if fluent_passed else "NOT_RUN",
            "readiness": "FLUENT_IMPORT_AUDITED" if fluent_passed else "STATICALLY_VALIDATED_NOT_IMPORTED",
            "source_case": str(inputs.case_dir),
            "source": canonical.summary,
            "exports": round_trips,
            "fluent": fluent_result,
            "tolerances": {
                "coordinate_m": inputs.coordinate_tolerance,
                "volume_relative": inputs.volume_relative_tolerance,
                "brep_relative_envelope": 5.0e-4,
            },
            "provenance": hashes,
            "cfd_solution_executed": False,
            "flow_initialized": False,
            "solver_iterations": 0,
            "final_solver_selected": None,
        }
        manifest = {
            "schema_version": 1,
            "overall": overall,
            "source_case": str(inputs.case_dir),
            "coordinate_unit": "m",
            "export_coordinate_scale": 1.0,
            "volume_zone": {"name": "fluid", "element_type": "PENTA_6", "count": len(case.mesh.prisms)},
            "boundary_zones": _patch_counts(case.mesh),
            "mouth": {
                "boundary_group": False,
                "internal_owner_neighbour_faces": len(case.mesh.mouth_triangles),
                "incident_cells_per_face": 2,
            },
            "files": EXPORT_NAMES,
            "fluent_import": fluent_result,
            "fluent_ready": fluent_passed,
            "solution_fields": 0,
            "cfd_solution_executed": False,
        }
        _write_json(stage / "interchange_report.json", report)
        _write_json(stage / "interchange_manifest.json", manifest)
        (stage / "README_OPEN_ME_FIRST.txt").write_text(
            _readme_text(inputs.outdir, fluent_passed), encoding="utf-8"
        )
        atomic_replace_directory(stage, inputs.outdir)
        rows = [canonical.summary, *round_trips]
        _print_table(rows, inputs.case_dir, report["fluent_import_status"])
        print("\nFinal handoff")
        print(f"1. Gmsh: {inputs.outdir / EXPORT_NAMES['gmsh41']} and {inputs.outdir / EXPORT_NAMES['gmsh22']}")
        print(f"2. Fluent CGNS import: {inputs.outdir / EXPORT_NAMES['cgns']}")
        print("3. Manual Fluent GUI: 3D Double > File > Import > CGNS > Mesh > Mesh Check; do not initialize or iterate")
        print(f"4. Automated audit: uv run python interchange/fluent_import_check.py --interchange-dir {inputs.outdir}")
        native = inputs.outdir / "bearing_prism_imported.msh.h5"
        print(f"5. Later Workbench file: {native if fluent_passed else 'none until a real Fluent audit creates bearing_prism_imported.msh.h5'}")
        print(f"6. Limitation: {fluent_result.get('reason', 'none')}")
        return report
    except Exception:
        if gmsh_initialized:
            try:
                gmsh.logger.stop()
            except Exception:
                pass
            gmsh.finalize()
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args(argv: Sequence[str] | None = None) -> InterchangeInputs:
    parser = argparse.ArgumentParser(description="Export the validated ported Prism6 mesh to clean Gmsh and CGNS interchange files.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fluent", choices=("auto", "required", "skip"), default="auto")
    parser.add_argument("--gui", action="store_true", help="show Fluent UI only when a real Fluent audit runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--coordinate-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--volume-relative-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    return InterchangeInputs(
        case_dir=args.case_dir,
        outdir=args.outdir,
        fluent=args.fluent,
        gui=args.gui,
        overwrite=args.overwrite,
        coordinate_tolerance=args.coordinate_tolerance,
        volume_relative_tolerance=args.volume_relative_tolerance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_interchange(parse_args(argv))
    except Exception as error:
        print(f"INTERCHANGE FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
