#!/usr/bin/env python3
"""Validate native bearing BREP geometry in Gmsh before any meshing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import gmsh

from bearing_cfd.artifacts import make_staging_directory, publish_generation


VOLUME_REL_TOL = 1.0e-10
AREA_REL_TOL = 1.0e-10
SCALE_REL_TOL = 1.0e-10
BBOX_ABS_TOL_MM = 1.0e-6
POSITION_ABS_TOL_MM = 1.0e-6
# OCC Bnd_Box adds 1e-7 in the file's coordinate unit at each side. Exact
# interface locations are checked separately with POSITION_ABS_TOL_MM * scale.
BBOX_SPAN_REL_TOL = 5.0e-5
TOPOLOGY_TOLERANCE_MAX_GAP_FRACTION = 1.0e-2
SI_SCALE = 1.0e-3

STRICT_OCC_OPTIONS = {
    "Geometry.OCCAutoFix": 0.0,
    "Geometry.OCCFixDegenerated": 0.0,
    "Geometry.OCCFixSmallEdges": 0.0,
    "Geometry.OCCFixSmallFaces": 0.0,
    "Geometry.OCCSewFaces": 0.0,
    "Geometry.OCCMakeSolids": 0.0,
    "Geometry.OCCBooleanGlue": 0.0,
    "Geometry.OCCBooleanNonDestructive": 0.0,
    "Geometry.ToleranceBoolean": 0.0,
    "Geometry.OCCBooleanSimplify": 0.0,
}

Vec3 = tuple[float, float, float]
BBox = tuple[float, float, float, float, float, float]


class GmshPreflightError(RuntimeError):
    """Base class for expected preflight failures."""


class InputValidationError(GmshPreflightError):
    """The BREP files and params.json are missing or inconsistent."""


class GeometryValidationError(GmshPreflightError):
    """A mandatory OCC geometry check failed."""


class GeometryClassificationError(GmshPreflightError):
    """OCC entities could not be classified uniquely from their geometry."""


class PreflightRunError(GmshPreflightError):
    """A failed run with a serializable diagnostic report."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class PreflightInputs:
    unsplit: Path = Path("out/conical_journal/geometry/default/film_unsplit.brep")
    zones: Path = Path("out/conical_journal/geometry/default/film_zones.brep")
    params: Path = Path("out/conical_journal/geometry/default/params.json")
    outdir: Path = Path("out/conical_journal/meshing/brep-preflight")
    gui: bool = False


@dataclass(frozen=True)
class CadReference:
    length: float
    mean_radius: float
    radial_clearance: float
    cone_slope: float
    eccentricity: float
    ex: float
    ey: float
    axial_cutter_extension: float
    hole_radius: float
    hole_axial_pos: float
    y_feed_end: float
    z1: float
    z2: float
    h_radial_min: float
    total_volume: float
    zone_volumes: dict[str, float]
    bounding_box: BBox
    inlet_area: float
    inlet_centre: Vec3
    source_overall: str

    def journal_radius(self, z: float) -> float:
        return self.mean_radius + (self.length / 2.0 - z) * self.cone_slope

    def bore_radius(self, z: float) -> float:
        return self.journal_radius(z) + self.radial_clearance


@dataclass(frozen=True)
class VolumeRecord:
    tag: int
    mass: float
    centre: Vec3
    bbox: BBox
    label: str | None = None


@dataclass(frozen=True)
class SurfaceRecord:
    tag: int
    entity_type: str
    area: float
    centre: Vec3
    bbox: BBox
    adjacent_volume_tags: tuple[int, ...]
    adjacent_volume_labels: tuple[str, ...]
    boundary_curve_tags: tuple[int, ...]
    boundary_curve_types: tuple[str, ...]
    properties: tuple[float, ...]
    classification: str | None = None


def relative_error(actual: float, expected: float) -> float:
    """Return a relative error, with an absolute fallback for a zero target."""
    return abs(actual - expected) / abs(expected) if expected else abs(actual)


def max_abs_difference(actual: Sequence[float], expected: Sequence[float]) -> float:
    if len(actual) != len(expected):
        return math.inf
    return max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def normalized(vector: Sequence[float]) -> Vec3:
    norm = vector_norm(vector)
    if norm == 0.0:
        raise GeometryClassificationError("zero-length OCC direction vector")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def parallel(vector: Sequence[float], axis: Vec3, tolerance: float = 1.0e-10) -> bool:
    direction = normalized(vector)
    return abs(sum(a * b for a, b in zip(direction, axis))) >= 1.0 - tolerance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: Path) -> tuple[CadReference, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        resolved = raw["resolved_parameters"]
        measurements = raw["measurements"]
        volumes = measurements["volumes_mm3"]
        bbox = measurements["final_bounding_box_mm"]
        inlet = measurements["inlet_face"]
        reference = CadReference(
            length=float(resolved["length"]),
            mean_radius=float(resolved["mean_radius"]),
            radial_clearance=float(resolved["radial_clearance"]),
            cone_slope=float(resolved["cone_slope"]),
            eccentricity=float(resolved["eccentricity"]),
            ex=float(resolved["ex"]),
            ey=float(resolved["ey"]),
            axial_cutter_extension=float(resolved["axial_cutter_extension"]),
            hole_radius=float(resolved["hole_radius"]),
            hole_axial_pos=float(resolved["hole_axial_pos"]),
            y_feed_end=float(resolved["y_feed_end"]),
            z1=float(resolved["z1"]),
            z2=float(resolved["z2"]),
            h_radial_min=float(resolved["h_radial_min"]),
            total_volume=float(volumes["total"]),
            zone_volumes={
                name: float(volumes["zones"][name])
                for name in ("ring_A", "hole_band", "ring_B")
            },
            bounding_box=tuple(float(value) for value in (*bbox["min"], *bbox["max"])),
            inlet_area=float(inlet["area_mm2"]),
            inlet_centre=tuple(float(value) for value in inlet["centre_mm"]),
            source_overall=str(raw.get("overall", "UNKNOWN")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InputValidationError(f"invalid CAD parameter record {path}: {error}") from error
    return reference, raw


def reference_from_dict(data: dict[str, Any]) -> CadReference:
    """Restore the CAD reference embedded in a preflight report."""
    values = dict(data)
    values["zone_volumes"] = {
        str(name): float(volume) for name, volume in values["zone_volumes"].items()
    }
    values["bounding_box"] = tuple(float(value) for value in values["bounding_box"])
    values["inlet_centre"] = tuple(float(value) for value in values["inlet_centre"])
    return CadReference(**values)


def classify_volumes(
    volumes: Sequence[VolumeRecord], reference: CadReference, scale: float = 1.0
) -> dict[str, VolumeRecord]:
    """Classify axial bodies from centre of mass and z extents, never tags."""
    if len(volumes) != 3:
        raise GeometryClassificationError(
            f"expected 3 volumes for axial classification, got {len(volumes)}"
        )

    z1 = reference.z1 * scale
    z2 = reference.z2 * scale
    length = reference.length * scale
    tolerance = POSITION_ABS_TOL_MM * scale
    classified: dict[str, VolumeRecord] = {}

    for volume in volumes:
        z = volume.centre[2]
        if z < z1 - tolerance:
            label = "ring_A"
            expected_ends = (0.0, z1)
        elif z <= z2 + tolerance:
            label = "hole_band"
            expected_ends = (z1, z2)
        else:
            label = "ring_B"
            expected_ends = (z2, length)

        if label in classified:
            raise GeometryClassificationError(f"multiple volumes classified as {label}")
        expected_span = expected_ends[1] - expected_ends[0]
        actual_span = volume.bbox[5] - volume.bbox[2]
        if not (volume.bbox[2] <= expected_ends[0] + tolerance < volume.centre[2]):
            raise GeometryClassificationError(
                f"{label} bounding box does not contain its expected lower plane"
            )
        if not (volume.centre[2] < expected_ends[1] - tolerance <= volume.bbox[5]):
            raise GeometryClassificationError(
                f"{label} bounding box does not contain its expected upper plane"
            )
        if relative_error(actual_span, expected_span) > BBOX_SPAN_REL_TOL:
            raise GeometryClassificationError(
                f"{label} z-span={actual_span:.12g}, expected {expected_span:.12g}"
            )
        classified[label] = replace(volume, label=label)

    expected_labels = {"ring_A", "hole_band", "ring_B"}
    if set(classified) != expected_labels:
        raise GeometryClassificationError(
            f"volume labels {sorted(classified)}, expected {sorted(expected_labels)}"
        )
    return classified


def _lateral_surface_classification(
    surface: SurfaceRecord, reference: CadReference, scale: float
) -> str:
    """Distinguish journal, bore and feed walls from analytic OCC support data."""
    if surface.entity_type not in {"Cone", "Cylinder", "Surface of Revolution"}:
        raise GeometryClassificationError(
            f"unsupported external surface type {surface.entity_type} on tag {surface.tag}"
        )
    if len(surface.properties) < 7:
        raise GeometryClassificationError(
            f"missing OCC axis/radius properties for surface {surface.tag}"
        )

    centre = surface.properties[0:3]
    axis = surface.properties[3:6]
    radius = surface.properties[6]
    tolerance = 1.0e-5 * scale

    if parallel(axis, (0.0, 1.0, 0.0)):
        feed_error = max(
            abs(centre[0]),
            abs(centre[2] - reference.hole_axial_pos * scale),
            abs(radius - reference.hole_radius * scale),
        )
        if feed_error <= tolerance:
            return "stationary_bushing_feed_wall"
        raise GeometryClassificationError(
            f"Y-axis cylinder {surface.tag} does not match the feed drilling"
        )

    if not parallel(axis, (0.0, 0.0, 1.0)):
        raise GeometryClassificationError(
            f"surface {surface.tag} has an unexpected lateral axis {axis}"
        )

    z_mm = centre[2] / scale
    journal_error = max(
        abs(centre[0] - reference.ex * scale),
        abs(centre[1] - reference.ey * scale),
        abs(radius - reference.journal_radius(z_mm) * scale),
    )
    bore_error = max(
        abs(centre[0]),
        abs(centre[1]),
        abs(radius - reference.bore_radius(z_mm) * scale),
    )
    if journal_error <= tolerance and journal_error < bore_error:
        return "journal_rotating_wall"
    if bore_error <= tolerance and bore_error < journal_error:
        return "stationary_bushing_feed_wall"
    raise GeometryClassificationError(
        f"surface {surface.tag} matches neither analytic journal nor bore "
        f"(journal error {journal_error:.3e}, bore error {bore_error:.3e})"
    )


def classify_surfaces(
    surfaces: Sequence[SurfaceRecord], reference: CadReference, scale: float = 1.0
) -> tuple[SurfaceRecord, ...]:
    """Assign physical boundary roles using geometry and volume adjacency."""
    position_tolerance = POSITION_ABS_TOL_MM * scale
    expected_positions = {
        "outlet_z0": 0.0,
        "interface_z1": reference.z1 * scale,
        "interface_z2": reference.z2 * scale,
        "outlet_zL": reference.length * scale,
    }
    classified: list[SurfaceRecord] = []

    for surface in surfaces:
        adjacent_count = len(surface.adjacent_volume_tags)
        role: str | None = None

        if adjacent_count == 2:
            if surface.entity_type != "Plane":
                raise GeometryClassificationError(
                    f"internal surface {surface.tag} is {surface.entity_type}, not Plane"
                )
            for candidate in ("interface_z1", "interface_z2"):
                if abs(surface.centre[2] - expected_positions[candidate]) <= position_tolerance:
                    role = candidate
                    break
        elif adjacent_count == 1 and surface.entity_type == "Plane":
            inlet_target = (
                0.0,
                reference.y_feed_end * scale,
                reference.hole_axial_pos * scale,
            )
            if max_abs_difference(surface.centre, inlet_target) <= position_tolerance:
                role = "pressure_inlet"
            elif abs(surface.centre[2] - expected_positions["outlet_z0"]) <= position_tolerance:
                role = "outlet_z0"
            elif abs(surface.centre[2] - expected_positions["outlet_zL"]) <= position_tolerance:
                role = "outlet_zL"
        elif adjacent_count == 1:
            role = _lateral_surface_classification(surface, reference, scale)

        if role is None:
            raise GeometryClassificationError(
                f"surface {surface.tag} ({surface.entity_type}, adjacency "
                f"{surface.adjacent_volume_tags}) has no unique physical role"
            )
        classified.append(replace(surface, classification=role))

    singular_roles = {
        "pressure_inlet",
        "outlet_z0",
        "outlet_zL",
        "interface_z1",
        "interface_z2",
    }
    for role in singular_roles:
        count = sum(surface.classification == role for surface in classified)
        if count != 1:
            raise GeometryClassificationError(f"expected one {role}, found {count}")

    journal_count = sum(
        surface.classification == "journal_rotating_wall" for surface in classified
    )
    stationary_count = sum(
        surface.classification == "stationary_bushing_feed_wall" for surface in classified
    )
    if journal_count != 3:
        raise GeometryClassificationError(
            f"expected 3 journal-wall patches after zoning, found {journal_count}"
        )
    if stationary_count != 4:
        raise GeometryClassificationError(
            f"expected 4 stationary bore/feed patches after zoning, found {stationary_count}"
        )
    return tuple(classified)


def inventory_volumes() -> tuple[VolumeRecord, ...]:
    records: list[VolumeRecord] = []
    for dim, tag in gmsh.model.getEntities(3):
        records.append(
            VolumeRecord(
                tag=int(tag),
                mass=float(gmsh.model.occ.getMass(dim, tag)),
                centre=tuple(
                    float(value) for value in gmsh.model.occ.getCenterOfMass(dim, tag)
                ),
                bbox=tuple(
                    float(value) for value in gmsh.model.occ.getBoundingBox(dim, tag)
                ),
            )
        )
    return tuple(records)


def inventory_surfaces(volume_labels: dict[int, str]) -> tuple[SurfaceRecord, ...]:
    records: list[SurfaceRecord] = []
    for dim, tag in gmsh.model.getEntities(2):
        upward, downward = gmsh.model.getAdjacencies(dim, tag)
        adjacent_tags = tuple(sorted(int(value) for value in upward))
        curve_tags = tuple(sorted(int(value) for value in downward))
        _integers, reals = gmsh.model.getEntityProperties(dim, tag)
        records.append(
            SurfaceRecord(
                tag=int(tag),
                entity_type=str(gmsh.model.getType(dim, tag)),
                area=float(gmsh.model.occ.getMass(dim, tag)),
                centre=tuple(
                    float(value) for value in gmsh.model.occ.getCenterOfMass(dim, tag)
                ),
                bbox=tuple(
                    float(value) for value in gmsh.model.occ.getBoundingBox(dim, tag)
                ),
                adjacent_volume_tags=adjacent_tags,
                adjacent_volume_labels=tuple(
                    sorted(volume_labels.get(value, f"unclassified:{value}") for value in adjacent_tags)
                ),
                boundary_curve_tags=curve_tags,
                boundary_curve_types=tuple(
                    str(gmsh.model.getType(1, curve_tag)) for curve_tag in curve_tags
                ),
                properties=tuple(float(value) for value in reals),
            )
        )
    return tuple(records)


def require(
    records: list[dict[str, Any]],
    name: str,
    condition: bool,
    actual: Any,
    expected: Any,
    tolerance: Any = None,
) -> None:
    record = {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "actual": actual,
        "expected": expected,
        "tolerance": tolerance,
        "mandatory": True,
    }
    records.append(record)
    if not condition:
        raise GeometryValidationError(
            f"{name}: actual={actual!r}, expected={expected!r}, tolerance={tolerance!r}"
        )


def require_relative(
    records: list[dict[str, Any]],
    name: str,
    actual: float,
    expected: float,
    tolerance: float,
) -> None:
    error = relative_error(actual, expected)
    require(
        records,
        name,
        error <= tolerance,
        {"value": actual, "relative_error": error},
        expected,
        tolerance,
    )


def configure_occ_options(
    records: list[dict[str, Any]],
    prefix: str,
    scale: float,
    boolean_simplify: float = 0.0,
) -> dict[str, float]:
    """Set and verify every OCC import/Boolean option used by this project."""
    expected = dict(STRICT_OCC_OPTIONS)
    expected["Geometry.OCCBooleanSimplify"] = boolean_simplify
    expected["Geometry.OCCScaling"] = scale
    actual: dict[str, float] = {}
    for name, value in expected.items():
        gmsh.option.setNumber(name, value)
        actual[name] = float(gmsh.option.getNumber(name))
    require(
        records,
        f"{prefix}.strict_occ_options",
        actual == expected,
        actual,
        expected,
    )
    return actual


def combined_bbox(volumes: Sequence[VolumeRecord]) -> BBox:
    if not volumes:
        raise GeometryValidationError("cannot bound an empty volume set")
    return (
        min(volume.bbox[0] for volume in volumes),
        min(volume.bbox[1] for volume in volumes),
        min(volume.bbox[2] for volume in volumes),
        max(volume.bbox[3] for volume in volumes),
        max(volume.bbox[4] for volume in volumes),
        max(volume.bbox[5] for volume in volumes),
    )


def validate_model_bbox(
    records: list[dict[str, Any]],
    prefix: str,
    volumes: Sequence[VolumeRecord],
    reference: CadReference,
    scale: float,
) -> BBox:
    """Check physical bounds without mistaking OCC's Bnd_Box padding for geometry."""
    radius = reference.bore_radius(0.0) * scale
    expected = (
        -radius,
        -radius,
        0.0,
        radius,
        reference.y_feed_end * scale,
        reference.length * scale,
    )
    actual = combined_bbox(volumes)
    actual_centre = tuple((actual[index] + actual[index + 3]) / 2.0 for index in range(3))
    expected_centre = tuple((expected[index] + expected[index + 3]) / 2.0 for index in range(3))
    actual_spans = tuple(actual[index + 3] - actual[index] for index in range(3))
    expected_spans = tuple(expected[index + 3] - expected[index] for index in range(3))
    centre_error = max_abs_difference(actual_centre, expected_centre)
    span_errors = tuple(
        relative_error(actual_span, expected_span)
        for actual_span, expected_span in zip(actual_spans, expected_spans)
    )
    require(
        records,
        f"{prefix}.bounding_box",
        centre_error <= POSITION_ABS_TOL_MM * scale
        and max(span_errors) <= BBOX_SPAN_REL_TOL,
        {
            "bbox": actual,
            "centre_error": centre_error,
            "span_relative_errors": span_errors,
        },
        expected,
        {
            "centre_absolute": POSITION_ABS_TOL_MM * scale,
            "span_relative": BBOX_SPAN_REL_TOL,
        },
    )
    return actual


def validate_zone_model(
    records: list[dict[str, Any]],
    prefix: str,
    reference: CadReference,
    scale: float,
) -> tuple[
    dict[str, VolumeRecord],
    tuple[SurfaceRecord, ...],
    dict[str, list[SurfaceRecord]],
]:
    """Validate a current Gmsh model containing the three conformal zones."""
    volumes = inventory_volumes()
    require(records, f"{prefix}.volume_count", len(volumes) == 3, len(volumes), 3)
    expected_total = reference.total_volume * scale**3
    require_relative(
        records,
        f"{prefix}.total_mass",
        sum(volume.mass for volume in volumes),
        expected_total,
        VOLUME_REL_TOL,
    )
    by_name = classify_volumes(volumes, reference, scale)
    require(
        records,
        f"{prefix}.geometric_volume_classification",
        set(by_name) == {"ring_A", "hole_band", "ring_B"},
        sorted(by_name),
        ["hole_band", "ring_A", "ring_B"],
    )
    for name, volume in by_name.items():
        require_relative(
            records,
            f"{prefix}.{name}.mass",
            volume.mass,
            reference.zone_volumes[name] * scale**3,
            VOLUME_REL_TOL,
        )
    labels = {volume.tag: name for name, volume in by_name.items()}
    surfaces = classify_surfaces(inventory_surfaces(labels), reference, scale)
    groups = validate_boundaries(records, surfaces, reference, scale, prefix)
    validate_model_bbox(records, prefix, tuple(by_name.values()), reference, scale)
    return by_name, surfaces, groups


def validate_fragment_mapping(
    records: list[dict[str, Any]],
    source_names: Sequence[str],
    output_dimtags: Sequence[tuple[int, int]],
    output_mapping: Sequence[Sequence[tuple[int, int]]],
) -> dict[str, list[tuple[int, int]]]:
    mapped = {
        name: [(int(dim), int(tag)) for dim, tag in mapping]
        for name, mapping in zip(source_names, output_mapping)
    }
    mapped_volumes = {
        name: [dimtag for dimtag in dimtags if dimtag[0] == 3]
        for name, dimtags in mapped.items()
    }
    output_volumes = {(int(dim), int(tag)) for dim, tag in output_dimtags if dim == 3}
    flattened = [dimtag for dimtags in mapped_volumes.values() for dimtag in dimtags]
    valid = (
        len(output_mapping) == len(source_names)
        and all(len(dimtags) == 1 for dimtags in mapped_volumes.values())
        and len(set(flattened)) == len(source_names)
        and set(flattened) == output_volumes
        and len(output_volumes) == 3
    )
    require(
        records,
        "fragment.source_to_output_mapping",
        valid,
        {name: dimtags for name, dimtags in mapped_volumes.items()},
        "one unique final 3D volume per source zone and no extra 3D output",
    )
    return mapped


def topology_signature(
    by_name: dict[str, VolumeRecord], surfaces: Sequence[SurfaceRecord]
) -> dict[str, Any]:
    groups = grouped_surfaces(surfaces)
    return {
        "volume_count": len(by_name),
        "surface_count": len(surfaces),
        "zone_volumes": {name: volume.mass for name, volume in by_name.items()},
        "surface_role_counts": {name: len(items) for name, items in sorted(groups.items())},
    }


def compare_topology_signatures(
    simplify_off: dict[str, Any], simplify_on: dict[str, Any]
) -> dict[str, Any]:
    structural_change = any(
        simplify_off[key] != simplify_on[key]
        for key in ("volume_count", "surface_count", "surface_role_counts")
    )
    volume_errors = {
        name: relative_error(
            simplify_on["zone_volumes"][name], simplify_off["zone_volumes"][name]
        )
        for name in simplify_off["zone_volumes"]
    }
    return {
        "changed": structural_change or max(volume_errors.values()) > VOLUME_REL_TOL,
        "simplify_off": simplify_off,
        "simplify_on": simplify_on,
        "zone_volume_relative_errors": volume_errors,
    }


def validate_brep_with_ocp(
    records: list[dict[str, Any]],
    prefix: str,
    path: Path,
    reference: CadReference,
    scale: float,
    coordinate_unit: str,
) -> dict[str, Any]:
    """Independently validate a Gmsh-written BREP through build123d/OCP."""
    from build123d import import_brep
    from OCP.BRep import BRep_Tool

    shape = import_brep(path)
    solids = sorted(shape.solids(), key=lambda solid: solid.center().Z)
    require(records, f"{prefix}.solid_count", len(solids) == 3, len(solids), 3)
    valid_flags = [bool(solid.is_valid) for solid in solids]
    manifold_flags = [bool(solid.is_manifold) for solid in solids]
    require(
        records,
        f"{prefix}.valid_manifold",
        all(valid_flags) and all(manifold_flags),
        {"valid": valid_flags, "manifold": manifold_flags},
        {"valid": [True] * 3, "manifold": [True] * 3},
    )
    labels = ("ring_A", "hole_band", "ring_B")
    volumes = {name: float(solid.volume) for name, solid in zip(labels, solids)}
    require_relative(
        records,
        f"{prefix}.total_volume",
        sum(volumes.values()),
        reference.total_volume * scale**3,
        VOLUME_REL_TOL,
    )
    for name in labels:
        require_relative(
            records,
            f"{prefix}.{name}.volume",
            volumes[name],
            reference.zone_volumes[name] * scale**3,
            VOLUME_REL_TOL,
        )

    tolerance_sets = {
        "vertex": [BRep_Tool.Tolerance_s(item.wrapped) for item in shape.vertices()],
        "edge": [BRep_Tool.Tolerance_s(item.wrapped) for item in shape.edges()],
        "face": [BRep_Tool.Tolerance_s(item.wrapped) for item in shape.faces()],
    }
    maxima = {name: max(values, default=0.0) for name, values in tolerance_sets.items()}
    maxima_mm = {name: value / scale for name, value in maxima.items()}
    maximum_fraction = max(maxima_mm.values(), default=0.0) / reference.h_radial_min
    require(
        records,
        f"{prefix}.topology_tolerance_vs_minimum_gap",
        maximum_fraction <= TOPOLOGY_TOLERANCE_MAX_GAP_FRACTION,
        {
            "maxima_in_file_units": maxima,
            "maxima_mm": maxima_mm,
            "maximum_gap_fraction": maximum_fraction,
        },
        f"<= {TOPOLOGY_TOLERANCE_MAX_GAP_FRACTION:.3%} of h_min",
        TOPOLOGY_TOLERANCE_MAX_GAP_FRACTION,
    )
    return {
        "path": str(path),
        "coordinate_unit": coordinate_unit,
        "coordinate_scale_per_mm": scale,
        "scale_to_m": SI_SCALE / scale,
        "sha256": sha256_file(path),
        "solid_count": len(solids),
        "all_valid": all(valid_flags),
        "all_manifold": all(manifold_flags),
        "zone_volumes": volumes,
        "total_volume": sum(volumes.values()),
        "topology_tolerance_maxima_file_units": maxima,
        "topology_tolerance_maxima_mm": maxima_mm,
        "topology_tolerance_maximum_gap_fraction": maximum_fraction,
        "minimum_radial_gap_mm": reference.h_radial_min,
    }


def write_validation_log(
    path: Path,
    title: str,
    gmsh_lines: Sequence[str],
    records: Sequence[dict[str, Any]],
) -> None:
    lines = [title, "", *(str(line) for line in gmsh_lines), "", "Validation records"]
    lines.extend(f"[{record['status']}] {record['name']}: {record['actual']}" for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_surfaces(surfaces: Sequence[SurfaceRecord]) -> dict[str, list[SurfaceRecord]]:
    groups: dict[str, list[SurfaceRecord]] = {}
    for surface in surfaces:
        if surface.classification is None:
            raise GeometryClassificationError(f"surface {surface.tag} is unclassified")
        groups.setdefault(surface.classification, []).append(surface)
    return groups


def validate_boundaries(
    records: list[dict[str, Any]],
    surfaces: Sequence[SurfaceRecord],
    reference: CadReference,
    scale: float,
    prefix: str,
) -> dict[str, list[SurfaceRecord]]:
    groups = grouped_surfaces(surfaces)
    require(
        records,
        f"{prefix}.all_surfaces_classified",
        sum(len(values) for values in groups.values()) == len(surfaces),
        sum(len(values) for values in groups.values()),
        len(surfaces),
    )

    expected_adjacency = {
        "interface_z1": {"ring_A", "hole_band"},
        "interface_z2": {"hole_band", "ring_B"},
    }
    for role, expected_labels in expected_adjacency.items():
        surface = groups[role][0]
        actual_labels = set(surface.adjacent_volume_labels)
        require(
            records,
            f"{prefix}.{role}.two_sided_adjacency",
            len(surface.adjacent_volume_tags) == 2 and actual_labels == expected_labels,
            sorted(actual_labels),
            sorted(expected_labels),
        )

    inlet = groups["pressure_inlet"][0]
    expected_area = reference.inlet_area * scale**2
    require_relative(
        records,
        f"{prefix}.pressure_inlet.area_vs_params",
        inlet.area,
        expected_area,
        AREA_REL_TOL,
    )
    require_relative(
        records,
        f"{prefix}.pressure_inlet.params_area_vs_diameter",
        reference.inlet_area,
        math.pi * reference.hole_radius**2,
        AREA_REL_TOL,
    )
    expected_centre = tuple(value * scale for value in reference.inlet_centre)
    centre_error = max_abs_difference(inlet.centre, expected_centre)
    centre_tolerance = POSITION_ABS_TOL_MM * scale
    require(
        records,
        f"{prefix}.pressure_inlet.centre",
        centre_error <= centre_tolerance,
        {"centre": inlet.centre, "max_abs_error": centre_error},
        expected_centre,
        centre_tolerance,
    )
    normal = normalized(inlet.properties[:3]) if len(inlet.properties) >= 3 else (0.0, 0.0, 0.0)
    require(
        records,
        f"{prefix}.pressure_inlet.support_plane_direction",
        parallel(normal, (0.0, 1.0, 0.0)) and normal[1] > 0.0,
        normal,
        (0.0, 1.0, 0.0),
        1.0e-10,
    )
    probe_distance = 0.05 * reference.hole_radius * scale
    adjacent_volume = inlet.adjacent_volume_tags[0]
    inward_point = (inlet.centre[0], inlet.centre[1] - probe_distance, inlet.centre[2])
    outward_point = (inlet.centre[0], inlet.centre[1] + probe_distance, inlet.centre[2])
    inward_is_inside = bool(
        gmsh.model.isInside(3, adjacent_volume, list(inward_point), parametric=False)
    )
    outward_is_inside = bool(
        gmsh.model.isInside(3, adjacent_volume, list(outward_point), parametric=False)
    )
    require(
        records,
        f"{prefix}.pressure_inlet.outward_orientation",
        inward_is_inside and not outward_is_inside,
        {
            "probe_distance": probe_distance,
            "minus_y_point_inside": inward_is_inside,
            "plus_y_point_inside": outward_is_inside,
        },
        "-Y probe inside and +Y probe outside",
    )
    require(
        records,
        f"{prefix}.pressure_inlet.circular_boundary",
        "Circle" in inlet.boundary_curve_types,
        inlet.boundary_curve_types,
        "contains Circle",
    )
    return groups


def import_brep(path: Path) -> tuple[VolumeRecord, ...]:
    gmsh.model.occ.importShapes(str(path), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    return inventory_volumes()


def volume_rows(
    model: str, unit: str, volumes: Iterable[VolumeRecord]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for volume in volumes:
        rows.append(
            {
                "model": model,
                "unit": unit,
                "volume_tag": volume.tag,
                "label": volume.label or "",
                "mass": volume.mass,
                "centre_x": volume.centre[0],
                "centre_y": volume.centre[1],
                "centre_z": volume.centre[2],
                "bbox_xmin": volume.bbox[0],
                "bbox_ymin": volume.bbox[1],
                "bbox_zmin": volume.bbox[2],
                "bbox_xmax": volume.bbox[3],
                "bbox_ymax": volume.bbox[4],
                "bbox_zmax": volume.bbox[5],
            }
        )
    return rows


def surface_rows(
    model: str, unit: str, surfaces: Iterable[SurfaceRecord]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface in surfaces:
        rows.append(
            {
                "model": model,
                "unit": unit,
                "surface_tag": surface.tag,
                "classification": surface.classification or "",
                "entity_type": surface.entity_type,
                "area": surface.area,
                "centre_x": surface.centre[0],
                "centre_y": surface.centre[1],
                "centre_z": surface.centre[2],
                "bbox_xmin": surface.bbox[0],
                "bbox_ymin": surface.bbox[1],
                "bbox_zmin": surface.bbox[2],
                "bbox_xmax": surface.bbox[3],
                "bbox_ymax": surface.bbox[4],
                "bbox_zmax": surface.bbox[5],
                "adjacent_volume_tags": ";".join(map(str, surface.adjacent_volume_tags)),
                "adjacent_volume_labels": ";".join(surface.adjacent_volume_labels),
                "boundary_curve_tags": ";".join(map(str, surface.boundary_curve_tags)),
                "boundary_curve_types": ";".join(surface.boundary_curve_types),
                "occ_properties": ";".join(f"{value:.17g}" for value in surface.properties),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise GeometryValidationError(f"refusing to write empty inventory {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def serializable_volume(volume: VolumeRecord) -> dict[str, Any]:
    return asdict(volume)


def serializable_surface(surface: SurfaceRecord) -> dict[str, Any]:
    return asdict(surface)


def _check_input_hashes(
    records: list[dict[str, Any]], inputs: PreflightInputs, params_raw: dict[str, Any]
) -> None:
    hashes = params_raw.get("sha256", {})
    for label, path in (("unsplit", inputs.unsplit), ("zones", inputs.zones)):
        expected = hashes.get(path.name)
        actual = sha256_file(path)
        require(
            records,
            f"input.{label}.sha256",
            isinstance(expected, str) and actual == expected,
            actual,
            expected,
        )


def _check_native_brep_records(
    records: list[dict[str, Any]], params_raw: dict[str, Any]
) -> None:
    round_trips = params_raw.get("measurements", {}).get("brep_round_trip", {})
    for filename, expected_solids in (("film_unsplit.brep", 1), ("film_zones.brep", 3)):
        result = round_trips.get(filename, {})
        passed = (
            result.get("solid_count") == expected_solids
            and result.get("all_valid") is True
            and result.get("all_manifold") is True
            and float(result.get("relative_volume_error", math.inf)) <= 1.0e-12
        )
        require(
            records,
            f"input.native_brep_record.{filename}",
            passed,
            result,
            {
                "solid_count": expected_solids,
                "valid": True,
                "manifold": True,
                "relative_volume_error_max": 1.0e-12,
            },
        )


def _run_occ_checks(
    inputs: PreflightInputs,
    reference: CadReference,
    records: list[dict[str, Any]],
    stage: Path,
) -> dict[str, Any]:
    all_volume_rows: list[dict[str, Any]] = []
    all_surface_rows: list[dict[str, Any]] = []
    option_sets: dict[str, dict[str, float]] = {}

    option_sets["unsplit_mm"] = configure_occ_options(records, "unsplit", 1.0)
    gmsh.model.add("film_unsplit_mm")
    unsplit = import_brep(inputs.unsplit)
    require(records, "unsplit.volume_count", len(unsplit) == 1, len(unsplit), 1)
    unsplit_volume = unsplit[0]
    require_relative(
        records,
        "unsplit.occ_mass_vs_native",
        unsplit_volume.mass,
        reference.total_volume,
        VOLUME_REL_TOL,
    )
    bbox_error = max_abs_difference(unsplit_volume.bbox, reference.bounding_box)
    require(
        records,
        "unsplit.bounding_box_vs_native_brep",
        bbox_error <= BBOX_ABS_TOL_MM,
        {"bbox": unsplit_volume.bbox, "max_abs_error_mm": bbox_error},
        reference.bounding_box,
        BBOX_ABS_TOL_MM,
    )
    axial_span = unsplit_volume.bbox[5] - unsplit_volume.bbox[2]
    unit_error = relative_error(axial_span, reference.length)
    require(
        records,
        "unsplit.coordinates_are_millimetres",
        unit_error <= 1.0e-8,
        {"z_span": axial_span, "reference_length_mm": reference.length},
        "1 model coordinate = 1 mm",
        1.0e-8,
    )
    all_volume_rows.extend(volume_rows("film_unsplit", "mm", unsplit))

    option_sets["zones_source_mm"] = configure_occ_options(records, "zones_import", 1.0)
    gmsh.model.add("film_zones_mm")
    zones_imported = import_brep(inputs.zones)
    require(records, "zones_import.volume_count", len(zones_imported) == 3, len(zones_imported), 3)
    imported_total = sum(volume.mass for volume in zones_imported)
    require_relative(
        records,
        "zones_import.total_mass_vs_native",
        imported_total,
        reference.total_volume,
        VOLUME_REL_TOL,
    )
    imported_by_name = classify_volumes(zones_imported, reference)
    require(
        records,
        "zones_import.geometric_classification",
        set(imported_by_name) == {"ring_A", "hole_band", "ring_B"},
        sorted(imported_by_name),
        ["hole_band", "ring_A", "ring_B"],
    )
    for name, volume in imported_by_name.items():
        require_relative(
            records,
            f"zones_import.{name}.mass_vs_native",
            volume.mass,
            reference.zone_volumes[name],
            VOLUME_REL_TOL,
        )
    all_volume_rows.extend(volume_rows("zones_imported", "mm", imported_by_name.values()))

    source_names = ("ring_A", "hole_band", "ring_B")
    source_dimtags = [(3, imported_by_name[name].tag) for name in source_names]
    fragment_output, fragment_mapping = gmsh.model.occ.fragment(
        [source_dimtags[0]],
        source_dimtags[1:],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    mapped_sources = validate_fragment_mapping(
        records, source_names, fragment_output, fragment_mapping
    )
    fragmented_by_name, mm_surfaces, mm_groups = validate_zone_model(
        records, "fragment", reference, 1.0
    )
    mapped_volume_tags = {
        name: next(tag for dim, tag in dimtags if dim == 3)
        for name, dimtags in mapped_sources.items()
    }
    classified_volume_tags = {
        name: volume.tag for name, volume in fragmented_by_name.items()
    }
    require(
        records,
        "fragment.source_mapping_preserves_zone_identity",
        mapped_volume_tags == classified_volume_tags,
        mapped_volume_tags,
        classified_volume_tags,
    )
    fragmented_total = sum(volume.mass for volume in fragmented_by_name.values())
    require_relative(
        records,
        "fragment.total_mass_conservation",
        fragmented_total,
        imported_total,
        VOLUME_REL_TOL,
    )
    all_volume_rows.extend(volume_rows("zones_fragmented", "mm", fragmented_by_name.values()))
    all_surface_rows.extend(surface_rows("zones_fragmented", "mm", mm_surfaces))
    simplify_off_signature = topology_signature(fragmented_by_name, mm_surfaces)

    option_sets["boolean_simplify_on_probe"] = configure_occ_options(
        records, "boolean_simplify_on_probe", 1.0, boolean_simplify=1.0
    )
    gmsh.model.add("boolean_simplify_on_probe")
    simplify_source = import_brep(inputs.zones)
    simplify_imported = classify_volumes(simplify_source, reference)
    simplify_dimtags = [(3, simplify_imported[name].tag) for name in source_names]
    gmsh.model.occ.fragment(
        [simplify_dimtags[0]], simplify_dimtags[1:], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()
    simplify_on_volumes, simplify_on_surfaces, _simplify_groups = validate_zone_model(
        records, "boolean_simplify_on_probe", reference, 1.0
    )
    simplify_comparison = compare_topology_signatures(
        simplify_off_signature,
        topology_signature(simplify_on_volumes, simplify_on_surfaces),
    )
    records.append(
        {
            "name": "fragment.boolean_simplify_topology_comparison",
            "status": "PASS",
            "actual": simplify_comparison,
            "expected": "difference reported; production remains OCCBooleanSimplify=0",
            "tolerance": VOLUME_REL_TOL,
            "mandatory": False,
        }
    )

    fragmented_path = stage / "film_zones_fragmented.brep"
    gmsh.model.setCurrent("film_zones_mm")
    gmsh.write(str(fragmented_path))
    require(
        records,
        "fragment.brep_export",
        fragmented_path.is_file() and fragmented_path.stat().st_size > 0,
        fragmented_path.stat().st_size if fragmented_path.exists() else 0,
        "> 0 bytes",
    )

    option_sets["fragment_disk_mm"] = configure_occ_options(
        records, "fragment_disk_mm", 1.0
    )
    gmsh.model.add("fragment_disk_mm")
    import_brep(fragmented_path)
    disk_mm_by_name, disk_mm_surfaces, disk_mm_groups = validate_zone_model(
        records, "fragment_disk_mm", reference, 1.0
    )
    disk_mm_comparison = compare_topology_signatures(
        simplify_off_signature,
        topology_signature(disk_mm_by_name, disk_mm_surfaces),
    )
    require(
        records,
        "fragment_disk_mm.topology_round_trip",
        not disk_mm_comparison["changed"],
        disk_mm_comparison,
        "no topology or volume change",
    )
    all_volume_rows.extend(volume_rows("fragment_disk", "mm", disk_mm_by_name.values()))
    all_surface_rows.extend(surface_rows("fragment_disk", "mm", disk_mm_surfaces))

    # Import-time OCC scaling transforms each shared subshape exactly once.
    option_sets["si_scaled_from_mm"] = configure_occ_options(
        records, "si_scaled_from_mm", SI_SCALE
    )
    gmsh.model.add("film_zones_si_scaled")
    import_brep(fragmented_path)
    si_by_name, si_surfaces, si_groups = validate_zone_model(
        records, "si_scaled_from_mm", reference, SI_SCALE
    )
    si_total = sum(volume.mass for volume in si_by_name.values())
    require_relative(
        records,
        "si.volume_scale",
        si_total / fragmented_total,
        SI_SCALE**3,
        SCALE_REL_TOL,
    )
    mm_area = sum(surface.area for surface in mm_surfaces)
    si_area = sum(surface.area for surface in si_surfaces)
    require_relative(
        records,
        "si.area_scale",
        si_area / mm_area,
        SI_SCALE**2,
        SCALE_REL_TOL,
    )
    for name in ("ring_A", "hole_band", "ring_B"):
        require_relative(
            records,
            f"si.{name}.volume_scale",
            si_by_name[name].mass / fragmented_by_name[name].mass,
            SI_SCALE**3,
            SCALE_REL_TOL,
        )

    length_metrics_mm = {
        "axial_length": mm_groups["outlet_zL"][0].centre[2]
        - mm_groups["outlet_z0"][0].centre[2],
        "split_z1": mm_groups["interface_z1"][0].centre[2],
        "split_z2": mm_groups["interface_z2"][0].centre[2],
        "feed_end_y": mm_groups["pressure_inlet"][0].centre[1],
        "feed_centre_z": mm_groups["pressure_inlet"][0].centre[2],
    }
    length_metrics_si = {
        "axial_length": si_groups["outlet_zL"][0].centre[2]
        - si_groups["outlet_z0"][0].centre[2],
        "split_z1": si_groups["interface_z1"][0].centre[2],
        "split_z2": si_groups["interface_z2"][0].centre[2],
        "feed_end_y": si_groups["pressure_inlet"][0].centre[1],
        "feed_centre_z": si_groups["pressure_inlet"][0].centre[2],
    }
    length_scale_errors: dict[str, float] = {}
    for name, mm_value in length_metrics_mm.items():
        ratio = length_metrics_si[name] / mm_value
        length_scale_errors[name] = relative_error(ratio, SI_SCALE)
    require(
        records,
        "si.length_scale",
        max(length_scale_errors.values()) <= SCALE_REL_TOL,
        {
            "mm": length_metrics_mm,
            "si": length_metrics_si,
            "relative_errors": length_scale_errors,
        },
        SI_SCALE,
        SCALE_REL_TOL,
    )

    si_path = stage / "film_zones_SI.brep"
    gmsh.write(str(si_path))
    require(
        records,
        "si.brep_export",
        si_path.is_file() and si_path.stat().st_size > 0,
        si_path.stat().st_size if si_path.exists() else 0,
        "> 0 bytes",
    )
    all_volume_rows.extend(volume_rows("zones_scaled", "m", si_by_name.values()))
    all_surface_rows.extend(surface_rows("zones_scaled", "m", si_surfaces))

    option_sets["si_disk"] = configure_occ_options(records, "si_disk", 1.0)
    gmsh.model.add("film_zones_si_disk")
    import_brep(si_path)
    si_disk_by_name, si_disk_surfaces, si_disk_groups = validate_zone_model(
        records, "si_disk", reference, SI_SCALE
    )
    si_disk_comparison = compare_topology_signatures(
        topology_signature(si_by_name, si_surfaces),
        topology_signature(si_disk_by_name, si_disk_surfaces),
    )
    require(
        records,
        "si_disk.topology_round_trip",
        not si_disk_comparison["changed"],
        si_disk_comparison,
        "no topology or volume change",
    )
    si_disk_axial_length = (
        si_disk_groups["outlet_zL"][0].centre[2]
        - si_disk_groups["outlet_z0"][0].centre[2]
    )
    require_relative(
        records,
        "si_disk.file_contains_si_coordinates",
        si_disk_axial_length,
        reference.length * SI_SCALE,
        SCALE_REL_TOL,
    )
    all_volume_rows.extend(volume_rows("zones_disk_round_trip", "m", si_disk_by_name.values()))
    all_surface_rows.extend(surface_rows("zones_disk_round_trip", "m", si_disk_surfaces))

    mm_ocp = validate_brep_with_ocp(
        records, "ocp_fragment_mm", fragmented_path, reference, 1.0, "mm"
    )
    si_ocp = validate_brep_with_ocp(
        records, "ocp_fragment_si", si_path, reference, SI_SCALE, "m"
    )
    brep_manifest = {
        "schema_version": 1,
        "overall": "PASS",
        "coordinate_units_explicit": True,
        "minimum_radial_gap_mm": reference.h_radial_min,
        "topology_tolerance_maximum_gap_fraction": TOPOLOGY_TOLERANCE_MAX_GAP_FRACTION,
        "files": {
            "film_zones_fragmented.brep": {**mm_ocp, "path": "film_zones_fragmented.brep"},
            "film_zones_SI.brep": {**si_ocp, "path": "film_zones_SI.brep"},
        },
    }
    (stage / "brep_manifest.json").write_text(
        json.dumps(brep_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "coordinate_units": {
            "input_brep": "mm",
            "film_zones_fragmented.brep": "mm",
            "film_zones_SI.brep": "m",
        },
        "occ_option_sets": option_sets,
        "unsplit": serializable_volume(unsplit_volume),
        "zones_imported": {
            name: serializable_volume(volume) for name, volume in imported_by_name.items()
        },
        "zones_fragmented_mm": {
            name: serializable_volume(volume) for name, volume in fragmented_by_name.items()
        },
        "zones_fragmented_si": {
            name: serializable_volume(volume) for name, volume in si_by_name.items()
        },
        "zones_fragmented_disk_mm": {
            name: serializable_volume(volume) for name, volume in disk_mm_by_name.items()
        },
        "zones_fragmented_disk_si": {
            name: serializable_volume(volume) for name, volume in si_disk_by_name.items()
        },
        "surfaces_mm": [serializable_surface(surface) for surface in mm_surfaces],
        "surfaces_si": [serializable_surface(surface) for surface in si_surfaces],
        "surface_rows": all_surface_rows,
        "volume_rows": all_volume_rows,
        "fragment_output_dimtags": [(int(dim), int(tag)) for dim, tag in fragment_output],
        "fragment_source_mapping": mapped_sources,
        "boolean_simplify_comparison": simplify_comparison,
        "brep_manifest": brep_manifest,
        "boundary_groups_mm": {
            name: [surface.tag for surface in surfaces] for name, surfaces in mm_groups.items()
        },
        "boundary_groups_disk_mm": {
            name: [surface.tag for surface in surfaces]
            for name, surfaces in disk_mm_groups.items()
        },
        "boundary_groups_si": {
            name: [surface.tag for surface in surfaces] for name, surfaces in si_groups.items()
        },
        "boundary_groups_disk_si": {
            name: [surface.tag for surface in surfaces]
            for name, surfaces in si_disk_groups.items()
        },
        "scaling": {
            "method": "Geometry.OCCScaling during import of the fragmented BREP",
            "factor": SI_SCALE,
            "length_metrics_mm": length_metrics_mm,
            "length_metrics_si": length_metrics_si,
            "length_relative_errors": length_scale_errors,
            "area_ratio": si_area / mm_area,
            "volume_ratio": si_total / fragmented_total,
        },
    }


def _publish_failure_bundle(
    outdir: Path,
    report: dict[str, Any],
    gmsh_lines: Sequence[str],
    records: Sequence[dict[str, Any]],
) -> None:
    stage = make_staging_directory(outdir)
    try:
        (stage / "preflight_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_validation_log(
            stage / "gmsh_preflight.log", "Gmsh BREP preflight", gmsh_lines, records
        )
        publish_generation(
            stage,
            outdir,
            stage="meshing",
            operation="brep-preflight",
            status=str(report["overall"]),
            resolved_inputs=report.get("inputs", {}),
            input_units=report.get("coordinate_units", {}),
            producer_files=(Path(__file__),),
            tool_versions=report.get("dependency_versions", {}),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _publish_preflight_success(
    stage: Path,
    outdir: Path,
    base_report: dict[str, Any],
    reference: CadReference,
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    gmsh_lines: Sequence[str],
) -> dict[str, Any]:
    write_csv(stage / "surfaces.csv", diagnostics.pop("surface_rows"))
    write_csv(stage / "volumes.csv", diagnostics.pop("volume_rows"))
    write_validation_log(
        stage / "gmsh_preflight.log", "Gmsh BREP preflight", gmsh_lines, records
    )
    output_names = (
        "surfaces.csv",
        "volumes.csv",
        "film_zones_fragmented.brep",
        "film_zones_SI.brep",
        "brep_manifest.json",
        "gmsh_preflight.log",
    )
    output_units = {
        "surfaces.csv": "explicit per row",
        "volumes.csv": "explicit per row",
        "film_zones_fragmented.brep": "mm",
        "film_zones_SI.brep": "m",
        "brep_manifest.json": "explicit per BREP entry",
        "gmsh_preflight.log": "n/a",
    }
    report = {
        **base_report,
        "cad_params_overall": reference.source_overall,
        "cad_reference": asdict(reference),
        "cad_params_note": (
            "STEP exchange status is independent of this native-BREP preflight."
        ),
        "validation_records": records,
        "diagnostics": diagnostics,
        "outputs": {
            name: {
                "sha256": sha256_file(stage / name),
                "bytes": (stage / name).stat().st_size,
                "coordinate_unit": output_units[name],
                "scale_to_m": 0.001
                if name == "film_zones_fragmented.brep"
                else (1.0 if name == "film_zones_SI.brep" else None),
            }
            for name in output_names
        },
        "overall": "PASS",
        "error": None,
    }
    (stage / "preflight_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    publish_generation(
        stage,
        outdir,
        stage="meshing",
        operation="brep-preflight",
        status="PASS",
        resolved_inputs=base_report["inputs"],
        input_units=base_report["coordinate_units"],
        producer_files=(Path(__file__),),
        upstream_artifacts=tuple(
            Path(base_report["inputs"][name])
            for name in ("unsplit", "zones", "params")
            if Path(base_report["inputs"][name]).is_file()
        ),
        tool_versions=base_report["dependency_versions"],
    )
    return report


def run_preflight(inputs: PreflightInputs) -> dict[str, Any]:
    """Run the complete headless preflight and publish only validated geometry."""
    inputs = PreflightInputs(
        unsplit=inputs.unsplit.resolve(),
        zones=inputs.zones.resolve(),
        params=inputs.params.resolve(),
        outdir=inputs.outdir.resolve(),
        gui=inputs.gui,
    )
    records: list[dict[str, Any]] = []
    base_report: dict[str, Any] = {
        "inputs": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(inputs).items()},
        "tolerances": {
            "volume_relative": VOLUME_REL_TOL,
            "area_relative": AREA_REL_TOL,
            "scale_relative": SCALE_REL_TOL,
            "bbox_absolute_mm": BBOX_ABS_TOL_MM,
            "position_absolute_mm": POSITION_ABS_TOL_MM,
            "bbox_span_relative": BBOX_SPAN_REL_TOL,
            "topology_tolerance_maximum_gap_fraction": TOPOLOGY_TOLERANCE_MAX_GAP_FRACTION,
        },
        "dependency_versions": {
            "gmsh": gmsh.__version__,
            "build123d": importlib.metadata.version("build123d"),
            "python": sys.version.split()[0],
        },
        "coordinate_units": {
            "input_brep": "mm",
            "fragmented_brep": "mm",
            "si_brep": "m",
        },
        "mesh_generated": False,
    }

    try:
        for label, path in (("unsplit", inputs.unsplit), ("zones", inputs.zones), ("params", inputs.params)):
            require(records, f"input.{label}.exists", path.is_file(), str(path), "readable file")
        reference, params_raw = load_reference(inputs.params)
        _check_input_hashes(records, inputs, params_raw)
        _check_native_brep_records(records, params_raw)
    except Exception as error:
        report = {
            **base_report,
            "validation_records": records,
            "overall": "FAIL",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _publish_failure_bundle(inputs.outdir, report, [], records)
        raise PreflightRunError(str(error), report) from error

    stage = make_staging_directory(inputs.outdir)
    try:
        gmsh_lines: list[str] = []
        diagnostics: dict[str, Any] = {}
        caught: Exception | None = None
        initialized = False
        logger_started = False

        try:
            gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
            initialized = True
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.logger.start()
            logger_started = True
            diagnostics = _run_occ_checks(inputs, reference, records, stage)
            if inputs.gui:
                gmsh.model.setCurrent("fragment_disk_mm")
                gmsh.option.setNumber("Geometry.Surfaces", 1)
                gmsh.option.setNumber("Geometry.SurfaceType", 2)
                require(
                    records,
                    "gui.solid_geometry_display",
                    gmsh.option.getNumber("Geometry.Surfaces") == 1
                    and gmsh.option.getNumber("Geometry.SurfaceType") == 2,
                    {
                        "Geometry.Surfaces": gmsh.option.getNumber("Geometry.Surfaces"),
                        "Geometry.SurfaceType": gmsh.option.getNumber("Geometry.SurfaceType"),
                    },
                    {"Geometry.Surfaces": 1, "Geometry.SurfaceType": 2},
                )
                gmsh.fltk.run()
        except Exception as error:
            caught = error
        finally:
            if logger_started:
                try:
                    gmsh_lines = [str(line) for line in gmsh.logger.get()]
                    gmsh.logger.stop()
                except Exception as logger_error:  # pragma: no cover - defensive cleanup
                    gmsh_lines.append(f"logger cleanup failed: {logger_error}")
            if initialized:
                gmsh.finalize()

        if caught is not None:
            report = {
                **base_report,
                "cad_params_overall": reference.source_overall,
                "cad_reference": asdict(reference),
                "validation_records": records,
                "diagnostics": diagnostics,
                "overall": "FAIL",
                "error": {"type": type(caught).__name__, "message": str(caught)},
            }
            if stage.exists():
                shutil.rmtree(stage)
            _publish_failure_bundle(inputs.outdir, report, gmsh_lines, records)
            raise PreflightRunError(str(caught), report) from caught

        try:
            return _publish_preflight_success(
                stage,
                inputs.outdir,
                base_report,
                reference,
                records,
                diagnostics,
                gmsh_lines,
            )
        except Exception as error:
            records.append(
                {
                    "name": "output.atomic_publication",
                    "status": "FAIL",
                    "actual": {"type": type(error).__name__, "message": str(error)},
                    "expected": "complete atomic output publication",
                    "tolerance": None,
                    "mandatory": True,
                }
            )
            if stage.exists():
                shutil.rmtree(stage)
            report = {
                **base_report,
                "cad_params_overall": reference.source_overall,
                "cad_reference": asdict(reference),
                "validation_records": records,
                "diagnostics": diagnostics,
                "overall": "FAIL",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            _publish_failure_bundle(inputs.outdir, report, gmsh_lines, records)
            raise PreflightRunError(str(error), report) from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def print_report(report: dict[str, Any]) -> None:
    print("\nGmsh native-BREP preflight")
    print(f"{'check':<54} {'status':<6}")
    print(f"{'-' * 54} {'-' * 6}")
    for record in report.get("validation_records", []):
        print(f"{record['name']:<54} {record['status']:<6}")
    print(f"\nNo 3D mesh generated: {not report.get('mesh_generated', True)}")
    print(f"OVERALL: {report.get('overall', 'FAIL')}")


def parse_args(argv: Sequence[str] | None = None) -> PreflightInputs:
    parser = argparse.ArgumentParser(
        description="Validate conical-bearing native BREP geometry in Gmsh OCC without meshing."
    )
    parser.add_argument("--unsplit", type=Path, default=PreflightInputs.unsplit)
    parser.add_argument("--zones", type=Path, default=PreflightInputs.zones)
    parser.add_argument("--params", type=Path, default=PreflightInputs.params)
    parser.add_argument("--outdir", type=Path, default=PreflightInputs.outdir)
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the validated fragmented millimetre model after every check passes.",
    )
    args = parser.parse_args(argv)
    return PreflightInputs(
        unsplit=args.unsplit,
        zones=args.zones,
        params=args.params,
        outdir=args.outdir,
        gui=args.gui,
    )


def main(argv: Sequence[str] | None = None) -> int:
    inputs = parse_args(argv)
    try:
        report = run_preflight(inputs)
    except PreflightRunError as error:
        print_report(error.report)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
