#!/usr/bin/env python3
"""Exact B-rep fluid volume for an eccentric conical journal bearing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import build123d
from build123d import (
    Align,
    Axis,
    Compound,
    Cylinder,
    GeomType,
    Keep,
    Part,
    Plane,
    Polyline,
    Pos,
    PrecisionMode,
    Rot,
    Shape,
    Solid,
    Unit,
    export_brep,
    export_step,
    import_brep,
    import_step,
    make_face,
    revolve,
    section,
    split,
)

from bearing_cfd.artifacts import make_staging_directory, publish_generation

from bearing_cfd.bearings.conical_journal.geometry.model import (
    GeometryInputs,
    GeometryParameterError,
    ResolvedGeometry,
    resolve_geometry,
)


STEP_ROUNDTRIP_REL_TOL = 1.0e-6
BREP_ROUNDTRIP_REL_TOL = 1.0e-12
PREVIEW_PORT = 3939
REJECTED_STEP_SCHEMA = "bearing-cfd.rejected-step.v1"
PRODUCER = "bearing_cfd.bearings.conical_journal.geometry.cad"


class BearingFilmError(RuntimeError):
    """Base class for expected project failures."""


class ParameterValidationError(BearingFilmError):
    """Invalid input parameters."""


class GeometryConstructionError(BearingFilmError):
    """Failed exact geometry construction."""


class BooleanOperationError(BearingFilmError):
    """Failed or unexpected OCCT Boolean result."""


class TopologyValidationError(BearingFilmError):
    """Invalid, non-manifold, or disconnected topology."""


class GeometryValidationError(BearingFilmError):
    """A mandatory geometric measurement failed."""

    def __init__(
        self,
        message: str,
        records: list[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.records = records
        self.diagnostics = diagnostics


class GeometryExportError(BearingFilmError):
    """STEP or diagnostic-file export failed."""


class RoundTripValidationError(BearingFilmError):
    """An exported CAD file failed re-import validation."""

    def __init__(
        self,
        message: str,
        records: list[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.records = records
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class RunOptions:
    """CAD validation, publication, and presentation policy."""

    max_face_count: int = 100
    export_debug_half: bool = False
    retain_failed_step: bool = False
    preview: bool = False
    outdir: Path = Path("out/conical_journal/geometry/default")


def resolve_params(inputs: GeometryInputs) -> ResolvedGeometry:
    """Translate geometry validation failures to the CAD command contract."""

    try:
        return resolve_geometry(inputs)
    except GeometryParameterError as error:
        raise ParameterValidationError(str(error)) from error


def validate_run_options(options: RunOptions) -> None:
    if options.max_face_count <= 0:
        raise ParameterValidationError(
            "Invalid parameters:\n- max_face_count must be > 0"
        )


def _revolved_frustum(radius_at: Any, z0: float, z1: float) -> Part:
    profile = Polyline(
        (0.0, z0),
        (radius_at(z0), z0),
        (radius_at(z1), z1),
        (0.0, z1),
        (0.0, z0),
    )
    return revolve(Plane.XZ * make_face(profile), axis=Axis.Z, revolution_arc=360.0)


def make_bore_blank(params: ResolvedGeometry) -> Part:
    return _revolved_frustum(params.bore_radius, 0.0, params.length)


def make_journal(params: ResolvedGeometry, *, extended: bool = True) -> Part:
    delta = params.axial_cutter_extension if extended else 0.0
    journal = _revolved_frustum(
        params.journal_radius,
        -delta,
        params.length + delta,
    )
    return Pos(params.ex, params.ey, 0.0) * journal


def _inlet_face_matches(face: Shape, params: ResolvedGeometry) -> bool:
    expected_area = math.pi * params.hole_radius**2
    center = tuple(face.center())
    normal = tuple(face.normal_at())
    return (
        face.geom_type == GeomType.PLANE
        and abs(face.area - expected_area) <= max(1e-7, expected_area * 1e-8)
        and math.dist(center, (0.0, params.y_feed_end, params.hole_axial_pos)) <= 1e-6
        and normal[1] >= 1.0 - 1e-10
    )


def make_feed_cylinder(params: ResolvedGeometry) -> Part:
    feed_plane = Plane(
        origin=(0.0, 0.0, params.hole_axial_pos),
        x_dir=(1.0, 0.0, 0.0),
        z_dir=(0.0, 1.0, 0.0),
    )
    feed = feed_plane * Cylinder(
        params.hole_radius,
        params.y_feed_end,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    inlet_faces = [face for face in feed.faces() if _inlet_face_matches(face, params)]
    if len(inlet_faces) != 1:
        raise GeometryConstructionError(
            f"feed cylinder must expose exactly one remote circular inlet face; found {len(inlet_faces)}"
        )
    edges = list(inlet_faces[0].edges())
    if len(edges) != 1 or edges[0].geom_type != GeomType.CIRCLE:
        raise GeometryConstructionError("remote inlet face does not have one circular boundary")
    return feed


def make_base_film(bore_blank: Shape, journal: Shape) -> Part:
    try:
        return bore_blank - journal
    except Exception as exc:
        raise BooleanOperationError(f"base_film = bore_blank - journal failed: {exc}") from exc


def _bounding_box_record(shape: Shape) -> dict[str, list[float]]:
    bbox = shape.bounding_box()
    return {
        "min": [bbox.min.X, bbox.min.Y, bbox.min.Z],
        "max": [bbox.max.X, bbox.max.Y, bbox.max.Z],
        "size": [bbox.size.X, bbox.size.Y, bbox.size.Z],
    }


def _shape_record(shape: Shape) -> dict[str, Any]:
    solids = list(shape.solids())
    return {
        "volume_mm3": shape.volume,
        "solid_count": len(solids),
        "solid_volumes_mm3": [solid.volume for solid in solids],
        "solid_validity": [
            {"is_valid": solid.is_valid, "is_manifold": solid.is_manifold}
            for solid in solids
        ],
        "all_solids_valid": all(solid.is_valid for solid in solids),
        "all_solids_manifold": all(solid.is_manifold for solid in solids),
        "face_count": len(shape.faces()),
        "is_valid": shape.is_valid,
        "is_manifold": shape.is_manifold,
        "bounding_box_mm": _bounding_box_record(shape),
    }


def make_full_film(
    bore_blank: Shape,
    feed_cylinder: Shape,
    journal: Shape,
) -> tuple[Part, Part]:
    try:
        wet = bore_blank + feed_cylinder
    except Exception as exc:
        raise BooleanOperationError(f"wet = bore_blank + feed_cylinder failed: {exc}") from exc
    if not wet.is_valid or not wet.is_manifold or len(wet.solids()) != 1:
        raise TopologyValidationError(
            "wet bore/feed union must be one valid manifold solid:\n"
            + json.dumps(_shape_record(wet), indent=2)
        )
    try:
        film = wet - journal
    except Exception as exc:
        raise BooleanOperationError(f"film = wet - journal failed: {exc}") from exc
    return wet, film


def split_axial_zones(params: ResolvedGeometry, film: Shape) -> dict[str, Solid]:
    first = list(
        split(film.solid(), Plane.XY.offset(params.z1), keep=Keep.BOTH).solids()
    )
    if len(first) != 2:
        raise GeometryConstructionError(f"first axial split produced {len(first)} solids, expected 2")
    first.sort(key=lambda solid: solid.center().Z)
    second = list(
        split(first[1], Plane.XY.offset(params.z2), keep=Keep.BOTH).solids()
    )
    if len(second) != 2:
        raise GeometryConstructionError(f"second axial split produced {len(second)} solids, expected 2")
    pieces = [first[0], *second]
    pieces.sort(key=lambda solid: solid.center().Z)
    zones = dict(zip(("ring_A", "hole_band", "ring_B"), pieces, strict=True))
    expected = {
        "ring_A": (0.0, params.z1),
        "hole_band": (params.z1, params.z2),
        "ring_B": (params.z2, params.length),
    }
    for name, solid in zones.items():
        bbox = solid.bounding_box()
        z_min, z_max = expected[name]
        if abs(bbox.min.Z - z_min) > 0.001 or abs(bbox.max.Z - z_max) > 0.001:
            raise GeometryConstructionError(
                f"{name} position test failed: got z=[{bbox.min.Z}, {bbox.max.Z}], "
                f"expected [{z_min}, {z_max}]"
            )
    return zones


def make_context_bushing(
    params: ResolvedGeometry,
    bore_blank: Shape,
    feed_cylinder: Shape,
) -> Part:
    outer = _revolved_frustum(params.outer_radius, 0.0, params.length)
    try:
        return outer - bore_blank - feed_cylinder
    except Exception as exc:
        raise BooleanOperationError(
            f"context bushing = outer - bore_blank - feed_cylinder failed: {exc}"
        ) from exc


def _validation_record(
    name: str,
    passed: bool,
    measured: Any,
    expected: Any,
    tolerance: Any,
    *,
    detail: str = "",
    mandatory: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "mandatory": mandatory,
        "measured": measured,
        "expected": expected,
        "tolerance": tolerance,
        "detail": detail,
    }


def _relative_error(measured: float, expected: float) -> float:
    return abs(measured - expected) / max(abs(expected), 1e-300)


def _lateral_face(shape: Shape) -> Shape:
    candidates = [
        face
        for face in shape.faces()
        if face.geom_type in (GeomType.CONE, GeomType.CYLINDER)
    ]
    if len(candidates) != 1:
        raise GeometryValidationError(
            f"expected one lateral cone/cylinder face, found {len(candidates)}",
            [],
            {"candidate_face_count": len(candidates)},
        )
    return candidates[0]


def validate_geometry(
    params: ResolvedGeometry,
    bore_blank: Shape,
    journal: Shape,
    feed_cylinder: Shape,
    base_film: Shape,
    wet: Shape,
    film: Shape,
    zones: dict[str, Solid],
    journal_context: Shape,
    bushing_context: Shape,
    *,
    max_face_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    shapes = {
        "bore_blank": bore_blank,
        "journal_cutter": journal,
        "feed_cylinder": feed_cylinder,
        "base_film": base_film,
        "wet": wet,
        "film": film,
        **zones,
        "journal_context": journal_context,
        "bushing_context": bushing_context,
    }
    topology = {name: _shape_record(shape) for name, shape in shapes.items()}

    film_top = topology["film"]
    records.extend(
        [
            _validation_record("topology.full_film_one_solid", film_top["solid_count"] == 1, film_top["solid_count"], 1, 0),
            _validation_record("topology.full_film_valid", film_top["is_valid"], film_top["is_valid"], True, 0),
            _validation_record("topology.full_film_manifold", film_top["is_manifold"], film_top["is_manifold"], True, 0),
            _validation_record(
                "topology.full_film_face_count",
                0 < film_top["face_count"] <= max_face_count,
                film_top["face_count"],
                f"1..{max_face_count}",
                0,
            ),
            _validation_record("topology.wet_connected", topology["wet"]["solid_count"] == 1, topology["wet"]["solid_count"], 1, 0),
            _validation_record("topology.wet_valid", topology["wet"]["is_valid"], topology["wet"]["is_valid"], True, 0),
            _validation_record("topology.wet_manifold", topology["wet"]["is_manifold"], topology["wet"]["is_manifold"], True, 0),
            _validation_record(
                "topology.all_solids_valid_manifold",
                all(
                    data["solid_count"] >= 1
                    and data["all_solids_valid"]
                    and data["all_solids_manifold"]
                    for data in topology.values()
                ),
                [
                    name
                    for name, data in topology.items()
                    if data["solid_count"] < 1
                    or not data["all_solids_valid"]
                    or not data["all_solids_manifold"]
                ],
                [],
                0,
            ),
        ]
    )
    for name in ("ring_A", "hole_band", "ring_B"):
        data = topology[name]
        records.append(
            _validation_record(
                f"topology.{name}",
                data["solid_count"] == 1 and data["is_valid"] and data["is_manifold"],
                {key: data[key] for key in ("solid_count", "is_valid", "is_manifold")},
                {"solid_count": 1, "is_valid": True, "is_manifold": True},
                0,
            )
        )
    for name in ("journal_context", "bushing_context"):
        data = topology[name]
        records.append(
            _validation_record(
                f"topology.{name}",
                data["solid_count"] == 1 and data["is_valid"] and data["is_manifold"],
                {key: data[key] for key in ("solid_count", "is_valid", "is_manifold")},
                {"solid_count": 1, "is_valid": True, "is_manifold": True},
                0,
            )
        )

    base_rel = _relative_error(base_film.volume, params.base_volume_exact)
    records.append(
        _validation_record(
            "volume.base_film_exact",
            base_rel <= 1e-8,
            {"volume_mm3": base_film.volume, "relative_error": base_rel},
            params.base_volume_exact,
            "relative <= 1e-8",
        )
    )

    added = film.volume - base_film.volume
    try:
        feed_outside_bore = feed_cylinder - bore_blank
    except Exception as exc:
        raise GeometryValidationError(
            f"V_added_boolean = (feed_cylinder - bore_blank).volume failed: {exc}",
            records,
            {"topology": topology},
        ) from exc
    added_boolean = feed_outside_bore.volume
    added_rel = _relative_error(added, added_boolean)
    records.extend(
        [
            _validation_record("volume.feed_added_positive", added > 0.0, added, "> 0", 0),
            _validation_record(
                "topology.feed_connected_to_film",
                film_top["solid_count"] == 1 and added > 0.0,
                {"film_solid_count": film_top["solid_count"], "feed_added_mm3": added},
                {"film_solid_count": 1, "feed_added_mm3": "> 0"},
                0,
            ),
            _validation_record(
                "volume.feed_boolean_identity",
                added_rel <= 5e-5,
                {
                    "film_minus_base_mm3": added,
                    "feed_minus_bore_mm3": added_boolean,
                    "relative_error": added_rel,
                },
                "equal",
                "relative <= 5e-5",
                detail=(
                    "Default OCCT Boolean tolerances perturb the shared bore/feed intersection by a few micrometres cubed; "
                    "no fuzzy tolerance or healing is used."
                ),
            ),
            _validation_record(
                "volume.feed_scale_estimate_diagnostic",
                True,
                {
                    "added_mm3": added,
                    "estimate_mm3": params.feed_scale_estimate,
                    "relative_difference": _relative_error(added, params.feed_scale_estimate),
                },
                "diagnostic only",
                None,
                mandatory=False,
                detail="Cone slope and finite hole radius make this estimate non-binding.",
            ),
        ]
    )

    radial_measurements: list[dict[str, float]] = []
    for station_index in range(1, 6):
        z = params.length * station_index / 6.0
        bore_section = section(bore_blank, section_by=Plane.XY.offset(z))
        journal_section = section(journal, section_by=Plane.XY.offset(z))
        rotated_bore = Rot(Z=-params.eccentricity_angle_deg) * bore_section
        rotated_journal = Rot(Z=-params.eccentricity_angle_deg) * journal_section
        bore_box = rotated_bore.bounding_box()
        journal_box = rotated_journal.bounding_box()
        radial_min = bore_box.max.X - journal_box.max.X
        radial_max = journal_box.min.X - bore_box.min.X
        measurement = {"z_mm": z, "radial_min_mm": radial_min, "radial_max_mm": radial_max}
        radial_measurements.append(measurement)
        records.extend(
            [
                _validation_record(
                    f"clearance.same_z_radial_min_z{station_index}",
                    abs(radial_min - params.h_radial_min) <= 0.0005,
                    radial_min,
                    params.h_radial_min,
                    "absolute <= 0.0005 mm",
                ),
                _validation_record(
                    f"clearance.same_z_radial_max_z{station_index}",
                    abs(radial_max - params.h_radial_max) <= 0.0005,
                    radial_max,
                    params.h_radial_max,
                    "absolute <= 0.0005 mm",
                ),
            ]
        )

    bore_lateral = _lateral_face(bore_blank)
    journal_lateral = _lateral_face(journal)
    normal_distance, closest_bore, closest_journal = (
        bore_lateral.distance_to_with_closest_points(journal_lateral)
    )
    closest_points = {
        "bore_mm": list(tuple(closest_bore)),
        "journal_mm": list(tuple(closest_journal)),
    }
    records.append(
        _validation_record(
            "clearance.shortest_surface_normal_min",
            abs(normal_distance - params.h_normal_min) <= 0.0005,
            {"distance_mm": normal_distance, "closest_points": closest_points},
            params.h_normal_min,
            "absolute <= 0.0005 mm",
            detail="Compared with (c-e)*cos(gamma), not the same-z radial clearance c-e.",
        )
    )

    zone_volumes = {name: zone.volume for name, zone in zones.items()}
    split_rel = abs(sum(zone_volumes.values()) - film.volume) / film.volume
    records.append(
        _validation_record(
            "volume.axial_split_conservation",
            split_rel <= 1e-8,
            {"zone_sum_mm3": sum(zone_volumes.values()), "relative_error": split_rel},
            film.volume,
            "relative <= 1e-8",
        )
    )

    bbox = film.bounding_box()
    rb0 = params.bore_radius(0.0)
    expected_y_max = max(rb0, params.y_feed_end)
    local_min_journal_radius = params.journal_radius(params.z_hole_max)
    records.append(
        _validation_record(
            "bounding_box.feed_radius_below_local_journal_radius",
            params.hole_radius < local_min_journal_radius,
            params.hole_radius,
            f"< {local_min_journal_radius}",
            0,
            detail="This proves the +Y feed does not alter the x-min, x-max, or y-min radial envelope.",
        )
    )
    bbox_checks = {
        "z_min": (bbox.min.Z, 0.0),
        "z_max": (bbox.max.Z, params.length),
        "x_min": (bbox.min.X, -rb0),
        "x_max": (bbox.max.X, rb0),
        "y_min": (bbox.min.Y, -rb0),
        "y_max": (bbox.max.Y, expected_y_max),
    }
    for name, (measured, expected) in bbox_checks.items():
        records.append(
            _validation_record(
                f"bounding_box.{name}",
                abs(measured - expected) <= 0.001,
                measured,
                expected,
                "absolute <= 0.001 mm",
            )
        )

    inlet_faces = [face for face in film.faces() if _inlet_face_matches(face, params)]
    records.append(
        _validation_record("inlet.identity_unique", len(inlet_faces) == 1, len(inlet_faces), 1, 0)
    )
    inlet: dict[str, Any] = {"matching_face_count": len(inlet_faces)}
    if len(inlet_faces) == 1:
        inlet_face = inlet_faces[0]
        inlet_edges = list(inlet_face.edges())
        inlet = {
            "area_mm2": inlet_face.area,
            "centre_mm": list(tuple(inlet_face.center())),
            "normal": list(tuple(inlet_face.normal_at())),
            "edge_count": len(inlet_edges),
            "edge_geometry": [edge.geom_type.name for edge in inlet_edges],
        }
        expected_area = math.pi * params.hole_radius**2
        records.extend(
            [
                _validation_record(
                    "inlet.area",
                    abs(inlet_face.area - expected_area) <= max(1e-7, expected_area * 1e-8),
                    inlet_face.area,
                    expected_area,
                    "max(1e-7 mm2, relative 1e-8)",
                ),
                _validation_record(
                    "inlet.centre",
                    math.dist(tuple(inlet_face.center()), (0.0, params.y_feed_end, params.hole_axial_pos)) <= 1e-6,
                    inlet["centre_mm"],
                    [0.0, params.y_feed_end, params.hole_axial_pos],
                    "distance <= 1e-6 mm",
                ),
                _validation_record(
                    "inlet.outward_normal_plus_y",
                    tuple(inlet_face.normal_at())[1] >= 1.0 - 1e-10,
                    inlet["normal"],
                    [0.0, 1.0, 0.0],
                    "dot >= 1-1e-10",
                ),
                _validation_record(
                    "inlet.circular_boundary",
                    len(inlet_edges) == 1 and inlet_edges[0].geom_type == GeomType.CIRCLE,
                    inlet["edge_geometry"],
                    ["CIRCLE"],
                    0,
                ),
            ]
        )
    else:
        for name in ("area", "centre", "outward_normal_plus_y", "circular_boundary"):
            records.append(
                _validation_record(f"inlet.{name}", False, "not measurable", "one inlet face", 0)
            )

    diagnostics = {
        "topology": topology,
        "volumes_mm3": {
            "base_exact": params.base_volume_exact,
            "base_boolean": base_film.volume,
            "feed_added": added,
            "feed_added_boolean": added_boolean,
            "feed_scale_estimate": params.feed_scale_estimate,
            "total": film.volume,
            "zones": zone_volumes,
            "split_relative_error": split_rel,
        },
        "same_z_radial_clearance_mm": {
            "target_min": params.h_radial_min,
            "target_max": params.h_radial_max,
            "stations": radial_measurements,
        },
        "shortest_surface_normal_clearance_mm": {
            "target_min": params.h_normal_min,
            "target_max": params.h_normal_max,
            "occt_minimum": normal_distance,
            "closest_points": closest_points,
        },
        "inlet_face": inlet,
        "final_bounding_box_mm": _bounding_box_record(film),
        "feed_start_disk_proof_margin_mm": params.feed_start_disk_margin,
    }
    failures = [record for record in records if record["mandatory"] and record["status"] == "FAIL"]
    if failures:
        raise GeometryValidationError(
            "mandatory geometry validation failed: " + ", ".join(record["name"] for record in failures),
            records,
            diagnostics,
        )
    return records, diagnostics


def _dependency_versions() -> dict[str, str | None]:
    def installed(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "build123d": build123d.__version__,
        "cadquery-ocp-novtk": installed("cadquery-ocp-novtk"),
        "matplotlib": installed("matplotlib"),
        "ocp-vscode": installed("ocp-vscode"),
    }


def _export_step_file(shape: Shape, path: Path) -> None:
    try:
        success = export_step(
            shape,
            path,
            unit=Unit.MM,
            write_pcurves=True,
            precision_mode=PrecisionMode.GREATEST,
            timestamp="2000-01-01T00:00:00",
        )
    except Exception as exc:
        raise GeometryExportError(f"STEP export failed for {path.name}: {exc}") from exc
    if not success or not path.is_file() or path.stat().st_size == 0:
        raise GeometryExportError(f"STEP export did not create a non-empty {path.name}")


def _export_brep_file(shape: Shape, path: Path) -> None:
    try:
        success = export_brep(shape, path)
    except Exception as exc:
        raise GeometryExportError(f"BREP export failed for {path.name}: {exc}") from exc
    if not success or not path.is_file() or path.stat().st_size == 0:
        raise GeometryExportError(f"BREP export did not create a non-empty {path.name}")


def preview_geometry(
    film: Shape,
    journal_context: Shape,
    bushing_context: Shape,
) -> None:
    try:
        from ocp_vscode import show

        show(
            film,
            journal_context,
            bushing_context,
            names=["fluid", "journal_context", "bushing_context"],
            colors=["deepskyblue", "silver", "goldenrod"],
            alphas=[1.0, 0.2, 0.2],
            port=PREVIEW_PORT,
            axes=True,
        )
    except Exception as exc:
        raise GeometryExportError(
            f"live preview failed on 127.0.0.1:{PREVIEW_PORT}: {exc}"
        ) from exc


def _plot_thickness_map(params: ResolvedGeometry, path: Path) -> tuple[bool, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        return False, f"film_thickness_map.png skipped: optional plotting dependency unavailable ({exc})"

    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    z = np.linspace(0.0, params.length, 161)
    theta_grid, z_grid = np.meshgrid(theta, z)
    alpha = theta_grid - params.phi_rad
    rj = params.mean_radius + (params.length / 2.0 - z_grid) * params.cone_slope
    journal_intersection = (
        params.eccentricity * np.cos(alpha)
        + np.sqrt(rj**2 - params.eccentricity**2 * np.sin(alpha) ** 2)
    )
    exact = rj + params.radial_clearance - journal_intersection
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.pcolormesh(np.degrees(theta), z, exact, shading="auto", cmap="viridis")
    eccentricity_deg = params.eccentricity_angle_deg % 360.0
    ax.axvline(eccentricity_deg, color="white", linestyle="--", linewidth=1.2, label="eccentricity / radial minimum")
    ax.axvline(90.0, color="red", linestyle=":", linewidth=1.4, label="+Y feed direction")
    ax.set(xlabel="bearing-frame angle theta (deg)", ylabel="axial coordinate z (mm)", xlim=(0.0, 360.0))
    ax.set_title("Exact same-z radial film thickness — geometry diagnostic, not CFD pressure")
    fig.colorbar(image, ax=ax, label="same-z radial thickness (mm)")
    ax.legend(loc="upper right")
    try:
        fig.savefig(path, dpi=180)
    except Exception as exc:
        plt.close(fig)
        raise GeometryExportError(f"plot export failed: {exc}") from exc
    plt.close(fig)
    return True, "film_thickness_map.png generated from the exact same-z radial expression"


def export_all(
    params: ResolvedGeometry,
    film: Shape,
    zones: dict[str, Solid],
    journal_context: Shape,
    bushing_context: Shape,
    outdir: Path,
    *,
    export_debug_half: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    target = outdir
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    film.label = "film_unsplit"
    film_path = target / "film_unsplit.step"
    _export_step_file(film, film_path)
    manifest[film_path.name] = {"solid_count": 1, "volume_mm3": film.volume, "labels": ["film_unsplit"]}
    film_brep_path = target / "film_unsplit.brep"
    _export_brep_file(film, film_brep_path)
    manifest[film_brep_path.name] = {
        "solid_count": 1,
        "volume_mm3": film.volume,
        "labels": [],
    }

    for name, zone in zones.items():
        zone.label = name
        path = target / f"{name}.step"
        _export_step_file(zone, path)
        manifest[path.name] = {"solid_count": 1, "volume_mm3": zone.volume, "labels": [name]}

    zones_assembly = Compound(label="film_zones", children=list(zones.values()))
    zones_path = target / "film_zones.step"
    _export_step_file(zones_assembly, zones_path)
    manifest[zones_path.name] = {
        "solid_count": 3,
        "volume_mm3": sum(zone.volume for zone in zones.values()),
        "labels": list(zones),
    }
    zones_brep_path = target / "film_zones.brep"
    _export_brep_file(zones_assembly, zones_brep_path)
    manifest[zones_brep_path.name] = {
        "solid_count": 3,
        "volume_mm3": sum(zone.volume for zone in zones.values()),
        "labels": [],
    }

    journal_context.label = "journal_context"
    bushing_context.label = "bushing_context"
    context = Compound(
        label="bearing_context",
        children=[journal_context, bushing_context],
    )
    context_path = target / "context_assembly.step"
    _export_step_file(context, context_path)
    manifest[context_path.name] = {
        "solid_count": 2,
        "volume_mm3": journal_context.volume + bushing_context.volume,
        "labels": ["journal_context", "bushing_context"],
    }

    if export_debug_half:
        print("GEOMETRY DEBUG ONLY — NOT VALID FOR ROTATING-JOURNAL CFD")
        halves: list[Solid] = []
        for name, zone in zones.items():
            half_parts = list(split(zone, Plane.YZ, keep=Keep.TOP).solids())
            if len(half_parts) != 1:
                raise GeometryExportError(f"debug half of {name} produced {len(half_parts)} solids")
            half = half_parts[0]
            half.label = f"{name}_half_DEBUG_ONLY"
            halves.append(half)
        half_assembly = Compound(label="GEOMETRY_DEBUG_ONLY_NOT_CFD", children=halves)
        half_path = target / "geometry_half_debug.step"
        _export_step_file(half_assembly, half_path)
        manifest[half_path.name] = {
            "solid_count": 3,
            "volume_mm3": sum(half.volume for half in halves),
            "labels": [half.label for half in halves],
        }

    plot_created, plot_message = _plot_thickness_map(params, target / "film_thickness_map.png")
    warnings.append(plot_message)
    versions = _dependency_versions()
    requirement_lines = [f"build123d=={versions['build123d']}"]
    if plot_created and versions["matplotlib"]:
        requirement_lines.append(f"matplotlib=={versions['matplotlib']}")
    if versions["ocp-vscode"]:
        requirement_lines.append(f"ocp-vscode=={versions['ocp-vscode']}")
    (target / "requirements.txt").write_text("\n".join(requirement_lines) + "\n", encoding="utf-8")
    return manifest, warnings


def _all_labels(shape: Shape) -> list[str]:
    nodes = [shape, *shape.descendants]
    return list(dict.fromkeys(node.label for node in nodes if node.label))


def _round_trip_all(
    outdir: Path,
    manifest: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    volume_failures: list[str] = []
    for filename, expected in manifest.items():
        path = outdir / filename
        if path.suffix.lower() == ".step":
            format_name = "STEP"
            importer = import_step
            volume_tolerance = STEP_ROUNDTRIP_REL_TOL
        elif path.suffix.lower() == ".brep":
            format_name = "BREP"
            importer = import_brep
            volume_tolerance = BREP_ROUNDTRIP_REL_TOL
        else:
            raise RoundTripValidationError(
                f"unsupported round-trip format: {filename}",
                records,
                diagnostics,
            )
        record_prefix = f"{format_name.lower()}_round_trip"
        try:
            imported = importer(path)
        except Exception as exc:
            records.append(
                _validation_record(
                    f"{record_prefix}.{filename}.import",
                    False,
                    str(exc),
                    "successful import",
                    0,
                )
            )
            continue
        solids = list(imported.solids())
        total_volume = sum(solid.volume for solid in solids)
        absolute_volume_error = abs(total_volume - expected["volume_mm3"])
        relative_volume_error = _relative_error(total_volume, expected["volume_mm3"])
        valid = all(solid.is_valid for solid in solids)
        manifold = all(solid.is_manifold for solid in solids)
        labels = _all_labels(imported) if format_name == "STEP" else []
        missing_labels = (
            [label for label in expected["labels"] if label not in labels]
            if format_name == "STEP"
            else []
        )
        if format_name == "STEP" and missing_labels:
            warnings.append(
                f"{filename}: imported STEP omitted labels {missing_labels}; label-independent fallback files are present"
            )
        volume_passed = relative_volume_error <= volume_tolerance
        if not volume_passed:
            volume_failures.append(
                f"{filename}: {format_name} round-trip error {relative_volume_error:.3e} "
                f"exceeds {volume_tolerance:.3e}"
            )
        diagnostics[filename] = {
            "format": format_name,
            "step_precision_mode": PrecisionMode.GREATEST.name if format_name == "STEP" else None,
            "solid_count": len(solids),
            "expected_solid_count": expected["solid_count"],
            "all_valid": valid,
            "all_manifold": manifold,
            "total_volume_mm3": total_volume,
            "expected_volume_mm3": expected["volume_mm3"],
            "absolute_volume_error_mm3": absolute_volume_error,
            "relative_volume_error": relative_volume_error,
            "labels": labels,
            "missing_expected_labels_warning": missing_labels,
        }
        records.extend(
            [
                _validation_record(
                    f"{record_prefix}.{filename}.solid_count",
                    len(solids) == expected["solid_count"],
                    len(solids),
                    expected["solid_count"],
                    0,
                ),
                _validation_record(f"{record_prefix}.{filename}.valid", valid, valid, True, 0),
                _validation_record(f"{record_prefix}.{filename}.manifold", manifold, manifold, True, 0),
                _validation_record(
                    f"{record_prefix}.{filename}.volume",
                    volume_passed,
                    {
                        "volume_mm3": total_volume,
                        "absolute_error_mm3": absolute_volume_error,
                        "relative_error": relative_volume_error,
                    },
                    expected["volume_mm3"],
                    f"relative <= {volume_tolerance:.1e}",
                    detail=(
                        "STEP is exported with PrecisionMode.GREATEST. Native BREP is checked separately "
                        "so an interchange translation cannot be mistaken for an in-memory geometry error."
                    ),
                ),
            ]
        )
        if format_name == "STEP":
            records.append(
                _validation_record(
                    f"{record_prefix}.{filename}.labels",
                    True,
                    labels,
                    expected["labels"],
                    "warning only",
                    mandatory=False,
                    detail="Missing labels are non-fatal because individual fallback STEP files are mandatory.",
                )
            )
    failures = [record for record in records if record["mandatory"] and record["status"] == "FAIL"]
    if failures:
        raise RoundTripValidationError(
            "mandatory CAD round-trip validation failed: "
            + "; ".join(volume_failures or [record["name"] for record in failures]),
            records,
            diagnostics,
        )
    return records, diagnostics, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if hasattr(value, "__version__"):
        return str(value.__version__)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rejected_step_target(outdir: Path) -> Path:
    return outdir.with_name(f"{outdir.name}.rejected-step")


def _failed_generation_target(outdir: Path) -> Path:
    return outdir.with_name(f"{outdir.name}.failed")


def _publish_failure_generation(
    inputs: GeometryInputs,
    options: RunOptions,
    params: ResolvedGeometry | None,
    error: BaseException,
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    shapes: dict[str, Shape],
    *,
    argv: Sequence[str],
) -> None:
    target = _failed_generation_target(options.outdir)
    stage = make_staging_directory(target)
    try:
        _write_json(
            stage / "params.json",
            _failure_payload(
                inputs,
                options,
                params,
                error,
                records,
                diagnostics,
                shapes,
            ),
        )
        publish_generation(
            stage,
            target,
            stage="geometry",
            operation="cad",
            status="FAIL",
            argv=argv,
            resolved_inputs=(
                _input_payload(inputs, options)
                if params is None
                else _resolved_payload(params, options)
            ),
            input_units={
                "lengths": "mm",
                "angles": "deg",
                "volumes": "mm^3",
            },
            producer_files=(Path(__file__), Path(__file__).with_name("model.py")),
            tool_versions=_dependency_versions(),
            acceptance_status=False,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _is_owned_rejected_step_batch(target: Path) -> bool:
    manifest = target / "REJECTED.json"
    if target.is_symlink() or not target.is_dir() or not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema") == REJECTED_STEP_SCHEMA
        and payload.get("producer") == PRODUCER
        and payload.get("status") == "REJECTED_DIAGNOSTIC_ONLY"
        and payload.get("solve_eligible") is False
    )


def _discard_rejected_step_batch(outdir: Path) -> None:
    target = _rejected_step_target(outdir)
    if not target.exists():
        return
    if not _is_owned_rejected_step_batch(target):
        raise GeometryExportError(f"refusing to remove unrecognized STEP quarantine: {target}")
    shutil.rmtree(target)


def _publish_rejected_step_batch(
    stage: Path,
    outdir: Path,
    manifest: dict[str, dict[str, Any]],
    error: RoundTripValidationError,
) -> dict[str, Any]:
    """Atomically retain a failed STEP batch outside the trusted CAD directory."""
    target = _rejected_step_target(outdir)
    if target.exists() and not _is_owned_rejected_step_batch(target):
        raise GeometryExportError(f"refusing to replace unrecognized STEP quarantine: {target}")

    quarantine_stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent)
    )
    try:
        files: dict[str, Any] = {}
        for filename, expected in manifest.items():
            source = stage / filename
            if not source.is_file():
                raise GeometryExportError(f"rejected STEP artifact is missing: {source}")
            destination = quarantine_stage / filename
            shutil.copy2(source, destination)
            files[filename] = {
                "sha256": _sha256(destination),
                "expected": expected,
                "round_trip": error.diagnostics.get(filename),
            }

        rejected = {
            "schema": REJECTED_STEP_SCHEMA,
            "producer": PRODUCER,
            "status": "REJECTED_DIAGNOSTIC_ONLY",
            "solve_eligible": False,
            "warning": "DO NOT USE FOR MESHING OR SOLVER INPUT",
            "source_output_directory": str(outdir),
            "strict_relative_volume_tolerance": STEP_ROUNDTRIP_REL_TOL,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "files": files,
        }
        _write_json(quarantine_stage / "REJECTED.json", rejected)

        publish_generation(
            quarantine_stage,
            target,
            stage="geometry",
            operation="rejected-step",
            status="FAIL",
            resolved_inputs={"source_output_directory": outdir},
            input_units={"geometry": "mm"},
            producer_files=(Path(__file__),),
            acceptance_status="REJECTED_DIAGNOSTIC_ONLY",
        )
        return {
            "schema": REJECTED_STEP_SCHEMA,
            "status": rejected["status"],
            "solve_eligible": False,
            "path": str(target),
            "manifest": "REJECTED.json",
            "files": sorted(files),
        }
    finally:
        if quarantine_stage.exists():
            shutil.rmtree(quarantine_stage)


def _table(title: str, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    print(f"\n{title}")
    text_rows = [[str(item) for item in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(item.ljust(widths[index]) for index, item in enumerate(row)))


def _input_payload(
    inputs: GeometryInputs,
    options: RunOptions,
) -> dict[str, Any]:
    return asdict(inputs) | asdict(options)


def _resolved_payload(
    params: ResolvedGeometry,
    options: RunOptions,
) -> dict[str, Any]:
    return asdict(params) | {
        "max_face_count": options.max_face_count,
        "export_debug_half": options.export_debug_half,
        "preview": options.preview,
        "outdir": options.outdir,
    }


def _console_report(
    inputs: GeometryInputs,
    options: RunOptions,
    params: ResolvedGeometry,
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    step_round_trip: dict[str, Any],
    brep_round_trip: dict[str, Any],
    messages: list[str],
) -> None:
    input_units = {
        "length": "mm",
        "mean_radius": "mm",
        "semicone_angle_deg": "deg",
        "radial_clearance": "mm radial",
        "eccentricity_ratio": "1",
        "eccentricity_angle_deg": "deg",
        "hole_diameter": "mm",
        "hole_axial_pos": "mm",
        "split_halfwidth": "mm",
        "bushing_wall_thickness": "mm",
        "inlet_extension": "mm",
        "axial_cutter_extension": "mm",
        "max_face_count": "1",
        "export_debug_half": "bool",
        "retain_failed_step": "bool",
        "preview": "bool",
        "outdir": "path",
    }
    _table(
        "Inputs",
        ("name", "value", "unit"),
        [
            (name, "auto (L/2)" if name == "hole_axial_pos" and value is None else value, input_units[name])
            for name, value in _input_payload(inputs, options).items()
        ],
    )
    _table(
        "Resolved placement",
        ("quantity", "value", "unit"),
        [
            ("zh", f"{params.hole_axial_pos:.9f}", "mm"),
            ("e", f"{params.eccentricity:.9f}", "mm"),
            ("ex", f"{params.ex:.9f}", "mm"),
            ("ey", f"{params.ey:.9f}", "mm"),
            ("eccentricity/min-clearance direction", f"{params.eccentricity_angle_deg:.9f}", "deg"),
            ("feed direction", "+Y (90 deg)", "bearing frame"),
        ],
    )
    _table(
        "Radii",
        ("z (mm)", "Rj (mm)", "Rb (mm)", "Ro (mm)"),
        [
            (
                f"{z:.6f}",
                f"{params.journal_radius(z):.9f}",
                f"{params.bore_radius(z):.9f}",
                f"{params.outer_radius(z):.9f}",
            )
            for z in (0.0, params.length / 2.0, params.length)
        ],
    )
    radial = diagnostics["same_z_radial_clearance_mm"]
    _table(
        "Same-z radial clearance (not surface-normal distance)",
        ("z (mm)", "measured min", "target min", "measured max", "target max"),
        [
            (
                f"{item['z_mm']:.6f}",
                f"{item['radial_min_mm']:.9f}",
                f"{radial['target_min']:.9f}",
                f"{item['radial_max_mm']:.9f}",
                f"{radial['target_max']:.9f}",
            )
            for item in radial["stations"]
        ],
    )
    normal = diagnostics["shortest_surface_normal_clearance_mm"]
    _table(
        "Shortest surface-normal clearance (not same-z radial clearance)",
        ("quantity", "value", "unit"),
        [
            ("target min = (c-e) cos(gamma)", f"{normal['target_min']:.9f}", "mm"),
            ("target max = (c+e) cos(gamma)", f"{normal['target_max']:.9f}", "mm"),
            ("OCCT lateral-face minimum", f"{normal['occt_minimum']:.9f}", "mm"),
            ("closest bore point", normal["closest_points"]["bore_mm"], "mm"),
            ("closest journal point", normal["closest_points"]["journal_mm"], "mm"),
        ],
    )
    volumes = diagnostics["volumes_mm3"]
    _table(
        "Fluid volumes",
        ("body", "volume (mm^3)"),
        [
            ("base exact", f"{volumes['base_exact']:.9f}"),
            ("base Boolean", f"{volumes['base_boolean']:.9f}"),
            ("feed added", f"{volumes['feed_added']:.9f}"),
            ("feed added Boolean identity", f"{volumes['feed_added_boolean']:.9f}"),
            ("feed scale estimate (diagnostic)", f"{volumes['feed_scale_estimate']:.9f}"),
            ("full film", f"{volumes['total']:.9f}"),
            *((name, f"{volume:.9f}") for name, volume in volumes["zones"].items()),
        ],
    )
    _table(
        "Topology",
        ("shape", "solids", "faces", "valid", "manifold"),
        [
            (
                name,
                data["solid_count"],
                data["face_count"],
                data["is_valid"],
                data["is_manifold"],
            )
            for name, data in diagnostics["topology"].items()
        ],
    )
    inlet = diagnostics["inlet_face"]
    _table(
        "Selectable remote inlet face",
        ("quantity", "value"),
        [
            ("area (mm^2)", inlet.get("area_mm2")),
            ("centre (mm)", inlet.get("centre_mm")),
            ("outward normal", inlet.get("normal")),
            ("boundary", inlet.get("edge_geometry")),
        ],
    )
    _table(
        f"STEP inspection round-trip (relative tolerance {STEP_ROUNDTRIP_REL_TOL:.1e})",
        ("file", "solids", "valid", "manifold", "volume rel. error", "labels"),
        [
            (
                filename,
                data["solid_count"],
                data["all_valid"],
                data["all_manifold"],
                f"{data['relative_volume_error']:.3e}",
                data["labels"],
            )
            for filename, data in step_round_trip.items()
        ],
    )
    _table(
        f"Native BREP round-trip (mandatory relative tolerance {BREP_ROUNDTRIP_REL_TOL:.1e})",
        ("file", "solids", "valid", "manifold", "volume rel. error"),
        [
            (
                filename,
                data["solid_count"],
                data["all_valid"],
                data["all_manifold"],
                f"{data['relative_volume_error']:.3e}",
            )
            for filename, data in brep_round_trip.items()
        ],
    )
    passed = sum(record["status"] == "PASS" for record in records)
    print(f"\nValidation records: {passed}/{len(records)} PASS")
    for message in messages:
        print(f"NOTE: {message}")
    print("NOTE: -Y is the default eccentricity/minimum-clearance line, not automatically the external load line.")
    print("NOTE: Hydrodynamic load direction is deferred to later CFD; this program creates CAD only.")
    print("OVERALL: PASS")


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[GeometryInputs, RunOptions]:
    geometry_defaults = GeometryInputs()
    option_defaults = RunOptions()
    parser = argparse.ArgumentParser(
        description="Generate and validate the exact eccentric conical-bearing lubricant volume."
    )
    parser.add_argument("--length", type=float, default=geometry_defaults.length)
    parser.add_argument("--mean-radius", type=float, default=geometry_defaults.mean_radius)
    parser.add_argument("--semicone-angle-deg", type=float, default=geometry_defaults.semicone_angle_deg)
    parser.add_argument("--radial-clearance", type=float, default=geometry_defaults.radial_clearance)
    parser.add_argument("--eccentricity-ratio", type=float, default=geometry_defaults.eccentricity_ratio)
    parser.add_argument("--eccentricity-angle-deg", type=float, default=geometry_defaults.eccentricity_angle_deg)
    parser.add_argument("--hole-diameter", type=float, default=geometry_defaults.hole_diameter)
    parser.add_argument("--hole-axial-pos", type=float, default=geometry_defaults.hole_axial_pos)
    parser.add_argument("--split-halfwidth", type=float, default=geometry_defaults.split_halfwidth)
    parser.add_argument("--bushing-wall-thickness", type=float, default=geometry_defaults.bushing_wall_thickness)
    parser.add_argument("--inlet-extension", type=float, default=geometry_defaults.inlet_extension)
    parser.add_argument("--axial-cutter-extension", type=float, default=geometry_defaults.axial_cutter_extension)
    parser.add_argument("--max-face-count", type=int, default=option_defaults.max_face_count)
    parser.add_argument(
        "--export-debug-half",
        action=argparse.BooleanOptionalAction,
        default=option_defaults.export_debug_half,
    )
    parser.add_argument(
        "--retain-failed-step",
        action=argparse.BooleanOptionalAction,
        default=option_defaults.retain_failed_step,
        help="retain a rejected STEP batch in a sibling diagnostic-only directory",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=option_defaults.preview,
        help=f"send fluid and context solids to an ocp_vscode viewer on port {PREVIEW_PORT}",
    )
    parser.add_argument("--outdir", type=Path, default=option_defaults.outdir)
    parsed = vars(parser.parse_args(argv))
    inputs = GeometryInputs(
        **{field.name: parsed[field.name] for field in fields(GeometryInputs)}
    )
    options = RunOptions(
        **{field.name: parsed[field.name] for field in fields(RunOptions)}
    )
    return inputs, options


def _failure_payload(
    inputs: GeometryInputs | None,
    options: RunOptions | None,
    params: ResolvedGeometry | None,
    error: BaseException,
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    shapes: dict[str, Shape],
) -> dict[str, Any]:
    trusted_hashes: dict[str, str] = {}
    if params is not None and options is not None:
        for filename in (
            "film_unsplit.brep",
            "film_zones.brep",
            "film_thickness_map.png",
            "requirements.txt",
        ):
            path = options.outdir / filename
            if path.is_file():
                trusted_hashes[filename] = _sha256(path)
    return {
        "overall": "FAIL",
        "inputs": (
            _input_payload(inputs, options)
            if inputs is not None and options is not None
            else None
        ),
        "resolved_parameters": (
            _resolved_payload(params, options)
            if params is not None and options is not None
            else None
        ),
        "dependency_versions": _dependency_versions(),
        "error": {"type": type(error).__name__, "message": str(error)},
        "validation_records": records,
        "measurements": diagnostics,
        "sha256": trusted_hashes,
        "available_shape_diagnostics": {
            name: _shape_record(shape) for name, shape in shapes.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    inputs: GeometryInputs | None = None
    options: RunOptions | None = None
    params: ResolvedGeometry | None = None
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    shapes: dict[str, Shape] = {}
    step_round_trip: dict[str, Any] = {}
    step_validation_records: list[dict[str, Any]] = []
    step_exchange_status = "PASS"
    brep_round_trip: dict[str, Any] = {}
    try:
        inputs, options = _parse_args(argv)
        params = resolve_params(inputs)
        validate_run_options(options)

        bore_blank = make_bore_blank(params)
        journal = make_journal(params, extended=True)
        feed_cylinder = make_feed_cylinder(params)
        shapes.update(bore_blank=bore_blank, journal_cutter=journal, feed_cylinder=feed_cylinder)
        base_film = make_base_film(bore_blank, journal)
        wet, film = make_full_film(bore_blank, feed_cylinder, journal)
        shapes.update(base_film=base_film, wet=wet, film=film)
        zones = split_axial_zones(params, film)
        shapes.update(zones)
        journal_context = make_journal(params, extended=False)
        bushing_context = make_context_bushing(params, bore_blank, feed_cylinder)
        shapes.update(journal_context=journal_context, bushing_context=bushing_context)

        records, diagnostics = validate_geometry(
            params,
            bore_blank,
            journal,
            feed_cylinder,
            base_film,
            wet,
            film,
            zones,
            journal_context,
            bushing_context,
            max_face_count=options.max_face_count,
        )
        if options.preview:
            preview_geometry(film, journal_context, bushing_context)

        options.outdir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{options.outdir.name}.stage-",
            dir=options.outdir.parent,
        ) as stage_name:
            stage = Path(stage_name)
            manifest, messages = export_all(
                params,
                film,
                zones,
                journal_context,
                bushing_context,
                stage,
                export_debug_half=options.export_debug_half,
            )
            brep_manifest = {
                filename: expected
                for filename, expected in manifest.items()
                if Path(filename).suffix.lower() == ".brep"
            }
            brep_records, brep_round_trip, brep_messages = _round_trip_all(
                stage,
                brep_manifest,
            )
            records.extend(brep_records)
            messages.extend(brep_messages)
            step_manifest = {
                filename: expected
                for filename, expected in manifest.items()
                if Path(filename).suffix.lower() == ".step"
            }
            try:
                step_validation_records, step_round_trip, step_messages = _round_trip_all(
                    stage,
                    step_manifest,
                )
            except RoundTripValidationError as step_error:
                step_exchange_status = "REJECTED"
                step_validation_records = step_error.records
                step_round_trip = step_error.diagnostics
                diagnostics["step_exchange"] = {
                    "status": step_exchange_status,
                    "strict_relative_volume_tolerance": STEP_ROUNDTRIP_REL_TOL,
                    "error": {
                        "type": type(step_error).__name__,
                        "message": str(step_error),
                    },
                }
                try:
                    if options.retain_failed_step:
                        diagnostics["rejected_step_quarantine"] = (
                            _publish_rejected_step_batch(
                                stage,
                                options.outdir,
                                step_manifest,
                                step_error,
                            )
                        )
                    else:
                        _discard_rejected_step_batch(options.outdir)
                except Exception as quarantine_error:
                    diagnostics["rejected_step_quarantine"] = {
                        "status": "PUBLICATION_FAILED",
                        "error": str(quarantine_error),
                    }
                    print(
                        f"ERROR [STEP quarantine publication]: {quarantine_error}",
                        file=sys.stderr,
                    )
                for filename in step_manifest:
                    (stage / filename).unlink(missing_ok=True)
                messages.append(
                    "STEP exchange was rejected at its strict round-trip tolerance; "
                    "the validated native BREP remains the geometry output."
                )
            else:
                try:
                    _discard_rejected_step_batch(options.outdir)
                except GeometryExportError as quarantine_error:
                    messages.append(str(quarantine_error))
                records.extend(step_validation_records)
                messages.extend(step_messages)
            hashes = {
                path.name: _sha256(path)
                for path in sorted(stage.iterdir())
                if path.is_file()
            }
            payload = {
                "overall": "PASS",
                "inputs": _input_payload(inputs, options),
                "resolved_parameters": _resolved_payload(params, options),
                "dependency_versions": _dependency_versions(),
                "validation_records": records,
                "measurements": diagnostics,
                "step_exchange": {
                    "status": step_exchange_status,
                    "strict_relative_volume_tolerance": STEP_ROUNDTRIP_REL_TOL,
                    "validation_records": step_validation_records,
                },
                "step_round_trip": step_round_trip,
                "brep_round_trip": brep_round_trip,
                "warnings": messages,
                "sha256": hashes,
            }
            _write_json(stage / "params.json", payload)
            publish_generation(
                stage,
                options.outdir,
                stage="geometry",
                operation="cad",
                status="PASS",
                argv=() if argv is None else argv,
                resolved_inputs=_resolved_payload(params, options),
                input_units={
                    "lengths": "mm",
                    "angles": "deg",
                    "volumes": "mm^3",
                },
                producer_files=(Path(__file__), Path(__file__).with_name("model.py")),
                tool_versions=_dependency_versions(),
            )

        _console_report(
            inputs,
            options,
            params,
            records,
            diagnostics,
            step_round_trip,
            brep_round_trip,
            messages,
        )
        return 0
    except GeometryValidationError as exc:
        records = exc.records
        diagnostics = exc.diagnostics
        error: BaseException = exc
    except RoundTripValidationError as exc:
        records.extend(exc.records)
        failed_step = {
            filename: data
            for filename, data in exc.diagnostics.items()
            if Path(filename).suffix.lower() == ".step"
        }
        failed_brep = {
            filename: data
            for filename, data in exc.diagnostics.items()
            if Path(filename).suffix.lower() == ".brep"
        }
        step_round_trip.update(failed_step)
        brep_round_trip.update(failed_brep)
        diagnostics = {
            **diagnostics,
            "step_round_trip": step_round_trip,
            "brep_round_trip": brep_round_trip,
        }
        error = exc
    except BearingFilmError as exc:
        error = exc
    except Exception as exc:
        error = GeometryConstructionError(f"unexpected {type(exc).__name__}: {exc}")

    quarantine = diagnostics.get("rejected_step_quarantine", {})
    if options is not None and quarantine.get("status") != "REJECTED_DIAGNOSTIC_ONLY":
        try:
            _discard_rejected_step_batch(options.outdir)
        except Exception as quarantine_error:
            diagnostics["rejected_step_quarantine_cleanup"] = {
                "status": "CLEANUP_FAILED",
                "error": str(quarantine_error),
            }
            print(
                f"ERROR [STEP quarantine cleanup]: {quarantine_error}",
                file=sys.stderr,
            )

    print(f"ERROR [{type(error).__name__}]: {error}", file=sys.stderr)
    if shapes:
        print("Resulting shape diagnostics:", file=sys.stderr)
        print(
            json.dumps(
                {name: _shape_record(shape) for name, shape in shapes.items()},
                indent=2,
                default=_json_default,
            ),
            file=sys.stderr,
        )
    if inputs is not None and options is not None:
        try:
            _publish_failure_generation(
                inputs,
                options,
                params,
                error,
                records,
                diagnostics,
                shapes,
                argv=() if argv is None else argv,
            )
        except Exception as json_error:
            print(f"ERROR [failure-report]: {json_error}", file=sys.stderr)
    print("OVERALL: FAIL", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
