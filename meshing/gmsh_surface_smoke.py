#!/usr/bin/env python3
"""Generate and validate a first-order 2D CFD-boundary smoke mesh."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import gmsh

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshing.gmsh_brep_preflight import (
    SI_SCALE,
    PreflightRunError,
    atomic_replace_directory,
    configure_occ_options,
    grouped_surfaces,
    import_brep,
    make_staging_directory,
    parallel,
    reference_from_dict,
    relative_error,
    require,
    sha256_file,
    validate_zone_model,
    write_validation_log,
)


INLET_AREA_REL_TOL = 5.0e-3
MESH_BBOX_REL_TOL = 1.0e-10


class SurfaceSmokeRunError(PreflightRunError):
    """A failed surface-smoke run with its diagnostic report."""


@dataclass(frozen=True)
class SurfaceSmokeInputs:
    brep: Path = Path("out/gmsh_preflight/film_zones_fragmented.brep")
    preflight_report: Path = Path("out/gmsh_preflight/preflight_report.json")
    outdir: Path = Path("out/gmsh_surface_smoke")
    global_size_mm: float = 0.75
    port_size_mm: float = 0.20
    port_refine_radius_mm: float = 6.0
    max_elements: int = 1_000_000
    gui: bool = False


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def triangle_area(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def load_preflight_contract(
    inputs: SurfaceSmokeInputs, records: list[dict[str, Any]]
) -> tuple[dict[str, Any], Any]:
    for label, path in (("brep", inputs.brep), ("preflight_report", inputs.preflight_report)):
        require(records, f"input.{label}.exists", path.is_file(), str(path), "readable file")
    report = json.loads(inputs.preflight_report.read_text(encoding="utf-8"))
    require(records, "input.preflight_overall", report.get("overall") == "PASS", report.get("overall"), "PASS")

    output = report.get("outputs", {}).get("film_zones_fragmented.brep", {})
    actual_hash = sha256_file(inputs.brep)
    require(
        records,
        "input.brep_sha256",
        output.get("sha256") == actual_hash,
        actual_hash,
        output.get("sha256"),
    )
    require(
        records,
        "input.brep_coordinate_contract",
        output.get("coordinate_unit") == "mm" and output.get("scale_to_m") == SI_SCALE,
        {
            "coordinate_unit": output.get("coordinate_unit"),
            "scale_to_m": output.get("scale_to_m"),
        },
        {"coordinate_unit": "mm", "scale_to_m": SI_SCALE},
    )
    manifest = report.get("diagnostics", {}).get("brep_manifest", {})
    manifest_entry = manifest.get("files", {}).get("film_zones_fragmented.brep", {})
    require(
        records,
        "input.brep_manifest_contract",
        manifest.get("overall") == "PASS"
        and manifest_entry.get("coordinate_unit") == "mm"
        and manifest_entry.get("scale_to_m") == SI_SCALE
        and manifest_entry.get("sha256") == actual_hash,
        manifest_entry,
        "validated millimetre BREP; apply 0.001 exactly once",
    )
    return report, reference_from_dict(report["cad_reference"])


def validate_cli(inputs: SurfaceSmokeInputs, records: list[dict[str, Any]]) -> None:
    require(records, "input.global_size_mm", inputs.global_size_mm > 0.0, inputs.global_size_mm, "> 0")
    require(records, "input.port_size_mm", inputs.port_size_mm > 0.0, inputs.port_size_mm, "> 0")
    require(
        records,
        "input.port_size_not_larger_than_global",
        inputs.port_size_mm <= inputs.global_size_mm,
        inputs.port_size_mm,
        f"<= {inputs.global_size_mm}",
    )
    require(
        records,
        "input.port_refine_radius_mm",
        inputs.port_refine_radius_mm > 0.0,
        inputs.port_refine_radius_mm,
        "> 0",
    )
    require(records, "input.max_elements", inputs.max_elements > 0, inputs.max_elements, "> 0")


def create_physical_groups(
    by_name: dict[str, Any], groups: dict[str, list[Any]]
) -> dict[str, dict[str, Any]]:
    definitions = {
        "journal_wall": (2, [surface.tag for surface in groups["journal_rotating_wall"]], False),
        "stationary_wall": (
            2,
            [surface.tag for surface in groups["stationary_bushing_feed_wall"]],
            False,
        ),
        "pressure_feed": (2, [groups["pressure_inlet"][0].tag], False),
        "axial_end_z0": (2, [groups["outlet_z0"][0].tag], False),
        "axial_end_zL": (2, [groups["outlet_zL"][0].tag], False),
        "internal_interface_z1": (2, [groups["interface_z1"][0].tag], True),
        "internal_interface_z2": (2, [groups["interface_z2"][0].tag], True),
        "ring_A": (3, [by_name["ring_A"].tag], False),
        "hole_band": (3, [by_name["hole_band"].tag], False),
        "ring_B": (3, [by_name["ring_B"].tag], False),
    }
    manifest: dict[str, dict[str, Any]] = {}
    for name, (dimension, entity_tags, internal) in definitions.items():
        physical_tag = int(gmsh.model.addPhysicalGroup(dimension, entity_tags))
        gmsh.model.setPhysicalName(dimension, physical_tag, name)
        manifest[name] = {
            "dimension": dimension,
            "physical_tag": physical_tag,
            "entity_tags": entity_tags,
            "internal": internal,
            "later_openfoam_wall_patch": dimension == 2 and not internal,
        }
    return manifest


def configure_mesh_sizes(
    inputs: SurfaceSmokeInputs, groups: dict[str, list[Any]]
) -> dict[str, Any]:
    global_size = inputs.global_size_mm * SI_SCALE
    port_size = inputs.port_size_mm * SI_SCALE
    refine_radius = inputs.port_refine_radius_mm * SI_SCALE

    feed_candidates = [
        surface
        for surface in groups["stationary_bushing_feed_wall"]
        if surface.entity_type == "Cylinder"
        and len(surface.properties) >= 6
        and parallel(surface.properties[3:6], (0.0, 1.0, 0.0))
    ]
    if len(feed_candidates) != 1:
        raise ValueError(f"expected one radial feed-wall cylinder, found {len(feed_candidates)}")
    feed_surface = feed_candidates[0]
    inlet_surface = groups["pressure_inlet"][0]
    refinement_curves = sorted(
        set(feed_surface.boundary_curve_tags) | set(inlet_surface.boundary_curve_tags)
    )

    settings = {
        "Mesh.MeshSizeFromPoints": 0.0,
        "Mesh.MeshSizeFromCurvature": 0.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.0,
        "Mesh.MeshSizeMin": port_size,
        "Mesh.MeshSizeMax": global_size,
        "Mesh.ElementOrder": 1.0,
    }
    for name, value in settings.items():
        gmsh.option.setNumber(name, value)

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", [feed_surface.tag])
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", refinement_curves)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", port_size)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", global_size)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", refine_radius)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    return {
        "user_units": "mm",
        "model_units": "m",
        "global_size_mm": inputs.global_size_mm,
        "port_size_mm": inputs.port_size_mm,
        "port_refine_radius_mm": inputs.port_refine_radius_mm,
        "global_size_m": global_size,
        "port_size_m": port_size,
        "port_refine_radius_m": refine_radius,
        "feed_surface_tag": feed_surface.tag,
        "refinement_curve_tags": refinement_curves,
        "distance_field": distance,
        "threshold_field": threshold,
        "implicit_sizing_disabled": {
            name: gmsh.option.getNumber(name) for name in settings if "From" in name
        },
    }


def collect_surface_elements(
    surfaces: Sequence[Any], records: list[dict[str, Any]]
) -> tuple[dict[int, list[int]], dict[int, tuple[int, int, int]]]:
    surface_elements: dict[int, list[int]] = {}
    connectivity: dict[int, tuple[int, int, int]] = {}
    element_properties: dict[int, dict[str, Any]] = {}

    for surface in surfaces:
        element_types, element_tags_by_type, node_tags_by_type = gmsh.model.mesh.getElements(
            2, surface.tag
        )
        tags_for_surface: list[int] = []
        for element_type, element_tags, node_tags in zip(
            element_types, element_tags_by_type, node_tags_by_type
        ):
            name, dimension, order, node_count, _local, primary_count = (
                gmsh.model.mesh.getElementProperties(int(element_type))
            )
            element_properties[int(element_type)] = {
                "name": name,
                "dimension": int(dimension),
                "order": int(order),
                "node_count": int(node_count),
                "primary_node_count": int(primary_count),
            }
            require(
                records,
                f"mesh.surface_{surface.tag}.element_type_{int(element_type)}",
                dimension == 2
                and order == 1
                and node_count == 3
                and primary_count == 3
                and str(name).startswith("Triangle"),
                element_properties[int(element_type)],
                "first-order 3-node triangle",
            )
            flat_nodes = [int(value) for value in node_tags]
            for index, raw_tag in enumerate(element_tags):
                tag = int(raw_tag)
                nodes = tuple(flat_nodes[index * 3 : index * 3 + 3])
                if len(nodes) != 3:
                    raise ValueError(f"triangle {tag} has {len(nodes)} nodes")
                connectivity[tag] = nodes  # entity ownership makes tags unique
                tags_for_surface.append(tag)
        require(
            records,
            f"mesh.surface_{surface.tag}.has_elements",
            bool(tags_for_surface),
            len(tags_for_surface),
            "> 0",
        )
        surface_elements[surface.tag] = tags_for_surface

    require(
        records,
        "mesh.element_tags_unique_across_surfaces",
        len(connectivity) == sum(len(tags) for tags in surface_elements.values()),
        len(connectivity),
        sum(len(tags) for tags in surface_elements.values()),
    )
    return surface_elements, connectivity


def mesh_node_coordinates() -> tuple[dict[int, tuple[float, float, float]], tuple[float, ...]]:
    node_tags, coordinates, _parametric = gmsh.model.mesh.getNodes()
    nodes = {
        int(tag): (
            float(coordinates[index * 3]),
            float(coordinates[index * 3 + 1]),
            float(coordinates[index * 3 + 2]),
        )
        for index, tag in enumerate(node_tags)
    }
    bbox = (
        min(point[0] for point in nodes.values()),
        min(point[1] for point in nodes.values()),
        min(point[2] for point in nodes.values()),
        max(point[0] for point in nodes.values()),
        max(point[1] for point in nodes.values()),
        max(point[2] for point in nodes.values()),
    )
    return nodes, bbox


def quality_statistics(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
    }


def validate_mesh(
    inputs: SurfaceSmokeInputs,
    records: list[dict[str, Any]],
    reference: Any,
    surfaces: Sequence[Any],
    groups: dict[str, list[Any]],
    physical_groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    surface_elements, connectivity = collect_surface_elements(surfaces, records)
    element_tags = sorted(connectivity)
    qualities = [float(value) for value in gmsh.model.mesh.getElementQualities(element_tags, "minSICN")]
    quality_by_element = dict(zip(element_tags, qualities))
    require(
        records,
        "mesh.minSICN_finite_positive_and_no_inversion",
        len(qualities) == len(element_tags)
        and all(math.isfinite(value) and value > 0.0 for value in qualities),
        {
            "count": len(qualities),
            "nonpositive_or_nonfinite": sum(
                not math.isfinite(value) or value <= 0.0 for value in qualities
            ),
            "minimum": min(qualities, default=None),
        },
        "all finite and > 0",
    )

    nodes, node_bbox = mesh_node_coordinates()
    triangle_areas = {
        tag: triangle_area(*(nodes[node_tag] for node_tag in node_tags))
        for tag, node_tags in connectivity.items()
    }
    require(
        records,
        "mesh.no_zero_area_triangles",
        all(math.isfinite(area) and area > 0.0 for area in triangle_areas.values()),
        {
            "minimum_area_m2": min(triangle_areas.values(), default=None),
            "invalid_count": sum(
                not math.isfinite(area) or area <= 0.0 for area in triangle_areas.values()
            ),
        },
        "all finite and > 0",
    )
    duplicate_nodes = [int(tag) for tag in gmsh.model.mesh.getDuplicateNodes()]
    require(records, "mesh.no_duplicate_nodes", not duplicate_nodes, duplicate_nodes, [])

    dimension_counts: dict[int, int] = {}
    for dimension in range(4):
        _types, tags_by_type, _nodes = gmsh.model.mesh.getElements(dimension)
        dimension_counts[dimension] = sum(len(tags) for tags in tags_by_type)
    require(
        records,
        "mesh.all_2d_elements_belong_to_classified_surfaces",
        dimension_counts[2] == len(element_tags),
        dimension_counts[2],
        len(element_tags),
    )
    require(records, "mesh.no_3d_elements", dimension_counts[3] == 0, dimension_counts[3], 0)
    total_elements = sum(dimension_counts.values())
    require(
        records,
        "mesh.element_count_limit",
        total_elements < inputs.max_elements,
        total_elements,
        f"< {inputs.max_elements}",
    )

    axial_span = node_bbox[5] - node_bbox[2]
    require(
        records,
        "mesh.node_bbox_in_metres",
        relative_error(axial_span, reference.length * SI_SCALE) <= MESH_BBOX_REL_TOL
        and max(abs(value) for value in node_bbox) < 1.0,
        {"bbox_m": node_bbox, "axial_span_m": axial_span},
        {"axial_span_m": reference.length * SI_SCALE, "max_abs_coordinate_m": "< 1"},
        MESH_BBOX_REL_TOL,
    )

    for name, group in physical_groups.items():
        entities = [
            int(tag)
            for tag in gmsh.model.getEntitiesForPhysicalGroup(
                int(group["dimension"]), int(group["physical_tag"])
            )
        ]
        require(records, f"physical_group.{name}.nonempty", bool(entities), entities, group["entity_tags"])

    interface_checks: dict[str, Any] = {}
    for role in ("interface_z1", "interface_z2"):
        interface_surfaces = groups[role]
        interface = interface_surfaces[0]
        passed = (
            len(interface_surfaces) == 1
            and len(interface.adjacent_volume_tags) == 2
            and bool(surface_elements[interface.tag])
        )
        interface_checks[role] = {
            "surface_tag": interface.tag,
            "adjacent_volumes": interface.adjacent_volume_tags,
            "triangle_count": len(surface_elements[interface.tag]),
        }
        require(
            records,
            f"mesh.{role}.single_shared_entity",
            passed,
            interface_checks[role],
            "one meshed surface adjacent to two volumes",
        )

    inlet = groups["pressure_inlet"][0]
    inlet_circle_tags = [
        tag
        for tag, kind in zip(inlet.boundary_curve_tags, inlet.boundary_curve_types)
        if kind == "Circle"
    ]
    inlet_segments = 0
    for curve_tag in sorted(set(inlet_circle_tags)):
        element_types, tags_by_type, _node_tags = gmsh.model.mesh.getElements(1, curve_tag)
        for element_type, tags in zip(element_types, tags_by_type):
            properties = gmsh.model.mesh.getElementProperties(int(element_type))
            require(
                records,
                f"mesh.inlet_curve_{curve_tag}.first_order_lines",
                properties[1] == 1 and properties[2] == 1 and properties[3] == 2,
                properties[:4],
                "first-order 2-node line",
            )
            inlet_segments += len(tags)
    require(records, "mesh.inlet_circle_segments", inlet_segments >= 48, inlet_segments, ">= 48")

    inlet_triangle_area = sum(triangle_areas[tag] for tag in surface_elements[inlet.tag])
    inlet_occ_area = inlet.area
    require(
        records,
        "mesh.inlet_triangulated_area",
        relative_error(inlet_triangle_area, inlet_occ_area) <= INLET_AREA_REL_TOL,
        {
            "triangulated_area_m2": inlet_triangle_area,
            "occ_area_m2": inlet_occ_area,
            "relative_error": relative_error(inlet_triangle_area, inlet_occ_area),
        },
        inlet_occ_area,
        INLET_AREA_REL_TOL,
    )

    surface_group_sources = {
        "journal_wall": groups["journal_rotating_wall"],
        "stationary_wall": groups["stationary_bushing_feed_wall"],
        "pressure_feed": groups["pressure_inlet"],
        "axial_end_z0": groups["outlet_z0"],
        "axial_end_zL": groups["outlet_zL"],
        "internal_interface_z1": groups["interface_z1"],
        "internal_interface_z2": groups["interface_z2"],
    }
    quality_rows: list[dict[str, Any]] = []
    for name, group_surfaces in surface_group_sources.items():
        tags = [tag for surface in group_surfaces for tag in surface_elements[surface.tag]]
        group_qualities = [quality_by_element[tag] for tag in tags]
        group_triangle_area = sum(triangle_areas[tag] for tag in tags)
        group_occ_area = sum(surface.area for surface in group_surfaces)
        stats = quality_statistics(group_qualities)
        quality_rows.append(
            {
                "boundary_group": name,
                "internal": name.startswith("internal_"),
                "surface_tags": ";".join(str(surface.tag) for surface in group_surfaces),
                "triangle_count": len(tags),
                "occ_area_m2": group_occ_area,
                "triangulated_area_m2": group_triangle_area,
                "area_relative_difference": relative_error(group_triangle_area, group_occ_area),
                **{f"minSICN_{key}": value for key, value in stats.items()},
            }
        )

    return {
        "node_count": len(nodes),
        "node_bbox_m": node_bbox,
        "element_counts_by_dimension": dimension_counts,
        "total_element_count": total_elements,
        "triangle_count": len(element_tags),
        "duplicate_node_count": len(duplicate_nodes),
        "inlet_circle_segment_count": inlet_segments,
        "inlet_occ_area_m2": inlet_occ_area,
        "inlet_triangulated_area_m2": inlet_triangle_area,
        "inlet_area_relative_error": relative_error(inlet_triangle_area, inlet_occ_area),
        "global_minSICN": quality_statistics(qualities),
        "interfaces": interface_checks,
        "quality_rows": quality_rows,
    }


def write_quality_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_surface_meshes(stage: Path, records: list[dict[str, Any]]) -> tuple[str, ...]:
    outputs = (
        "surface_mesh.msh",
        "surface_mesh_ascii.msh",
        "surface_mesh.vtk",
    )
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(stage / outputs[0]))
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(stage / outputs[1]))
    gmsh.write(str(stage / outputs[2]))
    for filename in outputs:
        path = stage / filename
        require(
            records,
            f"output.{filename}",
            path.is_file() and path.stat().st_size > 0,
            path.stat().st_size if path.exists() else 0,
            "> 0 bytes",
        )
    return outputs


def _run_mesh(
    inputs: SurfaceSmokeInputs,
    reference: Any,
    records: list[dict[str, Any]],
    stage: Path,
) -> dict[str, Any]:
    occ_options = configure_occ_options(records, "surface_smoke_import", SI_SCALE)
    gmsh.model.add("bearing_surface_smoke")
    import_brep(inputs.brep)
    by_name, surfaces, groups = validate_zone_model(
        records, "surface_smoke_geometry", reference, SI_SCALE
    )
    physical_groups = create_physical_groups(by_name, groups)
    sizing = configure_mesh_sizes(inputs, groups)
    require(
        records,
        "mesh.explicit_sizing_controls",
        all(value == 0.0 for value in sizing["implicit_sizing_disabled"].values()),
        sizing["implicit_sizing_disabled"],
        "all zero",
    )

    gmsh.model.mesh.generate(2)
    mesh = validate_mesh(inputs, records, reference, surfaces, groups, physical_groups)
    mesh_outputs = export_surface_meshes(stage, records)
    return {
        "coordinate_units": {
            "input_brep": "mm",
            "gmsh_model": "m",
            "mesh_nodes": "m",
            "areas": "m^2",
        },
        "scale_application": {
            "method": "Geometry.OCCScaling",
            "factor": SI_SCALE,
            "application_count": 1,
        },
        "occ_options": occ_options,
        "sizing": sizing,
        "physical_groups": physical_groups,
        "mesh": mesh,
        "mesh_outputs": mesh_outputs,
    }


def _publish_failure(
    outdir: Path,
    report: dict[str, Any],
    gmsh_lines: Sequence[str],
    records: Sequence[dict[str, Any]],
) -> None:
    stage = make_staging_directory(outdir)
    try:
        (stage / "surface_mesh_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_validation_log(
            stage / "gmsh_surface_mesh.log", "Gmsh 2D surface-mesh smoke test", gmsh_lines, records
        )
        atomic_replace_directory(stage, outdir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _publish_smoke_success(
    stage: Path,
    outdir: Path,
    base_report: dict[str, Any],
    preflight: dict[str, Any],
    preflight_report_path: Path,
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    gmsh_lines: Sequence[str],
) -> dict[str, Any]:
    quality_rows = diagnostics["mesh"].pop("quality_rows")
    write_quality_csv(stage / "surface_quality.csv", quality_rows)
    physical_manifest = {
        "schema_version": 1,
        "overall": "PASS",
        "coordinate_unit": "m",
        "input_coordinate_unit": "mm",
        "scale_to_m_applied_exactly_once": SI_SCALE,
        "internal_interfaces_are_not_wall_patches": True,
        "groups": diagnostics["physical_groups"],
    }
    (stage / "physical_groups.json").write_text(
        json.dumps(physical_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_validation_log(
        stage / "gmsh_surface_mesh.log",
        "Gmsh 2D surface-mesh smoke test",
        gmsh_lines,
        records,
    )
    output_names = (
        "surface_mesh.msh",
        "surface_mesh_ascii.msh",
        "surface_mesh.vtk",
        "physical_groups.json",
        "surface_quality.csv",
        "gmsh_surface_mesh.log",
    )
    report = {
        **base_report,
        "preflight_report_sha256": sha256_file(preflight_report_path),
        "preflight_overall": preflight["overall"],
        "validation_records": records,
        "diagnostics": diagnostics,
        "outputs": {
            name: {
                "sha256": sha256_file(stage / name),
                "bytes": (stage / name).stat().st_size,
                "coordinate_unit": "m" if name.startswith("surface_mesh") else "explicit in file",
            }
            for name in output_names
        },
        "overall": "PASS",
        "error": None,
    }
    (stage / "surface_mesh_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    atomic_replace_directory(stage, outdir)
    return report


def run_surface_smoke(inputs: SurfaceSmokeInputs) -> dict[str, Any]:
    inputs = SurfaceSmokeInputs(
        brep=inputs.brep.resolve(),
        preflight_report=inputs.preflight_report.resolve(),
        outdir=inputs.outdir.resolve(),
        global_size_mm=inputs.global_size_mm,
        port_size_mm=inputs.port_size_mm,
        port_refine_radius_mm=inputs.port_refine_radius_mm,
        max_elements=inputs.max_elements,
        gui=inputs.gui,
    )
    records: list[dict[str, Any]] = []
    base_report: dict[str, Any] = {
        "inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(inputs).items()
        },
        "coordinate_units": {
            "cli_sizes": "mm",
            "input_brep": "mm",
            "gmsh_model": "m",
            "mesh_nodes": "m",
            "surface_areas": "m^2",
        },
        "mesh_dimension": 2,
        "volume_mesh_generated": False,
        "dependency_versions": {"gmsh": gmsh.__version__, "python": sys.version.split()[0]},
    }
    try:
        validate_cli(inputs, records)
        preflight, reference = load_preflight_contract(inputs, records)
    except Exception as error:
        report = {
            **base_report,
            "validation_records": records,
            "overall": "FAIL",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _publish_failure(inputs.outdir, report, [], records)
        raise SurfaceSmokeRunError(str(error), report) from error

    stage = make_staging_directory(inputs.outdir)
    gmsh_lines: list[str] = []
    diagnostics: dict[str, Any] = {}
    caught: Exception | None = None
    initialized = False
    logger_started = False
    try:
        try:
            gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
            initialized = True
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.logger.start()
            logger_started = True
            diagnostics = _run_mesh(inputs, reference, records, stage)
            if inputs.gui:
                gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
                gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
                require(
                    records,
                    "gui.surface_mesh_display",
                    gmsh.option.getNumber("Mesh.SurfaceFaces") == 1
                    and gmsh.option.getNumber("Mesh.SurfaceEdges") == 1,
                    {
                        "Mesh.SurfaceFaces": gmsh.option.getNumber("Mesh.SurfaceFaces"),
                        "Mesh.SurfaceEdges": gmsh.option.getNumber("Mesh.SurfaceEdges"),
                    },
                    {"Mesh.SurfaceFaces": 1, "Mesh.SurfaceEdges": 1},
                )
                gmsh.fltk.run()
        except Exception as error:
            caught = error
        finally:
            if logger_started:
                gmsh_lines = [str(line) for line in gmsh.logger.get()]
                gmsh.logger.stop()
            if initialized:
                gmsh.finalize()

        if caught is not None:
            report = {
                **base_report,
                "preflight_report_sha256": sha256_file(inputs.preflight_report),
                "validation_records": records,
                "diagnostics": diagnostics,
                "overall": "FAIL",
                "error": {"type": type(caught).__name__, "message": str(caught)},
            }
            shutil.rmtree(stage)
            _publish_failure(inputs.outdir, report, gmsh_lines, records)
            raise SurfaceSmokeRunError(str(caught), report) from caught

        try:
            return _publish_smoke_success(
                stage,
                inputs.outdir,
                base_report,
                preflight,
                inputs.preflight_report,
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
                "preflight_report_sha256": sha256_file(inputs.preflight_report),
                "validation_records": records,
                "diagnostics": diagnostics,
                "overall": "FAIL",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            _publish_failure(inputs.outdir, report, gmsh_lines, records)
            raise SurfaceSmokeRunError(str(error), report) from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def print_report(report: dict[str, Any]) -> None:
    print("\nGmsh 2D surface-mesh smoke test")
    print(f"{'check':<62} {'status':<6}")
    print(f"{'-' * 62} {'-' * 6}")
    for record in report.get("validation_records", []):
        print(f"{record['name']:<62} {record['status']:<6}")
    print(f"\n3D volume mesh generated: {report.get('volume_mesh_generated', False)}")
    print(f"OVERALL: {report.get('overall', 'FAIL')}")


def parse_args(argv: Sequence[str] | None = None) -> SurfaceSmokeInputs:
    parser = argparse.ArgumentParser(description="Generate a validated 2D Gmsh boundary smoke mesh.")
    parser.add_argument("--brep", type=Path, default=SurfaceSmokeInputs.brep)
    parser.add_argument(
        "--preflight-report", type=Path, default=SurfaceSmokeInputs.preflight_report
    )
    parser.add_argument("--outdir", type=Path, default=SurfaceSmokeInputs.outdir)
    parser.add_argument("--global-size-mm", type=float, default=0.75)
    parser.add_argument("--port-size-mm", type=float, default=0.20)
    parser.add_argument("--port-refine-radius-mm", type=float, default=6.0)
    parser.add_argument("--max-elements", type=int, default=1_000_000)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    return SurfaceSmokeInputs(
        brep=args.brep,
        preflight_report=args.preflight_report,
        outdir=args.outdir,
        global_size_mm=args.global_size_mm,
        port_size_mm=args.port_size_mm,
        port_refine_radius_mm=args.port_refine_radius_mm,
        max_elements=args.max_elements,
        gui=args.gui,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_surface_smoke(parse_args(argv))
    except SurfaceSmokeRunError as error:
        print_report(error.report)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
