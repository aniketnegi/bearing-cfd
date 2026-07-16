#!/usr/bin/env python3
"""Run a real, import-only PyFluent audit of the exported CGNS mesh."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PATCHES = (
    "journal_wall",
    "bushing_bore_wall",
    "pressure_feed",
    "axial_end_z0",
    "axial_end_zL",
    "feed_tube_wall",
)
FORBIDDEN = {"feed_mouth", "mouth_cap", "internal_feed", "defaultFaces"}


def fluent_capability() -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec("ansys.fluent.core")
    except (ImportError, ModuleNotFoundError):
        spec = None
    return {
        "available": spec is not None,
        "backend": "PyFluent" if spec is not None else None,
        "reason": None if spec is not None else "PyFluent is not installed",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numbers(pattern: str, text: str) -> tuple[float, ...] | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return tuple(float(value) for value in match.groups()) if match else None


def _integer(pattern: str, text: str) -> int | None:
    values = _numbers(pattern, text)
    return int(values[0]) if values else None


def parse_fluent_transcript(text: str) -> dict[str, Any]:
    number = r"([-+0-9.eE]+)"
    nodes = _integer(rf"\b{number}\s+nodes?\b", text)
    faces = _integer(rf"\b{number}\s+faces?\b", text)
    cells = _integer(rf"\b{number}\s+cells?\b", text)
    bbox: list[float] = []
    for axis in "xyz":
        extent = _numbers(rf"{axis}[- ]coordinate\s*:\s*{number}\s+(?:to|through)\s+{number}", text)
        if extent:
            bbox.extend(extent)
    minimum_volume = _numbers(rf"minimum\s+(?:cell\s+)?volume\s*[:=]\s*{number}", text)
    total_volume = _numbers(rf"total\s+(?:cell\s+)?volume\s*[:=]\s*{number}", text)
    maximum_skewness = _numbers(rf"maximum\s+(?:cell\s+)?skewness\s*[:=]\s*{number}", text)
    regions = _integer(rf"(?:connected|disconnected)\s+regions?\s*[:=]\s*{number}", text)
    return {
        "nodes": nodes,
        "faces": faces,
        "cells": cells,
        "bounding_box_m": bbox if len(bbox) == 6 else None,
        "minimum_cell_volume_m3": minimum_volume[0] if minimum_volume else None,
        "total_volume_m3": total_volume[0] if total_volume else None,
        "maximum_skewness": maximum_skewness[0] if maximum_skewness else None,
        "disconnected_region_count": regions,
    }


def _version(session: Any, pyfluent: Any) -> str:
    getter = getattr(session, "get_fluent_version", None)
    if callable(getter):
        return str(getter())
    return str(getattr(pyfluent, "__version__", "unknown"))


def run_fluent_import_audit(
    *,
    cgns: Path,
    canonical: dict[str, Any],
    outdir: Path,
    gui: bool = False,
) -> dict[str, Any]:
    """Return PASS only after querying a live Fluent process and saving its native mesh."""
    capability = fluent_capability()
    if not capability["available"]:
        return {"status": "NOT_RUN", "reason": capability["reason"], "real_import": False}

    import ansys.fluent.core as pyfluent

    outdir.mkdir(parents=True, exist_ok=True)
    transcript_path = outdir / "fluent_import_transcript.txt"
    report_path = outdir / "fluent_import_report.json"
    native_path = outdir / "bearing_prism_imported.msh.h5"
    session = None
    launched = False
    try:
        try:
            session = pyfluent.launch_fluent(
                mode=pyfluent.FluentMode.MESHING,
                dimension=pyfluent.Dimension.THREE,
                precision=pyfluent.Precision.DOUBLE,
                processor_count=1,
                ui_mode="gui" if gui else "no_gui_or_graphics",
                cwd=str(outdir),
                start_transcript=False,
            )
            launched = True
        except Exception as error:
            return {
                "status": "NOT_RUN",
                "reason": f"Fluent launch/license unavailable: {type(error).__name__}: {error}",
                "real_import": False,
            }

        session.transcript.start(file_name=str(transcript_path), write_to_stdout=False)
        session.tui.file.import_.cgns_vol_mesh(str(cgns.resolve()))
        session.tui.mesh.check_quality_level(1)
        session.tui.mesh.check_mesh()
        session.tui.mesh.check_quality()
        utilities = session.meshing_utilities
        boundary_ids = [int(value) for value in utilities.get_face_zones(maximum_entity_count=1.0e30, only_boundary=True)]
        cell_ids = [int(value) for value in utilities.get_cell_zones(maximum_entity_count=1.0e30)]
        boundary_names = set(utilities.convert_zone_ids_to_name_strings(zone_id_list=boundary_ids))
        cell_names = set(utilities.convert_zone_ids_to_name_strings(zone_id_list=cell_ids))
        face_counts = {
            name: int(utilities.get_face_zone_count(face_zone_name_list=[name]))
            for name in PATCHES
        }
        cell_count = int(utilities.get_cell_zone_count(cell_zone_name_list=["fluid"]))
        cell_shapes = [str(utilities.get_cell_zone_shape(cell_zone_id=value)) for value in cell_ids]
        session.transcript.stop()
        transcript = transcript_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_fluent_transcript(transcript)
        required_metrics = (
            "nodes",
            "faces",
            "cells",
            "bounding_box_m",
            "minimum_cell_volume_m3",
            "total_volume_m3",
            "maximum_skewness",
            "disconnected_region_count",
        )
        missing_metrics = [name for name in required_metrics if parsed[name] is None]
        expected_counts = canonical["patch_counts"]
        checks = {
            "boundary_names_exact": boundary_names == set(PATCHES),
            "cell_zone_exact": cell_names == {"fluid"} and len(cell_ids) == 1,
            "forbidden_boundaries_absent": FORBIDDEN.isdisjoint(boundary_names),
            "patch_counts_exact": face_counts == expected_counts,
            "cell_count_exact": cell_count == int(canonical["prism6_cells"]),
            "transcript_metrics_complete": not missing_metrics,
            "prism_cell_type": all("prism" in value.lower() or "wedge" in value.lower() for value in cell_shapes),
        }
        if not missing_metrics:
            checks.update(
                {
                    "node_count_exact": parsed["nodes"] == int(canonical["points"]),
                    "face_count_exact": parsed["faces"] == int(canonical["total_faces"]),
                    "transcript_cell_count_exact": parsed["cells"] == int(canonical["prism6_cells"]),
                    "bounding_box_exact": bool(
                        np.max(
                            np.abs(np.asarray(parsed["bounding_box_m"]) - np.asarray(canonical["bounding_box_m"])),
                            initial=0.0,
                        )
                        <= 1.0e-12
                    ),
                    "volume_exact": abs(parsed["total_volume_m3"] - canonical["volume_m3"])
                    / max(abs(canonical["volume_m3"]), np.finfo(float).tiny)
                    <= 1.0e-10,
                    "positive_minimum_volume": parsed["minimum_cell_volume_m3"] > 0.0,
                    "valid_skewness": math.isfinite(parsed["maximum_skewness"])
                    and 0.0 <= parsed["maximum_skewness"] <= 1.0,
                    "one_connected_region": parsed["disconnected_region_count"] == 1,
                }
            )
        failure_markers = re.findall(
            r"(?:mesh check failed|invalid connectivity|negative-volume cells?\s*:\s*[1-9]\d*|zero-volume cells?\s*:\s*[1-9]\d*)",
            transcript,
            flags=re.IGNORECASE,
        )
        checks["mesh_check_no_failure"] = not failure_markers
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "real_import": True,
            "backend": "PyFluent",
            "ansys_fluent_version": _version(session, pyfluent),
            "mode": "3D double-precision meshing/import-only",
            "source_cgns": str(cgns.resolve()),
            "checks": checks,
            "missing_transcript_metrics": missing_metrics,
            "nodes": parsed["nodes"],
            "faces": parsed["faces"],
            "cells": cell_count,
            "cell_element_types": cell_shapes,
            "cell_zones": sorted(cell_names),
            "face_zones": sorted(boundary_names),
            "face_counts": face_counts,
            "bounding_box_m": parsed["bounding_box_m"],
            "total_volume_m3": parsed["total_volume_m3"],
            "minimum_cell_volume_m3": parsed["minimum_cell_volume_m3"],
            "maximum_skewness": parsed["maximum_skewness"],
            "disconnected_region_count": parsed["disconnected_region_count"],
            "transcript": transcript_path.name,
            "flow_initialized": False,
            "solver_iterations": 0,
            "solution_fields": 0,
        }
        if report["status"] == "PASS":
            session.tui.file.write_mesh(str(native_path.resolve()))
            if not native_path.is_file():
                report["status"] = "FAIL"
                report["checks"]["native_mesh_saved"] = False
                report["reason"] = "Fluent did not create the native .msh.h5 file"
            else:
                report["checks"]["native_mesh_saved"] = True
                report["native_mesh"] = native_path.name
        else:
            report["reason"] = "one or more mandatory live Fluent checks failed"
        _write_json(report_path, report)
        return report
    except Exception as error:
        report = {
            "status": "FAIL" if launched else "NOT_RUN",
            "real_import": launched,
            "reason": f"{type(error).__name__}: {error}",
            "flow_initialized": False,
            "solver_iterations": 0,
        }
        _write_json(report_path, report)
        return report
    finally:
        if session is not None:
            try:
                session.exit()
            except Exception:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and audit bearing_prism.cgns in a real Fluent session without solving.")
    parser.add_argument("--interchange-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "required"), default="required")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory = args.interchange_dir.resolve()
    report_file = directory / "interchange_report.json"
    if not report_file.is_file():
        print(f"missing static interchange report: {report_file}", file=sys.stderr)
        return 1
    static = json.loads(report_file.read_text(encoding="utf-8"))
    result = run_fluent_import_audit(
        cgns=directory / "bearing_prism.cgns",
        canonical=static["source"],
        outdir=directory,
        gui=args.gui,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "PASS":
        return 0
    if result["status"] == "NOT_RUN" and args.mode == "auto":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
