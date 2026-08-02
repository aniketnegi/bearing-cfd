#!/usr/bin/env python3
"""Run and compare the body-fitted circular surface-inlet cases."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import quote

from bearing_cfd.artifacts import make_staging_directory, publish_generation
from bearing_cfd.bearings.conical_journal.meshing import body_fitted_inlet as body
from bearing_cfd.bearings.conical_journal.meshing.brep_preflight import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARAMS = (
    REPO_ROOT
    / "tests/fixtures/conical_journal/geometry/strict_default/params.json"
)
CROSS_PARAMS = (
    REPO_ROOT
    / "tests/fixtures/conical_journal/geometry/strict_case_e03_g20/params.json"
)


@dataclass(frozen=True)
class CaseSpec:
    topology: Literal["tensor-warp", "ogrid"]
    geometry_mode: Literal["inscribed", "equal-area"]
    q: int
    inner_layers: int = 2
    outer_layers: int = 4


@dataclass(frozen=True)
class StudyInputs:
    preset: Literal["uniform-study", "cross-case", "inflation-audit"] = (
        "uniform-study"
    )
    params: Path = DEFAULT_PARAMS
    outdir: Path = Path("out/conical_journal/studies/body-fitted-mesh")
    cases: tuple[str, ...] = ()
    openfoam: Literal["auto", "required", "skip"] = "skip"
    ansys: Literal["auto", "required", "skip"] = "required"


BODY_CASES: dict[str, CaseSpec] = {
    "TW16-I": CaseSpec("tensor-warp", "inscribed", 4),
    "TW40-I": CaseSpec("tensor-warp", "inscribed", 10),
    "TW40-EA": CaseSpec("tensor-warp", "equal-area", 10),
    "TW80-I": CaseSpec("tensor-warp", "inscribed", 20),
    "TW80-EA": CaseSpec("tensor-warp", "equal-area", 20),
    "OG32-I": CaseSpec("ogrid", "inscribed", 8, 2, 4),
    "OG32-EA": CaseSpec("ogrid", "equal-area", 8, 2, 4),
    "OG64-I": CaseSpec("ogrid", "inscribed", 16, 2, 3),
    "OG64-EA": CaseSpec("ogrid", "equal-area", 16, 2, 3),
}

PRESET_CASES: dict[str, tuple[str, ...]] = {
    "uniform-study": (
        "S16",
        "S100",
        "S390",
        "TW16-I",
        "TW40-I",
        "TW40-EA",
        "TW80-I",
        "TW80-EA",
        "OG32-I",
        "OG32-EA",
        "OG64-I",
        "OG64-EA",
    ),
    "cross-case": ("TW80-I", "OG64-I"),
    "inflation-audit": ("S390", "TW80-I", "OG64-I"),
}

CSV_FIELDS = (
    "case",
    "status",
    "solver_eligible",
    "failure",
    "source_kind",
    "topology",
    "geometry_mode",
    "nominal_geometry",
    "research_variant",
    "n_theta",
    "n_axial",
    "n_gap",
    "gap_inflation_ratio",
    "q",
    "rim_segments",
    "inner_layers",
    "outer_layers",
    "support_size_cells",
    "pressure_face_count",
    "master_cell_count",
    "hex8_count",
    "point_count",
    "inlet_area_mm2",
    "nominal_circle_area_mm2",
    "area_relative_error",
    "perimeter_mm",
    "sagitta_mm",
    "effective_radius_mm",
    "radial_bias_mm",
    "area_correction_mm2",
    "minimum_master_scaled_jacobian",
    "minimum_minSICN",
    "minimum_minDetJac",
    "maximum_nonorthogonality_deg",
    "maximum_skewness",
    "mesh_volume_m3",
    "continuous_volume_relative_error",
    "source_report",
    "artifact_hashes",
)

_UNIFORM_REFERENCES = {
    "S16": REPO_ROOT
    / "out/professor_hex_inlet_mesh_comparison_ngap12/inlet16/mesh_report.json",
    "S100": REPO_ROOT
    / "out/professor_hex_inlet_mesh_comparison_ngap12/inlet100/mesh_report.json",
    "S390": REPO_ROOT
    / "out/professor_hex_inlet_mesh_comparison_ngap12/inlet390/mesh_report.json",
}
_INFLATION_REFERENCES = {
    "S390": REPO_ROOT
    / "out/structured_surface_inlet_default/nGap_12/mesh_report.json"
}
_UNIFORM_STAIRCASE_DIRS = {
    "S16": REPO_ROOT
    / "out/professor_surface_inlet_variants/inlet16_uniform_ngap12/nGap_12",
    "S100": REPO_ROOT
    / "out/professor_surface_inlet_variants/inlet100_moderate_ngap12/nGap_12",
    "S390": REPO_ROOT
    / "out/professor_surface_inlet_variants/inlet390_fine_ngap12/nGap_12",
}
_INFLATION_STAIRCASE_DIRS = {
    "S390": REPO_ROOT / "out/structured_surface_inlet_default/nGap_12"
}


def _value(data: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    return None


def _load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise body.BodyFittedError(f"cannot load reference report {path}: {error}") from error
    if report.get("overall") != "PASS":
        raise body.BodyFittedError(f"reference report is not PASS: {path}")
    return report


def _artifact_hashes(report: dict[str, Any], report_path: Path) -> dict[str, str]:
    hashes = {report_path.name: sha256_file(report_path)}
    for name, entry in report.get("files", {}).items():
        digest = entry.get("sha256") if isinstance(entry, dict) else entry
        if isinstance(digest, str):
            hashes[str(name)] = digest
    for name in ("manifest.json", "validation_report.json"):
        path = report_path.parent / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    for path in report_path.parent.iterdir():
        if path.is_file() and path.name not in hashes:
            hashes[path.name] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _reference_path(case_name: str, preset: str) -> Path:
    references = (
        _INFLATION_REFERENCES
        if preset == "inflation-audit"
        else _UNIFORM_REFERENCES
    )
    try:
        return references[case_name]
    except KeyError as error:
        raise body.BodyFittedError(
            f"{case_name} has no frozen reference for preset {preset}"
        ) from error


def _reference_row(
    case_name: str, preset: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _reference_path(case_name, preset)
    report = _load_report(path)
    analytic = report["analytic"]
    pressure = analytic["pressure_patch"]
    quality = report["quality"]
    counts = analytic["counts"]
    row = {
        "case": case_name,
        "status": "PASS",
        "solver_eligible": True,
        "failure": None,
        "source_kind": "frozen-staircase-reference",
        "topology": "staircase",
        "geometry_mode": None,
        "nominal_geometry": False,
        "research_variant": False,
        "n_theta": report["inputs"]["n_theta"],
        "n_axial": report["inputs"]["n_axial"],
        "n_gap": report["inputs"]["n_gap"],
        "gap_inflation_ratio": report["inputs"]["gap_inflation_ratio"],
        "q": None,
        "rim_segments": pressure["rim_edge_count"],
        "inner_layers": None,
        "outer_layers": None,
        "support_size_cells": None,
        "pressure_face_count": counts["boundary_quads"]["pressure_feed"],
        "master_cell_count": counts["hexes"] // report["inputs"]["n_gap"],
        "hex8_count": counts["hexes"],
        "point_count": counts["points"],
        "inlet_area_mm2": pressure["projected_xz_area_mm2"],
        "nominal_circle_area_mm2": pressure["analytic_circle_area_mm2"],
        "area_relative_error": pressure["projected_area_relative_error"],
        "perimeter_mm": None,
        "sagitta_mm": None,
        "effective_radius_mm": None,
        "radial_bias_mm": None,
        "area_correction_mm2": None,
        "minimum_master_scaled_jacobian": None,
        "minimum_minSICN": quality["minSICN"]["min"],
        "minimum_minDetJac": quality["minDetJac"]["min"],
        "maximum_nonorthogonality_deg": quality[
            "max_nonorthogonality_deg"
        ]["max"],
        "maximum_skewness": quality["max_skewness"]["max"],
        "mesh_volume_m3": analytic["mesh_volume_m3"],
        "continuous_volume_relative_error": analytic[
            "faceted_volume_relative_error"
        ],
        "source_report": str(path),
        "artifact_hashes": _artifact_hashes(report, path),
    }
    return row, report


def _body_row(
    case_name: str,
    spec: CaseSpec,
    report: dict[str, Any],
    report_path: Path,
    inlet: body.InletSpec,
    gap_inflation_ratio: float,
) -> dict[str, Any]:
    geometry = body.rim_geometry_diagnostics(
        inlet, 4 * spec.q, spec.geometry_mode
    )
    master = _value(
        report,
        ("master_validation",),
        ("validation", "master"),
        ("master", "validation"),
    ) or {}
    volume = _value(
        report,
        ("body_validation",),
        ("mesh_validation",),
        ("validation", "body"),
        ("validation", "mesh"),
    ) or {}
    counts = volume.get("counts", {})
    boundary_counts = counts.get("boundary_Quad4", {})
    quality = volume.get("quality", {})
    master_count = (
        256 * 96
        if spec.topology == "tensor-warp"
        else 256 * 96
        + 4 * spec.q * (spec.inner_layers + spec.outer_layers)
    )
    pressure_faces = (
        spec.q**2
        if spec.topology == "tensor-warp"
        else spec.q**2 + 4 * spec.q * spec.inner_layers
    )
    return {
        "case": case_name,
        "status": report["overall"],
        "solver_eligible": report["overall"] == "PASS",
        "failure": (
            report.get("error", {}).get("message")
            if report["overall"] != "PASS"
            else None
        ),
        "source_kind": "body-fitted",
        "topology": spec.topology,
        "geometry_mode": spec.geometry_mode,
        "nominal_geometry": geometry["nominal_geometry"],
        "research_variant": geometry["research_variant"],
        "n_theta": 256,
        "n_axial": 96,
        "n_gap": 12,
        "gap_inflation_ratio": gap_inflation_ratio,
        "q": spec.q,
        "rim_segments": 4 * spec.q,
        "inner_layers": (
            spec.inner_layers if spec.topology == "ogrid" else None
        ),
        "outer_layers": (
            spec.outer_layers if spec.topology == "ogrid" else None
        ),
        "support_size_cells": (
            f"{2 * spec.q}x{2 * spec.q}"
            if spec.topology == "tensor-warp"
            else f"{spec.q}x{spec.q}"
        ),
        "pressure_face_count": boundary_counts.get(
            "pressure_feed", pressure_faces
        ),
        "master_cell_count": master.get("counts", {}).get(
            "quads", master_count
        ),
        "hex8_count": counts.get("Hex8", master_count * 12),
        "point_count": counts.get("points"),
        "inlet_area_mm2": geometry["polygon_area_mm2"],
        "nominal_circle_area_mm2": geometry["nominal_circle_area_mm2"],
        "area_relative_error": geometry["polygon_area_relative_error"],
        "perimeter_mm": geometry["polygon_perimeter_mm"],
        "sagitta_mm": geometry["chord_sagitta_mm"],
        "effective_radius_mm": geometry["effective_radius_mm"],
        "radial_bias_mm": geometry["radial_bias_mm"],
        "area_correction_mm2": geometry["area_correction_mm2"],
        "minimum_master_scaled_jacobian": master.get("quality", {}).get(
            "minimum_scaled_jacobian"
        ),
        "minimum_minSICN": quality.get("minimum_minSICN"),
        "minimum_minDetJac": quality.get("minimum_minDetJac"),
        "maximum_nonorthogonality_deg": quality.get(
            "maximum_nonorthogonality_deg"
        ),
        "maximum_skewness": quality.get("maximum_skewness"),
        "mesh_volume_m3": volume.get("volume", {}).get("cell_sum_m3"),
        "continuous_volume_relative_error": volume.get("volume", {}).get(
            "continuous_relative_error"
        ),
        "source_report": f"{case_name}/{report_path.name}",
        "artifact_hashes": _artifact_hashes(report, report_path),
    }


def _case_names(inputs: StudyInputs) -> tuple[str, ...]:
    names = inputs.cases or PRESET_CASES[inputs.preset]
    known = set(BODY_CASES) | {"S16", "S100", "S390"}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise body.BodyFittedError(
            f"unknown cases {unknown}; choose from {sorted((*BODY_CASES, 'S16', 'S100', 'S390'))}"
        )
    if len(set(names)) != len(names):
        raise body.BodyFittedError("case names must be unique")
    return names


def _body_report_path(case_dir: Path) -> Path:
    for name in (
        "validation_report.json",
        "failure_report.json",
        "mesh_report.json",
        "run_report.json",
    ):
        path = case_dir / name
        if path.is_file():
            return path
    raise body.BodyFittedError(f"body-fitted case wrote no report in {case_dir}")


def _staircase_dir(case_name: str, preset: str) -> Path:
    directories = (
        _INFLATION_STAIRCASE_DIRS
        if preset == "inflation-audit"
        else _UNIFORM_STAIRCASE_DIRS
    )
    return directories.get(case_name, _UNIFORM_STAIRCASE_DIRS[case_name])


def _case_assets(
    row: dict[str, Any], preset: str, study_dir: Path
) -> dict[str, Path]:
    case_name = str(row["case"])
    rejected = row.get("status") != "PASS"
    if row.get("source_kind") == "frozen-staircase-reference":
        case_dir = _staircase_dir(case_name, preset)
        assets = {
            "paraview": case_dir / "boundary_quads.vtu",
            "paraview_cutaway": case_dir / "viz/cutaway_exact.vtu",
            "gmsh": case_dir / "structured_hex.msh",
            "footprint": case_dir / "images/pressure_feed_footprint.png",
            "local_mesh": case_dir / "images/pressure_feed_footprint.png",
            "quality": case_dir / "images/quality_unavailable.png",
        }
    else:
        case_dir = study_dir / case_name
        assets = {
            "paraview": case_dir / "viz/master_surface.vtu",
            "paraview_cutaway": case_dir / "viz/cutaway_exact.vtu",
            "gmsh": (
                case_dir / "viz/DIAGNOSTIC_ONLY_master_surface.msh"
                if rejected
                else case_dir / "structured_hex.msh"
            ),
            "footprint": case_dir / "plots/footprint.png",
            "local_mesh": case_dir / "plots/local_master_mesh.png",
            "quality": case_dir / "plots/master_quality.png",
        }
    assets["fallback_image"] = assets["local_mesh"]
    assets["render"] = study_dir / "renders" / f"{case_name}.png"
    return assets


def _copy_gallery_geometry(params_path: Path, outdir: Path) -> list[Path]:
    geometry_dir = outdir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        params_path.parent / "film_unsplit.brep",
        params_path.parent / "film_zones.brep",
    ]
    if params_path.parent.name == "strict_default":
        sources.append(
            REPO_ROOT
            / "out/ansys_surface_inlet_default"
            / "VISUAL_CONTEXT_ONLY_context_assembly.step"
        )
    copied: list[Path] = []
    for source in sources:
        if source.is_file():
            target = geometry_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
    warning = (
        "VISUAL GEOMETRY ONLY.\n"
        "These BREP/STEP files help FreeCAD show the bearing context. They do "
        "not contain the CFD volume-mesh physical groups, and the circular "
        "inlet is a mesh boundary partition rather than STEP solver geometry.\n"
    )
    missing = [source.name for source in sources if not source.is_file()]
    if missing:
        warning += "Not found: " + ", ".join(missing) + "\n"
    (geometry_dir / "README_VISUAL_ONLY.txt").write_text(
        warning, encoding="utf-8"
    )
    return copied


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _href(path: Path, base: Path) -> str:
    return quote(_relative_path(path, base), safe="/._-")


def _format_metric(value: Any, format_spec: str = ".3g") -> str:
    if value is None or value == "":
        return "—"
    try:
        return format(float(value), format_spec)
    except (TypeError, ValueError):
        return html.escape(str(value))


def _format_count(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _asset_link(
    path: Path,
    base: Path,
    label: str,
    *,
    css_class: str = "",
) -> str:
    if not path.is_file():
        return f'<span class="missing">{html.escape(label)} unavailable</span>'
    class_attr = f' class="{css_class}"' if css_class else ""
    return (
        f'<a{class_attr} href="{_href(path, base)}">'
        f"{html.escape(label)}</a>"
    )


def _render_study_images(
    outdir: Path, comparison: dict[str, Any]
) -> dict[str, str]:
    render_dir = outdir / "renders"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    pvpython = shutil.which("pvpython")
    if pvpython is None:
        return {"ParaView": "pvpython was not found; plot images are used."}
    script = Path(__file__).with_name("render.py")
    failures: dict[str, str] = {}
    render_dir.mkdir(parents=True, exist_ok=True)
    for row in comparison.get("cases", []):
        assets = _case_assets(row, comparison["preset"], outdir)
        source = assets["paraview"]
        if not source.is_file():
            failures[str(row["case"])] = f"missing input: {source}"
            continue
        command = [
            pvpython,
            "--force-offscreen-rendering",
            str(script),
            "--input",
            str(source),
            "--output",
            str(assets["render"]),
            "--title",
            str(row["case"]),
        ]
        if row.get("status") != "PASS":
            command.append("--rejected")
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        environment["XDG_CONFIG_HOME"] = str(outdir / ".paraview-config")
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures[str(row["case"])] = str(error)
            continue
        if result.returncode:
            message = (result.stderr or result.stdout or "unknown error").strip()
            failures[str(row["case"])] = message[-2000:]
    return failures


def _gallery_section(
    label: str,
    comparison: dict[str, Any],
    source_dir: Path,
    page_dir: Path,
) -> str:
    preset = str(comparison.get("preset", "study"))
    cards: list[str] = []
    table_rows: list[str] = []
    geometry = [
        source_dir / "geometry/film_unsplit.brep",
        source_dir / "geometry/film_zones.brep",
        source_dir
        / "geometry/VISUAL_CONTEXT_ONLY_context_assembly.step",
    ]
    geometry_links = " ".join(
        _asset_link(path, page_dir, path.name) for path in geometry
    )
    for row in comparison.get("cases", []):
        status = str(row.get("status", "FAIL"))
        rejected = status != "PASS"
        case_name = html.escape(str(row.get("case", "?")))
        assets = _case_assets(row, preset, source_dir)
        image_path = (
            assets["render"]
            if assets["render"].is_file()
            else assets["fallback_image"]
        )
        image = (
            f'<a href="{_href(image_path, page_dir)}">'
            f'<img src="{_href(image_path, page_dir)}" '
            f'alt="{case_name} mesh preview" loading="lazy"></a>'
            if image_path.is_file()
            else '<div class="no-image">No preview image was made.</div>'
        )
        extra_images = []
        seen_images = {image_path.resolve()}
        for key, caption in (
            ("local_mesh", "Local mesh"),
            ("footprint", "Inlet footprint"),
            ("quality", "Master quality"),
        ):
            path = assets[key]
            if path.is_file() and path.resolve() not in seen_images:
                seen_images.add(path.resolve())
                extra_images.append(
                    f'<figure><a href="{_href(path, page_dir)}">'
                    f'<img src="{_href(path, page_dir)}" '
                    f'alt="{case_name} {caption.lower()}" loading="lazy">'
                    f"</a><figcaption>{caption}</figcaption></figure>"
                )
        image_strip = (
            f'<div class="image-strip">{"".join(extra_images)}</div>'
            if extra_images
            else ""
        )
        gmsh_label = (
            "Gmsh diagnostic surface (DO NOT SOLVE)"
            if rejected
            else "Gmsh solver mesh"
        )
        links = " ".join(
            (
                _asset_link(
                    assets["paraview"], page_dir, "ParaView surface VTU"
                ),
                _asset_link(
                    assets["paraview_cutaway"],
                    page_dir,
                    "ParaView cutaway Hex8 VTU",
                    css_class="danger" if rejected else "",
                ),
                _asset_link(
                    assets["gmsh"],
                    page_dir,
                    gmsh_label,
                    css_class="danger" if rejected else "",
                ),
            )
        )
        failure = html.escape(str(row.get("failure") or ""))
        warning = (
            '<p class="rejected">REJECTED — VISUAL ONLY — DO NOT SOLVE</p>'
            if rejected
            else ""
        )
        research_warning = (
            '<p class="research">NON-NOMINAL RESEARCH GEOMETRY</p>'
            if row.get("research_variant")
            else ""
        )
        cards.append(
            f"""
            <article class="card {'fail' if rejected else 'pass'}">
              <header><h3>{case_name}</h3><span class="badge">{html.escape(status)}</span></header>
              {warning}{research_warning}{image}
              {image_strip}
              <dl>
                <dt>Topology</dt><dd>{html.escape(str(row.get('topology') or '—'))}</dd>
                <dt>Geometry</dt><dd>{html.escape(str(row.get('geometry_mode') or '—'))}</dd>
                <dt>Research variant</dt><dd>{'yes' if row.get('research_variant') else 'no'}</dd>
                <dt>Area error</dt><dd>{_format_metric(row.get('area_relative_error'), '.3e')}</dd>
                <dt>Pressure faces</dt><dd>{_format_count(row.get('pressure_face_count'))}</dd>
                <dt>Master cells</dt><dd>{_format_count(row.get('master_cell_count'))}</dd>
                <dt>Hex8 cells</dt><dd>{_format_count(row.get('hex8_count'))}</dd>
                <dt>Max non-orth</dt><dd>{_format_metric(row.get('maximum_nonorthogonality_deg'))}°</dd>
                <dt>Max skew</dt><dd>{_format_metric(row.get('maximum_skewness'))}</dd>
                <dt>Volume error</dt><dd>{_format_metric(row.get('continuous_volume_relative_error'), '.3e')}</dd>
              </dl>
              {f'<p class="failure">{failure}</p>' if failure else ''}
              <p class="links">{links}</p>
            </article>"""
        )
        table_rows.append(
            "<tr>"
            f"<th>{case_name}</th><td class=\"{status.lower()}\">{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('topology') or '—'))}</td>"
            f"<td>{_format_metric(row.get('area_relative_error'), '.3e')}</td>"
            f"<td>{_format_count(row.get('pressure_face_count'))}</td>"
            f"<td>{_format_count(row.get('master_cell_count'))}</td>"
            f"<td>{_format_count(row.get('hex8_count'))}</td>"
            f"<td>{_format_metric(row.get('maximum_nonorthogonality_deg'))}</td>"
            f"<td>{_format_metric(row.get('maximum_skewness'))}</td>"
            f"<td>{_format_metric(row.get('continuous_volume_relative_error'), '.3e')}</td>"
            f"<td>{failure or '—'}</td></tr>"
        )
    comparison_links = " ".join(
        _asset_link(source_dir / name, page_dir, name)
        for name in ("comparison.json", "comparison.csv", "viewer_commands.txt")
    )
    return f"""
    <section class="study">
      <h2>{html.escape(label)}</h2>
      <p><strong>Overall: {html.escape(str(comparison.get('overall', 'FAIL')))}</strong>
      · preset {html.escape(preset)} · {comparison_links}</p>
      <p class="geometry"><strong>FreeCAD visual geometry:</strong> {geometry_links}
      <br><small>Visual context only. The BREP/STEP files do not carry CFD mesh groups.</small></p>
      <div class="cards">{''.join(cards)}</div>
      <div class="table-wrap"><table>
        <thead><tr><th>Case</th><th>Status</th><th>Topology</th><th>Area err.</th>
        <th>Pressure faces</th><th>Master cells</th><th>Hex8</th><th>Max non-orth °</th>
        <th>Max skew</th><th>Volume err.</th><th>Failure</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table></div>
    </section>"""


def _gallery_document(title: str, sections: Sequence[str]) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--good:#16845b;--bad:#b42318;--ink:#182230;--pale:#f5f7fa}}
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.45 system-ui,sans-serif;color:var(--ink);background:var(--pale)}}
main{{max-width:1500px;margin:auto;padding:24px}} h1{{margin-bottom:4px}} h2{{margin-top:44px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.card{{background:white;border:2px solid #d0d5dd;border-radius:12px;padding:14px;overflow:hidden}}
.card.pass{{border-color:var(--good)}} .card.fail{{border-color:var(--bad);background:#fff7f6}}
.card header{{display:flex;align-items:center;justify-content:space-between}} .card h3{{margin:0}}
.badge{{font-weight:800}} .pass .badge,.pass td{{color:var(--good)}} .fail .badge,.fail td{{color:var(--bad)}}
.rejected,.danger,.failure{{color:var(--bad);font-weight:800}} .failure{{overflow-wrap:anywhere}}
.research{{color:#9a6700;font-weight:800}}
img{{display:block;width:100%;aspect-ratio:3/2;object-fit:contain;background:#eef2f6;border-radius:8px;margin:12px 0}}
.image-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}}
.image-strip figure{{margin:0}} .image-strip img{{margin:0;aspect-ratio:1/1;cursor:zoom-in}}
.image-strip figcaption{{font-size:12px;text-align:center;color:#475467;padding:3px}}
.no-image{{min-height:180px;display:grid;place-items:center;background:#eef2f6;margin:12px 0}}
dl{{display:grid;grid-template-columns:1fr 1fr;margin:8px 0}} dt,dd{{padding:3px 0;margin:0}} dd{{text-align:right;font-variant-numeric:tabular-nums}}
.links a,.geometry a{{display:inline-block;margin:4px 8px 4px 0}} .missing{{color:#667085;margin-right:8px}}
.geometry{{background:#fff6d8;padding:12px;border-left:5px solid #eaaa08}}
.table-wrap{{overflow:auto;margin-top:18px}} table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:8px;border:1px solid #d0d5dd;text-align:right;white-space:nowrap}} th:first-child,td:last-child{{text-align:left}}
td:last-child{{white-space:normal;min-width:240px}} thead th{{background:#e9edf2;position:sticky;top:0}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p>Passing meshes are shown in green. Rejected meshes are shown in red and are for visual inspection only.</p>
<p>Each study header links to its <code>viewer_commands.txt</code> file with commands for ParaView, Gmsh, and FreeCAD.</p>
{''.join(sections)}
</main></body></html>
"""


def _write_viewer_commands(outdir: Path, comparison: dict[str, Any]) -> Path:
    lines = [
        "# Run these commands from this study directory.",
        "# REJECTED means visual-only: never solve or convert that mesh.",
        "",
    ]
    for row in comparison.get("cases", []):
        case_name = str(row["case"])
        rejected = row.get("status") != "PASS"
        assets = _case_assets(row, comparison["preset"], outdir)
        lines.append(
            f"# {case_name}: {'REJECTED — VISUAL ONLY' if rejected else 'PASS'}"
        )
        if assets["paraview"].is_file():
            lines.append(
                "paraview "
                + shlex.quote(_relative_path(assets["paraview"], outdir))
            )
        if assets["paraview_cutaway"].is_file():
            if rejected:
                lines.append("# DIAGNOSTIC CUTAWAY ONLY — DO NOT SOLVE")
            lines.append(
                "paraview "
                + shlex.quote(
                    _relative_path(assets["paraview_cutaway"], outdir)
                )
            )
        if assets["gmsh"].is_file():
            if rejected:
                lines.append("# DIAGNOSTIC SURFACE ONLY — DO NOT SOLVE")
            lines.append(
                "gmsh "
                + shlex.quote(_relative_path(assets["gmsh"], outdir))
            )
        lines.append("")
    lines.append("# FreeCAD visual geometry (no CFD physical groups):")
    for name in (
        "film_unsplit.brep",
        "film_zones.brep",
        "VISUAL_CONTEXT_ONLY_context_assembly.step",
    ):
        if (outdir / "geometry" / name).is_file():
            lines.append("freecad " + shlex.quote(f"geometry/{name}"))
    lines.append("")
    path = outdir / "viewer_commands.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_study_gallery(
    outdir: Path, comparison: dict[str, Any], render_failures: dict[str, str]
) -> Path:
    failures_path = outdir / "render_failures.txt"
    if render_failures:
        failures_path.write_text(
            "\n\n".join(
                f"{case}:\n{message}"
                for case, message in sorted(render_failures.items())
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        failures_path.unlink(missing_ok=True)
    section = _gallery_section(
        str(comparison.get("preset", "Study")), comparison, outdir, outdir
    )
    path = outdir / "study_gallery.html"
    path.write_text(
        _gallery_document("Surface-inlet mesh study", (section,)),
        encoding="utf-8",
    )
    return path


def write_combined_gallery(
    outdir: Path,
    studies: Sequence[tuple[str, dict[str, Any], Path]],
) -> Path:
    """Write one image-and-results index for previously generated studies."""
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sections = [
        _gallery_section(label, comparison, source_dir.resolve(), outdir)
        for label, comparison, source_dir in studies
    ]
    path = outdir / "index.html"
    path.write_text(
        _gallery_document("All surface-inlet experiments", sections),
        encoding="utf-8",
    )
    return path


def run_study(inputs: StudyInputs) -> dict[str, Any]:
    params_path = inputs.params.resolve()
    outdir = inputs.outdir.resolve()
    if not params_path.is_file():
        raise body.BodyFittedError(f"params file not found: {params_path}")
    names = _case_names(inputs)
    outdir.mkdir(parents=True, exist_ok=True)
    inlet = body.load_inlet_spec(params_path)
    gap_inflation_ratio = 5.0 if inputs.preset == "inflation-audit" else 1.0
    rows: list[dict[str, Any]] = []
    for case_name in names:
        if case_name.startswith("S"):
            row, report = _reference_row(case_name, inputs.preset)
        else:
            spec = BODY_CASES[case_name]
            case_dir = outdir / case_name
            try:
                report = body.run_body_fitted_case(
                    body.BodyFittedCaseInputs(
                        params=params_path,
                        outdir=case_dir,
                        case_name=case_name,
                        topology=spec.topology,
                        geometry_mode=spec.geometry_mode,
                        q=spec.q,
                        inner_layers=spec.inner_layers,
                        outer_layers=spec.outer_layers,
                        n_theta=256,
                        n_axial=96,
                        n_gap=12,
                        gap_inflation_ratio=gap_inflation_ratio,
                        openfoam=inputs.openfoam,
                        ansys=inputs.ansys,
                    )
                )
            except body.BodyFittedRunError as error:
                report = error.report
            report_path = _body_report_path(case_dir)
            row = _body_row(
                case_name,
                spec,
                report,
                report_path,
                inlet,
                gap_inflation_ratio,
            )
        rows.append(row)

    comparison = {
        "schema_version": 1,
        "overall": (
            "PASS"
            if all(row["status"] == "PASS" for row in rows)
            else "FAIL"
        ),
        "preset": inputs.preset,
        "params": str(params_path),
        "params_sha256": sha256_file(params_path),
        "openfoam": inputs.openfoam,
        "ansys": inputs.ansys,
        "case_order": list(names),
        "cases": rows,
    }
    (outdir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (outdir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["artifact_hashes"] = json.dumps(
                csv_row["artifact_hashes"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(csv_row)
    _copy_gallery_geometry(params_path, outdir)
    render_failures = _render_study_images(outdir, comparison)
    _write_viewer_commands(outdir, comparison)
    _write_study_gallery(outdir, comparison, render_failures)
    return comparison


def generate_gallery(
    outdir: Path,
    *,
    default_params: Path = DEFAULT_PARAMS,
    cross_params: Path = CROSS_PARAMS,
    openfoam: str = "skip",
    ansys: str = "required",
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Generate the three retained comparisons as one atomic study artifact."""
    target = outdir.resolve()
    stage = make_staging_directory(target)
    try:
        configurations = (
            (
                "Uniform 12-layer representation study",
                StudyInputs(
                    preset="uniform-study",
                    params=default_params,
                    outdir=stage / "uniform-study",
                    openfoam=openfoam,
                    ansys=ansys,
                ),
            ),
            (
                "Second-cone cross case",
                StudyInputs(
                    preset="cross-case",
                    params=cross_params,
                    outdir=stage / "cross-case",
                    openfoam=openfoam,
                    ansys=ansys,
                ),
            ),
            (
                "Symmetric 5:1 inflation audit",
                StudyInputs(
                    preset="inflation-audit",
                    params=default_params,
                    outdir=stage / "inflation-audit",
                    openfoam=openfoam,
                    ansys=ansys,
                ),
            ),
        )
        studies = []
        for label, inputs in configurations:
            print(f"\n{label}")
            report = run_study(inputs)
            studies.append((label, report, inputs.outdir.resolve()))
        write_combined_gallery(stage, studies)
        rows = [
            row
            for _label, report, _source_dir in studies
            for row in report["cases"]
        ]
        summary = {
            "schema_version": 1,
            "purpose": "visual gallery plus canonical study results",
            "gallery": "index.html",
            "experiment_rows": len(rows),
            "passed_rows": sum(row["status"] == "PASS" for row in rows),
            "rejected_rows": sum(row["status"] != "PASS" for row in rows),
            "solver_safety": (
                "Rejected meshes expose only explicitly labelled visual previews; "
                "they are not solver eligible."
            ),
            "studies": [
                {
                    "label": label,
                    "preset": report["preset"],
                    "overall": report["overall"],
                    "directory": inputs.outdir.name,
                    "gallery": f"{inputs.outdir.name}/study_gallery.html",
                    "comparison_json": f"{inputs.outdir.name}/comparison.json",
                    "comparison_csv": f"{inputs.outdir.name}/comparison.csv",
                    "viewer_commands": f"{inputs.outdir.name}/viewer_commands.txt",
                }
                for (label, report, _source_dir), (_unused, inputs) in zip(
                    studies, configurations, strict=True
                )
            ],
        }
        (stage / "all_studies.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "OPEN_ME.txt").write_text(
            "BODY-FITTED MESH EXPERIMENT GALLERY\n\n"
            "Open index.html in a browser.\n"
            "Green PASS meshes may use their solver exports.\n"
            "Red REJECTED meshes are visual-only and must never be solved.\n",
            encoding="utf-8",
        )
        publish_generation(
            stage,
            target,
            stage="study",
            operation="body-fitted-mesh",
            status="COMPLETE",
            resolved_inputs={
                "default_params": default_params.resolve(),
                "cross_params": cross_params.resolve(),
                "openfoam": openfoam,
                "ansys": ansys,
            },
            input_units={"geometry": "mm", "mesh": "m"},
            producer_files=(Path(__file__), Path(__file__).with_name("render.py")),
            upstream_artifacts=(default_params.resolve(), cross_params.resolve()),
            tool_versions={
                "gmsh": body.gmsh.__version__,
                "meshio": body.meshio.__version__,
                "numpy": body.np.__version__,
            },
            argv=argv,
            acceptance_status={
                "passed_rows": summary["passed_rows"],
                "rejected_rows": summary["rejected_rows"],
            },
            repository=REPO_ROOT,
        )
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the retained body-fitted mesh studies and gallery."
    )
    parser.add_argument("--outdir", type=Path, default=StudyInputs.outdir)
    parser.add_argument("--default-params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--cross-params", type=Path, default=CROSS_PARAMS)
    parser.add_argument(
        "--openfoam", choices=("auto", "required", "skip"), default="skip"
    )
    parser.add_argument(
        "--ansys", choices=("auto", "required", "skip"), default="required"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_args(values)
    try:
        summary = generate_gallery(
            arguments.outdir,
            default_params=arguments.default_params,
            cross_params=arguments.cross_params,
            openfoam=arguments.openfoam,
            ansys=arguments.ansys,
            argv=values,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Gallery ready: {arguments.outdir.resolve() / summary['gallery']}\n"
        f"Rows: {summary['experiment_rows']} "
        f"({summary['passed_rows']} PASS, "
        f"{summary['rejected_rows']} REJECTED)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
