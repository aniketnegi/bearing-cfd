#!/usr/bin/env python3
"""Generate every body-fitted study and one browseable mesh gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshing.body_fitted_inlet_study import (
    StudyInputs,
    run_study,
    write_combined_gallery,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_gallery(
    outdir: Path,
    *,
    openfoam: str = "skip",
    ansys: str = "required",
) -> dict:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    configurations = (
        (
            "Uniform 12-layer representation study",
            StudyInputs(
                preset="uniform-study",
                params=_REPO_ROOT / "out/strict_default/params.json",
                outdir=outdir / "uniform-study",
                openfoam=openfoam,
                ansys=ansys,
            ),
        ),
        (
            "Second-cone cross case",
            StudyInputs(
                preset="cross-case",
                params=_REPO_ROOT
                / "out/strict_case_e03_g20/params.json",
                outdir=outdir / "cross-case",
                openfoam=openfoam,
                ansys=ansys,
            ),
        ),
        (
            "Symmetric 5:1 inflation audit",
            StudyInputs(
                preset="inflation-audit",
                params=_REPO_ROOT / "out/strict_default/params.json",
                outdir=outdir / "inflation-audit",
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
    index = write_combined_gallery(outdir, studies)
    rows = [
        (label, row)
        for label, report, _source_dir in studies
        for row in report["cases"]
    ]
    summary = {
        "schema_version": 1,
        "purpose": "visual gallery plus canonical study results",
        "gallery": str(index),
        "experiment_rows": len(rows),
        "passed_rows": sum(row["status"] == "PASS" for _label, row in rows),
        "rejected_rows": sum(row["status"] != "PASS" for _label, row in rows),
        "solver_safety": (
            "Rejected meshes expose only explicitly labeled visual previews; "
            "they are not solver eligible."
        ),
        "studies": [
            {
                "label": label,
                "preset": report["preset"],
                "overall": report["overall"],
                "directory": str(source_dir),
                "gallery": str(source_dir / "study_gallery.html"),
                "comparison_json": str(source_dir / "comparison.json"),
                "comparison_csv": str(source_dir / "comparison.csv"),
                "viewer_commands": str(source_dir / "viewer_commands.txt"),
            }
            for label, report, source_dir in studies
        ],
    }
    (outdir / "all_studies.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (outdir / "OPEN_ME.txt").write_text(
        "BODY-FITTED MESH EXPERIMENT GALLERY\n\n"
        "Open index.html in a browser.\n"
        "Each study page has copy-paste commands for ParaView, Gmsh, and "
        "FreeCAD.\n\n"
        "Green PASS meshes may use their solver exports.\n"
        "Red REJECTED meshes are visual-only and must never be solved.\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all mesh studies and one visual gallery."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/body_fitted_experiment_gallery"),
    )
    parser.add_argument(
        "--openfoam",
        choices=("auto", "required", "skip"),
        default="skip",
    )
    parser.add_argument(
        "--ansys",
        choices=("auto", "required", "skip"),
        default="required",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        summary = generate_gallery(
            arguments.outdir,
            openfoam=arguments.openfoam,
            ansys=arguments.ansys,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"\nGallery ready: {summary['gallery']}\n"
        f"Rows: {summary['experiment_rows']} "
        f"({summary['passed_rows']} PASS, "
        f"{summary['rejected_rows']} REJECTED)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
