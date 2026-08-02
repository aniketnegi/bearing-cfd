from __future__ import annotations

from pathlib import Path

import pytest

from bearing_cfd.bearings.conical_journal.meshing.brep_preflight import PreflightInputs, run_preflight
from bearing_cfd.bearings.conical_journal.meshing.surface_smoke import (
    SurfaceSmokeInputs,
    run_surface_smoke,
)


pytestmark = pytest.mark.integration


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "conical_journal"


@pytest.mark.parametrize("case", ["strict_default", "strict_case_e03_g20"])
def test_native_brep_preflight(case: str, tmp_path: Path) -> None:
    source = FIXTURES / "geometry" / case
    outdir = tmp_path / "preflight"
    report = run_preflight(
        PreflightInputs(
            unsplit=source / "film_unsplit.brep",
            zones=source / "film_zones.brep",
            params=source / "params.json",
            outdir=outdir,
        )
    )

    assert report["overall"] == "PASS"
    assert report["mesh_generated"] is False
    assert all(record["status"] == "PASS" for record in report["validation_records"])
    assert len(report["diagnostics"]["zones_fragmented_mm"]) == 3
    assert report["diagnostics"]["scaling"]["volume_ratio"] == pytest.approx(
        1.0e-9, rel=1.0e-10
    )
    assert report["diagnostics"]["scaling"]["area_ratio"] == pytest.approx(
        1.0e-6, rel=1.0e-10
    )

    for filename in (
        "preflight_report.json",
        "surfaces.csv",
        "volumes.csv",
        "film_zones_fragmented.brep",
        "film_zones_SI.brep",
        "brep_manifest.json",
        "gmsh_preflight.log",
        "run.json",
    ):
        assert (outdir / filename).is_file()

    smoke_outdir = tmp_path / "surface_smoke"
    smoke = run_surface_smoke(
        SurfaceSmokeInputs(
            brep=outdir / "film_zones_fragmented.brep",
            preflight_report=outdir / "preflight_report.json",
            outdir=smoke_outdir,
        )
    )
    assert smoke["overall"] == "PASS"
    assert smoke["volume_mesh_generated"] is False
    assert smoke["diagnostics"]["mesh"]["element_counts_by_dimension"][3] == 0
    assert smoke["diagnostics"]["mesh"]["inlet_circle_segment_count"] >= 48
    assert all(record["status"] == "PASS" for record in smoke["validation_records"])

    for filename in (
        "surface_mesh.msh",
        "surface_mesh_ascii.msh",
        "surface_mesh.vtk",
        "surface_mesh_report.json",
        "physical_groups.json",
        "surface_quality.csv",
        "gmsh_surface_mesh.log",
        "run.json",
    ):
        assert (smoke_outdir / filename).is_file()
