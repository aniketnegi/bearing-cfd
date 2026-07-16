from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from meshing.gmsh_brep_preflight import PreflightInputs, run_preflight
from meshing.gmsh_surface_smoke import SurfaceSmokeInputs, run_surface_smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["strict_default", "strict_case_e03_g20"])
def test_native_brep_preflight(case: str) -> None:
    source = PROJECT_ROOT / "out" / case
    temporary_parent = PROJECT_ROOT / "out"
    temporary_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temporary_parent, prefix=f".pytest-{case}-") as directory:
        outdir = Path(directory) / "preflight"
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
        ):
            assert (outdir / filename).is_file()

        smoke_outdir = Path(directory) / "surface_smoke"
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
        ):
            assert (smoke_outdir / filename).is_file()
