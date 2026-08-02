from __future__ import annotations

import json
from pathlib import Path

import bearing_cfd.bearings.conical_journal.geometry.cad as cad
import pytest


pytestmark = pytest.mark.integration


def _owned_quarantine_manifest() -> str:
    return json.dumps(
        {
            "schema": cad.REJECTED_STEP_SCHEMA,
            "producer": cad.PRODUCER,
            "status": "REJECTED_DIAGNOSTIC_ONLY",
            "solve_eligible": False,
        }
    )


def test_failed_step_does_not_block_native_brep_publication(tmp_path: Path) -> None:
    outdir = tmp_path / "strict"
    outdir.mkdir()
    (outdir / "film_unsplit.step").write_text("stale", encoding="utf-8")

    quarantine = tmp_path / "strict.rejected-step"
    quarantine.mkdir()
    (quarantine / "REJECTED.json").write_text(
        _owned_quarantine_manifest(), encoding="utf-8"
    )
    (quarantine / "stale.step").write_text("stale", encoding="utf-8")

    result = cad.main(
        ["--outdir", str(outdir), "--retain-failed-step"]
    )

    assert result == 0
    assert (outdir / "film_unsplit.brep").is_file()
    assert (outdir / "film_zones.brep").is_file()
    assert not tuple(outdir.glob("*.step"))
    assert not (tmp_path / "strict.failed").exists()
    assert json.loads((outdir / "run.json").read_text(encoding="utf-8"))["status"] == "PASS"
    params = json.loads((outdir / "params.json").read_text(encoding="utf-8"))
    assert params["overall"] == "PASS"
    assert params["step_exchange"]["status"] == "REJECTED"

    expected_steps = {
        "film_unsplit.step",
        "film_zones.step",
        "ring_A.step",
        "hole_band.step",
        "ring_B.step",
        "context_assembly.step",
    }
    assert {path.name for path in quarantine.glob("*.step")} == expected_steps
    assert not (quarantine / "stale.step").exists()

    rejected = json.loads((quarantine / "REJECTED.json").read_text(encoding="utf-8"))
    assert rejected["status"] == "REJECTED_DIAGNOSTIC_ONLY"
    assert rejected["solve_eligible"] is False
    assert rejected["schema"] == cad.REJECTED_STEP_SCHEMA
    assert rejected["producer"] == cad.PRODUCER
    assert set(rejected["files"]) == expected_steps
    assert all(record["sha256"] for record in rejected["files"].values())


def test_cleanup_refuses_unowned_quarantine(tmp_path: Path) -> None:
    outdir = tmp_path / "strict"
    quarantine = tmp_path / "strict.rejected-step"
    quarantine.mkdir()
    (quarantine / "REJECTED.json").write_text("{}", encoding="utf-8")

    with pytest.raises(cad.GeometryExportError):
        cad._discard_rejected_step_batch(outdir)

    assert quarantine.is_dir()


def test_early_failure_discards_owned_stale_quarantine(tmp_path: Path) -> None:
    outdir = tmp_path / "strict"
    quarantine = tmp_path / "strict.rejected-step"
    quarantine.mkdir()
    (quarantine / "REJECTED.json").write_text(
        _owned_quarantine_manifest(), encoding="utf-8"
    )

    assert cad.main(["--outdir", str(outdir), "--length", "0"]) == 2
    assert not quarantine.exists()


def test_geometry_and_run_options_preserve_flat_json_contract() -> None:
    inputs, options = cad._parse_args([])
    params = cad.resolve_params(inputs)

    assert set(cad._input_payload(inputs, options)) == {
        "length",
        "mean_radius",
        "semicone_angle_deg",
        "radial_clearance",
        "eccentricity_ratio",
        "eccentricity_angle_deg",
        "hole_diameter",
        "hole_axial_pos",
        "split_halfwidth",
        "bushing_wall_thickness",
        "inlet_extension",
        "axial_cutter_extension",
        "max_face_count",
        "export_debug_half",
        "retain_failed_step",
        "preview",
        "outdir",
    }
    assert set(cad._resolved_payload(params, options)) == {
        "length",
        "mean_radius",
        "semicone_angle_deg",
        "radial_clearance",
        "eccentricity_ratio",
        "eccentricity_angle_deg",
        "hole_diameter",
        "hole_axial_pos",
        "split_halfwidth",
        "bushing_wall_thickness",
        "inlet_extension",
        "axial_cutter_extension",
        "gamma_rad",
        "cone_slope",
        "hole_radius",
        "eccentricity",
        "phi_rad",
        "ex",
        "ey",
        "z_hole_min",
        "z_hole_max",
        "z1",
        "z2",
        "y_feed_end",
        "feed_start_disk_margin",
        "h_radial_min",
        "h_radial_max",
        "h_normal_min",
        "h_normal_max",
        "base_volume_exact",
        "feed_scale_estimate",
        "max_face_count",
        "export_debug_half",
        "preview",
        "outdir",
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--semicone-angle-deg", "inf"], "semicone_angle_deg must be finite"),
        (
            ["--eccentricity-angle-deg", "inf"],
            "eccentricity_angle_deg must be finite",
        ),
        (["--max-face-count", "0"], "max_face_count must be > 0"),
    ],
)
def test_invalid_cli_value_is_reported_as_parameter_error(
    tmp_path: Path,
    arguments: list[str],
    message: str,
) -> None:
    outdir = tmp_path / arguments[0].removeprefix("--")

    assert cad.main([*arguments, "--outdir", str(outdir)]) == 2

    failed = outdir.with_name(f"{outdir.name}.failed")
    report = json.loads((failed / "params.json").read_text(encoding="utf-8"))
    assert report["error"]["type"] == "ParameterValidationError"
    assert message in report["error"]["message"]
