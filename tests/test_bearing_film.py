from __future__ import annotations

import json
from pathlib import Path

import bearing_film
import pytest


def _owned_quarantine_manifest() -> str:
    return json.dumps(
        {
            "schema": bearing_film.REJECTED_STEP_SCHEMA,
            "producer": "bearing_film.py",
            "status": "REJECTED_DIAGNOSTIC_ONLY",
            "solve_eligible": False,
        }
    )


def test_failed_step_batch_is_quarantined(tmp_path: Path) -> None:
    outdir = tmp_path / "strict"
    outdir.mkdir()
    (outdir / "film_unsplit.step").write_text("stale", encoding="utf-8")

    quarantine = tmp_path / "strict.rejected-step"
    quarantine.mkdir()
    (quarantine / "REJECTED.json").write_text(
        _owned_quarantine_manifest(), encoding="utf-8"
    )
    (quarantine / "stale.step").write_text("stale", encoding="utf-8")

    result = bearing_film.main(
        ["--outdir", str(outdir), "--retain-failed-step"]
    )

    assert result == 2
    assert not list(outdir.glob("*.step"))
    assert (outdir / "film_unsplit.brep").is_file()
    assert (outdir / "film_zones.brep").is_file()

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
    assert rejected["schema"] == bearing_film.REJECTED_STEP_SCHEMA
    assert rejected["producer"] == "bearing_film.py"
    assert set(rejected["files"]) == expected_steps
    assert all(record["sha256"] for record in rejected["files"].values())


def test_cleanup_refuses_unowned_quarantine(tmp_path: Path) -> None:
    outdir = tmp_path / "strict"
    quarantine = tmp_path / "strict.rejected-step"
    quarantine.mkdir()
    (quarantine / "REJECTED.json").write_text("{}", encoding="utf-8")

    with pytest.raises(bearing_film.GeometryExportError):
        bearing_film._discard_rejected_step_batch(outdir)

    assert quarantine.is_dir()


def test_early_failure_discards_owned_stale_quarantine(tmp_path: Path) -> None:
    outdir = tmp_path / "strict"
    quarantine = tmp_path / "strict.rejected-step"
    quarantine.mkdir()
    (quarantine / "REJECTED.json").write_text(
        _owned_quarantine_manifest(), encoding="utf-8"
    )

    assert bearing_film.main(["--outdir", str(outdir), "--length", "0"]) == 2
    assert not quarantine.exists()
