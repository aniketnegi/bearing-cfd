import json
from pathlib import Path

from studies.conical_journal.nanolubricant_isothermal.run import (
    DEFAULT_INPUT,
    load_study,
    main,
)


def test_property_source_uses_reported_concentration_not_fraction() -> None:
    study = load_study(DEFAULT_INPUT)
    bdl0, bdl2 = study["property_sets"]

    assert study["property_temperature_c"] == 40
    assert bdl0["reported_tio2_volume_percent"] == 0
    assert bdl0["density_kg_m3"] == 918
    assert bdl2["reported_tio2_volume_percent"] == 0.02
    assert bdl2["dynamic_viscosity_pa_s"] == 0.03782
    assert study["source"]["source_pdf_sha256"] == (
        "96daa9070d30e5e21befe2e6f5561e8a6d35e36cece6c82fdc0a0635f5ffc490"
    )


def test_bdl0_bdl2_screen_writes_an_accepted_comparison(tmp_path: Path) -> None:
    outdir = tmp_path / "screen"

    assert main(
        [
            "--n-theta",
            "128",
            "--n-axial",
            "40",
            "--outdir",
            str(outdir),
        ]
    ) == 0

    comparison = json.loads((outdir / "comparison.json").read_text())
    assert comparison["status"] == "NUMERICAL_PASS_PHYSICALLY_UNVALIDATED"
    assert comparison["source"]["doi"] == "10.1177/1350650120981478"
    assert comparison["cases"]["BDL0"]["dynamic_viscosity_pa_s"] == 0.03462
    assert comparison["cases"]["BDL2"]["density_kg_m3"] == 979
    assert comparison["metrics"]["maximum_gauge_pressure_pa"]["BDL2"] > comparison[
        "metrics"
    ]["maximum_gauge_pressure_pa"]["BDL0"]
    assert comparison["metrics"]["feed_flow_m3_s"]["BDL2"] < comparison[
        "metrics"
    ]["feed_flow_m3_s"]["BDL0"]
    for case in ("bdl0", "bdl2"):
        summary = json.loads((outdir / case / "summary.json").read_text())
        assert summary["acceptance"]["accepted"] is True
        assert summary["flow"]["relative_imbalance"] < 0.005
    assert (outdir / "bdl0" / "state.npz").is_file()
    assert (outdir / "bdl2" / "state.npz").is_file()
    assert (outdir / "comparison.csv").is_file()
    assert (outdir / "run.json").is_file()
