#!/usr/bin/env python3
"""Run the controlled BDL0-versus-BDL2 isothermal Reynolds-JFO screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from bearing_cfd.artifacts import make_staging_directory, publish_generation
from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo as jfo


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = Path(__file__).with_name("inputs") / "lubricants.json"
DEFAULT_OUTPUT = Path("out/conical_journal/studies/nanolubricant-isothermal")
CASE_IDS = ("BDL0", "BDL2")
SHA256 = re.compile(r"[0-9a-f]{64}")
METRICS = (
    ("maximum_gauge_pressure_pa", "Pa"),
    ("total_load_magnitude_n", "N"),
    ("journal_torque_magnitude_nm", "N m"),
    ("feed_flow_m3_s", "m^3/s"),
    ("feed_flow_kg_s", "kg/s"),
    ("cavitated_area_fraction", "1"),
)


def load_study(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("lubricant input schema_version must be 1")
    conditions = document.get("conditions")
    properties = document.get("property_sets")
    if not isinstance(conditions, dict) or not isinstance(properties, list):
        raise ValueError("lubricant inputs require conditions and property_sets")
    source = document.get("source")
    if not isinstance(source, dict) or not SHA256.fullmatch(
        str(source.get("source_pdf_sha256", ""))
    ):
        raise ValueError("lubricant inputs require the source PDF SHA-256")
    temperature = document.get("property_temperature_c")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("property_temperature_c must be finite")
    labels = [item.get("id") for item in properties if isinstance(item, dict)]
    if labels != list(CASE_IDS):
        raise ValueError("property_sets must contain BDL0 then BDL2")
    for item in properties:
        if not isinstance(item, dict):
            raise ValueError("every property set must be an object")
        for field in (
            "reported_tio2_volume_percent",
            "dynamic_viscosity_pa_s",
            "density_kg_m3",
        ):
            value = item.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{item['id']} {field} must be finite")
            if field == "reported_tio2_volume_percent":
                if value < 0:
                    raise ValueError(f"{item['id']} {field} must be non-negative")
            elif value <= 0:
                raise ValueError(f"{item['id']} {field} must be positive")
    return document


def case_metrics(summary: dict[str, object]) -> dict[str, float]:
    inputs = summary["inputs"]
    pressure = summary["pressure"]
    loads = summary["loads"]
    flow = summary["flow"]
    film = summary["film"]
    assert isinstance(inputs, dict)
    assert isinstance(pressure, dict)
    assert isinstance(loads, dict)
    assert isinstance(flow, dict)
    assert isinstance(film, dict)
    return {
        "maximum_gauge_pressure_pa": float(pressure["maximum_absolute_pa"])
        - float(inputs["ambient_pressure_pa"]),
        "total_load_magnitude_n": float(np.linalg.norm(loads["total_force_n"])),
        "journal_torque_magnitude_nm": abs(float(loads["journal_torque_z_nm"])),
        "feed_flow_m3_s": float(flow["feed_in_m3_s"]),
        "feed_flow_kg_s": float(flow["feed_in_kg_s"]),
        "cavitated_area_fraction": float(film["cavitated_area_fraction"]),
    }


def compare(summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    values = {case: case_metrics(summaries[case]) for case in CASE_IDS}
    metrics: dict[str, object] = {}
    for name, unit in METRICS:
        baseline = values["BDL0"][name]
        candidate = values["BDL2"][name]
        change = candidate - baseline
        metrics[name] = {
            "unit": unit,
            "BDL0": baseline,
            "BDL2": candidate,
            "absolute_change": change,
            "percent_change": None if baseline == 0 else 100 * change / baseline,
        }
    return metrics


def write_comparison_csv(path: Path, metrics: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("metric", "unit", "BDL0", "BDL2", "absolute_change", "percent_change")
        )
        for name, _ in METRICS:
            row = metrics[name]
            assert isinstance(row, dict)
            writer.writerow(
                (
                    name,
                    row["unit"],
                    row["BDL0"],
                    row["BDL2"],
                    row["absolute_change"],
                    row["percent_change"],
                )
            )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-theta", type=int)
    parser.add_argument("--n-axial", type=int)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args(values)

    target = args.outdir.resolve()
    stage = make_staging_directory(target)
    try:
        study = load_study(args.input)
        conditions = dict(study["conditions"])
        if args.n_theta is not None:
            conditions["n_theta"] = args.n_theta
        if args.n_axial is not None:
            conditions["n_axial"] = args.n_axial
        properties = {
            str(item["id"]): item for item in study["property_sets"]
        }
        summaries: dict[str, dict[str, object]] = {}
        for case_id in CASE_IDS:
            properties_for_case = properties[case_id]
            inputs = jfo.Inputs(
                **conditions,
                dynamic_viscosity_pa_s=float(
                    properties_for_case["dynamic_viscosity_pa_s"]
                ),
                density_kg_m3=float(properties_for_case["density_kg_m3"]),
            )
            grid, state = jfo.solve(inputs, log_every=args.log_every)
            summaries[case_id] = jfo.write_results(
                stage / case_id.lower(), inputs, grid, state
            )

        accepted = all(
            bool(summary["acceptance"]["accepted"])
            for summary in summaries.values()
        )
        metrics = compare(summaries)
        comparison = {
            "status": (
                "NUMERICAL_PASS_PHYSICALLY_UNVALIDATED" if accepted else "NUMERICAL_FAIL"
            ),
            "source": study["source"],
            "interpretation": study["interpretation"],
            "property_temperature_c": study["property_temperature_c"],
            "model_scope": study["model_scope"],
            "property_sets": study["property_sets"],
            "cases": {case: summaries[case]["inputs"] for case in CASE_IDS},
            "metrics": metrics,
        }
        (stage / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        write_comparison_csv(stage / "comparison.csv", metrics)
        publish_generation(
            stage,
            target,
            stage="study",
            operation="nanolubricant-isothermal",
            status=str(comparison["status"]),
            argv=values,
            resolved_inputs={
                "input": args.input,
                "conditions": conditions,
                "property_temperature_c": study["property_temperature_c"],
                "property_sets": study["property_sets"],
            },
            input_units={
                "rpm": "rev/min",
                "feed_gauge_pressure_pa": "Pa gauge",
                "ambient_pressure_pa": "Pa absolute",
                "cavitation_pressure_abs_pa": "Pa absolute",
                "dynamic_viscosity_pa_s": "Pa s at 40 C",
                "density_kg_m3": "kg/m^3",
                "reported_tio2_volume_percent": "vol% as reported by source",
                "property_temperature_c": "degC",
            },
            producer_files=(Path(__file__), Path(jfo.__file__)),
            upstream_artifacts=(args.input,),
            tool_versions={"numpy": np.__version__},
            acceptance_status=accepted,
            repository=REPO_ROOT,
        )
        stage = None
        return 0 if accepted else 2
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    raise SystemExit(main())
