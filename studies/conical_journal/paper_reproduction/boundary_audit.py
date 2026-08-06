#!/usr/bin/env python3
"""Audit the paper's under-specified oil-supply boundary at one journal position."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy

from bearing_cfd.artifacts import make_staging_directory, publish_generation
from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo as film
from studies.conical_journal.paper_reproduction.run import (
    DEFAULT_INPUT,
    REPO_ROOT,
    equilibrium_case,
    load_study,
    make_inputs,
    resource_record,
    solve_case,
)


DEFAULT_OUTPUT = Path(
    "out/conical_journal/studies/paper-reproduction/supply-boundary-audit"
)


def supply_cases(
    diameters_mm: Sequence[float],
    feed_gauge_pressures_kpa: Sequence[float],
    nominal_diameter_mm: float,
) -> list[tuple[str, float, float]]:
    cases = [
        ("reynolds", 0.0, 0.0),
        ("reynolds", nominal_diameter_mm, 0.0),
        ("jfo", 0.0, 0.0),
        *(("jfo", diameter, 0.0) for diameter in diameters_mm),
        *(
            ("jfo", nominal_diameter_mm, pressure)
            for pressure in feed_gauge_pressures_kpa
        ),
    ]
    return list(dict.fromkeys(cases))


def fixed_supply_case(spec: tuple[object, ...]) -> dict[str, object]:
    (
        model,
        diameter_mm,
        pressure_kpa,
        base_conditions,
        eccentricity_ratio,
        eccentricity_angle_deg,
        n_theta,
        n_axial,
        max_revolutions,
    ) = spec
    conditions = dict(base_conditions)
    conditions["feed_diameter_m"] = float(diameter_mm) / 1000
    conditions["feed_gauge_pressure_pa"] = float(pressure_kpa) * 1000
    inputs = make_inputs(
        conditions,
        float(eccentricity_ratio),
        float(eccentricity_angle_deg),
        int(n_theta),
        int(n_axial),
        int(max_revolutions),
    )
    _, _, result = solve_case(str(model), inputs)
    result["case"] = (
        f"{model}-d{float(diameter_mm):g}mm-p{float(pressure_kpa):g}kpa"
    )
    result["supply_boundary"] = {
        "diameter_m": inputs.feed_diameter_m,
        "gauge_pressure_pa": inputs.feed_gauge_pressure_pa,
        "fixed_fill_fraction": 1.0 if inputs.feed_diameter_m > 0 else None,
    }
    return result


def run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    study = load_study(args.input)
    conditions = dict(study["conditions"])
    figure8 = study["figure8"]
    assert isinstance(figure8, dict)
    if any(value <= 0 for value in args.port_diameters_mm):
        raise ValueError("port diameters must be positive")
    if any(value <= 0 for value in args.feed_gauge_pressures_kpa):
        raise ValueError("feed gauge pressures must be positive")
    if args.nominal_diameter_mm <= 0 or args.jobs < 1:
        raise ValueError("nominal diameter and jobs must be positive")

    target_load = float(figure8["target_radial_load_n"])
    paper_conditions = dict(conditions)
    paper_conditions["feed_diameter_m"] = 0.0
    paper_conditions["feed_gauge_pressure_pa"] = 0.0
    equilibrium = equilibrium_case(
        (
            "reynolds",
            paper_conditions,
            args.n_theta,
            args.n_axial,
            args.max_revolutions,
            target_load,
            args.initial_eccentricity_ratio,
            args.initial_angle_deg,
        )
    )
    if not equilibrium["equilibrium_accepted"]:
        raise RuntimeError("paper-boundary Reynolds equilibrium failed")

    cases = supply_cases(
        args.port_diameters_mm,
        args.feed_gauge_pressures_kpa,
        args.nominal_diameter_mm,
    )
    specs = [
        (
            model,
            diameter_mm,
            pressure_kpa,
            conditions,
            equilibrium["eccentricity_ratio"],
            equilibrium["eccentricity_angle_deg"],
            args.n_theta,
            args.n_axial,
            args.max_revolutions,
        )
        for model, diameter_mm, pressure_kpa in cases
    ]
    jobs = min(args.jobs, len(specs), os.cpu_count() or 1)
    target = args.outdir.resolve()
    stage = make_staging_directory(target)
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(fixed_supply_case, specs))

        with (stage / "supply-boundary-audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "case",
                    "model",
                    "feed_diameter_mm",
                    "feed_gauge_pressure_kpa",
                    "maximum_gauge_pressure_kpa",
                    "radial_pressure_force_n",
                    "pressure_force_x_n",
                    "pressure_force_y_n",
                    "axial_pressure_force_n",
                    "numerical_accepted",
                    "relative_mass_imbalance",
                    "minimum_fill_fraction",
                    "ruptured_area_fraction",
                    "feed_cells",
                    "feed_area_mm2",
                )
            )
            for result in results:
                numerical = result["numerical"]
                supply = result["supply_boundary"]
                assert isinstance(numerical, dict) and isinstance(supply, dict)
                force = result["pressure_force_n"]
                assert isinstance(force, list)
                writer.writerow(
                    (
                        result["case"],
                        result["model"],
                        float(supply["diameter_m"]) * 1000,
                        float(supply["gauge_pressure_pa"]) / 1000,
                        float(result["maximum_gauge_pressure_pa"]) / 1000,
                        result["radial_pressure_force_n"],
                        force[0],
                        force[1],
                        result["axial_pressure_force_n"],
                        numerical["accepted"],
                        numerical.get("relative_mass_imbalance"),
                        numerical.get("minimum_fill_fraction"),
                        numerical["ruptured_area_fraction"],
                        result["feed_cells"],
                        float(result["feed_area_m2"]) * 1e6,
                    )
                )

        expected_invalid = {"jfo-d0mm-p0kpa"}
        unexpected_failures = [
            result["case"]
            for result in results
            if not result["numerical"]["accepted"]
            and result["case"] not in expected_invalid
        ]
        status = (
            "NUMERICAL_PASS_WITH_EXPECTED_NO_FEED_REJECTION"
            if not unexpected_failures
            else "PARTIAL_OR_FAILED"
        )
        compact_equilibrium = {
            key: value
            for key, value in equilibrium.items()
            if key not in {"profile_theta_deg", "profile_pressure_kpa"}
        }
        summary = {
            "status": status,
            "purpose": "boundary-condition sensitivity, not parameter calibration",
            "common_journal_position": compact_equilibrium,
            "paper_method_boundary": {
                "model": "pressure-only Reynolds with Reynolds rupture condition",
                "internal_feed_port": False,
                "axial_edges": "atmospheric pressure",
            },
            "results": results,
            "unexpected_failures": unexpected_failures,
            "resources": resource_record(started, jobs),
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        publish_generation(
            stage,
            target,
            stage="study",
            operation="paper-reproduction-supply-boundary-audit",
            status=status,
            argv=argv,
            resolved_inputs={
                "input": args.input,
                "target_radial_load_n": target_load,
                "n_theta": args.n_theta,
                "n_axial": args.n_axial,
                "max_revolutions": args.max_revolutions,
                "port_diameters_mm": args.port_diameters_mm,
                "nominal_diameter_mm": args.nominal_diameter_mm,
                "feed_gauge_pressures_kpa": args.feed_gauge_pressures_kpa,
                "jobs": jobs,
            },
            input_units={
                "pressure": "Pa unless key states kPa",
                "length": "m unless key states mm",
                "load": "N",
            },
            producer_files=(
                Path(__file__),
                Path(equilibrium_case.__code__.co_filename),
                Path(film.__file__),
            ),
            upstream_artifacts=(args.input,),
            tool_versions={"numpy": np.__version__, "scipy": scipy.__version__},
            acceptance_status=status,
            repository=REPO_ROOT,
        )
        stage = None
        return 0 if not unexpected_failures else 2
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-theta", type=int, default=256)
    parser.add_argument("--n-axial", type=int, default=80)
    parser.add_argument("--max-revolutions", type=int, default=24)
    parser.add_argument("--initial-eccentricity-ratio", type=float, default=0.014)
    parser.add_argument("--initial-angle-deg", type=float, default=-2.0)
    parser.add_argument(
        "--port-diameters-mm", type=float, nargs="+", default=[2.0, 4.0, 8.0]
    )
    parser.add_argument("--nominal-diameter-mm", type=float, default=4.0)
    parser.add_argument(
        "--feed-gauge-pressures-kpa",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 100.0, 150.0, 500.0],
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(values)
    try:
        return run(args, values)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
