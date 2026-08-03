#!/usr/bin/env python3
"""Compare Reynolds and JFO interpretations of Gangrade et al. Figures 6 and 8."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy
from scipy.optimize import least_squares

from bearing_cfd.artifacts import make_staging_directory, publish_generation
from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo as film


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = Path(__file__).with_name("inputs") / "paper_graph_read.json"
DEFAULT_OUTPUT = Path("out/conical_journal/studies/paper-reproduction")
MODELS = ("reynolds", "jfo")


def load_study(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("paper input schema_version must be 1")
    for key in ("source", "conditions", "interpretation", "figure6", "figure8"):
        if not isinstance(document.get(key), dict):
            raise ValueError(f"paper inputs require {key}")
    conditions = document["conditions"]
    assert isinstance(conditions, dict)
    if float(conditions["feed_gauge_pressure_pa"]) != 0:
        raise ValueError("paper-reproduction feed must be 0 Pa gauge")
    if float(conditions["paper_reference_pressure_pa"]) <= 0:
        raise ValueError("paper reference pressure must be positive")
    return document


def make_inputs(
    conditions: dict[str, object],
    eccentricity_ratio: float,
    eccentricity_angle_deg: float,
    n_theta: int,
    n_axial: int,
    max_revolutions: int,
) -> film.Inputs:
    return film.Inputs(
        rpm=float(conditions["rpm"]),
        n_theta=n_theta,
        n_axial=n_axial,
        length_m=float(conditions["length_m"]),
        mean_radius_m=float(conditions["mean_radius_m"]),
        semicone_angle_deg=float(conditions["semicone_angle_deg"]),
        radial_clearance_m=float(conditions["radial_clearance_m"]),
        eccentricity_m=(eccentricity_ratio * float(conditions["radial_clearance_m"])),
        eccentricity_angle_deg=eccentricity_angle_deg,
        feed_diameter_m=float(conditions["feed_diameter_m"]),
        feed_gauge_pressure_pa=float(conditions["feed_gauge_pressure_pa"]),
        ambient_pressure_pa=float(conditions["ambient_pressure_pa"]),
        cavitation_pressure_abs_pa=float(conditions["cavitation_pressure_abs_pa"]),
        dynamic_viscosity_pa_s=float(conditions["dynamic_viscosity_pa_s"]),
        density_kg_m3=float(conditions["density_kg_m3"]),
        max_revolutions=max_revolutions,
    )


def solve_case(
    model: str, inputs: film.Inputs
) -> tuple[film.Grid, object, dict[str, object]]:
    if model == "reynolds":
        grid, state = film.solve_reynolds(inputs)
        pressure = state.pressure_above_cavitation_pa
        numerical = {
            "accepted": bool(
                pressure.min(initial=0) >= -1e-6
                and state.complementarity_slack_m3.min(initial=0) >= -1e-18
            ),
            "active_set_iterations": state.active_set_iterations,
            "ruptured_area_fraction": float(
                np.sum(grid.area[state.rupture_mask]) / np.sum(grid.area)
            ),
            "minimum_slack_m3": float(state.complementarity_slack_m3.min(initial=0)),
            "mass_conserving_cavitation": False,
        }
    elif model == "jfo":
        grid, state = film.solve(inputs)
        pressure = state.pressure_above_cavitation_pa
        flows = film.flow_metrics(inputs, grid, state)
        numerical = {
            "accepted": bool(
                state.converged
                and pressure.min(initial=0) >= -1e-6
                and flows["relative_imbalance"] <= 0.005
            ),
            "converged": state.converged,
            "steps": state.steps,
            "revolutions": state.revolutions,
            "pressure_error": state.pressure_error,
            "fill_error": state.fill_error,
            "active_set_iterations_max": state.active_set_iterations_max,
            "minimum_fill_fraction": float(state.fill_fraction.min()),
            "ruptured_area_fraction": float(
                np.sum(grid.area[state.fill_fraction < 1 - 1e-8]) / np.sum(grid.area)
            ),
            "relative_mass_imbalance": flows["relative_imbalance"],
            "feed_in_m3_s": flows["feed_in_m3_s"],
            "mass_conserving_cavitation": True,
        }
    else:
        raise ValueError(f"unknown model: {model}")

    pressure_gauge = (
        inputs.cavitation_pressure_abs_pa + pressure - inputs.ambient_pressure_pa
    )
    force = film.pressure_force(inputs, grid, pressure)
    metrics = {
        "model": model,
        "eccentricity_ratio": inputs.eccentricity_m / inputs.radial_clearance_m,
        "eccentricity_angle_deg": inputs.eccentricity_angle_deg,
        "maximum_gauge_pressure_pa": float(pressure_gauge.max()),
        "minimum_gauge_pressure_pa": float(pressure_gauge.min()),
        "pressure_force_n": force.tolist(),
        "radial_pressure_force_n": float(np.linalg.norm(force[:2])),
        "axial_pressure_force_n": float(force[2]),
        "feed_cells": int(np.count_nonzero(grid.feed)),
        "feed_area_m2": float(np.sum(grid.area[grid.feed])),
        "numerical": numerical,
    }
    return grid, state, metrics


def fixed_case(
    spec: tuple[str, float, dict[str, object], int, int, int],
) -> dict[str, object]:
    model, epsilon, conditions, n_theta, n_axial, max_revolutions = spec
    inputs = make_inputs(conditions, epsilon, -90.0, n_theta, n_axial, max_revolutions)
    _, _, result = solve_case(model, inputs)
    reference = float(conditions["paper_reference_pressure_pa"])
    result["normalized_maximum_pressure"] = (
        float(result["maximum_gauge_pressure_pa"]) / reference
    )
    return result


def resource_record(start: float, jobs: int) -> dict[str, object]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "host": platform.node(),
        "logical_cpu_count": os.cpu_count(),
        "worker_processes": jobs,
        "wall_seconds": time.perf_counter() - start,
        "parent_cpu_seconds": own.ru_utime + own.ru_stime,
        "child_cpu_seconds": children.ru_utime + children.ru_stime,
        "parent_peak_rss_kib": own.ru_maxrss,
        "largest_child_peak_rss_kib": children.ru_maxrss,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def periodic_values(
    theta_deg: np.ndarray, values: np.ndarray, sample_deg: np.ndarray
) -> np.ndarray:
    wrapped = np.mod(sample_deg, 360.0)
    extended_theta = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    extended_values = np.tile(values, 3)
    return np.interp(wrapped, extended_theta, extended_values)


def best_phase_shift(
    theta_deg: np.ndarray,
    pressure_kpa: np.ndarray,
    reference_points: list[dict[str, object]],
) -> tuple[float, float]:
    angles = np.array([float(point["angle_deg"]) for point in reference_points])
    expected = np.array([float(point["pressure_kpa"]) for point in reference_points])
    shifts = np.linspace(0, 360, 1440, endpoint=False)
    errors = np.array(
        [
            np.sqrt(
                np.mean(
                    (
                        periodic_values(theta_deg, pressure_kpa, angles + shift)
                        - expected
                    )
                    ** 2
                )
            )
            for shift in shifts
        ]
    )
    index = int(np.argmin(errors))
    return float(shifts[index]), float(errors[index])


def write_figure6(
    stage: Path,
    study: dict[str, object],
    results: list[dict[str, object]],
    jobs: int,
    started: float,
) -> tuple[str, dict[str, object]]:
    figure6 = study["figure6"]
    conditions = study["conditions"]
    interpretation = study["interpretation"]
    assert isinstance(figure6, dict)
    assert isinstance(conditions, dict)
    assert isinstance(interpretation, dict)
    paper = {
        float(point["eccentricity_ratio"]): point
        for point in figure6["graph_read_points"]
    }
    pressure_limit = float(interpretation["pressure_limit_pa"])

    with (stage / "figure6.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "model",
                "eccentricity_ratio",
                "maximum_gauge_pressure_pa",
                "normalized_maximum_pressure",
                "radial_pressure_force_n",
                "axial_pressure_force_n",
                "paper_fea_graph_read",
                "paper_fluent_graph_read",
                "below_5mpa_design_gate",
                "numerical_accepted",
            )
        )
        for result in results:
            point = paper.get(float(result["eccentricity_ratio"]), {})
            numerical = result["numerical"]
            assert isinstance(numerical, dict)
            writer.writerow(
                (
                    result["model"],
                    result["eccentricity_ratio"],
                    result["maximum_gauge_pressure_pa"],
                    result["normalized_maximum_pressure"],
                    result["radial_pressure_force_n"],
                    result["axial_pressure_force_n"],
                    point.get("paper_fea"),
                    point.get("paper_fluent"),
                    float(result["maximum_gauge_pressure_pa"]) <= pressure_limit,
                    numerical["accepted"],
                )
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    points = list(figure6["graph_read_points"])
    paper_e = [float(point["eccentricity_ratio"]) for point in points]
    ax.plot(
        paper_e,
        [point["paper_fea"] for point in points],
        "ko-",
        label="paper FEA (graph read)",
    )
    ax.plot(
        paper_e,
        [point["paper_fluent"] for point in points],
        "k^--",
        label="paper Fluent (graph read)",
    )
    for model, marker in (("reynolds", "s"), ("jfo", "o")):
        rows = [item for item in results if item["model"] == model]
        ax.plot(
            [item["eccentricity_ratio"] for item in rows],
            [item["normalized_maximum_pressure"] for item in rows],
            marker=marker,
            label=f"local {model.upper()}",
        )
    ax.axhline(
        pressure_limit / float(conditions["paper_reference_pressure_pa"]),
        color="tab:red",
        linestyle=":",
        label="5 MPa provisional design ceiling",
    )
    ax.set(
        xlabel="eccentricity ratio",
        ylabel="maximum gauge pressure / 0.5 MPa",
        title="Figure 6 interpretation: prescribed eccentricity at 2000 rpm",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(stage / "figure6.png", dpi=180)
    plt.close(fig)

    numerical_pass = all(bool(result["numerical"]["accepted"]) for result in results)
    design_pass = all(
        float(result["maximum_gauge_pressure_pa"]) <= pressure_limit
        for result in results
    )
    status = (
        "NUMERICAL_PASS_DESIGN_PASS"
        if numerical_pass and design_pass
        else "NUMERICAL_PASS_DESIGN_PRESSURE_EXCEEDED"
        if numerical_pass
        else "NUMERICAL_FAIL"
    )
    summary = {
        "status": status,
        "control": figure6["control"],
        "conditions": conditions,
        "interpretation": interpretation,
        "results": results,
        "resources": resource_record(started, jobs),
    }
    (stage / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return status, summary


def equilibrium_case(
    spec: tuple[str, dict[str, object], int, int, int, float, float, float],
) -> dict[str, object]:
    (
        model,
        conditions,
        n_theta,
        n_axial,
        max_revolutions,
        target_load,
        initial_e,
        initial_angle,
    ) = spec
    target = np.array((0.0, target_load))
    cache: dict[tuple[float, float], dict[str, object]] = {}

    def evaluate(values: np.ndarray) -> dict[str, object]:
        epsilon = float(values[0])
        angle = float(values[1])
        key = (round(epsilon, 10), round(angle, 7))
        if key not in cache:
            inputs = make_inputs(
                conditions,
                epsilon,
                angle,
                n_theta,
                n_axial,
                max_revolutions,
            )
            _, _, cache[key] = solve_case(model, inputs)
        return cache[key]

    def residual(values: np.ndarray) -> np.ndarray:
        result = evaluate(values)
        numerical = result["numerical"]
        assert isinstance(numerical, dict)
        if not numerical["accepted"]:
            raise RuntimeError(
                f"{model} equilibrium evaluation failed its numerical gate"
            )
        force = np.asarray(result["pressure_force_n"], dtype=float)
        current = (force[:2] - target) / target_load
        print(
            f"{model}: epsilon={values[0]:.7f} angle={values[1]:.4f} "
            f"Fx={force[0]:.3f} Fy={force[1]:.3f} residual={np.linalg.norm(current):.3e}",
            flush=True,
        )
        return current

    fit = least_squares(
        residual,
        x0=np.array((initial_e, initial_angle)),
        bounds=(np.array((1e-5, -270.0)), np.array((0.95, 90.0))),
        x_scale=np.array((0.02, 90.0)),
        diff_step=np.array((0.02, 0.001)),
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        max_nfev=30,
    )
    epsilon = float(fit.x[0])
    angle = float((fit.x[1] + 180) % 360 - 180)
    inputs = make_inputs(conditions, epsilon, angle, n_theta, n_axial, max_revolutions)
    grid, state, result = solve_case(model, inputs)
    force = np.asarray(result["pressure_force_n"], dtype=float)
    load_residual = float(np.linalg.norm(force[:2] - target))
    pressure = (
        inputs.cavitation_pressure_abs_pa
        + state.pressure_above_cavitation_pa
        - inputs.ambient_pressure_pa
    )
    midplane = int(np.argmin(np.abs(grid.z - inputs.length_m / 2)))
    theta_deg = np.degrees(grid.theta)
    pressure_kpa = pressure[midplane] / 1000
    result.update(
        {
            "optimizer_success": bool(fit.success),
            "optimizer_message": fit.message,
            "optimizer_evaluations": fit.nfev,
            "target_pressure_force_xy_n": target.tolist(),
            "load_vector_residual_n": load_residual,
            "equilibrium_accepted": bool(fit.success and load_residual <= 5.0),
            "midplane_z_m": float(grid.z[midplane]),
            "profile_theta_deg": theta_deg.tolist(),
            "profile_pressure_kpa": pressure_kpa.tolist(),
        }
    )
    return result


def write_figure8(
    stage: Path,
    study: dict[str, object],
    results: list[dict[str, object]],
    jobs: int,
    started: float,
) -> tuple[str, dict[str, object]]:
    figure8 = study["figure8"]
    interpretation = study["interpretation"]
    assert isinstance(figure8, dict)
    assert isinstance(interpretation, dict)
    reference_curve = list(figure8["paper_fea_curve_graph_read"])
    taps = list(figure8["experimental_taps_graph_read"])
    pressure_limit = float(interpretation["pressure_limit_pa"])

    for result in results:
        theta = np.asarray(result["profile_theta_deg"], dtype=float)
        pressure = np.asarray(result["profile_pressure_kpa"], dtype=float)
        shift, rmse = best_phase_shift(theta, pressure, reference_curve)
        result["phase_alignment"] = {
            "model_theta_equals_paper_theta_plus_deg": shift,
            "rmse_against_paper_fea_graph_read_kpa": rmse,
            "predictive_validation": False,
        }
        with (stage / f"{result['model']}_profile.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("theta_deg", "pressure_gauge_kpa"))
            writer.writerows(zip(theta, pressure, strict=True))

    by_model = {str(result["model"]): result for result in results}
    with (stage / "figure8_taps.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "port",
                "paper_angle_deg",
                "experimental_graph_read_kpa",
                "reynolds_phase_aligned_kpa",
                "jfo_phase_aligned_kpa",
            )
        )
        for tap in taps:
            predictions = []
            for model in MODELS:
                result = by_model[model]
                alignment = result["phase_alignment"]
                assert isinstance(alignment, dict)
                predictions.append(
                    float(
                        periodic_values(
                            np.asarray(result["profile_theta_deg"]),
                            np.asarray(result["profile_pressure_kpa"]),
                            np.array(
                                [
                                    float(tap["angle_deg"])
                                    + float(
                                        alignment[
                                            "model_theta_equals_paper_theta_plus_deg"
                                        ]
                                    )
                                ]
                            ),
                        )[0]
                    )
                )
            writer.writerow(
                (
                    tap["port"],
                    tap["angle_deg"],
                    tap["pressure_kpa"],
                    *predictions,
                )
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    reference_angles = np.array(
        [float(point["angle_deg"]) for point in reference_curve]
    )
    ax.plot(
        reference_angles,
        [point["pressure_kpa"] for point in reference_curve],
        "k.--",
        label="paper FEA (graph read)",
    )
    ax.scatter(
        [tap["angle_deg"] for tap in taps],
        [tap["pressure_kpa"] for tap in taps],
        marker="x",
        s=60,
        color="black",
        label="paper experiment (graph read)",
    )
    plot_angles = np.linspace(0, 360, 721)
    for model in MODELS:
        result = by_model[model]
        alignment = result["phase_alignment"]
        assert isinstance(alignment, dict)
        values = periodic_values(
            np.asarray(result["profile_theta_deg"]),
            np.asarray(result["profile_pressure_kpa"]),
            plot_angles + float(alignment["model_theta_equals_paper_theta_plus_deg"]),
        )
        ax.plot(plot_angles, values, label=f"local {model.upper()} (phase aligned)")
    ax.set(
        xlabel="paper circumferential angle (degree)",
        ylabel="mid-plane gauge pressure (kPa)",
        title="Figure 8 interpretation: 1000 N radial-load equilibrium at 2000 rpm",
        xlim=(0, 360),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(stage / "figure8.png", dpi=180)
    plt.close(fig)

    numerical_pass = all(bool(result["numerical"]["accepted"]) for result in results)
    equilibrium_pass = all(bool(result["equilibrium_accepted"]) for result in results)
    design_pass = all(
        float(result["maximum_gauge_pressure_pa"]) <= pressure_limit
        for result in results
    )
    status = (
        "NUMERICAL_EQUILIBRIUM_PASS_DESIGN_PASS"
        if numerical_pass and equilibrium_pass and design_pass
        else "NUMERICAL_EQUILIBRIUM_PASS_DESIGN_PRESSURE_EXCEEDED"
        if numerical_pass and equilibrium_pass
        else "NUMERICAL_OR_EQUILIBRIUM_FAIL"
    )
    summary = {
        "status": status,
        "control": figure8["control"],
        "conditions": study["conditions"],
        "interpretation": interpretation,
        "angle_reference": figure8["angle_reference"],
        "target_radial_load_n": figure8["target_radial_load_n"],
        "reported_axial_load_n": figure8["reported_axial_load_n"],
        "results": results,
        "resources": resource_record(started, jobs),
    }
    (stage / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return status, summary


def run_figure6(args: argparse.Namespace, values: Sequence[str]) -> int:
    study = load_study(args.input)
    conditions = study["conditions"]
    assert isinstance(conditions, dict)
    if not args.eccentricity_ratios or any(
        not 0 < value < 1 for value in args.eccentricity_ratios
    ):
        raise ValueError("eccentricity ratios must be inside (0, 1)")
    specs = [
        (model, epsilon, conditions, args.n_theta, args.n_axial, args.max_revolutions)
        for model in MODELS
        for epsilon in args.eccentricity_ratios
    ]
    jobs = min(args.jobs, len(specs), os.cpu_count() or 1)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    target = args.outdir.resolve()
    stage = make_staging_directory(target)
    started = time.perf_counter()
    try:
        if jobs == 1:
            results = [fixed_case(spec) for spec in specs]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                results = list(executor.map(fixed_case, specs))
        status, _ = write_figure6(stage, study, results, jobs, started)
        publish_generation(
            stage,
            target,
            stage="study",
            operation="paper-reproduction-figure6",
            status=status,
            argv=values,
            resolved_inputs={
                "input": args.input,
                "eccentricity_ratios": args.eccentricity_ratios,
                "n_theta": args.n_theta,
                "n_axial": args.n_axial,
                "max_revolutions": args.max_revolutions,
                "jobs": jobs,
                "conditions": conditions,
            },
            input_units={
                "rpm": "rev/min",
                "pressure": "Pa",
                "length": "m",
                "dynamic_viscosity": "Pa s",
            },
            producer_files=(Path(__file__), Path(film.__file__)),
            upstream_artifacts=(args.input,),
            tool_versions={"numpy": np.__version__, "scipy": scipy.__version__},
            acceptance_status=status,
            repository=REPO_ROOT,
        )
        stage = None
        return 0 if status != "NUMERICAL_FAIL" else 2
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def run_figure8(args: argparse.Namespace, values: Sequence[str]) -> int:
    study = load_study(args.input)
    conditions = study["conditions"]
    figure8 = study["figure8"]
    assert isinstance(conditions, dict)
    assert isinstance(figure8, dict)
    target_load = float(figure8["target_radial_load_n"])
    specs = [
        (
            model,
            conditions,
            args.n_theta,
            args.n_axial,
            args.max_revolutions,
            target_load,
            args.initial_eccentricity_ratio,
            -90.0,
        )
        for model in MODELS
    ]
    jobs = min(args.jobs, len(specs), os.cpu_count() or 1)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    target = args.outdir.resolve()
    stage = make_staging_directory(target)
    started = time.perf_counter()
    try:
        if jobs == 1:
            results = [equilibrium_case(spec) for spec in specs]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                results = list(executor.map(equilibrium_case, specs))
        status, _ = write_figure8(stage, study, results, jobs, started)
        publish_generation(
            stage,
            target,
            stage="study",
            operation="paper-reproduction-figure8",
            status=status,
            argv=values,
            resolved_inputs={
                "input": args.input,
                "target_radial_load_n": target_load,
                "n_theta": args.n_theta,
                "n_axial": args.n_axial,
                "max_revolutions": args.max_revolutions,
                "jobs": jobs,
                "initial_eccentricity_ratio": args.initial_eccentricity_ratio,
                "conditions": conditions,
            },
            input_units={
                "rpm": "rev/min",
                "pressure": "Pa",
                "load": "N",
                "length": "m",
                "dynamic_viscosity": "Pa s",
            },
            producer_files=(Path(__file__), Path(film.__file__)),
            upstream_artifacts=(args.input,),
            tool_versions={"numpy": np.__version__, "scipy": scipy.__version__},
            acceptance_status=status,
            repository=REPO_ROOT,
        )
        stage = None
        return 0 if status != "NUMERICAL_OR_EQUILIBRIUM_FAIL" else 2
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="figure", required=True)

    figure6 = subparsers.add_parser("figure6", help="fixed-eccentricity sweep")
    figure6.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    figure6.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT / "figure6")
    figure6.add_argument("--n-theta", type=int, default=256)
    figure6.add_argument("--n-axial", type=int, default=80)
    figure6.add_argument("--max-revolutions", type=int, default=12)
    figure6.add_argument("--jobs", type=int, default=1)
    figure6.add_argument(
        "--eccentricity-ratios",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    figure8 = subparsers.add_parser("figure8", help="load-equilibrium comparison")
    figure8.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    figure8.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT / "figure8")
    figure8.add_argument("--n-theta", type=int, default=256)
    figure8.add_argument("--n-axial", type=int, default=80)
    figure8.add_argument("--max-revolutions", type=int, default=12)
    figure8.add_argument("--jobs", type=int, default=1)
    figure8.add_argument("--initial-eccentricity-ratio", type=float, default=0.02)

    args = parser.parse_args(values)
    try:
        return (
            run_figure6(args, values)
            if args.figure == "figure6"
            else run_figure8(args, values)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
