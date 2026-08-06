#!/usr/bin/env python3
"""Run the load-controlled Reynolds/JFO matrix behind paper Sections 4.1--4.8."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve

from bearing_cfd.artifacts import make_staging_directory, publish_generation
from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo as film
from studies.conical_journal.paper_reproduction.run import (
    DEFAULT_INPUT,
    MODELS,
    equilibrium_case,
    load_study,
    make_inputs,
    resource_record,
    solve_case,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("out/conical_journal/studies/paper-reproduction/section4")
DEFAULT_ANGLES = (5.0, 10.0, 20.0, 30.0)
DEFAULT_LOAD_RATIOS = tuple(value / 10 for value in range(1, 10))
DEFAULT_GRIDS = ((448, 140), (512, 160), (704, 220))


def write_partial(path: Path, kind: str, values: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {"kind": kind, "completed_count": len(values), "results": values},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def model_conditions(
    conditions: dict[str, object], models: Sequence[str]
) -> dict[str, dict[str, object]]:
    resolved: dict[str, dict[str, object]] = {}
    for model in models:
        values = dict(conditions)
        if model == "reynolds":
            values["feed_diameter_m"] = 0.0
            values["feed_gauge_pressure_pa"] = 0.0
        elif float(values["feed_diameter_m"]) <= 0:
            raise ValueError("JFO Section 4 sensitivity requires an explicit feed")
        resolved[model] = values
    return resolved


def parse_grid(value: str) -> tuple[int, int]:
    try:
        n_theta, n_axial = (int(item) for item in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "grid must have the form NTHETAxNAXIAL"
        ) from error
    if n_theta < 16 or n_axial < 4:
        raise argparse.ArgumentTypeError("grid is below the solver minimum")
    return n_theta, n_axial


def compact_equilibrium(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"profile_theta_deg", "profile_pressure_kpa"}
    }


def seed_chain(
    spec: tuple[
        str,
        float,
        tuple[float, ...],
        dict[str, object],
        int,
        int,
        int,
    ],
) -> dict[str, object]:
    model, angle, loads, base_conditions, n_theta, n_axial, max_revolutions = spec
    conditions = dict(base_conditions)
    conditions["semicone_angle_deg"] = angle
    pressure_scale = float(conditions["paper_reference_pressure_pa"])
    radius = float(conditions["mean_radius_m"])
    load_scale = pressure_scale * radius**2
    epsilon = 0.02
    attitude = -90.0
    results: list[dict[str, object]] = []
    try:
        for load_ratio in loads:
            result = equilibrium_case(
                (
                    model,
                    conditions,
                    n_theta,
                    n_axial,
                    max_revolutions,
                    load_ratio * load_scale,
                    epsilon,
                    attitude,
                )
            )
            if not result["equilibrium_accepted"]:
                raise RuntimeError(f"equilibrium rejected at load ratio {load_ratio:g}")
            if (
                float(result["load_vector_residual_n"])
                > 0.005 * load_ratio * load_scale
            ):
                raise RuntimeError(
                    f"equilibrium load residual exceeded 0.5% at load ratio {load_ratio:g}"
                )
            epsilon = float(result["eccentricity_ratio"])
            attitude = float(result["eccentricity_angle_deg"])
            results.append(
                {
                    "load_ratio": load_ratio,
                    "eccentricity_ratio": epsilon,
                    "eccentricity_angle_deg": attitude,
                    "equilibrium": compact_equilibrium(result),
                }
            )
        return {
            "status": "PASS",
            "model": model,
            "semicone_angle_deg": angle,
            "grid": [n_theta, n_axial],
            "results": results,
        }
    except Exception as error:  # Preserve the rest of an unattended matrix.
        return {
            "status": "FAIL",
            "model": model,
            "semicone_angle_deg": angle,
            "grid": [n_theta, n_axial],
            "results": results,
            "error": f"{type(error).__name__}: {error}",
        }


def inputs_at_xy(inputs: film.Inputs, x_m: float, y_m: float) -> film.Inputs:
    eccentricity = math.hypot(x_m, y_m)
    if eccentricity >= 0.95 * inputs.radial_clearance_m:
        raise RuntimeError("perturbation left the accepted eccentricity domain")
    angle = math.degrees(math.atan2(y_m, x_m)) if eccentricity else 0.0
    return replace(
        inputs,
        eccentricity_m=eccentricity,
        eccentricity_angle_deg=angle,
    )


def accepted_force(
    model: str,
    inputs: film.Inputs,
    initial_fill: np.ndarray | None,
) -> tuple[film.Grid, object, np.ndarray]:
    if model == "reynolds":
        grid, state = film.solve_reynolds(inputs)
        accepted = bool(
            state.pressure_above_cavitation_pa.min(initial=0) >= -1e-6
            and state.complementarity_slack_m3.min(initial=0) >= -1e-18
        )
    else:
        grid, state = film.solve(inputs, initial_fill)
        flows = film.flow_metrics(inputs, grid, state)
        accepted = bool(
            state.converged
            and state.pressure_above_cavitation_pa.min(initial=0) >= -1e-6
            and flows["relative_imbalance"] <= 0.005
        )
    if not accepted:
        raise RuntimeError(f"{model} perturbation failed its numerical gate")
    return (
        grid,
        state,
        film.pressure_force(inputs, grid, state.pressure_above_cavitation_pa),
    )


def stiffness_matrix(
    model: str,
    inputs: film.Inputs,
    base_state: object,
    perturbation_ratio: float,
) -> np.ndarray:
    angle = math.radians(inputs.eccentricity_angle_deg)
    center = np.array(
        [
            inputs.eccentricity_m * math.cos(angle),
            inputs.eccentricity_m * math.sin(angle),
        ]
    )
    delta = perturbation_ratio * inputs.radial_clearance_m
    matrix = np.empty((2, 2))
    initial_fill = np.asarray(base_state.fill_fraction) if model == "jfo" else None
    for column in range(2):
        offset = np.zeros(2)
        offset[column] = delta
        plus = inputs_at_xy(inputs, *(center + offset))
        minus = inputs_at_xy(inputs, *(center - offset))
        _, _, force_plus = accepted_force(model, plus, initial_fill)
        _, _, force_minus = accepted_force(model, minus, initial_fill)
        matrix[:, column] = -(force_plus[:2] - force_minus[:2]) / (2 * delta)
    return matrix


def damping_matrix(
    model: str,
    inputs: film.Inputs,
    grid: film.Grid,
    base_state: object,
) -> tuple[np.ndarray, float]:
    pressure = np.asarray(base_state.pressure_above_cavitation_pa)
    if model == "reynolds":
        free = ~np.asarray(base_state.rupture_mask, dtype=bool)
    else:
        tolerance = max(float(pressure.max(initial=0)) * 1e-10, 1e-7)
        free = pressure > tolerance
    free &= ~grid.feed
    free_flat = free.ravel()
    if not np.any(free_flat):
        raise RuntimeError("no pressurized cells available for damping linearization")

    operator, _ = film.diffusion_matrix(inputs, grid, 1.0)
    reduced = operator[free_flat][:, free_flat].tocsr()
    angle = math.radians(inputs.eccentricity_angle_deg)
    center = np.array(
        [
            inputs.eccentricity_m * math.cos(angle),
            inputs.eccentricity_m * math.sin(angle),
        ]
    )
    delta = 1e-4 * inputs.radial_clearance_m
    matrix = np.empty((2, 2))
    for column in range(2):
        offset = np.zeros(2)
        offset[column] = delta
        plus = film.make_grid(inputs_at_xy(inputs, *(center + offset)))
        minus = film.make_grid(inputs_at_xy(inputs, *(center - offset)))
        volume_derivative = (
            plus.area * plus.film_thickness - minus.area * minus.film_thickness
        ) / (2 * delta)
        pressure_derivative = np.zeros_like(pressure)
        pressure_derivative.ravel()[free_flat] = spsolve(
            reduced, -volume_derivative.ravel()[free_flat]
        )
        matrix[:, column] = -film.pressure_force(inputs, grid, pressure_derivative)[:2]
    return matrix, float(np.sum(grid.area[free]) / np.sum(grid.area))


def critical_mass(stiffness: np.ndarray, damping: np.ndarray) -> dict[str, object]:
    trace_c = float(np.trace(damping))
    trace_k = float(np.trace(stiffness))
    det_c = float(np.linalg.det(damping))
    det_k = float(np.linalg.det(stiffness))
    mixed = float(
        damping[0, 0] * stiffness[1, 1]
        + damping[1, 1] * stiffness[0, 0]
        - damping[0, 1] * stiffness[1, 0]
        - damping[1, 0] * stiffness[0, 1]
    )
    denominator = mixed**2 + trace_c**2 * det_k - trace_c * trace_k * mixed
    terms = {
        "trace_c": trace_c,
        "trace_k": trace_k,
        "det_c": det_c,
        "det_k": det_k,
        "mixed_ck": mixed,
        "denominator": denominator,
    }
    if min(trace_c, det_c, det_k, mixed, denominator) <= 0:
        return {"accepted": False, "critical_mass_kg": None, "routh_terms": terms}
    mass = trace_c * det_c * mixed / denominator
    return {
        "accepted": bool(math.isfinite(mass) and mass > 0),
        "critical_mass_kg": mass if math.isfinite(mass) and mass > 0 else None,
        "routh_terms": terms,
    }


def normalized_dynamics(
    inputs: film.Inputs,
    conditions: dict[str, object],
    stiffness: np.ndarray,
    damping: np.ndarray,
    axial_force_n: float,
    target_radial_load_n: float,
) -> dict[str, object]:
    pressure_scale = float(conditions["paper_reference_pressure_pa"])
    radius = inputs.mean_radius_m
    clearance = inputs.radial_clearance_m
    viscosity = inputs.dynamic_viscosity_pa_s
    load_scale = pressure_scale * radius**2
    stiffness_scale = pressure_scale * radius**2 / clearance
    damping_scale = viscosity * radius**4 / clearance**3
    mass_scale = viscosity**2 * radius**6 / (clearance**5 * pressure_scale)
    speed_scale = clearance**2 * pressure_scale / (viscosity * radius**2)
    result = critical_mass(stiffness, damping)
    result.update(
        {
            "stiffness_n_m": stiffness.tolist(),
            "stiffness_normalized": (stiffness / stiffness_scale).tolist(),
            "damping_n_s_m": damping.tolist(),
            "damping_normalized": (damping / damping_scale).tolist(),
            "normalization": {
                "load_scale_n": load_scale,
                "stiffness_scale_n_m": stiffness_scale,
                "damping_scale_n_s_m": damping_scale,
                "mass_scale_kg": mass_scale,
                "angular_speed_scale_rad_s": speed_scale,
            },
        }
    )
    mass = result["critical_mass_kg"]
    if mass is None:
        result["threshold"] = None
        return result
    mass_bar = float(mass) / mass_scale
    force_bar_full = math.hypot(target_radial_load_n, axial_force_n) / load_scale
    force_bar_radial = target_radial_load_n / load_scale
    omega_bar_full = math.sqrt(mass_bar / force_bar_full)
    omega_bar_radial = math.sqrt(mass_bar / force_bar_radial)
    result["critical_mass_normalized"] = mass_bar
    result["threshold"] = {
        "paper_resultant_force_interpretation": {
            "force_normalized": force_bar_full,
            "angular_speed_normalized": omega_bar_full,
            "angular_speed_rad_s": omega_bar_full * speed_scale,
            "speed_rpm": omega_bar_full * speed_scale * 60 / (2 * math.pi),
        },
        "radial_force_sensitivity": {
            "force_normalized": force_bar_radial,
            "angular_speed_normalized": omega_bar_radial,
            "angular_speed_rad_s": omega_bar_radial * speed_scale,
            "speed_rpm": omega_bar_radial * speed_scale * 60 / (2 * math.pi),
        },
        "operating_angular_speed_normalized": (
            inputs.rpm * 2 * math.pi / 60 / speed_scale
        ),
    }
    return result


def section4_case(spec: tuple[object, ...]) -> dict[str, object]:
    (
        model,
        angle,
        load_ratio,
        conditions_in,
        n_theta,
        n_axial,
        max_revolutions,
        initial_e,
        initial_angle,
        perturbations,
        main_grid,
    ) = spec
    conditions = dict(conditions_in)
    conditions["semicone_angle_deg"] = float(angle)
    pressure_scale = float(conditions["paper_reference_pressure_pa"])
    load_scale = pressure_scale * float(conditions["mean_radius_m"]) ** 2
    target_load = float(load_ratio) * load_scale
    identity = {
        "model": model,
        "semicone_angle_deg": angle,
        "load_ratio": load_ratio,
        "grid": [n_theta, n_axial],
        "grid_role": "main" if (n_theta, n_axial) == main_grid else "sensitivity",
    }
    try:
        equilibrium = equilibrium_case(
            (
                str(model),
                conditions,
                int(n_theta),
                int(n_axial),
                int(max_revolutions),
                target_load,
                float(initial_e),
                float(initial_angle),
            )
        )
        if not equilibrium["equilibrium_accepted"]:
            raise RuntimeError("load equilibrium failed")
        if float(equilibrium["load_vector_residual_n"]) > 0.005 * target_load:
            raise RuntimeError("load equilibrium residual exceeded 0.5%")
        inputs = make_inputs(
            conditions,
            float(equilibrium["eccentricity_ratio"]),
            float(equilibrium["eccentricity_angle_deg"]),
            int(n_theta),
            int(n_axial),
            int(max_revolutions),
        )
        grid, base_state, base = solve_case(str(model), inputs)
        stiffness = {
            str(value): stiffness_matrix(str(model), inputs, base_state, float(value))
            for value in perturbations
        }
        primary_ratio = float(perturbations[0])
        primary_stiffness = stiffness[str(primary_ratio)]
        damping, pressurized_area = damping_matrix(str(model), inputs, grid, base_state)
        force = np.asarray(base["pressure_force_n"], dtype=float)
        dynamics = normalized_dynamics(
            inputs,
            conditions,
            primary_stiffness,
            damping,
            abs(float(force[2])),
            target_load,
        )
        sensitivity = None
        if len(perturbations) > 1:
            comparison = stiffness[str(float(perturbations[1]))]
            sensitivity = float(
                np.linalg.norm(primary_stiffness - comparison)
                / max(np.linalg.norm(primary_stiffness), 1e-30)
            )
        numerical = base["numerical"]
        assert isinstance(numerical, dict)
        result = identity | {
            "status": "PASS",
            "target_radial_load_n": target_load,
            "equilibrium": compact_equilibrium(equilibrium),
            "sections": {
                "4.1_maximum_pressure": {
                    "gauge_pa": base["maximum_gauge_pressure_pa"],
                    "normalized": float(base["maximum_gauge_pressure_pa"])
                    / pressure_scale,
                    "below_5mpa_design_gate": float(base["maximum_gauge_pressure_pa"])
                    <= 5_000_000,
                },
                "4.2_axial_load": {
                    "signed_n": float(force[2]),
                    "magnitude_n": abs(float(force[2])),
                    "normalized_magnitude": abs(float(force[2])) / load_scale,
                },
                "4.3_minimum_film_thickness": {
                    "m": float(grid.film_thickness.min()),
                    "normalized": float(grid.film_thickness.min())
                    / inputs.radial_clearance_m,
                },
                "4.4_4.5_stiffness": {
                    "primary_perturbation_over_clearance": primary_ratio,
                    "matrices_n_m": {
                        key: value.tolist() for key, value in stiffness.items()
                    },
                    "two_step_relative_frobenius_difference": sensitivity,
                },
                "4.6_4.7_damping": {
                    "method": "frozen-cavity instantaneous Reynolds linearization",
                    "pressurized_area_fraction": pressurized_area,
                },
                "4.8_stability": dynamics,
            },
            "numerical": numerical,
            "paper_method_parity": False,
        }
        print(
            f"done {model} gamma={float(angle):g} Wbar={float(load_ratio):g} "
            f"grid={n_theta}x{n_axial}",
            flush=True,
        )
        return result
    except Exception as error:  # Preserve all other unattended cases.
        print(
            f"failed {model} gamma={angle} Wbar={load_ratio} grid={n_theta}x{n_axial}: {error}",
            flush=True,
        )
        return identity | {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
        }


def csv_row(result: dict[str, object], primary_ratio: float) -> dict[str, object]:
    row: dict[str, object] = {
        "status": result["status"],
        "model": result["model"],
        "semicone_angle_deg": result["semicone_angle_deg"],
        "load_ratio": result["load_ratio"],
        "grid": "x".join(str(value) for value in result["grid"]),
        "grid_role": result["grid_role"],
    }
    if result["status"] != "PASS":
        row["error"] = result.get("error")
        return row
    sections = result["sections"]
    equilibrium = result["equilibrium"]
    assert isinstance(sections, dict) and isinstance(equilibrium, dict)
    pressure = sections["4.1_maximum_pressure"]
    axial = sections["4.2_axial_load"]
    thickness = sections["4.3_minimum_film_thickness"]
    stiffness = sections["4.4_4.5_stiffness"]
    stability = sections["4.8_stability"]
    assert all(
        isinstance(value, dict)
        for value in (pressure, axial, thickness, stiffness, stability)
    )
    matrices = stiffness["matrices_n_m"]
    matrix = np.asarray(matrices[str(primary_ratio)])
    damping = np.asarray(stability["damping_n_s_m"])
    threshold = stability["threshold"]
    row.update(
        {
            "eccentricity_ratio": equilibrium["eccentricity_ratio"],
            "eccentricity_angle_deg": equilibrium["eccentricity_angle_deg"],
            "maximum_pressure_normalized": pressure["normalized"],
            "below_5mpa_design_gate": pressure["below_5mpa_design_gate"],
            "axial_load_normalized_magnitude": axial["normalized_magnitude"],
            "minimum_film_thickness_normalized": thickness["normalized"],
            "S11_n_m": matrix[0, 0],
            "S12_n_m": matrix[0, 1],
            "S21_n_m": matrix[1, 0],
            "S22_n_m": matrix[1, 1],
            "C11_n_s_m": damping[0, 0],
            "C12_n_s_m": damping[0, 1],
            "C21_n_s_m": damping[1, 0],
            "C22_n_s_m": damping[1, 1],
            "critical_mass_kg": stability["critical_mass_kg"],
            "threshold_speed_normalized": (
                threshold["paper_resultant_force_interpretation"][
                    "angular_speed_normalized"
                ]
                if isinstance(threshold, dict)
                else None
            ),
            "stiffness_two_step_relative_difference": stiffness[
                "two_step_relative_frobenius_difference"
            ],
        }
    )
    return row


def run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    study = load_study(args.input)
    base_conditions = study["conditions"]
    assert isinstance(base_conditions, dict)
    conditions = dict(base_conditions)
    radius = float(conditions["mean_radius_m"])
    conditions["rpm"] = args.surface_speed_m_s / (2 * math.pi * radius) * 60
    models = tuple(dict.fromkeys(args.models))
    conditions_by_model = model_conditions(conditions, models)
    grids = tuple(args.grids)
    main_grid = args.main_grid
    if main_grid not in grids:
        raise ValueError("main grid must be included in --grids")
    if any(not 0 < value < 1 for value in args.load_ratios):
        raise ValueError("load ratios must lie inside (0, 1)")
    if any(value <= 0 for value in args.perturbation_ratios):
        raise ValueError("perturbation ratios must be positive")
    if args.jobs < 1 or args.seed_jobs < 1:
        raise ValueError("worker counts must be positive")

    target = args.outdir.resolve()
    stage = make_staging_directory(target)
    started = time.perf_counter()
    try:
        seed_specs = [
            (
                model,
                angle,
                tuple(args.load_ratios),
                conditions_by_model[model],
                args.seed_grid[0],
                args.seed_grid[1],
                args.max_revolutions,
            )
            for model in models
            for angle in args.semicone_angles_deg
        ]
        seed_jobs = min(args.seed_jobs, len(seed_specs), os.cpu_count() or 1)
        seeds: list[dict[str, object]] = []
        seed_partial = stage / "seeds.partial.json"
        with ProcessPoolExecutor(max_workers=seed_jobs) as executor:
            futures = [executor.submit(seed_chain, spec) for spec in seed_specs]
            for future in as_completed(futures):
                seeds.append(future.result())
                write_partial(seed_partial, "coarse equilibrium seeds", seeds)
        seeds.sort(key=lambda item: (str(item["model"]), item["semicone_angle_deg"]))
        (stage / "seeds.json").write_text(
            json.dumps(seeds, indent=2) + "\n", encoding="utf-8"
        )
        seed_partial.unlink()
        seed_lookup = {
            (
                item["model"],
                float(item["semicone_angle_deg"]),
                float(seed["load_ratio"]),
            ): seed
            for item in seeds
            for seed in item["results"]
        }
        specs = []
        skipped: list[dict[str, object]] = []
        for model in models:
            for angle in args.semicone_angles_deg:
                for load_ratio in args.load_ratios:
                    seed = seed_lookup.get((model, float(angle), float(load_ratio)))
                    for grid in grids:
                        if seed is None:
                            skipped.append(
                                {
                                    "status": "SKIPPED",
                                    "model": model,
                                    "semicone_angle_deg": angle,
                                    "load_ratio": load_ratio,
                                    "grid": list(grid),
                                    "grid_role": "main"
                                    if grid == main_grid
                                    else "sensitivity",
                                    "error": "coarse equilibrium seed unavailable",
                                }
                            )
                            continue
                        specs.append(
                            (
                                model,
                                angle,
                                load_ratio,
                                conditions_by_model[model],
                                grid[0],
                                grid[1],
                                args.max_revolutions,
                                seed["eccentricity_ratio"],
                                seed["eccentricity_angle_deg"],
                                tuple(args.perturbation_ratios),
                                main_grid,
                            )
                        )
        jobs = min(args.jobs, len(specs), os.cpu_count() or 1) if specs else 0
        results: list[dict[str, object]] = []
        result_partial = stage / "results.partial.json"
        if specs:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(section4_case, spec) for spec in specs]
                for future in as_completed(futures):
                    results.append(future.result())
                    write_partial(result_partial, "section 4 cases", results)
            result_partial.unlink()
        results.extend(skipped)
        results.sort(
            key=lambda item: (
                str(item["model"]),
                float(item["semicone_angle_deg"]),
                float(item["load_ratio"]),
                tuple(item["grid"]),
            )
        )
        summary_status = (
            "NUMERICAL_PASS"
            if results and all(item["status"] == "PASS" for item in results)
            else "PARTIAL_OR_FAILED"
        )
        summary = {
            "status": summary_status,
            "paper_sections": ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8"],
            "base_conditions": conditions,
            "conditions_by_model": conditions_by_model,
            "models": models,
            "load_ratios": args.load_ratios,
            "semicone_angles_deg": args.semicone_angles_deg,
            "grids": [list(value) for value in grids],
            "main_grid": list(main_grid),
            "perturbation_ratios": args.perturbation_ratios,
            "method_notes": {
                "equilibrium": "Fx=0 and Fy=target radial load at every point",
                "stiffness": "central differences at fixed journal-center coordinates",
                "damping": "frozen-cavity instantaneous pressure linearization; the paper does not disclose its perturbation amplitude or moving-boundary implementation",
                "grid_comparison": "three-grid numerical sensitivity, not a formal GCI sequence",
                "paper_parity": False,
            },
            "results": results,
            "resources": resource_record(started, jobs),
        }
        summary_path = stage / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        rows = [csv_row(item, float(args.perturbation_ratios[0])) for item in results]
        columns = sorted({key for row in rows for key in row})
        with (stage / "section4.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        publish_generation(
            stage,
            target,
            stage="study",
            operation="paper-reproduction-section4",
            status=summary_status,
            argv=argv,
            resolved_inputs={
                "input": args.input,
                "base_conditions": conditions,
                "conditions_by_model": conditions_by_model,
                "models": models,
                "load_ratios": args.load_ratios,
                "semicone_angles_deg": args.semicone_angles_deg,
                "grids": grids,
                "main_grid": main_grid,
                "seed_grid": args.seed_grid,
                "max_revolutions": args.max_revolutions,
                "perturbation_ratios": args.perturbation_ratios,
                "jobs": jobs,
                "seed_jobs": seed_jobs,
            },
            input_units={
                "surface_speed": "m/s",
                "rpm": "rev/min",
                "pressure": "Pa",
                "load": "N",
                "length": "m",
                "dynamic_viscosity": "Pa s",
            },
            producer_files=(Path(__file__), Path(film.__file__)),
            upstream_artifacts=(args.input,),
            tool_versions={"numpy": np.__version__, "scipy": scipy.__version__},
            acceptance_status=summary_status,
            repository=REPO_ROOT,
        )
        stage = None
        return 0 if summary_status == "NUMERICAL_PASS" else 2
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface-speed-m-s", type=float, default=2.6)
    parser.add_argument(
        "--models", choices=MODELS, nargs="+", default=["reynolds"]
    )
    parser.add_argument(
        "--semicone-angles-deg", type=float, nargs="+", default=list(DEFAULT_ANGLES)
    )
    parser.add_argument(
        "--load-ratios", type=float, nargs="+", default=list(DEFAULT_LOAD_RATIOS)
    )
    parser.add_argument(
        "--grids", type=parse_grid, nargs="+", default=list(DEFAULT_GRIDS)
    )
    parser.add_argument("--main-grid", type=parse_grid, default=(512, 160))
    parser.add_argument("--seed-grid", type=parse_grid, default=(128, 40))
    parser.add_argument(
        "--perturbation-ratios", type=float, nargs="+", default=[0.001, 0.002]
    )
    parser.add_argument("--max-revolutions", type=int, default=24)
    parser.add_argument("--seed-jobs", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=28)
    args = parser.parse_args(values)
    try:
        return run(args, values)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
