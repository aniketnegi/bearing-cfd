#!/usr/bin/env python3
"""Run 3D single-phase diagnostics at equilibrium and prescribed positions."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from bearing_cfd.artifacts import record_generation, sha256_file
from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo as film
from studies.conical_journal.paper_reproduction.run import (
    DEFAULT_INPUT,
    equilibrium_case,
    load_study,
    periodic_values,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = REPO_ROOT / "cases/conical_journal/openfoam/single_phase_oq90"
DEFAULT_OUTPUT = Path("out/conical_journal/studies/paper-reproduction/three-d")
DEFAULT_FIXED_ECCENTRICITY_RATIOS = tuple(value / 100 for value in range(40, 91, 5))
FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
NUMBER_RE = re.compile(FLOAT)


def command_output(args: Sequence[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_command(
    args: Sequence[str],
    log_path: Path,
    *,
    check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("command: " + " ".join(args) + "\n")
        stream.flush()
        completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
    if check and completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")
    return completed.returncode


def seed_case(spec: tuple[object, ...]) -> dict[str, object]:
    label, model, conditions, target_load, initial_e, n_theta, n_axial = spec
    try:
        result = equilibrium_case(
            (
                str(model),
                dict(conditions),
                int(n_theta),
                int(n_axial),
                24,
                float(target_load),
                float(initial_e),
                -90.0,
            )
        )
        return {
            "status": "PASS" if result["equilibrium_accepted"] else "FAIL",
            "operating_point": label,
            "model": model,
            "target_radial_load_n": target_load,
            "result": result,
        }
    except Exception as error:  # Preserve other seeds in an unattended run.
        return {
            "status": "FAIL",
            "operating_point": label,
            "model": model,
            "target_radial_load_n": target_load,
            "error": f"{type(error).__name__}: {error}",
        }


def probe_block(inputs: film.Inputs, count: int = 72) -> tuple[str, list[float]]:
    angle = math.radians(inputs.eccentricity_angle_deg)
    ex = inputs.eccentricity_m * math.cos(angle)
    ey = inputs.eccentricity_m * math.sin(angle)
    theta = (np.arange(count) + 0.5) * 2 * math.pi / count
    radius = inputs.mean_radius_m
    q = ex * np.sin(theta) - ey * np.cos(theta)
    journal_ray = q + np.sqrt(radius**2 - inputs.eccentricity_m**2 + q**2)
    bore = radius + inputs.radial_clearance_m
    midpoint = 0.5 * (journal_ray + bore)
    locations = "\n".join(
        f"        ({r * math.sin(t):.15g} {-r * math.cos(t):.15g} {inputs.length_m / 2:.15g})"
        for r, t in zip(midpoint, theta, strict=True)
    )
    block = f"""

midplaneProbes
{{
    type            probes;
    libs            (\"libsampling.so\");
    writeControl    timeStep;
    writeInterval   1;
    fields          (p);
    probeLocations
    (
{locations}
    );
}}
"""
    return block, np.degrees(theta).tolist()


def last_numeric_row(path: Path) -> list[float]:
    rows = [
        [float(value) for value in NUMBER_RE.findall(line)]
        for line in path.read_text(errors="replace").splitlines()
        if line and not line.startswith("#")
    ]
    rows = [row for row in rows if row]
    if not rows:
        raise RuntimeError(f"no numeric rows in {path}")
    return rows[-1]


def post_file(case: Path, function: str, filename: str) -> Path:
    candidates = list((case / "postProcessing" / function).glob(f"*/{filename}"))
    if not candidates:
        raise RuntimeError(f"missing post-processing output {function}/{filename}")
    return max(candidates, key=lambda path: float(path.parent.name))


def parse_metrics(
    case: Path,
    seed: dict[str, object],
    inputs: film.Inputs,
    probe_theta_deg: list[float],
    converged: bool,
    mesh_ok: bool,
    extended_check_status: int,
) -> dict[str, object]:
    rho = inputs.density_kg_m3
    ambient = inputs.ambient_pressure_pa
    pmax_kinematic = last_numeric_row(post_file(case, "maxP", "volFieldValue.dat"))[1]
    pmin_kinematic = last_numeric_row(post_file(case, "minP", "volFieldValue.dat"))[1]
    umax = last_numeric_row(post_file(case, "maxU", "volFieldValue.dat"))[1]
    feed = last_numeric_row(post_file(case, "feedFlowRate", "surfaceFieldValue.dat"))[1]
    z0 = last_numeric_row(post_file(case, "z0FlowRate", "surfaceFieldValue.dat"))[1]
    zl = last_numeric_row(post_file(case, "zlFlowRate", "surfaceFieldValue.dat"))[1]
    net = last_numeric_row(post_file(case, "netBoundaryFlow", "surfaceFieldValue.dat"))[
        1
    ]
    force_values = last_numeric_row(post_file(case, "journalForces", "forces.dat"))
    if len(force_values) != 13:
        raise RuntimeError(f"expected 13 force columns, got {len(force_values)}")
    pressure_force = np.asarray(force_values[1:4])
    viscous_force = np.asarray(force_values[4:7])
    total_force = pressure_force + viscous_force
    reference_flow = max(abs(feed), abs(z0) + abs(zl), 1e-30)
    imbalance = abs(net) / reference_flow

    probe_values = last_numeric_row(post_file(case, "midplaneProbes", "p"))[1:]
    if len(probe_values) != len(probe_theta_deg):
        raise RuntimeError(
            f"expected {len(probe_theta_deg)} probes, got {len(probe_values)}"
        )
    probe_gauge_pa = rho * np.asarray(probe_values)
    seed_result = seed["result"]
    assert isinstance(seed_result, dict)
    has_seed_profile = {
        "profile_theta_deg",
        "profile_pressure_kpa",
    }.issubset(seed_result)
    seed_profile_pa = (
        1000
        * periodic_values(
            np.asarray(seed_result["profile_theta_deg"], dtype=float),
            np.asarray(seed_result["profile_pressure_kpa"], dtype=float),
            np.asarray(probe_theta_deg),
        )
        if has_seed_profile
        else None
    )
    profile_rmse = (
        float(np.sqrt(np.mean((probe_gauge_pa - seed_profile_pa) ** 2)))
        if seed_profile_pa is not None
        else None
    )
    pmax_gauge = rho * pmax_kinematic
    pmin_gauge = rho * pmin_kinematic
    target_load = (
        float(seed["target_radial_load_n"])
        if seed.get("target_radial_load_n") is not None
        else None
    )
    numerical_pass = bool(converged and mesh_ok and imbalance <= 0.005)
    return {
        "numerical_status": "PASS" if numerical_pass else "FAIL",
        "mesh": {
            "standard_check_mesh_ok": mesh_ok,
            "extended_check_mesh_exit_status": extended_check_status,
        },
        "solver": {"simple_converged": converged},
        "pressure": {
            "minimum_gauge_pa": pmin_gauge,
            "minimum_absolute_pa": ambient + pmin_gauge,
            "maximum_gauge_pa": pmax_gauge,
            "maximum_absolute_pa": ambient + pmax_gauge,
            "below_5mpa_design_gate": pmax_gauge <= 5_000_000,
            "single_phase_tension_present": pmin_gauge < 0,
        },
        "flow": {
            "feed_m3_s": feed,
            "axial_z0_m3_s": z0,
            "axial_zl_m3_s": zl,
            "net_boundary_m3_s": net,
            "relative_imbalance": imbalance,
        },
        "load": {
            "pressure_force_n": pressure_force.tolist(),
            "viscous_force_n": viscous_force.tolist(),
            "total_force_n": total_force.tolist(),
            "pressure_radial_magnitude_n": float(np.linalg.norm(pressure_force[:2])),
            "target_radial_load_n": target_load,
            "radial_magnitude_over_target": (
                float(np.linalg.norm(pressure_force[:2])) / target_load
                if target_load is not None
                else None
            ),
            "pressure_axial_magnitude_n": abs(float(pressure_force[2])),
        },
        "velocity": {"maximum_m_s": umax},
        "midplane_profile": {
            "theta_deg": probe_theta_deg,
            "pressure_gauge_pa": probe_gauge_pa.tolist(),
            "seed_pressure_gauge_pa": (
                seed_profile_pa.tolist() if seed_profile_pa is not None else None
            ),
            "rmse_against_2d_seed_pa": profile_rmse,
        },
        "physical_validation": {
            "status": "DIAGNOSTIC_ONLY_NO_CAVITATION_MODEL",
            "accepted": False,
            "reason": "single-phase Navier-Stokes does not impose Reynolds rupture or mass-conserving JFO cavitation",
        },
    }


def case_key(seed: dict[str, object]) -> str:
    return str(seed.get("case", f"{seed['operating_point']}-{seed['model']}"))


def run_3d_case(spec: tuple[object, ...]) -> dict[str, object]:
    seed, conditions_in, output_text, ranks = spec
    assert isinstance(seed, dict)
    conditions = dict(conditions_in)
    key = case_key(seed)
    output = Path(str(output_text))
    geometry_dir = output / "geometry" / key
    mesh_dir = output / "meshes" / key
    case = output / "cases" / key
    log_dir = output / "logs" / key
    identity = {
        "case": key,
        "operating_point": seed["operating_point"],
        "seed_model": seed["model"],
    }
    try:
        result = seed["result"]
        assert isinstance(result, dict)
        epsilon = float(result["eccentricity_ratio"])
        attitude = float(result["eccentricity_angle_deg"])
        rpm = float(conditions["rpm"])
        inputs = film.Inputs(
            rpm=rpm,
            n_theta=128,
            n_axial=40,
            length_m=float(conditions["length_m"]),
            mean_radius_m=float(conditions["mean_radius_m"]),
            semicone_angle_deg=float(conditions["semicone_angle_deg"]),
            radial_clearance_m=float(conditions["radial_clearance_m"]),
            eccentricity_m=epsilon * float(conditions["radial_clearance_m"]),
            eccentricity_angle_deg=attitude,
            feed_diameter_m=float(conditions["feed_diameter_m"]),
            feed_gauge_pressure_pa=0.0,
            ambient_pressure_pa=float(conditions["ambient_pressure_pa"]),
            cavitation_pressure_abs_pa=float(conditions["cavitation_pressure_abs_pa"]),
            dynamic_viscosity_pa_s=float(conditions["dynamic_viscosity_pa_s"]),
            density_kg_m3=float(conditions["density_kg_m3"]),
        )
        geometry_command = [
            sys.executable,
            "-m",
            "bearing_cfd",
            "conical-journal",
            "geometry",
            "--length",
            f"{inputs.length_m * 1000:.15g}",
            "--mean-radius",
            f"{inputs.mean_radius_m * 1000:.15g}",
            "--semicone-angle-deg",
            f"{inputs.semicone_angle_deg:.15g}",
            "--radial-clearance",
            f"{inputs.radial_clearance_m * 1000:.15g}",
            "--eccentricity-ratio",
            f"{epsilon:.15g}",
            "--eccentricity-angle-deg",
            f"{attitude:.15g}",
            "--hole-diameter",
            f"{inputs.feed_diameter_m * 1000:.15g}",
            "--hole-axial-pos",
            f"{inputs.feed_axial_position_m * 1000:.15g}",
            "--no-preview",
            "--outdir",
            str(geometry_dir),
        ]
        print(f"[{key}] geometry", flush=True)
        run_command(geometry_command, log_dir / "geometry.log")
        mesh_command = [
            sys.executable,
            "-m",
            "bearing_cfd",
            "conical-journal",
            "mesh",
            "body-fitted-inlet",
            "--params",
            str(geometry_dir / "params.json"),
            "--outdir",
            str(mesh_dir),
            "--case-name",
            key,
            "--q",
            "18",
            "--inner-layers",
            "1",
            "--outer-layers",
            "9",
            "--n-theta",
            "576",
            "--n-axial",
            "180",
            "--n-gap",
            "12",
            "--quality-optimized-ogrid",
            "--control-radius-factor",
            "2.87",
            "--control-square-blend",
            "0.36",
            "--central-corner-radius-factor",
            "0.9",
            "--smoothing-iterations",
            "100",
            "--smoothing-damping",
            "0.5",
            "--smoothing-fixed-nodes",
            "background-and-rim",
            "--minimum-fluent-orthogonal-quality",
            "0.9",
            "--openfoam",
            "skip",
            "--ansys",
            "skip",
        ]
        print(f"[{key}] 1,252,800-cell OQ90 mesh", flush=True)
        run_command(mesh_command, log_dir / "mesh.log")
        mesh_path = mesh_dir / "fluent" / f"{key}.msh"
        if not mesh_path.is_file():
            raise RuntimeError(f"mesh generator did not create {mesh_path}")

        shutil.copytree(TEMPLATE / "0", case / "0")
        shutil.copytree(TEMPLATE / "constant", case / "constant")
        shutil.copytree(TEMPLATE / "system", case / "system")
        run_command(
            ["fluentMeshToFoam", "-case", str(case), str(mesh_path)],
            log_dir / "fluentMeshToFoam.log",
        )
        imported = case / "0/polyMesh"
        if imported.is_dir():
            shutil.move(str(imported), str(case / "constant/polyMesh"))
        standard_log = log_dir / "checkMesh.standard.log"
        run_command(["checkMesh", "-case", str(case)], standard_log)
        mesh_ok = "Mesh OK" in standard_log.read_text(errors="replace")
        if not mesh_ok:
            raise RuntimeError("standard checkMesh did not report Mesh OK")
        extended_status = run_command(
            ["checkMesh", "-case", str(case), "-allGeometry", "-allTopology"],
            log_dir / "checkMesh.extended.log",
            check=False,
        )

        angle_rad = math.radians(attitude)
        ex = inputs.eccentricity_m * math.cos(angle_rad)
        ey = inputs.eccentricity_m * math.sin(angle_rad)
        dictionary_changes = [
            (
                case / "0/U",
                "boundaryField/journal_wall/origin",
                f"({ex:.15g} {ey:.15g} 0)",
            ),
            (
                case / "0/U",
                "boundaryField/journal_wall/omega",
                f"{rpm:.15g} [rpm]",
            ),
            (case / "0/p", "boundaryField/pressure_feed/value", "uniform 0"),
            (
                case / "system/functions",
                "journalForces/CofR",
                f"({ex:.15g} {ey:.15g} {inputs.length_m / 2:.15g})",
            ),
            (case / "system/decomposeParDict", "numberOfSubdomains", str(ranks)),
        ]
        for dictionary, entry, value in dictionary_changes:
            run_command(
                [
                    "foamDictionary",
                    "-writePrecision",
                    "15",
                    str(dictionary),
                    "-entry",
                    entry,
                    "-set",
                    value,
                ],
                log_dir
                / f"foamDictionary-{dictionary.name}-{entry.replace('/', '_')}.log",
            )
        block, theta_deg = probe_block(inputs)
        with (case / "system/functions").open("a", encoding="utf-8") as stream:
            stream.write(block)
        run_command(
            ["decomposePar", "-case", str(case), "-force"],
            log_dir / "decomposePar.log",
        )
        solver_log = log_dir / "foamRun.log"
        print(f"[{key}] OpenFOAM with {ranks} MPI ranks", flush=True)
        run_command(
            [
                "mpirun",
                "--bind-to",
                "none",
                "-np",
                str(ranks),
                "foamRun",
                "-parallel",
                "-case",
                str(case),
            ],
            solver_log,
        )
        converged = "SIMPLE solution converged" in solver_log.read_text(
            errors="replace"
        )
        if not converged:
            raise RuntimeError("foamRun ended without SIMPLE solution converged")
        run_command(
            ["reconstructPar", "-case", str(case), "-latestTime"],
            log_dir / "reconstructPar.log",
        )
        metrics = parse_metrics(
            case,
            seed,
            inputs,
            theta_deg,
            converged,
            mesh_ok,
            extended_status,
        )
        dictionaries = [
            case / "0/U",
            case / "0/p",
            case / "constant/physicalProperties",
            case / "constant/momentumTransport",
            case / "system/controlDict",
            case / "system/fvSchemes",
            case / "system/fvSolution",
            case / "system/functions",
            case / "system/decomposeParDict",
        ]
        result_payload = identity | {
            "status": metrics["numerical_status"],
            "seed": seed,
            "openfoam_conditions": {
                "rpm": rpm,
                "eccentricity_ratio": epsilon,
                "eccentricity_angle_deg": attitude,
                "feed_diameter_m": inputs.feed_diameter_m,
                "feed_gauge_pressure_pa": 0.0,
                "cavitation_model": None,
                "turbulence_model": "laminar",
                "cells": 1_252_800,
                "mpi_ranks": ranks,
                "mpi_binding": "none; scheduled across the workstation by the OS",
            },
            "provenance": {
                "mesh_path": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "dictionary_sha256": {
                    str(path.relative_to(case)): sha256_file(path)
                    for path in dictionaries
                },
            },
            "metrics": metrics,
        }
        (case / "case_summary.json").write_text(
            json.dumps(result_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{key}] {result_payload['status']}", flush=True)
        return result_payload
    except Exception as error:  # Preserve other 3D cases and all logs.
        print(f"[{key}] failed: {error}", flush=True)
        return identity | {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "case_directory": str(case),
            "log_directory": str(log_dir),
        }


def run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    required = (
        "fluentMeshToFoam",
        "checkMesh",
        "foamDictionary",
        "decomposePar",
        "foamRun",
        "reconstructPar",
        "mpirun",
    )
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError("missing required commands: " + ", ".join(missing))
    foam_version = (
        f"{os.environ.get('WM_PROJECT', '')}-"
        f"{os.environ.get('WM_PROJECT_VERSION', '')}"
    ).strip("-")
    if foam_version != "OpenFOAM-14":
        raise RuntimeError(f"OpenFOAM Foundation v14 required, found {foam_version}")
    if args.jobs < 1 or args.mpi_ranks < 1:
        raise ValueError("3D jobs and MPI ranks must be positive")
    if args.seed_n_theta < 16 or args.seed_n_axial < 4:
        raise ValueError("2D seed grid is below the solver minimum")
    if any(not 0 < value < 0.95 for value in args.fixed_eccentricity_ratios):
        raise ValueError("fixed 3D eccentricity ratios must lie inside (0, 0.95)")
    cpu_count = os.cpu_count() or 1
    if args.jobs * args.mpi_ranks > cpu_count:
        raise ValueError("3D jobs times MPI ranks exceeds logical CPU count")

    output = args.outdir.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    study = load_study(args.input)
    base = study["conditions"]
    figure8 = study["figure8"]
    assert isinstance(base, dict) and isinstance(figure8, dict)
    reference_pressure = float(base["paper_reference_pressure_pa"])
    radius = float(base["mean_radius_m"])
    section4_rpm = args.surface_speed_m_s / (2 * math.pi * radius) * 60
    operating_points = {
        "figure8": (
            dict(base) | {"rpm": 2000.0, "semicone_angle_deg": 10.0},
            float(figure8["target_radial_load_n"]),
            0.02,
        ),
        "section4-gamma10-wbar05": (
            dict(base) | {"rpm": section4_rpm, "semicone_angle_deg": 10.0},
            0.5 * reference_pressure * radius**2,
            0.1,
        ),
    }
    seed_specs = [
        (
            label,
            model,
            conditions,
            target,
            initial,
            args.seed_n_theta,
            args.seed_n_axial,
        )
        for label, (conditions, target, initial) in operating_points.items()
        for model in ("reynolds", "jfo")
    ]
    print(
        "computing four 2D equilibrium positions used only as 3D geometry seeds",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=4) as executor:
        seeds = list(executor.map(seed_case, seed_specs))

    output.mkdir(parents=True)
    seeds_path = output / "seeds.json"
    seeds_path.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
    valid = [seed for seed in seeds if seed["status"] == "PASS"]
    fixed_conditions = dict(base) | {"rpm": 2000.0, "semicone_angle_deg": 10.0}
    fixed_cases = [
        {
            "status": "PASS",
            "case": f"fixed-eccentricity-e{round(epsilon * 100):03d}",
            "operating_point": "fixed-eccentricity-sensitivity",
            "model": "prescribed",
            "target_radial_load_n": None,
            "result": {
                "eccentricity_ratio": epsilon,
                "eccentricity_angle_deg": -90.0,
            },
        }
        for epsilon in args.fixed_eccentricity_ratios
    ]
    specs = []
    for seed in valid:
        label = str(seed["operating_point"])
        conditions = operating_points[label][0]
        specs.append((seed, conditions, str(output), args.mpi_ranks))
    specs.extend(
        (case, fixed_conditions, str(output), args.mpi_ranks) for case in fixed_cases
    )
    started = time.perf_counter()
    jobs = min(args.jobs, len(specs)) if specs else 0
    if specs:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(run_3d_case, specs))
    else:
        results = []
    for seed in seeds:
        if seed["status"] != "PASS":
            results.append(
                {
                    "status": "SKIPPED",
                    "case": case_key(seed),
                    "operating_point": seed["operating_point"],
                    "seed_model": seed["model"],
                    "error": "2D equilibrium seed failed",
                }
            )
    results.sort(key=lambda value: str(value["case"]))
    expected_cases = len(seeds) + len(fixed_cases)
    status = (
        "NUMERICAL_PASS_PHYSICAL_MODEL_INCOMPLETE"
        if len(results) == expected_cases
        and all(item["status"] == "PASS" for item in results)
        else "PARTIAL_OR_FAILED"
    )
    summary = {
        "status": status,
        "purpose": "3D single-phase diagnostics at load-equilibrium seed positions and prescribed eccentricities",
        "not_claimed": [
            "a 3D Reynolds boundary-condition solve",
            "a 3D JFO solve",
            "a cavitation-valid pressure field",
            "a 3D load equilibrium",
        ],
        "openfoam_version": foam_version,
        "mpi_version": command_output(["mpirun", "--version"]).splitlines()[0],
        "wall_seconds": time.perf_counter() - started,
        "worker_cases": jobs,
        "mpi_ranks_per_case": args.mpi_ranks,
        "load_equilibrium_seed_case_count": len(seeds),
        "fixed_eccentricity_case_count": len(fixed_cases),
        "fixed_eccentricity_ratios": args.fixed_eccentricity_ratios,
        "results": results,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    record_generation(
        output,
        stage="study",
        operation="paper-reproduction-three-dimensional-diagnostics",
        status=status,
        argv=argv,
        resolved_inputs={
            "input": args.input,
            "surface_speed_m_s": args.surface_speed_m_s,
            "seed_grid": [args.seed_n_theta, args.seed_n_axial],
            "fixed_eccentricity_ratios": args.fixed_eccentricity_ratios,
            "fixed_eccentricity_angle_deg": -90.0,
            "fixed_eccentricity_rpm": 2000.0,
            "jobs": jobs,
            "mpi_ranks": args.mpi_ranks,
            "mesh": {
                "n_theta": 576,
                "n_axial": 180,
                "n_gap": 12,
                "cells": 1_252_800,
                "minimum_fluent_orthogonal_quality": 0.9,
            },
        },
        input_units={
            "surface_speed": "m/s",
            "rpm": "rev/min",
            "pressure": "Pa",
            "load": "N",
            "length": "m",
        },
        producer_files=(Path(__file__),),
        output_files=(seeds_path, summary_path),
        upstream_artifacts=(args.input,),
        tool_versions={"openfoam": foam_version},
        acceptance_status=status,
        repository=REPO_ROOT,
    )
    return 0 if status == "NUMERICAL_PASS_PHYSICAL_MODEL_INCOMPLETE" else 2


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface-speed-m-s", type=float, default=2.6)
    parser.add_argument("--seed-n-theta", type=int, default=448)
    parser.add_argument("--seed-n-axial", type=int, default=140)
    parser.add_argument(
        "--fixed-eccentricity-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_FIXED_ECCENTRICITY_RATIOS),
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    args = parser.parse_args(values)
    try:
        return run(args, values)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
