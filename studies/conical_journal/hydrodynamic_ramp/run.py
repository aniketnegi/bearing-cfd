#!/usr/bin/env python3
"""Guarded pseudo-time RPM continuation for the local OpenFOAM bearing case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from bearing_cfd.artifacts import record_generation


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CASE = (
    REPO_ROOT
    / "out/archive/conical_journal/simulation/openfoam/multiphase/openfoam_s100_pure_hydrodynamic"
)
DEFAULT_TEMPLATE = SOURCE_CASE
DEFAULT_SEED = SOURCE_CASE / "checkpoints" / "15rpm" / "400"
DEFAULT_WORK_CASE = Path("out/conical_journal/studies/hydrodynamic-ramp")
DEFAULT_BASHRC = (
    REPO_ROOT.parents[1] / ".openfoam/OpenFOAM-14/etc/bashrc"
)
MILESTONES_RPM = (20.0, 50.0, 100.0, 150.0, 200.0, 496.6, 1000.0, 1500.0, 2000.0)
SCALAR_NAMES = (
    "feedFlowRate",
    "z0FlowRate",
    "zLFlowRate",
    "maxU",
    "minPressure",
    "maxPressure",
    "minAlphaOil",
    "meanAlphaOil",
    "maxAlphaOil",
)
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
PMIN_RE = re.compile(r"min\(all\) of p_rgh =\s*(" + FLOAT_RE.pattern + r")")
AMIN_RE = re.compile(r"min\(all\) of alpha\.oil =\s*(" + FLOAT_RE.pattern + r")")
TIME_RE = re.compile(r"^Time =\s*(" + FLOAT_RE.pattern + r")s?\s*$")
CONTINUITY_RE = re.compile(r"time step continuity errors\s*: sum local =\s*(" + FLOAT_RE.pattern + r")")
BAD_NUMBER_RE = re.compile(r"(?<![A-Za-z])(nan|[-+]?inf)(?![A-Za-z])", re.IGNORECASE)
NORMAL_SIGFPE_STARTUP = "sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE)."
CSV_FIELDS = (
    "target_rpm",
    "observed_rpm",
    "observed_time",
    "pseudo_time",
    "status",
    "hold_steps",
    "p_min_pa",
    "p_max_pa",
    "alpha_min",
    "alpha_mean",
    "alpha_max",
    "u_max_m_s",
    "feed_flow_m3_s",
    "z0_flow_m3_s",
    "zL_flow_m3_s",
    "opening_imbalance",
    "corrected_local_continuity",
    "largest_plateau_drift",
    "reason",
)


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * math.pi / 30.0


def planned_targets(start_rpm: float, target_rpm: float) -> list[float]:
    if target_rpm <= start_rpm or math.isclose(target_rpm, start_rpm):
        return []
    targets = [rpm for rpm in MILESTONES_RPM if start_rpm < rpm <= target_rpm]
    if not targets or not math.isclose(targets[-1], target_rpm):
        targets.append(target_rpm)
    return targets


def rounded_steps(delta_rpm: float, rpm_per_step: float, chunk_steps: int) -> int:
    raw = math.ceil(delta_rpm / rpm_per_step)
    return max(chunk_steps, math.ceil(raw / chunk_steps) * chunk_steps)


def time_text(value: float) -> str:
    return str(int(round(value))) if math.isclose(value, round(value)) else f"{value:.12g}"


def rpm_text(value: float) -> str:
    return f"{value:.6g}"


def foam_command(bashrc: Path, words: list[str]) -> str:
    command = " ".join(shlex.quote(str(word)) for word in words)
    return f"source {shlex.quote(str(bashrc))} >/dev/null 2>&1 && {command}"


def run_foam_tool(bashrc: Path, words: list[str]) -> str:
    result = subprocess.run(
        ["bash", "-lc", foam_command(bashrc, words)],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def fatal_log_reason(line: str) -> str:
    if "FOAM FATAL" in line or "Floating point exception" in line:
        return line.strip()
    return ""


def set_dictionary(bashrc: Path, path: Path, entry: str, value: str) -> None:
    run_foam_tool(
        bashrc,
        ["foamDictionary", "-writePrecision", "17", path, "-entry", entry, "-set", value],
    )


def read_p_sat(fv_models: Path) -> float:
    text = fv_models.read_text(encoding="utf-8")
    match = re.search(r"\bpSat\s+(" + FLOAT_RE.pattern + r")\s*;", text)
    if not match:
        raise ValueError(f"could not read pSat from {fv_models}")
    return float(match.group(1))


def check_inputs(args: argparse.Namespace) -> None:
    if args.start_rpm < 0 or args.target_rpm <= args.start_rpm:
        raise ValueError("target RPM must be greater than the non-negative start RPM")
    if args.chunk_steps <= 0 or args.min_hold_steps < args.chunk_steps:
        raise ValueError("chunk and hold lengths must be positive")
    if args.min_hold_steps % args.chunk_steps or args.max_hold_steps % args.chunk_steps:
        raise ValueError("hold lengths must be multiples of chunk steps")
    if args.max_hold_steps < args.min_hold_steps or args.rpm_per_step <= 0:
        raise ValueError("maximum hold and ramp-rate values are inconsistent")
    if not args.template_case.is_dir():
        raise FileNotFoundError(f"template case not found: {args.template_case}")
    if not args.seed.is_dir():
        raise FileNotFoundError(f"accepted seed not found: {args.seed}")
    try:
        seed_time = float(args.seed.name)
    except ValueError as error:
        raise ValueError(f"seed directory must have a numeric time name: {args.seed}") from error
    if not math.isfinite(seed_time) or not math.isclose(seed_time, round(seed_time)):
        raise ValueError("this driver requires an integer pseudo-time seed")
    for name in ("constant", "system"):
        if not (args.template_case / name).is_dir():
            raise FileNotFoundError(f"template is missing {name}/")
    for name in ("U", "alpha.oil", "p_rgh", "phi"):
        if not (args.seed / name).is_file():
            raise FileNotFoundError(f"seed is missing {name}")
    if args.run and not args.foam_bashrc.is_file():
        raise FileNotFoundError(f"OpenFOAM bashrc not found: {args.foam_bashrc}")


def check_openfoam(bashrc: Path) -> None:
    subprocess.run(
        [
            "bash",
            "-lc",
            foam_command(bashrc, ["command", "-v", "foamDictionary"])
            + " >/dev/null && command -v foamRun >/dev/null",
        ],
        check=True,
    )


def openfoam_version(bashrc: Path) -> str:
    command = (
        f"source {shlex.quote(str(bashrc))} >/dev/null 2>&1"
        " && printf '%s' \"$WM_PROJECT_VERSION\""
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def provenance_files(case: Path) -> list[Path]:
    files = [case / "ramp-state.json", case / "ramp-results.csv"]
    files.extend((case / "logs").glob("*.log"))
    files.extend((case / "accepted").rglob("*"))
    return [path for path in files if path.is_file()]


def upstream_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for path in (args.template_case / "constant", args.template_case / "system", args.seed):
        files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    return files


def record_run(
    case: Path,
    args: argparse.Namespace,
    state: dict,
    argv: Sequence[str],
    *,
    status: str | None = None,
) -> None:
    record_generation(
        case,
        stage="study",
        operation="hydrodynamic-ramp",
        status=status or str(state["status"]),
        resolved_inputs=vars(args),
        input_units={
            "start_rpm": "rpm",
            "target_rpm": "rpm",
            "rpm_per_step": "rpm/pseudo-time-step",
            "chunk_steps": "pseudo-time-steps",
            "min_hold_steps": "pseudo-time-steps",
            "max_hold_steps": "pseudo-time-steps",
        },
        producer_files=(Path(__file__),),
        output_files=provenance_files(case),
        upstream_artifacts=upstream_files(args),
        tool_versions={"openfoam": openfoam_version(args.foam_bashrc)},
        argv=argv,
        case_name=args.work_case.name,
        acceptance_status=status or str(state["status"]),
        repository=REPO_ROOT,
    )


def seed_case(args: argparse.Namespace, argv: Sequence[str] = ()) -> dict:
    work_case = args.work_case
    if work_case.exists():
        raise FileExistsError(f"work case already exists: {work_case}; use --resume or another path")
    building = work_case.with_name(f".{work_case.name}.building")
    if building.exists():
        raise FileExistsError(f"incomplete build path already exists: {building}")
    building.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": 1,
        "status": "READY",
        "requested_target_rpm": args.target_rpm,
        "current_rpm": args.start_rpm,
        "current_time": float(args.seed.name),
        "last_accepted_rpm": args.start_rpm,
        "last_accepted_time": float(args.seed.name),
        "template_case": str(args.template_case.resolve()),
        "seed": str(args.seed.resolve()),
        "results": [],
        "reason": "",
    }
    try:
        building.mkdir()
        shutil.copytree(args.template_case / "constant", building / "constant")
        shutil.copytree(args.template_case / "system", building / "system")
        shutil.copytree(args.seed, building / args.seed.name)
        (building / "logs").mkdir()
        set_dictionary(args.foam_bashrc, building / "system" / "controlDict", "startFrom", "latestTime")
        set_dictionary(args.foam_bashrc, building / "system" / "controlDict", "writeInterval", str(args.chunk_steps))
        set_dictionary(args.foam_bashrc, building / "system" / "controlDict", "purgeWrite", "2")
        save_state(building, state)
        record_run(building, args, state, argv)
        os.replace(building, work_case)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise
    return state


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_state(case: Path, state: dict) -> None:
    atomic_text(case / "ramp-state.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary = case / ".ramp-results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["results"])
    os.replace(temporary, case / "ramp-results.csv")


def load_state(case: Path) -> dict:
    path = case / "ramp-state.json"
    if not path.is_file():
        raise FileNotFoundError(f"resume state not found: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    false_startup_failure = (
        state.get("status") == "FAILED"
        and state.get("reason") == NORMAL_SIGFPE_STARTUP
        and state.get("observed_time") is None
        and math.isclose(float(state["current_time"]), float(state["last_accepted_time"]))
        and math.isclose(float(state["current_rpm"]), float(state["last_accepted_rpm"]))
    )
    if false_startup_failure:
        state["results"] = [
            row for row in state["results"] if row.get("reason") != NORMAL_SIGFPE_STARTUP
        ]
        state.update(status="READY", reason="", observed_time=None, observed_rpm=None)
        save_state(case, state)
    if state.get("status") == "HOLDING":
        raise RuntimeError(
            "automatic resume from an unfinished target hold is disabled; "
            "start a new work case from the last accepted checkpoint"
        )
    if state.get("status") in {"SATURATION_THRESHOLD", "CAVITATION_ONSET", "FAILED", "COMPLETE"}:
        raise RuntimeError(f"refusing to resume terminal status {state['status']}")
    latest = latest_time(case)
    if not math.isclose(latest, float(state["current_time"])):
        raise RuntimeError(
            f"latest field time {latest:g} does not match safe state time "
            f"{float(state['current_time']):g}"
        )
    return state


def latest_time(case: Path) -> float:
    times = []
    for path in case.iterdir():
        if path.is_dir():
            try:
                times.append(float(path.name))
            except ValueError:
                pass
    if not times:
        raise RuntimeError(f"no numeric time directories in {case}")
    return max(times)


def find_time_dir(case: Path, value: float) -> Path:
    for path in case.iterdir():
        if path.is_dir():
            try:
                if math.isclose(float(path.name), value):
                    return path
            except ValueError:
                pass
    raise FileNotFoundError(f"field time {value:g} not found in {case}")


def scalar_series(case: Path, name: str) -> list[tuple[float, float]]:
    values: dict[float, float] = {}
    for path in sorted((case / "postProcessing" / name).glob("*/*.dat")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                values[float(fields[0])] = float(fields[1])
    return sorted(values.items())


def force_series(case: Path) -> list[tuple[float, tuple[float, ...]]]:
    values: dict[float, tuple[float, ...]] = {}
    for path in sorted((case / "postProcessing" / "journalForces").glob("*/forces.dat")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            numbers = [float(value) for value in FLOAT_RE.findall(line)]
            if len(numbers) == 13:
                values[numbers[0]] = tuple(numbers[1:])
    return sorted(values.items())


def recent(series: list[tuple[float, object]], end_time: float, count: int = 5) -> list:
    selected = [(time, value) for time, value in series if time <= end_time + 1e-9]
    if len(selected) < count or not math.isclose(selected[-1][0], end_time):
        raise RuntimeError(f"missing {count}-step monitor window ending at {end_time:g}")
    return [value for _, value in selected[-count:]]


def magnitude(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def relative_drift(values: list[float], floor: float = 1e-12) -> float:
    return (max(values) - min(values)) / max(max(abs(value) for value in values), floor)


def vector_drift(vectors: list[tuple[float, ...]], floor: float = 1e-12) -> float:
    scale = max(max(magnitude(vector) for vector in vectors), floor)
    return max(
        magnitude(tuple(value - reference for value, reference in zip(vector, vectors[-1])))
        for vector in vectors
    ) / scale


def last_continuity(log_path: Path) -> float:
    matches = CONTINUITY_RE.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise RuntimeError(f"no continuity result in {log_path}")
    return float(matches[-1])


def evaluate_hold(case: Path, end_time: float, log_path: Path, p_sat: float) -> dict:
    windows = {name: recent(scalar_series(case, name), end_time) for name in SCALAR_NAMES}
    forces = recent(force_series(case), end_time)
    p_min = windows["minPressure"][-1]
    p_max = windows["maxPressure"][-1]
    flows = [windows[name][-1] for name in ("feedFlowRate", "z0FlowRate", "zLFlowRate")]
    pressure_span = max(abs(p_max - p_min), 1.0)
    pressure_force = [values[0:3] for values in forces]
    viscous_force = [values[3:6] for values in forces]
    pressure_moment = [values[6:9] for values in forces]
    viscous_mz = [abs(values[11]) for values in forces]
    drifts = {
        "u": relative_drift(windows["maxU"]),
        "p_min": (max(windows["minPressure"]) - min(windows["minPressure"])) / pressure_span,
        "p_max": (max(windows["maxPressure"]) - min(windows["maxPressure"])) / pressure_span,
        "pressure_force": vector_drift(pressure_force),
        "viscous_force": vector_drift(viscous_force),
        "pressure_moment": vector_drift(pressure_moment),
        "viscous_mz": relative_drift(viscous_mz),
        "feed_flow": relative_drift(windows["feedFlowRate"]),
        "z0_flow": relative_drift(windows["z0FlowRate"]),
        "zL_flow": relative_drift(windows["zLFlowRate"]),
        "mean_alpha": max(windows["meanAlphaOil"]) - min(windows["meanAlphaOil"]),
    }
    imbalance = abs(sum(flows)) / max(sum(abs(value) for value in flows), 1e-14)
    continuity = last_continuity(log_path)
    finite = [
        p_min,
        p_max,
        *flows,
        continuity,
        windows["minAlphaOil"][-1],
        windows["meanAlphaOil"][-1],
        windows["maxAlphaOil"][-1],
        *drifts.values(),
    ]
    checks = {
        "finite": all(math.isfinite(value) for value in finite),
        "single_liquid": windows["minAlphaOil"][-1] >= 0.999999
        and windows["meanAlphaOil"][-1] >= 0.999999
        and windows["maxAlphaOil"][-1] <= 1.00000001,
        "above_saturation": p_min >= p_sat,
        "continuity": continuity < 1e-5,
        "opening_balance": imbalance < 0.005,
        "primary_plateau": max(
            drifts[name] for name in ("u", "p_min", "p_max", "pressure_force", "pressure_moment")
        )
        < 0.005,
        "secondary_plateau": max(
            drifts[name]
            for name in ("viscous_force", "viscous_mz", "feed_flow", "z0_flow", "zL_flow")
        )
        < 0.01,
        "phase_plateau": drifts["mean_alpha"] < 1e-5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "drifts": drifts,
        "p_min_pa": p_min,
        "p_max_pa": p_max,
        "alpha_min": windows["minAlphaOil"][-1],
        "alpha_mean": windows["meanAlphaOil"][-1],
        "alpha_max": windows["maxAlphaOil"][-1],
        "u_max_m_s": windows["maxU"][-1],
        "feed_flow_m3_s": flows[0],
        "z0_flow_m3_s": flows[1],
        "zL_flow_m3_s": flows[2],
        "opening_imbalance": imbalance,
        "corrected_local_continuity": continuity,
        "largest_plateau_drift": max(drifts.values()),
    }


def run_chunk(case: Path, bashrc: Path, end_time: float, p_sat: float, log_path: Path) -> dict:
    set_dictionary(bashrc, case / "system" / "controlDict", "endTime", time_text(end_time))
    command = foam_command(bashrc, ["exec", "foamRun", "-case", case])
    outcome = {"status": "OK", "reason": "", "observed_time": None}
    low_alpha_count = 0
    step_continuities: list[float] = []
    bad_continuity: float | None = None
    saw_end = False
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                time_match = TIME_RE.match(line.strip())
                if time_match:
                    outcome["observed_time"] = float(time_match.group(1))
                    step_continuities = []
                    bad_continuity = None
                if line.strip() == "End":
                    saw_end = True
                continuity_match = CONTINUITY_RE.search(line)
                if continuity_match:
                    step_continuities.append(float(continuity_match.group(1)))
                p_match = PMIN_RE.search(line)
                a_match = AMIN_RE.search(line)
                reason = ""
                status = "FAILED"
                if (
                    line.startswith("ExecutionTime")
                    and step_continuities
                    and step_continuities[-1] >= 1e-3
                ):
                    bad_continuity = step_continuities[-1]
                if p_match and float(p_match.group(1)) < p_sat:
                    status = "SATURATION_THRESHOLD"
                    reason = f"p_min={float(p_match.group(1)):.12g} Pa crossed pSat={p_sat:.12g} Pa"
                if a_match:
                    alpha = float(a_match.group(1))
                    if alpha < -1e-8 or alpha > 1.00000001:
                        reason = f"alpha.oil={alpha:.12g} left its bounded range"
                    low_alpha_count = low_alpha_count + 1 if alpha < 0.999999 else 0
                    if low_alpha_count >= 2:
                        status = "CAVITATION_ONSET"
                        reason = f"alpha.oil_min={alpha:.12g} remained below 0.999999"
                    elif bad_continuity is not None:
                        reason = (
                            "final-corrector local continuity reached "
                            f"{bad_continuity:.12g}"
                        )
                fatal_reason = fatal_log_reason(line)
                if fatal_reason:
                    reason = fatal_reason
                if BAD_NUMBER_RE.search(line):
                    reason = f"non-finite solver output: {line.strip()}"
                if "Solving for p_rgh" in line and re.search(r"No Iterations\s+100\b", line):
                    reason = "pressure solver reached its 100-iteration limit"
                if reason and outcome["status"] == "OK":
                    outcome.update(status=status, reason=reason)
                    process.terminate()
        return_code = process.wait()
    except BaseException:
        stop_process(process)
        raise
    if outcome["status"] != "OK":
        return outcome
    if return_code != 0:
        return {"status": "FAILED", "reason": f"foamRun exited with {return_code}", "observed_time": outcome["observed_time"]}
    if not saw_end:
        return {"status": "FAILED", "reason": "foamRun log has no End marker", "observed_time": outcome["observed_time"]}
    continuity = last_continuity(log_path)
    if continuity >= 1e-3:
        return {
            "status": "FAILED",
            "reason": f"final-corrector local continuity reached {continuity:.12g}",
            "observed_time": outcome["observed_time"],
        }
    return outcome


def halt(case: Path, state: dict, outcome: dict, target_rpm: float) -> None:
    state.update(
        status=outcome["status"],
        reason=outcome["reason"],
        attempted_target_rpm=target_rpm,
        observed_time=outcome.get("observed_time"),
        observed_rpm=outcome.get("observed_rpm"),
    )
    state["results"].append(
        {
            "target_rpm": target_rpm,
            "pseudo_time": state["current_time"],
            "status": outcome["status"],
            "reason": outcome["reason"],
            **{key: value for key, value in outcome.items() if key in CSV_FIELDS},
        }
    )
    if "checks" in outcome:
        state["last_evaluation"] = {
            key: value for key, value in outcome.items() if key not in {"status", "reason"}
        }
    save_state(case, state)


def campaign(args: argparse.Namespace, argv: Sequence[str] = ()) -> dict:
    check_openfoam(args.foam_bashrc)
    if not args.resume:
        seed_omega = float(
            run_foam_tool(
                args.foam_bashrc,
                [
                    "foamDictionary",
                    args.seed / "U",
                    "-entry",
                    "boundaryField/journal_wall/omega/value",
                    "-value",
                ],
            )
        )
        seed_rpm = seed_omega * 30.0 / math.pi
        if abs(seed_rpm - args.start_rpm) > 0.01:
            raise ValueError(
                f"seed wall speed is {seed_rpm:.6g} rpm, not requested {args.start_rpm:.6g} rpm"
            )
    state = load_state(args.work_case) if args.resume else seed_case(args, argv)
    case = args.work_case
    p_sat = read_p_sat(case / "constant" / "fvModels")
    current_rpm = float(state["current_rpm"])
    current_time = float(state["current_time"])
    if args.resume:
        requested = float(state["requested_target_rpm"])
        if not math.isclose(requested, args.target_rpm):
            raise ValueError(
                f"resume target must remain {requested:g} rpm, not {args.target_rpm:g} rpm"
            )
        if current_rpm > args.target_rpm and not math.isclose(current_rpm, args.target_rpm):
            raise ValueError("resume target is below the current safe RPM")
    for stage_index, target_rpm in enumerate(planned_targets(current_rpm, args.target_rpm), start=1):
        rate = min(args.rpm_per_step, 0.05) if current_rpm < 20.0 and target_rpm <= 20.0001 else args.rpm_per_step
        ramp_steps = rounded_steps(target_rpm - current_rpm, rate, args.chunk_steps)
        stage_start_rpm = current_rpm
        ramp_start_time = current_time
        ramp_end = ramp_start_time + ramp_steps
        anchor = ramp_end + args.max_hold_steps + args.chunk_steps
        omega = (
            "{ type table; values ("
            f"({time_text(ramp_start_time)} {rpm_to_rad_s(stage_start_rpm):.17g}) "
            f"({time_text(ramp_end)} {rpm_to_rad_s(target_rpm):.17g}) "
            f"({time_text(anchor)} {rpm_to_rad_s(target_rpm):.17g})"
            "); }"
        )
        set_dictionary(
            args.foam_bashrc,
            find_time_dir(case, current_time) / "U",
            "boundaryField/journal_wall/omega",
            omega,
        )
        state.update(status="RAMPING", reason="", attempted_target_rpm=target_rpm)
        save_state(case, state)
        for chunk_end in range(int(ramp_start_time) + args.chunk_steps, int(ramp_end) + 1, args.chunk_steps):
            log_path = case / "logs" / (
                f"{stage_index:02d}-ramp-{rpm_text(stage_start_rpm)}-to-{rpm_text(target_rpm)}"
                f"-t{time_text(current_time)}-{chunk_end}.log"
            )
            outcome = run_chunk(case, args.foam_bashrc, chunk_end, p_sat, log_path)
            if outcome["status"] != "OK":
                if outcome.get("observed_time") is not None:
                    fraction = max(
                        0.0,
                        min(1.0, (float(outcome["observed_time"]) - ramp_start_time) / ramp_steps),
                    )
                    outcome["observed_rpm"] = stage_start_rpm + (
                        target_rpm - stage_start_rpm
                    ) * fraction
                halt(case, state, outcome, target_rpm)
                return state
            fraction = (chunk_end - ramp_start_time) / ramp_steps
            current_rpm = stage_start_rpm + (target_rpm - stage_start_rpm) * fraction
            current_time = float(chunk_end)
            state.update(
                status="HOLDING" if math.isclose(current_rpm, target_rpm) else "RAMPING",
                current_rpm=current_rpm,
                current_time=current_time,
                hold_steps=0,
            )
            save_state(case, state)
        current_rpm = target_rpm
        state.update(
            status="HOLDING",
            current_rpm=current_rpm,
            current_time=current_time,
            hold_steps=0,
        )
        save_state(case, state)
        result = None
        hold_steps = 0
        while hold_steps < args.max_hold_steps:
            chunk_end = current_time + args.chunk_steps
            log_path = case / "logs" / (
                f"{stage_index:02d}-hold-{rpm_text(target_rpm)}"
                f"-t{time_text(current_time)}-{time_text(chunk_end)}.log"
            )
            outcome = run_chunk(case, args.foam_bashrc, chunk_end, p_sat, log_path)
            if outcome["status"] != "OK":
                outcome["observed_rpm"] = target_rpm
                halt(case, state, outcome, target_rpm)
                return state
            current_time = chunk_end
            hold_steps += args.chunk_steps
            state.update(current_time=current_time, hold_steps=hold_steps)
            save_state(case, state)
            if hold_steps >= args.min_hold_steps:
                result = evaluate_hold(case, current_time, log_path, p_sat)
                if result["passed"]:
                    break
        if result is None or not result["passed"]:
            failed_checks = ", ".join(
                name for name, passed in (result or {}).get("checks", {}).items() if not passed
            )
            failed = {
                "status": "FAILED",
                "reason": "target hold failed: " + (failed_checks or "monitor data unavailable"),
                "observed_time": current_time,
                "observed_rpm": target_rpm,
                **(result or {}),
            }
            halt(case, state, failed, target_rpm)
            return state
        checkpoint = case / "accepted" / f"{rpm_text(target_rpm)}rpm" / time_text(current_time)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_building = checkpoint.with_name(f".{checkpoint.name}.building")
        if checkpoint.exists() or checkpoint_building.exists():
            raise FileExistsError(f"checkpoint path already exists: {checkpoint}")
        try:
            shutil.copytree(find_time_dir(case, current_time), checkpoint_building)
            os.replace(checkpoint_building, checkpoint)
        except BaseException:
            if checkpoint_building.exists():
                shutil.rmtree(checkpoint_building)
            raise
        row = {
            "target_rpm": target_rpm,
            "observed_rpm": target_rpm,
            "observed_time": current_time,
            "pseudo_time": current_time,
            "status": "ACCEPTED",
            "hold_steps": hold_steps,
            "reason": "",
            **{key: value for key, value in result.items() if key in CSV_FIELDS},
        }
        state["results"].append(row)
        state.update(
            status="ACCEPTED_STAGE",
            current_rpm=target_rpm,
            current_time=current_time,
            last_accepted_rpm=target_rpm,
            last_accepted_time=current_time,
            reason="",
            last_evaluation=result,
        )
        save_state(case, state)
    state.update(status="COMPLETE", reason="")
    save_state(case, state)
    return state


def print_plan(args: argparse.Namespace) -> None:
    current = args.start_rpm
    if args.resume and (args.work_case / "ramp-state.json").is_file():
        current = float(
            json.loads((args.work_case / "ramp-state.json").read_text(encoding="utf-8"))[
                "current_rpm"
            ]
        )
    minimum = 0
    maximum = 0
    print("Guarded OpenFOAM pseudo-time continuation")
    print(f"  seed:   {args.seed}")
    print(f"  output: {args.work_case}")
    print()
    print("  from RPM -> target RPM   ramp steps   hold steps")
    for target in planned_targets(current, args.target_rpm):
        rate = min(args.rpm_per_step, 0.05) if current < 20.0 and target <= 20.0001 else args.rpm_per_step
        steps = rounded_steps(target - current, rate, args.chunk_steps)
        minimum += steps + args.min_hold_steps
        maximum += steps + args.max_hold_steps
        print(f"  {current:8.3f} -> {target:10.3f}   {steps:10d}   {args.min_hold_steps}-{args.max_hold_steps}")
        current = target
    print()
    print(f"  total pseudo-steps if every stage is reached: {minimum}-{maximum}")
    print("  safety stop: first p_min < configured pSat, sustained alpha.oil < 0.999999, or solver failure")
    if not args.run:
        print("  dry run only; add --run to create and execute the isolated case")


def self_test() -> None:
    assert planned_targets(15, 2000) == list(MILESTONES_RPM)
    assert planned_targets(15, 30) == [20.0, 30]
    assert planned_targets(20, 20) == []
    assert rounded_steps(5, 0.05, 25) == 100
    assert math.isclose(relative_drift([1, 2, 1, 2, 1]), 0.5)
    assert vector_drift([(1.0, 0.0), (0.0, 1.0)]) > 1.0
    assert not fatal_log_reason(NORMAL_SIGFPE_STARTUP)
    assert fatal_log_reason("Floating point exception (core dumped)")
    with tempfile.TemporaryDirectory() as directory:
        case = Path(directory)
        times = range(1, 6)
        scalar_values = {
            "feedFlowRate": -1.0e-10,
            "z0FlowRate": 0.8e-10,
            "zLFlowRate": 0.2e-10,
            "maxU": 0.1,
            "minPressure": 1000.0,
            "maxPressure": 2000.0,
            "minAlphaOil": 1.0,
            "meanAlphaOil": 1.0,
            "maxAlphaOil": 1.0,
        }
        for name, value in scalar_values.items():
            path = case / "postProcessing" / name / "0" / "value.dat"
            path.parent.mkdir(parents=True)
            path.write_text(
                "# test\n" + "".join(f"{time} {value}\n" for time in times),
                encoding="utf-8",
            )
        force_path = case / "postProcessing" / "journalForces" / "0" / "forces.dat"
        force_path.parent.mkdir(parents=True)
        force_path.write_text(
            "# test\n"
            + "".join(
                f"{time} ((1 0 0) (0.1 0 0)) ((0 1 0) (0 0 0.01))\n" for time in times
            ),
            encoding="utf-8",
        )
        log = case / "log"
        log.write_text(
            "time step continuity errors : sum local = 1e-8, global = 0, cumulative = 0\n",
            encoding="utf-8",
        )
        assert evaluate_hold(case, 5, log, 0.5)["passed"]
        alpha_path = case / "postProcessing" / "minAlphaOil" / "0" / "value.dat"
        alpha_path.write_text("# test\n" + "".join(f"{time} 0.9\n" for time in times), encoding="utf-8")
        assert not evaluate_hold(case, 5, log, 0.5)["passed"]
    print("self-test: PASS")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-case", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--work-case", type=Path, default=DEFAULT_WORK_CASE)
    parser.add_argument("--foam-bashrc", type=Path, default=DEFAULT_BASHRC)
    parser.add_argument("--start-rpm", type=float, default=15.0)
    parser.add_argument("--target-rpm", type=float, default=2000.0)
    parser.add_argument("--rpm-per-step", type=float, default=0.2)
    parser.add_argument("--chunk-steps", type=int, default=25)
    parser.add_argument("--min-hold-steps", type=int, default=100)
    parser.add_argument("--max-hold-steps", type=int, default=500)
    parser.add_argument("--run", action="store_true", help="create and execute the isolated case")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted work case")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def request_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, request_interrupt)
    values = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(values)
    work_case_existed = args.work_case.exists()
    if args.self_test:
        self_test()
        return 0
    try:
        check_inputs(args)
        print_plan(args)
        if not args.run:
            return 0
        state = campaign(args, values)
        record_run(args.work_case, args, state, values)
        print(f"\nstatus: {state['status']}")
        if state.get("reason"):
            print(f"reason: {state['reason']}")
        print(f"state:  {args.work_case / 'ramp-state.json'}")
        return 0 if state["status"] == "COMPLETE" else 3
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        state_path = args.work_case / "ramp-state.json"
        if state_path.is_file() and (args.resume or not work_case_existed):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            record_run(args.work_case, args, state, values, status="FAILED")
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        state_path = args.work_case / "ramp-state.json"
        if state_path.is_file() and (args.resume or not work_case_existed):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            record_run(args.work_case, args, state, values, status="INTERRUPTED")
        print("\ninterrupted; the last completed chunk remains resumable", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
