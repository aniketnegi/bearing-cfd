#!/usr/bin/env pvpython
"""Generate plots and field renders for the accepted OQ90 OpenFOAM stages."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image
from paraview.simple import (  # type: ignore[import-not-found]
    Calculator,
    CellDatatoPointData,
    ColorBy,
    CreateView,
    Delete,
    GetColorTransferFunction,
    GetScalarBar,
    LegacyVTKReader,
    Render,
    SaveScreenshot,
    Show,
    Text,
    _DisableFirstRenderCameraReset,
)
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.vtkIOLegacy import vtkDataSetReader

from bearing_cfd.artifacts import make_staging_directory, publish_generation


REPO_ROOT = Path(__file__).resolve().parents[3]


RHO = 860.0
P_ATM = 101_325.0
P_SUPPLY = 500_000.0
N_GAP = 12
FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
NUMBER_RE = re.compile(FLOAT)
TIME_RE = re.compile(r"Time = (" + FLOAT + r")s")
RESIDUAL_RE = re.compile(
    r"Solving for (Ux|Uy|Uz|p), Initial residual = (" + FLOAT + r")"
)

STAGES = (
    {
        "key": "atmospheric",
        "label": "0 rpm · atmospheric",
        "rpm": 0.0,
        "start": "0",
        "time": 1,
        "log": "log.0rpm-atmospheric",
        "physical": True,
    },
    {
        "key": "pressure_fed",
        "label": "0 rpm · 0.5 MPa feed",
        "rpm": 0.0,
        "start": "1",
        "time": 128,
        "log": "log.0rpm-pressure-fed",
        "physical": True,
    },
    {
        "key": "496rpm",
        "label": "496.563 rpm · 0.5 MPa feed",
        "rpm": 496.563,
        "start": "128",
        "time": 266,
        "log": "log.496p563rpm",
        "physical": False,
    },
    {
        "key": "2000rpm",
        "label": "2000 rpm · 0.5 MPa feed",
        "rpm": 2000.0,
        "start": "267",
        "time": 402,
        "log": "log.2000rpm",
        "physical": False,
    },
)


def numeric_columns(path: Path, count: int = 2) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in NUMBER_RE.findall(line)]
        rows.append(values[:count])
    return np.asarray(rows, dtype=float)


def force_history(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in NUMBER_RE.findall(line)]
        if len(values) != 13:
            raise ValueError(f"Expected 13 force columns in {path}, got {len(values)}")
        rows.append(values)
    data = np.asarray(rows, dtype=float)
    force = data[:, 1:4] + data[:, 4:7]
    moment = data[:, 7:10] + data[:, 10:13]
    return {
        "time": data[:, 0],
        "force": force,
        "load": np.linalg.norm(force, axis=1),
        "moment": moment,
    }


def residual_history(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[tuple[float, float]]] = {
        "Ux": [],
        "Uy": [],
        "Uz": [],
        "p": [],
    }
    current_time: float | None = None
    for line in path.read_text(errors="replace").splitlines():
        time_match = TIME_RE.search(line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
        residual_match = RESIDUAL_RE.search(line)
        if residual_match and current_time is not None:
            rows[residual_match.group(1)].append(
                (current_time, float(residual_match.group(2)))
            )
    return {field: np.asarray(values, dtype=float) for field, values in rows.items()}


def stage_histories(case: Path, stage: dict[str, object]) -> dict[str, object]:
    start = str(stage["start"])
    post = case / "postProcessing"
    return {
        "max_p": numeric_columns(post / "maxP" / start / "volFieldValue.dat"),
        "min_p": numeric_columns(post / "minP" / start / "volFieldValue.dat"),
        "max_u": numeric_columns(post / "maxU" / start / "volFieldValue.dat"),
        "feed": numeric_columns(
            post / "feedFlowRate" / start / "surfaceFieldValue.dat"
        ),
        "z0": numeric_columns(post / "z0FlowRate" / start / "surfaceFieldValue.dat"),
        "zl": numeric_columns(post / "zlFlowRate" / start / "surfaceFieldValue.dat"),
        "net": numeric_columns(
            post / "netBoundaryFlow" / start / "surfaceFieldValue.dat"
        ),
        "forces": force_history(post / "journalForces" / start / "forces.dat"),
        "residuals": residual_history(case / str(stage["log"])),
    }


def final_metrics(histories: dict[str, object]) -> dict[str, float]:
    max_p = histories["max_p"]
    min_p = histories["min_p"]
    max_u = histories["max_u"]
    feed = histories["feed"]
    z0 = histories["z0"]
    zl = histories["zl"]
    net = histories["net"]
    forces = histories["forces"]
    assert isinstance(max_p, np.ndarray)
    assert isinstance(min_p, np.ndarray)
    assert isinstance(max_u, np.ndarray)
    assert isinstance(feed, np.ndarray)
    assert isinstance(z0, np.ndarray)
    assert isinstance(zl, np.ndarray)
    assert isinstance(net, np.ndarray)
    assert isinstance(forces, dict)
    feed_final = float(feed[-1, 1])
    return {
        "pmax_abs": P_ATM + RHO * float(max_p[-1, 1]),
        "pmin_abs": P_ATM + RHO * float(min_p[-1, 1]),
        "umax": float(max_u[-1, 1]),
        "feed": feed_final,
        "z0": float(z0[-1, 1]),
        "zl": float(zl[-1, 1]),
        "net": float(net[-1, 1]),
        "imbalance_pct": (
            0.0
            if feed_final == 0
            else 100.0 * abs(float(net[-1, 1]) / feed_final)
        ),
        "load": float(forces["load"][-1]),
        "torque": float(forces["moment"][-1, 2]),
    }


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#30343b",
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#242830",
            "text.color": "#242830",
            "font.size": 10.5,
            "savefig.facecolor": "#f7f8fa",
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_run_summary(
    histories: dict[str, dict[str, object]],
    metrics: dict[str, dict[str, float]],
    output: Path,
) -> None:
    selected = ("pressure_fed", "496rpm", "2000rpm")
    rpm = np.asarray([0.0, 496.563, 2000.0])
    pmin = np.asarray([metrics[key]["pmin_abs"] for key in selected]) / 1e6
    pmax = np.asarray([metrics[key]["pmax_abs"] for key in selected]) / 1e6
    load = np.asarray([metrics[key]["load"] for key in selected]) / 1e3
    torque = np.abs(np.asarray([metrics[key]["torque"] for key in selected]))
    feed = np.abs(np.asarray([metrics[key]["feed"] for key in selected])) * 1e6
    z0 = np.asarray([metrics[key]["z0"] for key in selected]) * 1e6
    zl = np.asarray([metrics[key]["zl"] for key in selected]) * 1e6

    projection_rpm = np.linspace(2000.0, 4000.0, 81)
    pmin_fit = np.polyval(np.polyfit(rpm, pmin, 1), projection_rpm)
    pmax_fit = np.polyval(np.polyfit(rpm, pmax, 1), projection_rpm)
    load_fit = np.polyval(np.polyfit(rpm, load, 1), projection_rpm)
    torque_fit = np.polyval(np.polyfit(rpm, torque, 1), projection_rpm)

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    pressure_ax = axes[0, 0]
    pressure_ax.axhspan(-40, 0, color="#f6c9c9", alpha=0.45, label="pabs < 0")
    pressure_ax.plot(rpm, pmax, "o-", color="#c7334f", label="maximum")
    pressure_ax.plot(rpm, pmin, "o-", color="#255fa8", label="minimum")
    pressure_ax.plot(projection_rpm, pmax_fit, "--", color="#c7334f", alpha=0.65)
    pressure_ax.plot(projection_rpm, pmin_fit, "--", color="#255fa8", alpha=0.65)
    pressure_ax.axvspan(2000, 4000, color="#9299a3", alpha=0.12)
    pressure_ax.axvline(2000, color="#555b65", linewidth=0.9)
    pressure_ax.set(
        title="Absolute-pressure screen and higher-RPM extrapolation",
        xlabel="Journal speed (rpm)",
        ylabel="Absolute pressure (MPa)",
        xlim=(-80, 4050),
        ylim=(-37, 37),
    )
    pressure_ax.text(
        3050,
        -33,
        "dashed = fully-filled numerical trend only\nnot a physical cavitation prediction",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#8a2736",
    )
    pressure_ax.legend(loc="upper right", frameon=True)

    load_ax = axes[0, 1]
    torque_ax = load_ax.twinx()
    load_ax.plot(rpm, load, "o-", color="#1c7c67", label="load")
    torque_ax.plot(rpm, torque, "s-", color="#a55c16", label="|torque|")
    load_ax.plot(projection_rpm, load_fit, "--", color="#1c7c67", alpha=0.65)
    torque_ax.plot(projection_rpm, torque_fit, "--", color="#a55c16", alpha=0.65)
    load_ax.axvspan(2000, 4000, color="#9299a3", alpha=0.12)
    load_ax.set(
        title="Load and shaft torque",
        xlabel="Journal speed (rpm)",
        ylabel="Resultant load (kN)",
        xlim=(-80, 4050),
    )
    torque_ax.set_ylabel("Shaft torque magnitude (N m)")
    handles = load_ax.get_lines()[:1] + torque_ax.get_lines()[:1]
    load_ax.legend(handles, [line.get_label() for line in handles], loc="upper left")

    flow_ax = axes[1, 0]
    flow_ax.plot(rpm, feed, "o-", label="feed inflow", color="#6a3d9a")
    flow_ax.plot(rpm, z0, "o-", label="z0 discharge", color="#1f78b4")
    flow_ax.plot(rpm, zl, "o-", label="zL discharge", color="#33a02c")
    flow_ax.set(
        title="Volume-flow balance",
        xlabel="Journal speed (rpm)",
        ylabel="Volume flow (mL/s)",
    )
    flow_ax.legend(loc="best")
    for x, key in zip(rpm, selected):
        flow_ax.annotate(
            f"imbalance {metrics[key]['imbalance_pct']:.4g}%",
            (x, feed[list(rpm).index(x)]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )

    paper_ax = axes[1, 1]
    labels = ("Paper FEA", "Paper Fluent", "OpenFOAM OQ90")
    values = (35.6, 32.0, RHO * histories["2000rpm"]["max_p"][-1, 1] / P_SUPPLY)
    colors = ("#5c677d", "#7b8fa1", "#d1495b")
    bars = paper_ax.bar(labels, values, color=colors, width=0.62)
    paper_ax.set(
        title="2000 rpm peak-pressure scalar",
        ylabel=r"$P_{\max}/P_s$ using gauge numerator",
        ylim=(0, 40),
    )
    paper_ax.grid(axis="x", visible=False)
    paper_ax.bar_label(bars, fmt="%.3g", padding=4)
    paper_ax.text(
        0.98,
        0.05,
        "OpenFOAM: −2.74% vs FEA\n+8.20% vs paper Fluent\none scalar only",
        transform=paper_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbc0c7", "alpha": 0.9},
    )

    figure.suptitle(
        "OQ90 conical-bearing run sitrep · accepted numerical checkpoints",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "0.5 MPa feed is treated as gauge; rotating single-phase states are "
        "numerically converged but physically rejected where pabs < 0.",
        ha="center",
        fontsize=9.5,
        color="#7c2432",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(figure, output)


def plot_convergence(
    stage: dict[str, object],
    histories: dict[str, object],
    metrics: dict[str, float],
    output: Path,
) -> None:
    start_time = float(stage["start"])
    max_p = histories["max_p"]
    min_p = histories["min_p"]
    feed = histories["feed"]
    z0 = histories["z0"]
    zl = histories["zl"]
    net = histories["net"]
    forces = histories["forces"]
    residuals = histories["residuals"]
    assert isinstance(max_p, np.ndarray)
    assert isinstance(min_p, np.ndarray)
    assert isinstance(feed, np.ndarray)
    assert isinstance(z0, np.ndarray)
    assert isinstance(zl, np.ndarray)
    assert isinstance(net, np.ndarray)
    assert isinstance(forces, dict)
    assert isinstance(residuals, dict)

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    residual_ax = axes[0, 0]
    for field, color in (
        ("Ux", "#2166ac"),
        ("Uy", "#4393c3"),
        ("Uz", "#762a83"),
        ("p", "#d6604d"),
    ):
        values = residuals[field]
        residual_ax.semilogy(
            values[:, 0] - start_time,
            values[:, 1],
            label=field,
            color=color,
            linewidth=1.5,
        )
    residual_ax.axhline(1e-5, color="#4b4f56", linestyle="--", linewidth=1)
    residual_ax.axhline(5e-5, color="#a33b45", linestyle=":", linewidth=1)
    residual_ax.set(
        title="Initial residual history",
        xlabel="SIMPLE iterations in branch",
        ylabel="Initial residual",
    )
    residual_ax.legend(ncol=2)

    pressure_ax = axes[0, 1]
    pressure_ax.plot(
        max_p[:, 0] - start_time,
        (P_ATM + RHO * max_p[:, 1]) / 1e6,
        color="#c7334f",
        label="maximum",
    )
    pressure_ax.plot(
        min_p[:, 0] - start_time,
        (P_ATM + RHO * min_p[:, 1]) / 1e6,
        color="#255fa8",
        label="minimum",
    )
    pressure_ax.axhline(0, color="#7f1d2d", linewidth=1.1)
    pressure_ax.fill_between(
        min_p[:, 0] - start_time,
        (P_ATM + RHO * min_p[:, 1]) / 1e6,
        0,
        where=(P_ATM + RHO * min_p[:, 1]) < 0,
        color="#f1aeb5",
        alpha=0.35,
    )
    pressure_ax.set(
        title="Global absolute-pressure extrema",
        xlabel="SIMPLE iterations in branch",
        ylabel="Absolute pressure (MPa)",
    )
    pressure_ax.legend()

    force_ax = axes[1, 0]
    torque_ax = force_ax.twinx()
    branch_iterations = forces["time"] - start_time
    force_ax.plot(
        branch_iterations,
        forces["load"] / 1e3,
        color="#1c7c67",
        label="load",
    )
    torque_ax.plot(
        branch_iterations,
        forces["moment"][:, 2],
        color="#a55c16",
        label="Mz",
    )
    force_ax.set(
        title="Integrated journal response",
        xlabel="SIMPLE iterations in branch",
        ylabel="Resultant load (kN)",
    )
    torque_ax.set_ylabel("Shaft torque Mz (N m)")
    handles = force_ax.get_lines() + torque_ax.get_lines()
    force_ax.legend(handles, [line.get_label() for line in handles], loc="best")

    flow_ax = axes[1, 1]
    flow_ax.plot(
        feed[:, 0] - start_time,
        -feed[:, 1] * 1e6,
        label="feed inflow",
        color="#6a3d9a",
    )
    flow_ax.plot(
        z0[:, 0] - start_time,
        z0[:, 1] * 1e6,
        label="z0 discharge",
        color="#1f78b4",
    )
    flow_ax.plot(
        zl[:, 0] - start_time,
        zl[:, 1] * 1e6,
        label="zL discharge",
        color="#33a02c",
    )
    flow_ax.set(
        title="Boundary-flow histories",
        xlabel="SIMPLE iterations in branch",
        ylabel="Volume flow (mL/s)",
    )
    flow_ax.legend()
    imbalance_ax = flow_ax.twinx()
    imbalance = 100.0 * np.abs(net[:, 1] / feed[:, 1])
    imbalance_ax.semilogy(
        net[:, 0] - start_time,
        np.maximum(imbalance, 1e-12),
        color="#555b65",
        alpha=0.55,
        linewidth=1,
    )
    imbalance_ax.set_ylabel("Relative imbalance (%)", color="#555b65")

    figure.suptitle(
        f"{stage['label']} · convergence and acceptance monitors",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        f"Final: Umax={metrics['umax']:.4g} m/s · "
        f"|F|={metrics['load']/1e3:.4g} kN · "
        f"Mz={metrics['torque']:.5g} N m · "
        f"net/feed={metrics['imbalance_pct']:.4g}%",
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(figure, output)


def read_mid_gap(vtk_path: Path) -> dict[str, np.ndarray]:
    reader = vtkDataSetReader()
    reader.SetFileName(str(vtk_path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    dataset = reader.GetOutput()
    centers_filter = vtkCellCenters()
    centers_filter.SetInputData(dataset)
    centers_filter.Update()

    centers = vtk_to_numpy(centers_filter.GetOutput().GetPoints().GetData())
    cell_ids = vtk_to_numpy(dataset.GetCellData().GetArray("cellID")).astype(int)
    pressure = vtk_to_numpy(dataset.GetCellData().GetArray("p"))
    velocity = vtk_to_numpy(dataset.GetCellData().GetArray("U"))
    order = np.argsort(cell_ids)
    cell_ids = cell_ids[order]
    if not np.array_equal(cell_ids, np.arange(cell_ids.size)):
        raise ValueError(f"Unexpected cellID ordering in {vtk_path}")
    centers = centers[order].reshape(-1, N_GAP, 3)
    pressure = pressure[order].reshape(-1, N_GAP)
    speed = np.linalg.norm(velocity[order], axis=1).reshape(-1, N_GAP)

    mid_centers = centers[:, 5:7].mean(axis=1)
    mid_pressure = pressure[:, 5:7].mean(axis=1)
    mid_speed = speed[:, 5:7].mean(axis=1)
    theta = np.degrees(np.arctan2(mid_centers[:, 1], mid_centers[:, 0]))
    theta = (theta + 180.0) % 360.0 - 180.0
    return {
        "theta": theta,
        "z_mm": mid_centers[:, 2] * 1e3,
        "p_abs_mpa": (P_ATM + RHO * mid_pressure) / 1e6,
        "speed": mid_speed,
    }


def plot_unwrapped_fields(
    vtk_dir: Path,
    metrics: dict[str, dict[str, float]],
    output: Path,
) -> None:
    field_data = {
        "496rpm": read_mid_gap(vtk_dir / "openfoam_oq90_single_phase_266.vtk"),
        "2000rpm": read_mid_gap(vtk_dir / "openfoam_oq90_single_phase_402.vtk"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), sharex=True, sharey=True)
    p_norm = TwoSlopeNorm(vmin=-17.5, vcenter=0, vmax=17.5)
    u_norm = Normalize(vmin=0, vmax=12.0)
    pressure_mesh = None
    velocity_mesh = None
    for row, key in enumerate(("496rpm", "2000rpm")):
        values = field_data[key]
        triangulation = mtri.Triangulation(values["theta"], values["z_mm"])
        pressure_mesh = axes[row, 0].tripcolor(
            triangulation,
            values["p_abs_mpa"],
            shading="gouraud",
            cmap="coolwarm",
            norm=p_norm,
            rasterized=True,
        )
        velocity_mesh = axes[row, 1].tripcolor(
            triangulation,
            values["speed"],
            shading="gouraud",
            cmap="viridis",
            norm=u_norm,
            rasterized=True,
        )
        label = STAGES[2 + row]["label"]
        state = "pabs < 0 · numerical only"
        axes[row, 0].set_title(f"{label}\nAbsolute pressure · {state}")
        axes[row, 1].set_title(f"{label}\nMid-film speed magnitude")
        for axis in axes[row]:
            axis.set_ylabel("Axial position z (mm)")
            axis.axvline(90, color="white", linewidth=0.8, alpha=0.7)
            axis.text(
                92,
                96,
                "feed meridian",
                color="white",
                fontsize=8,
                rotation=90,
                va="top",
            )
    for axis in axes[-1]:
        axis.set_xlabel("Bearing-frame angle (degrees)")
        axis.set_xlim(-180, 180)
        axis.set_xticks(np.arange(-180, 181, 60))
    assert pressure_mesh is not None
    assert velocity_mesh is not None
    figure.colorbar(
        pressure_mesh,
        ax=axes[:, 0],
        label="Absolute pressure (MPa)",
        fraction=0.025,
        pad=0.02,
    )
    figure.colorbar(
        velocity_mesh,
        ax=axes[:, 1],
        label="Speed magnitude (m/s)",
        fraction=0.025,
        pad=0.02,
    )
    figure.suptitle(
        "Unwrapped mid-film OpenFOAM fields · common scales across speeds",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Average of the two central cells through the 12-layer oil film. "
        "These are converged fully-filled numerical fields, not cavitating solutions.",
        ha="center",
        fontsize=9.5,
        color="#7c2432",
    )
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.08, top=0.89, wspace=0.18, hspace=0.28)
    save_figure(figure, output)


def plot_pressure_feed_field(vtk_dir: Path, output: Path) -> None:
    values = read_mid_gap(vtk_dir / "openfoam_oq90_single_phase_128.vtk")
    triangulation = mtri.Triangulation(values["theta"], values["z_mm"])
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), sharex=True, sharey=True)
    pressure_mesh = axes[0].tripcolor(
        triangulation,
        values["p_abs_mpa"],
        shading="gouraud",
        cmap="turbo",
        vmin=0.10,
        vmax=0.605,
        rasterized=True,
    )
    velocity_mesh = axes[1].tripcolor(
        triangulation,
        values["speed"],
        shading="gouraud",
        cmap="viridis",
        vmin=0,
        vmax=2.13,
        rasterized=True,
    )
    axes[0].set_title("Absolute pressure")
    axes[1].set_title("Mid-film speed magnitude")
    for axis in axes:
        axis.set(
            xlabel="Bearing-frame angle (degrees)",
            ylabel="Axial position z (mm)",
            xlim=(-180, 180),
        )
        axis.set_xticks(np.arange(-180, 181, 60))
        axis.axvline(90, color="white", linewidth=0.8, alpha=0.75)
    figure.colorbar(pressure_mesh, ax=axes[0], label="Absolute pressure (MPa)")
    figure.colorbar(velocity_mesh, ax=axes[1], label="Speed magnitude (m/s)")
    figure.suptitle(
        "Zero-speed 0.5 MPa pressure-fed equilibrium · physically admissible",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, output)


def pressure_surface_proxies(vtk_dir: Path, time: int):
    readers = []
    interpolators = []
    calculators = []
    for patch in ("stationary_wall", "pressure_feed"):
        path = vtk_dir / patch / f"{patch}_{time}.vtk"
        reader = LegacyVTKReader(FileNames=[str(path)])
        interpolator = CellDatatoPointData(Input=reader)
        interpolator.ProcessAllArrays = 1
        calculator = Calculator(Input=interpolator)
        calculator.AttributeType = "Point Data"
        calculator.ResultArrayName = "p_abs_MPa"
        calculator.Function = "(101325 + 860*p)/1000000"
        readers.append(reader)
        interpolators.append(interpolator)
        calculators.append(calculator)
    return readers, interpolators, calculators


def prepare_pressure_view(
    vtk_dir: Path,
    stage: dict[str, object],
    metrics: dict[str, float],
    *,
    size: tuple[int, int],
):
    _DisableFirstRenderCameraReset()
    readers, interpolators, calculators = pressure_surface_proxies(
        vtk_dir, int(stage["time"])
    )
    view = CreateView("RenderView")
    view.ViewSize = list(size)
    view.UseColorPaletteForBackground = 0
    view.Background = [0.965, 0.972, 0.982]
    view.OrientationAxesVisibility = 1
    lut = GetColorTransferFunction("p_abs_MPa")
    lut.ApplyPreset("Cool to Warm", True)
    for calculator in calculators:
        display = Show(calculator, view)
        display.Representation = "Surface"
        ColorBy(display, ("POINTS", "p_abs_MPa"))
        display.LookupTable = lut
        display.SetScalarBarVisibility(view, False)
    # ColorBy auto-rescales on every patch; set the shared comparison range last.
    lut.RescaleTransferFunction(-17.5, 17.5)
    scalar_bar = GetScalarBar(lut, view)
    scalar_bar.Title = "Absolute pressure"
    scalar_bar.ComponentTitle = "MPa"
    scalar_bar.TitleFontSize = 15
    scalar_bar.LabelFontSize = 12
    scalar_bar.TitleColor = [0.08, 0.11, 0.16]
    scalar_bar.LabelColor = [0.08, 0.11, 0.16]
    scalar_bar.ScalarBarLength = 0.34
    scalar_bar.WindowLocation = "Lower Right Corner"
    scalar_bar.Visibility = 1

    title = Text()
    physical = "PHYSICAL PASS" if stage["physical"] else "NUMERICAL ONLY · pabs < 0"
    title.Text = (
        f"{stage['label']}\n"
        f"{physical}\n"
        f"pabs = {metrics['pmin_abs']/1e6:.3f} to "
        f"{metrics['pmax_abs']/1e6:.3f} MPa"
    )
    title_display = Show(title, view)
    title_display.WindowLocation = "Upper Left Corner"
    title_display.FontSize = 20
    title_display.Color = [0.08, 0.11, 0.16] if stage["physical"] else [0.63, 0.08, 0.13]

    view.CameraFocalPoint = [0.0, 0.0, 0.05]
    view.CameraPosition = [0.0, -0.23, 0.14]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.083
    Render(view)
    return view, readers, interpolators, calculators, title, lut


def delete_pressure_view(view, readers, interpolators, calculators, title) -> None:
    Delete(title)
    for calculator in calculators:
        Delete(calculator)
    for interpolator in interpolators:
        Delete(interpolator)
    for reader in readers:
        Delete(reader)
    Delete(view)


def render_stage_frames(
    vtk_dir: Path,
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> list[Path]:
    frame_dir = output_dir / "frames_stage"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, stage in enumerate(STAGES):
        view, readers, interpolators, calculators, title, _ = prepare_pressure_view(
            vtk_dir, stage, metrics[str(stage["key"])], size=(1400, 900)
        )
        frame = frame_dir / f"stage_{index:02d}.png"
        SaveScreenshot(str(frame), view, ImageResolution=[1400, 900])
        frames.append(frame)
        delete_pressure_view(view, readers, interpolators, calculators, title)
    shutil.copyfile(frames[2], output_dir / "pressure_3d_496rpm.png")
    shutil.copyfile(frames[3], output_dir / "pressure_3d_2000rpm.png")
    return frames


def render_orbit(
    vtk_dir: Path,
    metrics: dict[str, float],
    output_dir: Path,
    frame_count: int = 48,
) -> list[Path]:
    stage = STAGES[-1]
    frame_dir = output_dir / "frames_orbit_2000rpm"
    frame_dir.mkdir(parents=True, exist_ok=True)
    view, readers, interpolators, calculators, title, _ = prepare_pressure_view(
        vtk_dir, stage, metrics, size=(1280, 720)
    )
    frames: list[Path] = []
    radius = 0.23
    for index in range(frame_count):
        angle = -math.pi / 2 + 2 * math.pi * index / frame_count
        view.CameraPosition = [
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.14,
        ]
        Render(view)
        frame = frame_dir / f"orbit_{index:03d}.png"
        SaveScreenshot(str(frame), view, ImageResolution=[1280, 720])
        frames.append(frame)
    delete_pressure_view(view, readers, interpolators, calculators, title)
    return frames


def run_ffmpeg(arguments: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments],
        check=True,
    )


def stage_video(frames: list[Path], output: Path) -> None:
    concat = output.with_suffix(".concat.txt")
    lines: list[str] = []
    for frame in frames:
        lines.extend((f"file '{frame.resolve()}'", "duration 2.0"))
    lines.append(f"file '{frames[-1].resolve()}'")
    concat.write_text("\n".join(lines) + "\n")
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vf",
            "fps=24,format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def orbit_video(frames: list[Path], output: Path) -> None:
    run_ffmpeg(
        [
            "-framerate",
            "12",
            "-start_number",
            "0",
            "-i",
            str(frames[0].parent / "orbit_%03d.png"),
            "-frames:v",
            str(len(frames)),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def animated_gif(frames: list[Path], output: Path, duration_ms: int) -> None:
    images: list[Image.Image] = []
    for frame in frames:
        image = Image.open(frame).convert("RGB")
        image.thumbnail((960, 620), Image.Resampling.LANCZOS)
        images.append(image)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    for image in images:
        image.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=Path,
        default=REPO_ROOT / "out/openfoam_oq90_single_phase",
        help="OpenFOAM case containing postProcessing and VTK",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/conical_journal/studies/single-phase-oq90"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_args(values)
    case = arguments.case.resolve()
    target = arguments.output.resolve()
    vtk_dir = case / "VTK"
    required_vtk = [
        vtk_dir / f"openfoam_oq90_single_phase_{time}.vtk"
        for time in (1, 128, 266, 402)
    ]
    missing = [path for path in required_vtk if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing foamToVTK exports:\n" + "\n".join(str(path) for path in missing)
        )
    output = make_staging_directory(target)
    try:
        style()
        histories = {
            str(stage["key"]): stage_histories(case, stage) for stage in STAGES
        }
        metrics = {
            key: final_metrics(stage_history)
            for key, stage_history in histories.items()
        }
        plot_run_summary(histories, metrics, output / "run_summary.png")
        plot_convergence(
            STAGES[2],
            histories["496rpm"],
            metrics["496rpm"],
            output / "convergence_496rpm.png",
        )
        plot_convergence(
            STAGES[3],
            histories["2000rpm"],
            metrics["2000rpm"],
            output / "convergence_2000rpm.png",
        )
        plot_unwrapped_fields(
            vtk_dir, metrics, output / "unwrapped_rotating_fields.png"
        )
        plot_pressure_feed_field(vtk_dir, output / "pressure_feed_0rpm_fields.png")

        stage_frames = render_stage_frames(vtk_dir, metrics, output)
        orbit_frames = render_orbit(vtk_dir, metrics["2000rpm"], output)
        stage_video(stage_frames, output / "pressure_stage_sweep.mp4")
        orbit_video(orbit_frames, output / "pressure_orbit_2000rpm.mp4")
        animated_gif(stage_frames, output / "pressure_stage_sweep.gif", 2000)
        animated_gif(orbit_frames, output / "pressure_orbit_2000rpm.gif", 85)

        upstream = list(required_vtk)
        for stage in STAGES:
            start = str(stage["start"])
            upstream.extend(
                (
                    case / str(stage["log"]),
                    case / "postProcessing/maxP" / start / "volFieldValue.dat",
                    case / "postProcessing/minP" / start / "volFieldValue.dat",
                    case / "postProcessing/maxU" / start / "volFieldValue.dat",
                    case
                    / "postProcessing/feedFlowRate"
                    / start
                    / "surfaceFieldValue.dat",
                    case
                    / "postProcessing/z0FlowRate"
                    / start
                    / "surfaceFieldValue.dat",
                    case
                    / "postProcessing/zLFlowRate"
                    / start
                    / "surfaceFieldValue.dat",
                    case
                    / "postProcessing/netBoundaryFlow"
                    / start
                    / "surfaceFieldValue.dat",
                    case / "postProcessing/journalForces" / start / "forces.dat",
                )
            )
        publish_generation(
            output,
            target,
            stage="study",
            operation="single-phase-oq90",
            status="RENDERED",
            resolved_inputs={"case": case, "stages": STAGES},
            input_units={
                "pressure": "Pa absolute",
                "velocity": "m/s",
                "flow": "m^3/s",
                "speed": "rpm",
            },
            producer_files=(Path(__file__),),
            upstream_artifacts=tuple(upstream),
            tool_versions={
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "ffmpeg": shutil.which("ffmpeg") or "not found",
            },
            argv=values,
            acceptance_status="EVIDENCE_ONLY",
            repository=REPO_ROOT,
        )
    finally:
        if output.exists():
            shutil.rmtree(output)

    print(f"Generated visualization package in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
