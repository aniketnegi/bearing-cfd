#!/usr/bin/env python3
"""Plot and animate the stored native OpenFOAM JFO checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


# ponytail: OpenFOAM time dirs do not retain stage rpm; update this accepted
# RUN_RESULTS.md mapping when a new continuation is retained. Time 15.639... is
# the superseded 477.465 rpm run; 17.002... is the corrected 2.6 m/s point.
CHECKPOINTS = (
    (0.0, "6"),
    (20.0, "14.90625"),
    (496.563, "17.002574820373"),
    (1000.0, "16.061490272848"),
    (2000.0, "17.261090445375"),
)
CAVITY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Checkpoint:
    rpm: float
    time: str
    pressure_gauge_mpa: np.ndarray
    fill: np.ndarray
    area_weight: np.ndarray

    @property
    def pressure_max_mpa(self) -> float:
        return float(self.pressure_gauge_mpa.max())

    @property
    def fill_min(self) -> float:
        return float(self.fill.min())

    @property
    def cavity_area_percent(self) -> float:
        cavity = self.fill < 1.0 - CAVITY_TOLERANCE
        return float(100.0 * self.area_weight[cavity].sum() / self.area_weight.sum())


def _scalar(dictionary: Path, name: str) -> float:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s+(?:\[[^\]]+\]\s+)?([-+0-9.eE]+)\s*;",
        dictionary.read_text(encoding="utf-8"),
    )
    if not match:
        raise ValueError(f"{name!r} not found in {dictionary}")
    return float(match.group(1))


def _mesh_shape(block_mesh: Path) -> tuple[int, int]:
    match = re.search(
        r"hex\s*\([^)]+\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)",
        block_mesh.read_text(encoding="utf-8"),
    )
    if not match:
        raise ValueError(f"cell counts not found in {block_mesh}")
    n_theta, n_thickness, n_axial = map(int, match.groups())
    if n_thickness != 1:
        raise ValueError("this thin-film plotter requires one pseudo-thickness cell")
    return n_axial, n_theta


def _field(path: Path, shape: tuple[int, int]) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform:
        return np.full(shape, float(uniform.group(1)))
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*"
        r"\(\s*(.*?)\s*\)\s*;",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"nonuniform scalar internalField not found in {path}")
    count = int(match.group(1))
    values = np.fromstring(match.group(2), sep=" ")
    if count != values.size or count != shape[0] * shape[1]:
        raise ValueError(
            f"{path}: declared {count}, read {values.size}, expected {shape[0] * shape[1]}"
        )
    return values.reshape(shape)


def load_checkpoints(case: Path) -> tuple[list[Checkpoint], float]:
    shape = _mesh_shape(case / "system/blockMeshDict")
    properties = case / "constant/jfoProperties"
    ambient = _scalar(properties, "ambientPressure")
    rupture = _scalar(properties, "cavitationPressure")
    checkpoints: list[Checkpoint] = []
    for rpm, time_name in CHECKPOINTS:
        time = case / time_name
        pressure_gauge_mpa = (
            _field(time / "p", shape) + rupture - ambient
        ) / 1e6
        fill = _field(time / "thetaFill", shape)
        area_weight = _field(time / "surfaceMetric", shape)
        if pressure_gauge_mpa.min() < -1e-9:
            raise ValueError(f"{time}: gauge pressure fell below the declared floor")
        if fill.min() < -1e-9 or fill.max() > 1.0 + 1e-9:
            raise ValueError(f"{time}: thetaFill is outside [0, 1]")
        checkpoints.append(
            Checkpoint(rpm, time_name, pressure_gauge_mpa, fill, area_weight)
        )
    return checkpoints, rupture


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def write_metrics(checkpoints: list[Checkpoint], output: Path) -> None:
    with (output / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "rpm",
                "openfoam_time",
                "max_gauge_pressure_mpa",
                "minimum_fill_fraction",
                "cavitated_area_percent",
            )
        )
        for state in checkpoints:
            writer.writerow(
                (
                    f"{state.rpm:g}",
                    state.time,
                    f"{state.pressure_max_mpa:.9g}",
                    f"{state.fill_min:.9g}",
                    f"{state.cavity_area_percent:.9g}",
                )
            )


def plot_speed_sweep(checkpoints: list[Checkpoint], output: Path) -> None:
    rpm = np.array([state.rpm for state in checkpoints])
    series = (
        (
            np.array([state.pressure_max_mpa for state in checkpoints]),
            "Maximum gauge pressure (MPa)",
            "#0072B2",
        ),
        (
            np.array([state.cavity_area_percent for state in checkpoints]),
            "Cavitated surface area (%)",
            "#D55E00",
        ),
        (
            np.array([state.fill_min for state in checkpoints]),
            r"Minimum liquid fill fraction, $\theta_\mathrm{fill}$",
            "#009E73",
        ),
    )
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    for axis, (values, label, color) in zip(axes, series, strict=True):
        axis.plot(rpm, values, "o-", color=color, linewidth=1.8, markersize=5)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        for index, (x, y) in enumerate(zip(rpm, values, strict=True)):
            x_offset = -6 if index == 0 else 6 if index == 1 else 0
            alignment = "right" if index == 0 else "left" if index == 1 else "center"
            axis.annotate(
                f"{y:.3g}",
                (x, y),
                xytext=(x_offset, 6),
                textcoords="offset points",
                ha=alignment,
                fontsize=8,
            )
    axes[-1].set_xlabel("Journal speed (rpm)")
    axes[-1].set_xscale("symlog", linthresh=20)
    axes[-1].set_xticks(rpm)
    axes[-1].set_xticklabels(("0", "20", "496.563", "1000", "2000"))
    figure.suptitle("Native OpenFOAM JFO: converged speed checkpoints")
    figure.text(
        0.5,
        0.005,
        "Current thin-film Reynolds/JFO implementation; independent validation pending.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    _save(figure, output / "speed_sweep")


def plot_final_fields(state: Checkpoint, rupture_pressure_pa: float, output: Path) -> None:
    extent = (0.0, 360.0, 0.0, 100.0)
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    pressure = axes[0].imshow(
        state.pressure_gauge_mpa,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=state.pressure_max_mpa,
        interpolation="nearest",
    )
    figure.colorbar(pressure, ax=axes[0], label="Gauge pressure (MPa)")
    axes[0].set_title(
        f"Pressure: max {state.pressure_max_mpa:.4f} MPa gauge; "
        f"absolute floor {rupture_pressure_pa / 1e6:.6f} MPa"
    )

    fill = axes[1].imshow(
        state.fill,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    figure.colorbar(fill, ax=axes[1], label=r"Liquid fill fraction, $\theta_\mathrm{fill}$")
    axes[1].set_title(f"Fill fraction: minimum {state.fill_min:.6f}")

    cavity = state.fill < 1.0 - CAVITY_TOLERANCE
    cavity_plot = axes[2].imshow(
        cavity,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=colors.ListedColormap(("#306998", "#E66101")),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    colorbar = figure.colorbar(cavity_plot, ax=axes[2], ticks=(0.25, 0.75))
    colorbar.ax.set_yticklabels(("Full film", "Cavitated"))
    axes[2].set_title(
        f"JFO rupture mask: {state.cavity_area_percent:.3f}% of surface area"
    )
    axes[2].set_xlabel(r"Circumferential angle, $\theta$ (degrees)")
    for axis in axes:
        axis.set_ylabel("Axial position, z (mm)")
        axis.set_xticks(np.arange(0, 361, 45))
    figure.suptitle("Native OpenFOAM JFO — converged 2000 rpm fields")
    figure.text(
        0.5,
        0.005,
        "Thin-film Reynolds/JFO result; not a 3-D multiphase Fluent solution.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    _save(figure, output / "final_fields_2000rpm")


def plot_fill_severity(state: Checkpoint, output: Path) -> None:
    thresholds = np.linspace(0.0, 1.0, 401)
    total_area = state.area_weight.sum()
    area_below = np.array(
        [
            100.0 * state.area_weight[state.fill < threshold].sum() / total_area
            for threshold in thresholds
        ]
    )
    cavity_area = state.cavity_area_percent
    area_below_half = float(
        100.0 * state.area_weight[state.fill < 0.5].sum() / total_area
    )
    full_film_area = 100.0 - cavity_area

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(thresholds, area_below, color="#D55E00", linewidth=2)
    axes[0].axhline(
        cavity_area,
        color="#A50F15",
        linestyle="--",
        linewidth=1.2,
        label=f"JFO rupture footprint: {cavity_area:.3f}%",
    )
    axes[0].plot(0.5, area_below_half, "o", color="#0072B2")
    axes[0].annotate(
        f"{area_below_half:.3f}% below 0.5",
        (0.5, area_below_half),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].set(
        xlabel=r"Fill threshold, $\theta_\mathrm{fill}$",
        ylabel="Surface area below threshold (%)",
        xlim=(0.2, 1.0),
        ylim=(0.0, 70.0),
        title="Area-weighted fill severity",
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right", frameon=True)

    weights = 100.0 * state.area_weight.ravel() / total_area
    axes[1].hist(
        state.fill.ravel(),
        bins=np.linspace(0.25, 1.0, 31),
        weights=weights,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].axvline(
        state.fill_min,
        color="#A50F15",
        linestyle="--",
        linewidth=1.2,
        label=f"minimum: {state.fill_min:.6f}",
    )
    axes[1].annotate(
        f"Exact full film\n{full_film_area:.3f}%",
        (0.998, full_film_area),
        xytext=(-8, -8),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
    )
    axes[1].set(
        xlabel=r"Liquid fill fraction, $\theta_\mathrm{fill}$",
        ylabel="Surface area per bin (%)",
        xlim=(0.24, 1.01),
        title="Area-weighted fill distribution",
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="upper left", frameon=True)

    figure.suptitle("Native OpenFOAM JFO — 2000 rpm cavitation severity")
    figure.text(
        0.5,
        0.005,
        "Current model diagnostic: rupture means $\\theta_\\mathrm{fill}<1$; "
        "external validation pending.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.94))
    _save(figure, output / "fill_severity_2000rpm")


def animate_checkpoints(checkpoints: list[Checkpoint], output: Path) -> None:
    extent = (0.0, 360.0, 0.0, 100.0)
    pressure_max = max(state.pressure_max_mpa for state in checkpoints)
    fill_min = min(state.fill_min for state in checkpoints)
    pressure_norm = colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=pressure_max)
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 6.6), sharex=True)
    pressure = axes[0].imshow(
        checkpoints[0].pressure_gauge_mpa,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        norm=pressure_norm,
        interpolation="nearest",
        animated=True,
    )
    figure.colorbar(
        pressure,
        ax=axes[0],
        label="Gauge pressure (MPa; fixed power-scaled colors)",
    )
    fill = axes[1].imshow(
        checkpoints[0].fill,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=fill_min,
        vmax=1.0,
        interpolation="nearest",
        animated=True,
    )
    figure.colorbar(fill, ax=axes[1], label=r"Liquid fill fraction, $\theta_\mathrm{fill}$")
    axes[1].set_xlabel(r"Circumferential angle, $\theta$ (degrees)")
    for axis in axes:
        axis.set_ylabel("Axial position, z (mm)")
        axis.set_xticks(np.arange(0, 361, 45))
    title = figure.suptitle("")
    note = figure.text(
        0.5,
        0.005,
        "Five converged checkpoints; frames are not temporal interpolation.",
        ha="center",
        fontsize=8,
    )

    def update(frame: int) -> tuple[object, ...]:
        state = checkpoints[frame]
        pressure.set_data(state.pressure_gauge_mpa)
        fill.set_data(state.fill)
        axes[0].set_title(f"Gauge pressure; max {state.pressure_max_mpa:.4f} MPa")
        axes[1].set_title(
            f"Fill fraction; min {state.fill_min:.6f}, "
            f"cavitated area {state.cavity_area_percent:.3f}%"
        )
        title.set_text(f"Native OpenFOAM JFO — {state.rpm:g} rpm")
        return pressure, fill, title, note

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(checkpoints),
        interval=1200,
        repeat=True,
        blit=False,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    movie.save(
        output / "converged_checkpoints.mp4",
        writer=animation.FFMpegWriter(
            fps=1,
            codec="libx264",
            bitrate=2500,
            extra_args=("-pix_fmt", "yuv420p"),
        ),
        dpi=160,
    )
    movie.save(
        output / "converged_checkpoints.gif",
        writer=animation.PillowWriter(fps=1),
        dpi=100,
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("out/openfoam_jfo_native_256x80"),
        help="native OpenFOAM JFO case containing the stored checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/handoff/media/jfo_candidate"),
        help="directory for figures, animation, and checkpoint CSV",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()
    checkpoints, rupture_pressure_pa = load_checkpoints(args.case)
    write_metrics(checkpoints, args.output)
    plot_speed_sweep(checkpoints, args.output)
    plot_final_fields(checkpoints[-1], rupture_pressure_pa, args.output)
    plot_fill_severity(checkpoints[-1], args.output)
    animate_checkpoints(checkpoints, args.output)
    print(f"wrote JFO figures and animation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
