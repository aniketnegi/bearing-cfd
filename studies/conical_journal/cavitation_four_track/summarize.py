"""Create the four-track cavitation result ledger and comparison figures."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bearing_cfd.artifacts import make_staging_directory, publish_generation


REPO_ROOT = Path(__file__).resolve().parents[3]


def read_series(case: Path, function: str) -> dict[float, float]:
    values: dict[float, float] = {}
    paths = sorted(
        (case / "postProcessing" / function).glob("*/*.dat"),
        key=lambda path: float(path.parent.name),
    )
    for path in paths:
        for line in path.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            values[float(fields[0])] = float(fields[1])
    return dict(sorted(values.items()))


def aligned(*series: dict[float, float]) -> tuple[np.ndarray, ...]:
    times = sorted(set.intersection(*(set(item) for item in series)))
    return (
        np.asarray(times),
        *(np.asarray([item[time] for time in times]) for item in series),
    )


def load_jfo(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def save_figure(fig: plt.Figure, stem: str, output: Path) -> None:
    fig.savefig(output / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(
    vapour: Path, gas: Path, ventilation: Path, output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    t, amin, amean, net, feed = aligned(
        read_series(vapour, "minAlphaOil"),
        read_series(vapour, "meanAlphaOil"),
        read_series(vapour, "netBoundaryMassFlow"),
        read_series(vapour, "feedMassFlow"),
    )
    mask = t >= 100
    imbalance = 100 * np.abs(net) / np.abs(feed)
    axes[0].plot(t[mask], amin[mask], label=r"$\alpha_{oil,min}$", lw=2)
    axes[0].plot(t[mask], amean[mask], label=r"$\bar{\alpha}_{oil}$", lw=2)
    axes[0].axvline(116, color="#d97706", ls="--", label="onset ~24.4 rpm")
    axes[0].axvspan(130, 145, color="#ef4444", alpha=0.12, label="28 rpm hold")
    axes[0].set(title="B · oil vapour", xlabel="pseudo-step", ylabel="oil fraction", ylim=(0, 1.03))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    ax0b = axes[0].twinx()
    ax0b.plot(t[mask], imbalance[mask], color="#7c3aed", alpha=0.65)
    ax0b.set_ylabel("boundary mass imbalance [%]", color="#7c3aed")

    t, amin, amean, net, feed = aligned(
        read_series(gas, "minAlphaOil"),
        read_series(gas, "meanAlphaOil"),
        read_series(gas, "netBoundaryMassFlow"),
        read_series(gas, "feedMassFlow"),
    )
    imbalance = 100 * np.abs(net) / np.abs(feed)
    mask = t >= 10
    axes[1].plot(t[mask], amin[mask], label=r"$\alpha_{oil,min}$", lw=2)
    axes[1].plot(t[mask], amean[mask], label=r"$\bar{\alpha}_{oil}$", lw=2)
    axes[1].set(title="C · non-condensable gas", xlabel="pseudo-step", ylabel="oil fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    ax1b = axes[1].twinx()
    ax1b.plot(t[mask], imbalance[mask], color="#dc2626", lw=1.8, label="mass imbalance")
    ax1b.axhline(0.5, color="#dc2626", ls="--", alpha=0.55)
    ax1b.set_ylabel("boundary mass imbalance [%]", color="#dc2626")

    t, amin, pmin, pmax = aligned(
        read_series(ventilation, "minAlphaOil"),
        read_series(ventilation, "minPressure"),
        read_series(ventilation, "maxPressure"),
    )
    axes[2].plot(t * 1e6, amin, color="#0f766e", lw=2)
    axes[2].set(
        title="D · atmospheric ventilation",
        xlabel=r"physical startup time [$\mu$s]",
        ylabel=r"$\alpha_{oil,min}$",
        ylim=(0.99, 1.0005),
    )
    axes[2].grid(alpha=0.25)
    ax2b = axes[2].twinx()
    ax2b.plot(t * 1e6, pmin / 1e6, color="#dc2626", label=r"$p_{min}$")
    ax2b.plot(t * 1e6, pmax / 1e6, color="#2563eb", label=r"$p_{max}$")
    ax2b.axhline(0, color="black", lw=0.8)
    ax2b.set_ylabel("absolute pressure [MPa]")
    ax2b.legend(fontsize=8, loc="lower left")

    fig.suptitle("3-D screening diagnostics — none passed a steady quantitative gate", fontsize=15)
    save_figure(fig, "screening_diagnostics", output)


def plot_status(output: Path) -> None:
    columns = ["Track", "Mechanism", "Published surrogate", "Highest state", "Gate result", "Status"]
    rows = [
        ["A", "Reynolds–JFO", "SAE 10W-40", "4000 rpm", "converged + cross-code parity", "ACCEPTED NUMERICAL"],
        ["B", "oil vapour", "SAE 10W-40", "28 rpm hold", "phase field still drifting", "UNSETTLED"],
        ["C", "non-condensable gas", "ISO VG32", "0 rpm, step 40", "2.04% mass imbalance", "UNSETTLED"],
        ["D", "atmospheric ventilation", "VG22", "3500 rpm, 4.37 us", "p_abs fell below zero", "REJECTED"],
    ]
    fig, ax = plt.subplots(figsize=(16, 3.3))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    widths = [0.05, 0.16, 0.17, 0.15, 0.27, 0.20]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(widths[col])
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif col == 5:
            cell.set_facecolor(["#dcfce7", "#fef3c7", "#fef3c7", "#fee2e2"][row - 1])
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("Four mechanisms are separate models, not four settings of one model", fontsize=15, pad=16)
    save_figure(fig, "four_track_status", output)


def plot_paper_and_speed(jfo_metrics: Path, output: Path) -> None:
    jfo = load_jfo(jfo_metrics)
    rpm = np.asarray([row["rpm"] for row in jfo])
    pressure = np.asarray([row["max_gauge_pressure_mpa"] for row in jfo])
    cavity = np.asarray([row["cavitated_area_percent"] for row in jfo])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    labels = ["paper FEA\n(graph read)", "paper Fluent\n(graph read)", "paper-input JFO", "named SAE\nsurrogate"]
    values = [35.6, 32.0, 34.8411, pressure[rpm == 2000][0] / 0.5]
    colors = ["#2563eb", "#60a5fa", "#16a34a", "#f59e0b"]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel(r"$p_{max,gauge}/p_s$ at 2000 rpm")
    axes[0].set_title("Narrow paper comparison")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(
        3,
        values[3] + 1,
        "not comparable:\nviscosity changed",
        ha="center",
        fontsize=8,
        color="#92400e",
    )

    ax = axes[1]
    ax.plot(rpm, pressure, marker="o", lw=2, color="#2563eb", label=r"$p_{max,gauge}$")
    ax.set(xlabel="speed [rpm]", ylabel="maximum gauge pressure [MPa]", title="Named SAE 10W-40 JFO sensitivity")
    ax.axvspan(2000, 4100, color="#f59e0b", alpha=0.13)
    ax.text(3000, pressure.max() * 0.18, "extrapolative", ha="center", color="#92400e")
    ax.grid(alpha=0.25)
    axb = ax.twinx()
    axb.plot(rpm, cavity, marker="s", lw=1.8, color="#dc2626", label="ruptured area")
    axb.set_ylabel("ruptured area [%]", color="#dc2626")
    save_figure(fig, "paper_comparison_and_higher_rpm", output)


def write_summary_csv(
    vapour: Path,
    gas: Path,
    ventilation: Path,
    jfo_metrics: Path,
    output: Path,
) -> None:
    jfo = load_jfo(jfo_metrics)[-1]
    b_min = read_series(vapour, "minAlphaOil")
    b_mean = read_series(vapour, "meanAlphaOil")
    b_net = read_series(vapour, "netBoundaryMassFlow")
    b_feed = read_series(vapour, "feedMassFlow")
    c_min = read_series(gas, "minAlphaOil")
    c_mean = read_series(gas, "meanAlphaOil")
    c_net = read_series(gas, "netBoundaryMassFlow")
    c_feed = read_series(gas, "feedMassFlow")
    d_min = read_series(ventilation, "minAlphaOil")
    d_pmin = read_series(ventilation, "minPressure")
    d_pmax = read_series(ventilation, "maxPressure")
    rows = [
        ["A", "Reynolds-JFO", "accepted numerical", 4000, jfo["minimum_fill_fraction"], "", "", ""],
        [
            "B",
            "oil vapour",
            "unsettled",
            28,
            b_min[145],
            b_mean[145],
            100 * abs(b_net[145] / b_feed[145]),
            "pressure pinned at pSat; phase field not plateaued",
        ],
        [
            "C",
            "non-condensable gas",
            "unsettled",
            0,
            c_min[40],
            c_mean[40],
            100 * abs(c_net[40] / c_feed[40]),
            "zero-speed mass gate failed",
        ],
        [
            "D",
            "atmospheric ventilation",
            "rejected startup",
            3500,
            d_min[max(d_min)],
            "",
            "",
            f"worst p_abs={min(d_pmin.values()):.6g} Pa; max={max(d_pmax.values()):.6g} Pa",
        ],
    ]
    with (output / "mechanism_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["track", "mechanism", "status", "highest_rpm", "alpha_oil_min", "alpha_oil_mean", "mass_imbalance_percent", "note"]
        )
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vapour-case",
        type=Path,
        default=REPO_ROOT
        / "out/archive/conical_journal/simulation/openfoam/multiphase/openfoam_oq90_vapour_sae10w40_schnerr_screen",
    )
    parser.add_argument(
        "--gas-case",
        type=Path,
        default=REPO_ROOT
        / "out/archive/conical_journal/simulation/openfoam/multiphase/openfoam_oq90_pseudocavitation_iso_vg32_air",
    )
    parser.add_argument(
        "--ventilation-case",
        type=Path,
        default=REPO_ROOT
        / "out/archive/conical_journal/simulation/openfoam/multiphase/openfoam_oq90_ventilation_vg22_air_boundary_screen",
    )
    parser.add_argument(
        "--jfo-metrics",
        type=Path,
        default=(
            REPO_ROOT
            / "studies/conical_journal/cavitation_four_track/inputs"
            / "track_a_jfo/checkpoint_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/conical_journal/studies/cavitation-four-track"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_args(values)
    vapour = arguments.vapour_case.resolve()
    gas = arguments.gas_case.resolve()
    ventilation = arguments.ventilation_case.resolve()
    jfo_metrics = arguments.jfo_metrics.resolve()
    target = arguments.output.resolve()
    output = make_staging_directory(target)
    try:
        plot_diagnostics(vapour, gas, ventilation, output)
        plot_status(output)
        plot_paper_and_speed(jfo_metrics, output)
        write_summary_csv(vapour, gas, ventilation, jfo_metrics, output)
        sources = [jfo_metrics]
        for case, functions in (
            (
                vapour,
                ("minAlphaOil", "meanAlphaOil", "netBoundaryMassFlow", "feedMassFlow"),
            ),
            (
                gas,
                ("minAlphaOil", "meanAlphaOil", "netBoundaryMassFlow", "feedMassFlow"),
            ),
            (ventilation, ("minAlphaOil", "minPressure", "maxPressure")),
        ):
            for function in functions:
                sources.extend((case / "postProcessing" / function).glob("*/*.dat"))
        publish_generation(
            output,
            target,
            stage="study",
            operation="cavitation-four-track",
            status="RENDERED",
            resolved_inputs={
                "vapour_case": vapour,
                "gas_case": gas,
                "ventilation_case": ventilation,
                "jfo_metrics": jfo_metrics,
            },
            input_units={
                "pressure": "Pa absolute",
                "mass_flow": "kg/s",
                "speed": "rpm",
            },
            producer_files=(Path(__file__),),
            upstream_artifacts=tuple(sorted(sources)),
            tool_versions={
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
            },
            argv=values,
            acceptance_status="MIXED_RESEARCH_STATUS",
            repository=REPO_ROOT,
        )
    finally:
        if output.exists():
            shutil.rmtree(output)
    print(f"wrote four-track evidence to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
