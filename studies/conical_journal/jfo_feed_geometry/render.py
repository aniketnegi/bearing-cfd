#!/usr/bin/env python3
"""Compare the old stair-stepped JFO feed with the body-fitted O-grid."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import meshio
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from bearing_cfd.artifacts import make_staging_directory, publish_generation
from bearing_cfd.bearings.conical_journal.simulation import jfo_surface_mesh
from bearing_cfd.bearings.conical_journal.meshing import no_port as no_port
from bearing_cfd.bearings.conical_journal.meshing.surface_inlet import InletSpec
from bearing_cfd.bearings.conical_journal.simulation.reynolds_jfo import Inputs, make_grid


REPO_ROOT = Path(__file__).resolve().parents[3]


def _paper_geometry(inputs: Inputs) -> tuple[no_port.BearingParams, InletSpec]:
    angle = math.radians(inputs.eccentricity_angle_deg)
    return (
        no_port.BearingParams(
            source=Path("paper-exact"),
            source_sha256="visualization",
            length_mm=inputs.length_m * 1.0e3,
            mean_radius_mm=inputs.mean_radius_m * 1.0e3,
            semicone_angle_deg=inputs.semicone_angle_deg,
            radial_clearance_mm=inputs.radial_clearance_m * 1.0e3,
            eccentricity_ratio=(
                inputs.eccentricity_m / inputs.radial_clearance_m
            ),
            eccentricity_mm=inputs.eccentricity_m * 1.0e3,
            ex_mm=inputs.eccentricity_m * math.cos(angle) * 1.0e3,
            ey_mm=inputs.eccentricity_m * math.sin(angle) * 1.0e3,
            cone_slope=math.tan(
                math.radians(inputs.semicone_angle_deg)
            ),
        ),
        InletSpec(
            axial_position_mm=inputs.feed_axial_position_m * 1.0e3,
            diameter_mm=inputs.feed_diameter_m * 1.0e3,
            radius_mm=inputs.feed_diameter_m * 0.5e3,
        ),
    )


def _old_polygons(
    inputs: Inputs,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    grid = make_grid(inputs)
    gamma = math.radians(inputs.semicone_angle_deg)
    dtheta = 2.0 * math.pi / inputs.n_theta
    dz = inputs.length_m / inputs.n_axial
    polygons: list[np.ndarray] = []
    selected: list[bool] = []
    for axial in range(inputs.n_axial):
        for circumferential in range(inputs.n_theta):
            theta0 = circumferential * dtheta
            theta1 = theta0 + dtheta
            z0 = axial * dz
            z1 = z0 + dz
            polygon = np.asarray(
                [
                    (
                        inputs.mean_radius_m * (theta0 - math.pi) * 1.0e3,
                        (z0 - inputs.feed_axial_position_m)
                        / math.cos(gamma)
                        * 1.0e3,
                    ),
                    (
                        inputs.mean_radius_m * (theta1 - math.pi) * 1.0e3,
                        (z0 - inputs.feed_axial_position_m)
                        / math.cos(gamma)
                        * 1.0e3,
                    ),
                    (
                        inputs.mean_radius_m * (theta1 - math.pi) * 1.0e3,
                        (z1 - inputs.feed_axial_position_m)
                        / math.cos(gamma)
                        * 1.0e3,
                    ),
                    (
                        inputs.mean_radius_m * (theta0 - math.pi) * 1.0e3,
                        (z1 - inputs.feed_axial_position_m)
                        / math.cos(gamma)
                        * 1.0e3,
                    ),
                ]
            )
            if np.all(np.abs(polygon.mean(axis=0)) <= 6.5):
                polygons.append(polygon)
                selected.append(bool(grid.feed[axial, circumferential]))
    return (
        polygons,
        np.asarray(selected),
        float(grid.area[grid.feed].sum() * 1.0e6),
    )


def _new_polygons(
    inputs: Inputs,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    params, inlet = _paper_geometry(inputs)
    mesh, feed = jfo_surface_mesh.build_mesh(
        params,
        inlet,
        n_theta=inputs.n_theta,
        n_axial=inputs.n_axial,
        rim_segments=32,
    )
    front = mesh.hexes[:, [0, 1, 5, 4]].astype(np.int64) - 1
    vertices = mesh.points_m[front]
    gamma = math.radians(inputs.semicone_angle_deg)
    polygons: list[np.ndarray] = []
    selected: list[bool] = []
    for cell, points in enumerate(vertices):
        polygon = np.column_stack(
            (
                (
                    points[:, 0]
                    - math.pi * inputs.mean_radius_m
                )
                * 1.0e3,
                (
                    points[:, 2] - inputs.feed_axial_position_m
                )
                / math.cos(gamma)
                * 1.0e3,
            )
        )
        if np.all(np.abs(polygon.mean(axis=0)) <= 6.5):
            polygons.append(polygon)
            selected.append(bool(feed[cell]))
    return (
        polygons,
        np.asarray(selected),
        float(mesh.metadata["feed_area_m2"] * 1.0e6),
    )


def _draw(
    axis: plt.Axes,
    polygons: list[np.ndarray],
    selected: np.ndarray,
    title: str,
    area_mm2: float,
) -> None:
    colors = np.where(selected, "#e76f51", "#f6f7f8")
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors=colors,
            edgecolors="#667085",
            linewidths=0.45,
        )
    )
    angle = np.linspace(0.0, 2.0 * math.pi, 500)
    axis.plot(
        2.0 * np.cos(angle),
        2.0 * np.sin(angle),
        color="#111827",
        linewidth=2.0,
        label="intended 4 mm rim",
    )
    axis.set(
        aspect="equal",
        xlim=(-6, 6),
        ylim=(-6, 6),
        xlabel="circumferential distance from feed centre [mm]",
        ylabel="conical axial distance from feed centre [mm]",
        title=f"{title}\nselected feed area = {area_mm2:.3f} mm²",
    )
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right")


def _accepted_mesh(
    inputs: Inputs,
) -> tuple[
    no_port.MeshData,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    params, inlet = _paper_geometry(inputs)
    mesh, feed = jfo_surface_mesh.build_mesh(
        params,
        inlet,
        n_theta=inputs.n_theta,
        n_axial=inputs.n_axial,
        rim_segments=32,
    )
    quads = mesh.hexes[:, [0, 1, 5, 4]].astype(np.int64) - 1
    point_count = len(mesh.points_m) // 2
    planar_m = mesh.points_m[:point_count][:, [0, 2]]
    theta = planar_m[:, 0] / inputs.mean_radius_m
    z_m = planar_m[:, 1]
    gamma = math.radians(inputs.semicone_angle_deg)
    journal_radius = (
        inputs.mean_radius_m
        + (inputs.length_m / 2.0 - z_m) * math.tan(gamma)
    )
    angle = math.radians(inputs.eccentricity_angle_deg)
    ex = inputs.eccentricity_m * math.cos(angle)
    ey = inputs.eccentricity_m * math.sin(angle)
    q = ex * np.sin(theta) - ey * np.cos(theta)
    journal_ray = q + np.sqrt(
        journal_radius**2 - inputs.eccentricity_m**2 + q**2
    )
    surface_radius = 0.5 * (
        journal_radius + inputs.radial_clearance_m + journal_ray
    )
    curved_m = np.column_stack(
        (
            surface_radius * np.sin(theta),
            -surface_radius * np.cos(theta),
            z_m,
        )
    )
    transition = (mesh.logical_cell_indices[:, 1] != 9) & ~feed
    return mesh, feed, transition, quads, curved_m


def _cell_colors(feed: np.ndarray, transition: np.ndarray) -> np.ndarray:
    colors = np.full(len(feed), "#f8fafc", dtype=object)
    colors[transition] = "#bfdbfe"
    colors[feed] = "#f97316"
    return colors


def _write_paraview(
    outdir: Path,
    mesh: no_port.MeshData,
    feed: np.ndarray,
    transition: np.ndarray,
    quads: np.ndarray,
    curved_m: np.ndarray,
) -> tuple[Path, Path]:
    unwrapped = outdir / "accepted_jfo_unwrapped_hex.vtu"
    curved = outdir / "accepted_jfo_conical_surface.vtu"
    common = {
        "block_id": [mesh.logical_cell_indices[:, 1].astype(np.int32)],
        "pressure_feed": [feed.astype(np.uint8)],
        "transition_zone": [transition.astype(np.uint8)],
    }
    meshio.write(
        unwrapped,
        meshio.Mesh(
            points=mesh.points_m,
            cells=[
                (
                    "hexahedron",
                    mesh.hexes.astype(np.int64) - 1,
                )
            ],
            cell_data=common
            | {
                "max_nonorthogonality_deg": [
                    mesh.cell_metrics["max_nonorthogonality_deg"]
                ],
                "max_skewness": [
                    mesh.cell_metrics["max_skewness"]
                ],
                "solver_eligible": [
                    np.ones(len(mesh.hexes), dtype=np.uint8)
                ],
            },
        ),
        file_format="vtu",
        binary=True,
    )
    meshio.write(
        curved,
        meshio.Mesh(
            points=curved_m,
            cells=[("quad", quads)],
            cell_data=common
            | {
                "visualization_only": [
                    np.ones(len(quads), dtype=np.uint8)
                ]
            },
        ),
        file_format="vtu",
        binary=True,
    )
    for path, expected_type in (
        (unwrapped, "hexahedron"),
        (curved, "quad"),
    ):
        reopened = meshio.read(path)
        assert reopened.cells[0].type == expected_type
        assert len(reopened.cells[0].data) == len(mesh.hexes)
        assert int(reopened.cell_data["pressure_feed"][0].sum()) == int(
            feed.sum()
        )
    return unwrapped, curved


def _draw_accepted_package(
    outdir: Path,
    inputs: Inputs,
    mesh: no_port.MeshData,
    feed: np.ndarray,
    transition: np.ndarray,
    quads: np.ndarray,
    curved_m: np.ndarray,
) -> tuple[Path, Path]:
    planar_mm = (
        mesh.points_m[: len(mesh.points_m) // 2][:, [0, 2]] * 1.0e3
    )
    gamma = math.radians(inputs.semicone_angle_deg)
    local_mm = np.column_stack(
        (
            planar_mm[:, 0] - math.pi * inputs.mean_radius_m * 1.0e3,
            (
                planar_mm[:, 1]
                - inputs.feed_axial_position_m * 1.0e3
            )
            / math.cos(gamma),
        )
    )
    polygons = local_mm[quads]
    colors = _cell_colors(feed, transition)

    figure = plt.figure(figsize=(18, 7), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.45, 1.0, 1.0))
    full = figure.add_subplot(grid[0, 0])
    cone = figure.add_subplot(grid[0, 1], projection="3d")
    close = figure.add_subplot(grid[0, 2])

    full.add_collection(
        PolyCollection(
            polygons,
            facecolors=colors,
            edgecolors="#64748b",
            linewidths=0.12,
            rasterized=True,
        )
    )
    full.autoscale()
    full.set(
        aspect="equal",
        xlabel="circumferential distance from feed centre [mm]",
        ylabel="conical axial distance from feed centre [mm]",
        title=(
            "Complete unwrapped solver mesh\n"
            f"{len(mesh.hexes):,} Hex8 cells"
        ),
    )

    curved_mm = curved_m * 1.0e3
    cone.add_collection3d(
        Poly3DCollection(
            curved_mm[quads],
            facecolors=colors,
            edgecolors="#64748b",
            linewidths=0.08,
            alpha=0.9,
            rasterized=True,
        )
    )
    limits = np.ptp(curved_mm, axis=0)
    centres = np.mean(
        np.column_stack(
            (curved_mm.min(axis=0), curved_mm.max(axis=0))
        ),
        axis=1,
    )
    half = 0.5 * limits.max()
    cone.set(
        xlim=(centres[0] - half, centres[0] + half),
        ylim=(centres[1] - half, centres[1] + half),
        zlim=(0.0, inputs.length_m * 1.0e3),
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="axial z [mm]",
        title="Same nodes on the conical mid-film surface",
    )
    cone.set_box_aspect((limits[0], limits[1], limits[2]))
    cone.view_init(elev=10, azim=90)

    centres_2d = polygons.mean(axis=1)
    near = np.all(np.abs(centres_2d) <= 7.0, axis=1)
    close.add_collection(
        PolyCollection(
            polygons[near],
            facecolors=colors[near],
            edgecolors="#475569",
            linewidths=0.5,
        )
    )
    angle = np.linspace(0.0, 2.0 * math.pi, 500)
    close.plot(
        2.0 * np.cos(angle),
        2.0 * np.sin(angle),
        color="#111827",
        linewidth=2.0,
    )
    close.set(
        aspect="equal",
        xlim=(-6.5, 6.5),
        ylim=(-6.5, 6.5),
        xlabel="circumferential distance [mm]",
        ylabel="conical axial distance [mm]",
        title=(
            "4 mm inlet close-up\n"
            f"{int(feed.sum())} feed cells, 32 rim faces"
        ),
    )
    legend = [
        Patch(facecolor="#f97316", label="pressure-feed zone"),
        Patch(facecolor="#bfdbfe", label="body-fitted transition"),
        Patch(facecolor="#f8fafc", label="structured background"),
    ]
    close.legend(handles=legend, loc="upper right", fontsize=8)
    figure.suptitle(
        "Accepted reduced JFO geometry and mesh — 256×80 background",
        fontsize=15,
    )
    png = outdir / "accepted_geometry_and_mesh.png"
    pdf = outdir / "accepted_geometry_and_mesh.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _ordered_result_fields(
    vtk: Path,
    mesh: no_port.MeshData,
    feed: np.ndarray,
    inputs: Inputs,
) -> dict[str, np.ndarray]:
    result = meshio.read(vtk)
    if len(result.cells) != 1 or result.cells[0].type != "hexahedron":
        raise ValueError(f"{vtk} is not the expected single Hex8 mesh")
    if len(result.cells[0].data) != len(mesh.hexes):
        raise ValueError(f"{vtk} does not match the accepted mesh cell count")

    source_centres = mesh.points_m[
        mesh.hexes.astype(np.int64) - 1
    ].mean(axis=1)
    result_centres = result.points[result.cells[0].data].mean(axis=1)
    mismatch, source_to_result = cKDTree(
        result_centres[:, [0, 2]]
    ).query(
        source_centres[:, [0, 2]],
        k=1,
    )
    if mismatch.max() > 5.0e-8:
        raise ValueError(
            f"{vtk} cell centres do not match the accepted mesh "
            f"(max mismatch {mismatch.max():.3g} m)"
        )
    if len(np.unique(source_to_result)) != len(mesh.hexes):
        raise ValueError(f"{vtk} cell-centre map is not one-to-one")

    fields = {
        name: np.asarray(result.cell_data[name][0])[
            source_to_result
        ].astype(np.float64)
        for name in (
            "p",
            "thetaFill",
            "filmThickness",
            "surfaceRadius",
            "surfaceMetric",
        )
    }
    if np.any(fields["p"] < -1.0e-6):
        raise ValueError("pressure fell below the JFO rupture floor")
    if (
        fields["thetaFill"].min() < -1.0e-7
        or fields["thetaFill"].max() > 1.0 + 1.0e-7
    ):
        raise ValueError("thetaFill fell outside [0, 1]")
    if not np.allclose(
        fields["p"][feed],
        inputs.feed_gauge_pressure_pa,
        rtol=0.0,
        atol=1.0e-3,
    ):
        raise ValueError("the topological feed cells lost feed pressure")
    return fields


def _write_result_surface(
    outdir: Path,
    inputs: Inputs,
    feed: np.ndarray,
    transition: np.ndarray,
    quads: np.ndarray,
    curved_m: np.ndarray,
    fields: dict[str, np.ndarray],
) -> Path:
    output = outdir / "accepted_2000rpm_conical_fields.vtu"
    pressure_absolute = (
        fields["p"] + inputs.cavitation_pressure_abs_pa
    )
    pressure_gauge = pressure_absolute - inputs.ambient_pressure_pa
    ruptured = fields["thetaFill"] < 1.0 - 1.0e-8
    meshio.write(
        output,
        meshio.Mesh(
            points=curved_m,
            cells=[("quad", quads)],
            cell_data={
                "pressure_absolute_pa": [pressure_absolute],
                "pressure_gauge_pa": [pressure_gauge],
                "thetaFill": [fields["thetaFill"]],
                "filmThickness_m": [fields["filmThickness"]],
                "surfaceRadius_m": [fields["surfaceRadius"]],
                "surfaceMetric": [fields["surfaceMetric"]],
                "pressure_feed": [feed.astype(np.uint8)],
                "transition_zone": [
                    transition.astype(np.uint8)
                ],
                "ruptured": [ruptured.astype(np.uint8)],
                "visualization_only": [
                    np.ones(len(quads), dtype=np.uint8)
                ],
            },
        ),
        file_format="vtu",
        binary=True,
    )
    reopened = meshio.read(output)
    assert len(reopened.cells[0].data) == len(quads)
    assert int(reopened.cell_data["pressure_feed"][0].sum()) == int(
        feed.sum()
    )
    return output


def _result_metrics(
    inputs: Inputs,
    mesh: no_port.MeshData,
    fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    pressure_gauge_mpa = (
        fields["p"]
        + inputs.cavitation_pressure_abs_pa
        - inputs.ambient_pressure_pa
    ) / 1.0e6
    ruptured = fields["thetaFill"] < 1.0 - 1.0e-8
    area = (
        fields["surfaceMetric"]
        * mesh.cell_metrics["signed_volume_m3"]
        / float(mesh.metadata["pseudo_thickness_m"])
    )
    cavity_percent = 100.0 * area[ruptured].sum() / area.sum()
    return pressure_gauge_mpa, ruptured, float(cavity_percent)


def _draw_result_fields(
    outdir: Path,
    inputs: Inputs,
    mesh: no_port.MeshData,
    feed: np.ndarray,
    quads: np.ndarray,
    curved_m: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[Path, Path, Path, Path]:
    pressure, ruptured, cavity_percent = _result_metrics(
        inputs, mesh, fields
    )
    planar = mesh.points_m[: len(mesh.points_m) // 2]
    unwrapped = np.column_stack(
        (
            np.degrees(planar[:, 0] / inputs.mean_radius_m),
            planar[:, 2] * 1.0e3,
        )
    )
    polygons = unwrapped[quads]
    values = (
        (
            pressure,
            "magma",
            (0.0, float(pressure.max())),
            "Gauge pressure [MPa]",
            f"Pressure: max {pressure.max():.5f} MPa gauge",
        ),
        (
            fields["thetaFill"],
            "viridis",
            (0.0, 1.0),
            r"Liquid fill fraction, $\theta_\mathrm{fill}$",
            f"Fill fraction: min {fields['thetaFill'].min():.6f}",
        ),
        (
            ruptured.astype(float),
            mpl_colors.ListedColormap(("#306998", "#e66101")),
            (0.0, 1.0),
            "0 = full film, 1 = ruptured",
            f"JFO rupture mask: {cavity_percent:.3f}% of area",
        ),
    )
    figure, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, constrained_layout=True
    )
    for axis, (data, cmap, limits, label, title) in zip(
        axes, values, strict=True
    ):
        collection = PolyCollection(
            polygons,
            array=data,
            cmap=cmap,
            edgecolors="#0f172a",
            linewidths=0.04,
            rasterized=True,
        )
        collection.set_clim(*limits)
        axis.add_collection(collection)
        axis.autoscale()
        axis.set(
            xlim=(0.0, 360.0),
            ylim=(0.0, inputs.length_m * 1.0e3),
            ylabel="axial z [mm]",
            title=title,
        )
        figure.colorbar(collection, ax=axis, label=label)
    axes[-1].set_xlabel(
        r"circumferential angle, $\theta$ [degrees]"
    )
    figure.suptitle(
        "Body-fitted OpenFOAM JFO numerical candidate — 2000 rpm"
    )
    unwrapped_png = outdir / "accepted_2000rpm_unwrapped_fields.png"
    unwrapped_pdf = outdir / "accepted_2000rpm_unwrapped_fields.pdf"
    figure.savefig(unwrapped_png, dpi=250)
    figure.savefig(unwrapped_pdf)
    plt.close(figure)

    curved_mm = curved_m * 1.0e3
    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    cone_values = (
        (
            pressure,
            "magma",
            mpl_colors.PowerNorm(
                gamma=0.6, vmin=0.0, vmax=float(pressure.max())
            ),
            "Gauge pressure [MPa] (power-scaled colors)",
            "Pressure on conical mid-film surface",
        ),
        (
            fields["thetaFill"],
            "viridis",
            mpl_colors.Normalize(vmin=0.0, vmax=1.0),
            r"Liquid fill fraction, $\theta_\mathrm{fill}$",
            "Fill fraction on conical mid-film surface",
        ),
    )
    limits = np.ptp(curved_mm, axis=0)
    centres = 0.5 * (
        curved_mm.min(axis=0) + curved_mm.max(axis=0)
    )
    half = 0.5 * limits.max()
    for index, (data, cmap, norm, label, title) in enumerate(
        cone_values, start=1
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        collection = Poly3DCollection(
            curved_mm[quads],
            array=data,
            cmap=cmap,
            norm=norm,
            edgecolors="#334155",
            linewidths=0.03,
            rasterized=True,
        )
        axis.add_collection3d(collection)
        axis.set(
            xlim=(centres[0] - half, centres[0] + half),
            ylim=(centres[1] - half, centres[1] + half),
            zlim=(0.0, inputs.length_m * 1.0e3),
            xlabel="x [mm]",
            ylabel="y [mm]",
            zlabel="axial z [mm]",
            title=title,
        )
        axis.set_box_aspect((limits[0], limits[1], limits[2]))
        axis.view_init(elev=10, azim=90)
        figure.colorbar(
            collection, ax=axis, label=label, shrink=0.68, pad=0.08
        )
    figure.suptitle(
        "Converged 2000 rpm fields on the accepted mesh\n"
        "Reduced JFO model; physical validation pending"
    )
    conical_png = outdir / "accepted_2000rpm_conical_fields.png"
    conical_pdf = outdir / "accepted_2000rpm_conical_fields.pdf"
    figure.savefig(conical_png, dpi=230)
    figure.savefig(conical_pdf)
    plt.close(figure)
    return unwrapped_png, unwrapped_pdf, conical_png, conical_pdf


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/conical_journal/studies/jfo-feed-geometry"),
    )
    parser.add_argument(
        "--result-case",
        type=Path,
        default=REPO_ROOT
        / "out/archive/conical_journal/simulation/openfoam/jfo/jfo_body_fitted_sim_256x80",
        help="optional solved case; fields are omitted when its VTK directory is absent",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_args(values)
    target = arguments.outdir.resolve()
    result_case = arguments.result_case.resolve()
    stage = make_staging_directory(target)
    inputs = Inputs(n_theta=256, n_axial=80)
    result_vtks: list[Path] = []
    try:
        old_polygons, old_selected, old_area = _old_polygons(inputs)
        new_polygons, new_selected, new_area = _new_polygons(inputs)

        figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        _draw(axes[0], old_polygons, old_selected, "Old cell-centre mask", old_area)
        _draw(axes[1], new_polygons, new_selected, "New body-fitted O-grid", new_area)
        figure.suptitle(
            "JFO pressure-feed geometry at 256×80\n"
            "orange cells are fixed at feed pressure",
            fontsize=14,
        )
        figure.savefig(stage / "feed_geometry_comparison.png", dpi=220)
        figure.savefig(stage / "feed_geometry_comparison.pdf")
        plt.close(figure)

        accepted_outdir = stage / "accepted"
        accepted_outdir.mkdir()
        mesh, feed, transition, quads, curved_m = _accepted_mesh(inputs)
        _draw_accepted_package(
            accepted_outdir,
            inputs,
            mesh,
            feed,
            transition,
            quads,
            curved_m,
        )
        _write_paraview(
            accepted_outdir,
            mesh,
            feed,
            transition,
            quads,
            curved_m,
        )

        result_vtks = sorted(
            (result_case / "VTK").glob(f"{result_case.name}_*.vtk")
        )
        if result_vtks:
            result_outdir = stage / "result"
            result_outdir.mkdir()
            fields = _ordered_result_fields(result_vtks[-1], mesh, feed, inputs)
            _write_result_surface(
                result_outdir,
                inputs,
                feed,
                transition,
                quads,
                curved_m,
                fields,
            )
            _draw_result_fields(
                result_outdir,
                inputs,
                mesh,
                feed,
                quads,
                curved_m,
                fields,
            )
        publish_generation(
            stage,
            target,
            stage="study",
            operation="jfo-feed-geometry",
            status="RENDERED",
            resolved_inputs={
                "n_theta": inputs.n_theta,
                "n_axial": inputs.n_axial,
                "result_case": result_case,
                "result_fields_included": bool(result_vtks),
            },
            input_units={"geometry": "m internally; mm in figures"},
            producer_files=(Path(__file__),),
            upstream_artifacts=(result_vtks[-1],) if result_vtks else (),
            tool_versions={
                "matplotlib": matplotlib.__version__,
                "meshio": meshio.__version__,
                "numpy": np.__version__,
            },
            argv=values,
            acceptance_status="EVIDENCE_ONLY",
            repository=REPO_ROOT,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"wrote JFO feed-geometry evidence to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
