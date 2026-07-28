#!/usr/bin/env python3
"""Mass-conserving Reynolds/JFO companion model for the conical bearing."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import meshio
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Inputs:
    rpm: float = 0.0
    n_theta: int = 128
    n_axial: int = 40
    length_m: float = 0.100
    mean_radius_m: float = 0.050
    semicone_angle_deg: float = 10.0
    radial_clearance_m: float = 50e-6
    eccentricity_m: float = 30e-6
    eccentricity_angle_deg: float = -90.0
    feed_diameter_m: float = 0.004
    feed_axial_position_m: float = 0.050
    feed_gauge_pressure_pa: float = 500_000.0
    ambient_pressure_pa: float = 101_325.0
    cavitation_pressure_abs_pa: float = 101_325.0
    dynamic_viscosity_pa_s: float = 0.0277
    density_kg_m3: float = 860.0
    convergence_tolerance: float = 1e-8
    max_revolutions: int = 8


@dataclass
class Grid:
    theta: np.ndarray
    z: np.ndarray
    journal_radius: np.ndarray
    journal_ray_radius: np.ndarray
    surface_radius: np.ndarray
    film_thickness: np.ndarray
    area: np.ndarray
    feed: np.ndarray
    radial_x: np.ndarray
    radial_y: np.ndarray


@dataclass
class State:
    pressure_above_cavitation_pa: np.ndarray
    fill_fraction: np.ndarray
    converged: bool
    steps: int
    revolutions: float
    pressure_error: float
    fill_error: float
    active_set_iterations_max: int


def validate(inputs: Inputs) -> None:
    if inputs.n_theta < 16 or inputs.n_axial < 4:
        raise ValueError("n_theta must be >=16 and n_axial must be >=4")
    if inputs.length_m <= 0 or inputs.mean_radius_m <= 0:
        raise ValueError("length and mean radius must be positive")
    if not 0 <= inputs.eccentricity_m < inputs.radial_clearance_m:
        raise ValueError("eccentricity must be in [0, radial clearance)")
    if inputs.dynamic_viscosity_pa_s <= 0 or inputs.density_kg_m3 <= 0:
        raise ValueError("oil properties must be positive")
    if inputs.feed_gauge_pressure_pa < 0:
        raise ValueError("feed gauge pressure must be non-negative")
    if inputs.cavitation_pressure_abs_pa > inputs.ambient_pressure_pa:
        raise ValueError("cavitation pressure cannot exceed ambient pressure")


def make_grid(inputs: Inputs) -> Grid:
    validate(inputs)
    gamma = math.radians(inputs.semicone_angle_deg)
    dtheta = 2 * math.pi / inputs.n_theta
    dz = inputs.length_m / inputs.n_axial
    theta = (np.arange(inputs.n_theta) + 0.5) * dtheta
    z = (np.arange(inputs.n_axial) + 0.5) * dz
    journal_radius_1d = (
        inputs.mean_radius_m
        + (inputs.length_m / 2 - z) * math.tan(gamma)
    )
    journal_radius = np.broadcast_to(
        journal_radius_1d[:, None], (inputs.n_axial, inputs.n_theta)
    )
    angle = math.radians(inputs.eccentricity_angle_deg)
    ex = inputs.eccentricity_m * math.cos(angle)
    ey = inputs.eccentricity_m * math.sin(angle)
    q = ex * np.sin(theta) - ey * np.cos(theta)
    journal_ray_radius = q[None, :] + np.sqrt(
        journal_radius**2
        - inputs.eccentricity_m**2
        + q[None, :] ** 2
    )
    bore_radius = journal_radius + inputs.radial_clearance_m
    radial_gap = bore_radius - journal_ray_radius
    film_thickness = radial_gap * math.cos(gamma)
    surface_radius = 0.5 * (bore_radius + journal_ray_radius)
    area = surface_radius * dtheta * dz / math.cos(gamma)

    wrapped = (
        theta - math.pi + math.pi
    ) % (2 * math.pi) - math.pi
    feed_distance = np.hypot(
        surface_radius * wrapped[None, :],
        (z[:, None] - inputs.feed_axial_position_m) / math.cos(gamma),
    )
    feed = feed_distance <= inputs.feed_diameter_m / 2
    if inputs.feed_diameter_m > 0 and not np.any(feed):
        raise ValueError("grid is too coarse to resolve the 4 mm feed patch")

    x = journal_ray_radius * np.sin(theta)[None, :]
    y = -journal_ray_radius * np.cos(theta)[None, :]
    radial_x = (x - ex) / journal_radius
    radial_y = (y - ey) / journal_radius
    return Grid(
        theta=theta,
        z=z,
        journal_radius=journal_radius,
        journal_ray_radius=journal_ray_radius,
        surface_radius=surface_radius,
        film_thickness=film_thickness,
        area=area,
        feed=feed,
        radial_x=radial_x,
        radial_y=radial_y,
    )


def _harmonic(left: float, right: float) -> float:
    return 2 * left * right / (left + right)


def diffusion_matrix(
    inputs: Inputs, grid: Grid, dt: float
) -> tuple[csr_matrix, np.ndarray]:
    nt, nz = inputs.n_theta, inputs.n_axial
    dtheta = 2 * math.pi / nt
    dz = inputs.length_m / nz
    gamma = math.radians(inputs.semicone_angle_deg)
    permeability = grid.film_thickness**3 / (
        12 * inputs.dynamic_viscosity_pa_s
    )
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(nt * nz)
    boundary_source = np.zeros(nt * nz)

    def connect(a: int, b: int, conductance: float) -> None:
        value = dt * conductance
        diagonal[a] += value
        diagonal[b] += value
        rows.extend((a, b))
        cols.extend((b, a))
        values.extend((-value, -value))

    for j in range(nz):
        edge = dz / math.cos(gamma)
        for i in range(nt):
            other = (i + 1) % nt
            distance = 0.5 * (
                grid.surface_radius[j, i] + grid.surface_radius[j, other]
            ) * dtheta
            conductance = (
                _harmonic(permeability[j, i], permeability[j, other])
                * edge
                / distance
            )
            connect(j * nt + i, j * nt + other, conductance)

    axial_distance = dz / math.cos(gamma)
    for j in range(nz - 1):
        for i in range(nt):
            edge = (
                0.5
                * (grid.surface_radius[j, i] + grid.surface_radius[j + 1, i])
                * dtheta
            )
            conductance = (
                _harmonic(permeability[j, i], permeability[j + 1, i])
                * edge
                / axial_distance
            )
            connect(j * nt + i, (j + 1) * nt + i, conductance)

    end_pressure = (
        inputs.ambient_pressure_pa - inputs.cavitation_pressure_abs_pa
    )
    half_axial_distance = 0.5 * axial_distance
    for j in (0, nz - 1):
        for i in range(nt):
            index = j * nt + i
            edge = grid.surface_radius[j, i] * dtheta
            value = (
                dt
                * permeability[j, i]
                * edge
                / half_axial_distance
            )
            diagonal[index] += value
            boundary_source[index] += value * end_pressure

    indices = np.arange(nt * nz)
    rows.extend(indices.tolist())
    cols.extend(indices.tolist())
    values.extend(diagonal.tolist())
    return (
        coo_matrix((values, (rows, cols)), shape=(nt * nz, nt * nz)).tocsr(),
        boundary_source,
    )


def solve_lcp(
    matrix: csr_matrix,
    constant: np.ndarray,
    pressure_scale: float,
    free: np.ndarray | None,
    max_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    size = len(constant)
    free = np.ones(size, dtype=bool) if free is None else free.copy()
    scaled_diagonal = matrix.diagonal() * pressure_scale
    if np.any(scaled_diagonal <= 0):
        raise RuntimeError("non-positive diffusion diagonal")

    for iteration in range(1, max_iterations + 1):
        scaled_pressure = np.zeros(size)
        if np.any(free):
            submatrix = matrix[free][:, free] * pressure_scale
            scaled_pressure[free] = spsolve(submatrix, -constant[free])
        pressure = pressure_scale * scaled_pressure
        slack = matrix @ pressure + constant
        next_free = (
            scaled_pressure - slack / scaled_diagonal
        ) > 1e-12
        if np.array_equal(next_free, free):
            pressure[np.abs(pressure) < 1e-7] = 0
            slack[np.abs(slack) < 1e-18] = 0
            if pressure.min(initial=0) < -1e-5 or slack.min(initial=0) < -1e-18:
                raise RuntimeError("active-set solve violated JFO complementarity")
            return pressure, slack, free, iteration
        free = next_free
    raise RuntimeError("JFO active-set solve did not converge")


def solve(
    inputs: Inputs,
    initial_fill_fraction: np.ndarray | None = None,
    *,
    log_every: int = 0,
) -> tuple[Grid, State]:
    grid = make_grid(inputs)
    omega = inputs.rpm * 2 * math.pi / 60
    dtheta = 2 * math.pi / inputs.n_theta
    dt = 1.0 if omega == 0 else 2 * dtheta / abs(omega)
    matrix, boundary_source = diffusion_matrix(inputs, grid, dt)
    feed_flat = grid.feed.ravel()
    unknown = ~feed_flat
    unknown_indices = np.flatnonzero(unknown)
    feed_indices = np.flatnonzero(feed_flat)
    unknown_matrix = matrix[unknown][:, unknown].tocsr()
    feed_pressure = (
        inputs.ambient_pressure_pa
        + inputs.feed_gauge_pressure_pa
        - inputs.cavitation_pressure_abs_pa
    )
    fixed_pressure = np.full(len(feed_indices), feed_pressure)
    feed_term = (
        matrix[unknown][:, feed_flat] @ fixed_pressure
        if len(feed_indices)
        else np.zeros(len(unknown_indices))
    )
    area_h = (grid.area * grid.film_thickness).ravel()
    fill = (
        np.ones_like(grid.film_thickness)
        if initial_fill_fraction is None
        else np.asarray(initial_fill_fraction, dtype=float).copy()
    )
    if fill.shape != grid.film_thickness.shape:
        raise ValueError("initial fill-fraction shape does not match the grid")
    fill[grid.feed] = 1
    pressure = np.zeros_like(grid.film_thickness)
    pressure[grid.feed] = feed_pressure
    free: np.ndarray | None = np.ones(len(unknown_indices), dtype=bool)
    min_steps = 2 if omega == 0 else inputs.n_theta
    max_steps = 7 if omega == 0 else inputs.max_revolutions * inputs.n_theta
    consecutive = 0
    maximum_active_iterations = 0
    pressure_error = math.inf
    fill_error = math.inf

    for step in range(1, max_steps + 1):
        old_pressure = pressure.copy()
        old_fill = fill.copy()
        volume = grid.area * grid.film_thickness * fill
        advected_volume = (
            volume
            if omega == 0
            else np.roll(volume, 1 if omega > 0 else -1, axis=1)
        )
        constant = (
            area_h[unknown_indices]
            - advected_volume.ravel()[unknown_indices]
            - boundary_source[unknown_indices]
            + feed_term
        )
        pressure_unknown, slack, free, active_iterations = solve_lcp(
            unknown_matrix,
            constant,
            pressure_scale=max(
                inputs.feed_gauge_pressure_pa,
                abs(omega)
                * inputs.dynamic_viscosity_pa_s
                * inputs.mean_radius_m**2
                / inputs.radial_clearance_m**2,
                1.0,
            ),
            free=free,
        )
        maximum_active_iterations = max(
            maximum_active_iterations, active_iterations
        )
        pressure.ravel()[unknown_indices] = pressure_unknown
        pressure.ravel()[feed_indices] = feed_pressure
        fill.ravel()[unknown_indices] = (
            1 - slack / area_h[unknown_indices]
        )
        fill.ravel()[feed_indices] = 1
        if fill.min(initial=1) < -1e-7 or fill.max(initial=0) > 1 + 1e-7:
            raise RuntimeError("JFO transport produced fill fraction outside [0, 1]")
        np.clip(fill, 0, 1, out=fill)

        pressure_error = float(
            np.max(np.abs(pressure - old_pressure))
            / max(feed_pressure, float(np.max(pressure)), 1.0)
        )
        fill_error = float(np.max(np.abs(fill - old_fill)))
        converged_now = (
            pressure_error <= inputs.convergence_tolerance
            and fill_error <= inputs.convergence_tolerance
        )
        consecutive = consecutive + 1 if converged_now else 0
        if log_every and (step == 1 or step % log_every == 0):
            print(
                f"rpm={inputs.rpm:g} step={step} "
                f"pmax={pressure.max():.6g} "
                f"theta_min={fill.min():.8f} "
                f"dp={pressure_error:.3e} dtheta={fill_error:.3e}"
            )
        if step >= min_steps and consecutive >= 5:
            break

    state = State(
        pressure_above_cavitation_pa=pressure,
        fill_fraction=fill,
        converged=consecutive >= 5,
        steps=step,
        revolutions=0.0 if omega == 0 else step / inputs.n_theta,
        pressure_error=pressure_error,
        fill_error=fill_error,
        active_set_iterations_max=maximum_active_iterations,
    )
    return grid, state


def flow_metrics(inputs: Inputs, grid: Grid, state: State) -> dict[str, float]:
    nt, nz = inputs.n_theta, inputs.n_axial
    dtheta = 2 * math.pi / nt
    dz = inputs.length_m / nz
    gamma = math.radians(inputs.semicone_angle_deg)
    omega = inputs.rpm * 2 * math.pi / 60
    pressure = state.pressure_above_cavitation_pa
    content = grid.film_thickness * state.fill_fraction
    permeability = grid.film_thickness**3 / (
        12 * inputs.dynamic_viscosity_pa_s
    )
    feed_in = 0.0

    for j in range(nz):
        edge = dz / math.cos(gamma)
        for i in range(nt):
            other = (i + 1) % nt
            distance = 0.5 * (
                grid.surface_radius[j, i] + grid.surface_radius[j, other]
            ) * dtheta
            pressure_flow = (
                _harmonic(permeability[j, i], permeability[j, other])
                * edge
                * (pressure[j, i] - pressure[j, other])
                / distance
            )
            upstream = i if omega >= 0 else other
            couette_flow = (
                0.5
                * omega
                * grid.surface_radius[j, upstream]
                * content[j, upstream]
                * edge
            )
            flow = pressure_flow + couette_flow
            if grid.feed[j, i] and not grid.feed[j, other]:
                feed_in += flow
            elif not grid.feed[j, i] and grid.feed[j, other]:
                feed_in -= flow

    axial_distance = dz / math.cos(gamma)
    for j in range(nz - 1):
        for i in range(nt):
            if grid.feed[j, i] == grid.feed[j + 1, i]:
                continue
            edge = (
                0.5
                * (grid.surface_radius[j, i] + grid.surface_radius[j + 1, i])
                * dtheta
            )
            flow = (
                _harmonic(permeability[j, i], permeability[j + 1, i])
                * edge
                * (pressure[j, i] - pressure[j + 1, i])
                / axial_distance
            )
            feed_in += flow if grid.feed[j, i] else -flow

    end_pressure = (
        inputs.ambient_pressure_pa - inputs.cavitation_pressure_abs_pa
    )
    end_out = []
    for j in (0, nz - 1):
        outflow = 0.0
        for i in range(nt):
            edge = grid.surface_radius[j, i] * dtheta
            outflow += (
                permeability[j, i]
                * edge
                * (pressure[j, i] - end_pressure)
                / (0.5 * axial_distance)
            )
        end_out.append(outflow)

    net_out = end_out[0] + end_out[1] - feed_in
    reference = max(
        abs(feed_in), abs(end_out[0]) + abs(end_out[1]), 1e-30
    )
    return {
        "feed_in_m3_s": feed_in,
        "axial_z0_out_m3_s": end_out[0],
        "axial_zL_out_m3_s": end_out[1],
        "net_out_m3_s": net_out,
        "relative_imbalance": abs(net_out) / reference,
        "feed_in_kg_s": feed_in * inputs.density_kg_m3,
        "net_out_kg_s": net_out * inputs.density_kg_m3,
    }


def load_metrics(inputs: Inputs, grid: Grid, state: State) -> dict[str, object]:
    gamma = math.radians(inputs.semicone_angle_deg)
    dtheta = 2 * math.pi / inputs.n_theta
    dz = inputs.length_m / inputs.n_axial
    omega = inputs.rpm * 2 * math.pi / 60
    pressure_gauge = (
        inputs.cavitation_pressure_abs_pa
        + state.pressure_above_cavitation_pa
        - inputs.ambient_pressure_pa
    )
    area = grid.journal_radius * dtheta * dz / math.cos(gamma)
    normal = np.stack(
        (
            math.cos(gamma) * grid.radial_x,
            math.cos(gamma) * grid.radial_y,
            np.full_like(grid.radial_x, math.sin(gamma)),
        ),
        axis=-1,
    )
    pressure_force = np.sum(
        -pressure_gauge[..., None] * normal * area[..., None],
        axis=(0, 1),
    )

    tangent = np.stack(
        (-grid.radial_y, grid.radial_x, np.zeros_like(grid.radial_x)),
        axis=-1,
    )
    pressure_gradient = (
        np.roll(state.pressure_above_cavitation_pa, -1, axis=1)
        - np.roll(state.pressure_above_cavitation_pa, 1, axis=1)
    ) / (2 * grid.journal_radius * dtheta)
    shear = state.fill_fraction * (
        -inputs.dynamic_viscosity_pa_s
        * omega
        * grid.journal_radius
        / grid.film_thickness
        - 0.5 * grid.film_thickness * pressure_gradient
    )
    shear_force = np.sum(shear[..., None] * tangent * area[..., None], axis=(0, 1))
    total_force = pressure_force + shear_force
    torque = float(np.sum(shear * grid.journal_radius * area))
    return {
        "pressure_force_n": pressure_force.tolist(),
        "viscous_force_n": shear_force.tolist(),
        "total_force_n": total_force.tolist(),
        "journal_torque_z_nm": torque,
    }


def summarize(inputs: Inputs, grid: Grid, state: State) -> dict[str, object]:
    pressure_abs = (
        inputs.cavitation_pressure_abs_pa
        + state.pressure_above_cavitation_pa
    )
    minimum = np.unravel_index(np.argmin(pressure_abs), pressure_abs.shape)
    maximum = np.unravel_index(np.argmax(pressure_abs), pressure_abs.shape)
    flows = flow_metrics(inputs, grid, state)
    accepted = bool(
        state.converged
        and pressure_abs.min() >= inputs.cavitation_pressure_abs_pa - 1e-6
        and flows["relative_imbalance"] <= 0.005
    )
    return {
        "inputs": asdict(inputs),
        "model": {
            "equation": "Reynolds thin-film equation",
            "cavitation": "mass-conserving JFO/Elrod-Adams complementarity",
            "feed_representation": "4 mm geodesic fixed-pressure surface patch",
            "pressure_floor_abs_pa": inputs.cavitation_pressure_abs_pa,
        },
        "convergence": {
            "converged": state.converged,
            "steps": state.steps,
            "characteristic_revolutions": state.revolutions,
            "pressure_error": state.pressure_error,
            "fill_error": state.fill_error,
            "active_set_iterations_max": state.active_set_iterations_max,
        },
        "pressure": {
            "minimum_absolute_pa": float(pressure_abs[minimum]),
            "minimum_location": {
                "theta_deg": float(np.degrees(grid.theta[minimum[1]])),
                "z_m": float(grid.z[minimum[0]]),
            },
            "maximum_absolute_pa": float(pressure_abs[maximum]),
            "maximum_location": {
                "theta_deg": float(np.degrees(grid.theta[maximum[1]])),
                "z_m": float(grid.z[maximum[0]]),
            },
        },
        "film": {
            "minimum_thickness_m": float(grid.film_thickness.min()),
            "maximum_thickness_m": float(grid.film_thickness.max()),
            "minimum_fill_fraction": float(state.fill_fraction.min()),
            "mean_fill_fraction_area_weighted": float(
                np.average(state.fill_fraction, weights=grid.area)
            ),
            "cavitated_area_fraction": float(
                np.sum(grid.area[state.fill_fraction < 1 - 1e-8])
                / np.sum(grid.area)
            ),
        },
        "flow": flows,
        "loads": load_metrics(inputs, grid, state),
        "acceptance": {
            "accepted": accepted,
            "requirements": {
                "converged": True,
                "minimum_absolute_pressure_pa": (
                    f">={inputs.cavitation_pressure_abs_pa:g}"
                ),
                "relative_mass_imbalance": "<=0.005",
            },
        },
    }


def write_results(
    outdir: Path, inputs: Inputs, grid: Grid, state: State
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = summarize(inputs, grid, state)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pressure_abs = (
        inputs.cavitation_pressure_abs_pa
        + state.pressure_above_cavitation_pa
    )
    np.savez_compressed(
        outdir / "state.npz",
        theta_rad=grid.theta,
        z_m=grid.z,
        film_thickness_m=grid.film_thickness,
        pressure_absolute_pa=pressure_abs,
        fill_fraction=state.fill_fraction,
        feed_mask=grid.feed,
    )

    theta_nodes = np.linspace(0, 2 * math.pi, inputs.n_theta + 1)
    z_nodes = np.linspace(0, inputs.length_m, inputs.n_axial + 1)
    radius_nodes = (
        inputs.mean_radius_m
        + (inputs.length_m / 2 - z_nodes)
        * math.tan(math.radians(inputs.semicone_angle_deg))
        + 0.5 * inputs.radial_clearance_m
    )
    points = np.array(
        [
            [radius * math.sin(theta), -radius * math.cos(theta), z]
            for z, radius in zip(z_nodes, radius_nodes, strict=True)
            for theta in theta_nodes
        ]
    )
    stride = inputs.n_theta + 1
    quads = np.array(
        [
            [
                j * stride + i,
                j * stride + i + 1,
                (j + 1) * stride + i + 1,
                (j + 1) * stride + i,
            ]
            for j in range(inputs.n_axial)
            for i in range(inputs.n_theta)
        ]
    )
    meshio.write(
        outdir / "jfo_surface.vtu",
        meshio.Mesh(
            points,
            [("quad", quads)],
            cell_data={
                "pressure_absolute_pa": [pressure_abs.ravel()],
                "pressure_gauge_pa": [
                    (
                        pressure_abs - inputs.ambient_pressure_pa
                    ).ravel()
                ],
                "fill_fraction": [state.fill_fraction.ravel()],
                "film_thickness_m": [grid.film_thickness.ravel()],
                "feed_mask": [grid.feed.astype(np.uint8).ravel()],
            },
        ),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (0, 360, 0, inputs.length_m * 1000)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    pressure_plot = axes[0].imshow(
        (pressure_abs - inputs.ambient_pressure_pa) / 1e6,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
    )
    axes[0].set_ylabel("z (mm)")
    axes[0].set_title(f"{inputs.rpm:g} rpm: gauge pressure (MPa)")
    figure.colorbar(pressure_plot, ax=axes[0])
    fill_plot = axes[1].imshow(
        state.fill_fraction,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axes[1].set_xlabel("theta (deg; 0 = minimum gap, 180 = feed)")
    axes[1].set_ylabel("z (mm)")
    axes[1].set_title("JFO liquid fill fraction")
    figure.colorbar(fill_plot, ax=axes[1])
    figure.savefig(outdir / "fields.png", dpi=160)
    plt.close(figure)
    return summary


def _stage_name(rpm: float) -> str:
    return f"{rpm:g}".replace("-", "m").replace(".", "p") + "rpm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpm", type=float, nargs="+", default=[0.0, 20.0])
    parser.add_argument("--n-theta", type=int, default=128)
    parser.add_argument("--n-axial", type=int, default=40)
    parser.add_argument("--max-revolutions", type=int, default=8)
    parser.add_argument("--cavitation-pressure-abs-pa", type=float, default=101_325.0)
    parser.add_argument(
        "--outdir", type=Path, default=Path("out/reynolds_jfo_0p5mpa")
    )
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--log-every", type=int, default=32)
    args = parser.parse_args()

    previous_fill: np.ndarray | None = (
        np.load(args.initial_state)["fill_fraction"]
        if args.initial_state
        else None
    )
    ramp: list[dict[str, object]] = []
    for rpm in args.rpm:
        inputs = Inputs(
            rpm=rpm,
            n_theta=args.n_theta,
            n_axial=args.n_axial,
            max_revolutions=args.max_revolutions,
            cavitation_pressure_abs_pa=args.cavitation_pressure_abs_pa,
        )
        grid, state = solve(inputs, previous_fill, log_every=args.log_every)
        summary = write_results(
            args.outdir / _stage_name(rpm), inputs, grid, state
        )
        ramp.append(summary)
        previous_fill = state.fill_fraction
        if not summary["acceptance"]["accepted"]:
            break

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "ramp_summary.json").write_text(
        json.dumps(ramp, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if len(ramp) == len(args.rpm) and all(
        item["acceptance"]["accepted"] for item in ramp
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
