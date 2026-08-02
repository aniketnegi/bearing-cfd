from pathlib import Path

import numpy as np

from bearing_cfd.interchange import fluent_legacy
from bearing_cfd.bearings.conical_journal.simulation import jfo_surface_mesh
from bearing_cfd.bearings.conical_journal.meshing import body_fitted_inlet as body_fitted
from bearing_cfd.bearings.conical_journal.meshing import no_port as no_port
from bearing_cfd.bearings.conical_journal.meshing.surface_inlet import InletSpec
from bearing_cfd.bearings.conical_journal.simulation.reynolds_jfo import (
    Inputs,
    main,
    solve,
    summarize,
)


def test_cli_rejects_a_feed_grid_that_is_too_coarse(
    tmp_path: Path, capsys
) -> None:
    outdir = tmp_path / "result"

    assert main(
        [
            "--rpm",
            "0",
            "--n-theta",
            "16",
            "--n-axial",
            "4",
            "--outdir",
            str(outdir),
        ]
    ) == 2
    assert "grid is too coarse to resolve the 4 mm feed patch" in capsys.readouterr().err
    assert not outdir.exists()


def test_uniform_stationary_film_stays_full_and_at_ambient() -> None:
    inputs = Inputs(
        n_theta=32,
        n_axial=8,
        eccentricity_m=0,
        feed_diameter_m=0,
        feed_gauge_pressure_pa=0,
    )
    grid, state = solve(inputs)
    result = summarize(inputs, grid, state)

    assert state.converged
    assert np.all(state.fill_fraction == 1)
    assert np.all(state.pressure_above_cavitation_pa == 0)
    assert result["pressure"]["minimum_absolute_pa"] == 101_325
    assert result["flow"]["net_out_m3_s"] == 0


def test_rotating_jfo_case_cavitates_without_losing_mass() -> None:
    inputs = Inputs(
        rpm=20,
        n_theta=64,
        n_axial=20,
        feed_diameter_m=0.008,
    )
    grid, state = solve(inputs)
    result = summarize(inputs, grid, state)

    assert result["acceptance"]["accepted"]
    assert state.fill_fraction.min() < 1
    assert result["pressure"]["minimum_absolute_pa"] == 101_325
    assert result["flow"]["relative_imbalance"] < 1e-6


def test_body_fitted_jfo_feed_has_a_conforming_circular_rim() -> None:
    params = no_port.BearingParams(
        source=Path("paper-exact"),
        source_sha256="test",
        length_mm=100,
        mean_radius_mm=50,
        semicone_angle_deg=10,
        radial_clearance_mm=0.05,
        eccentricity_ratio=0.6,
        eccentricity_mm=0.03,
        ex_mm=0,
        ey_mm=-0.03,
        cone_slope=np.tan(np.deg2rad(10)),
    )
    mesh, feed = jfo_surface_mesh.build_mesh(
        params,
        InletSpec(axial_position_mm=50, diameter_mm=4, radius_mm=2),
        n_theta=256,
        n_axial=80,
        rim_segments=32,
    )
    census = body_fitted.hex_face_census(mesh.hexes)
    internal = census["counts"] == 2

    assert mesh.metadata["feed_cells"] == 96
    assert mesh.metadata["feed_rim_faces"] == 32
    assert np.count_nonzero(
        feed[census["owner"][internal]]
        != feed[census["neighbour"][internal]]
    ) == 32
    np.testing.assert_allclose(
        mesh.metadata["source_feed_area_m2"],
        mesh.metadata["feed_area_m2"],
        rtol=1e-3,
    )
    assert np.all(mesh.cell_metrics["signed_volume_m3"] > 0)

    occurrences = mesh.hexes[
        :, fluent_legacy.HEX_FACES_INWARD
    ].reshape(-1, 4)
    _, inverse, counts = np.unique(
        np.sort(occurrences, axis=1),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(
        ([0], np.cumsum(counts[:-1], dtype=np.int64))
    )
    owners = np.repeat(
        np.arange(1, len(mesh.hexes) + 1, dtype=np.int64), 6
    )
    first = order[starts]
    neighbours = np.zeros(len(counts), dtype=np.int64)
    internal = counts == 2
    neighbours[internal] = owners[order[starts[internal] + 1]]
    quality = fluent_legacy.orthogonal_quality(
        fluent_legacy.FluentLegacyMesh(
            mesh.points_m,
            mesh.hexes,
            mesh.cell_centres_m,
            occurrences[first],
            owners[first],
            neighbours,
            {},
            1.0,
        )
    )

    assert mesh.metadata["outer_layers"] == 4
    assert quality.min() > 0.95
