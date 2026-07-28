import numpy as np

from reynolds_jfo import Inputs, make_grid, solve, summarize


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
