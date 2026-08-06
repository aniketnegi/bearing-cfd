import numpy as np
import pytest

from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo as film
from studies.conical_journal.paper_reproduction.boundary_audit import supply_cases
from studies.conical_journal.paper_reproduction.section4 import (
    DEFAULT_LOAD_RATIOS,
    critical_mass,
    damping_matrix,
    inputs_at_xy,
    parse_grid,
    write_partial,
)
from studies.conical_journal.paper_reproduction.three_d import (
    DEFAULT_FIXED_ECCENTRICITY_RATIOS,
    last_numeric_row,
    probe_block,
)


def test_grid_and_xy_inputs() -> None:
    assert DEFAULT_LOAD_RATIOS == tuple(value / 10 for value in range(1, 10))
    assert DEFAULT_FIXED_ECCENTRICITY_RATIOS == tuple(
        value / 100 for value in range(40, 91, 5)
    )
    assert parse_grid("512x160") == (512, 160)
    inputs = inputs_at_xy(film.Inputs(), 3e-6, 4e-6)
    assert inputs.eccentricity_m == pytest.approx(5e-6)
    assert inputs.eccentricity_angle_deg == pytest.approx(53.1301023542)
    with pytest.raises(RuntimeError, match="eccentricity domain"):
        inputs_at_xy(film.Inputs(), 49e-6, 0)


def test_boundary_cases_and_partial_checkpoint(tmp_path) -> None:
    assert supply_cases([2.0, 4.0, 8.0], [25.0, 50.0], 4.0) == [
        ("reynolds", 0.0, 0.0),
        ("reynolds", 4.0, 0.0),
        ("jfo", 0.0, 0.0),
        ("jfo", 2.0, 0.0),
        ("jfo", 4.0, 0.0),
        ("jfo", 8.0, 0.0),
        ("jfo", 4.0, 25.0),
        ("jfo", 4.0, 50.0),
    ]
    path = tmp_path / "results.partial.json"
    write_partial(path, "test", [{"status": "PASS"}])
    assert path.read_text(encoding="utf-8").startswith(
        '{\n  "kind": "test",\n  "completed_count": 1'
    )


def test_frozen_cavity_damping_has_positive_direct_terms() -> None:
    inputs = film.Inputs(
        rpm=496.563,
        n_theta=32,
        n_axial=8,
        eccentricity_m=10e-6,
        feed_diameter_m=0,
        feed_gauge_pressure_pa=0,
    )
    grid, state = film.solve_reynolds(inputs)
    damping, pressurized_area = damping_matrix("reynolds", inputs, grid, state)

    assert damping[0, 0] > 0
    assert damping[1, 1] > 0
    assert 0 < pressurized_area < 1


def test_probe_dictionary_and_numeric_parser(tmp_path) -> None:
    block, theta = probe_block(film.Inputs(), count=12)
    assert len(theta) == 12
    assert block.count("\n        (") == 12
    data = tmp_path / "values.dat"
    data.write_text("# header\n0 (1 2 3)\n1 (4 5 6)\n", encoding="utf-8")
    assert last_numeric_row(data) == [1.0, 4.0, 5.0, 6.0]


def test_critical_mass_satisfies_the_quartic_routh_boundary() -> None:
    stiffness = np.array([[2e6, 1e6], [-0.5e6, 3e6]])
    damping = np.array([[1000.0, 100.0], [-50.0, 1200.0]])
    result = critical_mass(stiffness, damping)

    assert result["accepted"] is True
    mass = result["critical_mass_kg"]
    assert mass == pytest.approx(12.05)
    terms = result["routh_terms"]
    a4 = mass**2
    a3 = mass * terms["trace_c"]
    a2 = mass * terms["trace_k"] + terms["det_c"]
    a1 = terms["mixed_ck"]
    a0 = terms["det_k"]
    assert a3 * a2 * a1 == pytest.approx(a4 * a1**2 + a3**2 * a0)
