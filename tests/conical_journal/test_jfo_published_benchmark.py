"""Published 1-D JFO graph-consistency check for the active-set LCP solver.

Giacopini et al., J. Tribol. 132 (2010), 041702, cases 1 and 2:
https://doi.org/10.1115/1.4002215

Miraskari et al., J. Tribol. 139 (2017), 031703, Fig. 9:
https://doi.org/10.1115/1.4034244

Miraskari's compressible Elrod comparison uses a 100 GPa bulk modulus; its
Figure 9 also plots Giacopini's incompressible LCP result.  This deliberately
test-only assembly takes that incompressible limit and checks only broad
graph-read peak/front bands through ``solve_lcp``.  It does not compare the
full published curve or validate the production bearing model.
"""

import numpy as np
from scipy.sparse import csr_matrix

from bearing_cfd.bearings.conical_journal.simulation.reynolds_jfo import solve_lcp


def _published_case(end_pressure_pa: float) -> tuple[np.ndarray, ...]:
    cells = 60
    length = 0.125
    viscosity = 0.015
    speed = 4.0
    h_min, h_max = 15e-6, 25e-6
    dx = length / cells
    x = (np.arange(cells) + 0.5) * dx
    faces = np.arange(cells + 1) * dx

    def film(position: np.ndarray) -> np.ndarray:
        return (
            0.5 * (h_max + h_min)
            + 0.5 * (h_max - h_min)
            * np.cos(2 * np.pi * position / length)
        )

    h_face = film(faces)
    conductance = h_face**3 / (12 * viscosity)
    advection = 0.5 * speed
    diffusion = np.zeros((cells, cells))
    void_term = np.zeros_like(diffusion)
    constant = np.zeros(cells)

    for i in range(cells):
        west = conductance[i] / (0.5 * dx if i == 0 else dx)
        east = conductance[i + 1] / (
            0.5 * dx if i == cells - 1 else dx
        )
        diffusion[i, i] = west + east
        if i:
            diffusion[i, i - 1] = -west
        else:
            constant[i] -= west * end_pressure_pa
        if i < cells - 1:
            diffusion[i, i + 1] = -east
        else:
            constant[i] -= east * end_pressure_pa

        void_term[i, i] = -advection * h_face[i + 1]
        if i:
            void_term[i, i - 1] = advection * h_face[i]
        constant[i] += advection * (h_face[i + 1] - h_face[i])

    # ponytail: dense elimination is clearest for this fixed 60-cell benchmark.
    lcp_matrix = -np.linalg.solve(diffusion, void_term)
    lcp_constant = -np.linalg.solve(diffusion, constant)
    scale = 1e12
    void, pressure_scaled, _, _ = solve_lcp(
        csr_matrix(lcp_matrix / scale),
        lcp_constant / scale,
        pressure_scale=1.0,
        free=None,
    )
    return x / length, pressure_scaled * scale, 1 - void


def test_published_jfo_peaks_and_fronts_are_graph_consistent() -> None:
    x0, pressure0, fill0 = _published_case(0.0)
    cavity0 = x0[fill0 < 1 - 1e-8]

    assert 5.5e6 < pressure0.max() < 6.3e6
    assert 0.60 < cavity0[0] < 0.70
    assert cavity0[-1] > 0.98

    x1, pressure1, fill1 = _published_case(1e6)
    cavity1 = x1[fill1 < 1 - 1e-8]

    assert 6.2e6 < pressure1.max() < 7.0e6
    assert 0.60 < cavity1[0] < 0.70
    assert 0.90 < cavity1[-1] < 0.99
