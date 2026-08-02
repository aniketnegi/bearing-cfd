"""Through-gap layer grading shared by conical-journal meshers."""

from __future__ import annotations

import math

import numpy as np


def symmetric_gap_coordinates(n_gap: int, inflation_ratio: float) -> np.ndarray:
    """Return wall-clustered coordinates with equal grading at both walls."""
    if n_gap < 1:
        raise ValueError("nGap must be positive")
    if not math.isfinite(inflation_ratio) or inflation_ratio < 1.0:
        raise ValueError("gap inflation ratio must be finite and >=1")
    if n_gap <= 2 or inflation_ratio == 1.0:
        return np.linspace(0.0, 1.0, n_gap + 1, dtype=np.float64)

    half = n_gap // 2
    if n_gap % 2 == 0:
        growth = inflation_ratio ** (1.0 / (half - 1))
        one_side = growth ** np.arange(half, dtype=np.float64)
        widths = np.concatenate([one_side, one_side[::-1]])
    else:
        growth = inflation_ratio ** (1.0 / half)
        one_side = growth ** np.arange(half, dtype=np.float64)
        widths = np.concatenate([one_side, [inflation_ratio], one_side[::-1]])
    widths /= widths.max()
    widths /= widths.sum()
    coordinates = np.concatenate([[0.0], np.cumsum(widths)])
    coordinates[-1] = 1.0
    return coordinates
