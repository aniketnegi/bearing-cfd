from __future__ import annotations

from dataclasses import asdict
import math

import pytest

from bearing_cfd.bearings.conical_journal.geometry.model import (
    GeometryInputs,
    GeometryParameterError,
    resolve_geometry,
)


def test_default_geometry_matches_existing_resolved_values() -> None:
    assert asdict(resolve_geometry(GeometryInputs())) == {
        "length": 60.0,
        "mean_radius": 50.0,
        "semicone_angle_deg": 10.0,
        "radial_clearance": 0.05,
        "eccentricity_ratio": 0.6,
        "eccentricity_angle_deg": -90.0,
        "hole_diameter": 4.0,
        "hole_axial_pos": 30.0,
        "split_halfwidth": 4.0,
        "bushing_wall_thickness": 10.0,
        "inlet_extension": 3.0,
        "axial_cutter_extension": 1.0,
        "gamma_rad": 0.17453292519943295,
        "cone_slope": 0.17632698070846498,
        "hole_radius": 2.0,
        "eccentricity": 0.03,
        "phi_rad": -1.5707963267948966,
        "ex": 1.8369701987210296e-18,
        "ey": -0.03,
        "z_hole_min": 28.0,
        "z_hole_max": 32.0,
        "z1": 26.0,
        "z2": 34.0,
        "y_feed_end": 63.402653961416924,
        "feed_start_disk_margin": 47.6471210512379,
        "h_radial_min": 0.020000000000000004,
        "h_radial_max": 0.08,
        "h_normal_min": 0.019696155060244164,
        "h_normal_max": 0.07878462024097664,
        "base_volume_exact": 942.9490349749764,
        "feed_scale_estimate": 163.36281798666926,
    }


def test_concentric_zero_cone_limit() -> None:
    geometry = resolve_geometry(
        GeometryInputs(semicone_angle_deg=0.0, eccentricity_ratio=0.0)
    )

    assert geometry.cone_slope == 0.0
    assert geometry.eccentricity == 0.0
    assert geometry.ex == 0.0
    assert geometry.ey == 0.0
    assert geometry.h_radial_min == geometry.radial_clearance
    assert geometry.h_radial_max == geometry.radial_clearance
    assert geometry.h_normal_min == geometry.radial_clearance
    assert geometry.h_normal_max == geometry.radial_clearance
    assert geometry.journal_radius(0.0) == geometry.mean_radius
    assert geometry.journal_radius(geometry.length) == geometry.mean_radius
    assert geometry.base_volume_exact == math.pi * geometry.length * (
        2.0 * geometry.mean_radius * geometry.radial_clearance
        + geometry.radial_clearance**2
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"length": 0.0}, "length must be > 0"),
        ({"semicone_angle_deg": float("inf")}, "semicone_angle_deg must be finite"),
        (
            {"eccentricity_angle_deg": float("inf")},
            "eccentricity_angle_deg must be finite",
        ),
        ({"semicone_angle_deg": 90.0}, "0 <= gamma < 90 deg"),
        ({"eccentricity_ratio": 1.0}, "0 <= epsilon < 1"),
        ({"hole_axial_pos": 2.0}, "feed cylinder must stay inside"),
        ({"split_halfwidth": 2.0}, "split_halfwidth must be greater"),
    ],
)
def test_invalid_raw_geometry_has_field_specific_error(
    changes: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(GeometryParameterError, match=message):
        resolve_geometry(GeometryInputs(**changes))
