"""Validated geometry contract for the eccentric conical bearing.

Lengths use millimetres and angles use degrees unless a field name states
otherwise.  The bearing axis is +z; the journal offset is measured in the
bearing x-y frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


class GeometryParameterError(ValueError):
    """Input values cannot define the required bearing geometry."""


@dataclass(frozen=True)
class GeometryInputs:
    """User-supplied physical and construction geometry in millimetres."""

    length: float = 60.0
    mean_radius: float = 50.0
    semicone_angle_deg: float = 10.0
    radial_clearance: float = 0.050
    eccentricity_ratio: float = 0.6
    eccentricity_angle_deg: float = -90.0
    hole_diameter: float = 4.0
    hole_axial_pos: float | None = None
    split_halfwidth: float = 4.0
    bushing_wall_thickness: float = 10.0
    inlet_extension: float = 3.0
    axial_cutter_extension: float = 1.0


@dataclass(frozen=True)
class ResolvedGeometry:
    """Validated construction-ready geometry derived from ``GeometryInputs``."""

    length: float
    mean_radius: float
    semicone_angle_deg: float
    radial_clearance: float
    eccentricity_ratio: float
    eccentricity_angle_deg: float
    hole_diameter: float
    hole_axial_pos: float
    split_halfwidth: float
    bushing_wall_thickness: float
    inlet_extension: float
    axial_cutter_extension: float
    gamma_rad: float
    cone_slope: float
    hole_radius: float
    eccentricity: float
    phi_rad: float
    ex: float
    ey: float
    z_hole_min: float
    z_hole_max: float
    z1: float
    z2: float
    y_feed_end: float
    feed_start_disk_margin: float
    h_radial_min: float
    h_radial_max: float
    h_normal_min: float
    h_normal_max: float
    base_volume_exact: float
    feed_scale_estimate: float

    def journal_radius(self, z: float) -> float:
        return self.mean_radius + (self.length / 2.0 - z) * self.cone_slope

    def bore_radius(self, z: float) -> float:
        return self.journal_radius(z) + self.radial_clearance

    def outer_radius(self, z: float) -> float:
        return self.bore_radius(z) + self.bushing_wall_thickness


def _raise_if_invalid(errors: list[str]) -> None:
    if errors:
        raise GeometryParameterError(
            "Invalid parameters:\n- " + "\n- ".join(dict.fromkeys(errors))
        )


def validate_geometry_inputs(inputs: GeometryInputs) -> None:
    """Validate raw values before trigonometric or derived calculations."""

    numeric = {
        name: value for name, value in asdict(inputs).items() if value is not None
    }
    errors = [
        f"{name} must be finite"
        for name, value in numeric.items()
        if not math.isfinite(value)
    ]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(inputs.length > 0.0, "length must be > 0")
    require(inputs.mean_radius > 0.0, "mean_radius must be > 0")
    require(inputs.radial_clearance > 0.0, "radial_clearance must be > 0")
    require(inputs.hole_diameter > 0.0, "hole_diameter must be > 0")
    require(
        inputs.bushing_wall_thickness > 0.0,
        "bushing_wall_thickness must be > 0",
    )
    require(inputs.inlet_extension > 0.0, "inlet_extension must be > 0")
    require(
        inputs.axial_cutter_extension > 0.0,
        "axial_cutter_extension must be > 0",
    )
    require(
        0.0 <= inputs.eccentricity_ratio < 1.0,
        "eccentricity_ratio must satisfy 0 <= epsilon < 1",
    )
    require(
        0.0 <= inputs.semicone_angle_deg < 90.0,
        "semicone_angle_deg must satisfy 0 <= gamma < 90 deg",
    )
    require(
        inputs.split_halfwidth > inputs.hole_diameter / 2.0,
        "split_halfwidth must be greater than dh/2",
    )
    _raise_if_invalid(errors)

    hole_radius = inputs.hole_diameter / 2.0
    hole_axial_pos = (
        inputs.length / 2.0 if inputs.hole_axial_pos is None else inputs.hole_axial_pos
    )
    require(
        0.0 < hole_axial_pos < inputs.length,
        "resolved hole_axial_pos must satisfy 0 < zh < L",
    )
    require(
        hole_radius < hole_axial_pos < inputs.length - hole_radius,
        "the complete feed cylinder must stay inside both axial ends",
    )
    require(
        0.0 < hole_axial_pos - inputs.split_halfwidth
        and hole_axial_pos + inputs.split_halfwidth < inputs.length,
        "split planes must satisfy 0 < zh-w < zh+w < L",
    )
    _raise_if_invalid(errors)


def _validate_resolved_geometry(geometry: ResolvedGeometry) -> None:
    numeric = {
        name: value
        for name, value in asdict(geometry).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    errors = [
        f"{name} must be finite"
        for name, value in numeric.items()
        if not math.isfinite(value)
    ]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        geometry.journal_radius(geometry.length + geometry.axial_cutter_extension)
        > 0.0,
        "Rj(L+delta) must be > 0",
    )
    require(geometry.bore_radius(geometry.length) > 0.0, "Rb(L) must be > 0")
    require(
        geometry.feed_start_disk_margin > 0.0,
        "the complete circular feed-start disk at y=0 is not proven inside "
        "the journal cutter",
    )
    require(
        geometry.hole_radius < geometry.journal_radius(geometry.z_hole_max),
        "feed radius must be smaller than the local journal and bore radii",
    )
    require(
        geometry.y_feed_end > geometry.outer_radius(geometry.z_hole_min),
        "feed outer endpoint must lie outside the context bushing",
    )
    require(
        geometry.y_feed_end > 0.0,
        "feed cylinder must have positive length and intersect the bore blank",
    )
    _raise_if_invalid(errors)


def resolve_geometry(inputs: GeometryInputs) -> ResolvedGeometry:
    """Validate and resolve one immutable geometry snapshot."""

    validate_geometry_inputs(inputs)
    try:
        hole_axial_pos = (
            inputs.length / 2.0
            if inputs.hole_axial_pos is None
            else inputs.hole_axial_pos
        )
        gamma = math.radians(inputs.semicone_angle_deg)
        slope = math.tan(gamma)
        hole_radius = inputs.hole_diameter / 2.0
        eccentricity = inputs.eccentricity_ratio * inputs.radial_clearance
        phi = math.radians(inputs.eccentricity_angle_deg)
        ex = eccentricity * math.cos(phi)
        ey = eccentricity * math.sin(phi)

        def journal_radius(z: float) -> float:
            return inputs.mean_radius + (inputs.length / 2.0 - z) * slope

        def bore_radius(z: float) -> float:
            return journal_radius(z) + inputs.radial_clearance

        def outer_radius(z: float) -> float:
            return bore_radius(z) + inputs.bushing_wall_thickness

        z_hole_min = hole_axial_pos - hole_radius
        z_hole_max = hole_axial_pos + hole_radius
        y_feed_end = outer_radius(z_hole_min) + inputs.inlet_extension
        # The worst disk point combines the x offset and feed radius; this
        # bound proves the complete y=0 feed-start disk lies in the journal.
        disk_max_axis_distance = math.hypot(abs(ex) + hole_radius, ey)
        disk_margin = journal_radius(z_hole_max) - disk_max_axis_distance
        radial_min = inputs.radial_clearance - eccentricity
        radial_max = inputs.radial_clearance + eccentricity
        geometry = ResolvedGeometry(
            length=inputs.length,
            mean_radius=inputs.mean_radius,
            semicone_angle_deg=inputs.semicone_angle_deg,
            radial_clearance=inputs.radial_clearance,
            eccentricity_ratio=inputs.eccentricity_ratio,
            eccentricity_angle_deg=inputs.eccentricity_angle_deg,
            hole_diameter=inputs.hole_diameter,
            hole_axial_pos=hole_axial_pos,
            split_halfwidth=inputs.split_halfwidth,
            bushing_wall_thickness=inputs.bushing_wall_thickness,
            inlet_extension=inputs.inlet_extension,
            axial_cutter_extension=inputs.axial_cutter_extension,
            gamma_rad=gamma,
            cone_slope=slope,
            hole_radius=hole_radius,
            eccentricity=eccentricity,
            phi_rad=phi,
            ex=ex,
            ey=ey,
            z_hole_min=z_hole_min,
            z_hole_max=z_hole_max,
            z1=hole_axial_pos - inputs.split_halfwidth,
            z2=hole_axial_pos + inputs.split_halfwidth,
            y_feed_end=y_feed_end,
            feed_start_disk_margin=disk_margin,
            h_radial_min=radial_min,
            h_radial_max=radial_max,
            h_normal_min=radial_min * math.cos(gamma),
            h_normal_max=radial_max * math.cos(gamma),
            base_volume_exact=(
                math.pi
                * inputs.length
                * (
                    2.0 * inputs.mean_radius * inputs.radial_clearance
                    + inputs.radial_clearance**2
                )
            ),
            feed_scale_estimate=(
                math.pi
                * hole_radius**2
                * (inputs.bushing_wall_thickness + inputs.inlet_extension)
            ),
        )
    except (OverflowError, ValueError) as error:
        raise GeometryParameterError(
            f"Invalid parameters:\n- geometry derivation failed: {error}"
        ) from error

    _validate_resolved_geometry(geometry)
    return geometry
