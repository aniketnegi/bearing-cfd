"""Unwrap the existing circular O-grid for the native Reynolds/JFO solver."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from meshing import structured_hex_body_fitted_inlet as body_fitted
from meshing import structured_hex_no_port as base
from meshing.structured_hex_surface_inlet import InletSpec


PATCH_NAMES = (
    "thetaMin",
    "thetaMax",
    "axialEndZ0",
    "axialEndZL",
    "frontAndBack",
)


def build_mesh(
    params: base.BearingParams,
    inlet: InletSpec,
    *,
    n_theta: int,
    n_axial: int,
    rim_segments: int,
    pseudo_thickness_m: float = 1.0e-3,
    outer_layers: int | None = None,
    control_radius_factor: float = 2.87,
    control_square_blend: float = 0.36,
    central_corner_radius_factor: float = 0.9,
    smoothing_iterations: int = 100,
) -> tuple[base.MeshData, np.ndarray]:
    """Return the audited rounded-square JFO mesh and topological feed mask.

    The dimensionless defaults are shared by every resolution; only the
    concentric transition-layer count scales with the circular rim.
    """
    if outer_layers is None:
        outer_layers = max(2, rim_segments // 8)
    master = body_fitted.build_ogrid_master(
        params,
        inlet,
        rim_segments,
        1,
        outer_layers,
        n_theta=n_theta,
        n_axial=n_axial,
        quality_optimized=True,
        control_radius_factor=control_radius_factor,
        control_square_blend=control_square_blend,
        central_corner_radius_factor=central_corner_radius_factor,
    )
    if smoothing_iterations:
        fixed = np.unique(
            np.concatenate(
                (
                    master.unchanged_node_tags,
                    master.rim_node_tags,
                )
            )
        )
        master = body_fitted.smooth_master_mesh(
            replace(master, fixed_node_tags=fixed),
            params,
            iterations=smoothing_iterations,
            damping=0.5,
        )
        corner_tags = np.asarray(
            master.metadata["central_square_corner_node_tags"],
            dtype=np.uint64,
        )
        corner_points = master.points_mm[
            body_fitted._indices_for_tags(master.node_tags, corner_tags)
        ]
        corner_radii = np.hypot(
            corner_points[:, 0],
            corner_points[:, 2] - inlet.axial_position_mm,
        )
        if np.ptp(corner_radii) > 1.0e-10:
            raise body_fitted.BodyFittedError(
                "smoothed central corners lost fourfold symmetry"
            )
        master = replace(
            master,
            metadata=dict(master.metadata)
            | {
                "central_square_corner_radius_mm": float(
                    corner_radii.mean()
                )
            },
        )
    body_fitted.validate_master_mesh(master, params, inlet, [])

    master_indices = body_fitted._indices_for_tags(
        master.node_tags, master.quads
    )
    theta = np.mod(
        np.arctan2(master.points_mm[:, 0], -master.points_mm[:, 1]),
        2.0 * math.pi,
    )
    seam_cells = np.ptp(theta[master_indices], axis=1) > math.pi
    seam_nodes = np.unique(
        master_indices[seam_cells][
            theta[master_indices[seam_cells]] < math.pi
        ]
    )
    duplicate = {
        int(node): len(master.points_mm) + index
        for index, node in enumerate(seam_nodes)
    }

    mean_radius_m = params.mean_radius_mm * base.SI_PER_MM
    planar_x = np.concatenate(
        (
            theta * mean_radius_m,
            (theta[seam_nodes] + 2.0 * math.pi) * mean_radius_m,
        )
    )
    planar_z = np.concatenate(
        (
            master.points_mm[:, 2] * base.SI_PER_MM,
            master.points_mm[seam_nodes, 2] * base.SI_PER_MM,
        )
    )
    quads = master_indices.copy()
    for cell in np.flatnonzero(seam_cells):
        for corner in range(4):
            node = int(quads[cell, corner])
            if theta[node] < math.pi:
                quads[cell, corner] = duplicate[node]

    planar = np.column_stack((planar_x, planar_z))
    vertices = planar[quads]
    signed_twice_area = np.sum(
        vertices[:, :, 0] * np.roll(vertices[:, :, 1], -1, axis=1)
        - np.roll(vertices[:, :, 0], -1, axis=1) * vertices[:, :, 1],
        axis=1,
    )
    if np.any(np.abs(signed_twice_area) <= 1.0e-18):
        raise body_fitted.BodyFittedError(
            "the unwrapped JFO mesh contains a zero-area quad"
        )
    reversed_cells = signed_twice_area < 0
    quads[reversed_cells] = quads[reversed_cells][:, [0, 3, 2, 1]]

    point_count = len(planar)
    points_m = np.empty((2 * point_count, 3), dtype=np.float64)
    points_m[:point_count] = np.column_stack(
        (planar_x, np.zeros(point_count), planar_z)
    )
    points_m[point_count:] = np.column_stack(
        (
            planar_x,
            np.full(point_count, pseudo_thickness_m),
            planar_z,
        )
    )
    first, second, third, fourth = quads.T
    hexes = (
        np.column_stack(
            (
                first,
                second,
                second + point_count,
                first + point_count,
                fourth,
                third,
                third + point_count,
                fourth + point_count,
            )
        )
        + 1
    ).astype(np.uint64)
    node_tags = np.arange(1, len(points_m) + 1, dtype=np.uint64)
    cell_tags = np.arange(1, len(hexes) + 1, dtype=np.uint64)
    centres, metrics, census = body_fitted._generic_hex_metrics(
        points_m, node_tags, hexes
    )

    external = census["counts"] == 1
    external_faces = census["oriented_faces"][external]
    face_points = points_m[
        body_fitted._indices_for_tags(node_tags, external_faces)
    ]
    tolerance = 1.0e-12
    at_theta_min = np.all(
        np.abs(face_points[:, :, 0]) <= tolerance, axis=1
    )
    at_theta_max = np.all(
        np.abs(
            face_points[:, :, 0] - 2.0 * math.pi * mean_radius_m
        )
        <= tolerance,
        axis=1,
    )
    at_z0 = np.all(np.abs(face_points[:, :, 2]) <= tolerance, axis=1)
    at_zl = np.all(
        np.abs(
            face_points[:, :, 2] - params.length_mm * base.SI_PER_MM
        )
        <= tolerance,
        axis=1,
    )
    at_front_or_back = np.all(
        np.abs(face_points[:, :, 1]) <= tolerance, axis=1
    ) | np.all(
        np.abs(face_points[:, :, 1] - pseudo_thickness_m)
        <= tolerance,
        axis=1,
    )
    classifications = np.column_stack(
        (at_theta_min, at_theta_max, at_z0, at_zl, at_front_or_back)
    )
    if not np.all(classifications.sum(axis=1) == 1):
        raise body_fitted.BodyFittedError(
            "unwrapped JFO boundary faces do not classify exactly once"
        )
    boundary_quads = {
        name: external_faces[classifications[:, index]]
        for index, name in enumerate(PATCH_NAMES)
    }

    internal = census["counts"] == 2
    if (
        body_fitted._component_count(
            len(hexes),
            census["owner"][internal],
            census["neighbour"][internal],
        )
        != 1
    ):
        raise body_fitted.BodyFittedError(
            "the unwrapped JFO mesh is disconnected"
        )
    feed_mask = np.ascontiguousarray(master.pressure_mask)
    feed_rim_faces = np.count_nonzero(
        feed_mask[census["owner"][internal]]
        != feed_mask[census["neighbour"][internal]]
    )
    if feed_rim_faces != rim_segments:
        raise body_fitted.BodyFittedError(
            "the feed-cell zone does not have the expected circular rim"
        )

    gamma = math.radians(params.semicone_angle_deg)
    cell_theta = centres[:, 0] / mean_radius_m
    cell_z_mm = centres[:, 2] / base.SI_PER_MM
    journal_radius_mm = np.asarray(
        params.journal_radius_mm(cell_z_mm), dtype=np.float64
    )
    q = (
        params.ex_mm * np.sin(cell_theta)
        - params.ey_mm * np.cos(cell_theta)
    )
    journal_ray_mm = q + np.sqrt(
        journal_radius_mm**2
        - params.ex_mm**2
        - params.ey_mm**2
        + q**2
    )
    surface_radius_mm = 0.5 * (
        np.asarray(params.bore_radius_mm(cell_z_mm)) + journal_ray_mm
    )
    surface_metric = surface_radius_mm / (
        params.mean_radius_mm * math.cos(gamma)
    )
    jfo_area_m2 = (
        surface_metric * metrics["signed_volume_m3"] / pseudo_thickness_m
    )

    source_vertices = (
        master.points_mm[master_indices] * base.SI_PER_MM
    )
    _, source_area_vectors = base._quad_geometry(source_vertices)
    source_area_m2 = np.linalg.norm(source_area_vectors, axis=1)
    metadata = {
        "n_theta": n_theta,
        "n_axial": n_axial,
        "rim_segments": rim_segments,
        "pseudo_thickness_m": pseudo_thickness_m,
        "feed_cells": int(feed_mask.sum()),
        "feed_rim_faces": int(feed_rim_faces),
        "feed_area_m2": float(jfo_area_m2[feed_mask].sum()),
        "source_feed_area_m2": float(source_area_m2[feed_mask].sum()),
        "source_polygon_area_m2": (
            float(master.metadata["polygon_area_mm2"]) * 1.0e-6
        ),
        "topology": "body-fitted circular O-grid",
        "outer_layers": outer_layers,
        "control_radius_factor": control_radius_factor,
        "control_square_blend": control_square_blend,
        "central_corner_radius_factor": central_corner_radius_factor,
        "smoothing": master.metadata.get("smoothing"),
    }
    logical = np.column_stack(
        (
            np.arange(len(hexes), dtype=np.int64),
            master.block_id.astype(np.int64),
            feed_mask.astype(np.int64),
        )
    )
    mesh = base.MeshData(
        points_m=np.ascontiguousarray(points_m),
        hexes=np.ascontiguousarray(hexes),
        boundary_quads={
            name: np.ascontiguousarray(values)
            for name, values in boundary_quads.items()
        },
        logical_cell_indices=np.ascontiguousarray(logical),
        cell_tags=cell_tags,
        node_tags=node_tags,
        cell_centres_m=np.ascontiguousarray(centres),
        cell_metrics={
            name: np.ascontiguousarray(values)
            for name, values in metrics.items()
        },
        metadata=metadata,
    )
    feed_mask.setflags(write=False)
    return mesh, feed_mask
