#!/usr/bin/env python3
"""Write a validated native Fluent 5/6 ASCII mesh from canonical Hex8 arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HEX_FACES_INWARD = np.asarray(
    (
        (0, 1, 2, 3),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (3, 2, 6, 7),
        (0, 3, 7, 4),
        (4, 7, 6, 5),
    ),
    dtype=np.uint8,
)

# source name, Fluent name, zone id, Fluent boundary-condition id, zone type
FACE_ZONES = (
    ("interior", "interior", 2, 2, "interior"),
    ("journal_wall", "journal_wall", 3, 3, "wall"),
    ("stationary_wall", "stationary_wall", 4, 3, "wall"),
    ("axial_end_z0", "axial_end_z0", 5, 5, "pressure-outlet"),
    ("axial_end_zL", "axial_end_zl", 6, 5, "pressure-outlet"),
    ("pressure_feed", "pressure_feed", 7, 4, "pressure-inlet"),
)
FLUID_ZONE_ID = 8


class FluentLegacyMeshError(RuntimeError):
    """The canonical arrays cannot form the promised Fluent mesh."""


@dataclass(frozen=True)
class FluentLegacyMesh:
    points_m: np.ndarray
    hexes: np.ndarray
    cell_centres_m: np.ndarray
    faces: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    zone_face_indices: dict[str, np.ndarray]
    minimum_c0_orientation_dot: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FluentLegacyMeshError(message)


def _load_canonical(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["points_m"], dtype=np.float64)
            hexes = np.asarray(archive["hexes"], dtype=np.uint64)
            cell_centres = (
                np.asarray(archive["cell_centres_m"], dtype=np.float64)
                if "cell_centres_m" in archive
                else None
            )
            node_tags = np.asarray(archive["node_tags"], dtype=np.uint64)
            cell_tags = np.asarray(archive["cell_tags"], dtype=np.uint64)
            boundaries = {
                name: np.asarray(archive[f"boundary_{name}"], dtype=np.uint64)
                for name, *_ in FACE_ZONES
                if name != "interior"
            }
    except (OSError, KeyError, ValueError) as error:
        raise FluentLegacyMeshError(f"cannot load canonical arrays from {path}: {error}") from error

    _require(points.ndim == 2 and points.shape[1] == 3, "points_m must have shape [N,3]")
    _require(hexes.ndim == 2 and hexes.shape[1] == 8, "hexes must have shape [M,8]")
    _require(len(points) > 0 and len(hexes) > 0, "mesh must contain nodes and Hex8 cells")
    _require(np.all(np.isfinite(points)), "node coordinates contain NaN or Inf")
    _require(
        np.array_equal(node_tags, np.arange(1, len(points) + 1, dtype=np.uint64)),
        "node_tags must be contiguous and one-based",
    )
    _require(
        np.array_equal(cell_tags, np.arange(1, len(hexes) + 1, dtype=np.uint64)),
        "cell_tags must be contiguous and one-based",
    )
    _require(int(hexes.min()) >= 1 and int(hexes.max()) <= len(points), "Hex8 node tag out of range")
    if cell_centres is not None:
        _require(
            cell_centres.shape == (len(hexes), 3),
            "cell_centres_m must have shape [M,3]",
        )
        _require(
            np.all(np.isfinite(cell_centres)),
            "cell centres contain NaN or Inf",
        )
    for name, quads in boundaries.items():
        _require(quads.ndim == 2 and quads.shape[1] == 4, f"boundary_{name} must have shape [F,4]")
        _require(len(quads) > 0, f"boundary_{name} is empty")
        _require(
            int(quads.min()) >= 1 and int(quads.max()) <= len(points),
            f"boundary_{name} node tag out of range",
        )
    return points, hexes, cell_centres, boundaries


def _quad_geometry(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    average = vertices.mean(axis=1)
    following = np.roll(vertices, -1, axis=1)
    triangles_twice = np.cross(
        following - vertices,
        average[:, None, :] - vertices,
    )
    summed = triangles_twice.sum(axis=1)
    norm = np.linalg.norm(summed, axis=1)
    _require(np.all(norm > 0.0), "degenerate quadrilateral face")
    normal = summed / norm[:, None]
    weights = np.einsum("mfc,mc->mf", triangles_twice, normal)
    weight_sum = weights.sum(axis=1)
    _require(np.all(weight_sum > 0.0), "inconsistent quadrilateral winding")
    centre = (
        weights[:, :, None]
        * (vertices + following + average[:, None, :])
    ).sum(axis=1) / (3.0 * weight_sum[:, None])
    return centre, 0.5 * summed


def _volume_centres(
    points: np.ndarray,
    hexes: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    centres = np.empty((len(hexes), 3), dtype=np.float64)
    for start in range(0, len(hexes), chunk_size):
        stop = min(start + chunk_size, len(hexes))
        cell_points = points[hexes[start:stop].astype(np.int64) - 1]
        face_centres = np.empty((stop - start, 6, 3), dtype=np.float64)
        face_areas = np.empty_like(face_centres)
        for face_index, local_face in enumerate(HEX_FACES_INWARD):
            face_centres[:, face_index], face_areas[:, face_index] = (
                _quad_geometry(cell_points[:, local_face])
            )
        estimate = face_centres.mean(axis=1)
        pyramid_three = np.einsum(
            "mfc,mfc->mf",
            face_areas,
            estimate[:, None, :] - face_centres,
        )
        _require(
            np.all(pyramid_three > 0.0),
            "a Hex8 cell has a nonpositive face pyramid",
        )
        centres[start:stop] = (
            pyramid_three[:, :, None]
            * (0.75 * face_centres + 0.25 * estimate[:, None, :])
        ).sum(axis=1) / pyramid_three.sum(axis=1)[:, None]
    return centres


def orthogonal_quality(
    mesh: FluentLegacyMesh,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Return Fluent's face-normal/cell-centroid Orthogonal Quality per cell."""
    quality = np.ones(len(mesh.hexes), dtype=np.float64)
    for start in range(0, len(mesh.faces), chunk_size):
        stop = min(start + chunk_size, len(mesh.faces))
        vertices = mesh.points_m[
            mesh.faces[start:stop].astype(np.int64) - 1
        ]
        face_centres, area = _quad_geometry(vertices)
        area_norm = np.linalg.norm(area, axis=1)
        c0 = mesh.c0[start:stop] - 1
        c1_tags = mesh.c1[start:stop]

        c0_vector = mesh.cell_centres_m[c0] - face_centres
        c0_quality = np.einsum(
            "ij,ij->i", area, c0_vector
        ) / (area_norm * np.linalg.norm(c0_vector, axis=1))
        face_quality_c0 = c0_quality.copy()

        internal = c1_tags > 0
        if np.any(internal):
            c1 = c1_tags[internal] - 1
            centre_vector = (
                mesh.cell_centres_m[c0[internal]]
                - mesh.cell_centres_m[c1]
            )
            centre_quality = np.einsum(
                "ij,ij->i", area[internal], centre_vector
            ) / (
                area_norm[internal]
                * np.linalg.norm(centre_vector, axis=1)
            )
            c1_vector = face_centres[internal] - mesh.cell_centres_m[c1]
            c1_quality = np.einsum(
                "ij,ij->i", area[internal], c1_vector
            ) / (
                area_norm[internal]
                * np.linalg.norm(c1_vector, axis=1)
            )
            face_quality_c0[internal] = np.minimum(
                face_quality_c0[internal],
                centre_quality,
            )
            np.minimum.at(
                quality,
                c1,
                np.minimum(c1_quality, centre_quality),
            )

        np.minimum.at(quality, c0, face_quality_c0)

    quality = np.clip(quality, -1.0, 1.0)
    _require(
        np.all(np.isfinite(quality)) and np.all(quality > 0.0),
        "Orthogonal Quality is nonpositive or non-finite",
    )
    return quality


def orthogonal_quality_summary(
    mesh: FluentLegacyMesh,
    threshold: float = 0.8,
) -> dict[str, Any]:
    quality = orthogonal_quality(mesh)
    quantile_levels = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 1.0)
    quantiles = np.quantile(quality, quantile_levels)
    return {
        "metric": "standard hexahedral Orthogonal Quality",
        "enhanced_orthogonal_quality": False,
        "formula": (
            "minimum normalized face-area dot cell-to-face and "
            "cell-to-neighbour vectors"
        ),
        "minimum": float(quality.min()),
        "mean": float(quality.mean()),
        "quantiles": {
            f"{level:g}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "threshold": threshold,
        "cells_below_threshold": int(np.count_nonzero(quality < threshold)),
        "fraction_below_threshold": float(np.mean(quality < threshold)),
        "threshold_passed": bool(np.all(quality >= threshold)),
    }


def independent_centroid_orthogonal_quality_audit(
    mesh: FluentLegacyMesh,
    threshold: float = 0.8,
) -> dict[str, Any]:
    """Recompute cell centres from Hex8 geometry and repeat the OQ audit."""
    _require(
        0.0 < threshold <= 1.0,
        "minimum Orthogonal Quality must be in (0, 1]",
    )
    reconstructed_centres = _volume_centres(mesh.points_m, mesh.hexes)
    reconstructed_mesh = replace(
        mesh,
        cell_centres_m=reconstructed_centres,
    )
    quality = orthogonal_quality(reconstructed_mesh)
    worst = int(np.argmin(quality))
    return {
        "overall": "PASS" if np.all(quality >= threshold) else "FAIL",
        "passed": bool(np.all(quality >= threshold)),
        "required_minimum": threshold,
        "minimum_orthogonal_quality": float(quality[worst]),
        "mean_orthogonal_quality": float(quality.mean()),
        "cells_below_threshold": int(np.count_nonzero(quality < threshold)),
        "maximum_stored_vs_reconstructed_centre_delta_m": float(
            np.linalg.norm(
                mesh.cell_centres_m - reconstructed_centres,
                axis=1,
            ).max()
        ),
        "worst_cell_tag": worst + 1,
        "worst_cell_centre_m": reconstructed_centres[worst].tolist(),
        "method": (
            "cell centres independently reconstructed from Hex8 geometry; "
            "emitted Fluent ASCII already strict-round-trip matched to these "
            "canonical nodes and faces"
        ),
    }


def _face_orientation_minimum(
    points: np.ndarray,
    hexes: np.ndarray,
    faces: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    chunk_size: int = 100_000,
) -> float:
    minimum = float("inf")
    for start in range(0, len(faces), chunk_size):
        stop = min(start + chunk_size, len(faces))
        vertices = points[faces[start:stop].astype(np.int64) - 1]
        area = 0.5 * (
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            + np.cross(vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0])
        )
        face_centres = vertices.mean(axis=1)
        c0_centres = points[hexes[c0[start:stop] - 1].astype(np.int64) - 1].mean(axis=1)
        toward_c0 = np.einsum("ij,ij->i", area, c0_centres - face_centres)
        _require(np.all(np.isfinite(toward_c0)), "non-finite face orientation")
        _require(np.all(toward_c0 > 0.0), "a face right-hand normal does not point toward c0")
        internal = c1[start:stop] > 0
        if np.any(internal):
            c1_centres = points[
                hexes[c1[start:stop][internal] - 1].astype(np.int64) - 1
            ].mean(axis=1)
            away_from_c1 = np.einsum(
                "ij,ij->i",
                area[internal],
                c1_centres - face_centres[internal],
            )
            _require(np.all(away_from_c1 < 0.0), "an internal face does not separate c0 and c1")
        minimum = min(minimum, float(toward_c0.min()))
    return minimum


def build_fluent_legacy_mesh(npz_path: Path) -> FluentLegacyMesh:
    """Build and validate the complete face-owner representation Fluent stores."""
    points, hexes, stored_centres, boundaries = _load_canonical(Path(npz_path))
    occurrences = hexes[:, HEX_FACES_INWARD].reshape(-1, 4)
    keys = np.sort(occurrences, axis=1)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    _require(np.all((counts == 1) | (counts == 2)), "a face has other than one or two owners")

    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    owners = np.repeat(np.arange(1, len(hexes) + 1, dtype=np.int64), 6)
    first = order[starts]
    faces = occurrences[first].astype(np.uint64, copy=False)
    c0 = owners[first]
    c1 = np.zeros(len(unique), dtype=np.int64)
    internal = counts == 2
    c1[internal] = owners[order[starts[internal] + 1]]
    _require(np.all(c0 != c1), "a face lists the same cell on both sides")

    boundary_by_key: dict[tuple[int, int, int, int], str] = {}
    for name, quads in boundaries.items():
        for row in np.sort(quads, axis=1):
            key = tuple(int(value) for value in row)
            _require(key not in boundary_by_key, f"duplicate declared boundary face {key}")
            boundary_by_key[key] = name

    zone_face_indices: dict[str, np.ndarray] = {
        "interior": np.flatnonzero(internal)
    }
    classified: dict[str, list[int]] = {name: [] for name in boundaries}
    for face_index in np.flatnonzero(~internal):
        key = tuple(int(value) for value in unique[face_index])
        name = boundary_by_key.get(key)
        _require(name is not None, f"undeclared exterior face {key}")
        classified[name].append(int(face_index))
    for name, indices in classified.items():
        zone_face_indices[name] = np.asarray(indices, dtype=np.int64)
    _require(
        sum(len(indices) for name, indices in zone_face_indices.items() if name != "interior")
        == len(boundary_by_key),
        "declared and actual exterior face sets differ",
    )
    _require(
        len(boundary_by_key) == int(np.count_nonzero(~internal)),
        "a declared boundary face is absent from the Hex8 census",
    )

    incidence = np.bincount(
        np.concatenate((c0, c1[c1 > 0])),
        minlength=len(hexes) + 1,
    )[1:]
    _require(np.all(incidence == 6), "each Hex8 cell must have exactly six incident faces")
    orientation = _face_orientation_minimum(points, hexes, faces, c0, c1)
    return FluentLegacyMesh(
        points_m=points,
        hexes=hexes,
        cell_centres_m=(
            stored_centres
            if stored_centres is not None
            else _volume_centres(points, hexes)
        ),
        faces=faces,
        c0=c0,
        c1=c1,
        zone_face_indices=zone_face_indices,
        minimum_c0_orientation_dot=orientation,
    )


def _face_header(zone_id: int, first: int, count: int, bc_type: int) -> str:
    last = first + count - 1
    return f"(13 ({zone_id:x} {first:x} {last:x} {bc_type:x} 4)(\n"


def _write_ascii(stream: Any, mesh: FluentLegacyMesh) -> None:
    node_count = len(mesh.points_m)
    cell_count = len(mesh.hexes)
    face_count = len(mesh.faces)
    stream.write('(0 "Eccentric conical bearing; native Fluent legacy ASCII; SI metres")\n')
    stream.write("(2 3)\n")
    stream.write(f"(10 (0 1 {node_count:x} 1 3))\n")
    stream.write(f"(12 (0 1 {cell_count:x} 0))\n")
    stream.write(f"(13 (0 1 {face_count:x} 0))\n")
    stream.write(f"(10 (1 1 {node_count:x} 1 3)(\n")
    for point in mesh.points_m:
        stream.write(f"{point[0]:.17e} {point[1]:.17e} {point[2]:.17e}\n")
    stream.write("))\n")
    stream.write(f"(12 ({FLUID_ZONE_ID:x} 1 {cell_count:x} 1 4))\n")

    first = 1
    for source_name, _, zone_id, bc_type, _ in FACE_ZONES:
        indices = mesh.zone_face_indices[source_name]
        _require(len(indices) > 0, f"Fluent face zone {source_name} is empty")
        stream.write(_face_header(zone_id, first, len(indices), bc_type))
        for face_index in indices:
            nodes = mesh.faces[face_index]
            stream.write(
                " ".join(
                    f"{int(value):x}"
                    for value in (*nodes, mesh.c0[face_index], mesh.c1[face_index])
                )
                + "\n"
            )
        stream.write("))\n")
        first += len(indices)
    _require(first - 1 == face_count, "face zone ranges do not cover every face")

    stream.write(f"(45 ({FLUID_ZONE_ID} fluid fluid 1)())\n")
    for _, fluent_name, zone_id, _, zone_type in FACE_ZONES:
        stream.write(f"(45 ({zone_id} {zone_type} {fluent_name} 1)())\n")


def _expect_line(stream: Any, expected: str) -> None:
    actual = stream.readline()
    _require(actual == expected, f"written Fluent mesh differs at {actual.rstrip()!r}")


def audit_written_mesh(
    path: Path,
    mesh: FluentLegacyMesh,
    minimum_orthogonal_quality: float = 0.8,
) -> dict[str, Any]:
    """Strictly reparse the emitted subset and compare it with the in-memory model."""
    _require(
        0.0 < minimum_orthogonal_quality <= 1.0,
        "minimum Orthogonal Quality must be in (0, 1]",
    )
    with Path(path).open("r", encoding="ascii", newline="\n") as stream:
        _expect_line(
            stream,
            '(0 "Eccentric conical bearing; native Fluent legacy ASCII; SI metres")\n',
        )
        _expect_line(stream, "(2 3)\n")
        _expect_line(stream, f"(10 (0 1 {len(mesh.points_m):x} 1 3))\n")
        _expect_line(stream, f"(12 (0 1 {len(mesh.hexes):x} 0))\n")
        _expect_line(stream, f"(13 (0 1 {len(mesh.faces):x} 0))\n")
        _expect_line(stream, f"(10 (1 1 {len(mesh.points_m):x} 1 3)(\n")
        for expected in mesh.points_m:
            values = np.fromstring(stream.readline(), sep=" ")
            _require(
                values.shape == (3,) and np.array_equal(values, expected),
                "node coordinates changed during ASCII round trip",
            )
        _expect_line(stream, "))\n")
        _expect_line(stream, f"(12 ({FLUID_ZONE_ID:x} 1 {len(mesh.hexes):x} 1 4))\n")

        first = 1
        for source_name, _, zone_id, bc_type, _ in FACE_ZONES:
            indices = mesh.zone_face_indices[source_name]
            _expect_line(stream, _face_header(zone_id, first, len(indices), bc_type))
            for face_index in indices:
                values = np.asarray(
                    [int(token, 16) for token in stream.readline().split()],
                    dtype=np.int64,
                )
                expected = np.asarray(
                    (*mesh.faces[face_index], mesh.c0[face_index], mesh.c1[face_index]),
                    dtype=np.int64,
                )
                _require(
                    values.shape == (6,) and np.array_equal(values, expected),
                    "face connectivity changed during ASCII round trip",
                )
            _expect_line(stream, "))\n")
            first += len(indices)
        _expect_line(stream, f"(45 ({FLUID_ZONE_ID} fluid fluid 1)())\n")
        for _, fluent_name, zone_id, _, zone_type in FACE_ZONES:
            _expect_line(stream, f"(45 ({zone_id} {zone_type} {fluent_name} 1)())\n")
        _require(stream.readline() == "", "unexpected trailing Fluent mesh data")

    return {
        "overall": "STATIC_PASS_FLUENT_NOT_RUN",
        "format": "ANSYS Fluent legacy ASCII mesh",
        "coordinate_unit": "m",
        "nodes": len(mesh.points_m),
        "hex8_cells": len(mesh.hexes),
        "quad4_faces": len(mesh.faces),
        "minimum_c0_orientation_dot_m3": mesh.minimum_c0_orientation_dot,
        "face_zones": {
            fluent_name: len(mesh.zone_face_indices[source_name])
            for source_name, fluent_name, *_ in FACE_ZONES
        },
        "bbox_m": [
            *mesh.points_m.min(axis=0).tolist(),
            *mesh.points_m.max(axis=0).tolist(),
        ],
        "fluent_equivalent_orthogonal_quality": (
            orthogonal_quality_summary(
                mesh,
                threshold=minimum_orthogonal_quality,
            )
        ),
    }


def write_fluent_legacy_mesh(
    npz_path: Path,
    output_path: Path,
    minimum_orthogonal_quality: float = 0.8,
) -> dict[str, Any]:
    mesh = build_fluent_legacy_mesh(npz_path)
    mesh = replace(
        mesh,
        cell_centres_m=_volume_centres(mesh.points_m, mesh.hexes),
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            _write_ascii(stream, mesh)
        report = audit_written_mesh(
            Path(temporary_name),
            mesh,
            minimum_orthogonal_quality,
        )
        quality = report["fluent_equivalent_orthogonal_quality"]
        _require(
            quality["threshold_passed"],
            f"minimum Orthogonal Quality {quality['minimum']:.12g} is below "
            f"required {minimum_orthogonal_quality:.12g}",
        )
        os.replace(temporary_name, output_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return report | {
        "source_npz": str(Path(npz_path).resolve()),
        "source_npz_sha256": _sha256(Path(npz_path)),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write native Fluent legacy ASCII from canonical Hex8 NPZ arrays."
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-oq", type=float, default=0.8)
    args = parser.parse_args(argv)
    try:
        report = write_fluent_legacy_mesh(
            args.npz,
            args.out,
            args.min_oq,
        )
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text, encoding="utf-8")
        print(text, end="")
    except Exception as error:
        print(f"FLUENT LEGACY MESH FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
