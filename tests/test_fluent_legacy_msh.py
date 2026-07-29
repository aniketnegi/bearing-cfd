from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interchange.fluent_legacy_msh import (
    FluentLegacyMeshError,
    write_fluent_legacy_mesh,
)


OUTWARD = np.asarray(
    (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
    )
)


def _two_hex_npz(
    path: Path,
    *,
    omit_boundary: bool = False,
    skew: bool = False,
    forged_centres: bool = False,
) -> None:
    points = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (2, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (2, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 1),
        ],
        dtype=np.float64,
    )
    if skew:
        points[11, 0] = 2.25
    hexes = np.asarray(
        ((1, 2, 5, 4, 7, 8, 11, 10), (2, 3, 6, 5, 8, 9, 12, 11)),
        dtype=np.uint64,
    )
    occurrences = hexes[:, OUTWARD].reshape(-1, 4)
    keys, counts = np.unique(np.sort(occurrences, axis=1), axis=0, return_counts=True)
    exterior = keys[counts == 1]
    if omit_boundary:
        exterior = exterior[:-1]
    boundaries = {
        "journal_wall": exterior[0:2],
        "stationary_wall": exterior[2:6],
        "axial_end_z0": exterior[6:7],
        "axial_end_zL": exterior[7:8],
        "pressure_feed": exterior[8:],
    }
    arrays = {
        "points_m": points,
        "hexes": hexes,
        "node_tags": np.arange(1, len(points) + 1, dtype=np.uint64),
        "cell_tags": np.arange(1, len(hexes) + 1, dtype=np.uint64),
        **{f"boundary_{name}": values for name, values in boundaries.items()},
    }
    if forged_centres:
        arrays["cell_centres_m"] = np.asarray(
            ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5)),
            dtype=np.float64,
        )
    np.savez(
        path,
        **arrays,
    )


def test_native_fluent_ascii_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "mesh_arrays.npz"
    output = tmp_path / "bearing.msh"
    _two_hex_npz(source)
    report = write_fluent_legacy_mesh(source, output)

    assert report["overall"] == "STATIC_PASS_FLUENT_NOT_RUN"
    assert report["nodes"] == 12
    assert report["hex8_cells"] == 2
    assert report["quad4_faces"] == 11
    assert report["face_zones"]["interior"] == 1
    assert report["face_zones"]["pressure_feed"] == 2
    assert report["fluent_equivalent_orthogonal_quality"]["minimum"] == pytest.approx(1.0)
    assert report["fluent_equivalent_orthogonal_quality"]["threshold_passed"] is True
    text = output.read_text()
    for record in (
        "(45 (2 interior interior 1)())",
        "(45 (3 wall journal_wall 1)())",
        "(45 (4 wall stationary_wall 1)())",
        "(45 (5 pressure-outlet axial_end_z0 1)())",
        "(45 (6 pressure-outlet axial_end_zl 1)())",
        "(45 (7 pressure-inlet pressure_feed 1)())",
        "(45 (8 fluid fluid 1)())",
    ):
        assert record in text


def test_missing_boundary_face_fails(tmp_path: Path) -> None:
    source = tmp_path / "bad.npz"
    output = tmp_path / "bad.msh"
    _two_hex_npz(source, omit_boundary=True)
    with pytest.raises(FluentLegacyMeshError, match="undeclared exterior face"):
        write_fluent_legacy_mesh(source, output)
    assert not output.exists()


def test_minimum_orthogonal_quality_gate_prevents_publish(
    tmp_path: Path,
) -> None:
    source = tmp_path / "skewed.npz"
    output = tmp_path / "rejected.msh"
    _two_hex_npz(source, skew=True)

    with pytest.raises(
        FluentLegacyMeshError,
        match="minimum Orthogonal Quality .* is below required",
    ):
        write_fluent_legacy_mesh(
            source,
            output,
            minimum_orthogonal_quality=0.99,
        )
    assert not output.exists()


def test_stored_centres_cannot_override_orthogonal_quality(
    tmp_path: Path,
) -> None:
    source = tmp_path / "skewed.npz"
    output = tmp_path / "rejected.msh"
    _two_hex_npz(source, skew=True, forged_centres=True)

    with pytest.raises(
        FluentLegacyMeshError,
        match="minimum Orthogonal Quality .* is below required",
    ):
        write_fluent_legacy_mesh(
            source,
            output,
            minimum_orthogonal_quality=0.99,
        )
    assert not output.exists()
