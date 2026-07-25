#!/usr/bin/env python3
"""Correct Gmsh surface-BC locations in HDF5-backed CGNS files."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


SURFACE_ELEMENT_TYPES = {5, 7}  # TRI_3, QUAD_4


class CgnsCompatibilityError(RuntimeError):
    """A CGNS file does not match the narrowly supported Gmsh layout."""


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii")
    return str(value)


def _label(node: h5py.Group) -> str:
    return _text(node.attrs.get("label", ""))


def _dataset_text(dataset: h5py.Dataset) -> str:
    return bytes(np.asarray(dataset[...], dtype=np.uint8)).decode("ascii")


def _mesh_payload_digest(
    root: h5py.File, excluded_datasets: set[str]
) -> str:
    """Hash every dataset except the exact surface locations being corrected."""
    digest = hashlib.sha256()

    def visit(name: str, node: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(node, h5py.Dataset) or name in excluded_datasets:
            return
        values = np.asarray(node[...])
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())

    root.visititems(visit)
    return digest.hexdigest()


def _surface_boundary_locations(root: h5py.File) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    bases = [node for node in root.values() if isinstance(node, h5py.Group) and _label(node) == "CGNSBase_t"]
    if len(bases) != 1:
        raise CgnsCompatibilityError(f"expected one CGNSBase_t, found {len(bases)}")

    zones = [node for node in bases[0].values() if isinstance(node, h5py.Group) and _label(node) == "Zone_t"]
    if len(zones) != 1:
        raise CgnsCompatibilityError(f"expected one Zone_t, found {len(zones)}")
    zone = zones[0]

    surface_ranges: list[tuple[int, int, str]] = []
    for section in zone.values():
        if not isinstance(section, h5py.Group) or _label(section) != "Elements_t":
            continue
        element_type = int(np.asarray(section[" data"][...]).reshape(-1)[0])
        if element_type not in SURFACE_ELEMENT_TYPES:
            continue
        start, end = (int(value) for value in np.asarray(section["ElementRange"][" data"][...]).reshape(-1))
        surface_ranges.append((start, end, section.name))

    zone_bc = next(
        (
            node
            for node in zone.values()
            if isinstance(node, h5py.Group) and _label(node) == "ZoneBC_t"
        ),
        None,
    )
    if zone_bc is None:
        raise CgnsCompatibilityError("CGNS zone has no ZoneBC_t")

    for boundary in zone_bc.values():
        if not isinstance(boundary, h5py.Group) or _label(boundary) != "BC_t":
            continue
        if "PointRange" not in boundary or "GridLocation" not in boundary:
            continue
        start, end = (
            int(value)
            for value in np.asarray(boundary["PointRange"][" data"][...]).reshape(-1)
        )
        section = next(
            (name for lower, upper, name in surface_ranges if lower <= start and end <= upper),
            None,
        )
        if section is None:
            continue
        location = boundary["GridLocation"][" data"]
        records.append(
            {
                "boundary": boundary.name,
                "section": section,
                "range": [start, end],
                "location": _dataset_text(location),
                "dataset": location,
            }
        )
    return records


def audit_surface_boundary_locations(path: Path) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as root:
        records = _surface_boundary_locations(root)
        return {
            "path": str(path),
            "surface_boundaries": len(records),
            "face_center": sum(item["location"] == "FaceCenter" for item in records),
            "cell_center": sum(item["location"] == "CellCenter" for item in records),
            "other": sorted(
                {item["location"] for item in records}
                - {"FaceCenter", "CellCenter"}
            ),
            "records": [
                {key: value for key, value in item.items() if key != "dataset"}
                for item in records
            ],
        }


def sanitize_gmsh_cgns(
    path: Path,
    *,
    expected_surface_boundaries: int | None = None,
) -> dict[str, Any]:
    """Atomically change surface BC GridLocation from CellCenter to FaceCenter."""
    path = Path(path).resolve()
    if not path.is_file():
        raise CgnsCompatibilityError(f"CGNS file does not exist: {path}")

    with h5py.File(path, "r") as source:
        source_records = _surface_boundary_locations(source)
        surface_location_paths = {
            item["dataset"].name.lstrip("/") for item in source_records
        }
        version_before = np.asarray(source["CGNSLibraryVersion"][" data"][...]).copy()
        payload_before = _mesh_payload_digest(source, surface_location_paths)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(path, temporary)
        with h5py.File(temporary, "r+") as target:
            records = _surface_boundary_locations(target)
            if expected_surface_boundaries is not None and len(records) != expected_surface_boundaries:
                raise CgnsCompatibilityError(
                    f"expected {expected_surface_boundaries} surface BCs, found {len(records)}"
                )
            invalid = [
                item for item in records if item["location"] not in {"CellCenter", "FaceCenter"}
            ]
            if invalid:
                raise CgnsCompatibilityError(
                    f"unsupported surface GridLocation values: {[item['location'] for item in invalid]}"
                )
            corrected = [item["boundary"] for item in records if item["location"] == "CellCenter"]
            face_center = np.frombuffer(b"FaceCenter", dtype=np.int8)
            for item in records:
                if item["location"] == "CellCenter":
                    dataset = item["dataset"]
                    if dataset.shape != (10,) or dataset.dtype != np.dtype("int8"):
                        raise CgnsCompatibilityError(
                            f"unexpected GridLocation storage at {item['boundary']}"
                        )
                    dataset[...] = face_center
            target.flush()

        after = audit_surface_boundary_locations(temporary)
        if after["face_center"] != after["surface_boundaries"] or after["cell_center"]:
            raise CgnsCompatibilityError("surface BC correction did not persist")
        with h5py.File(temporary, "r") as target:
            if not np.array_equal(
                version_before, np.asarray(target["CGNSLibraryVersion"][" data"][...])
            ):
                raise CgnsCompatibilityError("CGNS library version changed")
            if payload_before != _mesh_payload_digest(target, surface_location_paths):
                raise CgnsCompatibilityError("CGNS mesh payload changed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        **audit_surface_boundary_locations(path),
        "corrected": corrected,
        "mesh_payload_unchanged": True,
        "cgns_library_version_unchanged": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct Gmsh CGNS surface BC locations without changing the mesh."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--expected-surface-boundaries", type=int)
    args = parser.parse_args(argv)
    try:
        for path in args.files:
            report = sanitize_gmsh_cgns(
                path,
                expected_surface_boundaries=args.expected_surface_boundaries,
            )
            print(
                f"{path}: PASS, FaceCenter={report['face_center']}, "
                f"corrected={len(report['corrected'])}"
            )
    except Exception as error:
        print(f"CGNS COMPATIBILITY FAIL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
