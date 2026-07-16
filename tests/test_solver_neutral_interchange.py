from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import gmsh
import numpy as np
import pytest

from interchange import fluent_import_check as fluent
from interchange.export_solver_neutral import (
    PATCHES,
    CanonicalCase,
    InterchangeError,
    InterchangeInputs,
    audit_canonical,
    enforce_fluent_mode,
    load_canonical_case,
    run_interchange,
    validate_group_names,
)
from meshing.layered_prism_central_feed import validate_external_face_orientation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINI_CASE = PROJECT_ROOT / "out" / "ported_prism_probe" / "nGap_02"


@pytest.fixture(scope="module")
def clean_export(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    outdir = tmp_path_factory.mktemp("solver-neutral") / "nGap_02"
    report = run_interchange(
        InterchangeInputs(case_dir=MINI_CASE, outdir=outdir, fluent="skip")
    )
    return {"outdir": outdir, "report": report}


def test_clean_default_mini_mesh_exports_all_formats(clean_export: dict[str, Any]) -> None:
    outdir = clean_export["outdir"]
    for name in (
        "bearing_prism_gmsh41.msh",
        "bearing_prism_gmsh22_ascii.msh",
        "bearing_prism.cgns",
        "interchange_report.json",
        "interchange_manifest.json",
        "zones.csv",
        "file_hashes.json",
        "README_OPEN_ME_FIRST.txt",
    ):
        assert (outdir / name).is_file()
    assert clean_export["report"]["overall"] == "STATIC_PASS_FLUENT_NOT_RUN"
    assert clean_export["report"]["readiness"] == "STATICALLY_VALIDATED_NOT_IMPORTED"


def test_all_three_round_trips_retain_counts_units_and_connectivity(
    clean_export: dict[str, Any],
) -> None:
    report = clean_export["report"]
    source = report["source"]
    assert [item["format"] for item in report["exports"]] == [
        "GMSH_4_1",
        "GMSH_2_2_ASCII",
        "CGNS",
    ]
    for item in report["exports"]:
        assert item["points"] == source["points"]
        assert item["prism6_cells"] == source["prism6_cells"]
        assert item["patch_counts"] == source["patch_counts"]
        assert item["mouth_internal_faces"] == source["mouth_internal_faces"]
        assert item["bounding_box_m"] == pytest.approx(source["bounding_box_m"], abs=1.0e-14)
        assert item["volume_m3"] == pytest.approx(source["volume_m3"], rel=1.0e-12)
        assert item["minimum_signed_volume_m3"] > 0.0
        assert item["connected_regions"] == 1
        assert item["units"] == "m"
        assert item["coordinate_max_error_m"] <= 1.0e-14
        assert item["solution_fields"] == 0


def test_physical_groups_are_exact_and_mouth_is_not_a_boundary(
    clean_export: dict[str, Any],
) -> None:
    expected = set(PATCHES) | {"fluid"}
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        for name in (
            "bearing_prism_gmsh41.msh",
            "bearing_prism_gmsh22_ascii.msh",
            "bearing_prism.cgns",
        ):
            gmsh.open(str(clean_export["outdir"] / name))
            groups = {
                gmsh.model.getPhysicalName(int(dimension), int(tag))
                for dimension, tag in gmsh.model.getPhysicalGroups()
            }
            assert groups == expected
            assert not {"feed_mouth", "mouth_cap", "internal_feed", "defaultFaces"} & groups
            assert len(gmsh.view.getTags()) == 0
            gmsh.clear()
    finally:
        gmsh.finalize()


def test_mouth_has_two_incident_cells_and_pressure_paths_exist(
    clean_export: dict[str, Any],
) -> None:
    checks = clean_export["report"]["source"]["connectivity_checks"]
    assert checks["mouth_two_incident_cells"]
    assert checks["mouth_is_internal"]
    assert checks["pressure_feed_through_mouth_to_film"]
    assert checks["pressure_feed_to_axial_end_z0"]
    assert checks["pressure_feed_to_axial_end_zL"]
    assert checks["one_connected_fluid_region"]


def test_every_exported_boundary_face_is_outward(clean_export: dict[str, Any]) -> None:
    for item in clean_export["report"]["exports"]:
        assert set(item["external_face_minimum_outward_projection_m"]) == set(PATCHES)
        assert all(
            value > 0.0
            for value in item["external_face_minimum_outward_projection_m"].values()
        )


def test_reversed_face_and_negative_cell_are_rejected(tmp_path: Path) -> None:
    case = load_canonical_case(MINI_CASE)
    triangles = {name: faces.copy() for name, faces in case.mesh.boundary_triangles.items()}
    triangles["journal_wall"][0] = triangles["journal_wall"][0, ::-1]
    reversed_mesh = replace(case.mesh, boundary_triangles=triangles)
    with pytest.raises(RuntimeError, match="external_face_orientation.journal_wall"):
        validate_external_face_orientation(reversed_mesh, [])

    case = load_canonical_case(MINI_CASE)
    prisms = case.mesh.prisms.copy()
    prisms[0, [0, 1]] = prisms[0, [1, 0]]
    negative = CanonicalCase(case.path, replace(case.mesh, prisms=prisms), case.manifest, case.report)
    with pytest.raises(InterchangeError, match="nonpositive Prism6"):
        audit_canonical(negative, InterchangeInputs(MINI_CASE, tmp_path / "negative"))


def test_missing_patch_and_duplicate_cell_are_rejected(tmp_path: Path) -> None:
    case = load_canonical_case(MINI_CASE)
    triangles = dict(case.mesh.boundary_triangles)
    triangles.pop("pressure_feed")
    missing = CanonicalCase(case.path, replace(case.mesh, boundary_triangles=triangles), case.manifest, case.report)
    with pytest.raises(InterchangeError, match="all six external patches"):
        audit_canonical(missing, InterchangeInputs(MINI_CASE, tmp_path / "missing"))

    case = load_canonical_case(MINI_CASE)
    prisms = case.mesh.prisms.copy()
    prisms[1] = prisms[0]
    duplicate = CanonicalCase(case.path, replace(case.mesh, prisms=prisms), case.manifest, case.report)
    with pytest.raises(InterchangeError, match="duplicate Prism6 cells"):
        audit_canonical(duplicate, InterchangeInputs(MINI_CASE, tmp_path / "duplicate"))


def test_cgns_boundary_name_loss_is_rejected() -> None:
    with pytest.raises(InterchangeError, match="GEOMETRY_ONLY_NOT_FLUENT_READY"):
        validate_group_names((set(PATCHES) - {"pressure_feed"}) | {"fluid"}, "CGNS")


def test_fluent_unavailable_auto_skip_and_required_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        fluent,
        "fluent_capability",
        lambda: {"available": False, "backend": None, "reason": "not installed"},
    )
    result = fluent.run_fluent_import_audit(
        cgns=tmp_path / "unused.cgns",
        canonical={},
        outdir=tmp_path,
    )
    assert result == {"status": "NOT_RUN", "reason": "not installed", "real_import": False}
    assert enforce_fluent_mode("auto", result) is False
    with pytest.raises(InterchangeError, match="--fluent required"):
        enforce_fluent_mode("required", result)


@pytest.mark.integration
def test_real_fluent_import_when_available(clean_export: dict[str, Any]) -> None:
    capability = fluent.fluent_capability()
    if not capability["available"]:
        pytest.skip(capability["reason"])
    result = fluent.run_fluent_import_audit(
        cgns=clean_export["outdir"] / "bearing_prism.cgns",
        canonical=clean_export["report"]["source"],
        outdir=clean_export["outdir"],
    )
    if result["status"] == "NOT_RUN":
        pytest.skip(result["reason"])
    assert result["status"] == "PASS", result
    assert result["real_import"] is True
