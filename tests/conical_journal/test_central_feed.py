from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import gmsh
import numpy as np
import pytest

import bearing_cfd.bearings.conical_journal.meshing.central_feed as ported
from bearing_cfd.bearings.conical_journal.meshing.central_feed import (
    PRISM_TRI_FACES,
    PortedInputs,
    PortedMeshError,
    PortedRunError,
    add_discrete_prism_model,
    add_element_data_views,
    add_gmsh_quality,
    build_master_mesh,
    build_prism_mesh,
    expected_physical_groups,
    load_contract,
    open_optional_gui,
    rim_coordinates,
    validate_geometry,
    validate_external_face_orientation,
    validate_gmsh_round_trip,
    validate_inputs,
    validate_npz_round_trip,
    validate_physical_groups,
    validate_topology,
    validate_vtu_round_trip,
    write_gmsh_files,
    write_mesh_arrays,
    write_vtu_files,
    write_visualizations,
)
from bearing_cfd.bearings.conical_journal.meshing.gap_grading import (
    symmetric_gap_coordinates,
)


pytestmark = pytest.mark.integration


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "conical_journal"
CASES = {
    "strict_default": (
        FIXTURES / "geometry" / "strict_default",
        FIXTURES / "meshing" / "brep_preflight" / "preflight_report.json",
    ),
    "strict_case_e03_g20": (
        FIXTURES / "geometry" / "strict_case_e03_g20",
        FIXTURES
        / "meshing"
        / "brep_preflight_e03_g20"
        / "preflight_report.json",
    ),
}


def _record(bundle: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in bundle["records"] if item["name"] == name]
    assert len(matches) == 1, name
    return matches[0]


def _build_bundle(case: str, directory: Path) -> dict[str, Any]:
    source, preflight = CASES[case]
    inputs = PortedInputs(
        params=source / "params.json",
        brep=source / "film_unsplit.brep",
        preflight=preflight,
        outdir=directory / "published",
        n_theta=128,
        n_axial=12,
        gap_levels=(2,),
        preview_ngap=2,
        rim_segments=128,
        tube_layers=2,
        openfoam="skip",
    )
    records: list[dict[str, Any]] = []
    params, _raw, _preflight = load_contract(inputs, records)
    validate_inputs(inputs, params)

    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        master_path = directory / "master_surface.msh"
        master = build_master_mesh(params, inputs, master_path, records)
        mesh = build_prism_mesh(
            master, params, n_gap=2, tube_layers=2, tube_grading=1.0
        )
        topology = validate_topology(mesh, master, records)
        geometry = validate_geometry(mesh, master, params, inputs, records)

        gmsh.clear()
        discrete = add_discrete_prism_model(mesh, records, f"test_{case}")
        physical = validate_physical_groups(records, "test_model")
        mesh = add_gmsh_quality(mesh, records)
        views = add_element_data_views(mesh)
        msh41, msh22 = write_gmsh_files(directory, views)
        gmsh_round_trips = {
            "msh41": validate_gmsh_round_trip(
                msh41, mesh, records, "test_round_trip.msh41"
            ),
            "msh22": validate_gmsh_round_trip(
                msh22, mesh, records, "test_round_trip.msh22"
            ),
        }
    finally:
        gmsh.finalize()

    volume_vtu, boundary_vtu = write_vtu_files(mesh, directory)
    vtu = validate_vtu_round_trip(
        mesh, volume_vtu, boundary_vtu, records
    )
    npz_path = directory / "mesh_arrays.npz"
    write_mesh_arrays(mesh, npz_path)
    npz = validate_npz_round_trip(mesh, npz_path, records)
    assert all(item["status"] == "PASS" for item in records)
    return {
        "case": case,
        "inputs": inputs,
        "params": params,
        "master": master,
        "mesh": mesh,
        "topology": topology,
        "geometry": geometry,
        "discrete": discrete,
        "physical": physical,
        "gmsh_round_trips": gmsh_round_trips,
        "vtu": vtu,
        "npz": npz,
        "records": records,
        "directory": directory,
    }


@pytest.fixture(scope="module")
def default_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _build_bundle(
        "strict_default", tmp_path_factory.mktemp("ported-prism-default")
    )


@pytest.fixture(scope="module")
def second_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _build_bundle(
        "strict_case_e03_g20", tmp_path_factory.mktemp("ported-prism-e03-g20")
    )


@pytest.mark.parametrize("keep", ["hole_radius", "hole_diameter"])
def test_contract_accepts_radius_only_or_diameter_only_parameter_records(
    keep: str, tmp_path: Path
) -> None:
    source, preflight = CASES["strict_default"]
    raw = json.loads((source / "params.json").read_text(encoding="utf-8"))
    raw["resolved_parameters"].pop(
        "hole_diameter" if keep == "hole_radius" else "hole_radius"
    )
    params_path = tmp_path / f"params_{keep}.json"
    params_path.write_text(json.dumps(raw), encoding="utf-8")
    params, _raw, _preflight = load_contract(
        PortedInputs(
            params=params_path,
            brep=source / "film_unsplit.brep",
            preflight=preflight,
        ),
        [],
    )
    assert params.hole_radius_mm == pytest.approx(2.0)


def test_concentric_case_is_rejected_explicitly(default_bundle: dict[str, Any]) -> None:
    params = replace(
        default_bundle["params"],
        eccentricity_mm=0.0,
        eccentricity_ratio=0.0,
        ex_mm=0.0,
        ey_mm=0.0,
    )
    with pytest.raises(PortedMeshError, match="concentric geometry is unsupported"):
        validate_inputs(default_bundle["inputs"], params)


def test_analytic_mouth_rim_and_closed_degree_two_loop(
    default_bundle: dict[str, Any],
) -> None:
    params = default_bundle["params"]
    master = default_bundle["master"]
    rim = rim_coordinates(params, 128)
    cylinder = rim["x"] ** 2 + (rim["z"] - params.hole_axial_pos_mm) ** 2
    bore = rim["x"] ** 2 + rim["y"] ** 2
    assert np.max(np.abs(cylinder - params.hole_radius_mm**2)) <= 1.0e-10
    assert np.max(
        np.abs(bore - np.asarray(params.bore_radius_mm(rim["z"])) ** 2)
    ) <= 1.0e-10
    assert np.all(rim["y"] > 0.0)
    assert master.metadata["rim_sagitta_mm"] == pytest.approx(
        params.hole_radius_mm * (1.0 - math.cos(math.pi / 128)), abs=1.0e-15
    )
    assert master.metadata["rim_sagitta_mm"] <= 0.001

    degree = np.bincount(master.rim_edges.ravel().astype(np.int64))
    assert np.all(degree[master.rim_nodes.astype(np.int64)] == 2)
    neighbours: dict[int, set[int]] = {}
    for left, right in master.rim_edges:
        neighbours.setdefault(int(left), set()).add(int(right))
        neighbours.setdefault(int(right), set()).add(int(left))
    seen = {int(master.rim_nodes[0])}
    pending = list(seen)
    while pending:
        for node in neighbours[pending.pop()] - seen:
            seen.add(node)
            pending.append(node)
    assert seen == set(master.rim_nodes.astype(int))


def test_symmetric_through_gap_inflation(
    default_bundle: dict[str, Any],
) -> None:
    ratio = 5.0
    xi = symmetric_gap_coordinates(4, ratio)
    fractions = np.diff(xi)
    assert fractions == pytest.approx(fractions[::-1], abs=1.0e-15)
    assert fractions.sum() == pytest.approx(1.0, abs=1.0e-15)
    assert fractions.max() / fractions.min() == pytest.approx(ratio, rel=1.0e-12)

    master = default_bundle["master"]
    mesh = build_prism_mesh(
        master,
        default_bundle["params"],
        n_gap=4,
        tube_layers=2,
        tube_grading=1.0,
        gap_inflation_ratio=ratio,
    )
    master_count = len(master.points_uz_mm)
    node = master.centre_node
    radial_line = mesh.points_m[
        np.asarray([layer * master_count + node for layer in range(5)])
    ]
    thicknesses = np.linalg.norm(np.diff(radial_line, axis=0), axis=1)
    assert thicknesses == pytest.approx(thicknesses[::-1], rel=1.0e-10)
    assert thicknesses.max() / thicknesses.min() == pytest.approx(
        ratio, rel=1.0e-10
    )
    assert mesh.metadata["gap_inflation_ratio_achieved"] == pytest.approx(ratio)


def test_periodic_seam_and_axial_partitions_are_internal(
    default_bundle: dict[str, Any],
) -> None:
    master = default_bundle["master"]
    assert len(master.seam_edges) > 0
    assert len(master.z1_edges) > 0
    assert len(master.z2_edges) > 0
    for name in (
        "master.periodic_seam_internal",
        "master.only_axial_external_edges",
        "topology.theta_seam_internal_conformal",
        "topology.z1_partition_internal_conformal",
        "topology.z2_partition_internal_conformal",
    ):
        assert _record(default_bundle, name)["status"] == "PASS"


def test_mouth_nodes_faces_and_continuous_journal(
    default_bundle: dict[str, Any],
) -> None:
    mesh = default_bundle["mesh"]
    master = default_bundle["master"]
    tri_faces = mesh.prisms[:, PRISM_TRI_FACES].reshape(-1, 3)
    owners = np.repeat(np.arange(len(mesh.prisms)), len(PRISM_TRI_FACES))
    census: dict[tuple[int, int, int], list[int]] = {}
    for face, owner in zip(tri_faces, owners):
        census.setdefault(tuple(sorted(map(int, face))), []).append(int(owner))

    boundary_keys = {
        tuple(sorted(map(int, face)))
        for faces in mesh.boundary_triangles.values()
        for face in faces
    }
    for mouth in mesh.mouth_triangles:
        key = tuple(sorted(map(int, mouth)))
        assert key not in boundary_keys
        mouth_owners = census[key]
        assert len(mouth_owners) == 2
        assert set(mesh.cell_fields["region_id"][mouth_owners]) == {0, 1}
        for node in mouth:
            assert np.any(mesh.prisms[mouth_owners[0]] == node)
            assert np.any(mesh.prisms[mouth_owners[1]] == node)

    assert len(np.unique(mesh.points_m, axis=0)) == len(mesh.points_m)
    journal = mesh.boundary_triangles["journal_wall"]
    expected = master.triangles[:, [0, 2, 1]] + 1
    assert len(journal) == mesh.master_triangle_count
    assert {
        tuple(sorted(map(int, face))) for face in journal
    } == {tuple(sorted(map(int, face))) for face in expected}


def test_one_component_and_complete_disjoint_boundaries(
    default_bundle: dict[str, Any],
) -> None:
    assert default_bundle["topology"]["connected_components"] == 1
    for name in (
        "topology.face_owner_counts",
        "topology.declared_boundaries_one_owner",
        "topology.boundary_groups_disjoint",
        "topology.boundary_union_complete",
        "topology.exact_patch_face_counts",
        "topology.pressure_to_film_path",
        "topology.only_Prism6_Tri3_Quad4",
    ):
        assert _record(default_bundle, name)["status"] == "PASS"
    assert default_bundle["physical"] == {
        f"{dimension}:{physical_id}": data
        for (dimension, physical_id), data in expected_physical_groups().items()
    }
    assert set(default_bundle["mesh"].boundary_triangles) == {
        "journal_wall",
        "bushing_bore_wall",
        "pressure_feed",
    }
    assert set(default_bundle["mesh"].boundary_quads) == {
        "axial_end_z0",
        "axial_end_zL",
        "feed_tube_wall",
    }
    counts = default_bundle["topology"]["boundary_face_counts"]
    assert set(counts) == set(ported.SURFACE_ENTITIES)
    assert sum(counts.values()) == default_bundle["topology"]["total_boundary_faces"]


def test_every_external_face_is_locally_outward_and_reversal_is_rejected(
    default_bundle: dict[str, Any],
) -> None:
    mesh = default_bundle["mesh"]
    orientation = default_bundle["geometry"]["external_face_orientation"]
    assert set(orientation) == set(ported.SURFACE_ENTITIES)
    assert all(item["minimum_outward_projection_m"] > 0.0 for item in orientation.values())

    triangles = {name: faces.copy() for name, faces in mesh.boundary_triangles.items()}
    triangles["journal_wall"][0] = triangles["journal_wall"][0, ::-1]
    reversed_mesh = replace(mesh, boundary_triangles=triangles)
    with pytest.raises(RuntimeError, match="external_face_orientation.journal_wall"):
        validate_external_face_orientation(reversed_mesh, [])


def test_positive_prism_jacobians_and_gmsh_quality(
    default_bundle: dict[str, Any],
) -> None:
    mesh = default_bundle["mesh"]
    assert mesh.prisms.shape[1] == 6
    assert np.all(np.isfinite(mesh.points_m))
    for name in ("volume_m3", "minDetJac", "minSICN", "aspect_ratio"):
        values = mesh.cell_fields[name]
        assert values.shape == (len(mesh.prisms),)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
    assert _record(default_bundle, "gmsh.quality.Prism6_positive")["status"] == "PASS"
    assert default_bundle["discrete"]["gmsh_generated_3d_mesh"] is False


def test_inlet_geometry_radial_gaps_and_volume_agreement(
    default_bundle: dict[str, Any],
) -> None:
    params = default_bundle["params"]
    mesh = default_bundle["mesh"]
    geometry = default_bundle["geometry"]
    inlet = geometry["inlet"]
    volumes = geometry["volumes"]
    assert inlet["polygon_relative_error"] <= 1.0e-10
    assert inlet["circle_relative_error"] <= 5.0e-4
    assert inlet["centroid_mm"] == pytest.approx(
        [0.0, params.y_feed_end_mm, params.hole_axial_pos_mm], abs=1.0e-10
    )
    assert inlet["mean_normal"] == pytest.approx([0.0, 1.0, 0.0], abs=1.0e-12)
    assert geometry["minimum_radial_gap_mm"] == pytest.approx(
        params.minimum_gap_mm, abs=1.0e-10
    )
    assert geometry["maximum_radial_gap_mm"] == pytest.approx(
        params.maximum_gap_mm, abs=1.0e-10
    )
    assert volumes["boundary_relative_error"] <= 1.0e-9
    assert volumes["native_brep_relative_error"] <= 5.0e-4
    pressure_points = mesh.points_m[
        mesh.boundary_triangles["pressure_feed"].astype(np.int64) - 1
    ]
    assert np.max(
        np.abs(pressure_points[:, :, 1] / ported.SI_PER_MM - params.y_feed_end_mm)
    ) <= 1.0e-10
    assert mesh.metadata["scale_to_m_applied_once"] == 0.001


def test_all_round_trips_and_both_strict_cases(
    default_bundle: dict[str, Any], second_bundle: dict[str, Any]
) -> None:
    for bundle in (default_bundle, second_bundle):
        mesh = bundle["mesh"]
        assert bundle["gmsh_round_trips"]["msh41"]["cell_count"] == len(mesh.prisms)
        assert bundle["gmsh_round_trips"]["msh22"]["cell_count"] == len(mesh.prisms)
        assert bundle["gmsh_round_trips"]["msh41"][
            "mouth_triangles_with_two_incident_cells"
        ] == len(mesh.mouth_triangles)
        assert bundle["gmsh_round_trips"]["msh22"][
            "mouth_triangles_with_two_incident_cells"
        ] == len(mesh.mouth_triangles)
        assert bundle["vtu"]["prisms"] == len(mesh.prisms)
        assert all(bundle["npz"]["checks"].values())
        assert bundle["geometry"]["volumes"]["native_brep_relative_error"] <= 5.0e-4
        for filename in (
            "ported_prism.msh",
            "ported_prism_openfoam.msh",
            "volume_prism.vtu",
            "boundary_faces.vtu",
            "mesh_arrays.npz",
        ):
            assert (bundle["directory"] / filename).is_file()
    assert default_bundle["params"].eccentricity_ratio == pytest.approx(0.6)
    assert second_bundle["params"].eccentricity_ratio == pytest.approx(0.3)
    assert second_bundle["params"].semicone_angle_deg == pytest.approx(20.0)


def test_diagnostic_msh_files_open_in_gmsh(
    default_bundle: dict[str, Any], tmp_path: Path
) -> None:
    bundle = default_bundle
    write_vtu_files(bundle["mesh"], tmp_path)
    write_visualizations(
        bundle["mesh"],
        bundle["master"],
        bundle["params"],
        tmp_path,
        bundle["directory"] / "master_surface.msh",
    )
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        for name in ("feed_cutaway_exact.msh", "feed_boundary_only.msh"):
            gmsh.open(str(tmp_path / "viz" / name))
            assert gmsh.model.getEntities()
            gmsh.clear()
    finally:
        gmsh.finalize()


def test_failure_atomically_removes_staged_solver_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, preflight = CASES["strict_default"]
    outdir = tmp_path / "result"
    stale = outdir / "nGap_01"
    stale.mkdir(parents=True)
    (stale / "ported_prism_openfoam.msh").write_text("stale", encoding="utf-8")
    observed: dict[str, bool] = {}

    def fail_after_staged_outputs(*args, **kwargs):
        output_path = args[2]
        staged = output_path.parent / "nGap_01"
        staged.mkdir(parents=True)
        for filename in ("ported_prism.msh", "volume_prism.vtu", "mesh_arrays.npz"):
            (staged / filename).write_text("staged", encoding="utf-8")
        observed["staged_outputs_existed"] = True
        raise RuntimeError("induced failure after staged solver artifacts")

    monkeypatch.setattr(ported, "build_master_mesh", fail_after_staged_outputs)
    with pytest.raises(PortedRunError, match="induced failure"):
        ported.run_ported(
            PortedInputs(
                params=source / "params.json",
                brep=source / "film_unsplit.brep",
                preflight=preflight,
                outdir=outdir,
                n_theta=8,
                n_axial=3,
                gap_levels=(1,),
                preview_ngap=1,
                rim_segments=32,
                tube_layers=1,
                openfoam="skip",
            )
        )
    assert observed == {"staged_outputs_existed": True}
    assert sorted(path.name for path in outdir.iterdir()) == [
        "failure_report.json",
        "run.json",
    ]
    failure = json.loads((outdir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure["overall"] == "FAIL"
    assert failure["solve_eligible_outputs_published"] is False


def test_gui_failure_after_publication_is_warning_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outdir = tmp_path / "published"
    mesh_path = outdir / "nGap_01" / "mesh_arrays.npz"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_bytes(b"validated")

    def fail_gui(_inputs: PortedInputs) -> None:
        raise RuntimeError("no display")

    inputs = PortedInputs(outdir=outdir, gap_levels=(1,), preview_ngap=1, gui=True)
    monkeypatch.setattr(ported, "open_gui", fail_gui)
    open_optional_gui(inputs)

    assert mesh_path.read_bytes() == b"validated"
    assert not (outdir / "failure_report.json").exists()
    assert "optional Gmsh GUI failed after validated mesh publication" in capsys.readouterr().err


def test_openfoam_auto_skips_when_utilities_are_absent(
    default_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ported.shutil, "which", lambda _name: None)
    records: list[dict[str, Any]] = []
    result = ported.audit_openfoam(
        "auto",
        tmp_path,
        tmp_path / "ported_prism_openfoam.msh",
        default_bundle["mesh"],
        records,
        tmp_path / "published",
    )
    assert result["status"] == "SKIPPED"
    assert result["reason"] == "gmshToFoam/checkMesh unavailable"
    assert records == [
        {
            "name": "openfoam.audit",
            "status": "SKIPPED",
            "actual": "gmshToFoam/checkMesh unavailable",
            "expected": "optional audited Prism6 conversion",
            "tolerance": None,
            "mandatory": False,
        }
    ]
