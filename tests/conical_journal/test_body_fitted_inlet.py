from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from studies.conical_journal.body_fitted_mesh import study as body_fitted_inlet_study
import bearing_cfd.bearings.conical_journal.meshing.body_fitted_inlet as body_fitted
from bearing_cfd.bearings.conical_journal.meshing import no_port as no_port
from bearing_cfd.bearings.conical_journal.meshing.surface_inlet import load_inlet_spec


pytestmark = pytest.mark.integration


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "conical_journal"


def _geometry(case: str = "strict_default"):
    path = FIXTURES / "geometry" / case / "params.json"
    return no_port.load_params(path), load_inlet_spec(path)


def _rows(values: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(sorted(map(int, row))) for row in values}


@pytest.fixture(scope="module")
def exported_tw16(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict, bytes]:
    root = tmp_path_factory.mktemp("body-fitted-export")
    context = root / "context.step"
    context.write_text(
        "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="ascii",
    )
    outdir = root / "TW16-I"
    report = body_fitted.run_body_fitted_case(
        body_fitted.BodyFittedCaseInputs(
            params=FIXTURES / "geometry" / "strict_default" / "params.json",
            outdir=outdir,
            case_name="TW16-I",
            topology="tensor-warp",
            geometry_mode="inscribed",
            q=4,
            n_gap=1,
            minimum_fluent_orthogonal_quality=0.1,
            openfoam="auto",
            ansys="required",
            context_step=context,
        )
    )
    return outdir, report, context.read_bytes()


@pytest.mark.parametrize("case", ["strict_default", "strict_case_e03_g20"])
@pytest.mark.parametrize("mode", ["inscribed", "equal-area"])
def test_analytic_rim_lies_on_drilling_cylinder_and_conical_bore(
    case: str, mode: str
) -> None:
    params, inlet = _geometry(case)
    segments = 32
    effective_radius = inlet.radius_mm
    if mode == "equal-area":
        effective_radius *= math.sqrt(
            2.0 * math.pi
            / (segments * math.sin(2.0 * math.pi / segments))
        )

    points = body_fitted.analytic_rim_nodes(
        params, inlet, segments, mode
    )
    theta = np.mod(np.arctan2(points[:, 0], -points[:, 1]), 2.0 * math.pi)

    assert points.shape == (segments, 3)
    np.testing.assert_allclose(
        np.hypot(points[:, 0], points[:, 2] - inlet.axial_position_mm),
        effective_radius,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.hypot(points[:, 0], points[:, 1]),
        params.bore_radius_mm(points[:, 2]),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        points[:, 0],
        params.bore_radius_mm(points[:, 2]) * np.sin(theta),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        points[:, 1],
        -params.bore_radius_mm(points[:, 2]) * np.cos(theta),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_inscribed_and_equal_area_diagnostics_are_honest() -> None:
    _params, inlet = _geometry()
    segments = 32
    inscribed = body_fitted.rim_geometry_diagnostics(
        inlet, segments, "inscribed"
    )
    equal_area = body_fitted.rim_geometry_diagnostics(
        inlet, segments, "equal-area"
    )
    expected_inscribed_area = (
        0.5
        * segments
        * inlet.radius_mm**2
        * math.sin(2.0 * math.pi / segments)
    )

    assert inscribed["effective_radius_mm"] == inlet.radius_mm
    assert inscribed["polygon_area_mm2"] == pytest.approx(
        expected_inscribed_area
    )
    assert inscribed["chord_sagitta_mm"] == pytest.approx(
        inlet.radius_mm * (1.0 - math.cos(math.pi / segments))
    )
    assert inscribed["nominal_geometry"] is True
    assert inscribed["research_variant"] is False
    assert equal_area["effective_radius_mm"] > inlet.radius_mm
    assert equal_area["radial_bias_mm"] > 0.0
    assert equal_area["polygon_area_mm2"] == pytest.approx(
        math.pi * inlet.radius_mm**2
    )
    assert equal_area["polygon_area_relative_error"] == pytest.approx(0.0)
    assert equal_area["nominal_geometry"] is False
    assert equal_area["research_variant"] is True


@pytest.mark.parametrize(
    ("topology", "resolution", "inner_layers", "outer_layers", "quads", "pressure"),
    [
        ("tensor-warp", 4, 0, 0, 24_576, 16),
        ("tensor-warp", 10, 0, 0, 24_576, 100),
        ("tensor-warp", 20, 0, 0, 24_576, 400),
        ("ogrid", 32, 2, 4, 24_768, 128),
        ("ogrid", 64, 2, 3, 24_896, 384),
    ],
)
def test_production_master_counts_interfaces_and_far_field(
    topology: str,
    resolution: int,
    inner_layers: int,
    outer_layers: int,
    quads: int,
    pressure: int,
) -> None:
    params, inlet = _geometry()
    if topology == "tensor-warp":
        master = body_fitted.build_tensor_warp_master(
            params, inlet, resolution
        )
    else:
        master = body_fitted.build_ogrid_master(
            params,
            inlet,
            resolution,
            inner_layers,
            outer_layers,
        )
    master = body_fitted.smooth_master_mesh(master, params)

    records: list[dict] = []
    report = body_fitted.validate_master_mesh(
        master, params, inlet, records
    )
    census = body_fitted.quad_edge_census(master.quads)
    edge_index = {
        tuple(map(int, edge)): index
        for index, edge in enumerate(census["edges"])
    }
    rim_indices = np.asarray(
        [
            edge_index[tuple(sorted((int(first), int(second))))]
            for first, second in zip(
                master.rim_node_tags,
                np.roll(master.rim_node_tags, -1),
            )
        ]
    )

    assert all(record["status"] == "PASS" for record in records)
    assert report["connected_regions"] == 1
    assert len(master.quads) == quads
    assert int(master.pressure_mask.sum()) == pressure
    assert len(master.rim_node_tags) == (
        4 * resolution if topology == "tensor-warp" else resolution
    )
    assert np.all(census["counts"][rim_indices] == 2)
    for edge in rim_indices:
        owners = (census["owner"][edge], census["neighbour"][edge])
        assert sorted(master.pressure_mask[list(owners)].tolist()) == [
            False,
            True,
        ]

    if topology == "ogrid":
        control_indices = [
            edge_index[tuple(sorted((int(first), int(second))))]
            for first, second in zip(
                master.control_loop_node_tags,
                np.roll(master.control_loop_node_tags, -1),
            )
        ]
        assert len(control_indices) == resolution
        assert np.all(census["counts"][control_indices] == 2)
        corner_tags = np.asarray(
            master.metadata["central_square_corner_node_tags"],
            dtype=np.uint64,
        )
        corners = master.points_mm[
            body_fitted._indices_for_tags(master.node_tags, corner_tags)
        ]
        np.testing.assert_allclose(
            np.hypot(
                corners[:, 0],
                corners[:, 2] - inlet.axial_position_mm,
            ),
            0.5 * master.metadata["effective_radius_mm"],
            rtol=0.0,
            atol=1.0e-12,
        )
    else:
        assert len(master.control_loop_node_tags) == 0

    unchanged = master.points_mm[
        body_fitted._indices_for_tags(
            master.node_tags, master.unchanged_node_tags
        )
    ]
    assert np.array_equal(unchanged, master.unchanged_points_mm)
    assert not master.points_mm.flags.writeable
    assert not master.quads.flags.writeable


def test_quality_optimized_ogrid_uses_circular_control_and_wide_support() -> None:
    params, inlet = _geometry()
    master = body_fitted.build_ogrid_master(
        params,
        inlet,
        32,
        1,
        1,
        n_theta=512,
        n_axial=96,
        quality_optimized=True,
    )
    records: list[dict] = []
    body_fitted.validate_master_mesh(master, params, inlet, records)
    control = master.points_mm[
        body_fitted._indices_for_tags(
            master.node_tags, master.control_loop_node_tags
        )
    ]

    assert all(record["status"] == "PASS" for record in records)
    assert len(master.quads) == 512 * 96 + 64
    assert int(master.pressure_mask.sum()) == 96
    assert master.metadata["quality_optimized"] is True
    assert master.metadata["support_size_cells"] == [16, 16]
    assert master.metadata["central_square_corner_radius_mm"] == pytest.approx(
        0.9 * master.metadata["effective_radius_mm"]
    )
    np.testing.assert_allclose(
        np.hypot(control[:, 0], control[:, 2] - inlet.axial_position_mm),
        1.4 * master.metadata["effective_radius_mm"],
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("mode", "nominal"), [("inscribed", 1), ("equal-area", 0)]
)
def test_generic_sweep_fields_boundaries_and_general_eccentricity(
    mode: str, nominal: int
) -> None:
    original, inlet = _geometry()
    params = replace(original, ex_mm=0.018, ey_mm=-0.024)
    master = body_fitted.build_tensor_warp_master(
        params,
        inlet,
        4,
        geometry_mode=mode,
        n_theta=32,
        n_axial=16,
    )
    mesh = body_fitted.sweep_master_mesh(master, params, n_gap=2)
    grid_mm = mesh.points_m.reshape(len(master.points_mm), 3, 3) / 1.0e-3
    journal, bore = grid_mm[:, 0], grid_mm[:, -1]

    np.testing.assert_allclose(
        np.hypot(
            journal[:, 0] - params.ex_mm,
            journal[:, 1] - params.ey_mm,
        ),
        params.journal_radius_mm(journal[:, 2]),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.hypot(bore[:, 0], bore[:, 1]),
        params.bore_radius_mm(bore[:, 2]),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(bore, master.points_mm, rtol=0.0, atol=1.0e-11)

    assert set(mesh.cell_fields) == set(body_fitted.CELL_FIELD_NAMES)
    assert np.array_equal(
        mesh.cell_fields["master_quad_index"],
        np.repeat(np.arange(len(master.quads)), 2),
    )
    assert np.array_equal(
        mesh.cell_fields["gap_index"],
        np.tile(np.arange(2), len(master.quads)),
    )
    assert np.array_equal(
        mesh.cell_fields["block_id"], np.repeat(master.block_id, 2)
    )
    assert set(mesh.cell_fields["nominal_geometry"]) == {nominal}
    assert set(mesh.cell_fields["research_variant"]) == {1 - nominal}
    assert np.all(mesh.cell_fields["gap_um"] > 0.0)

    assert set(mesh.boundary_quads) == set(body_fitted.SURFACE_ENTITIES)
    assert len(mesh.boundary_quads["journal_wall"]) == len(master.quads)
    assert len(mesh.boundary_quads["pressure_feed"]) == int(
        master.pressure_mask.sum()
    )
    assert len(mesh.boundary_quads["stationary_wall"]) == int(
        (~master.pressure_mask).sum()
    )
    census = body_fitted.hex_face_census(mesh.hexes)
    external = census["counts"] == 1
    declared = np.concatenate(list(mesh.boundary_quads.values()))
    assert _rows(declared) == _rows(census["faces"][external])
    assert len(_rows(declared)) == len(declared)
    assert np.all((census["counts"] == 1) | (census["counts"] == 2))
    internal = census["counts"] == 2
    assert (
        body_fitted._component_count(
            len(mesh.hexes),
            census["owner"][internal],
            census["neighbour"][internal],
        )
        == 1
    )
    for name in (
        "signed_volume_m3",
        "gauss_volume_m3",
        "gauss_min_det",
        "min_face_pyramid_m3",
    ):
        assert np.all(mesh.cell_metrics[name] > 0.0)


def test_stable_physical_group_ids_and_names() -> None:
    assert body_fitted.PHYSICAL_IDS == {
        "journal_wall": 101,
        "stationary_wall": 102,
        "axial_end_z0": 103,
        "axial_end_zL": 104,
        "pressure_feed": 106,
        "fluid": 201,
    }
    assert set(body_fitted.SURFACE_ENTITIES) == {
        "journal_wall",
        "stationary_wall",
        "axial_end_z0",
        "axial_end_zL",
        "pressure_feed",
    }


def test_ogrid_sweep_supports_symmetric_five_to_one_inflation() -> None:
    params, inlet = _geometry()
    master = body_fitted.smooth_master_mesh(
        body_fitted.build_ogrid_master(
            params,
            inlet,
            16,
            1,
            1,
            n_theta=32,
            n_axial=16,
        ),
        params,
    )
    mesh = body_fitted.sweep_master_mesh(
        master, params, n_gap=4, gap_inflation_ratio=5.0
    )
    coordinates = np.asarray(mesh.metadata["gap_layer_coordinates"])
    widths = np.diff(coordinates)

    assert widths.max() / widths.min() == pytest.approx(5.0)
    np.testing.assert_allclose(widths, widths[::-1], rtol=0.0, atol=1.0e-15)
    assert len(mesh.hexes) == 4 * len(master.quads)
    assert np.all(mesh.cell_metrics["signed_volume_m3"] > 0.0)


def test_export_round_trips_and_visual_context(
    exported_tw16: tuple[Path, dict, bytes],
) -> None:
    outdir, report, context_bytes = exported_tw16
    expected_ids = {
        name: (
            3 if name == "fluid" else 2,
            body_fitted.PHYSICAL_IDS[name],
        )
        for name in body_fitted.PHYSICAL_IDS
    }
    expected_hexes = 256 * 96
    expected_quads = 2 * 256 * 96 + 2 * 256

    assert report["overall"] == "PASS"
    assert report["openfoam"]["status"] in {"PASS", "SKIPPED"}
    assert report["ansys"] == {
        "mode": "required",
        "cgns_status": "WRITTEN",
        "path": "bearing_body_fitted_hex.cgns",
    }
    assert report["fluent"]["overall"] == "STATIC_PASS_FLUENT_NOT_RUN"
    assert report["fluent"]["path"] == "fluent/TW16-I.msh"
    assert report["fluent"]["output"] == "fluent/TW16-I.msh"
    assert report["fluent"]["source_npz"] == "mesh_arrays.npz"
    assert report["fluent"]["fluent_equivalent_orthogonal_quality"][
        "threshold_passed"
    ]
    assert all(
        record["status"] in {"PASS", "SKIPPED"}
        for record in report["validation_records"]
    )
    round_trips = {
        value["format"]: value for value in report["gmsh"]["round_trips"]
    }
    assert set(round_trips) == {"GMSH_4_1", "GMSH_2_2_ASCII", "CGNS"}
    for name in ("GMSH_4_1", "GMSH_2_2_ASCII"):
        groups = round_trips[name]["physical_groups"]
        assert {
            group: (value["dimension"], value["physical_id"])
            for group, value in groups.items()
        } == expected_ids
    cgns_groups = round_trips["CGNS"]["physical_groups"]
    assert {
        name: value["dimension"] for name, value in cgns_groups.items()
    } == {name: dimension for name, (dimension, _tag) in expected_ids.items()}
    assert round_trips["CGNS"]["Hex8"] == expected_hexes
    assert round_trips["CGNS"]["patch_counts"]["pressure_feed"] == 16

    for name in ("structured_hex.msh", "structured_hex_openfoam.msh"):
        exported = meshio.read(outdir / name)
        assert set(exported.cells_dict) == {"quad", "hexahedron"}
        assert len(exported.cells_dict["hexahedron"]) == expected_hexes
        assert len(exported.cells_dict["quad"]) == expected_quads
        assert {
            group: tuple(map(int, exported.field_data[group]))
            for group in expected_ids
        } == {
            group: (physical_id, dimension)
            for group, (dimension, physical_id) in expected_ids.items()
        }

    volume = meshio.read(outdir / "volume_hex.vtu")
    boundary = meshio.read(outdir / "boundary_quads.vtu")
    assert set(volume.cells_dict) == {"hexahedron"}
    assert len(volume.cells_dict["hexahedron"]) == expected_hexes
    assert (
        volume.cell_data_dict["fluent_orthogonal_quality"]["hexahedron"].shape
        == (expected_hexes,)
    )
    assert set(boundary.cells_dict) == {"quad"}
    assert len(boundary.cells_dict["quad"]) == expected_quads
    assert set(boundary.cell_data_dict["patch_id"]["quad"]) == {
        body_fitted.PHYSICAL_IDS[name]
        for name in body_fitted.SURFACE_ENTITIES
    }

    with np.load(outdir / "mesh_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["hexes"].shape == (expected_hexes, 8)
        assert arrays["boundary_pressure_feed"].shape == (16, 4)
        assert arrays["metric_minSICN"].shape == (expected_hexes,)
        assert arrays["metric_minDetJac"].shape == (expected_hexes,)
        metadata = json.loads(str(arrays["metadata_json"]))
        assert metadata["nominal_geometry"] is True
        assert metadata["research_variant"] is False

    physical = json.loads((outdir / "physical_groups.json").read_text())
    assert physical["volume"]["fluid"]["physical_id"] == 201
    assert {
        name: value["physical_id"]
        for name, value in physical["boundaries"].items()
    } == {
        name: body_fitted.PHYSICAL_IDS[name]
        for name in body_fitted.SURFACE_ENTITIES
    }
    assert (outdir / "bearing_body_fitted_hex.cgns").is_file()
    assert (outdir / "fluent" / "TW16-I.msh").is_file()
    independent_audit = json.loads(
        (
            outdir
            / "fluent"
            / "independent_centroid_oq_audit.json"
        ).read_text()
    )
    assert independent_audit["passed"] is True
    assert independent_audit["cells_below_threshold"] == 0
    master_surface = meshio.read(outdir / "viz" / "master_surface.vtu")
    assert (
        master_surface.cell_data_dict[
            "fluent_orthogonal_quality_min_through_gap"
        ]["quad"].shape
        == (256 * 96,)
    )
    for name in (
        "viz/full_3d_fluent_oq_overview.png",
        "viz/full_3d_fluent_oq_overview.pdf",
        "viz/full_3d_fluent_oq_summary.json",
    ):
        assert (outdir / name).is_file()
    copied_context = outdir / "VISUAL_CONTEXT_ONLY_context_assembly.step"
    assert copied_context.read_bytes() == context_bytes
    manifest = json.loads((outdir / "manifest.json").read_text())
    assert manifest["context_step"]["status"] == "COPIED"
    assert "VISUAL CONTEXT ONLY" in manifest["context_step"]["warning"]
    instructions = (outdir / "ANSYS_IMPORT.txt").read_text()
    assert "Preferred import: read fluent/TW16-I.msh" in instructions
    assert "must not be used as solver geometry" in instructions
    assert "not a live Fluent import pass" in instructions


def test_failure_atomically_removes_staged_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outdir = tmp_path / "result"
    outdir.mkdir()
    (outdir / "stale_solver_mesh.msh").write_text("stale", encoding="utf-8")
    observed: dict[str, bool] = {}

    def fail_after_msh(mesh, case_dir):
        observed["msh41"] = (case_dir / "structured_hex.msh").is_file()
        observed["msh22"] = (
            case_dir / "structured_hex_openfoam.msh"
        ).is_file()
        raise RuntimeError("induced failure after staged MSH publication")

    monkeypatch.setattr(body_fitted, "_write_vtu", fail_after_msh)
    with pytest.raises(
        body_fitted.BodyFittedRunError,
        match="induced failure after staged MSH publication",
    ):
        body_fitted.run_body_fitted_case(
            body_fitted.BodyFittedCaseInputs(
                params=FIXTURES / "geometry" / "strict_default" / "params.json",
                outdir=outdir,
                case_name="TW16-I-failure",
                topology="tensor-warp",
                geometry_mode="inscribed",
                q=4,
                n_theta=128,
                n_axial=20,
                n_gap=1,
                openfoam="skip",
                ansys="skip",
            )
        )

    assert observed == {"msh41": True, "msh22": True}
    assert sorted(path.name for path in outdir.iterdir()) == [
        "failure_report.json",
        "run.json",
    ]
    failure = json.loads((outdir / "failure_report.json").read_text())
    assert failure["overall"] == "FAIL"
    assert failure["solve_eligible_outputs_published"] is False


def test_quality_rejection_publishes_visual_only_previews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reject_quality(
        mesh,
        master,
        params,
        inlet,
        records=None,
        *,
        require_gmsh_metrics=False,
    ):
        records = [] if records is None else records
        body_fitted.require(
            records,
            "body.test.forced_quality_rejection",
            False,
            46.0,
            "<=45 deg",
        )

    monkeypatch.setattr(
        body_fitted, "validate_body_fitted_mesh", reject_quality
    )
    outdir = tmp_path / "rejected"
    with pytest.raises(body_fitted.BodyFittedRunError):
        body_fitted.run_body_fitted_case(
            body_fitted.BodyFittedCaseInputs(
                params=FIXTURES / "geometry" / "strict_default" / "params.json",
                outdir=outdir,
                case_name="TW16-I-rejected-preview",
                topology="tensor-warp",
                geometry_mode="inscribed",
                q=4,
                n_theta=32,
                n_axial=16,
                n_gap=1,
                openfoam="skip",
                ansys="skip",
            )
        )

    assert not any(
        (outdir / name).exists()
        for name in (
            "structured_hex.msh",
            "structured_hex_openfoam.msh",
            "bearing_body_fitted_hex.cgns",
            "volume_hex.vtu",
            "mesh_arrays.npz",
        )
    )
    expected = (
        "VISUAL_ONLY_DO_NOT_SOLVE.txt",
        "plots/footprint.png",
        "plots/local_master_mesh.png",
        "plots/master_quality.png",
        "viz/DIAGNOSTIC_ONLY_master_surface.msh",
        "viz/master_surface.vtu",
        "viz/pressure_feed_only.vtu",
        "viz/cutaway_exact.vtu",
    )
    assert all((outdir / name).is_file() for name in expected)
    gmsh_preview = meshio.read(
        outdir / "viz" / "DIAGNOSTIC_ONLY_master_surface.msh"
    )
    assert set(gmsh_preview.cells_dict) == {"quad"}
    failure = json.loads((outdir / "failure_report.json").read_text())
    assert failure["visual_only_preview"]["solve_eligible"] is False
    assert failure["solve_eligible_outputs_published"] is False


def test_study_records_rejected_case_and_writes_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reject(inputs: body_fitted.BodyFittedCaseInputs):
        report = {
            "overall": "FAIL",
            "master_validation": {
                "counts": {"quads": 24_576},
                "quality": {"minimum_scaled_jacobian": 0.15},
            },
            "body_validation": {
                "counts": {
                    "points": 322_816,
                    "Hex8": 294_912,
                    "boundary_Quad4": {"pressure_feed": 100},
                },
                "quality": {
                    "minimum_minSICN": None,
                    "minimum_minDetJac": None,
                    "maximum_nonorthogonality_deg": 61.2,
                    "maximum_skewness": 0.75,
                },
                "volume": {
                    "cell_sum_m3": 9.4e-7,
                    "continuous_relative_error": 1.0e-4,
                },
            },
            "error": {"message": "maximum non-orthogonality exceeds 45 deg"},
        }
        inputs.outdir.mkdir(parents=True)
        (inputs.outdir / "failure_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        raise body_fitted.BodyFittedRunError(
            report["error"]["message"], report
        )

    monkeypatch.setattr(body_fitted_inlet_study.body, "run_body_fitted_case", reject)
    outdir = tmp_path / "study"
    report = body_fitted_inlet_study.run_study(
        body_fitted_inlet_study.StudyInputs(
            params=FIXTURES / "geometry" / "strict_default" / "params.json",
            outdir=outdir,
            cases=("TW40-EA",),
            ansys="skip",
        )
    )

    assert report["overall"] == "FAIL"
    assert report["cases"][0]["solver_eligible"] is False
    assert report["cases"][0]["maximum_nonorthogonality_deg"] == 61.2
    assert (outdir / "comparison.csv").is_file()
    assert json.loads((outdir / "comparison.json").read_text())["overall"] == "FAIL"
    gallery = (outdir / "study_gallery.html").read_text(encoding="utf-8")
    assert "TW40-EA" in gallery
    assert "REJECTED — VISUAL ONLY — DO NOT SOLVE" in gallery
    assert "NON-NOMINAL RESEARCH GEOMETRY" in gallery
    assert "61.2" in gallery
    assert (outdir / "viewer_commands.txt").is_file()


def test_gallery_refresh_removes_stale_render_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    (render_dir / "old.png").write_bytes(b"stale")
    (tmp_path / "render_failures.txt").write_text("stale\n")
    monkeypatch.setattr(
        body_fitted_inlet_study.shutil, "which", lambda _name: None
    )

    body_fitted_inlet_study._render_study_images(
        tmp_path, {"preset": "uniform-study", "cases": []}
    )
    body_fitted_inlet_study._write_study_gallery(
        tmp_path, {"preset": "uniform-study", "cases": []}, {}
    )

    assert not render_dir.exists()
    assert not (tmp_path / "render_failures.txt").exists()
