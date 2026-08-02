from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
import pytest

import bearing_cfd.bearings.conical_journal.meshing.surface_inlet as surface_inlet
from bearing_cfd.bearings.conical_journal.meshing import no_port as no_port
from bearing_cfd.bearings.conical_journal.meshing.surface_inlet import (
    PHYSICAL_IDS,
    SURFACE_ENTITIES,
    SurfaceInletInputs,
    SurfaceInletRunError,
    build_surface_inlet_mesh,
    load_inlet_spec,
    run_surface_inlet,
    validate_surface_inlet_mesh,
)


pytestmark = pytest.mark.integration


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "conical_journal"


def _params(case: str = "strict_default") -> Path:
    return FIXTURES / "geometry" / case / "params.json"


@pytest.mark.parametrize("case", ["strict_default", "strict_case_e03_g20"])
def test_analytic_inflated_hex_mesh_and_surface_patch(case: str) -> None:
    params_path = _params(case)
    params = no_port.load_params(params_path)
    inlet = load_inlet_spec(params_path)
    mesh, original_outer, uniform_volume = build_surface_inlet_mesh(
        params,
        inlet,
        n_theta=256,
        n_axial=96,
        n_gap=4,
        gap_inflation_ratio=5.0,
    )
    records: list[dict] = []
    diagnostics = validate_surface_inlet_mesh(
        mesh,
        original_outer,
        uniform_volume,
        params,
        inlet,
        max_projected_area_relative_error=0.01,
        records=records,
        max_inlet_rim_error_mm=0.16,
    )

    assert all(record["status"] == "PASS" for record in records)
    assert mesh.hexes.shape == (256 * 96 * 4, 8)
    assert set(mesh.boundary_quads) == set(SURFACE_ENTITIES)
    assert all(quads.shape[1] == 4 for quads in mesh.boundary_quads.values())
    assert mesh.metadata["contains_feed_volume"] is False
    assert mesh.metadata["gap_inflation_ratio_achieved"] == pytest.approx(5.0)
    assert mesh.metadata["inlet_cluster_strength"] == pytest.approx(0.82)
    assert mesh.metadata["theta_max_adjacent_width_growth"] < 1.1
    assert mesh.metadata["axial_max_adjacent_height_growth"] < 1.1
    assert np.all(
        np.diff(mesh.metadata["theta_edge_coordinates_rad"]) > 0.0
    )
    assert np.all(np.diff(mesh.metadata["z_node_coordinates_mm"]) > 0.0)

    reconstructed = np.concatenate(
        (
            mesh.boundary_quads["stationary_wall"],
            mesh.boundary_quads["pressure_feed"],
        )
    )
    assert np.array_equal(
        no_port._sorted_rows(np.sort(reconstructed, axis=1)),
        no_port._sorted_rows(np.sort(original_outer, axis=1)),
    )
    patch = diagnostics["pressure_patch"]
    assert patch["quad_count"] >= 300
    assert patch["projected_area_relative_error"] < 0.01
    assert patch["rim_radius_mm"]["maximum_absolute_error"] < 0.16
    assert patch["rim_connected_components"] == 1
    assert patch["rim_vertex_degrees"] == {"2": patch["rim_vertex_count"]}
    assert len(mesh.boundary_quads["journal_wall"]) == 256 * 96
    assert diagnostics["inflation"]["achieved_ratio"] == pytest.approx(5.0)
    assert diagnostics["topology"]["connected_regions"] == 1
    assert np.all(mesh.cell_metrics["signed_volume_m3"] > 0.0)
    assert np.all(mesh.cell_metrics["gauss_min_det"] > 0.0)


def test_odd_theta_count_is_rejected() -> None:
    params_path = _params()
    params = no_port.load_params(params_path)
    inlet = load_inlet_spec(params_path)
    with pytest.raises(surface_inlet.SurfaceInletError, match="even n-theta"):
        surface_inlet.validate_inputs(
            SurfaceInletInputs(n_theta=127), params, inlet
        )


@pytest.mark.parametrize("case", ["strict_default", "strict_case_e03_g20"])
def test_round_trips_keep_hex8_quad4_and_pressure_group(
    case: str, tmp_path: Path
) -> None:
    outdir = tmp_path / case
    report = run_surface_inlet(
        SurfaceInletInputs(
            params=_params(case),
            outdir=outdir,
            n_theta=128,
            n_axial=48,
            gap_levels=(4,),
            preview_ngap=4,
            gap_inflation_ratio=5.0,
            inlet_cluster_strength=0.45,
            max_projected_area_relative_error=0.11,
            max_inlet_rim_error_mm=0.8,
            openfoam="skip",
        )
    )

    assert report["overall"] == "PASS"
    case_dir = outdir / "nGap_04"
    mesh_report = json.loads((case_dir / "mesh_report.json").read_text())
    manifest = json.loads((case_dir / "manifest.json").read_text())
    assert mesh_report["overall"] == "PASS"
    assert all(
        record["status"] in {"PASS", "SKIPPED"}
        for record in mesh_report["validation_records"]
    )
    assert manifest["contains_only_hex8_volume_cells"] is True
    assert manifest["contains_feed_tube"] is False
    assert manifest["contains_pressure_feed_patch"] is True

    expected_groups = {
        f"2:{PHYSICAL_IDS[name]}": name for name in SURFACE_ENTITIES
    } | {f"3:{PHYSICAL_IDS['fluid']}": "fluid"}
    physical = mesh_report["gmsh"]["physical_groups"]
    assert {
        key: physical[key]["name"] for key in expected_groups
    } == expected_groups

    hex_count = 128 * 48 * 4
    quad_count = 2 * 128 * 48 + 2 * 128 * 4
    for round_trip in mesh_report["gmsh"]["round_trips"].values():
        assert round_trip["element_type_counts"] == {
            "3": quad_count,
            "5": hex_count,
        }

    volume = meshio.read(case_dir / "volume_hex.vtu")
    boundary = meshio.read(case_dir / "boundary_quads.vtu")
    inlet_viz = meshio.read(case_dir / "viz" / "pressure_feed_only.vtu")
    assert set(volume.cells_dict) == {"hexahedron"}
    assert len(volume.cells_dict["hexahedron"]) == hex_count
    assert set(boundary.cells_dict) == {"quad"}
    assert len(boundary.cells_dict["quad"]) == quad_count
    assert set(boundary.cell_data_dict["patch_id"]["quad"]) == {
        PHYSICAL_IDS[name] for name in SURFACE_ENTITIES
    }
    assert PHYSICAL_IDS["pressure_feed"] in set(
        boundary.cell_data_dict["patch_id"]["quad"]
    )
    assert set(inlet_viz.cells_dict) == {"quad"}
    assert len(inlet_viz.cells_dict["quad"]) > 0
    assert np.ptp(inlet_viz.points[:, 0]) < 0.005
    assert np.ptp(inlet_viz.points[:, 2]) < 0.005

    with np.load(case_dir / "mesh_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["hexes"].shape == (hex_count, 8)
        assert arrays["boundary_pressure_feed"].shape[0] > 0
        assert arrays["boundary_pressure_feed"].shape[1] == 4
        metadata = json.loads(str(arrays["metadata_json"]))
        assert metadata["contains_feed_volume"] is False
        assert metadata["gap_inflation_ratio_achieved"] == pytest.approx(5.0)
        assert metadata["inlet_cluster_strength"] == pytest.approx(0.45)


def test_failure_publishes_only_failure_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outdir = tmp_path / "result"
    stale = outdir / "nGap_04"
    stale.mkdir(parents=True)
    (stale / "structured_hex_openfoam.msh").write_text(
        "stale", encoding="utf-8"
    )
    observed: dict[str, bool] = {}

    def fail_after_msh_publication(mesh, case_dir):
        observed["msh41"] = (case_dir / "structured_hex.msh").is_file()
        observed["msh22"] = (
            case_dir / "structured_hex_openfoam.msh"
        ).is_file()
        raise RuntimeError("induced failure after staged MSH publication")

    monkeypatch.setattr(
        surface_inlet, "write_vtu_files", fail_after_msh_publication
    )
    with pytest.raises(SurfaceInletRunError):
        run_surface_inlet(
            SurfaceInletInputs(
                params=_params(),
                outdir=outdir,
                n_theta=128,
                n_axial=48,
                gap_levels=(4,),
                preview_ngap=4,
                gap_inflation_ratio=5.0,
                inlet_cluster_strength=0.45,
                max_projected_area_relative_error=0.11,
                max_inlet_rim_error_mm=0.8,
                openfoam="skip",
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


def test_gui_failure_does_not_replace_published_mesh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outdir = tmp_path / "result"

    def fake_gap_case(params, inlet, inputs, n_gap, case_dir):
        case_dir.mkdir(parents=True)
        return {"n_gap": n_gap}

    monkeypatch.setattr(surface_inlet, "generate_gap_case", fake_gap_case)
    monkeypatch.setattr(
        surface_inlet,
        "write_convergence_report",
        lambda stage, params, inputs, cases: {"overall": "PASS"},
    )
    monkeypatch.setattr(
        surface_inlet,
        "open_gui",
        lambda inputs, published: (_ for _ in ()).throw(
            RuntimeError("display unavailable")
        ),
    )
    report = run_surface_inlet(
        SurfaceInletInputs(
            params=_params(),
            outdir=outdir,
            n_theta=128,
            n_axial=48,
            gap_levels=(4,),
            preview_ngap=4,
            gap_inflation_ratio=5.0,
            gui=True,
            openfoam="skip",
        )
    )

    assert report["overall"] == "PASS"
    assert report["gui"]["status"] == "WARNING"
    assert (outdir / "run_report.json").is_file()
    assert json.loads((outdir / "gui_status.json").read_text())["status"] == "WARNING"
    assert not (outdir / "failure_report.json").exists()
