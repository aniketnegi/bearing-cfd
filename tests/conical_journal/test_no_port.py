from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import meshio
import numpy as np
import pytest

import bearing_cfd.bearings.conical_journal.meshing.no_port as structured_hex
from bearing_cfd.bearings.conical_journal.meshing.no_port import (
    HEX_FACES,
    PHYSICAL_IDS,
    StructuredInputs,
    StructuredRunError,
    generate_mesh,
    load_params,
    run_structured,
    validate_analytic_mesh,
)


pytestmark = pytest.mark.integration


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "conical_journal"
DEFAULT_PARAMS = FIXTURES / "geometry" / "strict_default" / "params.json"


def test_analytic_geometry_orientation_counts_and_periodic_seam() -> None:
    params = load_params(DEFAULT_PARAMS)
    mesh = generate_mesh(params, n_theta=16, n_axial=4, n_gap=2)
    records: list[dict] = []
    diagnostics = validate_analytic_mesh(mesh, params, records)

    assert all(record["status"] == "PASS" for record in records)
    assert mesh.cell_centres_m.shape == (len(mesh.hexes), 3)
    assert all(values.shape == (len(mesh.hexes),) for values in mesh.cell_metrics.values())
    assert all(values.dtype == np.float64 for values in mesh.cell_metrics.values())
    assert diagnostics["radial_gap_mm"]["minimum"] == pytest.approx(0.020)
    assert diagnostics["radial_gap_mm"]["maximum"] == pytest.approx(0.080)
    assert diagnostics["counts"] == {
        "points": 16 * 5 * 3,
        "hexes": 16 * 4 * 2,
        "boundary_quads": {
            "journal_wall": 16 * 4,
            "stationary_wall": 16 * 4,
            "axial_end_z0": 16 * 2,
            "axial_end_zL": 16 * 2,
        },
    }
    grid = mesh.points_m.reshape(5, 16, 3, 3)
    middle = grid[2]
    radial_gap_mm = (
        np.linalg.norm(middle[:, -1, :2], axis=1)
        - np.linalg.norm(middle[:, 0, :2], axis=1)
    ) / 1.0e-3
    assert radial_gap_mm[0] == pytest.approx(params.h_min_mm, abs=1.0e-12)
    assert radial_gap_mm[8] == pytest.approx(params.h_max_mm, abs=1.0e-12)
    for index, theta in zip((0, 4, 8, 12), (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)):
        point = middle[index, 0]
        assert point[0] == pytest.approx(np.linalg.norm(point[:2]) * math.sin(theta), abs=1.0e-14)
        assert point[1] == pytest.approx(-np.linalg.norm(point[:2]) * math.cos(theta), abs=1.0e-14)
    assert np.all(mesh.cell_metrics["signed_volume_m3"] > 0.0)
    assert np.all(mesh.cell_metrics["min_face_pyramid_m3"] > 0.0)
    last_cells = np.arange(4 * 2) + 15 * 4 * 2
    first_cells = np.arange(4 * 2)
    assert np.array_equal(
        np.sort(mesh.hexes[last_cells][:, HEX_FACES[3]], axis=1),
        np.sort(mesh.hexes[first_cells][:, HEX_FACES[2]], axis=1),
    )


def test_gap_subdivision_invariance_and_theta_convergence() -> None:
    params = load_params(DEFAULT_PARAMS)
    volumes = [
        generate_mesh(params, 16, 4, gap).cell_metrics["signed_volume_m3"].sum()
        for gap in (1, 2, 3)
    ]
    assert max(volumes) - min(volumes) <= abs(np.mean(volumes)) * 1.0e-12
    theta_errors = [
        abs(
            generate_mesh(params, theta, 4, 1).cell_metrics["signed_volume_m3"].sum()
            - params.exact_volume_m3
        )
        / params.exact_volume_m3
        for theta in (8, 16, 32)
    ]
    assert theta_errors[0] > theta_errors[1] > theta_errors[2]


@pytest.mark.parametrize("case", ["strict_default", "strict_case_e03_g20"])
def test_all_true_scale_round_trips_and_visualization_markers(case: str) -> None:
    params_path = FIXTURES / "geometry" / case / "params.json"
    with tempfile.TemporaryDirectory(prefix=f".structured-{case}-") as directory:
        outdir = Path(directory) / "result"
        report = run_structured(
            StructuredInputs(
                params=params_path,
                outdir=outdir,
                n_theta=16,
                n_axial=4,
                gap_levels=(2,),
                preview_ngap=2,
                openfoam="skip",
            )
        )
        assert report["overall"] == "PASS"
        case_dir = outdir / "nGap_02"
        mesh_report = json.loads((case_dir / "mesh_report.json").read_text())
        assert mesh_report["overall"] == "PASS"
        assert all(
            record["status"] in {"PASS", "SKIPPED"}
            for record in mesh_report["validation_records"]
        )
        assert mesh_report["gmsh"]["round_trips"]["gmsh_4_1_binary"]["element_type_counts"]["5"] == 128
        assert mesh_report["gmsh"]["round_trips"]["gmsh_2_2_ascii"]["element_type_counts"]["5"] == 128
        physical = mesh_report["gmsh"]["physical_groups"]
        for name, physical_id in PHYSICAL_IDS.items():
            dimension = 3 if name == "fluid" else 2
            assert physical[f"{dimension}:{physical_id}"]["name"] == name

        volume = meshio.read(case_dir / "volume_hex.vtu")
        boundary = meshio.read(case_dir / "boundary_quads.vtu")
        assert volume.cells[0].type == "hexahedron"
        assert len(volume.cells[0].data) == 128
        assert boundary.cells[0].type == "quad"
        assert set(boundary.cell_data_dict["patch_id"]["quad"]) == {101, 102, 103, 104}

        viz_manifest = json.loads((case_dir / "viz" / "manifest.json").read_text())
        assert viz_manifest["exaggerated"]["solve_eligible"] is False
        assert viz_manifest["exaggerated"]["distorted_geometry"] is True
        assert viz_manifest["exaggerated"]["gap_scale"] == 100
        viz = meshio.read(
            case_dir / "viz" / "gap_x100_VISUALIZATION_ONLY_DO_NOT_SOLVE.vtu"
        )
        assert np.all(viz.cell_data_dict["solve_eligible"]["hexahedron"] == 0)
        assert np.all(viz.cell_data_dict["distorted_geometry"]["hexahedron"] == 1)
        assert not (case_dir / "pressure_feed").exists()


def test_failure_atomically_removes_staged_solver_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(prefix=".structured-fail-") as directory:
        outdir = Path(directory) / "result"
        stale = outdir / "nGap_02"
        stale.mkdir(parents=True)
        (stale / "structured_hex_openfoam.msh").write_text("stale", encoding="utf-8")
        observed: dict[str, bool] = {}

        def fail_after_msh_publication(mesh, case_dir):
            observed["msh41"] = (case_dir / "structured_hex.msh").is_file()
            observed["msh22"] = (case_dir / "structured_hex_openfoam.msh").is_file()
            raise RuntimeError("induced failure after staged MSH publication")

        monkeypatch.setattr(structured_hex, "write_vtu_files", fail_after_msh_publication)
        with pytest.raises(StructuredRunError):
            run_structured(
                StructuredInputs(
                    params=DEFAULT_PARAMS,
                    outdir=outdir,
                    n_theta=16,
                    n_axial=4,
                    gap_levels=(2,),
                    preview_ngap=2,
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


def test_openfoam_boundary_patch_types_are_parsed(tmp_path: Path) -> None:
    boundary = tmp_path / "boundary"
    boundary.write_text(
        """FoamFile
{
    class polyBoundaryMesh;
}
4
(
journal_wall
{
    type wall;
    nFaces 4;
}
stationary_wall
{
    type patch;
    nFaces 4;
}
axial_end_z0
{
    type patch;
    nFaces 2;
}
axial_end_zL
{
    type patch;
    nFaces 2;
}
)
""",
        encoding="utf-8",
    )
    assert structured_hex._openfoam_boundary_patches(boundary) == {
        "journal_wall": {"type": "wall"},
        "stationary_wall": {"type": "patch"},
        "axial_end_z0": {"type": "patch"},
        "axial_end_zL": {"type": "patch"},
    }
