from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = REPO_ROOT / "cases/conical_journal/openfoam"
ARCHIVED_CASES = {
    "jfo_body_fitted",
    "jfo_grid_128x40",
    "jfo_grid_512x160",
    "jfo_grid_512x160_mapped",
    "multiphase_paper_exact_cavitation",
    "multiphase_s100_cavitation_exploratory",
    "single_phase_paper_exact",
    "single_phase_s100_baseline",
}
CURRENT_TEMPLATE_HASHES = {
    "0/U": "d21f6a034dbdc20718edf17c9cd011e1f9ccf5ac83adef4fcfe7bd81493d0db8",
    "0/p": "676ac8856a5939f66e6b8901ff542a519b695478d266d5b74d5ff73e11e66cf4",
    "constant/momentumTransport": "7a054071ee4dd4342437c9d1b38811dd0987bfc838d57949a03f877ed0b69638",
    "constant/physicalProperties": "ffcc4ad1340cde844c9a411418661f87b83975697b362b54aba34caef358739a",
    "system/controlDict": "4c700ab207b347d5565135f8026e0d70b5932ed8e897b3debb0ac406d547b8f7",
    "system/decomposeParDict": "d720401777cdffca4fcb8654feea4468ce20432c6ba3105ea1a2d44ced778798",
    "system/functions": "3a7e35c241756d43fce69b6f7e1846f7b68a30615757bde55b281d7f2bc49b7d",
    "system/fvSchemes": "b59f194bcd6ecc1bf8047693b54db39478445c9edf2e3dc21171c1a9a271da12",
    "system/fvSolution": "b9a3e01fc13cbd5f6b80085be2968aff65551a6de72e1fd574794a6c7ff39a81",
}


def test_case_definitions_are_complete_and_documented() -> None:
    assert {path.name for path in (CASE_ROOT / "archived").iterdir() if path.is_dir()} == ARCHIVED_CASES
    assert (CASE_ROOT / "archived/README.org").is_file()
    assert (CASE_ROOT / "archived/RESULTS.org").is_file()
    for name in ARCHIVED_CASES:
        case = CASE_ROOT / "archived" / name
        assert (case / "constant").is_dir()
        assert (case / "system").is_dir()
        if name != "jfo_grid_512x160_mapped":
            assert (case / "0").is_dir()


def test_current_single_phase_template_matches_retained_case() -> None:
    template = CASE_ROOT / "single_phase_oq90"
    for relative, expected in CURRENT_TEMPLATE_HASHES.items():
        assert hashlib.sha256((template / relative).read_bytes()).hexdigest() == expected


def test_case_tree_contains_no_generated_openfoam_outputs() -> None:
    generated_directories = {"polyMesh", "postProcessing", "VTK", "checkpoints"}
    for path in CASE_ROOT.rglob("*"):
        relative = path.relative_to(CASE_ROOT)
        assert not generated_directories.intersection(relative.parts)
        assert not any(part.startswith("processor") for part in relative.parts)
        assert not path.name.startswith("log.")
        assert not (
            path.is_dir()
            and path.name != "0"
            and path.name.replace(".", "", 1).isdigit()
        )


def test_single_phase_runner_parses_and_exposes_staged_commands() -> None:
    runner = CASE_ROOT / "single_phase_oq90/run.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    result = subprocess.run(
        ["bash", str(runner), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare FLUENT_MESH" in result.stdout
    assert "2000rpm" in result.stdout
    assert "python3 -" not in runner.read_text(encoding="utf-8")
