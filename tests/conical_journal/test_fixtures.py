import json
from pathlib import Path

from bearing_cfd.artifacts import sha256_file


def test_fixture_hashes_match_manifest() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "conical_journal"
    manifest = json.loads((root / "fixture_manifest.json").read_text(encoding="utf-8"))

    assert manifest["bearing"] == "conical_journal"
    assert {
        name: sha256_file(root / name) for name in manifest["files"]
    } == manifest["files"]
