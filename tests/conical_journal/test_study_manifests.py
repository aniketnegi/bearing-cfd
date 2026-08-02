from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPO_ROOT / "studies/conical_journal"
EXPECTED_STUDIES = {
    "body_fitted_mesh",
    "cavitation_four_track",
    "hydrodynamic_ramp",
    "jfo_checkpoint_evidence",
    "jfo_feed_geometry",
    "single_phase_oq90",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_artifact(entry: dict[str, object]) -> None:
    relative = Path(str(entry["path"]))
    assert not relative.is_absolute()
    digest = str(entry["sha256"])
    assert SHA256.fullmatch(digest)
    path = REPO_ROOT / relative
    if entry.get("tracked"):
        assert path.exists(), path
    if not path.exists():
        return
    kind = str(entry.get("hash_kind", "sha256-file"))
    actual = _tree_sha256(path) if kind == "sha256-tree-v1" else _sha256(path)
    assert actual == digest, path


def test_curated_study_manifests_and_referenced_hashes() -> None:
    manifests = sorted(STUDY_ROOT.glob("*/study.json"))
    assert {path.parent.name for path in manifests} == EXPECTED_STUDIES
    for path in manifests:
        study = json.loads(path.read_text(encoding="utf-8"))
        assert study["schema_version"] == 1
        assert study["question"].strip()
        assert study["status"].strip()
        assert COMMIT.fullmatch(study["source_commit"])
        assert study["commands"]
        assert all(
            isinstance(command, list)
            and command
            and all(isinstance(argument, str) and argument for argument in command)
            for command in study["commands"]
        )
        assert isinstance(study["required_external_tools"], list)
        assert study["expected_outputs"]
        assert all(
            output["path"] and output["role"]
            for output in study["expected_outputs"]
        )
        for artifact in study["source_artifacts"]:
            _validate_artifact(artifact)
        large = study.get("large_artifacts", [])
        if large:
            assert study.get("large_artifact_regeneration")
        for artifact in large:
            _validate_artifact(artifact)
        origin = study.get("origin")
        if origin:
            assert SHA256.fullmatch(origin["sha256"])
            origin_path = REPO_ROOT / origin["path"]
            if origin_path.is_file():
                assert _sha256(origin_path) == origin["sha256"]


def test_evidence_manifest_and_collection_hashes() -> None:
    manifest = json.loads(
        (REPO_ROOT / "evidence/conical_journal/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 1
    assert manifest["bearing"] == "conical_journal"
    assert manifest["hash_kind"] == "sha256-tree-v1"
    assert manifest["collections"]
    for collection in manifest["collections"]:
        _validate_artifact(
            {
                "path": collection["path"],
                "sha256": collection["sha256"],
                "hash_kind": manifest["hash_kind"],
                "tracked": True,
            }
        )
        assert (REPO_ROOT / collection["source_manifest"]).is_file()
        assert collection["status"]
