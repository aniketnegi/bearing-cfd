from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from bearing_cfd.artifacts import (
    make_staging_directory,
    publish_generation,
    record_generation,
)


def _publish(staging: Path, target: Path, value: str) -> dict:
    (staging / "result.txt").write_text(value, encoding="utf-8")
    return publish_generation(
        staging,
        target,
        stage="test",
        operation="atomic-publication",
        status="PASS",
        resolved_inputs={"length": 1.0},
        input_units={"length": "mm"},
        producer_files=(Path(__file__),),
        argv=("bearing-cfd", "test"),
        repository=Path(__file__).resolve().parents[1],
    )


def test_publication_records_hashes_and_supersedes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "published"
    first = _publish(make_staging_directory(target), target, "first")
    manifest = json.loads((target / "run.json").read_text(encoding="utf-8"))
    assert manifest == first
    assert manifest["argv"] == ["bearing-cfd", "test"]
    assert manifest["resolved_inputs"] == {
        "units": {"length": "mm"},
        "values": {"length": 1.0},
    }
    assert manifest["outputs"] == [
        {
            "path": "result.txt",
            "role": "output",
            "sha256": "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e",
        }
    ]

    second = _publish(make_staging_directory(target), target, "second")
    assert second["superseded_artifact_id"] == first["artifact_id"]
    assert (target / "result.txt").read_text(encoding="utf-8") == "second"

    staging = make_staging_directory(target)
    (staging / "result.txt").write_text("third", encoding="utf-8")
    real_replace = os.replace

    def fail_publication(source: str | Path, destination: str | Path) -> None:
        if Path(source) == staging and Path(destination) == target:
            raise OSError("induced publication failure")
        real_replace(source, destination)

    monkeypatch.setattr("bearing_cfd.artifacts.os.replace", fail_publication)
    with pytest.raises(OSError, match="induced publication failure"):
        publish_generation(
            staging,
            target,
            stage="test",
            operation="atomic-publication",
            status="PASS",
            resolved_inputs={},
            input_units={},
            producer_files=(Path(__file__),),
        )
    assert (target / "result.txt").read_text(encoding="utf-8") == "second"


def test_mutable_generation_records_only_declared_outputs(tmp_path: Path) -> None:
    target = tmp_path / "campaign"
    target.mkdir()
    ledger = target / "ledger.csv"
    fields = target / "100" / "field"
    fields.parent.mkdir()
    ledger.write_text("status\nREADY\n", encoding="utf-8")
    fields.write_text("bulk mutable state", encoding="utf-8")

    first = record_generation(
        target,
        stage="study",
        operation="campaign",
        status="READY",
        resolved_inputs={},
        input_units={},
        producer_files=(Path(__file__),),
        output_files=(ledger,),
    )
    second = record_generation(
        target,
        stage="study",
        operation="campaign",
        status="COMPLETE",
        resolved_inputs={},
        input_units={},
        producer_files=(Path(__file__),),
        output_files=(ledger,),
    )

    assert [output["path"] for output in second["outputs"]] == ["ledger.csv"]
    assert second["superseded_artifact_id"] == first["artifact_id"]


def test_generation_dirty_state_includes_untracked_sources(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", repository], check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("tracked", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    target = tmp_path / "artifact"
    target.mkdir()
    output = target / "result.txt"
    output.write_text("result", encoding="utf-8")

    clean = record_generation(
        target,
        stage="test",
        operation="dirty-state",
        status="PASS",
        resolved_inputs={},
        input_units={},
        producer_files=(Path(__file__),),
        output_files=(output,),
        repository=repository,
    )
    (repository / "untracked.py").write_text("source", encoding="utf-8")
    dirty = record_generation(
        target,
        stage="test",
        operation="dirty-state",
        status="PASS",
        resolved_inputs={},
        input_units={},
        producer_files=(Path(__file__),),
        output_files=(output,),
        repository=repository,
    )

    assert clean["producer"]["dirty"] is False
    assert dirty["producer"]["dirty"] is True
