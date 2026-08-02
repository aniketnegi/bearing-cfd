"""Atomic generation publication and explicit provenance records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_staging_directory(outdir: Path) -> Path:
    outdir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=outdir.parent, prefix=f".{outdir.name}-staging-"))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _git_state(repository: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _previous_artifact_id(target: Path) -> str | None:
    try:
        value = json.loads((target / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifact_id = value.get("artifact_id")
    return artifact_id if isinstance(artifact_id, str) else None


def _write_run_json(
    directory: Path,
    *,
    bearing: str,
    stage: str,
    operation: str,
    case_name: str,
    status: str,
    argv: Sequence[str],
    resolved_inputs: Any,
    input_units: Mapping[str, Any],
    repository: Path,
    producer_files: Sequence[Path],
    upstream_artifacts: Sequence[Path],
    tool_versions: Mapping[str, Any],
    acceptance_status: str | bool,
    superseded_artifact_id: str | None,
    output_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    commit, dirty = _git_state(repository)
    files = (
        sorted(directory.rglob("*"))
        if output_files is None
        else sorted(path.resolve() for path in output_files)
    )
    outputs = [
        {
            "path": str(path.resolve().relative_to(directory.resolve())),
            "role": "output",
            "sha256": sha256_file(path),
        }
        for path in files
        if path.is_file() and path.name != "run.json"
    ]
    payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "bearing": bearing,
        "stage": stage,
        "operation": operation,
        "case_name": case_name,
        "status": status,
        "argv": list(argv),
        "resolved_inputs": {
            "values": _jsonable(resolved_inputs),
            "units": _jsonable(input_units),
        },
        "producer": {
            "commit": commit,
            "dirty": dirty,
            "files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in producer_files
                if path.is_file()
            ],
        },
        "upstream_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in upstream_artifacts
            if path.is_file()
        ],
        "tool_versions": {
            "python": platform.python_version(),
            **_jsonable(tool_versions),
        },
        "outputs": outputs,
        "acceptance_status": acceptance_status,
        "superseded_artifact_id": superseded_artifact_id,
    }
    identity = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["artifact_id"] = hashlib.sha256(identity).hexdigest()
    temporary = directory / f".run-{uuid.uuid4().hex}.json"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, directory / "run.json")
    return payload


def record_generation(
    directory: Path,
    *,
    stage: str,
    operation: str,
    status: str,
    resolved_inputs: Any,
    input_units: Mapping[str, Any],
    producer_files: Sequence[Path],
    output_files: Sequence[Path],
    upstream_artifacts: Sequence[Path] = (),
    tool_versions: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
    bearing: str = "conical_journal",
    case_name: str | None = None,
    acceptance_status: str | bool | None = None,
    repository: Path | None = None,
) -> dict[str, Any]:
    """Atomically refresh provenance for an intentionally mutable generation."""
    if not producer_files:
        raise ValueError("producer_files must identify at least one source file")
    directory = directory.resolve()
    return _write_run_json(
        directory,
        bearing=bearing,
        stage=stage,
        operation=operation,
        case_name=case_name or directory.name,
        status=status,
        argv=argv,
        resolved_inputs=resolved_inputs,
        input_units=input_units,
        repository=repository or producer_files[0].resolve().parent,
        producer_files=producer_files,
        upstream_artifacts=upstream_artifacts,
        tool_versions={} if tool_versions is None else tool_versions,
        acceptance_status=status if acceptance_status is None else acceptance_status,
        superseded_artifact_id=_previous_artifact_id(directory),
        output_files=output_files,
    )


def publish_generation(
    staging_directory: Path,
    target_directory: Path,
    *,
    stage: str,
    operation: str,
    status: str,
    resolved_inputs: Any,
    input_units: Mapping[str, Any],
    producer_files: Sequence[Path],
    upstream_artifacts: Sequence[Path] = (),
    tool_versions: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
    bearing: str = "conical_journal",
    case_name: str | None = None,
    acceptance_status: str | bool | None = None,
    repository: Path | None = None,
    superseded_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Publish a complete generation or restore the previous one on failure."""
    if not producer_files:
        raise ValueError("producer_files must identify at least one source file")
    staging_directory = staging_directory.resolve()
    target_directory = target_directory.resolve()
    previous_id = superseded_artifact_id or _previous_artifact_id(target_directory)
    manifest = _write_run_json(
        staging_directory,
        bearing=bearing,
        stage=stage,
        operation=operation,
        case_name=case_name or target_directory.name,
        status=status,
        argv=argv,
        resolved_inputs=resolved_inputs,
        input_units=input_units,
        repository=repository or producer_files[0].resolve().parent,
        producer_files=producer_files,
        upstream_artifacts=upstream_artifacts,
        tool_versions={} if tool_versions is None else tool_versions,
        acceptance_status=status if acceptance_status is None else acceptance_status,
        superseded_artifact_id=previous_id,
    )
    backup = target_directory.with_name(
        f".{target_directory.name}-backup-{uuid.uuid4().hex}"
    )
    had_previous = target_directory.exists()
    if had_previous:
        os.replace(target_directory, backup)
    try:
        os.replace(staging_directory, target_directory)
    except Exception:
        if had_previous and backup.exists():
            os.replace(backup, target_directory)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return manifest
