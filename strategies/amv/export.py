from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import polars as pl


@dataclass(frozen=True)
class SignalArtifactConfig:
    sleeve_id: str
    model_name: str
    output_root: Path
    config: Mapping[str, Any] = field(default_factory=dict)
    strategy: str = "amv_static_sleeve_topn"
    label: str | None = None
    feature_mode: str | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)
    extra_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalArtifact:
    output_dir: Path
    signal_path: Path
    selected_path: Path
    meta_path: Path


def timestamp_token(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def git_commit(root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def build_signal_meta(
    *,
    artifact: SignalArtifact,
    artifact_config: SignalArtifactConfig,
    started_at: datetime,
    finished_at: datetime,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    signal_id = artifact.output_dir.name
    sleeve_id = artifact_config.sleeve_id
    meta: dict[str, Any] = {
        "strategy": artifact_config.strategy,
        "signal_id": signal_id,
        "signal_run_id": f"{artifact_config.strategy}_{signal_id}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": artifact_config.label or f"static_sleeve:{sleeve_id}",
        "model_name": artifact_config.model_name,
        "feature_mode": artifact_config.feature_mode or sleeve_id,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": git_commit(repo_root),
        "config": dict(artifact_config.config),
        "summary": dict(artifact_config.summary),
        "files": {
            "signal": relative_path(artifact.signal_path, artifact.output_dir),
            "selected_signals": relative_path(artifact.selected_path, artifact.output_dir),
        },
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
    }
    meta.update(dict(artifact_config.extra_meta))
    return meta


def write_signal_artifact(
    *,
    export: pl.DataFrame,
    selected: pl.DataFrame,
    artifact_config: SignalArtifactConfig,
    started_at: datetime | None = None,
    repo_root: Path | None = None,
) -> SignalArtifact:
    started = started_at or datetime.now()
    output_dir = artifact_config.output_root / f"{timestamp_token()}_{artifact_config.sleeve_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = SignalArtifact(
        output_dir=output_dir,
        signal_path=output_dir / "signal.parquet",
        selected_path=output_dir / "selected_signals.csv",
        meta_path=output_dir / "signal.meta.json",
    )
    export.write_parquet(artifact.signal_path)
    selected.write_csv(artifact.selected_path)

    finished = datetime.now()
    meta = build_signal_meta(
        artifact=artifact,
        artifact_config=artifact_config,
        started_at=started,
        finished_at=finished,
        repo_root=repo_root,
    )
    artifact.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return artifact
