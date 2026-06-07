from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class SignalArtifact:
    output_dir: Path
    signal_path: Path


def timestamp_token(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def write_signal_artifact(
    *,
    export: pl.DataFrame,
    output_dir: Path,
) -> SignalArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signal.parquet"
    export.write_parquet(signal_path)
    return SignalArtifact(output_dir=output_dir, signal_path=signal_path)
