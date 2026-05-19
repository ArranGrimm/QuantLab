from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import (
    DEFAULT_QMT_DB,
    ROOT,
    _finite_expr,
    _git_commit,
    _rel_path,
    build_feature_frame,
)
from scripts.amv_static_sleeve_signal_export import parse_sleeves, sleeve_score_expr


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_close_to_close_diagnostic_signals"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_one_diagnostic_signal(
    market: pl.DataFrame,
    *,
    sleeve_id: str,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    score_expr, required_cols = sleeve_score_expr(sleeve_id)
    valid_expr = _finite_expr(required_cols[0])
    for col_name in required_cols[1:]:
        valid_expr = valid_expr & _finite_expr(col_name)

    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= args.mv_min)
        & (pl.col("amount_ma20") >= args.amount_ma20_min)
        & valid_expr
    )

    scored = (
        market.with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(score_expr)
            .otherwise(None)
            .alias("_signal_score")
        )
        .with_columns(
            pl.col("_signal_score")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("_signal_rank")
        )
    )

    signal_rows = (
        scored.filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= args.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )

    export = (
        market.select(
            [
                "date",
                "code",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "pre_close_adj",
                "is_bull_regime",
                "amv_mechanical_regime",
                "market_cap_100m",
                "amount_ma20",
            ]
        )
        .join(signal_rows, left_on=["date", "code"], right_on=["signal_date", "code"], how="left")
        .with_columns(
            [
                pl.col("sleeve_id").is_not_null().alias("is_signal"),
                pl.col("score").fill_null(0.0),
                pl.col("rank").fill_null(9999).cast(pl.UInt32),
                pl.col("sleeve_id").fill_null(""),
            ]
        )
        .sort(["date", "code"])
    )

    summary = {
        "sleeve_id": sleeve_id,
        "signal_timing": "unshifted_close_to_close",
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows": int(export["is_signal"].sum()),
        "signal_days": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
    }
    return export, signal_rows, summary


def write_one_signal(
    output_root: Path,
    *,
    sleeve_id: str,
    export: pl.DataFrame,
    selected: pl.DataFrame,
    summary: dict[str, Any],
    args: argparse.Namespace,
    started_at: datetime,
) -> Path:
    output_dir = output_root / f"{timestamp_token()}_{sleeve_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signal.parquet"
    selected_path = output_dir / "selected_signals.csv"
    meta_path = output_dir / "signal.meta.json"
    export.write_parquet(signal_path)
    selected.write_csv(selected_path)

    meta = {
        "strategy": "amv_close_to_close_diagnostic_topn",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_close_to_close_diagnostic_topn_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"close_to_close_diagnostic:{sleeve_id}",
        "model_name": "static_factor_sleeve",
        "feature_mode": sleeve_id,
        "signal_timing": "unshifted_close_to_close",
        "feature_count": None,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "sleeve_id": sleeve_id,
            "top_n": args.top_n,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "summary": summary,
        "files": {
            "signal": _rel_path(signal_path, output_dir),
            "selected_signals": _rel_path(selected_path, output_dir),
        },
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AMV close-to-close diagnostic signals")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sleeves", type=parse_sleeves, default=["manual_p2_k0p5_r0"])
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building close-to-close diagnostic signal exports...")
    market = build_feature_frame(args)
    output_paths: list[str] = []
    for sleeve_id in args.sleeves:
        print(f"Exporting diagnostic sleeve: {sleeve_id}")
        export, selected, summary = build_one_diagnostic_signal(
            market,
            sleeve_id=sleeve_id,
            args=args,
        )
        meta_path = write_one_signal(
            args.output_root,
            sleeve_id=sleeve_id,
            export=export,
            selected=selected,
            summary=summary,
            args=args,
            started_at=started_at,
        )
        output_paths.append(str(meta_path))
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print("Saved diagnostic signal metas:")
    for path in output_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
