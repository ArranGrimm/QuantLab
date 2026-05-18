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
    _score_component,
    build_feature_frame,
)


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_static_sleeve_signals"

SLEEVE_IDS = [
    "ret_5d",
    "ret_20d",
    "klen",
    "kmid2",
    "manual_p2_k0p5_r0",
    "manual_p3_k0p5_r0",
    "pkm_p1_k0p5_m1",
    "pkm_p2_k0p5_m0p5",
    "pkm_p3_k1_m2",
]


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_sleeves(value: str) -> list[str]:
    sleeves = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(sleeves) - set(SLEEVE_IDS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown sleeves: {', '.join(unknown)}")
    if not sleeves:
        raise argparse.ArgumentTypeError("sleeves must not be empty")
    return sleeves


def sleeve_score_expr(sleeve_id: str) -> tuple[pl.Expr, list[str]]:
    if sleeve_id == "ret_5d":
        return pl.col("ret_5d"), ["ret_5d"]
    if sleeve_id == "ret_20d":
        return pl.col("ret_20d"), ["ret_20d"]
    if sleeve_id == "klen":
        return -pl.col("KLEN"), ["KLEN"]
    if sleeve_id == "kmid2":
        return pl.col("KMID2"), ["KMID2"]
    if sleeve_id == "manual_p2_k0p5_r0":
        cols = ["price_pos_20d", "close_to_high_20d", "KLEN", "KMID2"]
        return (
            (
                _score_component("price_pos_20d", higher_is_better=True, weight=2.0)
                + _score_component("close_to_high_20d", higher_is_better=False, weight=2.0)
                + _score_component("KLEN", higher_is_better=False, weight=0.5)
                + _score_component("KMID2", higher_is_better=True, weight=0.5)
            )
            / 5.0,
            cols,
        )
    if sleeve_id == "manual_p3_k0p5_r0":
        cols = ["price_pos_20d", "close_to_high_20d", "KLEN", "KMID2"]
        return (
            (
                _score_component("price_pos_20d", higher_is_better=True, weight=3.0)
                + _score_component("close_to_high_20d", higher_is_better=False, weight=3.0)
                + _score_component("KLEN", higher_is_better=False, weight=0.5)
                + _score_component("KMID2", higher_is_better=True, weight=0.5)
            )
            / 7.0,
            cols,
        )
    if sleeve_id == "pkm_p1_k0p5_m1":
        return pkm_score_expr(price_weight=1.0, kbar_weight=0.5, momentum_weight=1.0)
    if sleeve_id == "pkm_p2_k0p5_m0p5":
        return pkm_score_expr(price_weight=2.0, kbar_weight=0.5, momentum_weight=0.5)
    if sleeve_id == "pkm_p3_k1_m2":
        return pkm_score_expr(price_weight=3.0, kbar_weight=1.0, momentum_weight=2.0)
    raise ValueError(f"unknown sleeve_id: {sleeve_id}")


def pkm_score_expr(
    *,
    price_weight: float,
    kbar_weight: float,
    momentum_weight: float,
) -> tuple[pl.Expr, list[str]]:
    cols = ["price_pos_20d", "close_to_high_20d", "KLEN", "KMID2", "ret_5d", "ret_20d"]
    total_weight = 2.0 * (price_weight + kbar_weight + momentum_weight)
    if total_weight <= 0:
        raise ValueError("P/K/M total weight must be positive")
    return (
        (
            _score_component("price_pos_20d", higher_is_better=True, weight=price_weight)
            + _score_component("close_to_high_20d", higher_is_better=False, weight=price_weight)
            + _score_component("KLEN", higher_is_better=False, weight=kbar_weight)
            + _score_component("KMID2", higher_is_better=True, weight=kbar_weight)
            + _score_component("ret_5d", higher_is_better=True, weight=momentum_weight)
            + _score_component("ret_20d", higher_is_better=True, weight=momentum_weight)
        )
        / total_weight,
        cols,
    )


def build_one_sleeve_signal(
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

    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(
        pl.col("date").shift(-1).alias("execution_date")
    ).drop_nulls("execution_date")
    execution_signals = (
        signal_rows.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(
            [
                pl.col("execution_date").alias("date"),
                "code",
                "signal_date",
                "sleeve_id",
                "score",
                "rank",
            ]
        )
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
        .join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            [
                pl.col("signal_date").is_not_null().alias("is_signal"),
                pl.col("score").fill_null(0.0),
                pl.col("rank").fill_null(9999).cast(pl.UInt32),
                pl.col("sleeve_id").fill_null(""),
            ]
        )
        .sort(["date", "code"])
    )

    summary = {
        "sleeve_id": sleeve_id,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "signals_blocked_by_execution_bear_regime": int(
            export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
        ),
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
        "strategy": "amv_static_sleeve_topn",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_static_sleeve_topn_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"static_sleeve:{sleeve_id}",
        "model_name": "static_factor_sleeve",
        "feature_mode": sleeve_id,
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
    parser = argparse.ArgumentParser(description="Export AMV static factor sleeve signals")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sleeves", type=parse_sleeves, default=["ret_5d"])
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
    print("Building static sleeve signal exports...")
    market = build_feature_frame(args)
    output_paths: list[str] = []
    for sleeve_id in args.sleeves:
        print(f"Exporting sleeve: {sleeve_id}")
        export, selected, summary = build_one_sleeve_signal(market, sleeve_id=sleeve_id, args=args)
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

    print("Saved signal metas:")
    for path in output_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
