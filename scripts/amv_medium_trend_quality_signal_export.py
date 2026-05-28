"""Export P3 medium-trend quality rerank signals for bt-amv-topn.

The first executable second-stage candidate is a soft penalty for P3 candidates
whose 128-day structure and 128-day trend quality are both weak. This keeps the
same AMV bull candidate pool and changes only Top3 ordering.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT, _finite_expr, _git_commit, _rel_path
from scripts.amv_medium_trend_quality_diagnostic import add_medium_trend_features
from scripts.amv_sector_tailwind_signal_export import build_market_frame
from scripts.amv_static_sleeve_signal_export import pullback_combo_score_expr


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_static_sleeve_signals"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def value_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def parse_penalties(value: str) -> list[float]:
    penalties = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not penalties:
        raise argparse.ArgumentTypeError("penalties must not be empty")
    if any(penalty < 0.0 for penalty in penalties):
        raise argparse.ArgumentTypeError("penalties must be non-negative")
    return penalties


def build_signal_for_penalty(
    market: pl.DataFrame,
    trend_features: pl.DataFrame,
    *,
    penalty: float,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    sleeve_id = (
        "p3_medium128_quality"
        f"_{args.penalty_mode}_t{value_token(args.weak_threshold)}_p{value_token(penalty)}"
    )
    base_score_expr, required_cols = pullback_combo_score_expr(
        price_weight=3.0,
        kbar_weight=0.5,
        bias_weight=0.0,
        close_pullback_weight=0.0,
        risk_weight=0.0,
    )
    valid_expr = _finite_expr(required_cols[0])
    for col_name in required_cols[1:]:
        valid_expr = valid_expr & _finite_expr(col_name)

    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= args.mv_min)
        & (pl.col("amount_ma20") >= args.amount_ma20_min)
        & valid_expr
    )

    structure = pl.col("structure_score_128d").fill_null(1.0)
    quality = pl.col("trend_quality_score_128d").fill_null(1.0)
    weak_expr = (structure < args.weak_threshold) & (quality < args.weak_threshold)
    structure_shortfall = (args.weak_threshold - structure) / args.weak_threshold
    quality_shortfall = (args.weak_threshold - quality) / args.weak_threshold
    weak_strength = (
        pl.when(weak_expr)
        .then(((structure_shortfall + quality_shortfall) / 2.0).clip(lower_bound=0.0, upper_bound=1.0))
        .otherwise(0.0)
    )
    if args.penalty_mode == "bucket":
        penalty_expr = pl.when(weak_expr).then(penalty).otherwise(0.0)
    elif args.penalty_mode == "linear":
        penalty_expr = weak_strength * penalty
    else:
        raise ValueError(f"unknown penalty mode: {args.penalty_mode}")

    scored = (
        market.join(trend_features, on=["date", "code"], how="left")
        .with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            [
                base_score_expr.alias("_base_signal_score"),
                weak_expr.alias("_medium_weak"),
                weak_strength.alias("_medium_weak_strength"),
                penalty_expr.alias("_medium_penalty"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(pl.col("_base_signal_score") - pl.col("_medium_penalty"))
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

    raw_top3 = (
        scored.filter(pl.col("_is_signal_candidate"))
        .with_columns(
            pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_rank")
        )
        .filter(pl.col("_base_rank") <= args.top_n)
        .select(["date", "code"])
    )

    signal_rows = (
        scored.filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= args.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_base_signal_score").alias("base_score"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
                pl.col("structure_score_64d"),
                pl.col("trend_quality_score_64d"),
                pl.col("structure_score_128d"),
                pl.col("trend_quality_score_128d"),
                pl.col("ret_128d"),
                pl.col("pos_128d"),
                pl.col("trend_eff_128d"),
                pl.col("ret_vol_128d"),
                pl.col("_medium_weak").alias("medium_weak"),
                pl.col("_medium_weak_strength").alias("medium_weak_strength"),
                pl.col("_medium_penalty").alias("medium_penalty"),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )

    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date")).drop_nulls(
        "execution_date"
    )
    execution_signals = (
        signal_rows.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(["execution_date", "code", "signal_date", "sleeve_id", "score", "rank"])
        .rename({"execution_date": "date"})
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

    selected_key = signal_rows.select([pl.col("signal_date").alias("date"), "code"])
    overlap = selected_key.join(raw_top3, on=["date", "code"], how="inner").height
    penalized_count = int((signal_rows["medium_penalty"] > 0.0).sum()) if signal_rows.height else 0
    summary = {
        "sleeve_id": sleeve_id,
        "base_sleeve_id": "candidate_p3_k0p5_b0_c0_r0",
        "penalty_mode": args.penalty_mode,
        "weak_threshold": args.weak_threshold,
        "penalty": penalty,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "raw_top3_overlap_rows": overlap,
        "raw_top3_overlap_ratio": overlap / signal_rows.height if signal_rows.height else None,
        "selected_medium_weak_rows": int(signal_rows["medium_weak"].sum()) if signal_rows.height else 0,
        "selected_medium_weak_ratio": float(signal_rows["medium_weak"].mean()) if signal_rows.height else None,
        "selected_penalized_rows": penalized_count,
        "selected_penalized_ratio": penalized_count / signal_rows.height if signal_rows.height else None,
        "selected_penalty_mean": float(signal_rows["medium_penalty"].mean()) if signal_rows.height else None,
        "selected_penalty_max": float(signal_rows["medium_penalty"].max()) if signal_rows.height else None,
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
    }
    return export, signal_rows, summary


def write_signal_artifact(
    *,
    output_root: Path,
    export: pl.DataFrame,
    selected: pl.DataFrame,
    summary: dict[str, Any],
    args: argparse.Namespace,
    started_at: datetime,
) -> Path:
    output_dir = output_root / f"{timestamp_token()}_{summary['sleeve_id']}"
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
        "label": f"static_sleeve:{summary['sleeve_id']}",
        "model_name": "static_factor_sleeve_medium_trend_quality",
        "feature_mode": summary["sleeve_id"],
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
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
            "penalty_mode": args.penalty_mode,
            "weak_threshold": args.weak_threshold,
            "penalty": summary["penalty"],
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
    parser = argparse.ArgumentParser(description="Export P3 medium-trend quality rerank signals")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument("--penalty-mode", choices=["bucket", "linear"], default="linear")
    parser.add_argument("--weak-threshold", type=float, default=0.50)
    parser.add_argument("--penalties", type=parse_penalties, default=[0.01, 0.02, 0.03])
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building P3 market feature frame ...")
    market = build_market_frame(args)
    print("Building medium trend quality features ...")
    trend_features = add_medium_trend_features(market)

    output_paths: list[str] = []
    for penalty in args.penalties:
        print(f"Exporting medium trend penalty={penalty:.4f}")
        export, selected, summary = build_signal_for_penalty(
            market,
            trend_features,
            penalty=penalty,
            args=args,
        )
        meta_path = write_signal_artifact(
            output_root=args.output_root,
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
