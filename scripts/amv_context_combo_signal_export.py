"""Export P3 combined context rerank signals for bt-amv-topn.

Combines the two currently strongest P3 context challengers:
- sector tailwind linear penalty
- 128-day medium structure / trend-quality linear penalty
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
from scripts.amv_sector_tailwind_diagnostic import DEFAULT_SECTOR_MAP
from scripts.amv_sector_tailwind_signal_export import (
    build_market_frame,
    build_sector_features,
    rank_source_token,
    relative_confirm_expr,
    sector_rank_expr,
    threshold_token,
)
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


def build_signal_for_penalties(
    market: pl.DataFrame,
    sector_features: pl.DataFrame,
    trend_features: pl.DataFrame,
    *,
    sector_penalty: float,
    medium_penalty: float,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    sleeve_id = (
        f"p3_ctx_sector{rank_source_token(args.rank_source)}_{args.sector_penalty_mode}"
        f"_b{threshold_token(args.bottom_rank_threshold)}_sp{value_token(sector_penalty)}"
        f"_medium128_{args.medium_penalty_mode}_t{value_token(args.weak_threshold)}"
        f"_mp{value_token(medium_penalty)}"
    )
    if args.relative_confirm != "none":
        sleeve_id = f"{sleeve_id}_{args.relative_confirm}"

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

    sector_rank = sector_rank_expr(args).fill_null(1.0)
    sector_bottom_distance = (args.bottom_rank_threshold - sector_rank) / args.bottom_rank_threshold
    sector_bottom_strength = pl.when(sector_bottom_distance > 0.0).then(sector_bottom_distance).otherwise(0.0)
    sector_confirm = relative_confirm_expr(args)
    if args.sector_penalty_mode == "linear":
        sector_penalty_expr = pl.when(sector_confirm).then(sector_bottom_strength * sector_penalty).otherwise(0.0)
    elif args.sector_penalty_mode == "bucket":
        sector_penalty_expr = (
            pl.when((sector_rank < args.bottom_rank_threshold) & sector_confirm).then(sector_penalty).otherwise(0.0)
        )
    else:
        raise ValueError(f"unknown sector penalty mode: {args.sector_penalty_mode}")

    structure = pl.col("structure_score_128d").fill_null(1.0)
    quality = pl.col("trend_quality_score_128d").fill_null(1.0)
    medium_weak = (structure < args.weak_threshold) & (quality < args.weak_threshold)
    structure_shortfall = (args.weak_threshold - structure) / args.weak_threshold
    quality_shortfall = (args.weak_threshold - quality) / args.weak_threshold
    medium_strength = (
        pl.when(medium_weak)
        .then(((structure_shortfall + quality_shortfall) / 2.0).clip(lower_bound=0.0, upper_bound=1.0))
        .otherwise(0.0)
    )
    if args.medium_penalty_mode == "linear":
        medium_penalty_expr = medium_strength * medium_penalty
    elif args.medium_penalty_mode == "bucket":
        medium_penalty_expr = pl.when(medium_weak).then(medium_penalty).otherwise(0.0)
    else:
        raise ValueError(f"unknown medium penalty mode: {args.medium_penalty_mode}")

    scored = (
        market.join(sector_features, on=["date", "code"], how="left")
        .join(trend_features, on=["date", "code"], how="left")
        .with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            [
                base_score_expr.alias("_base_signal_score"),
                sector_rank.alias("_sector_rank_score"),
                sector_confirm.alias("_relative_confirm"),
                sector_penalty_expr.alias("_sector_penalty"),
                medium_weak.alias("_medium_weak"),
                medium_strength.alias("_medium_weak_strength"),
                medium_penalty_expr.alias("_medium_penalty"),
            ]
        )
        .with_columns((pl.col("_sector_penalty") + pl.col("_medium_penalty")).alias("_context_penalty"))
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(pl.col("_base_signal_score") - pl.col("_context_penalty"))
            .otherwise(None)
            .alias("_signal_score")
        )
        .with_columns(pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank"))
    )

    raw_top3 = (
        scored.filter(pl.col("_is_signal_candidate"))
        .with_columns(pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_rank"))
        .filter(pl.col("_base_rank") <= args.top_n)
        .select(["date", "code"])
    )

    signal_rows = (
        scored.filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= args.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                "industry",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_base_signal_score").alias("base_score"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
                pl.col("_sector_rank_score").alias("sector_rank_score"),
                pl.col("_relative_confirm").alias("relative_confirm"),
                pl.col("_sector_penalty").alias("sector_penalty"),
                pl.col("stock_rel_sector_ret_20d"),
                pl.col("structure_score_128d"),
                pl.col("trend_quality_score_128d"),
                pl.col("_medium_weak").alias("medium_weak"),
                pl.col("_medium_weak_strength").alias("medium_weak_strength"),
                pl.col("_medium_penalty").alias("medium_penalty"),
                pl.col("_context_penalty").alias("context_penalty"),
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
                "open_raw",
                "high_raw",
                "low_raw",
                "close_raw",
                "pre_close_raw",
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
    sector_penalized = int((signal_rows["sector_penalty"] > 0.0).sum()) if signal_rows.height else 0
    medium_penalized = int((signal_rows["medium_penalty"] > 0.0).sum()) if signal_rows.height else 0
    context_penalized = int((signal_rows["context_penalty"] > 0.0).sum()) if signal_rows.height else 0
    summary = {
        "sleeve_id": sleeve_id,
        "base_sleeve_id": "candidate_p3_k0p5_b0_c0_r0",
        "sector_rank_source": args.rank_source,
        "sector_penalty_mode": args.sector_penalty_mode,
        "sector_penalty": sector_penalty,
        "relative_confirm": args.relative_confirm,
        "bottom_rank_threshold": args.bottom_rank_threshold,
        "medium_penalty_mode": args.medium_penalty_mode,
        "medium_penalty": medium_penalty,
        "weak_threshold": args.weak_threshold,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "raw_top3_overlap_rows": overlap,
        "raw_top3_overlap_ratio": overlap / signal_rows.height if signal_rows.height else None,
        "selected_sector_penalized_rows": sector_penalized,
        "selected_medium_penalized_rows": medium_penalized,
        "selected_context_penalized_rows": context_penalized,
        "selected_context_penalized_ratio": context_penalized / signal_rows.height if signal_rows.height else None,
        "selected_sector_penalty_mean": float(signal_rows["sector_penalty"].mean()) if signal_rows.height else None,
        "selected_medium_penalty_mean": float(signal_rows["medium_penalty"].mean()) if signal_rows.height else None,
        "selected_context_penalty_mean": float(signal_rows["context_penalty"].mean()) if signal_rows.height else None,
        "selected_context_penalty_max": float(signal_rows["context_penalty"].max()) if signal_rows.height else None,
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
        "model_name": "static_factor_sleeve_context_combo",
        "feature_mode": summary["sleeve_id"],
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "sector_map": str(args.sector_map),
            "sector_start_date": args.sector_start_date,
            "top_n": args.top_n,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "sector_rank_source": args.rank_source,
            "sector_penalty_mode": args.sector_penalty_mode,
            "sector_penalty": summary["sector_penalty"],
            "relative_confirm": args.relative_confirm,
            "bottom_rank_threshold": args.bottom_rank_threshold,
            "medium_penalty_mode": args.medium_penalty_mode,
            "medium_penalty": summary["medium_penalty"],
            "weak_threshold": args.weak_threshold,
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
    parser = argparse.ArgumentParser(description="Export P3 combined context rerank signals")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--sector-start-date", default="2019-01-01")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    parser.add_argument("--rank-source", choices=["5d", "10d", "20d", "mix_10_20"], default="mix_10_20")
    parser.add_argument("--sector-penalty-mode", choices=["bucket", "linear"], default="linear")
    parser.add_argument(
        "--relative-confirm",
        choices=["none", "rel5_under0", "rel10_under0", "rel20_under0"],
        default="rel20_under0",
    )
    parser.add_argument("--bottom-rank-threshold", type=float, default=0.40)
    parser.add_argument("--sector-penalties", type=parse_penalties, default=[0.02, 0.03])
    parser.add_argument("--medium-penalty-mode", choices=["bucket", "linear"], default="linear")
    parser.add_argument("--weak-threshold", type=float, default=0.50)
    parser.add_argument("--medium-penalties", type=parse_penalties, default=[0.03])
    parser.add_argument("--refresh-sector-map", action="store_true")
    parser.add_argument("--sector-map-request-sleep", type=float, default=0.35)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building P3 market feature frame ...", flush=True)
    market = build_market_frame(args)
    print("Building sector tailwind features ...", flush=True)
    sector_features = build_sector_features(args)
    print("Building medium trend quality features ...", flush=True)
    trend_features = add_medium_trend_features(market)

    output_paths: list[str] = []
    for sector_penalty in args.sector_penalties:
        for medium_penalty in args.medium_penalties:
            print(
                f"Exporting context combo sector={sector_penalty:.4f} medium={medium_penalty:.4f}",
                flush=True,
            )
            export, selected, summary = build_signal_for_penalties(
                market,
                sector_features,
                trend_features,
                sector_penalty=sector_penalty,
                medium_penalty=medium_penalty,
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
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)

    print("Saved signal metas:", flush=True)
    for path in output_paths:
        print(f"- {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
