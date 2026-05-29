"""Export P3 sector-tailwind rerank signals for bt-amv-topn.

This is intentionally a narrow prototype: apply a soft penalty to P3 candidates
whose industry return rank is in the bottom bucket, then re-rank the same
candidate pool and export signal.parquet artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT, _finite_expr, _git_commit, _rel_path
from scripts.amv_sector_tailwind_diagnostic import (
    DEFAULT_SECTOR_MAP,
    build_sector_tailwind_features,
    load_daily_with_industry,
    load_sector_map,
)
from scripts.amv_static_sleeve_signal_export import pullback_combo_score_expr


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_static_sleeve_signals"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def penalty_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def threshold_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def rank_source_token(value: str) -> str:
    return value.replace("_", "")


def parse_penalties(value: str) -> list[float]:
    penalties = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not penalties:
        raise argparse.ArgumentTypeError("penalties must not be empty")
    if any(penalty < 0.0 for penalty in penalties):
        raise argparse.ArgumentTypeError("penalties must be non-negative")
    return penalties


def build_market_frame(args: argparse.Namespace) -> pl.DataFrame:
    from scripts.amv_static_sleeve_signal_export import build_feature_frame

    return build_feature_frame(args)


def build_sector_features(args: argparse.Namespace) -> pl.DataFrame:
    sector_map = load_sector_map(
        args.sector_map,
        refresh=args.refresh_sector_map,
        request_sleep=args.sector_map_request_sleep,
    )
    daily = load_daily_with_industry(args.qmt_db, sector_map, args.sector_start_date)
    return build_sector_tailwind_features(daily).select(
        [
            "date",
            "code",
            "industry",
            "sector_ret_5d_rank_pct",
            "sector_ret_10d_rank_pct",
            "sector_ret_20d_rank_pct",
            "stock_rel_sector_ret_5d",
            "stock_rel_sector_ret_10d",
            "stock_rel_sector_ret_20d",
            "sector_breadth_ma20",
            "sector_amount_ratio_20",
            "sector_tailwind_ok",
        ]
    )


def sector_rank_expr(args: argparse.Namespace) -> pl.Expr:
    if args.rank_source == "5d":
        return pl.col("sector_ret_5d_rank_pct")
    if args.rank_source == "10d":
        return pl.col("sector_ret_10d_rank_pct")
    if args.rank_source == "20d":
        return pl.col("sector_ret_20d_rank_pct")
    if args.rank_source == "mix_10_20":
        return (pl.col("sector_ret_10d_rank_pct") + pl.col("sector_ret_20d_rank_pct")) / 2.0
    raise ValueError(f"unknown rank source: {args.rank_source}")


def relative_confirm_expr(args: argparse.Namespace) -> pl.Expr:
    if args.relative_confirm == "none":
        return pl.lit(True)
    if args.relative_confirm == "rel5_under0":
        return pl.col("stock_rel_sector_ret_5d").fill_null(0.0) < 0.0
    if args.relative_confirm == "rel10_under0":
        return pl.col("stock_rel_sector_ret_10d").fill_null(0.0) < 0.0
    if args.relative_confirm == "rel20_under0":
        return pl.col("stock_rel_sector_ret_20d").fill_null(0.0) < 0.0
    raise ValueError(f"unknown relative confirm: {args.relative_confirm}")


def build_signal_for_penalty(
    market: pl.DataFrame,
    sector_features: pl.DataFrame,
    *,
    penalty: float,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    sleeve_id = (
        f"p3_sector_{rank_source_token(args.rank_source)}_{args.penalty_mode}"
        f"_b{threshold_token(args.bottom_rank_threshold)}_p{penalty_token(penalty)}"
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

    rank_expr = sector_rank_expr(args).fill_null(1.0)
    bottom_distance_expr = (args.bottom_rank_threshold - rank_expr) / args.bottom_rank_threshold
    bottom_strength_expr = pl.when(bottom_distance_expr > 0.0).then(bottom_distance_expr).otherwise(0.0)
    confirm_expr = relative_confirm_expr(args)
    if args.penalty_mode == "bucket":
        penalty_expr = pl.when((rank_expr < args.bottom_rank_threshold) & confirm_expr).then(penalty).otherwise(0.0)
    elif args.penalty_mode == "linear":
        penalty_expr = pl.when(confirm_expr).then(bottom_strength_expr * penalty).otherwise(0.0)
    else:
        raise ValueError(f"unknown penalty mode: {args.penalty_mode}")

    scored = (
        market.join(sector_features, on=["date", "code"], how="left")
        .with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            [
                rank_expr.alias("_sector_rank_score"),
                confirm_expr.alias("_relative_confirm"),
                base_score_expr.alias("_base_signal_score"),
            ]
        )
        .with_columns(
            [
                (pl.col("_sector_rank_score") < args.bottom_rank_threshold).alias("_sector_bottom_rank"),
                penalty_expr.alias("_sector_penalty"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(pl.col("_base_signal_score") - pl.col("_sector_penalty"))
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
                "industry",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_base_signal_score").alias("base_score"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
                pl.col("_sector_rank_score").alias("sector_rank_score"),
                pl.col("_sector_penalty").alias("sector_penalty"),
                pl.col("_relative_confirm").alias("relative_confirm"),
                pl.col("sector_ret_5d_rank_pct"),
                pl.col("sector_ret_10d_rank_pct"),
                pl.col("sector_ret_20d_rank_pct"),
                pl.col("stock_rel_sector_ret_5d"),
                pl.col("stock_rel_sector_ret_10d"),
                pl.col("stock_rel_sector_ret_20d"),
                pl.col("sector_breadth_ma20"),
                pl.col("sector_amount_ratio_20"),
                pl.col("sector_tailwind_ok"),
                pl.col("_sector_bottom_rank").alias("sector_bottom_rank"),
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
    bottom_selected = int(signal_rows["sector_bottom_rank"].sum()) if signal_rows.height else 0
    summary = {
        "sleeve_id": sleeve_id,
        "base_sleeve_id": "candidate_p3_k0p5_b0_c0_r0",
        "sector_rank_source": args.rank_source,
        "penalty_mode": args.penalty_mode,
        "relative_confirm": args.relative_confirm,
        "penalty": penalty,
        "bottom_rank_threshold": args.bottom_rank_threshold,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "raw_top3_overlap_rows": overlap,
        "raw_top3_overlap_ratio": overlap / signal_rows.height if signal_rows.height else None,
        "selected_sector_bottom_rank_rows": bottom_selected,
        "selected_sector_bottom_rank_ratio": bottom_selected / signal_rows.height if signal_rows.height else None,
        "selected_relative_confirm_rows": int(signal_rows["relative_confirm"].sum()) if signal_rows.height else 0,
        "selected_penalized_rows": int((signal_rows["sector_penalty"] > 0.0).sum()) if signal_rows.height else 0,
        "selected_penalized_ratio": float((signal_rows["sector_penalty"] > 0.0).mean()) if signal_rows.height else None,
        "selected_penalty_mean": float(signal_rows["sector_penalty"].mean()) if signal_rows.height else None,
        "selected_penalty_max": float(signal_rows["sector_penalty"].max()) if signal_rows.height else None,
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
        "model_name": "static_factor_sleeve_sector_tailwind",
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
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
            "sector_rank_source": args.rank_source,
            "penalty_mode": args.penalty_mode,
            "relative_confirm": args.relative_confirm,
            "bottom_rank_threshold": args.bottom_rank_threshold,
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
    parser = argparse.ArgumentParser(description="Export P3 sector-tailwind rerank signals")
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
    parser.add_argument("--rank-window", type=int, choices=[5, 10, 20], default=10)
    parser.add_argument("--rank-source", choices=["5d", "10d", "20d", "mix_10_20"], default=None)
    parser.add_argument("--penalty-mode", choices=["bucket", "linear"], default="bucket")
    parser.add_argument(
        "--relative-confirm",
        choices=["none", "rel5_under0", "rel10_under0", "rel20_under0"],
        default="none",
    )
    parser.add_argument("--bottom-rank-threshold", type=float, default=0.40)
    parser.add_argument("--penalties", type=parse_penalties, default=[0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--refresh-sector-map", action="store_true")
    parser.add_argument("--sector-map-request-sleep", type=float, default=0.35)
    args = parser.parse_args()
    if args.rank_source is None:
        args.rank_source = f"{args.rank_window}d"

    started_at = datetime.now()
    print("Building P3 market feature frame ...")
    market = build_market_frame(args)
    print("Building sector tailwind features ...")
    sector_features = build_sector_features(args)

    output_paths: list[str] = []
    for penalty in args.penalties:
        print(f"Exporting sector penalty={penalty:.4f}")
        export, selected, summary = build_signal_for_penalty(
            market,
            sector_features,
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
