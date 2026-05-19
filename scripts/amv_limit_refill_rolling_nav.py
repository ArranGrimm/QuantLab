from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import _finite_expr, _git_commit, _rel_path
from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_QMT_DB,
    ROOT,
    _rolling_sleeve_nav,
    build_dataset,
)
from scripts.amv_static_sleeve_signal_export import parse_sleeves, sleeve_score_expr


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_limit_refill_rolling_nav"
LIMIT_TOLERANCE = 0.001


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _price_limit_pct_expr() -> pl.Expr:
    is_20pct_board = (
        pl.col("code").str.starts_with("sz.300")
        | pl.col("code").str.starts_with("sz.301")
        | pl.col("code").str.starts_with("sh.688")
        | pl.col("code").str.starts_with("sh.689")
        | pl.col("code").str.starts_with("300")
        | pl.col("code").str.starts_with("301")
        | pl.col("code").str.starts_with("688")
        | pl.col("code").str.starts_with("689")
    )
    return pl.when(is_20pct_board).then(0.20).otherwise(0.10)


def _quantile(series: pl.Series, q: float) -> float | None:
    if series.is_empty():
        return None
    value = series.quantile(q)
    return None if value is None else float(value)


def _mean_bool(series: pl.Series) -> float:
    if series.is_empty():
        return 0.0
    return float(series.cast(pl.Float64).mean())


def _selected_daily_frame(
    *,
    dates: pl.DataFrame,
    selected: pl.DataFrame,
    horizon: int,
) -> pl.DataFrame:
    if selected.is_empty():
        daily = pl.DataFrame(
            schema={
                "date": pl.Date,
                "daily_ret": pl.Float64,
                "hit15": pl.Float64,
                "n_picks": pl.UInt32,
            }
        )
    else:
        daily = (
            selected.group_by("date")
            .agg(
                [
                    pl.col(f"fwd_ret_{horizon}d").mean().alias("daily_ret"),
                    (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("hit15"),
                    pl.len().cast(pl.UInt32).alias("n_picks"),
                ]
            )
            .sort("date")
        )

    return (
        dates.join(daily, on="date", how="left")
        .with_columns(
            [
                pl.col("daily_ret").fill_null(0.0),
                pl.col("hit15").fill_null(0.0),
                pl.col("n_picks").fill_null(0).cast(pl.UInt32),
            ]
        )
        .sort("date")
    )


def _evaluate_selected(
    *,
    dates: pl.DataFrame,
    selected: pl.DataFrame,
    sleeve_id: str,
    scenario: str,
    horizon: int,
    top_n: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    daily = _selected_daily_frame(dates=dates, selected=selected, horizon=horizon)
    nav_end, max_dd = _rolling_sleeve_nav(daily["daily_ret"].to_numpy(), horizon)

    metrics = {
        "sleeve_id": sleeve_id,
        "scenario": scenario,
        "horizon": horizon,
        "eligible_days": dates.height,
        "selected_rows": selected.height,
        "full_days": int((daily["n_picks"] == top_n).sum()),
        "avg_picks_per_day": float(daily["n_picks"].mean()) if daily.height else 0.0,
        "mean_daily_ret": float(daily["daily_ret"].mean()) if daily.height else 0.0,
        "hit15": float(daily["hit15"].mean()) if daily.height else 0.0,
        "nav_end": nav_end,
        "max_dd": max_dd,
    }
    return metrics, daily.with_columns(
        [
            pl.lit(sleeve_id).alias("sleeve_id"),
            pl.lit(scenario).alias("scenario"),
            pl.lit(horizon).alias("horizon"),
        ]
    )


def _select_scenarios(
    df_scored: pl.DataFrame,
    *,
    eligible_dates: list[object],
    top_n: int,
    max_scan_rank: int,
) -> dict[str, pl.DataFrame]:
    df_eligible = df_scored.filter(pl.col("date").is_in(eligible_dates))
    if max_scan_rank > 0:
        df_eligible = df_eligible.filter(pl.col("rank") <= max_scan_rank)

    original_top = (
        df_eligible.sort(["date", "score", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .sort(["date", "rank", "code"])
    )
    no_refill = original_top.filter(~pl.col("is_close_limit_up"))
    refill = (
        df_eligible.filter(~pl.col("is_close_limit_up"))
        .sort(["date", "score", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .sort(["date", "rank", "code"])
    )
    return {
        "original_top3": original_top,
        "drop_limit_no_refill": no_refill,
        "skip_limit_refill_top3": refill,
    }


def _build_scored_pool(df_pool: pl.DataFrame, sleeve_id: str) -> pl.DataFrame:
    score_expr, required_cols = sleeve_score_expr(sleeve_id)
    valid_expr = _finite_expr(required_cols[0])
    for col_name in required_cols[1:]:
        valid_expr = valid_expr & _finite_expr(col_name)

    return (
        df_pool.with_columns(
            [
                valid_expr.alias("is_valid_candidate"),
                (
                    (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0)
                    >= (_price_limit_pct_expr() - LIMIT_TOLERANCE)
                ).alias("is_close_limit_up"),
            ]
        )
        .with_columns(
            pl.when(pl.col("is_valid_candidate"))
            .then(score_expr)
            .otherwise(None)
            .alias("score")
        )
        .filter(pl.col("is_valid_candidate") & _finite_expr("score"))
        .with_columns(
            pl.col("score").rank(method="ordinal", descending=True).over("date").cast(pl.UInt32).alias("rank")
        )
    )


def _scenario_diagnostics(
    *,
    selected: dict[str, pl.DataFrame],
    eligible_dates: list[object],
    top_n: int,
) -> dict[str, Any]:
    dates_df = pl.DataFrame({"date": eligible_dates})
    original_daily = (
        dates_df.join(
            selected["original_top3"]
            .group_by("date")
            .agg(
                [
                    pl.col("is_close_limit_up").sum().cast(pl.UInt32).alias("original_top_limit_up_count"),
                    pl.len().cast(pl.UInt32).alias("original_top_count"),
                ]
            ),
            on="date",
            how="left",
        )
        .with_columns(
            [
                pl.col("original_top_limit_up_count").fill_null(0).cast(pl.UInt32),
                pl.col("original_top_count").fill_null(0).cast(pl.UInt32),
            ]
        )
    )
    refill = selected["skip_limit_refill_top3"]
    if refill.is_empty():
        refill_daily = dates_df.with_columns(
            [
                pl.lit(0, dtype=pl.UInt32).alias("refill_count"),
                pl.lit(None, dtype=pl.UInt32).alias("refill_max_rank"),
                pl.lit(False).alias("refill_used_beyond_top3"),
            ]
        )
    else:
        refill_daily = (
            dates_df.join(
                refill.group_by("date").agg(
                    [
                        pl.len().cast(pl.UInt32).alias("refill_count"),
                        pl.col("rank").max().cast(pl.UInt32).alias("refill_max_rank"),
                        (pl.col("rank") > top_n).any().alias("refill_used_beyond_top3"),
                    ]
                ),
                on="date",
                how="left",
            )
            .with_columns(
                [
                    pl.col("refill_count").fill_null(0).cast(pl.UInt32),
                    pl.col("refill_used_beyond_top3").fill_null(False),
                ]
            )
        )

    selected_ranks = refill["rank"] if not refill.is_empty() else pl.Series("rank", [], dtype=pl.UInt32)
    return {
        "original_top_limit_up_rows": int(original_daily["original_top_limit_up_count"].sum()),
        "original_top_limit_up_days": int((original_daily["original_top_limit_up_count"] > 0).sum()),
        "original_top_limit_up_day_share": _mean_bool(original_daily["original_top_limit_up_count"] > 0),
        "original_top_limit_up_rows_per_day": float(original_daily["original_top_limit_up_count"].mean()),
        "refill_full_days": int((refill_daily["refill_count"] == top_n).sum()),
        "refill_used_beyond_top3_rows": int((selected_ranks > top_n).sum()) if not selected_ranks.is_empty() else 0,
        "refill_used_beyond_top3_days": int(refill_daily["refill_used_beyond_top3"].sum()),
        "refill_used_beyond_top3_day_share": _mean_bool(refill_daily["refill_used_beyond_top3"]),
        "refill_rank_mean": float(selected_ranks.mean()) if not selected_ranks.is_empty() else None,
        "refill_rank_q95": _quantile(selected_ranks, 0.95),
        "refill_rank_max": int(selected_ranks.max()) if not selected_ranks.is_empty() else None,
    }


def evaluate_sleeve(df_pool: pl.DataFrame, sleeve_id: str, args: argparse.Namespace) -> dict[str, Any]:
    scored = _build_scored_pool(df_pool, sleeve_id)
    daily_counts = (
        scored.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= args.top_n)
        .sort("date")
    )
    eligible_dates = daily_counts["date"].to_list()
    dates_df = daily_counts.select("date")
    selected = _select_scenarios(
        scored,
        eligible_dates=eligible_dates,
        top_n=args.top_n,
        max_scan_rank=args.max_scan_rank,
    )

    summary_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []
    for horizon in args.horizons:
        for scenario, scenario_selected in selected.items():
            metrics, daily = _evaluate_selected(
                dates=dates_df,
                selected=scenario_selected,
                sleeve_id=sleeve_id,
                scenario=scenario,
                horizon=horizon,
                top_n=args.top_n,
            )
            summary_rows.append(metrics)
            daily_frames.append(daily)

    selected_frames = []
    for scenario, scenario_selected in selected.items():
        selected_frames.append(
            scenario_selected.with_columns(
                [
                    pl.lit(sleeve_id).alias("sleeve_id"),
                    pl.lit(scenario).alias("scenario"),
                ]
            )
        )

    return {
        "sleeve_id": sleeve_id,
        "eligible_days": len(eligible_dates),
        "candidate_count": {
            "median": float(daily_counts["n_candidates"].median()) if daily_counts.height else None,
            "mean": float(daily_counts["n_candidates"].mean()) if daily_counts.height else None,
            "min": int(daily_counts["n_candidates"].min()) if daily_counts.height else None,
            "max": int(daily_counts["n_candidates"].max()) if daily_counts.height else None,
        },
        "diagnostics": _scenario_diagnostics(
            selected=selected,
            eligible_dates=eligible_dates,
            top_n=args.top_n,
        ),
        "summary": summary_rows,
        "daily": pl.concat(daily_frames, how="vertical") if daily_frames else pl.DataFrame(),
        "selected": pl.concat(selected_frames, how="vertical") if selected_frames else pl.DataFrame(),
    }


def write_outputs(
    *,
    output_root: Path,
    started_at: datetime,
    args: argparse.Namespace,
    sleeve_results: list[dict[str, Any]],
) -> Path:
    output_dir = output_root / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pl.DataFrame([row for result in sleeve_results for row in result["summary"]])
    daily = pl.concat([result["daily"] for result in sleeve_results], how="vertical")
    selected = pl.concat([result["selected"] for result in sleeve_results], how="vertical")

    summary_path = output_dir / "summary.csv"
    daily_path = output_dir / "daily.csv"
    selected_path = output_dir / "selected.csv"
    payload_path = output_dir / "summary.json"

    summary.write_csv(summary_path)
    daily.write_csv(daily_path)
    selected.write_csv(selected_path)

    payload = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "sleeves": args.sleeves,
            "horizons": args.horizons,
            "top_n": args.top_n,
            "max_scan_rank": args.max_scan_rank,
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
        "sleeves": [
            {
                "sleeve_id": result["sleeve_id"],
                "eligible_days": result["eligible_days"],
                "candidate_count": result["candidate_count"],
                "diagnostics": result["diagnostics"],
            }
            for result in sleeve_results
        ],
        "files": {
            "summary": _rel_path(summary_path, output_dir),
            "daily": _rel_path(daily_path, output_dir),
            "selected": _rel_path(selected_path, output_dir),
        },
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload_path


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV limit-up skip/refill rolling NAV diagnostic")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sleeves",
        type=parse_sleeves,
        default=["manual_p2_k0p5_r0", "pkm_p1_k0p5_m1", "pkm_p2_k0p5_m0p5", "pkm_p3_k1_m2"],
    )
    parser.add_argument("--horizons", type=_parse_int_list, default=[5, 6])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-scan-rank", type=int, default=0, help="0 表示从全候选池顺位补位")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.max_scan_rank < 0:
        raise ValueError("--max-scan-rank must be non-negative")

    started_at = datetime.now()
    print("Building AMV bull pool dataset...")
    df_pool = build_dataset(args)
    print(f"Rows: {df_pool.height:,}")
    print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")

    sleeve_results = []
    for sleeve_id in args.sleeves:
        print(f"\nEvaluating {sleeve_id}...")
        result = evaluate_sleeve(df_pool, sleeve_id, args)
        sleeve_results.append(result)
        diagnostics = result["diagnostics"]
        print(
            f"  limit-up rows={diagnostics['original_top_limit_up_rows']} "
            f"days={diagnostics['original_top_limit_up_days']} "
            f"refill_rank_max={diagnostics['refill_rank_max']}"
        )
        for row in result["summary"]:
            if row["scenario"] != "skip_limit_refill_top3":
                continue
            print(
                f"  {row['horizon']}d refill: "
                f"ret={row['mean_daily_ret'] * 100:+.3f}% "
                f"nav={row['nav_end'] * 100:+.2f}% "
                f"dd={row['max_dd'] * 100:.2f}% "
                f"full_days={row['full_days']}/{row['eligible_days']}"
            )

    payload_path = write_outputs(
        output_root=args.output_root,
        started_at=started_at,
        args=args,
        sleeve_results=sleeve_results,
    )
    print(f"\nSaved: {payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
