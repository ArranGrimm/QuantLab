from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    add_combo_scores,
    build_dataset,
)
from utils.active_market_value_regime import build_active_market_value_regime_frame


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_bull_pool_factor_regime"

FACTOR_SPECS: list[dict[str, Any]] = [
    {
        "id": "combo_p2_k0p5_r0",
        "label": "当前组合 P2/K0.5/R0",
        "factor": "_score_combo_p2_k0p5_r0",
        "higher": True,
        "group": "组合",
    },
    {"id": "price_pos_20d", "label": "20日高位", "factor": "price_pos_20d", "higher": True, "group": "位置"},
    {
        "id": "near_high_20d",
        "label": "接近20日新高",
        "factor": "close_to_high_20d",
        "higher": False,
        "group": "位置",
    },
    {"id": "klen_contract", "label": "K线振幅收缩", "factor": "KLEN", "higher": False, "group": "K线"},
    {"id": "kmid2_strong", "label": "实体占比偏强", "factor": "KMID2", "higher": True, "group": "K线"},
    {"id": "ret_5d", "label": "5日动量", "factor": "ret_5d", "higher": True, "group": "动量"},
    {"id": "ret_20d", "label": "20日动量", "factor": "ret_20d", "higher": True, "group": "动量"},
    {"id": "atr_14_pct_low", "label": "ATR低风险", "factor": "atr_14_pct", "higher": False, "group": "风险"},
    {
        "id": "panic_vol_low",
        "label": "卖压低",
        "factor": "panic_vol_ratio_20d",
        "higher": False,
        "group": "风险",
    },
    {"id": "intraday_pos", "label": "收盘靠高", "factor": "intraday_pos", "higher": True, "group": "日内"},
    {
        "id": "turnover_ma_ratio",
        "label": "换手放大",
        "factor": "turnover_ma_ratio",
        "higher": True,
        "group": "量能",
    },
]


def combo_ranker() -> dict[str, Any]:
    return {
        "id": "combo_p2_k0p5_r0",
        "label": "当前组合 P2/K0.5/R0",
        "group": "组合",
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KLEN", "higher_is_better": False, "weight": 0.5},
            {"factor": "KMID2", "higher_is_better": True, "weight": 0.5},
        ],
    }


def build_bull_phase_frame() -> pl.DataFrame:
    regime = (
        build_active_market_value_regime_frame()
        .select(["date", "is_bull_regime"])
        .sort("date")
    )
    bull_run_id = 0
    bull_day = 0
    prev_bull = False
    run_ids: list[int | None] = []
    run_days: list[int | None] = []
    phases: list[str] = []

    for row in regime.select("is_bull_regime").to_dicts():
        is_bull = bool(row["is_bull_regime"])
        if is_bull and not prev_bull:
            bull_run_id += 1
            bull_day = 1
        elif is_bull:
            bull_day += 1
        else:
            bull_day = 0

        if not is_bull:
            run_ids.append(None)
            run_days.append(None)
            phases.append("non_bull")
        else:
            run_ids.append(bull_run_id)
            run_days.append(bull_day)
            if bull_day <= 5:
                phases.append("early")
            elif bull_day <= 20:
                phases.append("middle")
            else:
                phases.append("late")
        prev_bull = is_bull

    return regime.with_columns(
        [
            pl.Series("bull_run_id", run_ids, dtype=pl.Int64),
            pl.Series("bull_day", run_days, dtype=pl.Int64),
            pl.Series("bull_phase", phases),
        ]
    ).select(["date", "bull_run_id", "bull_day", "bull_phase"])


def prepare_dataset(args: argparse.Namespace) -> pl.DataFrame:
    df_pool = add_combo_scores(build_dataset(args), [combo_ranker()])
    required_cols = sorted(
        {
            "date",
            "code",
            "ret_5d",
            f"fwd_ret_{args.horizon}d",
            f"fwd_mfe_{args.horizon}d",
            *(str(spec["factor"]) for spec in FACTOR_SPECS),
        }
    )
    df = df_pool.select(required_cols).with_columns(pl.col("date").dt.year().alias("year"))
    daily_env = (
        df.group_by("date")
        .agg(
            [
                pl.col("ret_5d").mean().alias("trail_pool_ret_5d"),
                pl.col(f"fwd_ret_{args.horizon}d").mean().alias("forward_pool_ret_6d"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("trail_pool_ret_5d") >= pl.col("trail_pool_ret_5d").median())
                .then(pl.lit("profit_strong"))
                .otherwise(pl.lit("profit_weak"))
                .alias("trail_profit_regime"),
                pl.when(pl.col("forward_pool_ret_6d") >= pl.col("forward_pool_ret_6d").median())
                .then(pl.lit("forward_strong"))
                .otherwise(pl.lit("forward_weak"))
                .alias("forward_env"),
            ]
        )
    )
    return (
        df.join(daily_env, on="date", how="left")
        .join(build_bull_phase_frame(), on="date", how="left")
        .sort(["date", "code"])
    )


def daily_factor_metrics(
    df: pl.DataFrame,
    spec: dict[str, Any],
    *,
    horizon: int,
    top_n: int,
) -> pl.DataFrame:
    factor = str(spec["factor"])
    label_col = f"fwd_ret_{horizon}d"
    mfe_col = f"fwd_mfe_{horizon}d"
    score_expr = pl.col(factor) if bool(spec["higher"]) else -pl.col(factor)

    valid = (
        df.filter(
            pl.col(factor).is_not_null()
            & pl.col(factor).is_not_nan()
            & pl.col(label_col).is_not_null()
            & pl.col(label_col).is_not_nan()
        )
        .with_columns(
            [
                score_expr.alias("_score"),
                score_expr.rank(method="average").over("date").alias("_score_rank"),
                pl.col(label_col).rank(method="average").over("date").alias("_label_rank"),
            ]
        )
    )
    baseline = (
        valid.group_by("date")
        .agg(
            [
                pl.col("year").first(),
                pl.col("bull_phase").first(),
                pl.col("trail_profit_regime").first(),
                pl.col("forward_env").first(),
                pl.corr("_score_rank", "_label_rank").alias("ic"),
                pl.col(label_col).mean().alias("baseline_ret"),
                (pl.col(mfe_col) >= 0.15).mean().alias("baseline_hit15"),
                pl.len().alias("n_candidates"),
            ]
        )
        .filter(pl.col("n_candidates") >= top_n)
    )
    top = (
        valid.sort(["date", "_score", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .group_by("date")
        .agg(
            [
                pl.col(label_col).mean().alias("top_ret"),
                (pl.col(mfe_col) >= 0.15).mean().alias("top_hit15"),
            ]
        )
    )
    return (
        baseline.join(top, on="date", how="inner")
        .with_columns(
            [
                pl.lit(str(spec["id"])).alias("factor_id"),
                pl.lit(str(spec["label"])).alias("label"),
                pl.lit(str(spec["group"])).alias("factor_group"),
                (pl.col("top_ret") - pl.col("baseline_ret")).alias("edge_ret"),
            ]
        )
        .select(
            [
                "date",
                "year",
                "bull_phase",
                "trail_profit_regime",
                "forward_env",
                "factor_id",
                "label",
                "factor_group",
                "ic",
                "top_ret",
                "baseline_ret",
                "edge_ret",
                "top_hit15",
                "baseline_hit15",
                "n_candidates",
            ]
        )
        .sort("date")
    )


def summarize(daily: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    keys = ["factor_id", "label", "factor_group", *group_cols]
    return (
        daily.group_by(keys)
        .agg(
            [
                pl.col("date").n_unique().alias("days"),
                pl.col("ic").mean().alias("mean_ic"),
                (pl.col("ic") > 0).mean().alias("positive_ic_ratio"),
                pl.col("top_ret").mean().alias("mean_top_ret"),
                pl.col("baseline_ret").mean().alias("mean_baseline_ret"),
                pl.col("edge_ret").mean().alias("mean_edge_ret"),
                (pl.col("edge_ret") > 0).mean().alias("positive_edge_ratio"),
                pl.col("top_hit15").mean().alias("mean_top_hit15"),
                pl.col("baseline_hit15").mean().alias("mean_baseline_hit15"),
                pl.col("n_candidates").median().alias("median_candidates"),
            ]
        )
        .with_columns((pl.col("mean_top_hit15") - pl.col("mean_baseline_hit15")).alias("mean_edge_hit15"))
        .sort([*group_cols, "mean_edge_ret"] if group_cols else ["mean_edge_ret"], descending=True)
    )


def top_records(df: pl.DataFrame, metric: str, n: int = 12) -> list[dict[str, Any]]:
    return df.sort(metric, descending=True).head(n).to_dicts()


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool factor label analysis by regime")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--horizons", default=None, help="兼容 build_dataset；默认自动使用 --horizon")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    args.horizons = [args.horizon]

    started_at = datetime.now()
    print("Building AMV bull pool 6d dataset...")
    df = prepare_dataset(args)
    date_min = df["date"].min()
    date_max = df["date"].max()
    print(f"Rows: {df.height:,}; dates: {date_min} -> {date_max}")

    daily_frames = []
    for spec in FACTOR_SPECS:
        print(f"Evaluating {spec['label']}...")
        daily_frames.append(daily_factor_metrics(df, spec, horizon=args.horizon, top_n=args.top_n))
    daily = pl.concat(daily_frames, how="vertical")

    by_factor = summarize(daily, [])
    by_year = summarize(daily, ["year"])
    by_phase = summarize(daily, ["bull_phase"])
    by_profit = summarize(daily, ["trail_profit_regime"])
    by_forward_env = summarize(daily, ["forward_env"])

    output_dir = args.output_root / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.write_csv(output_dir / "daily_factor_metrics.csv")
    by_factor.write_csv(output_dir / "by_factor.csv")
    by_year.write_csv(output_dir / "by_year.csv")
    by_phase.write_csv(output_dir / "by_bull_phase.csv")
    by_profit.write_csv(output_dir / "by_trail_profit_regime.csv")
    by_forward_env.write_csv(output_dir / "by_forward_env.csv")

    summary = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizon": args.horizon,
            "top_n": args.top_n,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
        },
        "pool": {
            "rows": df.height,
            "date_min": str(date_min),
            "date_max": str(date_max),
            "unique_codes": df["code"].n_unique(),
            "unique_dates": df["date"].n_unique(),
        },
        "files": {
            "daily": "daily_factor_metrics.csv",
            "by_factor": "by_factor.csv",
            "by_year": "by_year.csv",
            "by_bull_phase": "by_bull_phase.csv",
            "by_trail_profit_regime": "by_trail_profit_regime.csv",
            "by_forward_env": "by_forward_env.csv",
        },
        "top_by_factor_edge": top_records(by_factor, "mean_edge_ret"),
        "top_by_factor_ic": top_records(by_factor, "mean_ic"),
        "top_by_year_edge": top_records(by_year, "mean_edge_ret", n=20),
        "top_by_phase_edge": top_records(by_phase, "mean_edge_ret", n=20),
        "top_by_profit_regime_edge": top_records(by_profit, "mean_edge_ret", n=20),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_dir / 'summary.json'}")
    print("\nTop factors by all-sample edge:")
    for row in summary["top_by_factor_edge"][:8]:
        print(
            f"- {row['label']}: edge={row['mean_edge_ret'] * 100:+.3f}pp "
            f"IC={row['mean_ic']:+.4f} days={row['days']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
