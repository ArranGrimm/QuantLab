"""轻量因子探索脚本 — 改 FACTOR_EXPR → uv run → 看 Rank IC + 分组收益。

用法: uv run python scripts/explore_factor.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_source import TdxDailyReader, DataSourceSettings

# ═══════════════════════════════════════════════════════════════════════════
# 配置 — 改这里
# ═══════════════════════════════════════════════════════════════════════════

START_DATE = "2019-01-01"
END_DATE   = "2026-06-03"
FORWARD    = 5          # 前向收益天数
K          = 3000       # 风险厌恶系数（高质量动量用）
FACTOR_TAG = "terrified_score_20d"

def make_factor_expr() -> tuple[list[list[pl.Expr]], pl.Expr]:
    """返回 (steps, final_factor_expr)。
    steps 是 [[step1_exprs], [step2_exprs], ...]，每个内层 list 对应一次 with_columns。
    """
    # ── Terrified Score (凸显理论 STR 的简化版，同篇研报 Cell 9) ──
    ret_expr = (pl.col("close_adj") / pl.col("close_adj").shift(1).over("code") - 1.0).alias("_ret")
    mkt_expr = pl.col("_ret").mean().over("date").alias("_mkt")
    sigma = (pl.col("_ret") - pl.col("_mkt")).abs() / (pl.col("_ret").abs() + pl.col("_mkt").abs() + 0.1)
    weighted = sigma * pl.col("_ret")
    avg = weighted.rolling_mean(20).over("code")
    std = weighted.rolling_std(20).over("code")
    return [[ret_expr], [mkt_expr]], ((avg + std) * 0.5).alias("factor")

    # ── 高质量动量 ──
    # r_60 = pl.col("close_adj") / pl.col("close_adj").shift(60).over("code") - 1.0
    # sigma_60 = pl.col("close_adj").pct_change().rolling_std(60).over("code")
    # return [], (r_60 - 3000 * sigma_60.pow(2)).alias("factor")

    # ── 上影线压力 ──
    # us = pl.col("high_adj") - pl.max_horizontal(pl.col("close_adj"), pl.col("open_adj"))
    # rng = pl.col("high_adj") - pl.col("low_adj")
    # return [], (us / pl.max_horizontal(rng, pl.lit(1e-12))).rolling_mean(20).over("code").alias("factor")

    # ── MA 收敛 PCF ──
    # ma5  = pl.col("close_adj").rolling_mean(5).over("code")
    # ma10 = pl.col("close_adj").rolling_mean(10).over("code")
    # ma20 = pl.col("close_adj").rolling_mean(20).over("code")
    # ma60 = pl.col("close_adj").rolling_mean(60).over("code")
    # std_ma = pl.sum_horizontal([(ma5 - ma20).abs(), (ma10 - ma20).abs(), (ma60 - ma20).abs()]) / 3.0
    # return [], (-std_ma).alias("factor")

    # ── 价格位置 ──
    # return [], ((pl.col("close_adj") - pl.col("close_adj").rolling_min(20).over("code"))
    #         / pl.max_horizontal(pl.col("close_adj").rolling_max(20).over("code")
    #                             - pl.col("close_adj").rolling_min(20).over("code"),
    #                             pl.lit(0.01))).alias("factor")


# ═══════════════════════════════════════════════════════════════════════════
# 加载 + 计算
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()

    tdx_db = ROOT.parent / "QuantData" / "Ashare" / "tdx.db"
    ds = DataSourceSettings(provider="tdx", tdx_db=tdx_db, start_date=START_DATE)
    reader = TdxDailyReader(ds)

    steps, factor_expr = make_factor_expr()

    lf = reader.load_daily_full().filter(
        pl.col("date").is_between(
            pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
            pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
        )
    ).sort(["code", "date"])

    for step in steps:
        lf = lf.with_columns(step)

    lf = lf.with_columns([
        factor_expr,
        (pl.col("close_adj").shift(-FORWARD).over("code")
         / pl.col("close_adj") - 1.0).alias(f"_fwd_{FORWARD}d"),
    ])

    lf = lf.select(["code", "date", "close_adj", "market_cap_100m", "amount",
                    "factor", f"_fwd_{FORWARD}d"])

    df = lf.collect()
    reader.close()
    n_stocks = df["code"].n_unique()

    logger.info(f"加载: {df.height:,} 行, {n_stocks:,} 只, {time.perf_counter() - t0:.1f}s")

    # ── Rank IC ──
    ic = (
        df.filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan(),
                  pl.col(f"_fwd_{FORWARD}d").is_not_null())
        .group_by("date")
        .agg(
            pl.corr(pl.col("factor").rank("average"),
                    pl.col(f"_fwd_{FORWARD}d").rank("average")).alias("rank_ic"),
            pl.len().alias("n"),
        )
        .sort("date")
        .filter(pl.col("n") >= 100)
    )

    mean_ic = ic["rank_ic"].mean()
    std_ic  = ic["rank_ic"].std()
    ic_ir   = mean_ic / std_ic if std_ic is not None and std_ic > 0 else 0
    pos_ratio = (ic["rank_ic"] > 0).sum() / ic.height if ic.height > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Factor: {FACTOR_TAG}  |  Forward: {FORWARD}d  |  {ic.height} days")
    print(f"{'='*60}")
    print(f"  Mean Rank IC:  {mean_ic:>8.4f}")
    print(f"  IC Std:        {std_ic:>8.4f}")
    print(f"  IC IR:         {ic_ir:>8.2f}")
    print(f"  Positive %:    {pos_ratio:>8.1%}")

    # ── 年度 Rank IC ──
    ic_yearly = ic.with_columns(pl.col("date").dt.year().alias("year")).filter(
        pl.col("rank_ic").is_not_null() & pl.col("rank_ic").is_not_nan()
    )
    yearly = (
        ic_yearly.group_by("year")
        .agg(pl.col("rank_ic").mean().alias("mean_ic"),
             pl.col("rank_ic").std().alias("std_ic"),
             (pl.col("rank_ic") > 0).sum().alias("pos"),
             pl.len().alias("n"))
        .sort("year")
    )
    print(f"\n  {'Year':>6}  {'Mean IC':>8}  {'Std':>8}  {'IR':>6}  {'Pos%':>7}  {'Days':>5}")
    print(f"  {'-'*50}")
    for r in yearly.iter_rows(named=True):
        yr_std = r["std_ic"] or 0
        yr_ir = r["mean_ic"] / yr_std if yr_std > 0 else 0
        print(f"  {r['year']:>6}  {r['mean_ic']:>8.4f}  {r['std_ic']:>8.4f}  "
              f"{yr_ir:>6.2f}  {r['pos']/r['n']:>7.1%}  {r['n']:>5}")

    # ── 分组收益 ──
    valid = df.filter(
        pl.col("factor").is_not_null() & pl.col("factor").is_not_nan(),
        pl.col(f"_fwd_{FORWARD}d").is_not_null(),
    )
    if valid.height == 0:
        print("\n  无有效因子值，跳过分组分析\n")
        return

    grouped = (
        valid
        .with_columns(
            (pl.col("factor").rank("average").over("date")
             / pl.len().over("date") * 5).ceil().cast(pl.Int32).alias("_q"),
        )
        .group_by("_q")
        .agg(pl.col(f"_fwd_{FORWARD}d").mean().alias("avg_ret"),
             pl.len().alias("n"))
        .sort("_q")
    )

    print(f"\n  {'Quintile':>10}  {'Avg Ret':>10}  {'Ann(est)':>10}  {'Rows':>12}")
    print(f"  {'-'*50}")
    for r in grouped.iter_rows(named=True):
        ann = r["avg_ret"] * 252 / FORWARD * 100
        print(f"  {'Q' + str(r['_q']):>10}  {r['avg_ret']*100:>10.4f}%  "
              f"{ann:>10.1f}%  {r['n']:>12,}")

    # 多空
    q1 = grouped.filter(pl.col("_q") == 1)["avg_ret"][0]
    q5 = grouped.filter(pl.col("_q") == 5)["avg_ret"][0]
    print(f"  {'Q5-Q1':>10}  {(q5 - q1)*100:>10.4f}%")
    print(f"\n  耗时: {time.perf_counter() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
