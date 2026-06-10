"""择时 regime 探索 — B1 alpha proof 方法：随机买 vs 择时买。

核心问题：不在 gate=开 的日子开仓，能省多少？

方法来源：b1_alpha_proof.py + Q7 关键洞察
  - 每天全市场等权买，拿 N 天
  - 按 gate 状态分两组：gate=开时买的回报 vs gate=关时买的回报
  - 差距 = gate 的择时 alpha（不是选股 alpha）

用法:
  uv run python research/explore_regime.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategies.amv.data import MarketConfig, build_market_lazy
from utils.baostock_utils import get_sh_index_daily
from utils.data_source import resolve_data_source

# ═══════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════

START_DATE = "2019-01-01"
END_DATE   = "2026-06-03"
FWD_DAYS   = [5, 10, 15, 20]   # 测多个持有天数
TOP_K      = 5                  # 随机挑 K 只
MC_DRAWS   = 100                # 蒙特卡洛模拟次数


# ═══════════════════════════════════════════════════════════════════════════
# Gate 信号（RSRS / Breadth / CSVC 保留）
# ═══════════════════════════════════════════════════════════════════════════

def _ols_beta(high: pl.Series, low: pl.Series) -> float:
    x, y = low.to_numpy(), high.to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 5 else np.nan


def build_rsrs_gate() -> pl.DataFrame:
    sh = get_sh_index_daily()
    sh = sh.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")).sort("date")
    n, m = 18, 600
    betas = [None] * len(sh)
    for i in range(n - 1, len(sh)):
        betas[i] = _ols_beta(sh.slice(i - n + 1, n)["high"], sh.slice(i - n + 1, n)["low"])
    sh = sh.with_columns(pl.Series("_beta", betas))
    z = (sh["_beta"] - sh["_beta"].rolling_mean(m).shift(1)) / sh["_beta"].rolling_std(m).shift(1)
    return sh.with_columns(
        pl.when(z > 0.7).then(1).when(z < -0.7).then(0).otherwise(None)
        .forward_fill().fill_null(1).cast(pl.Int8).alias("rsrs")
    ).select(["date", "rsrs"]).with_columns(pl.lit("RSRS").alias("gate_name"))


def build_breadth_gate(market: pl.DataFrame) -> pl.DataFrame:
    return (
        market.sort(["code", "date"])
        .with_columns(pl.col("close_adj").rolling_mean(20).over("code").alias("_ma"))
        .group_by("date").agg((pl.col("close_adj") > pl.col("_ma")).mean().alias("pct"))
        .sort("date")
        .with_columns(pl.when(pl.col("pct") > 0.70).then(1).when(pl.col("pct") < 0.30).then(0)
                       .otherwise(None).forward_fill().fill_null(1).cast(pl.Int8).alias("breadth"))
        .select(["date", "breadth"]).with_columns(pl.lit("Breadth").alias("gate_name"))
    )


def build_csvc_gate(market: pl.DataFrame) -> pl.DataFrame:
    n = 120
    daily = (market.sort(["code", "date"])
        .with_columns((pl.col("close_adj") / pl.col("close_adj").shift(1).over("code") - 1).alias("_r"))
        .group_by("date").agg([pl.col("turnover").mean().alias("tov"), pl.col("_r").std().alias("std")])
        .sort("date").with_columns((pl.col("tov") / pl.col("std")).alias("_csvc")))
    z = (daily["_csvc"] - daily["_csvc"].rolling_mean(n)) / daily["_csvc"].rolling_std(n)
    return daily.with_columns((z > 0).cast(pl.Int8).alias("csvc")).select(["date", "csvc"]).with_columns(pl.lit("CSVC").alias("gate_name"))


# ═══════════════════════════════════════════════════════════════════════════
# 核心评估：按 gate 分组统计持有期回报
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gates(market: pl.DataFrame) -> None:
    # 构建所有前向收益列
    fwd_exprs = [
        (pl.col("close_adj").shift(-d).over("code") / pl.col("close_adj") - 1).alias(f"fwd_{d}d")
        for d in FWD_DAYS
    ]
    market = market.sort(["code", "date"]).with_columns(fwd_exprs)
    max_fwd = max(FWD_DAYS)
    market = market.with_columns(pl.col("date").dt.year().alias("year"))

    # AMV gate + 其他三个 gate
    amv     = market.group_by("date").agg(pl.col("is_bull_regime").first().alias("amv")).sort("date")
    rsrs    = build_rsrs_gate()
    breadth = build_breadth_gate(market)
    csvc    = build_csvc_gate(market)

    gates = [
        ("amv", "AMV 活跃市值"),
        ("g_rsrs", "RSRS"),
        ("g_breadth", "Breadth"),
        ("g_csvc", "CSVC"),
    ]

    print(f"\n  MC draws: {MC_DRAWS}  |  TopK: {TOP_K}  |  Windows: {FWD_DAYS}")
    print(f"  {'='*80}")

    for fwd in FWD_DAYS:
        col_fwd = f"fwd_{fwd}d"
        valid = market.drop_nulls([col_fwd])

        # ── Monte Carlo: N 次随机抽 K 只 ──
        mc_on: dict[str, list[float]] = {}   # gate_name -> [mc_draw_means]
        mc_off: dict[str, list[float]] = {}

        for gate_col, _ in gates:
            mc_on[gate_col] = []
            mc_off[gate_col] = []

        for _ in range(MC_DRAWS):
            daily_mc = valid.group_by("date").agg(
                pl.col(col_fwd).sample(n=TOP_K, with_replacement=False, shuffle=True).mean().alias("mc_ret")
            ).sort("date")
            joined = daily_mc.join(amv, on="date", how="inner")
            joined = joined.join(rsrs.select(["date", pl.col("rsrs").alias("g_rsrs")]), on="date", how="left")
            joined = joined.join(breadth.select(["date", pl.col("breadth").alias("g_breadth")]), on="date", how="left")
            joined = joined.join(csvc.select(["date", pl.col("csvc").alias("g_csvc")]), on="date", how="left")
            for gate_col, _ in gates:
                on_m  = joined.filter(pl.col(gate_col) == 1)["mc_ret"].mean()
                off_m = joined.filter(pl.col(gate_col) == 0)["mc_ret"].mean()
                mc_on[gate_col].append(on_m if on_m is not None else 0.0)
                mc_off[gate_col].append(off_m if off_m is not None else 0.0)

        # ── 输出 ──
        print(f"\n  {'─'*80}")
        print(f"  持有 {fwd} 天")
        print(f"  {'Gate':<14} {'开仓mean':>9} {'开仓MC±std':>13} {'关仓mean':>9} {'Δ':>9} {'判断':>6}")
        print(f"  {'-'*60}")
        for gate_col, name in gates:
            on_arr  = np.array(mc_on[gate_col])
            off_arr = np.array(mc_off[gate_col])
            r_on    = float(on_arr.mean())
            r_off   = float(off_arr.mean())
            r_on_s  = float(on_arr.std())
            diff    = r_on - r_off
            judge   = "✅" if diff > 0.005 else "❌" if diff < -0.005 else "≈"
            print(f"  {name:<14} {r_on:>+8.4%}  {r_on:>+8.4%}±{r_on_s:.4%}  {r_off:>+8.4%}  {diff:>+8.4%}  {judge:>4}")

        # ── AMV 分年 ──
        print(f"\n  AMV 分年 (持有 {fwd} 天):")
        print(f"  {'Year':>6}  {'开':>5} {'开仓ret':>9} {'关':>5} {'关仓ret':>9}  {'Δ':>9}")
        print(f"  {'-'*46}")
        for yr in sorted(valid["year"].unique().drop_nulls().cast(int).to_list()):
            yv = valid.filter(pl.col("year") == yr)
            # MC: N draws, each draw's yearly mean, then average over draws
            yr_on_vals = []
            yr_off_vals = []
            for _ in range(MC_DRAWS):
                d = yv.group_by("date").agg(
                    pl.col(col_fwd).sample(n=TOP_K, with_replacement=False, shuffle=True).mean().alias("r")
                ).join(amv, on="date", how="inner")
                yr_on_vals.append(d.filter(pl.col("amv") == 1)["r"].mean() or 0.0)
                yr_off_vals.append(d.filter(pl.col("amv") == 0)["r"].mean() or 0.0)
            ro = float(np.mean(yr_on_vals))
            rf = float(np.mean(yr_off_vals))
            no = yv.filter(pl.col("is_bull_regime")).select("date").unique().height
            nf = yv.filter(~pl.col("is_bull_regime")).select("date").unique().height
            print(f"  {yr:>6}  {no:>5} {ro:>+8.4%} {nf:>5} {rf:>+8.4%}  {ro-rf:>+8.4%}")


# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    t0 = time.perf_counter()
    logger.info("加载数据...")
    cfg = MarketConfig(
        data_source=resolve_data_source(),
        start_date=START_DATE, end_date=END_DATE, st_snapshot_date="2026-03-31",
    )
    reader, lf = build_market_lazy(cfg)
    market = lf.collect(); reader.close()
    logger.info(f"market: {market.height:,} rows, {market['code'].n_unique():,} codes, {time.perf_counter()-t0:.1f}s")

    evaluate_gates(market)
    logger.info(f"耗时: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
