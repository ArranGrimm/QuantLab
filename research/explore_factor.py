"""轻量因子探索脚本 — 改 FACTOR_TAG + make_factor_expr → uv run → 看 Rank IC + 分组收益。

自动记录到 research/factor_ledger.jsonl，避免重复探索。

用法: uv run python research/explore_factor.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER_PATH = Path(__file__).resolve().parent / "factor_ledger.jsonl"

from factors.registry import FACTOR_REGISTRY, compute_required_factors
from utils.data_source import TdxDailyReader, DataSourceSettings

# ═══════════════════════════════════════════════════════════════════════════
# 配置 — 改这里
# ═══════════════════════════════════════════════════════════════════════════

START_DATE = "2019-01-01"
END_DATE   = "2026-06-03"
FORWARD    = 5          # 前向收益天数
FACTOR_TAG = "cgo_100d"

# -------- 因子定义 --------
# 优先从 FACTOR_REGISTRY 读取 → 找不到才走 make_factor_expr()
# 切换因子只需改 FACTOR_TAG + make_factor_expr() 返回 None

def make_factor_expr() -> tuple[list[list[pl.Expr]], pl.Expr] | None:
    """返回 (steps, final_factor_expr)。返回 None = 使用 registry。"""

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 因子账本
# ═══════════════════════════════════════════════════════════════════════════

def _read_ledger() -> list[dict]:
    """读取历史因子实验结果。"""
    if not LEDGER_PATH.exists():
        return []
    entries = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _check_history(factor_tag: str) -> None:
    """检查当前因子是否已有历史记录，打印对比。"""
    entries = _read_ledger()
    matches = [e for e in entries if e.get("factor") == factor_tag]
    if not matches:
        return
    latest = matches[-1]  # 最新一条
    qr = latest.get("quintile_returns")
    spread = f"{qr[-1] - qr[0]:.4%}" if qr and len(qr) >= 5 else "n/a"
    print(f"\n  [ledger] 已有 {len(matches)} 条历史记录，最近一次 ({latest['ran_at']}):")
    print(f"    IC={latest['mean_ic']:.4f}, IR={latest['ic_ir']:.2f}, "
          f"spread={spread}")
    print(f"  本次将重新计算并追加新记录。\n")


def _write_ledger(
    factor_tag: str,
    forward: int,
    mean_ic: float,
    std_ic: float | None,
    ic_ir: float,
    pos_ratio: float,
    n_days: int,
    quintile_returns: list[float] | None,
) -> None:
    """追加一条实验结果到账本。"""
    entry = {
        "ran_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "factor": factor_tag,
        "forward_days": forward,
        "mean_ic": round(mean_ic, 6) if mean_ic is not None else None,
        "ic_std": round(std_ic, 6) if std_ic is not None else None,
        "ic_ir": round(ic_ir, 2),
        "positive_ratio": round(pos_ratio, 4),
        "n_days": n_days,
        "quintile_returns": [round(r, 6) for r in quintile_returns] if quintile_returns else None,
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"实验结果已记录到 {LEDGER_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# 加载 + 计算
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    _check_history(FACTOR_TAG)
    t0 = time.perf_counter()

    tdx_db = ROOT.parent / "QuantData" / "Ashare" / "tdx.db"
    ds = DataSourceSettings(provider="tdx", tdx_db=tdx_db, start_date=START_DATE)
    reader = TdxDailyReader(ds)

    factor_col = FACTOR_TAG  # registry factors use their own name

    lf = reader.load_daily_full().filter(
        pl.col("date").is_between(
            pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
            pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
        )
    )

    spec = FACTOR_REGISTRY.get(FACTOR_TAG)
    if spec:
        lf = compute_required_factors(lf, [FACTOR_TAG])
        print(f"  [registry] {FACTOR_TAG}: {spec['label']} (status={spec.get('status')})")
    else:
        # ── 实验因子：make_factor_expr() → 自定义 pipeline ──
        result = make_factor_expr()
        if result is None:
            print(f"  [skip] FACTOR_TAG={FACTOR_TAG!r} not in registry and make_factor_expr() returned None")
            return
        steps, final_expr = result
        for step in steps:
            lf = lf.with_columns(step)
        lf = lf.with_columns(final_expr.alias("factor"))
        factor_col = "factor"

    lf = lf.with_columns(
        (pl.col("close_adj").shift(-FORWARD).over("code")
         / pl.col("close_adj") - 1.0).alias(f"_fwd_{FORWARD}d"),
    )

    lf = lf.select(["code", "date", "close_adj", "market_cap_100m", "amount",
                    factor_col, f"_fwd_{FORWARD}d"])

    df = lf.collect()
    reader.close()
    n_stocks = df["code"].n_unique()

    logger.info(f"加载: {df.height:,} 行, {n_stocks:,} 只, {time.perf_counter() - t0:.1f}s")

    # ── Rank IC ──
    ic = (
        df.filter(pl.col(factor_col).is_not_null() & pl.col(factor_col).is_not_nan(),
                  pl.col(f"_fwd_{FORWARD}d").is_not_null())
        .group_by("date")
        .agg(
            pl.corr(pl.col(factor_col).rank("average"),
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
        pl.col(factor_col).is_not_null() & pl.col(factor_col).is_not_nan(),
        pl.col(f"_fwd_{FORWARD}d").is_not_null(),
    )
    if valid.height == 0:
        print("\n  无有效因子值，跳过分组分析\n")
        return

    grouped = (
        valid
        .with_columns(
            (pl.col(factor_col).rank("average").over("date")
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

    # ── 记录到因子账本 ──
    q_returns = [grouped.filter(pl.col("_q") == q)["avg_ret"][0] for q in range(1, 6)]
    _write_ledger(FACTOR_TAG, FORWARD, mean_ic, std_ic, ic_ir, pos_ratio, ic.height, q_returns)

    print(f"\n  耗时: {time.perf_counter() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
