"""
Fast Factor IC Analysis — Polars-native Spearman rank correlation

替代原来的 Python 双层循环 (factors x dates x scipy.spearmanr)，
使用 Polars group_by + pl.corr 一次性算完所有因子的逐日 IC。
"""
import polars as pl
import numpy as np
from typing import Optional


def calc_factor_ic(
    df: pl.DataFrame,
    factor_cols: list[str],
    label: str = "fwd_mfe_10d",
    min_samples: int = 30,
    prefix_highlight: str = "",
) -> dict:
    """
    计算所有因子的 Spearman 截面 IC (vs label)，并打印排行榜。

    Args:
        df: DataFrame, 必须包含 "date" 列 + factor_cols + label 列
        factor_cols: 因子列名列表
        label: 标签列名
        min_samples: 每日最少样本数 (不足的天数跳过)
        prefix_highlight: 因子名前缀高亮标记 (如 "b1_", "rk_")

    Returns:
        dict: {factor_name: {"ic_mean", "ic_std", "icir", "t_stat", "n_days"}}
    """
    print(f"📊 计算因子 IC (Polars Spearman vs {label})...")

    available = [f for f in factor_cols if f in df.columns]
    if len(available) < len(factor_cols):
        missing = set(factor_cols) - set(available)
        print(f"   ⚠️ 跳过缺失因子: {missing}")

    df_valid = df.filter(
        pl.col(label).is_not_null() & pl.col(label).is_not_nan()
    )

    # 过滤日样本数不足的天数
    date_counts = df_valid.group_by("date").agg(pl.len().alias("n"))
    valid_dates = date_counts.filter(pl.col("n") >= min_samples)["date"]
    df_valid = df_valid.filter(pl.col("date").is_in(valid_dates))

    n_dates = df_valid["date"].n_unique()
    print(f"   有效天数: {n_dates}, 样本: {df_valid.height:,}")

    # Polars 原生 Spearman IC: group_by(date) + pl.corr
    ic_df = (
        df_valid
        .group_by("date")
        .agg([
            pl.corr(f, label, method="spearman").alias(f)
            for f in available
        ])
        .sort("date")
    )

    ic_results = {}
    for f in available:
        ic_series = ic_df[f].drop_nulls().drop_nans()
        if len(ic_series) < 20:
            continue
        arr = ic_series.to_numpy()
        ic_mean = arr.mean()
        ic_std = arr.std()
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0
        t = ic_mean / (ic_std / np.sqrt(len(arr))) if ic_std > 1e-8 else 0
        ic_results[f] = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "t_stat": t,
            "n_days": len(arr),
        }

    # 打印排行榜
    sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]["icir"]), reverse=True)

    print(f"\n{'因子':<25} {'IC_Mean':>8} {'IC_Std':>8} {'ICIR':>8} {'t-stat':>8} {'显著':>4}")
    print("-" * 75)
    for name, r in sorted_factors:
        sig = "✅" if abs(r["t_stat"]) > 2 else ""
        prefix = "🔵" if (prefix_highlight and name.startswith(prefix_highlight)) else "  "
        print(f"{prefix}{name:<23} {r['ic_mean']:>+8.4f} {r['ic_std']:>8.4f} "
              f"{r['icir']:>+8.4f} {r['t_stat']:>+8.2f} {sig:>4}")
    print("-" * 75)

    n_sig = sum(1 for _, r in sorted_factors if abs(r["t_stat"]) > 2)
    n_total = len(sorted_factors)
    if prefix_highlight:
        n_hl_sig = sum(1 for n, r in sorted_factors if n.startswith(prefix_highlight) and abs(r["t_stat"]) > 2)
        n_hl = sum(1 for n, _ in sorted_factors if n.startswith(prefix_highlight))
        n_other_sig = n_sig - n_hl_sig
        n_other = n_total - n_hl
        print(f"\n   {prefix_highlight}* 因子显著: {n_hl_sig}/{n_hl}")
        print(f"   通用因子显著: {n_other_sig}/{n_other}")
    else:
        print(f"\n   显著因子: {n_sig} / {n_total}")

    return ic_results


def select_factors_by_ic(
    ic_results: dict,
    t_threshold: float = 1.5,
) -> list[str]:
    """
    根据 IC t-stat 筛选因子。

    Args:
        ic_results: calc_factor_ic 返回的字典
        t_threshold: |t-stat| 阈值

    Returns:
        筛选后的因子名列表 (按 |ICIR| 降序)
    """
    selected = [
        name for name, r in sorted(
            ic_results.items(),
            key=lambda x: abs(x[1]["icir"]),
            reverse=True,
        )
        if abs(r["t_stat"]) > t_threshold
    ]
    print(f"   自动入选 (|t|>{t_threshold}): {len(selected)} 个")
    return selected
