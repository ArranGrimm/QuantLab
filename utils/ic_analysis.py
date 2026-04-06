"""
Factor Analysis Toolkit — IC analysis + correlation matrix + pruning

- calc_factor_ic:        逐日截面 Spearman IC 分析
- select_factors_by_ic:  按 t-stat 筛选因子
- calc_factor_corr:      因子间 Spearman/Pearson 相关矩阵
- find_redundant_factors: 基于相关性 + ICIR 的冗余因子剪枝
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


# ═══════════════════════════════════════════════════════════════════════
# 因子相关性矩阵 + 冗余剪枝
# ═══════════════════════════════════════════════════════════════════════


def calc_factor_corr(
    df: pl.DataFrame,
    factor_cols: list[str],
    method: str = "spearman",
    sample_n: Optional[int] = 500_000,
) -> np.ndarray:
    """
    计算因子间两两相关矩阵 (Polars 原生, 一次 select 完成)。

    Args:
        df: DataFrame, 必须包含 factor_cols 列
        factor_cols: 因子列名列表
        method: "spearman" 或 "pearson"
        sample_n: 随机采样行数 (加速, None=全量)

    Returns:
        (corr_matrix, factor_names)
        corr_matrix: np.ndarray shape (n, n), 对称, 对角线为 1.0
        factor_names: 实际计算的因子名列表
    """
    available = [f for f in factor_cols if f in df.columns]
    n = len(available)
    print(f"📊 计算因子相关矩阵 ({method}, {n} 个因子)...")

    df_work = df.select(available).drop_nulls()
    if sample_n and df_work.height > sample_n:
        df_work = df_work.sample(n=sample_n, seed=42)
        print(f"   采样 {sample_n:,} 行 (原始 {df.height:,})")

    corr_exprs = []
    pair_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            corr_exprs.append(
                pl.corr(available[i], available[j], method=method)
                .alias(f"c_{i}_{j}")
            )
            pair_idx.append((i, j))

    result = df_work.select(corr_exprs)

    mat = np.eye(n)
    for (i, j), col_name in zip(pair_idx, result.columns):
        val = result[col_name].item()
        if val is None or np.isnan(val):
            val = 0.0
        mat[i, j] = val
        mat[j, i] = val

    return mat, available


def print_corr_clusters(
    corr_matrix: np.ndarray,
    factor_names: list[str],
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """
    打印高相关因子对, 按 |corr| 降序排列。

    Returns:
        [(factor_a, factor_b, corr), ...] 所有 |corr| > threshold 的对
    """
    n = len(factor_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_matrix[i, j]
            if abs(c) > threshold:
                pairs.append((factor_names[i], factor_names[j], c))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n{'='*65}")
    print(f" 高相关因子对 (|corr| > {threshold}): {len(pairs)} 对")
    print(f"{'='*65}")
    print(f" {'因子A':<22} {'因子B':<22} {'Corr':>8}")
    print(f" {'-'*22} {'-'*22} {'-'*8}")
    for a, b, c in pairs:
        print(f" {a:<22} {b:<22} {c:>+8.4f}")
    print(f"{'='*65}")

    return pairs


def find_redundant_factors(
    corr_matrix: np.ndarray,
    factor_names: list[str],
    ic_results: Optional[dict] = None,
    threshold: float = 0.85,
) -> tuple[list[str], list[str], list[tuple[str, str, float, str]]]:
    """
    基于相关性的贪心剪枝: 从最高相关对开始, 逐对淘汰 |ICIR| 较低的因子。

    Args:
        corr_matrix: calc_factor_corr 返回的相关矩阵
        factor_names: 因子名列表
        ic_results: calc_factor_ic 返回的 IC 字典 (可选, 无则按因子序号保留靠前的)
        threshold: |corr| 剪枝阈值

    Returns:
        (keep_list, drop_list, decisions)
        keep_list: 保留的因子列表
        drop_list: 淘汰的因子列表
        decisions: [(kept, dropped, corr, reason), ...]
    """
    n = len(factor_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr_matrix[i, j])
            if c > threshold:
                pairs.append((i, j, c))
    pairs.sort(key=lambda x: x[2], reverse=True)

    def get_icir(name: str) -> float:
        if ic_results and name in ic_results:
            return abs(ic_results[name]["icir"])
        return 0.0

    dropped = set()
    decisions = []

    for i, j, c in pairs:
        fa, fb = factor_names[i], factor_names[j]
        if fa in dropped or fb in dropped:
            continue
        icir_a, icir_b = get_icir(fa), get_icir(fb)
        if icir_a >= icir_b:
            dropped.add(fb)
            reason = f"|ICIR| {icir_a:.4f} > {icir_b:.4f}" if ic_results else "保留靠前"
            decisions.append((fa, fb, corr_matrix[i, j], reason))
        else:
            dropped.add(fa)
            reason = f"|ICIR| {icir_b:.4f} > {icir_a:.4f}"
            decisions.append((fb, fa, corr_matrix[i, j], reason))

    keep = [f for f in factor_names if f not in dropped]

    print(f"\n{'='*75}")
    print(f" 相关性剪枝 (|corr| > {threshold}): 淘汰 {len(dropped)}, 保留 {len(keep)}")
    print(f"{'='*75}")
    print(f" {'保留':<22} {'淘汰':<22} {'Corr':>8}  {'依据'}")
    print(f" {'-'*22} {'-'*22} {'-'*8}  {'-'*20}")
    for kept, drop, c, reason in decisions:
        print(f" {kept:<22} {drop:<22} {c:>+8.4f}  {reason}")
    print(f"{'='*75}")
    print(f" 保留 ({len(keep)}): {keep}")

    return keep, list(dropped), decisions
