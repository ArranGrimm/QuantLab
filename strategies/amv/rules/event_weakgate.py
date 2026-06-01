"""
首板后回踩事件弱窗口风控规则。

设计思路
--------
涨停生态是 A 股上下文因子路线的第三阶段。首板后回踩事件有独立 alpha，但回撤极重（base MaxDD > 45%）。
弱窗口 gate 的目标：在不牺牲太多收益的前提下压缩 extreme drawdown。

验证历程 (2026-05-29):
- base 5td: total +130.95%, MaxDD 45.38%
- weakgate 5td: total +155.04%, MaxDD 34.12%
- weaktop1/weaktier: MaxDD > 50%, 已否决
- 当前判断: weakgate 是防守上限，尚未 allocation-ready

规则定义
--------
弱窗口评分由 7 个子评分加总，每项 0~1，阈值基于历史样本分布经验：
1. limit_low: 全市场涨停数 rank_pct < 0.55 → 短线情绪降温
2. pool_thin: 候选数 < 12 → 事件稀缺，信号质量下降
3. score_low: Top3 均分 < 8.0 → 事件吸引力不足
4. atr_high: Top3 ATR rank > 0.75 → 参与标的风险偏大
5. stale_high: Top3 陈旧事件占比高 → 新鲜事件稀缺
6. reclaim_low: Top3 修复型占比 < 0.45 → 弱市修复动力不足
7. amv_flat: AMV 5 日涨速 < 2.0% → 活跃市值趋势不支持

当日 _weak_window_score >= 3.0 时，Top3 不允许开仓。

适用策略
--------
known_compatible: ["event-firstboard", "event-firstboard-base"]
known_incompatible: ["trend-p2", "trend-p3", "trend-p3-enhanced", "pullback-pb3"]
"""

from __future__ import annotations

from typing import Any

import polars as pl

from strategies.amv.regime import build_amv_phase_frame
from strategies.amv.scoring import finite_expr


def bool_score(col_name: str, weight: float) -> pl.Expr:
    return pl.when(pl.col(col_name).fill_null(False)).then(weight).otherwise(0.0)


def first_board_event_score_expr() -> pl.Expr:
    """首板后回踩事件的评分公式。

    核心逻辑：事件质量越高（反包/昨日涨停/修复）分越高，
    连板数越少（首板优先）分越高，时间越近分越高。
    被 workflows._apply_event_ranker 和 build_event_weakgate_signal 使用。
    """
    base_score = (
        bool_score("is_reboard_after_pullback", 5.0)  # 最强：回踩后再反包
        + bool_score("was_limit_up_yesterday", 4.0)   # 昨日涨停（热度还在）
        + bool_score("is_reclaim_after_limit", 3.0)    # 涨停后修复承接
        + bool_score("has_limit_up_5d", 2.0)           # 近 5 日有涨停
        + bool_score("has_limit_up_10d", 1.5)          # 近 10 日有涨停
        + bool_score("has_limit_up_20d", 1.0)          # 近 20 日有涨停（弱信号）
        + bool_score("has_one_word_limit_up_10d", 0.5) # 一字板（参考价值低）
        - bool_score("has_failed_limit_up_5d", 1.0)    # 近 5 日炸板（惩罚）
    )
    recency_bonus = (
        pl.when(pl.col("days_since_prior_limit_up").is_between(1, 3))
        .then(2.0)
        .when(pl.col("days_since_prior_limit_up").is_between(4, 10))
        .then(1.0)
        .otherwise(0.0)
    )
    first_board_bonus = (
        pl.when(pl.col("_last_lu_streak_before") == 1).then(1.0)
        .when(pl.col("_last_lu_streak_before") == 2).then(-1.0)
        .otherwise(0.0)
    )
    liquidity_bonus = pl.when(pl.col("amount_ratio_5_20").fill_null(1.0) <= 1.20).then(0.5).otherwise(0.0)
    risk_penalty = (
        pl.col("atr_14_pct_rank_pct").fill_null(0.5) + pl.col("panic_vol_ratio_20d_rank_pct").fill_null(0.5)
    ) / 2.0
    return base_score + recency_bonus + first_board_bonus + liquidity_bonus - risk_penalty


def add_weak_window_context(scored: pl.DataFrame, config: Any) -> pl.DataFrame:
    """构建 7 维弱窗口上下文评分，加总为 _weak_window_score。"""
    # 全市场涨停数 → rank 越低，情绪越冷
    market_breadth = (
        scored.group_by("date")
        .agg(pl.col("is_close_limit_up").fill_null(False).sum().alias("_weak_limit_up_count"))
        .sort("date")
        .with_columns((pl.col("_weak_limit_up_count").rank("average") / pl.len()).alias("_weak_limit_up_count_rank_pct"))
    )

    candidates = scored.filter(pl.col("_is_signal_candidate"))
    top3 = candidates.filter(pl.col("_base_signal_rank") <= config.top_n)
    candidate_health = (
        candidates.group_by("date")
        .agg(
            [
                pl.len().alias("_weak_candidate_count"),
                pl.col("_base_signal_score").mean().alias("_weak_candidate_avg_score"),
            ]
        )
        .join(
            top3.group_by("date").agg(
                [
                    pl.col("_base_signal_score").mean().alias("_weak_top3_avg_score"),
                    pl.col("atr_14_pct_rank_pct").mean().alias("_weak_top3_avg_atr_rank"),
                    (pl.col("days_since_prior_limit_up") >= 7).mean().alias("_weak_top3_stale_share"),
                    pl.col("is_reclaim_after_limit").mean().alias("_weak_top3_reclaim_share"),
                ]
            ),
            on="date", how="left",
        )
    )

    amv_phase = build_amv_phase_frame(
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        effective_lag_days=config.amv_effective_lag_days,
    ).select(
        [
            "date",
            pl.col("amv_slope_5d").alias("_weak_amv_slope_5d"),
            pl.col("amv_dd_from_high").alias("_weak_amv_dd_from_high"),
            pl.col("amv_neg_streak").alias("_weak_amv_neg_streak"),
        ]
    )

    enriched = (
        scored.join(market_breadth, on="date", how="left")
        .join(candidate_health, on="date", how="left")
        .join(amv_phase, on="date", how="left")
    )

    # 子评分 1: 全市场涨停数偏低 → rank_pct < 0.55 开始惩罚
    limit_low = ((0.55 - pl.col("_weak_limit_up_count_rank_pct").fill_null(0.55)) / 0.55).clip(0, 1)
    # 子评分 2: 候选池太薄 → 候选数 < 12 开始惩罚
    pool_thin = ((12.0 - pl.col("_weak_candidate_count").fill_null(12).cast(pl.Float64)) / 8.0).clip(0, 1)
    # 子评分 3: Top3 平均分偏低 → 均分 < 8.0 开始惩罚
    score_low = ((8.0 - pl.col("_weak_top3_avg_score").fill_null(8.0)) / 3.0).clip(0, 1)
    # 子评分 4: Top3 波动率偏高 → ATR rank > 0.75 开始惩罚
    atr_high = ((pl.col("_weak_top3_avg_atr_rank").fill_null(0.75) - 0.75) / 0.20).clip(0, 1)
    # 子评分 5: Top3 陈旧事件占比高 → 直接使用占比
    stale_high = pl.col("_weak_top3_stale_share").fill_null(0.0).clip(0, 1)
    # 子评分 6: Top3 修复型占比低 → reclaim < 0.45 开始惩罚
    reclaim_low = ((0.45 - pl.col("_weak_top3_reclaim_share").fill_null(0.45)) / 0.45).clip(0, 1)
    # 子评分 7: AMV 5 日涨速偏平/偏弱 → slope < 2.0% 开始惩罚
    amv_flat = ((2.0 - pl.col("_weak_amv_slope_5d").fill_null(2.0)) / 4.0).clip(0, 1)

    return enriched.with_columns(
        [
            limit_low.alias("_weak_limit_low_score"),
            pool_thin.alias("_weak_pool_thin_score"),
            score_low.alias("_weak_score_low_score"),
            atr_high.alias("_weak_atr_high_score"),
            stale_high.alias("_weak_stale_high_score"),
            reclaim_low.alias("_weak_reclaim_low_score"),
            amv_flat.alias("_weak_amv_flat_score"),
        ]
    ).with_columns(
        (
            pl.col("_weak_limit_low_score")
            + pl.col("_weak_pool_thin_score")
            + pl.col("_weak_score_low_score")
            + pl.col("_weak_atr_high_score")
            + pl.col("_weak_stale_high_score")
            + pl.col("_weak_reclaim_low_score")
            + pl.col("_weak_amv_flat_score")
        ).alias("_weak_window_score")
    )


def build_event_weakgate_signal(
    *,
    market: pl.DataFrame,
    sleeve_id: str,
    config: Any,
    skip_gate: bool = False,
) -> pl.DataFrame:
    """首板回踩评分 + 可选弱窗口过滤，返回 signal_rows。"""
    event_candidate = pl.col("is_first_board_pullback_setup").fill_null(False)
    valid_expr = finite_expr("price_pos_20d") & finite_expr("amount_ma20")
    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= config.mv_min)
        & (pl.col("amount_ma20") >= config.amount_ma20_min)
        & valid_expr
        & event_candidate
    )
    scored = market.with_columns(
        [
            candidate_expr.alias("_is_signal_candidate"),
            pl.when(candidate_expr).then(first_board_event_score_expr()).otherwise(None).alias("_base_signal_score"),
        ]
    ).with_columns(
        pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_signal_rank")
    )

    if skip_gate:
        # Base: no weak window context, pure event scoring
        scored = scored.with_columns(pl.col("_base_signal_score").alias("_signal_score"))
    else:
        scored = add_weak_window_context(scored, config)
        scored = scored.with_columns(pl.col("_base_signal_score").alias("_signal_score"))

    scored = scored.with_columns(
        pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank")
    )

    if skip_gate:
        select_expr = pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= config.top_n)
    else:
        weak_score = pl.col("_weak_window_score").fill_null(0.0)
        select_expr = pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= config.top_n) & (weak_score < 3.0)

    return (
        scored.filter(select_expr)
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )
