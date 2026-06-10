from __future__ import annotations

import polars as pl

from strategies.amv.specs import FactorSpec
from utils.alpha158_factors import resolve_alpha158_group_config

_A_SHARE_LOT_SIZE = 100.0
_EPS = 1e-12


AMV_BASE_FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec("price_pos_20d", "20日价格位置", "higher", family="price"),
    FactorSpec("close_to_high_20d", "接近20日新高", "lower", family="price"),
    FactorSpec("KLEN", "K线振幅收缩", "lower", family="kbar"),
    FactorSpec("KMID2", "实体占比偏强", "higher", family="kbar"),
    FactorSpec("ret_5d", "5日动量", "higher", family="momentum"),
    FactorSpec("ret_20d", "20日动量", "higher", family="momentum"),
    FactorSpec("ma_bias_20", "20日均线偏离", "lower", family="pullback"),
    FactorSpec("disp_bias_20", "20日成本偏离", "lower", family="pullback"),
    FactorSpec("intraday_pos", "日内收盘位置", "higher", family="pullback"),
    FactorSpec("atr_14_pct", "ATR 风险", "lower", role="risk", family="risk"),
    FactorSpec("panic_vol_ratio_20d", "恐慌量比", "lower", role="risk", family="risk"),
)


AMV_KBAR_COLS = tuple(resolve_alpha158_group_config("kbar_shape")["factor_cols"])


def calc_amv_core_factors(frame: pl.LazyFrame) -> pl.LazyFrame:
    """AMV 主线用到的 rotation 子集（~11 列），避免全量 46 因子 + 大量中间列。"""

    schema_names = frame.collect_schema().names()
    if "turnover" in schema_names:
        turnover_rate_expr = (pl.col("turnover") * 100.0).fill_nan(0.0)
    else:
        turnover_rate_expr = (
            (pl.col("volume") * _A_SHARE_LOT_SIZE) / pl.col("circulating_capital").fill_null(1) * 100
        ).fill_nan(0.0)

    open_den = pl.max_horizontal(pl.col("open_adj"), pl.lit(_EPS))
    range_den = pl.col("high_adj") - pl.col("low_adj") + _EPS

    return (
        frame.sort(["code", "date"])
        .with_columns(pl.col("close_adj").shift(1).over("code").alias("_pc"))
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("_pc") - 1).alias("_ret"),
                pl.max_horizontal(
                    pl.col("high_adj") - pl.col("low_adj"),
                    (pl.col("high_adj") - pl.col("_pc")).abs(),
                    (pl.col("low_adj") - pl.col("_pc")).abs(),
                ).alias("_tr"),
                turnover_rate_expr.alias("turnover_rate"),
            ]
        )
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("close_adj").shift(5).over("code") - 1).alias("ret_5d"),
                (pl.col("close_adj") / pl.col("close_adj").shift(20).over("code") - 1).alias("ret_20d"),
                pl.when(pl.col("_ret") < 0)
                .then(pl.col("volume"))
                .otherwise(0.0)
                .rolling_sum(20)
                .over("code")
                .alias("_down_vol_sum_20"),
                pl.col("volume").rolling_sum(20).over("code").alias("_total_vol_sum_20"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("_down_vol_sum_20")
                    / pl.max_horizontal(pl.col("_total_vol_sum_20"), pl.lit(1.0))
                ).alias("panic_vol_ratio_20d"),
                pl.col("_tr").rolling_mean(14).over("code").alias("_atr14"),
                pl.col("close_adj").rolling_mean(20).over("code").alias("_ma20"),
                pl.col("high_adj").rolling_max(20).over("code").alias("_high_20d"),
                pl.col("close_adj").rolling_min(20).over("code").alias("_c_min_20"),
                pl.col("close_adj").rolling_max(20).over("code").alias("_c_max_20"),
            ]
        )
        .with_columns(
            [
                (pl.col("_atr14") / pl.max_horizontal(pl.col("close_adj"), pl.lit(0.01))).alias("atr_14_pct"),
                ((pl.col("close_adj") - pl.col("_ma20")) / pl.max_horizontal(pl.col("_ma20"), pl.lit(0.01)) * 100).alias(
                    "ma_bias_20"
                ),
                (1 - pl.col("close_adj") / pl.max_horizontal(pl.col("_high_20d"), pl.lit(0.01))).alias("close_to_high_20d"),
                (
                    (pl.col("close_adj") - pl.col("low_adj"))
                    / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8))
                ).alias("intraday_pos"),
                (
                    (pl.col("close_adj") - pl.col("_c_min_20"))
                    / pl.max_horizontal(pl.col("_c_max_20") - pl.col("_c_min_20"), pl.lit(0.01))
                ).alias("price_pos_20d"),
                ((pl.col("close_adj") + pl.col("high_adj") + pl.col("low_adj")) / 3 * pl.col("volume")).alias("_tp_v"),
            ]
        )
        .with_columns(
            [
                pl.col("_tp_v").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_pv_20"),
                pl.col("volume").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_v_20"),
            ]
        )
        .with_columns(
            (
                pl.col("close_adj")
                / pl.max_horizontal(pl.col("_ewm_pv_20") / pl.max_horizontal(pl.col("_ewm_v_20"), pl.lit(1e-10)), pl.lit(0.01))
                - 1
            ).alias("disp_bias_20")
        )
        .with_columns(
            [
                ((pl.col("close_adj") - pl.col("open_adj")) / open_den).alias("KMID"),
                ((pl.col("high_adj") - pl.col("low_adj")) / open_den).alias("KLEN"),
                ((pl.col("close_adj") - pl.col("open_adj")) / range_den).alias("KMID2"),
                (
                    (pl.col("high_adj") - pl.max_horizontal(pl.col("open_adj"), pl.col("close_adj"))) / open_den
                ).alias("KUP"),
                (
                    (pl.col("high_adj") - pl.max_horizontal(pl.col("open_adj"), pl.col("close_adj"))) / range_den
                ).alias("KUP2"),
                (
                    (pl.min_horizontal(pl.col("open_adj"), pl.col("close_adj")) - pl.col("low_adj")) / open_den
                ).alias("KLOW"),
                (
                    (pl.min_horizontal(pl.col("open_adj"), pl.col("close_adj")) - pl.col("low_adj")) / range_den
                ).alias("KLOW2"),
                ((2.0 * pl.col("close_adj") - pl.col("high_adj") - pl.col("low_adj")) / open_den).alias("KSFT"),
                ((2.0 * pl.col("close_adj") - pl.col("high_adj") - pl.col("low_adj")) / range_den).alias("KSFT2"),
            ]
        )
        .drop(
            [
                "_pc",
                "_ret",
                "_tr",
                "turnover_rate",
                "_down_vol_sum_20",
                "_total_vol_sum_20",
                "_atr14",
                "_ma20",
                "_high_20d",
                "_c_min_20",
                "_c_max_20",
                "_tp_v",
                "_ewm_pv_20",
                "_ewm_v_20",
            ]
        )
    )


def build_amv_base_factors(frame: pl.LazyFrame, required_factors: list[str] | None = None) -> pl.LazyFrame:
    """计算 AMV 主线候选池需要的基础量价因子。

    Args:
        frame: 市场数据 LazyFrame
        required_factors: 需要的因子名列表；None 时计算全部（兼容旧流程）
    """
    if required_factors is not None:
        from factors.registry import compute_required_factors
        return compute_required_factors(frame, required_factors)

    return calc_amv_core_factors(frame)
