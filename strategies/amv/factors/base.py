from __future__ import annotations

import polars as pl

from strategies.amv.specs import FactorSpec
from utils.alpha158_factors import calc_alpha158_factors, resolve_alpha158_group_config
from utils.rotation_factors import calc_rotation_factors


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


def build_amv_base_factors(frame: pl.LazyFrame) -> pl.LazyFrame:
    """计算 AMV 主线候选池需要的基础量价因子。"""

    return calc_alpha158_factors(
        calc_rotation_factors(frame),
        use_kbar=True,
        price_fields=(),
        include=(),
    )
