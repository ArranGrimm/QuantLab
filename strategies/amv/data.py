"""AMV 市场数据层。

Reader 生命周期由调用方管理——build_market_lazy 返回 reader + LazyFrame，
调用方持 reader 直到 collect 完成，然后自行 close。
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from utils import get_st_blacklist_pl
from utils.data_source import (
    DailyMarketReader,
    DataSourceSettings,
    daily_reader,
    open_daily_reader,
)
from utils.active_market_value_regime import build_active_market_value_regime_frame

BASE_OHLC_COLS = [
    "code", "date",
    "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    "open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw",
    "market_cap_100m", "amount", "volume",
    "turnover", "circulating_capital",
]


@dataclass(frozen=True)
class MarketConfig:
    """构造 AMV 市场 LazyFrame 所需的不可变配置。"""
    data_source: DataSourceSettings
    start_date: str
    end_date: str
    st_snapshot_date: str
    bull_trigger_pct: float = 4.0
    bull_lookback_days: int = 2
    bear_trigger_1d_pct: float = -2.3
    effective_lag_days: int = 1


def build_market_lazy(config: MarketConfig) -> tuple[DailyMarketReader, pl.LazyFrame]:
    """构造 AMV 市场 LazyFrame，返回 (reader, lf)。

    reader 由调用方管理——必须在 collect 后 close。
    不 collect，不 .collect().lazy()——直接返回 DuckDB-backed LazyFrame。
    """
    reader = open_daily_reader(config.data_source)

    st_list = get_st_blacklist_pl(config.st_snapshot_date)

    q_raw = (
        reader.load_daily_full()
        .filter(pl.col("date") >= pl.lit(config.start_date).str.strptime(pl.Date, "%Y-%m-%d"))
        .filter(pl.col("date") <= pl.lit(config.end_date).str.strptime(pl.Date, "%Y-%m-%d"))
        .filter(~pl.col("code").is_in(st_list))
        .sort(["code", "date"])
    )
    available_base = [c for c in BASE_OHLC_COLS if c in q_raw.collect_schema().names()]
    lf = (
        q_raw.with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
        .select([*available_base, "amount_ma20"])
    )

    df_regime = build_active_market_value_regime_frame(
        bull_trigger_pct=config.bull_trigger_pct,
        bull_lookback_days=config.bull_lookback_days,
        bear_trigger_1d_pct=config.bear_trigger_1d_pct,
        effective_lag_days=config.effective_lag_days,
        date_col="date",
    ).select(["date", "is_bull_regime", "amv_mechanical_regime"]).lazy()

    lf = (
        lf.join(df_regime, on="date", how="left")
        .with_columns(
            pl.col("is_bull_regime").fill_null(False),
            pl.col("amv_mechanical_regime").fill_null("unknown"),
        )
        .sort(["date", "code"])
    )

    return reader, lf


def resolve_end_date(data_source: DataSourceSettings) -> str:
    """短暂开连接获取最新交易日。"""
    with daily_reader(data_source) as reader:
        return reader.resolve_end_date()


def build_market_frame(
    config: MarketConfig,
    *,
    required_factors: list[str] | None = None,
) -> pl.DataFrame:
    """Compat wrapper: build and collect the full AMV market frame in one call.

    Opens/closes the reader internally. Prefer build_market_lazy + manual
    collect for new code.
    """
    from strategies.amv.factors.base import build_amv_base_factors

    reader, lf = build_market_lazy(config)
    try:
        result = build_amv_base_factors(lf, required_factors).collect(streaming=True)
    finally:
        reader.close()
    return result
