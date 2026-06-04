from __future__ import annotations

import argparse

import polars as pl

from utils import get_st_blacklist_pl
from utils.data_source import daily_reader
from utils.active_market_value_regime import build_active_market_value_regime_frame
BASE_OHLC_COLS = [
    "code", "date",
    "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    "open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw",
    "market_cap_100m", "amount", "volume",
    "turnover", "circulating_capital",
]


def build_market_raw(args: argparse.Namespace) -> pl.LazyFrame:
    """构造 AMV 基础市场 LazyFrame——只含 OHLC + regime，不含因子。不 collect。"""

    st_blacklist = get_st_blacklist_pl(args.st_snapshot_date)
    st_blacklist_df = pl.DataFrame(
        {"code": st_blacklist},
        schema={"code": pl.Utf8},
    ).lazy()

    with daily_reader(args.data_source) as reader:
        q_raw = (
            reader.load_daily_full()
            .filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .join(st_blacklist_df, on="code", how="anti")
            .sort(["code", "date"])
        )
        available_base = [c for c in BASE_OHLC_COLS if c in q_raw.collect_schema().names()]
        lf = (
            q_raw.with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
            .select([*available_base, "amount_ma20"])
            .collect(streaming=True)
            .lazy()
        )

    df_regime = build_active_market_value_regime_frame(
        bull_trigger_pct=args.amv_bull_trigger_pct,
        bull_lookback_days=args.amv_bull_lookback_days,
        bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
        effective_lag_days=args.amv_effective_lag_days,
        date_col="date",
    ).select(["date", "is_bull_regime", "amv_mechanical_regime"]).lazy()

    return (
        lf.join(df_regime, on="date", how="left")
        .with_columns([
            pl.col("is_bull_regime").fill_null(False),
            pl.col("amv_mechanical_regime").fill_null("unknown"),
        ])
        .sort(["date", "code"])
    )


def build_market_frame(args: argparse.Namespace, required_factors: list[str] | None = None) -> pl.DataFrame:
    """构造 AMV 主线市场表（兼容旧调用）。required_factors 为 None 时计算全部因子。"""
    from strategies.amv.factors.base import build_amv_base_factors

    raw = build_market_raw(args)
    result = build_amv_base_factors(raw, required_factors).collect(streaming=True)
    return result
