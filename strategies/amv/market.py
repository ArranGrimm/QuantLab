from __future__ import annotations

import argparse

import duckdb
import polars as pl

from utils import get_st_blacklist_pl, load_daily_data_full
from utils.active_market_value_regime import build_active_market_value_regime_frame
from strategies.amv.factors.base import AMV_KBAR_COLS, build_amv_base_factors


def build_market_frame(args: argparse.Namespace) -> pl.DataFrame:
    """构造 AMV 主线候选池所需的市场特征表。"""

    conn = duckdb.connect(str(args.qmt_db), read_only=True)
    try:
        st_blacklist = get_st_blacklist_pl(args.st_snapshot_date)
        st_blacklist_df = pl.DataFrame(
            {"code": st_blacklist},
            schema={"code": pl.Utf8},
        ).lazy()

        q_full = (
            load_daily_data_full(conn, db_source=getattr(args, "db_source", "qmt"), tdx_db=getattr(args, "tdx_db", ""))
            .filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .join(st_blacklist_df, on="code", how="anti")
            .sort(["code", "date"])
        )

        q_factor = build_amv_base_factors(q_full)

        keep_cols = [
            "date",
            "code",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "pre_close_adj",
            "open_raw",
            "high_raw",
            "low_raw",
            "close_raw",
            "pre_close_raw",
            "market_cap_100m",
            "amount",
            "ret_5d",
            "ret_20d",
            "price_pos_20d",
            "close_to_high_20d",
            "ma_bias_20",
            "disp_bias_20",
            "atr_14_pct",
            "panic_vol_ratio_20d",
            "intraday_pos",
            *AMV_KBAR_COLS,
        ]

        df = (
            q_factor.with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
            .select([*keep_cols, "amount_ma20"])
            .collect()
        )

        df_regime = build_active_market_value_regime_frame(
            bull_trigger_pct=args.amv_bull_trigger_pct,
            bull_lookback_days=args.amv_bull_lookback_days,
            bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
            effective_lag_days=args.amv_effective_lag_days,
            date_col="date",
        ).select(["date", "is_bull_regime", "amv_mechanical_regime"])

        return (
            df.join(df_regime, on="date", how="left")
            .with_columns(
                [
                    pl.col("is_bull_regime").fill_null(False),
                    pl.col("amv_mechanical_regime").fill_null("unknown"),
                ]
            )
            .sort(["date", "code"])
        )
    finally:
        conn.close()

