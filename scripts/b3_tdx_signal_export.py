from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from scripts.amv_bull_pool_export_signals import (
    DEFAULT_QMT_DB,
    ROOT,
    _git_commit,
    _rel_path,
)
from utils import get_st_blacklist_pl, load_daily_data_full
from utils.active_market_value_regime import build_active_market_value_regime_frame


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "b3_tdx_signals"
PRICE_LIMIT_TOLERANCE = 0.001


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_20pct_board_expr() -> pl.Expr:
    return (
        pl.col("code").str.starts_with("sz.300")
        | pl.col("code").str.starts_with("sz.301")
        | pl.col("code").str.starts_with("sh.688")
        | pl.col("code").str.starts_with("sh.689")
        | pl.col("code").str.starts_with("300")
        | pl.col("code").str.starts_with("301")
        | pl.col("code").str.starts_with("688")
        | pl.col("code").str.starts_with("689")
    )


def price_limit_pct_expr() -> pl.Expr:
    return pl.when(is_20pct_board_expr()).then(0.20).otherwise(0.10)


def ref(col_name: str, periods: int = 1) -> pl.Expr:
    return pl.col(col_name).shift(periods).over("code")


def ma(col_name: str, window: int) -> pl.Expr:
    return pl.col(col_name).rolling_mean(window).over("code")


def hhv(col_name: str, window: int) -> pl.Expr:
    return pl.col(col_name).rolling_max(window).over("code")


def llv(col_name: str, window: int) -> pl.Expr:
    return pl.col(col_name).rolling_min(window).over("code")


def build_market_frame(args: argparse.Namespace) -> pl.DataFrame:
    conn = duckdb.connect(str(args.qmt_db), read_only=True)
    try:
        st_blacklist = get_st_blacklist_pl(args.st_snapshot_date)
        st_blacklist_df = pl.DataFrame({"code": st_blacklist}, schema={"code": pl.Utf8}).lazy()
        q_full = (
            load_daily_data_full(conn)
            .filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .join(st_blacklist_df, on="code", how="anti")
            .sort(["code", "date"])
            .with_columns(
                [
                    pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"),
                    price_limit_pct_expr().alias("_limit_pct"),
                ]
            )
            .collect()
        )

        regime = build_active_market_value_regime_frame(
            bull_trigger_pct=args.amv_bull_trigger_pct,
            bull_lookback_days=args.amv_bull_lookback_days,
            bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
            effective_lag_days=args.amv_effective_lag_days,
            date_col="date",
        ).select(["date", "is_bull_regime", "amv_mechanical_regime"])

        return (
            q_full.join(regime, on="date", how="left")
            .with_columns(
                [
                    pl.col("is_bull_regime").fill_null(False),
                    pl.col("amv_mechanical_regime").fill_null("unknown"),
                ]
            )
            .sort(["code", "date"])
        )
    finally:
        conn.close()


def add_running_weekly_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """复用 B1 的 running weekly MACD 强度，作为 B3 候选排序字段。"""
    df_with_time = df.sort(["code", "date"]).with_columns(
        pl.col("date").dt.truncate("1w").alias("_week_start")
    )
    weekly = (
        df_with_time.group_by(["code", "_week_start"])
        .agg(pl.col("close_adj").sort_by("date").last().alias("_weekly_close"))
        .sort(["code", "_week_start"])
        .with_columns(
            [
                pl.col("_weekly_close").ewm_mean(span=12, adjust=False).over("code").alias("_w_ema12"),
                pl.col("_weekly_close").ewm_mean(span=26, adjust=False).over("code").alias("_w_ema26"),
            ]
        )
        .with_columns((pl.col("_w_ema12") - pl.col("_w_ema26")).alias("_w_dif"))
        .with_columns(pl.col("_w_dif").ewm_mean(span=9, adjust=False).over("code").alias("_w_dea"))
    )
    weekly_prev = weekly.select(
        [
            "code",
            "_week_start",
            pl.col("_w_ema12").shift(1).over("code").alias("_prev_w_ema12"),
            pl.col("_w_ema26").shift(1).over("code").alias("_prev_w_ema26"),
            pl.col("_w_dea").shift(1).over("code").alias("_prev_w_dea"),
        ]
    )
    a12, a26, a9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0
    return (
        df_with_time.join(weekly_prev, on=["code", "_week_start"], how="left")
        .with_columns(
            [
                (a12 * pl.col("close_adj") + (1.0 - a12) * pl.col("_prev_w_ema12")).alias("rw_ema12"),
                (a26 * pl.col("close_adj") + (1.0 - a26) * pl.col("_prev_w_ema26")).alias("rw_ema26"),
            ]
        )
        .with_columns((pl.col("rw_ema12") - pl.col("rw_ema26")).alias("rw_dif"))
        .with_columns((a9 * pl.col("rw_dif") + (1.0 - a9) * pl.col("_prev_w_dea")).alias("rw_dea"))
        .with_columns(
            [
                (2.0 * (pl.col("rw_dif") - pl.col("rw_dea"))).alias("rw_hist"),
                (pl.col("rw_dif") / pl.col("close_adj") * 100.0).alias("rw_dif_pct"),
            ]
        )
        .drop(["_week_start", "_prev_w_ema12", "_prev_w_ema26", "_prev_w_dea"])
    )


def add_b3_features(df: pl.DataFrame) -> pl.DataFrame:
    tr_expr = pl.max_horizontal(
        [
            pl.col("high_adj") - pl.col("low_adj"),
            (pl.col("high_adj") - ref("close_adj")).abs(),
            (pl.col("low_adj") - ref("close_adj")).abs(),
        ]
    )
    return (
        df.with_columns(
            [
                pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").alias("_ema_c_10"),
                ref("close_adj").alias("_prev_close"),
                ref("open_adj").alias("_prev_open"),
                ref("high_adj").alias("_prev_high"),
                ref("low_adj").alias("_prev_low"),
                ref("volume").alias("_prev_volume"),
                ref("volume", 2).alias("_prev2_volume"),
                tr_expr.alias("_tr"),
                is_20pct_board_expr().alias("_is_20pct_board"),
            ]
        )
        .with_columns(
            [
                pl.col("_ema_c_10").ewm_mean(span=10, adjust=False).over("code").alias("_zhixing_short"),
                (
                    (
                        ma("close_adj", 14)
                        + ma("close_adj", 28)
                        + ma("close_adj", 57)
                        + ma("close_adj", 114)
                    )
                    / 4.0
                ).alias("_zhixing_long"),
                ((pl.col("close_adj") / pl.col("_prev_close") - 1.0) * 100.0).alias("_ret_pct"),
                (pl.col("high_adj") - pl.col("low_adj")).alias("_k_len"),
                (pl.col("high_adj") - pl.max_horizontal(["close_adj", "open_adj"])).alias("_upper_shadow"),
                (pl.min_horizontal(["close_adj", "open_adj"]) - pl.col("low_adj")).alias("_lower_shadow"),
                (pl.col("open_adj") - pl.col("close_adj")).abs().alias("_body"),
                ma("_tr", 30).alias("_tr_ma30"),
                ma("_tr", 60).alias("_tr_ma60"),
                ma("volume", 40).alias("_vol_ma40"),
                ma("volume", 60).alias("_vol_ma60"),
                hhv("high_adj", 30).alias("_hhv_h_30"),
                llv("low_adj", 30).alias("_llv_l_30"),
            ]
        )
        .with_columns(
            [
                (pl.col("_tr_ma30") / pl.col("_prev_close") * 100.0).alias("_volatility_30"),
                (pl.col("_tr_ma60") / pl.col("_prev_close") * 100.0).alias("_volatility_60"),
                pl.when(pl.col("_prev_volume") <= pl.col("volume") / 8.0)
                .then(pl.col("_prev2_volume"))
                .otherwise(pl.col("_prev_volume"))
                .alias("_ref_volume"),
                pl.max_horizontal(["_zhixing_short", "_zhixing_long"]).alias("_line_max"),
                ((pl.col("_hhv_h_30") - pl.col("_llv_l_30")) / pl.col("_llv_l_30") * 100.0).alias(
                    "_range_amp_30"
                ),
                pl.when(pl.col("_is_20pct_board")).then(9.0).otherwise(4.5).alias("_ret_limit"),
                (
                    pl.col("_zhixing_long").is_not_null().cum_sum().over("code")
                ).alias("_zhixing_long_bars"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("close_adj") > pl.col("open_adj"))
                    & (pl.col("_ret_pct") > pl.col("_volatility_30") * 1.5)
                    & (pl.col("_ret_pct") > 2.0)
                ).alias("_big_long_yang"),
                (
                    (pl.col("open_adj") - pl.col("_line_max")) / pl.col("open_adj") * 100.0
                ).alias("_deviation_distance"),
                (
                    (ref("_zhixing_long", 60) * 0.94 < pl.col("_zhixing_long"))
                    & (ref("_zhixing_long", 45) * 0.945 < pl.col("_zhixing_long"))
                    & (ref("_zhixing_long", 30) * 0.95 < pl.col("_zhixing_long"))
                ).alias("_yellow_limit_main"),
                (
                    (ref("_zhixing_long", 60) * 0.92 < pl.col("_zhixing_long"))
                    & (ref("_zhixing_long", 45) * 0.93 < pl.col("_zhixing_long"))
                    & (ref("_zhixing_long", 30) * 0.94 < pl.col("_zhixing_long"))
                ).alias("_yellow_limit_20pct"),
                (
                    (pl.col("close_adj") < pl.col("open_adj"))
                    & (pl.col("volume") > pl.col("_prev_volume") * 0.66)
                ).alias("_bad_volume_bar"),
                (
                    pl.when(pl.col("_is_20pct_board"))
                    .then(0.0)
                    .otherwise(ref("_upper_shadow"))
                    + pl.col("_upper_shadow")
                    <= ref("_body")
                ).alias("_upper_shadow_limit"),
                (ref("_lower_shadow") < ref("_body")).alias("_lower_shadow_limit"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_is_20pct_board"))
                .then(pl.col("_yellow_limit_20pct"))
                .otherwise(pl.col("_yellow_limit_main"))
                .alias("_yellow_limit"),
                (~pl.col("_bad_volume_bar")).alias("_volume_condition"),
                (pl.col("_upper_shadow_limit") & pl.col("_lower_shadow_limit")).alias(
                    "_shadow_condition"
                ),
                (
                    (pl.col("close_adj") > pl.col("_prev_close"))
                    & (pl.col("volume") > pl.col("_ref_volume") * 1.8)
                    & pl.col("_big_long_yang")
                    & (pl.col("volume") > pl.col("_vol_ma40"))
                ).alias("_key_k"),
                (
                    (pl.col("close_adj") > pl.col("_prev_close"))
                    & (pl.col("volume") > pl.col("_ref_volume") * 1.8)
                    & (pl.col("_ret_pct") > 4.0)
                    & (pl.col("_upper_shadow") <= pl.col("_k_len") / 4.0)
                    & (pl.col("volume") > pl.col("_vol_ma60"))
                    & ((ref("close_adj") - ref("open_adj", 9)) / ref("open_adj", 9) * 100.0 < 4.0)
                    & ((ref("close_adj") - ref("open_adj", 4)) / ref("open_adj", 4) * 100.0 < 4.0)
                ).alias("_violent_k"),
                (ref("_zhixing_short") >= ref("_zhixing_short", 2)).alias(
                    "_prev_short_line_up_strict"
                ),
            ]
        )
        .with_columns(
            [
                (ref("_key_k") | ref("_violent_k")).alias("_prev_key_or_violent_k"),
                ref("_key_k").alias("_prev_key_k"),
                ref("_violent_k").alias("_prev_violent_k"),
                ref("_ret_pct").alias("_prev_trigger_ret_pct"),
                (ref("volume") / ref("_ref_volume")).alias("_prev_trigger_volume_ratio"),
                ref("_deviation_distance").alias("_prev_deviation_distance"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("_prev_key_or_violent_k")
                    & (pl.col("volume") > 0)
                    & (pl.col("close_adj") >= pl.col("_prev_close"))
                    & (pl.col("_prev_close") > pl.col("_prev_open"))
                    & ((pl.col("high_adj") - pl.col("low_adj")) * 0.9 < (pl.col("_prev_high") - pl.col("_prev_low")))
                    & (pl.col("_zhixing_short") >= ref("_zhixing_short"))
                    & (pl.col("_zhixing_long") >= ref("_zhixing_long"))
                    # TDX 原式 `(REF(short,1) >= REF(short,2)) + 0.01` 在布尔上下文近似恒真。
                    & (pl.col("_zhixing_long") >= ref("_zhixing_long", 2))
                    & pl.col("_volume_condition")
                    & (pl.col("_ret_pct") < pl.col("_ret_limit"))
                    & pl.col("_shadow_condition")
                    & pl.when(pl.col("_zhixing_long_bars") >= 60)
                    .then(pl.col("_yellow_limit"))
                    .otherwise(True)
                    & (ref("_ret_pct", 2) < 3.0)
                    & (pl.col("_range_amp_30") < 40.0)
                    & (
                        pl.col("_prev_deviation_distance")
                        < pl.when(pl.col("_is_20pct_board"))
                        .then(ref("_volatility_60") * 1.3)
                        .otherwise(ref("_volatility_60") * 1.2)
                    )
                )
                .fill_null(False)
                .alias("_b3_raw_tdx")
            ]
        )
        .with_columns(
            [
                (
                    pl.col("_b3_raw_tdx") & pl.col("_prev_short_line_up_strict").fill_null(False)
                ).alias("_b3_raw_strict_short_line"),
                (
                    pl.col("open_adj") / pl.col("pre_close_adj") - 1.0
                    >= pl.col("_limit_pct") - PRICE_LIMIT_TOLERANCE
                ).alias("_open_limit_up"),
            ]
        )
    )


def build_signal_export(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    market = add_b3_features(add_running_weekly_momentum(build_market_frame(args)))
    liquidity_expr = (pl.col("market_cap_100m") >= args.mv_min) & (
        pl.col("amount_ma20") >= args.amount_ma20_min
    )
    candidate_expr = pl.col("_b3_raw_tdx") & pl.col("is_bull_regime") & liquidity_expr
    sort_col_name = "_prev_trigger_ret_pct" if args.sort_field == "prev_trigger_ret_pct" else args.sort_field
    sort_expr = pl.col(sort_col_name)
    scored = (
        market.with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(sort_expr)
            .otherwise(None)
            .alias("_signal_score")
        )
        .with_columns(
            pl.col("_signal_score")
            .rank(method="ordinal", descending=not args.sort_ascending)
            .over("date")
            .alias("_signal_rank")
        )
    )
    raw_candidates = (
        scored.filter(pl.col("_is_signal_candidate"))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
                pl.col("_prev_key_k").alias("prev_key_k"),
                pl.col("_prev_violent_k").alias("prev_violent_k"),
                pl.when(pl.col("_prev_violent_k"))
                .then(pl.lit("violent_k"))
                .when(pl.col("_prev_key_k"))
                .then(pl.lit("key_k"))
                .otherwise(pl.lit("unknown"))
                .alias("trigger_type"),
                pl.col("_prev_low").alias("trigger_low"),
                pl.col("_prev_high").alias("trigger_high"),
                pl.col("_prev_close").alias("trigger_close"),
                pl.col("_prev_trigger_ret_pct").alias("prev_trigger_ret_pct"),
                pl.col("_prev_trigger_volume_ratio").alias("prev_trigger_volume_ratio"),
                pl.col("_prev_deviation_distance").alias("prev_deviation_distance"),
                pl.col("_range_amp_30").alias("range_amp_30"),
                pl.col("_zhixing_short").alias("white_line"),
                pl.col("_zhixing_long").alias("yellow_line"),
                "rw_dif_pct",
                "rw_hist",
                "rw_dif",
                "market_cap_100m",
                "amount_ma20",
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )
    signal_rows = raw_candidates.filter(pl.col("rank") <= args.top_n)

    trading_dates = scored.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date")).drop_nulls(
        "execution_date"
    )
    execution_signals = (
        signal_rows.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(
            [
                "execution_date",
                "code",
                "signal_date",
                "score",
                "rank",
                "trigger_type",
                "trigger_low",
                "trigger_high",
                "trigger_close",
                "prev_trigger_ret_pct",
                "prev_trigger_volume_ratio",
                "rw_dif_pct",
            ]
        )
        .rename({"execution_date": "date"})
    )

    export = (
        scored.select(
            [
                "date",
                "code",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "pre_close_adj",
                "is_bull_regime",
                "amv_mechanical_regime",
                "market_cap_100m",
                "amount_ma20",
                pl.col("_zhixing_short").alias("white_line"),
                pl.col("_zhixing_long").alias("yellow_line"),
                "rw_dif_pct",
                "rw_hist",
                "rw_dif",
                pl.col("_b3_raw_tdx").alias("b3_raw_tdx"),
                pl.col("_b3_raw_strict_short_line").alias("b3_raw_strict_short_line"),
                pl.col("_open_limit_up").alias("is_open_limit_up"),
            ]
        )
        .join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            [
                pl.col("signal_date").is_not_null().alias("is_signal"),
                pl.col("score").fill_null(0.0),
                pl.col("rank").fill_null(9999).cast(pl.UInt32),
                pl.col("trigger_type").fill_null(""),
                pl.col("trigger_low").fill_null(0.0),
                pl.col("trigger_high").fill_null(0.0),
                pl.col("trigger_close").fill_null(0.0),
                pl.col("prev_trigger_ret_pct").fill_null(0.0),
                pl.col("prev_trigger_volume_ratio").fill_null(0.0),
            ]
        )
        .sort(["date", "code"])
    )

    signal_exec = export.filter(pl.col("is_signal"))
    summary = {
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "b3_raw_tdx_rows": int(scored["_b3_raw_tdx"].sum()),
        "b3_raw_tdx_days": scored.filter(pl.col("_b3_raw_tdx")).select("date").n_unique(),
        "b3_raw_strict_short_line_rows": int(scored["_b3_raw_strict_short_line"].sum()),
        "b3_amv_bull_liquid_rows_before_topn": raw_candidates.height,
        "b3_amv_bull_liquid_days_before_topn": raw_candidates.select("signal_date").n_unique(),
        "signal_rows_before_shift": signal_rows.height,
        "signal_days_before_shift": signal_rows.select("signal_date").n_unique(),
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": signal_exec.select("date").n_unique(),
        "signals_blocked_by_execution_bear_regime": int(
            signal_exec.filter(~pl.col("is_bull_regime")).height
        ),
        "signals_open_limit_up": int(signal_exec.filter(pl.col("is_open_limit_up")).height),
        "signals_open_limit_up_ratio": (
            float(signal_exec.filter(pl.col("is_open_limit_up")).height / signal_exec.height)
            if signal_exec.height
            else None
        ),
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
        "sort_field": args.sort_field,
        "sort_ascending": args.sort_ascending,
        "tdx_translation_note": (
            "TDX line `(REF(知行短期趋势线,1) >= REF(知行短期趋势线,2))+0.01` "
            "is treated as truthy for b3_raw_tdx; strict variant is reported separately."
        ),
    }
    return export, raw_candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export B3 TDX formula signals for bt-amv-topn")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--sort-field", default="prev_trigger_ret_pct")
    parser.add_argument("--sort-ascending", action="store_true")
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signal.parquet"
    raw_candidates_path = output_dir / "raw_candidate_signals.csv"
    meta_path = output_dir / "signal.meta.json"

    print("Building B3 TDX signal export...")
    export, raw_candidates, summary = build_signal_export(args)
    export.write_parquet(signal_path)
    raw_candidates.write_csv(raw_candidates_path)

    meta: dict[str, Any] = {
        "strategy": "b3_tdx_amv_bull",
        "signal_id": output_dir.name,
        "signal_run_id": f"b3_tdx_amv_bull_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": "b3_tdx_core_amv_bull_top3",
        "model_name": "tdx_formula_rewrite",
        "feature_mode": "b3_core_tdx_polars",
        "feature_count": None,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "top_n": args.top_n,
            "sort_field": args.sort_field,
            "sort_ascending": args.sort_ascending,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
            "price_limit_tolerance": PRICE_LIMIT_TOLERANCE,
        },
        "summary": summary,
        "files": {
            "signal": _rel_path(signal_path, output_dir),
            "raw_candidate_signals": _rel_path(raw_candidates_path, output_dir),
        },
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Saved signal: {signal_path}")
    print(f"Saved meta:   {meta_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Relative signal: {_rel_path(signal_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
