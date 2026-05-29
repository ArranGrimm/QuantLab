from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from strategies.amv.scoring import build_score_expr, finite_expr, required_factor_names
from strategies.amv.specs import RankerSpec


AMV_SIGNAL_EXPORT_COLUMNS: tuple[str, ...] = (
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
    "is_bull_regime",
    "amv_mechanical_regime",
    "market_cap_100m",
    "amount_ma20",
)


@dataclass(frozen=True)
class SignalAssemblyConfig:
    sleeve_id: str
    top_n: int = 3
    mv_min: float = 100.0
    amount_ma20_min: float = 5e7
    candidate_col: str = "_is_signal_candidate"
    score_col: str = "_signal_score"
    rank_col: str = "_signal_rank"
    require_bull_regime: bool = True
    export_columns: tuple[str, ...] = AMV_SIGNAL_EXPORT_COLUMNS


def ranker_score_expr(ranker: RankerSpec) -> pl.Expr:
    if ranker.components:
        return build_score_expr(ranker.components)
    if ranker.factor is None or ranker.descending is None:
        raise ValueError(f"ranker {ranker.id} must define either components or factor/descending")
    return pl.col(ranker.factor).rank(method="average", descending=not ranker.descending).over("date") / pl.len().over(
        "date"
    )


def ranker_required_columns(ranker: RankerSpec) -> list[str]:
    if ranker.components:
        return required_factor_names(ranker.components)
    if ranker.factor is None:
        raise ValueError(f"ranker {ranker.id} must define factor or components")
    return [ranker.factor]


def base_candidate_expr(
    required_cols: Sequence[str],
    *,
    mv_min: float,
    amount_ma20_min: float,
    require_bull_regime: bool = True,
) -> pl.Expr:
    candidate = (pl.col("market_cap_100m") >= mv_min) & (pl.col("amount_ma20") >= amount_ma20_min)
    if require_bull_regime:
        candidate = pl.col("is_bull_regime") & candidate
    for col_name in required_cols:
        candidate = candidate & finite_expr(col_name)
    return candidate


def with_signal_scores(
    frame: pl.DataFrame,
    *,
    score_expr: pl.Expr,
    candidate_expr: pl.Expr,
    config: SignalAssemblyConfig,
) -> pl.DataFrame:
    return (
        frame.with_columns(candidate_expr.alias(config.candidate_col))
        .with_columns(
            pl.when(pl.col(config.candidate_col))
            .then(score_expr)
            .otherwise(None)
            .alias(config.score_col)
        )
        .with_columns(
            pl.col(config.score_col)
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias(config.rank_col)
        )
    )


def _select_extra_columns(extra_columns: Sequence[str | pl.Expr]) -> list[str | pl.Expr]:
    return [pl.col(col) if isinstance(col, str) else col for col in extra_columns]


def select_signal_rows(
    scored: pl.DataFrame,
    *,
    config: SignalAssemblyConfig,
    extra_columns: Sequence[str | pl.Expr] = (),
) -> pl.DataFrame:
    return (
        scored.filter(pl.col(config.candidate_col) & (pl.col(config.rank_col) <= config.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.lit(config.sleeve_id).alias("sleeve_id"),
                pl.col(config.score_col).alias("score"),
                pl.col(config.rank_col).cast(pl.UInt32).alias("rank"),
                *_select_extra_columns(extra_columns),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )


def shift_signal_rows_to_execution(market: pl.DataFrame, signal_rows: pl.DataFrame) -> pl.DataFrame:
    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date")).drop_nulls(
        "execution_date"
    )
    return (
        signal_rows.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(["execution_date", "code", "signal_date", "sleeve_id", "score", "rank"])
        .rename({"execution_date": "date"})
    )


def build_backtest_signal_frame(
    market: pl.DataFrame,
    signal_rows: pl.DataFrame,
    *,
    export_columns: Sequence[str] = AMV_SIGNAL_EXPORT_COLUMNS,
) -> pl.DataFrame:
    execution_signals = shift_signal_rows_to_execution(market, signal_rows)
    return (
        market.select(list(export_columns))
        .join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            [
                pl.col("signal_date").is_not_null().alias("is_signal"),
                pl.col("score").fill_null(0.0),
                pl.col("rank").fill_null(9999).cast(pl.UInt32),
                pl.col("sleeve_id").fill_null(""),
            ]
        )
        .sort(["date", "code"])
    )


def assemble_ranker_signal(
    market: pl.DataFrame,
    ranker: RankerSpec,
    config: SignalAssemblyConfig,
    *,
    candidate_expr: pl.Expr | None = None,
    extra_columns: Sequence[str | pl.Expr] = (),
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    required_cols = ranker_required_columns(ranker)
    resolved_candidate = candidate_expr or base_candidate_expr(
        required_cols,
        mv_min=config.mv_min,
        amount_ma20_min=config.amount_ma20_min,
        require_bull_regime=config.require_bull_regime,
    )
    scored = with_signal_scores(
        market,
        score_expr=ranker_score_expr(ranker),
        candidate_expr=resolved_candidate,
        config=config,
    )
    signal_rows = select_signal_rows(scored, config=config, extra_columns=extra_columns)
    export = build_backtest_signal_frame(market, signal_rows, export_columns=config.export_columns)
    return export, signal_rows, scored
