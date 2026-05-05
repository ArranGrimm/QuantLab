import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 活跃市值 Regime Lab

    这个 notebook 用新的 `active_market_value.duckdb` 做两件事:

    1. 把活跃市值按 K 线形态画出来, 先用肉眼确认它和指南针里的走势是否一致。
    2. 从活跃市值 OHLC 机械生成 bull / bear regime, 并和旧的手工 `LOOSE_PERIODS` 做交叉对账。

    第一版只做探索, 不写回 `utils/manual_bull_periods.py`。
    """)
    return


@app.cell
def _():
    import duckdb
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from datetime import datetime

    from utils import get_st_blacklist_pl, load_daily_data_full
    from utils.manual_bull_periods import LOOSE_PERIODS

    pl.Config(tbl_rows=-1, tbl_cols=-1)
    return LOOSE_PERIODS, datetime, duckdb, get_st_blacklist_pl, go, load_daily_data_full, mo, pl


@app.cell
def _(mo):
    AMV_DB_PATH = r"../QuantData/Ashare/active_market_value.duckdb"

    lookback_days = mo.ui.slider(
        start=120,
        stop=1776,
        value=520,
        step=20,
        label="K 线显示最近 N 个交易日",
        full_width=True,
    )
    bull_trigger_pct = mo.ui.slider(
        value=4.0,
        start=1.0,
        stop=10.0,
        step=0.1,
        label="Bull trigger: 1/2/3 日累计涨幅 >= (%)",
        full_width=True,
    )
    bear_trigger_pct = mo.ui.slider(
        value=-2.3,
        start=-10.0,
        stop=-0.5,
        step=0.1,
        label="Bear trigger: 单日跌幅 <= (%)",
        full_width=True,
    )

    mo.vstack([lookback_days, mo.hstack([bull_trigger_pct, bear_trigger_pct])])
    return AMV_DB_PATH, bear_trigger_pct, bull_trigger_pct, lookback_days


@app.cell
def _(AMV_DB_PATH, duckdb, pl):
    _conn = duckdb.connect(AMV_DB_PATH, read_only=True)
    try:
        df_amv = _conn.execute(
            """
            SELECT
                trade_date,
                amv_open,
                amv_high,
                amv_low,
                amv_close,
                chg_abs_pct,
                volume_100m,
                amount_100m,
                position_100m,
                turnover_pct,
                amplitude_pct,
                quality_flags
            FROM active_market_value
            ORDER BY trade_date
            """
        ).pl()
    finally:
        _conn.close()

    df_amv = df_amv.with_columns(
        [
            ((pl.col("amv_close") / pl.col("amv_close").shift(1) - 1) * 100).alias("ret_1d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(2) - 1) * 100).alias("ret_2d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(3) - 1) * 100).alias("ret_3d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(5) - 1) * 100).alias("ret_5d"),
        ]
    )
    df_amv.head(), df_amv.tail()
    return (df_amv,)


@app.cell
def _(LOOSE_PERIODS, bear_trigger_pct, bull_trigger_pct, datetime, df_amv, pl):
    def _manual_bull_expr(date_col: str = "trade_date") -> pl.Expr:
        expr = pl.lit(False)
        for start, end in LOOSE_PERIODS:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
            expr = expr | pl.col(date_col).is_between(start_date, end_date, closed="both")
        return expr

    EFFECTIVE_LAG_DAYS = 1

    def _regime_state(rows: list[dict]) -> tuple[list[str], list[str]]:
        state = "neutral"
        observed_states: list[str] = []
        for row in rows:
            if row["bear_trigger"]:
                state = "bear"
            if row["bull_trigger"]:
                state = "bull"
            observed_states.append(state)

        effective_states = (
            ["neutral"] * EFFECTIVE_LAG_DAYS + observed_states[:-EFFECTIVE_LAG_DAYS]
            if EFFECTIVE_LAG_DAYS
            else observed_states
        )
        return observed_states, effective_states

    df_regime = df_amv.with_columns(
        [
            pl.max_horizontal("ret_1d", "ret_2d", "ret_3d").alias("max_ret_3d"),
            pl.min_horizontal("ret_1d", "ret_2d", "ret_3d").alias("min_ret_3d"),
        ]
    ).with_columns(
        [
            (pl.col("max_ret_3d") >= bull_trigger_pct.value).fill_null(False).alias("bull_trigger"),
            (pl.col("ret_1d") <= bear_trigger_pct.value).fill_null(False).alias("bear_trigger"),
            _manual_bull_expr().alias("is_manual_bull"),
        ]
    )

    observed_regime_states, effective_regime_states = _regime_state(
        df_regime.select(["bull_trigger", "bear_trigger"]).to_dicts()
    )
    df_regime = df_regime.with_columns(
        [
            pl.Series("observed_mechanical_regime", observed_regime_states),
            pl.Series("mechanical_regime", effective_regime_states),
            (pl.Series("mechanical_regime", effective_regime_states) == "bull").alias(
                "is_mechanical_bull"
            ),
        ]
    )
    df_regime.tail(20)
    return (df_regime,)


@app.cell
def _(df_regime, pl):
    def _compress_periods(records: list[dict], flag_col: str) -> list[dict]:
        periods: list[dict] = []
        start = None
        prev_date = None
        n_days = 0

        for row in records:
            current = bool(row[flag_col])
            trade_date = row["trade_date"]
            if current and start is None:
                start = trade_date
                n_days = 0
            if current:
                n_days += 1
            if not current and start is not None:
                periods.append({"start": start, "end": prev_date, "n_trading_days": n_days})
                start = None
                n_days = 0
            prev_date = trade_date

        if start is not None:
            periods.append({"start": start, "end": prev_date, "n_trading_days": n_days})
        return periods

    manual_periods_df = pl.DataFrame(_compress_periods(df_regime.to_dicts(), "is_manual_bull"))
    mechanical_periods_df = pl.DataFrame(_compress_periods(df_regime.to_dicts(), "is_mechanical_bull"))
    mechanical_periods_df
    return manual_periods_df, mechanical_periods_df


@app.cell
def _(
    df_regime,
    go,
    lookback_days,
    manual_periods_df,
    mechanical_periods_df,
    mo,
    pl,
):
    df_plot = df_regime.tail(lookback_days.value).with_columns(
        [
            pl.col("trade_date").cast(pl.String).alias("date_str"),
            pl.when(pl.col("is_manual_bull") & pl.col("is_mechanical_bull"))
            .then(pl.lit("both"))
            .when(pl.col("is_manual_bull"))
            .then(pl.lit("manual_only"))
            .when(pl.col("is_mechanical_bull"))
            .then(pl.lit("mechanical_only"))
            .otherwise(pl.lit("neither"))
            .alias("overlap_state"),
        ]
    )
    pdf = df_plot.to_pandas()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=pdf["date_str"],
            open=pdf["amv_open"],
            high=pdf["amv_high"],
            low=pdf["amv_low"],
            close=pdf["amv_close"],
            name="AMV K线",
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
        )
    )

    bull_trigger_pdf = pdf[pdf["bull_trigger"]]
    if len(bull_trigger_pdf):
        fig.add_trace(
            go.Scatter(
                x=bull_trigger_pdf["date_str"],
                y=bull_trigger_pdf["amv_low"] * 0.995,
                mode="markers",
                name="机械 bull trigger",
                marker=dict(symbol="triangle-up", size=9, color="#ff7f0e"),
                hovertemplate="%{x}<br>bull trigger<extra></extra>",
            )
        )

    bear_trigger_pdf = pdf[pdf["bear_trigger"]]
    if len(bear_trigger_pdf):
        fig.add_trace(
            go.Scatter(
                x=bear_trigger_pdf["date_str"],
                y=bear_trigger_pdf["amv_high"] * 1.005,
                mode="markers",
                name="机械 bear trigger",
                marker=dict(symbol="triangle-down", size=9, color="#1f77b4"),
                hovertemplate="%{x}<br>bear trigger<extra></extra>",
            )
        )

    plot_start = df_plot["trade_date"].min()
    plot_end = df_plot["trade_date"].max()
    for period in manual_periods_df.filter(
        (pl.col("end") >= plot_start) & (pl.col("start") <= plot_end)
    ).to_dicts():
        fig.add_vrect(
            x0=str(max(period["start"], plot_start)),
            x1=str(min(period["end"], plot_end)),
            fillcolor="#d62728",
            opacity=0.08,
            layer="below",
            line_width=0,
        )

    for period in mechanical_periods_df.filter(
        (pl.col("end") >= plot_start) & (pl.col("start") <= plot_end)
    ).to_dicts():
        fig.add_vrect(
            x0=str(max(period["start"], plot_start)),
            x1=str(min(period["end"], plot_end)),
            fillcolor="#ffbf00",
            opacity=0.07,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=f"活跃市值 K线: {pdf['date_str'].iloc[0]} -> {pdf['date_str'].iloc[-1]}",
        height=720,
        xaxis=dict(type="category", nticks=24, tickangle=-45),
        yaxis_title="活跃市值",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(l=50, r=40, t=70, b=60),
    )
    mo.ui.plotly(fig)
    return


@app.cell
def _(df_regime, mo, pl):
    overlap_summary = (
        df_regime.with_columns(
            pl.when(pl.col("is_manual_bull") & pl.col("is_mechanical_bull"))
            .then(pl.lit("both"))
            .when(pl.col("is_manual_bull"))
            .then(pl.lit("manual_only"))
            .when(pl.col("is_mechanical_bull"))
            .then(pl.lit("mechanical_only"))
            .otherwise(pl.lit("neither"))
            .alias("overlap_state")
        )
        .group_by("overlap_state")
        .agg(pl.len().alias("n_trading_days"))
        .sort("overlap_state")
    )

    manual_days = df_regime.filter(pl.col("is_manual_bull")).height
    mechanical_days = df_regime.filter(pl.col("is_mechanical_bull")).height
    both_days = df_regime.filter(pl.col("is_manual_bull") & pl.col("is_mechanical_bull")).height

    coverage = pl.DataFrame(
        {
            "metric": [
                "manual_days",
                "mechanical_bull_days",
                "both_days",
                "both / manual",
                "both / mechanical",
            ],
            "value": [
                float(manual_days),
                float(mechanical_days),
                float(both_days),
                both_days / manual_days if manual_days else None,
                both_days / mechanical_days if mechanical_days else None,
            ],
        }
    )
    mo.vstack([mo.md("## 交叉对账摘要"), overlap_summary, coverage])
    return


@app.cell
def _(LOOSE_PERIODS, datetime, df_regime, mo, pl):
    manual_rows = []
    for start, end in LOOSE_PERIODS:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        window = df_regime.filter(pl.col("trade_date").is_between(start_date, end_date, closed="both"))
        if window.is_empty():
            manual_rows.append(
                {
                    "manual_start": start,
                    "manual_end": end,
                    "manual_days": 0,
                    "first_bull_trigger": None,
                    "first_mechanical_bull": None,
                    "overlap_days": 0,
                    "overlap_ratio": None,
                }
            )
            continue

        bull_trigger_dates = window.filter(pl.col("bull_trigger"))["trade_date"].to_list()
        mechanical_dates = window.filter(pl.col("is_mechanical_bull"))["trade_date"].to_list()
        overlap_days = window.filter(pl.col("is_mechanical_bull")).height
        manual_rows.append(
            {
                "manual_start": start,
                "manual_end": end,
                "manual_days": window.height,
                "first_bull_trigger": bull_trigger_dates[0] if bull_trigger_dates else None,
                "first_mechanical_bull": mechanical_dates[0] if mechanical_dates else None,
                "overlap_days": overlap_days,
                "overlap_ratio": overlap_days / window.height if window.height else None,
            }
        )

    manual_compare_df = pl.DataFrame(manual_rows)
    mo.vstack([mo.md("## 每个手工 LOOSE_PERIODS 的机械解释度"), manual_compare_df])
    return


@app.cell
def _(df_regime, mechanical_periods_df, mo):
    recent_triggers = df_regime.filter(
        df_regime["bull_trigger"] | df_regime["bear_trigger"]
    ).select(
        [
            "trade_date",
            "amv_close",
            "ret_1d",
            "ret_2d",
            "ret_3d",
            "ret_5d",
            "max_ret_3d",
            "min_ret_3d",
            "bull_trigger",
            "bear_trigger",
            "mechanical_regime",
            "is_manual_bull",
        ]
    )

    mo.vstack(
        [
            mo.md("## 机械区间与最近触发点"),
            mechanical_periods_df.tail(30),
            recent_triggers.tail(80),
        ]
    )
    return


@app.cell
def _(duckdb, get_st_blacklist_pl, load_daily_data_full, pl):
    # ==============================================================================
    # AMV regime 随机买基线
    #
    # 目的: 不使用任何模型分数, 只测试机械活跃市值 regime 是否能区分
    # "随便买也相对更好 / 更差" 的市场状态。
    # ==============================================================================
    QMT_DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    RANDOM_BUY_START_DATE = "2022-09-01"
    RANDOM_BUY_MV_MIN = 50
    RANDOM_BUY_MV_MAX = None
    RANDOM_BUY_MIN_LIST_DAYS = 60
    RANDOM_BUY_AMOUNT_MA20_MIN = 5e7
    RANDOM_BUY_HORIZONS = [1, 3, 5, 10, 15, 20]

    _conn = duckdb.connect(QMT_DB_PATH, read_only=True)
    try:
        st_blacklist = get_st_blacklist_pl("2026-03-31")
        st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()
        df_random_pool = (
            load_daily_data_full(_conn)
            .join(st_blacklist_df, on="code", how="anti")
            .sort(["code", "date"])
            .with_columns(
                [
                    pl.col("date").cum_count().over("code").alias("_list_days"),
                    pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"),
                    *[
                        (
                            pl.col("close_adj").shift(-h).over("code") / pl.col("close_adj") - 1
                        ).alias(f"fwd_ret_{h}d")
                        for h in RANDOM_BUY_HORIZONS
                    ],
                ]
            )
            .filter(pl.col("date") >= pl.lit(RANDOM_BUY_START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(
                (pl.col("_list_days") >= RANDOM_BUY_MIN_LIST_DAYS)
                & (pl.col("market_cap_100m") >= RANDOM_BUY_MV_MIN)
                & (
                    pl.lit(True)
                    if RANDOM_BUY_MV_MAX is None
                    else (pl.col("market_cap_100m") <= RANDOM_BUY_MV_MAX)
                )
                & (pl.col("amount_ma20") >= RANDOM_BUY_AMOUNT_MA20_MIN)
            )
            .select(["date", "code", "market_cap_100m", "amount_ma20", *[f"fwd_ret_{h}d" for h in RANDOM_BUY_HORIZONS]])
            .collect()
        )
    finally:
        _conn.close()

    print(
        f"随机买池: rows={df_random_pool.height:,}, "
        f"dates={df_random_pool['date'].min()} -> {df_random_pool['date'].max()}, "
        f"codes={df_random_pool['code'].n_unique():,}"
    )
    return (
        RANDOM_BUY_HORIZONS,
        RANDOM_BUY_START_DATE,
        df_random_pool,
    )


@app.cell
def _(RANDOM_BUY_HORIZONS, df_random_pool, df_regime, mo, pl):
    df_regime_daily = (
        df_regime
        .rename({"trade_date": "date"})
        .select(["date", "mechanical_regime", "is_mechanical_bull", "bull_trigger", "bear_trigger"])
    )

    df_random_eval = (
        df_random_pool
        .join(df_regime_daily, on="date", how="left")
        .with_columns(
            [
                pl.col("mechanical_regime").fill_null("unknown"),
                pl.col("is_mechanical_bull").fill_null(False),
            ]
        )
    )

    def _summary_for_filter(label: str, condition: pl.Expr) -> pl.DataFrame:
        filtered = df_random_eval.filter(condition)
        if filtered.is_empty():
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "bucket": label,
                    "rows": filtered.height,
                    "trading_days": filtered["date"].n_unique(),
                    "avg_candidates_per_day": filtered.height / max(filtered["date"].n_unique(), 1),
                    **{
                        f"mean_ret_{h}d_pct": filtered[f"fwd_ret_{h}d"].mean() * 100
                        for h in RANDOM_BUY_HORIZONS
                    },
                    **{
                        f"win_{h}d_pct": (
                            filtered.filter(pl.col(f"fwd_ret_{h}d") > 0).height
                            / max(filtered.filter(pl.col(f"fwd_ret_{h}d").is_not_null()).height, 1)
                            * 100
                        )
                        for h in RANDOM_BUY_HORIZONS
                    },
                }
            ]
        )

    random_regime_summary = pl.concat(
        [
            _summary_for_filter("all", pl.lit(True)),
            _summary_for_filter("mechanical_bull", pl.col("mechanical_regime") == "bull"),
            _summary_for_filter("mechanical_bear", pl.col("mechanical_regime") == "bear"),
            _summary_for_filter("mechanical_neutral", pl.col("mechanical_regime") == "neutral"),
        ],
        how="diagonal",
    ).with_columns(
        [
            pl.col("avg_candidates_per_day").round(1),
            *[pl.col(f"mean_ret_{h}d_pct").round(3) for h in RANDOM_BUY_HORIZONS],
            *[pl.col(f"win_{h}d_pct").round(1) for h in RANDOM_BUY_HORIZONS],
        ]
    )

    mo.vstack(
        [
            mo.md("## 机械 AMV regime 随机买基线"),
            mo.md(
                "Universe: 非 ST, 流通市值 `>=50` 亿且不设上限, 上市满 `60` 天, "
                "`amount_ma20 >= 5000万`. 这里不使用模型分数, 只看 AMV 状态本身。"
            ),
            random_regime_summary,
        ]
    )
    return df_random_eval, random_regime_summary


@app.cell
def _(RANDOM_BUY_HORIZONS, df_random_eval, mo, pl):
    year_regime_summary = (
        df_random_eval
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by(["year", "mechanical_regime"])
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("date").n_unique().alias("trading_days"),
                *[
                    (pl.col(f"fwd_ret_{h}d").mean() * 100).alias(f"mean_ret_{h}d_pct")
                    for h in RANDOM_BUY_HORIZONS
                ],
            ]
        )
        .with_columns(
            [
                *[pl.col(f"mean_ret_{h}d_pct").round(3) for h in RANDOM_BUY_HORIZONS],
            ]
        )
        .sort(["year", "mechanical_regime"])
    )

    mo.vstack(
        [
            mo.md("## 随机买基线: 分年 / regime"),
            year_regime_summary,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
