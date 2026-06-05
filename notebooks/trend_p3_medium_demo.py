"""trend-p3-medium 单文件管道 Demo。

目的：验证"数据 + 因子 + 惩罚 + 排序 → signal"整条链路压缩到一个 notebook 里
到底有多少行、吃多少内存、跑多长时间。
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # trend-p3-medium 单文件管道

    在**一个 notebook** 里跑完整条 export 链路：
    1. TDX 加载全量 OHLC
    2. ST 过滤 + regime join
    3. 计算 4 个趋势因子 + medium penalty（lazy 链，不 collect）
    4. 一次 collect → 截面 rank → Top3
    5. 写出 signal.parquet

    观察：总代码行数、内存峰值、耗时。
    """)
    return


@app.cell
def _():
    import time
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from loguru import logger

    # 项目根路径
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from utils.data_source import TdxDailyReader, DataSourceSettings
    from utils import get_st_blacklist_pl
    from utils.active_market_value_regime import build_active_market_value_regime_frame

    pl.Config.set_tbl_rows(20)
    return (
        DataSourceSettings,
        ROOT,
        TdxDailyReader,
        build_active_market_value_regime_frame,
        get_st_blacklist_pl,
        logger,
        mo,
        pl,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 配置
    """)
    return


@app.cell
def _(DataSourceSettings, ROOT):
    # ── 策略参数 ──
    START_DATE = "2019-01-01"
    END_DATE   = "2026-06-03"
    ST_SNAPSHOT = "2026-03-31"
    MV_MIN      = 100.0
    AMOUNT_MA20_MIN = 5e7
    TOP_N       = 3

    # ── 趋势 P-block 权重 ──
    P_WEIGHT = 3.0
    K_WEIGHT = 0.5

    # ── Medium penalty 参数 ──
    MEDIUM_PENALTY       = 0.03
    MEDIUM_WEAK_THRESHOLD = 0.50

    # ── 数据源 ──
    TDX_DB = ROOT.parent / "QuantData" / "Ashare" / "tdx.db"
    data_source = DataSourceSettings(provider="tdx", tdx_db=TDX_DB, start_date=START_DATE)

    (START_DATE, END_DATE, MV_MIN, AMOUNT_MA20_MIN, TOP_N,
     P_WEIGHT, K_WEIGHT, MEDIUM_PENALTY, MEDIUM_WEAK_THRESHOLD,
     ST_SNAPSHOT, TDX_DB, data_source)
    return (
        AMOUNT_MA20_MIN,
        END_DATE,
        K_WEIGHT,
        MEDIUM_PENALTY,
        MEDIUM_WEAK_THRESHOLD,
        MV_MIN,
        P_WEIGHT,
        START_DATE,
        ST_SNAPSHOT,
        TOP_N,
        data_source,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 数据加载（DuckDB → LazyFrame）
    """)
    return


@app.cell
def _(
    END_DATE,
    START_DATE,
    ST_SNAPSHOT,
    TdxDailyReader,
    build_active_market_value_regime_frame,
    data_source,
    get_st_blacklist_pl,
    logger,
    pl,
    time,
):
    _t0 = time.perf_counter()

    # ── ST 黑名单 ──
    st_list = get_st_blacklist_pl(ST_SNAPSHOT)

    # ── 读 TDX 全量日线 ──
    reader = TdxDailyReader(data_source)

    q_raw = (
        reader.load_daily_full()
        .filter(pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
        .filter(pl.col("date") <= pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
        .filter(~pl.col("code").is_in(st_list))
        .sort(["code", "date"])
    )

    # ── 选列 + amount_ma20 ──
    BASE_COLS = [
        "code", "date",
        "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
        "open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw",
        "market_cap_100m", "amount", "volume", "turnover",
    ]
    available = [c for c in BASE_COLS if c in q_raw.collect_schema().names()]
    lf = (
        q_raw.with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
        .select([*available, "amount_ma20"])
    )

    # ── regime join ──
    df_regime = (
        build_active_market_value_regime_frame(
            bull_trigger_pct=4.0, bull_lookback_days=2,
            bear_trigger_1d_pct=-2.3, effective_lag_days=1, date_col="date",
        )
        .select(["date", "is_bull_regime", "amv_mechanical_regime"])
        .lazy()
    )

    lf = (
        lf.join(df_regime, on="date", how="left")
        .with_columns(
            pl.col("is_bull_regime").fill_null(False),
            pl.col("amv_mechanical_regime").fill_null("unknown"),
        )
        .sort(["date", "code"])
    )

    t_load = time.perf_counter() - _t0
    logger.info(f"LazyFrame 构造完成，{t_load:.1f}s，{lf.collect_schema().len()} 列")
    return lf, reader, t_load


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 因子 + Medium penalty（挂在 LazyFrame 上，不 collect）
    """)
    return


@app.cell
def _(K_WEIGHT, MEDIUM_PENALTY, MEDIUM_WEAK_THRESHOLD, P_WEIGHT, lf, pl):
    from strategies.amv.factors.base import calc_amv_core_factors
    from strategies.amv.factors import ranker_score_expr
    from strategies.amv.specs import RankerSpec, ScoreComponent

    # ── 构造 trend-p3 Ranker（唯一真相源 + 可读） ──
    ranker = RankerSpec(
        id="trend-p3-ranker",
        label="趋势突破 P3",
        group="trend",
        components=(
            ScoreComponent(factor="price_pos_20d", direction="higher", weight=P_WEIGHT),
            ScoreComponent(factor="close_to_high_20d", direction="lower", weight=P_WEIGHT),
            ScoreComponent(factor="KLEN", direction="lower", weight=K_WEIGHT),
            ScoreComponent(factor="KMID2", direction="higher", weight=K_WEIGHT),
        ),
    )

    # ── 用框架计算全部因子 + base score ──
    f = calc_amv_core_factors(lf)
    f = f.with_columns(
        ranker_score_expr(ranker).alias("_base_signal_score")
    )

    # ═══════════════════════════════════════════
    # 3b. Medium penalty（128d 滚动特征）
    # ═══════════════════════════════════════════
    W = 128

    _safe_div = lambda n, d: n / pl.when(d.abs() > 1e-12).then(d).otherwise(None)

    f = f.with_columns([
        (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("_ret1d"),
        (pl.col("close_adj") > pl.col("pre_close_adj")).alias("_upday"),
        ((pl.col("close_adj") - pl.col("open_adj")).abs()
         / pl.max_horizontal((pl.col("high_adj") - pl.col("low_adj")).abs(), pl.lit(1e-12)))
        .alias("_body_eff"),
    ]).with_columns(pl.col("_ret1d").abs().alias("_abs_ret"))

    f = f.with_columns([
        (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).alias(f"_ret_{W}d"),
        _safe_div(
            pl.col("close_adj") - pl.col("close_adj").rolling_min(W).over("code"),
            pl.col("close_adj").rolling_max(W).over("code") - pl.col("close_adj").rolling_min(W).over("code"),
        ).alias(f"_pos_{W}d"),
        _safe_div(
            (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).abs(),
            pl.col("_abs_ret").rolling_sum(W).over("code"),
        ).alias(f"_trend_eff_{W}d"),
        pl.col("_upday").rolling_mean(W).over("code").alias(f"_up_ratio_{W}d"),
        _safe_div(
            pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0,
            pl.col("_ret1d").rolling_std(W).over("code"),
        ).alias(f"_ret_vol_{W}d"),
        (_safe_div(
            pl.col("close_adj").rolling_mean(W).over("code"),
            pl.col("close_adj").rolling_mean(W).over("code").shift(20).over("code"),
        ) - 1.0).alias(f"_ma_slope_{W}d"),
        pl.col("_body_eff").rolling_mean(W).over("code").alias(f"_body_eff_{W}d"),
    ])

    # ── 截面百分位 rank ──
    _pct = lambda col, hi: (
        pl.col(col).rank("average", descending=not hi).over("date") / pl.len().over("date")
    )
    _feats = [
        (f"_ret_{W}d", True), (f"_pos_{W}d", True), (f"_trend_eff_{W}d", True),
        (f"_up_ratio_{W}d", True), (f"_ret_vol_{W}d", True), (f"_ma_slope_{W}d", True),
        (f"_body_eff_{W}d", True),
    ]
    for col, hi in _feats:
        f = f.with_columns(_pct(col, hi).alias(f"{col}_rank_pct"))

    # ── structure + quality → penalty → final score ──
    structure = (pl.col(f"_ret_{W}d_rank_pct") + pl.col(f"_pos_{W}d_rank_pct") + pl.col(f"_ma_slope_{W}d_rank_pct")) / 3.0
    quality   = (pl.col(f"_trend_eff_{W}d_rank_pct") + pl.col(f"_up_ratio_{W}d_rank_pct") + pl.col(f"_ret_vol_{W}d_rank_pct") + pl.col(f"_body_eff_{W}d_rank_pct")) / 4.0

    medium_weak = (structure.fill_null(1.0) < MEDIUM_WEAK_THRESHOLD) & (quality.fill_null(1.0) < MEDIUM_WEAK_THRESHOLD)
    s_sh = (MEDIUM_WEAK_THRESHOLD - structure.fill_null(1.0)) / MEDIUM_WEAK_THRESHOLD
    q_sh = (MEDIUM_WEAK_THRESHOLD - quality.fill_null(1.0)) / MEDIUM_WEAK_THRESHOLD
    strength = pl.when(medium_weak).then(((s_sh + q_sh) / 2.0).clip(0.0, 1.0)).otherwise(0.0)

    f = f.with_columns(
        (pl.col("_base_signal_score") - strength * MEDIUM_PENALTY).alias("_signal_score")
    )

    (f,)
    return (f,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Collect + 候选人过滤 + TopN
    """)
    return


@app.cell
def _(AMOUNT_MA20_MIN, MV_MIN, TOP_N, f, logger, pl, reader, time):
    _t0 = time.perf_counter()

    # ── 一次 collect ──
    market = f.collect(streaming=True)
    reader.close()

    t_collect = time.perf_counter() - _t0
    n_rows, n_cols = market.shape
    mem_mb = market.estimated_size() / 1024 / 1024

    # ── 候选人过滤 ──
    candidate = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= MV_MIN)
        & (pl.col("amount_ma20") >= AMOUNT_MA20_MIN)
        & pl.col("price_pos_20d").is_not_null()
        & pl.col("close_to_high_20d").is_not_null()
        & pl.col("KLEN").is_not_null()
        & pl.col("KMID2").is_not_null()
    )

    # ── 在全量 market 上 rank（用 lazy 合并两次 with_columns 为一次执行，避免中间副本）──
    # 非候选人 score 设 None → ordinal rank 把 null 排最后 → Top3 一定是候选人
    market = (
        market.lazy()
        .with_columns(
            pl.when(candidate).then(pl.col("_signal_score")).otherwise(None).alias("_csig")
        )
        .with_columns(
            pl.col("_csig").rank("ordinal", descending=True).over("date").alias("_signal_rank")
        )
        .collect(streaming=True)
    )

    signals = (
        market.filter(pl.col("_signal_rank") <= TOP_N)
        .select(
            signal_date=pl.col("date"),
            code="code",
            sleeve_id=pl.lit("trend-p3-medium"),
            score=pl.col("_csig"),
            rank=pl.col("_signal_rank").cast(pl.UInt32),
        )
        .sort(["signal_date", "rank", "code"])
    )

    n_signals = signals.height
    n_dates = signals.select("signal_date").n_unique()

    # ── 裁到导出列，释放因子列内存（同 cell 内重定义，旧 market 立即释放）──
    EXPORT_COLS = [
        "date", "code",
        "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
        "open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw",
        "is_bull_regime", "amv_mechanical_regime",
        "market_cap_100m", "amount_ma20",
    ]
    market = market.select(EXPORT_COLS)

    logger.info(
        f"collect={t_collect:.1f}s, market={n_rows:,}行×{n_cols}列, "
        f"内存≈{mem_mb:.0f}MB, 信号={n_signals}条/{n_dates}天"
    )

    (candidate, market, mem_mb, n_cols, n_dates, n_rows, n_signals, signals, t_collect)
    return market, mem_mb, n_signals, signals, t_collect


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 写出 signal.parquet
    """)
    return


@app.cell
def _(ROOT, logger, market, pl, signals, time):
    _t0 = time.perf_counter()

    # ── 执行日偏移 → 全量面板（market 已是瘦版 18 列）──
    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date")).drop_nulls()

    execution_signals = (
        signals.join(next_dates, left_on="signal_date", right_on="date")
        .select("execution_date", "code", "signal_date", "sleeve_id", "score", "rank")
        .rename({"execution_date": "date"})
    )

    export = (
        market.join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            is_signal=pl.col("signal_date").is_not_null(),
            score=pl.col("score").fill_null(0.0),
            rank=pl.col("rank").fill_null(9999).cast(pl.UInt32),
            sleeve_id=pl.col("sleeve_id").fill_null(""),
        )
        .sort(["date", "code"])
    )

    out_dir = ROOT / "artifacts" / "_demo_trend_p3_medium"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signal.parquet"
    export.write_parquet(out_path)

    t_write = time.perf_counter() - _t0
    logger.info(f"写出 {out_path} ({export.height:,}行), {t_write:.1f}s")
    return (t_write,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 总览
    """)
    return


@app.cell
def _(mem_mb, mo, n_signals, signals, t_collect, t_load, t_write):
    total = t_load + t_collect + t_write
    mo.md(f"""
    | 阶段 | 耗时 |
    |------|------|
    | LazyFrame 构造 | {t_load:.1f}s |
    | collect (含全部因子+penalty) | {t_collect:.1f}s |
    | 写出 parquet | {t_write:.1f}s |
    | **总耗时** | **{total:.1f}s** |
    | **内存峰值** | **≈{mem_mb:.0f} MB** |

    **信号**: {n_signals} 条

    ### 信号预览（最近 10 条）
    {mo.as_html(signals.tail(10).to_pandas().to_html(index=False))}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
