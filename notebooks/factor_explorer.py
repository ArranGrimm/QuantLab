"""Factor Explorer — 通用因子探索 notebook。

流程：
1. 从 TDX 加载日线数据
2. 定义因子表达式（Polars lazy）
3. 计算 Rank IC（日截面 Spearman 相关系数 × 未来 N 日收益）
4. 可视化 Rank IC 时序 + 分布
5. 可选：简单多空分组收益模拟
"""
import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Factor Explorer

    通用因子探索工作台。改 `FACTOR_EXPR` 即可测试任意新因子。
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from loguru import logger

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from utils.data_source import TdxDailyReader, DataSourceSettings

    pl.Config.set_tbl_rows(10)
    return DataSourceSettings, Path, ROOT, TdxDailyReader, go, logger, mo, pl, sys


@app.cell
def _(mo):
    mo.md(r"""## 1. 配置""")
    return


@app.cell
def _(DataSourceSettings, ROOT):
    # ── 时间范围 ──
    START_DATE = "2019-01-01"
    END_DATE = "2026-06-03"

    # ── 数据源 ──
    TDX_DB = ROOT.parent / "QuantData" / "Ashare" / "tdx.db"
    data_source = DataSourceSettings(provider="tdx", tdx_db=TDX_DB, start_date=START_DATE)

    # ── 因子参数 ──
    FORWARD_RETURN = 5       # 前向收益天数（1/5/10/20）
    FACTOR_NAME = "quality_momentum_60d"  # 用于图表标签

    (START_DATE, END_DATE, FORWARD_RETURN, FACTOR_NAME, data_source)
    return END_DATE, FACTOR_NAME, FORWARD_RETURN, START_DATE, data_source


@app.cell
def _(mo):
    mo.md(r"""## 2. 定义因子表达式

    修改 `FACTOR_EXPR` 即可测试不同因子。
    表达式在 **全量截面** 上求值，返回一个数值列。
    """)
    return


@app.cell
def _(pl):
    # ═══════════════════════════════════════════════════════════
    # 改这里！定义你要测试的因子
    # ═══════════════════════════════════════════════════════════

    # 示例 1: 价格位置（默认）
    # FACTOR_EXPR = (
    #     (pl.col("close_adj") - pl.col("close_adj").rolling_min(20).over("code"))
    #     / pl.max_horizontal(
    #         pl.col("close_adj").rolling_max(20).over("code")
    #         - pl.col("close_adj").rolling_min(20).over("code"),
    #         pl.lit(0.01),
    #     )
    # ).alias("factor_value")

    # 高质量动量: r_60 - k * sigma^2
    r_60 = pl.col("close_adj") / pl.col("close_adj").shift(60).over("code") - 1.0
    sigma_60 = pl.col("close_adj").pct_change().rolling_std(60).over("code")
    FACTOR_EXPR = (r_60 - 3000 * sigma_60.pow(2)).alias("factor_value")

    # 示例 2: 高质量动量（取消注释测试）
    # r_60 = pl.col("close_adj") / pl.col("close_adj").shift(60).over("code") - 1
    # sigma_60 = pl.col("close_adj").pct_change().rolling_std(60).over("code")
    # FACTOR_EXPR = (r_60 - 3000 * sigma_60.pow(2)).alias("factor_value")

    # 示例 3: 上影线压力（取消注释测试）
    # upper_shadow = pl.col("high_adj") - pl.max_horizontal(pl.col("close_adj"), pl.col("open_adj"))
    # range_ = pl.col("high_adj") - pl.col("low_adj")
    # FACTOR_EXPR = (upper_shadow / pl.max_horizontal(range_, pl.lit(1e-12))).alias("factor_value")

    (FACTOR_EXPR,)
    return (FACTOR_EXPR,)


@app.cell
def _(mo):
    mo.md(r"""## 3. 加载数据 + 计算因子 + 前向收益""")
    return


@app.cell
def _(END_DATE, FACTOR_EXPR, FORWARD_RETURN, START_DATE, TdxDailyReader, data_source, logger, pl, time):
    reader = TdxDailyReader(data_source)
    _t0 = time.perf_counter()

    # ── 基础数据 ──
    lf = reader.load_daily_full().filter(
        pl.col("date").is_between(
            pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
            pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
        )
    )

    # ── 因子 + 前向收益 ──
    factor_col = FACTOR_EXPR.meta.output_name() if hasattr(FACTOR_EXPR, 'meta') else "factor_value"

    lf = (
        lf.sort(["code", "date"])
        .with_columns([
            FACTOR_EXPR,
            # 前向 N 日收益
            (pl.col("close_adj").shift(-FORWARD_RETURN).over("code")
             / pl.col("close_adj") - 1.0).alias(f"_fwd_ret_{FORWARD_RETURN}d"),
        ])
        .select(["code", "date", "close_adj", "market_cap_100m", "amount",
                 factor_col, f"_fwd_ret_{FORWARD_RETURN}d"])
    )

    df = lf.collect()
    reader.close()

    _elapsed = time.perf_counter() - _t0
    logger.info(f"加载完成: {df.height:,} 行, {df['code'].n_unique():,} 只股票, {_elapsed:.1f}s")

    (df, factor_col)
    return df, factor_col


@app.cell
def _(mo):
    mo.md(r"""## 4. Rank IC 分析""")
    return


@app.cell
def _(
    FACTOR_NAME,
    FORWARD_RETURN,
    df,
    factor_col,
    go,
    mo,
    pl,
):
    # ── 日截面 Rank IC ──
    ic = (
        df.filter(
            pl.col(factor_col).is_not_null() & pl.col(factor_col).is_not_nan(),
            pl.col(f"_fwd_ret_{FORWARD_RETURN}d").is_not_null(),
        )
        .group_by("date")
        .agg(
            # Rank IC = Spearman correlation
            pl.corr(
                pl.col(factor_col).rank("average"),
                pl.col(f"_fwd_ret_{FORWARD_RETURN}d").rank("average"),
            ).alias("rank_ic"),
            pl.len().alias("n"),
        )
        .sort("date")
        .filter(pl.col("n") >= 100)  # 至少 100 只股票
    )

    mean_ic = ic["rank_ic"].mean()
    ic_ir = mean_ic / ic["rank_ic"].std() if ic["rank_ic"].std() > 0 else 0
    positive_ratio = (ic["rank_ic"] > 0).sum() / ic.height

    # ── 累计 Rank IC ──
    ic = ic.with_columns(pl.col("rank_ic").cum_sum().alias("cum_ic"))

    # ── 图表 ──
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Scatter(
        x=ic["date"], y=ic["cum_ic"],
        mode="lines", name="Cumulative Rank IC",
        line=dict(color="steelblue", width=2),
    ))
    fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ic.update_layout(
        title=f"{FACTOR_NAME} — Cumulative Rank IC (fwd {FORWARD_RETURN}d)",
        xaxis_title="Date", yaxis_title="Cumulative Rank IC",
        height=400, template="plotly_white",
    )

    # ── 月度 Rank IC 热力 ──
    ic_monthly = ic.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    ic_pivot = (
        ic_monthly.group_by(["year", "month"])
        .agg(pl.col("rank_ic").mean().alias("avg_ic"))
        .sort(["year", "month"])
    )

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=ic_pivot["avg_ic"].to_list(),
        x=[f"{m:02d}" for m in ic_pivot["month"].unique().sort().to_list()],
        y=[str(y) for y in ic_pivot["year"].unique().sort().to_list()],
        colorscale="RdBu", zmid=0,
    ))
    fig_heatmap.update_layout(
        title=f"{FACTOR_NAME} — Monthly Mean Rank IC",
        height=300, template="plotly_white",
    )

    mo.md(f"""
    | 指标 | 值 |
    |------|-----|
    | Mean Rank IC | {mean_ic:.4f} |
    | IC IR (IC / std) | {ic_ir:.2f} |
    | Positive Ratio | {positive_ratio:.1%} |
    | 样本天数 | {ic.height} |
    """)
    return fig_heatmap, fig_ic, ic, ic_ir, mean_ic


@app.cell
def _(fig_heatmap, fig_ic, mo):
    mo.ui.plotly(fig_ic)
    return


@app.cell
def _(fig_heatmap, mo):
    mo.ui.plotly(fig_heatmap)
    return


@app.cell
def _(mo):
    mo.md(r"""## 5. 分组收益模拟

    按因子值分 5 组，等权持有 N 天，看各组累计收益。
    """)
    return


@app.cell
def _(
    FACTOR_NAME,
    FORWARD_RETURN,
    df,
    factor_col,
    go,
    mo,
    pl,
):
    # ── 每日按因子值分 5 组 ──
    grouped = (
        df.filter(
            pl.col(factor_col).is_not_null() & pl.col(factor_col).is_not_nan(),
            pl.col(f"_fwd_ret_{FORWARD_RETURN}d").is_not_null(),
        )
        .with_columns(
            # 截面分 5 组（1=最低, 5=最高）
            (pl.col(factor_col).rank("average").over("date")
             / pl.len().over("date") * 5).ceil().cast(pl.Int32).alias("_group"),
        )
    )

    group_returns = (
        grouped.group_by(["date", "_group"])
        .agg(pl.col(f"_fwd_ret_{FORWARD_RETURN}d").mean().alias("group_ret"))
        .sort(["date", "_group"])
    )

    cum_returns = group_returns.with_columns(
        (1 + pl.col("group_ret")).cum_prod().over("_group").alias("cum_ret")
    )

    fig_groups = go.Figure()
    colors = ["red", "orange", "gray", "lightblue", "steelblue"]
    for g in range(1, 6):
        g_data = cum_returns.filter(pl.col("_group") == g)
        fig_groups.add_trace(go.Scatter(
            x=g_data["date"], y=g_data["cum_ret"],
            mode="lines", name=f"Q{g}",
            line=dict(color=colors[g - 1], width=1.5),
        ))

    # 多空：Q5 - Q1
    long_short = (
        group_returns.pivot(values="group_ret", index="date", columns="_group")
        .with_columns(((pl.col("5") - pl.col("1"))).alias("ls_ret"))
        .with_columns((1 + pl.col("ls_ret")).cum_prod().alias("ls_cum"))
    )
    fig_groups.add_trace(go.Scatter(
        x=long_short["date"], y=long_short["ls_cum"],
        mode="lines", name="Q5-Q1 (long-short)",
        line=dict(color="black", width=2, dash="dash"),
    ))

    fig_groups.update_layout(
        title=f"{FACTOR_NAME} — Quintile Cumulative Returns (fwd {FORWARD_RETURN}d, equal-weight)",
        yaxis_title="Cumulative Return", xaxis_title="Date",
        height=450, template="plotly_white",
    )

    print(f"分组收益模拟完成，{group_returns.height:,} 行")
    return fig_groups, group_returns, long_short


@app.cell
def _(fig_groups, mo):
    mo.ui.plotly(fig_groups)
    return
