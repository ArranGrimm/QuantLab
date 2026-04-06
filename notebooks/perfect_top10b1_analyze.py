import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime
    from utils import load_daily_data_full

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    PERFECT_CASES_CONFIG = [
        # 教科书案例, 10大完美案例
        {"code": "sh.688799", "date": "2025-05-12", "name": "华纳药厂(标准)"},
        {"code": "sz.300689", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
        {"code": "sh.600601", "date": "2025-07-23", "name": "方正科技(蓄势)"},
        {"code": "sh.688321", "date": "2025-06-20", "name": "微芯生物(双底)"},
        {"code": "sz.002940", "date": "2025-07-11", "name": "昂利康(压轴)"},
        {"code": "sz.301076", "date": "2025-08-01", "name": "新瀚新材(激进)"},
        {"code": "sh.600184", "date": "2025-07-10", "name": "光电股份(回踩)"},
        {"code": "sz.002074", "date": "2025-08-01", "name": "国轩高科(趋势)"},
        {"code": "sh.605378", "date": "2025-07-31", "name": "野马电池(突破)"},
        {"code": "sh.600366", "date": "2025-08-06", "name": "宁波韵升(反包)"},
        # 以下是自己发现的案例
        {"code": "sz.000547", "date": "2025-11-13", "name": "航天发展(标准)"}
    ]

    print("🚀 [Step 1] 加载原始行情数据...")
    perfect_case_list = [item.get("code") for item in PERFECT_CASES_CONFIG]
    q_full = load_daily_data_full(conn, perfect_case_list)
    print(f"✅ 加载完成, 共 {q_full.collect().height} 行")
    return PERFECT_CASES_CONFIG, go, make_subplots, mo, pl, q_full


@app.cell
def _(PERFECT_CASES_CONFIG, pl, q_full):
    """Phase 1: 计算 Running Weekly & Monthly MACD"""

    df_sorted = q_full.sort(["code", "date"])

    df_with_time = df_sorted.with_columns([
        pl.col("date").dt.truncate("1w").alias("week_start"),
        pl.col("date").dt.truncate("1mo").alias("month_start"),
    ])

    # --- 周线 MACD ---
    df_weekly = (
        df_with_time.sort(["code", "date"])
        .group_by(["code", "week_start"])
        .agg(pl.col("close_adj").last().alias("weekly_close"))
        .sort(["code", "week_start"])
        .with_columns([
            pl.col("weekly_close").ewm_mean(span=12, adjust=False).over("code").alias("w_ema12"),
            pl.col("weekly_close").ewm_mean(span=26, adjust=False).over("code").alias("w_ema26"),
        ])
        .with_columns((pl.col("w_ema12") - pl.col("w_ema26")).alias("w_dif"))
        .with_columns(pl.col("w_dif").ewm_mean(span=9, adjust=False).over("code").alias("w_dea"))
    )

    df_weekly_prev = df_weekly.select([
        "code", "week_start",
        pl.col("w_ema12").shift(1).over("code").alias("prev_w_ema12"),
        pl.col("w_ema26").shift(1).over("code").alias("prev_w_ema26"),
        pl.col("w_dea").shift(1).over("code").alias("prev_w_dea"),
        (2 * (pl.col("w_dif").shift(1).over("code") - pl.col("w_dea").shift(1).over("code"))).alias("prev_w_hist"),
    ])

    # --- 月线 MACD ---
    df_monthly = (
        df_with_time.sort(["code", "date"])
        .group_by(["code", "month_start"])
        .agg(pl.col("close_adj").last().alias("monthly_close"))
        .sort(["code", "month_start"])
        .with_columns([
            pl.col("monthly_close").ewm_mean(span=12, adjust=False).over("code").alias("m_ema12"),
            pl.col("monthly_close").ewm_mean(span=26, adjust=False).over("code").alias("m_ema26"),
        ])
        .with_columns((pl.col("m_ema12") - pl.col("m_ema26")).alias("m_dif"))
        .with_columns(pl.col("m_dif").ewm_mean(span=9, adjust=False).over("code").alias("m_dea"))
    )

    df_monthly_prev = df_monthly.select([
        "code", "month_start",
        pl.col("m_ema12").shift(1).over("code").alias("prev_m_ema12"),
        pl.col("m_ema26").shift(1).over("code").alias("prev_m_ema26"),
        pl.col("m_dea").shift(1).over("code").alias("prev_m_dea"),
    ])

    # --- Running MACD ---
    a12, a26, a9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0

    df_daily = (
        df_with_time
        .join(df_weekly_prev, on=["code", "week_start"], how="left")
        .join(df_monthly_prev, on=["code", "month_start"], how="left")
        .with_columns([
            (a12 * pl.col("close_adj") + (1 - a12) * pl.col("prev_w_ema12")).alias("rw_ema12"),
            (a26 * pl.col("close_adj") + (1 - a26) * pl.col("prev_w_ema26")).alias("rw_ema26"),
            (a12 * pl.col("close_adj") + (1 - a12) * pl.col("prev_m_ema12")).alias("rm_ema12"),
            (a26 * pl.col("close_adj") + (1 - a26) * pl.col("prev_m_ema26")).alias("rm_ema26"),
        ])
        .with_columns([
            (pl.col("rw_ema12") - pl.col("rw_ema26")).alias("rw_dif"),
            (pl.col("rm_ema12") - pl.col("rm_ema26")).alias("rm_dif"),
        ])
        .with_columns([
            (a9 * pl.col("rw_dif") + (1 - a9) * pl.col("prev_w_dea")).alias("rw_dea"),
            (a9 * pl.col("rm_dif") + (1 - a9) * pl.col("prev_m_dea")).alias("rm_dea"),
        ])
        .with_columns([
            (2 * (pl.col("rw_dif") - pl.col("rw_dea"))).alias("rw_hist"),
            (2 * (pl.col("rm_dif") - pl.col("rm_dea"))).alias("rm_hist"),
        ])
        .with_columns([
            # 红柱变化率: 当前周 running hist vs 上一完整周 hist
            pl.when(pl.col("prev_w_hist").abs() > 0.001)
              .then((pl.col("rw_hist") - pl.col("prev_w_hist")) / pl.col("prev_w_hist").abs())
              .otherwise(None)
              .alias("w_hist_chg_pct"),
        ])
        .collect()
    )

    # --- 提取完美案例触发日的特征 ---
    cases_df = pl.DataFrame(PERFECT_CASES_CONFIG).with_columns(
        pl.col("date").str.to_date().alias("date")
    )

    features = df_daily.join(cases_df, on=["code", "date"], how="inner").select([
        "name", "code", "date",
        "rw_dif", "rw_dea", "rw_hist",
        "prev_w_hist", "w_hist_chg_pct",
        "rm_dif", "rm_dea", "rm_hist",
    ])

    print(f"✅ 成功匹配 {features.height}/{cases_df.height} 个完美案例")
    return df_daily, features


@app.cell
def _(mo):
    """Phase 2: 完美案例 B1 触发日的周线/月线 MACD 特征总览"""
    mo.md("## 📊 完美案例 B1 触发日 — 周线/月线 MACD 特征")
    return


@app.cell
def _(features, mo, pl):
    display_df = features.select([
        "name",
        pl.col("rw_dif").round(4).alias("周DIF"),
        pl.col("rw_dea").round(4).alias("周DEA"),
        pl.col("rw_hist").round(4).alias("周HIST(红柱)"),
        pl.col("prev_w_hist").round(4).alias("上周HIST"),
        pl.col("w_hist_chg_pct").round(2).alias("HIST变化%"),
        pl.col("rm_dif").round(4).alias("月DIF"),
        pl.col("rm_hist").round(4).alias("月HIST"),
        # 标记
        (pl.col("rw_hist") > 0).alias("周红柱?"),
        (pl.col("rw_dif") > 0).alias("周水上?"),
        (pl.col("rm_dif") > 0).alias("月水上?"),
    ])
    mo.ui.table(display_df.to_pandas())
    return


@app.cell
def _(features, mo, pl):
    """Phase 3: 统计摘要 — 找阈值"""
    stats = features.select([
        pl.col("rw_dif").alias("周DIF"),
        pl.col("rw_hist").alias("周HIST"),
        pl.col("prev_w_hist").alias("上周HIST"),
        pl.col("w_hist_chg_pct").alias("HIST变化%"),
        pl.col("rm_dif").alias("月DIF"),
        pl.col("rm_hist").alias("月HIST"),
    ]).describe()

    all_weekly_red = (features["rw_hist"] > 0).all()
    all_weekly_above = (features["rw_dif"] > 0).all()
    all_monthly_above = (features["rm_dif"] > 0).all()

    summary = f"""
    ## 📈 统计摘要

    | 检查项 | 结果 |
    |--------|------|
    | 全部周线红柱 (rw_hist > 0) | {'✅ 是' if all_weekly_red else '❌ 否'} |
    | 全部周线水上 (rw_dif > 0) | {'✅ 是' if all_weekly_above else '❌ 否'} |
    | 全部月线水上 (rm_dif > 0) | {'✅ 是' if all_monthly_above else '❌ 否'} |

    ### 关键指标分布

    | 指标 | 最小值 | 中位数 | 最大值 |
    |------|--------|--------|--------|
    | 周 DIF | {features['rw_dif'].min():.4f} | {features['rw_dif'].median():.4f} | {features['rw_dif'].max():.4f} |
    | 周 HIST | {features['rw_hist'].min():.4f} | {features['rw_hist'].median():.4f} | {features['rw_hist'].max():.4f} |
    | 上周 HIST | {features['prev_w_hist'].min():.4f} | {features['prev_w_hist'].median():.4f} | {features['prev_w_hist'].max():.4f} |
    | HIST 变化% | {features['w_hist_chg_pct'].drop_nulls().min():.2f} | {features['w_hist_chg_pct'].drop_nulls().median():.2f} | {features['w_hist_chg_pct'].drop_nulls().max():.2f} |
    | 月 DIF | {features['rm_dif'].min():.4f} | {features['rm_dif'].median():.4f} | {features['rm_dif'].max():.4f} |
    | 月 HIST | {features['rm_hist'].min():.4f} | {features['rm_hist'].median():.4f} | {features['rm_hist'].max():.4f} |

    > **阈值建议**: 基于 min 值可以推导出过滤条件的下限。
    > 例如: 如果所有完美案例的周 HIST 最小值为 X, 则 `rw_hist >= X` 就是一个有数据支撑的阈值。
    """
    mo.md(summary)
    return


@app.cell
def _(features, go, make_subplots, mo):
    """Phase 4: 可视化 — 每个案例的周线MACD指标"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["周线 DIF (水上?)", "周线 HIST (红柱大小)", "月线 DIF (水上?)", "HIST 变化率%"],
        vertical_spacing=0.12,
    )

    names = features["name"].to_list()

    fig.add_trace(go.Bar(
        x=names, y=features["rw_dif"].to_list(),
        marker_color=["#ef5350" if v > 0 else "#26a69a" for v in features["rw_dif"].to_list()],
        name="周DIF"
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Bar(
        x=names, y=features["rw_hist"].to_list(),
        marker_color=["#ef5350" if v > 0 else "#26a69a" for v in features["rw_hist"].to_list()],
        name="周HIST"
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.add_trace(go.Bar(
        x=names, y=features["rm_dif"].to_list(),
        marker_color=["#ef5350" if v > 0 else "#26a69a" for v in features["rm_dif"].to_list()],
        name="月DIF"
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    chg_vals = features["w_hist_chg_pct"].fill_null(0).to_list()
    fig.add_trace(go.Bar(
        x=names, y=chg_vals,
        marker_color=["#ef5350" if v > 0 else "#26a69a" for v in chg_vals],
        name="HIST变化%"
    ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_layout(
        height=700, showlegend=False,
        title_text="完美案例 B1 触发日 — 周线/月线 MACD 特征画像",
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=45)

    mo.ui.plotly(fig)
    return


@app.cell
def _(df_daily, features, go, make_subplots, mo, pl):
    """Phase 5: 每个案例触发日前后的周线 MACD 走势 (上下文视图)"""
    case_rows = features.select(["code", "date"]).to_dicts()
    n_cases = len(case_rows)
    cols = 3
    rows = (n_cases + cols - 1) // cols

    names_map = {r["code"]: r["name"] for r in features.select(["code", "name"]).to_dicts()}

    fig2 = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[names_map.get(c["code"], c["code"]) for c in case_rows],
        vertical_spacing=0.08, horizontal_spacing=0.06,
    )

    for i, case in enumerate(case_rows):
        r, c = i // cols + 1, i % cols + 1
        signal_date = case["date"]

        # 取触发日前后 60 天的数据
        ctx = df_daily.filter(
            (pl.col("code") == case["code"]) &
            (pl.col("date") >= signal_date - pl.duration(days=120)) &
            (pl.col("date") <= signal_date + pl.duration(days=30))
        ).sort("date")

        if ctx.height == 0:
            continue

        dates = ctx["date"].to_list()
        hist_vals = ctx["rw_hist"].to_list()

        fig2.add_trace(go.Bar(
            x=dates, y=hist_vals,
            marker_color=["#ef5350" if v and v > 0 else "#26a69a" for v in hist_vals],
            showlegend=False,
        ), row=r, col=c)

        fig2.add_vline(x=signal_date, line_dash="dash", line_color="blue", line_width=1.5, row=r, col=c)

    fig2.update_layout(
        height=300 * rows,
        title_text="各案例 B1 触发日前后 — Running Weekly MACD HIST 走势 (蓝线=触发日)",
        template="plotly_white",
        showlegend=False,
    )
    fig2.update_xaxes(tickangle=45, tickfont_size=8)

    mo.ui.plotly(fig2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
