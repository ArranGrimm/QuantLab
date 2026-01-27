import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import marimo as mo
    import polars as pl
    import duckdb
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import date
    from utils import load_daily_data_single, get_adj_factor_frame
    return date, duckdb, go, load_daily_data_single, make_subplots, mo, os, pl


@app.cell
def _(mo):
    # ==============================================================================
    # 1. UI 控制区
    # ==============================================================================
    mo.md(r"""
    # 📈 Ztalk 快速复盘驾驶舱
    无需加载全量数据，输入代码直连 数据库。
    """)
    return


@app.cell
def _(mo):
    # 输入框：股票代码
    input_code = mo.ui.text(
        value="sh.600570", 
        label="🔍 股票代码 (例如: sh.600570):",
        full_width=True
    )

    # 滑块：查看最近多少天
    input_days = mo.ui.slider(
        start=50, 
        stop=800, 
        value=500, 
        step=10, 
        label="🕒 查看最近 N 天:",
        full_width=True
    )

    # 布局
    mo.vstack([input_code, input_days])
    return input_code, input_days


@app.cell
def _(
    duckdb,
    go,
    input_code,
    input_days,
    load_daily_data_single,
    make_subplots,
    mo,
    pl,
):
    # ==============================================================================
    # 2. 数据读取与计算核心
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    current_code = input_code.value.strip()
    lookback = input_days.value
    # 1. 极速读取数据库
    df = load_daily_data_single(conn, current_code)

    # 2. 特征计算 (计算涨跌幅 + Ztalk 核心指标)
    df = df.sort("date").with_columns([
        # 计算涨跌幅 (PctChange)
        pl.col("close_adj").shift(1).alias("prev_close")
    ]).with_columns([
        ((pl.col("close_adj") - pl.col("prev_close")) / pl.col("prev_close") * 100).alias("pct_change")
    ]).with_columns([
        # 白线 WL
        pl.col("close_adj").ewm_mean(span=10, adjust=False).ewm_mean(span=10, adjust=False).alias("WL"),
        # 黄线 YL
        ((pl.col("close_adj").rolling_mean(14) + 
          pl.col("close_adj").rolling_mean(28) + 
          pl.col("close_adj").rolling_mean(57) + 
          pl.col("close_adj").rolling_mean(114)) / 4).alias("YL"),
        # KDJ-J值
        (pl.col("high_adj").rolling_max(9) - pl.col("low_adj").rolling_min(9)).alias("kdj_den"),
    ]).with_columns([
        pl.when(pl.col("kdj_den") == 0).then(50.0).otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9)) / pl.col("kdj_den") * 100).alias("rsv")
    ]).with_columns([
        pl.col("rsv").ewm_mean(com=2, adjust=False).alias("K")
    ]).with_columns([
        pl.col("K").ewm_mean(com=2, adjust=False).alias("D")
    ]).with_columns([
        (3 * pl.col("K") - 2 * pl.col("D")).alias("J")
    ])

    # 3. 数据截取与类型转换
    # 🔥 关键点：将 date 转为 String，这是 Plotly 去除周末空缺的最简单方法
    df_plot = df.tail(lookback).with_columns(pl.col("date").cast(pl.String))

    # 转 Pandas 绘图
    pdf = df_plot.to_pandas()

    # ==============================================================================
    # 3. Plotly 绘图 (专业护眼配色)
    # ==============================================================================

    # 🔥 配色定义区 (方便随时改)
    COLOR_UP = '#F56C6C'     # 柔和红
    COLOR_DOWN = '#67C23A'   # 柔和绿 (松石绿)
    COLOR_BG = '#F7F9FC'     # 淡蓝灰背景
    COLOR_GRID = '#E4E7ED'   # 浅灰网格线
    COLOR_WL = '#409EFF'     # 白线 (深天蓝)
    COLOR_YL = '#E6A23C'     # 黄线 (橘黄)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    hover_text = [
        f"日期: {date}<br>"
        f"开盘: {open_p:.2f}<br>"
        f"最高: {high_p:.2f}<br>"
        f"最低: {low_p:.2f}<br>"
        f"收盘: {close:.2f}<br>"
        f"涨幅: {pct:+.2f}%<br>"
        f"J值: {j:.2f}"
        for date, pct, open_p, high_p, low_p, close, j in zip(
            pdf["date"], 
            pdf["pct_change"], 
            pdf["open_adj"],  # 新增
            pdf["high_adj"],  # 新增
            pdf["low_adj"],   # 新增
            pdf["close_adj"], 
            pdf["J"]
        )
    ]

    # A. K线图
    fig.add_trace(go.Candlestick(
        x=pdf["date"],
        open=pdf["open_adj"],
        high=pdf["high_adj"],
        low=pdf["low_adj"],
        close=pdf["close_adj"],
        name="K线",
        text=hover_text,
        hoverinfo="text",
        increasing_line_color=COLOR_UP,   # 应用柔和红
        decreasing_line_color=COLOR_DOWN  # 应用柔和绿
    ), secondary_y=False)

    # B. 成交量
    vol_colors = [
        COLOR_UP if close >= open else COLOR_DOWN 
        for close, open in zip(pdf["close_adj"], pdf["open_adj"])
    ]

    fig.add_trace(go.Bar(
        x=pdf["date"],
        y=pdf["volume"],
        name="成交量",
        marker_color=vol_colors,
        opacity=0.4  # 透明度稍微提高一点，更不抢眼
    ), secondary_y=True)

    # C. 黄白线
    fig.add_trace(go.Scatter(x=pdf["date"], y=pdf["WL"], mode='lines', name='WL(白)', line=dict(color=COLOR_WL, width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=pdf["date"], y=pdf["YL"], mode='lines', name='YL(黄)', line=dict(color=COLOR_YL, width=2)), secondary_y=False)

    # D. B1 信号
    b1_mask = (pdf["J"] <= 13) & (pdf["close_adj"] > pdf["YL"] * 0.97)
    if b1_mask.any():
        b1_data = pdf[b1_mask]
        fig.add_trace(go.Scatter(
            x=b1_data["date"],
            y=b1_data["low_adj"] * 0.98,
            mode='markers',
            name='B1潜在点',
            marker=dict(symbol='triangle-up', size=10, color='#909399'), # 灰色标记，低调点
            hoverinfo='skip'
        ), secondary_y=False)

    # --- 布局核心调整 ---
    fig.update_layout(
        title=dict(
            text=f"<b>{current_code}</b> 走势图 (Data to {pdf['date'].iloc[-1]})",
            font=dict(size=20, color='#303133')
        ),
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",

        # 🔥 背景色修改
        paper_bgcolor=COLOR_BG,  # 外框背景
        plot_bgcolor=COLOR_BG,   # 图表背景

        xaxis=dict(
            type='category', 
            nticks=20, 
            tickangle=-45,
            showgrid=True,
            gridcolor=COLOR_GRID, # 网格线颜色
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            gridwidth=1
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)'),
        margin=dict(l=50, r=50, t=60, b=50),
    )

    # 坐标轴设置
    max_vol = pdf["volume"].max()
    fig.update_yaxes(title="价格", secondary_y=False, gridcolor=COLOR_GRID)
    fig.update_yaxes(
        title="", 
        secondary_y=True, 
        showgrid=False, 
        showticklabels=False, 
        range=[0, max_vol * 5]
    )

    # 最终显示
    result_view = mo.ui.plotly(fig)

    # 显示结果
    result_view
    return (df,)


@app.cell
def _(date, mo):
    # 日期选择器 (默认为你刚才的案例时间)
    ui_buy_date = mo.ui.date(
        value=date(2025, 3, 28), 
        label="📅 买入日期 (T日)"
    )

    ui_sell_date = mo.ui.date(
        value=date(2025, 4, 3), 
        label="📅 卖出日期 (T+N日)"
    )

    # 将 UI 横向排列，美观一点
    mo.hstack([ui_buy_date, ui_sell_date], justify="start")
    return ui_buy_date, ui_sell_date


@app.cell
def _(df, file_path, input_code, mo, os, pl, ui_buy_date, ui_sell_date):
    code_val = input_code.value.strip()
    buy_date_val = ui_buy_date.value
    sell_date_val = ui_sell_date.value

    profit_view = None

    if not os.path.exists(file_path):
        profit_view = mo.callout(f"❌ 找不到文件: {file_path}", kind="danger")
    else:
        try:
            # 提取买卖数据
            buy_row = df.filter(pl.col("date") == buy_date_val)
            sell_row = df.filter(pl.col("date") == sell_date_val)

            # 逻辑判定
            if buy_row.height == 0:
                profit_view = mo.callout(f"⚠️ 买入日期 {buy_date_val} 无数据 (可能是停牌或非交易日)", kind="warn")
            elif sell_row.height == 0:
                profit_view = mo.callout(f"⚠️ 卖出日期 {sell_date_val} 无数据 (可能是停牌或非交易日)", kind="warn")
            else:
                # 按照你的逻辑：买入按开盘价 (Open)，卖出按收盘价 (Close)
                buy_price = buy_row["open_adj"][0]
                sell_price = sell_row["close_adj"][0]

                # 收益率计算
                profit_amount = sell_price - buy_price
                profit_pct = (profit_amount / buy_price) * 100

                # 持仓天数计算 (自然交易日)
                hold_days = df.filter(
                    (pl.col("date") >= buy_date_val) & 
                    (pl.col("date") <= sell_date_val)
                ).height

                # 判定盈亏颜色
                color_style = "green" if profit_pct > 0 else "red" # 这里的颜色逻辑：红涨绿跌? 还是国际惯例? 
                # A股习惯：红是赚，绿是亏。这里按 A 股习惯。
                txt_color = "red" if profit_pct > 0 else "green"
                emoji = "🚀" if profit_pct > 0 else "😭"

                # 生成漂亮的 Markdown 报告
                md_content = f"""
                ## {emoji} 交易复盘: {code_val}

                | 指标 | 数值 | 说明 |
                | :--- | :--- | :--- |
                | **买入** | `{buy_date_val}` | 价格: **{buy_price:.2f}** (Open) |
                | **卖出** | `{sell_date_val}` | 价格: **{sell_price:.2f}** (Close) |
                | **持仓** | **{hold_days}** 天 | 交易日数量 |
                | **每股盈亏** | {profit_amount:+.2f} 元 | 差价 |
                | **最终收益** | <span style="color:{txt_color}; font-size:24px; font-weight:bold">{profit_pct:+.2f}%</span> | {emoji} |
                """

                profit_view = mo.md(md_content)

        except Exception as e:
            profit_view = mo.callout(f"发生错误: {str(e)}", kind="danger")


    profit_view
    return


@app.cell
def _():
    import datetime
    datetime.datetime.now().weekday()
    return


if __name__ == "__main__":
    app.run()
