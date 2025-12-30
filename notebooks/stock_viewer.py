import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import date
    import os
    return date, go, make_subplots, mo, os, pl


@app.cell
def _(mo):
    # ==============================================================================
    # 1. UI 控制区
    # ==============================================================================
    mo.md(r"""
    # 📈 Ztalk 快速复盘驾驶舱
    无需加载全量数据，输入代码直连 Parquet 文件。
    """)
    return


@app.cell
def _(mo):
    # 输入框：股票代码
    input_code = mo.ui.text(
        value="600519_SH", 
        label="🔍 股票代码 (例如: 000001_SZ):",
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
def _(go, input_code, input_days, make_subplots, mo, os, pl):
    # ==============================================================================
    # 2. 数据读取与计算核心
    # ==============================================================================

    # 配置数据路径 (请根据你的实际路径修改)
    DATA_PATH = r"../QuantData/Ashare/stock_day_adj"

    current_code = input_code.value.strip()
    lookback = input_days.value

    # 构造文件路径
    file_path = os.path.join(DATA_PATH, f"{current_code}.parquet")

    if not os.path.exists(file_path):
        # 如果文件不存在，显示错误提示
        result_view = mo.callout(
            f"❌ 找不到文件: {file_path}\n请确认代码格式是否为 000001_SZ 或 600519_SH", 
            kind="danger"
        )
    else:
        # 1. 极速读取单文件
        df = pl.read_parquet(file_path)
        df = df.with_columns([
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            ])
        # 2. 现场计算 B1 因子 (只为这张图计算)
        # 为了画黄线和J值，我们需要整列计算
        df = df.sort("date").with_columns([
            pl.col("close").alias("close_adj"),
            pl.col("high").alias("high_adj"), 
            pl.col("low").alias("low_adj"),
            pl.col("open").alias("open_adj"),
        ]).with_columns([
            # 白线 WL
            pl.col("close_adj").ewm_mean(span=10, adjust=False).ewm_mean(span=10, adjust=False).alias("WL"),
            # 黄线 YL
            ((pl.col("close_adj").rolling_mean(14) + 
              pl.col("close_adj").rolling_mean(28) + 
              pl.col("close_adj").rolling_mean(57) + 
              pl.col("close_adj").rolling_mean(114)) / 4).alias("YL"),

            # KDJ (为了看J值)
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

        # 3. 截取最近 N 天用于展示
        df_plot = df.tail(lookback)

        # ==============================================================================
        # 3. Plotly 绘图
        # ==============================================================================

        # 创建子图：上图K线，下图成交量
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.7, 0.3]
        )

        # A. K线图 (Candlestick)
        fig.add_trace(go.Candlestick(
            x=df_plot["date"],
            open=df_plot["open_adj"],
            high=df_plot["high_adj"],
            low=df_plot["low_adj"],
            close=df_plot["close_adj"],
            name="K线",
            increasing_line_color='red', 
            decreasing_line_color='green'
        ), row=1, col=1)

        # B. 黄白线 (YL/WL) - 灵魂指标
        fig.add_trace(go.Scatter(
            x=df_plot["date"],
            y=df_plot["WL"],
            mode='lines',
            name='WL (白线)',
            # 🔥 修改点：将 color='white' 改为 'RoyalBlue' (或者 'Blue', 'Purple', 'Teal')
            line=dict(color='RoyalBlue', width=2), 
            opacity=0.8
        ), row=1, col=1)

    
        fig.add_trace(go.Scatter(
            x=df_plot["date"],
            y=df_plot["YL"],
            mode='lines',
            name='YL (黄线)',
            # 黄线在白底上其实有点淡，建议稍微加深一点，比如 'DarkOrange'
            line=dict(color='DarkOrange', width=2), 
            opacity=0.8
        ), row=1, col=1)

        # C. 成交量 (Volume)
        # 根据涨跌设置颜色
        colors = ['red' if row['open_adj'] < row['close_adj'] else 'green' 
                  for i, row in df_plot.to_pandas().iterrows()]

        fig.add_trace(go.Bar(
            x=df_plot["date"],
            y=df_plot["volume"],
            name='成交量',
            marker_color=colors
        ), row=2, col=1)

        # D. 标记 B1 信号点 (J<13 且 站上黄线)
        # 找出符合条件的日期
        b1_points = df_plot.filter((pl.col("J") < 13) & (pl.col("close_adj") > pl.col("YL")))
        if b1_points.height > 0:
            fig.add_trace(go.Scatter(
                x=b1_points["date"],
                y=b1_points["low_adj"] * 0.98, # 标在最低价下方一点点
                mode='markers',
                name='B1信号',
                marker=dict(symbol='triangle-up', size=10, color='purple')
            ), row=1, col=1)

        # 布局美化
        fig.update_layout(
            title=f"{current_code} - 前复权行情 (含黄线)",
            xaxis_rangeslider_visible=False, # 隐藏自带的滑块，用UI控制
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified" # 统一十字光标
        )

        # 最终显示
        result_view = mo.ui.plotly(fig)

    # 显示结果
    result_view
    return df, file_path


@app.cell
def _(date, mo):
    # 日期选择器 (默认为你刚才的案例时间)
    ui_buy_date = mo.ui.date(
        value=date(2024, 3, 28), 
        label="📅 买入日期 (T日)"
    )

    ui_sell_date = mo.ui.date(
        value=date(2024, 4, 3), 
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
    return


if __name__ == "__main__":
    app.run()
