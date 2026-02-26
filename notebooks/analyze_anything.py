import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import duckdb

    # ==========================================
    # 1. UI 配置与输入区域
    # ==========================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"

    # 创建侧边栏或顶部控件
    stock_input = mo.ui.text(
        value="sh.600941", 
        label="🔍 股票代码 (e.g. sh.600941)",
        full_width=True
    )

    growth_slider = mo.ui.slider(
        start=-0.10, stop=0.30, step=0.01, value=0.05, 
        label="📈 预期未来利润增长率 (Growth)",
        show_value=True
    )

    # 布局：将控件放在顶部
    controls = mo.vstack([
        mo.md("# 🚀 个股估值看板 (Marimo版)"),
        stock_input,
        growth_slider
    ])

    controls
    return DB_PATH, duckdb, growth_slider, mo, pl, px, stock_input


@app.cell
def _(DB_PATH, duckdb, mo, pl, stock_input):
    # ==========================================
    # 2. 数据加载函数 (响应式)
    # ==========================================
    # 当 stock_input 变化时，这个 cell 会自动重新运行
    code_val = stock_input.value

    def load_data(code, db_path):
        """从 DuckDB 加载所有必要的数据"""
        conn = duckdb.connect(db_path, read_only=True)
        try:
            count = conn.execute(
                f"SELECT COUNT(*) FROM stock_daily WHERE code = '{code}'"
            ).fetchone()[0]
            if count == 0:
                return None, "数据未找到，请检查股票代码"

            # 1. 日线
            df_d = pl.read_database(
                f"SELECT date, open, high, low, close, volume, amount "
                f"FROM stock_daily WHERE code = '{code}' ORDER BY date",
                conn
            ).with_columns(
                pl.col("date").cast(pl.Date)
            ).filter(pl.col("close").is_not_null()).sort("date")

            # 2. 股本 (pub_date 对应旧逻辑的 m_anntime)
            df_c = pl.read_database(
                f"SELECT pub_date, total_capital FROM finance_capital "
                f"WHERE code = '{code}' AND pub_date IS NOT NULL ORDER BY pub_date",
                conn
            ).with_columns(
                pl.col("pub_date").cast(pl.Date).alias("date"),
                pl.col("total_capital").cast(pl.Float64)
            ).select(["date", "total_capital"]).sort("date")

            # 3. 资产负债 (pub_date 对应旧逻辑的 m_anntime)
            df_b = pl.read_database(
                f"SELECT pub_date, net_assets FROM finance_balance "
                f"WHERE code = '{code}' AND pub_date IS NOT NULL ORDER BY pub_date",
                conn
            ).with_columns(
                pl.col("pub_date").cast(pl.Date).alias("date"),
                pl.col("net_assets").cast(pl.Float64)
            ).select(["date", "net_assets"]).sort("date")

            # 4. 利润表 (pub_date=m_anntime, date=m_timetag 即报告期)
            df_i = pl.read_database(
                f"SELECT pub_date, date AS report_date, net_profit AS cum_profit "
                f"FROM finance_income WHERE code = '{code}' ORDER BY pub_date",
                conn
            ).with_columns([
                pl.col("pub_date").cast(pl.Date),
                pl.col("report_date").cast(pl.Date),
                pl.col("cum_profit").cast(pl.Float64)
            ]).sort("pub_date")

            # 5. 分红 (从 qmt_factors 提取 interest 字段)
            df_div_raw = pl.read_database(
                f"SELECT date, interest FROM qmt_factors "
                f"WHERE code = '{code}' AND interest IS NOT NULL ORDER BY date",
                conn
            ).with_columns(
                pl.col("date").cast(pl.Date),
                pl.col("interest").cast(pl.Float64)
            ).group_by("date").agg(pl.col("interest").sum()).sort("date")

            return (df_d, df_c, df_b, df_i, df_div_raw), None
        except Exception as e:
            return None, str(e)
        finally:
            conn.close()

    # 执行加载
    data_pack, error_msg = load_data(code_val, DB_PATH)

    # 如果出错，停止后续运行并显示错误
    if error_msg:
        mo.stop(True, mo.md(f"❌ **错误**: {error_msg}"))

    (df_daily, df_cap, df_bal, df_inc, df_div) = data_pack
    return df_bal, df_cap, df_daily, df_div, df_inc


@app.cell
def _(df_bal, df_cap, df_daily, df_div, df_inc, pl):
    # ==========================================
    # 3. 核心计算逻辑 (分红率 & TTM)
    # ==========================================

    # --- A. 历史分红率计算 ---
    # 整理每年净利润
    df_yearly_profit = df_inc.with_columns(
        pl.col("report_date").dt.year().alias("year"),
        pl.col("report_date").dt.month().alias("month")
    ).filter(pl.col("month") == 12).group_by("year").last().select(["year", "cum_profit"]).sort("year")

    # 整理分红 (错位对齐)
    df_yearly_div = df_div.with_columns(
        (pl.col("date").dt.year() - 1).alias("profit_year") 
    ).group_by("profit_year").agg(pl.col("interest").sum().alias("total_div")).sort("profit_year")

    # 整理股本
    df_yearly_cap = df_cap.with_columns(
        pl.col("date").dt.year().alias("year")
    ).group_by("year").last().select(["year", "total_capital"])

    # 合并计算
    df_payout = df_yearly_profit.join(df_yearly_div, left_on="year", right_on="profit_year", how="inner")
    df_payout = df_payout.join(df_yearly_cap, on="year", how="left")
    df_payout = df_payout.with_columns(
        (pl.col("total_div") * pl.col("total_capital") / pl.col("cum_profit")).alias("payout_ratio")
    ).filter((pl.col("payout_ratio") > 0) & (pl.col("payout_ratio") < 1.2))

    # 计算平均分红率
    if df_payout.height >= 1:
        avg_payout_ratio = df_payout.tail(3)["payout_ratio"].mean()
        payout_msg = f"根据最近 {df_payout.height} 年数据，自适应计算分红率为: **{avg_payout_ratio*100:.2f}%**"
    else:
        avg_payout_ratio = 0.30
        payout_msg = "⚠️ 数据不足，使用默认分红率 30%"

    # --- B. TTM 数据处理 ---
    min_date = df_daily["date"].min()
    max_date = df_daily["date"].max()
    df_calendar = pl.date_range(min_date, max_date, "1d", eager=True).to_frame("date")

    df_div_ttm = df_calendar.join(df_div, on="date", how="left").with_columns(
        pl.col("interest").fill_null(0.0)
    ).with_columns(
        pl.col("interest").rolling_sum(window_size=365).alias("div_ttm")
    )

    # 简单年化因子
    df_inc_fix = df_inc.with_columns(
        pl.col("report_date").dt.month().alias("rpt_month")
    ).with_columns(
        pl.when(pl.col("rpt_month") == 3).then(4.0)
          .when(pl.col("rpt_month") == 6).then(2.0)
          .when(pl.col("rpt_month") == 9).then(1.33333333)
          .otherwise(1.0).alias("annual_factor")
    )
    df_inc_simple = df_inc_fix.group_by("pub_date").last().rename({"pub_date": "date"}).sort("date")

    # --- C. 主表合并 ---
    df_main = df_daily.join_asof(df_cap, on="date", strategy="backward")
    df_main = df_main.join_asof(df_bal, on="date", strategy="backward")
    df_main = df_main.join_asof(df_inc_simple.select(["date", "cum_profit", "annual_factor"]), on="date", strategy="backward")

    df_main = df_main.with_columns((pl.col("cum_profit") * pl.col("annual_factor")).alias("net_profit_ttm_approx"))
    df_main = df_main.join(df_div_ttm.select(["date", "div_ttm"]), on="date", how="left")

    df_final = df_main.with_columns([
        (pl.col("close") * pl.col("total_capital")).alias("market_cap")
    ]).with_columns([
        (pl.col("market_cap") / pl.col("net_assets")).alias("PB"),
        (pl.col("market_cap") / pl.col("net_profit_ttm_approx")).alias("PE"),
        (pl.col("div_ttm") / pl.col("close") * 100).alias("Yield_Pct")
    ]).select([
            "date", "close", "total_capital", "net_assets", "net_profit_ttm_approx", "div_ttm", "PB", "PE", "Yield_Pct"
        ]).drop_nulls()

    # 获取最新一行数据用于展示
    latest = df_final.tail(1)
    curr_close = latest["close"][0]
    curr_pe = latest["PE"][0]
    curr_pb = latest["PB"][0]
    curr_yield = latest["Yield_Pct"][0]
    return (
        avg_payout_ratio,
        curr_close,
        curr_pb,
        curr_pe,
        curr_yield,
        df_final,
        latest,
        payout_msg,
    )


@app.cell
def _(df_daily):
    df_daily
    return


@app.cell
def _(curr_close, curr_pb, curr_pe, curr_yield, mo, payout_msg):
    # ==========================================
    # 4. 可视化：核心指标看板 (KPIs)
    # ==========================================
    # 使用 mo.stat 展示大号数字
    stats = mo.ui.run_button(label="Refresh") # 占位符，这里其实是自动刷新的

    kpi_cards = mo.hstack([
        mo.stat(value=f"{curr_close:.2f}", label="当前股价", bordered=True),
        mo.stat(value=f"{curr_pe:.2f}", label="PE (TTM)", bordered=True),
        mo.stat(value=f"{curr_pb:.2f}", label="PB (LF)", bordered=True),
        mo.stat(value=f"{curr_yield:.2f}%", label="股息率 (TTM)", bordered=True, caption=payout_msg),
    ])

    kpi_cards
    return


@app.cell
def _(avg_payout_ratio, curr_close, df_inc, growth_slider, latest, mo, pl):
    # ==========================================
    # 5. 可视化：前瞻预测矩阵 (Sensitivity Matrix)
    # ==========================================
    # 这里我们将原来的 print 循环逻辑改造成生成一个 Polars DataFrame
    # 这样 marimo 就可以渲染成漂亮的交互式表格

    # 准备基础数据
    latest_report = df_inc.filter(pl.col("pub_date") <= latest["date"][0]).tail(1)
    rpt_profit = latest_report["cum_profit"][0]
    rpt_month = latest_report["report_date"][0].month
    ttm_profit = latest["net_profit_ttm_approx"][0]
    total_shares = latest["total_capital"][0]

    # 预估今年全年的基准利润
    if rpt_month == 9: est_full_year_profit = rpt_profit / 3 * 4
    elif rpt_month == 6: est_full_year_profit = rpt_profit * 2
    elif rpt_month == 12: est_full_year_profit = rpt_profit
    else: est_full_year_profit = ttm_profit

    # 构造矩阵数据
    base_p = int(avg_payout_ratio * 100)
    payout_scenarios = [base_p-5, base_p-2, base_p, base_p+2, base_p+5]
    growth_scenarios_list = [-0.05, -0.02, 0.00, 0.02, 0.05, 0.08, 0.10, growth_slider.value] 
    # 把用户输入的 growth 也加进去并去重
    growth_scenarios_list = sorted(list(set(growth_scenarios_list)))

    matrix_data = []

    for g in growth_scenarios_list:
        row = {"利润增长": f"{g*100:+.1f}%"}
        for p in payout_scenarios:
            profit_next = est_full_year_profit * (1 + g)
            div_next = (profit_next * (p/100.0)) / total_shares
            yield_next = (div_next / curr_close) * 100

            # 为了表格好看，我们格式化成字符串，或者保留数字由前端渲染
            # 这里为了标记 🔥，我们存成字符串
            val_str = f"{yield_next:.2f}%"
            if yield_next >= 6.0: val_str = "🔥 " + val_str
            elif yield_next >= 5.0: val_str = "✅ " + val_str

            row[f"分红率 {p}%"] = val_str
        matrix_data.append(row)

    df_sensitivity = pl.DataFrame(matrix_data)

    mo.vstack([
        mo.md("### 🔮 敏感性分析：2026 预期股息率"),
        mo.md(f"当前预估基准利润: **{est_full_year_profit/1e8:.2f} 亿** | 当前选中增长率: **{growth_slider.value*100:.1f}%**"),
        mo.ui.table(df_sensitivity, selection=None) # 渲染成漂亮表格
    ])
    return


@app.cell
def _(df_final, mo):
    # ==========================================
    # 6. 历史数据图表 (可选)
    # ==========================================
    # 展示最近3年的 PE/PB 走势
    df_chart = df_final.tail(365*3).select(["date", "PE", "PB", "Yield_Pct"])

    mo.vstack([
        mo.md("### 📉 历史估值走势 (近3年)"),
        mo.ui.table(df_chart, pagination=True, page_size=5) # 也可以换成 mo.ui.chart 做折线图
    ])
    return


@app.cell
def _(df_daily, mo, px):
    # 2. 创建一个图表对象
    fig = px.line(df_daily, x="date", y="close", title="股价走势", markers=True)

    # 3. 将图表转换为 UI 组件 (这就是 Dash 的能力)
    # 当你在图表上框选时，chart_ui.value 会包含被选中的数据点
    chart_ui = mo.ui.plotly(fig)
    return (chart_ui,)


@app.cell
def _(chart_ui, mo):
    # 4. 响应式显示选中内容
    # 这里的 callout 会随着你在图表上的操作自动更新
    mo.hstack([
        chart_ui,
        mo.callout(
            mo.md(f"您选中了 **{len(chart_ui.value)}** 个数据点"), 
            kind="info"
        )
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
