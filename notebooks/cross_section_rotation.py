import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from datetime import datetime
    from scipy import stats

    from utils import load_daily_data_full
    from utils import get_st_blacklist_pl
    from utils.rotation_factors import (
        calc_rotation_factors,
        cross_section_normalize,
        FACTOR_COLS,
    )

    # ==============================================================================
    # Cell 1: 配置与数据加载
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Universe 参数
    MV_MIN = 8       # 最小市值 (亿) → 80 亿
    MV_MAX = 50      # 最大市值 (亿) → 500 亿
    MIN_LIST_DAYS = 60  # 最少上市天数
    START_DATE = "2020-09-01"  # 创业板注册制后

    print("🚀 [Step 1] 加载全量日线数据...")
    st_blacklist = get_st_blacklist_pl("2026-03-17")

    q_full = (
        load_daily_data_full(conn)
        .filter(~pl.col("code").is_in(st_blacklist))
        .filter(pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
    )

    # 上市天数过滤 + 市值过滤 (动态, 每天重新筛选)
    q_universe = (
        q_full
        .sort(["code", "date"])
        .with_columns(
            pl.col("date").cum_count().over("code").alias("list_days")
        )
        .filter(
            (pl.col("list_days") >= MIN_LIST_DAYS) &
            (pl.col("market_cap_100m") >= MV_MIN) &
            (pl.col("market_cap_100m") <= MV_MAX)
        )
    )

    print(f"✅ Universe: 市值 {MV_MIN*10}~{MV_MAX*10} 亿, 上市>{MIN_LIST_DAYS}天, 起始={START_DATE}")
    return (
        FACTOR_COLS,
        calc_rotation_factors,
        cross_section_normalize,
        go,
        make_subplots,
        np,
        pl,
        q_universe,
        stats,
    )


@app.cell
def _(
    FACTOR_COLS,
    calc_rotation_factors,
    cross_section_normalize,
    pl,
    q_universe,
):
    import os

    os.environ['RUST_BACKTRACE']='1'
    # ==============================================================================
    # Cell 2: 因子计算 + 截面标准化 + Label
    # ==============================================================================
    print("⏳ [Step 2] 计算截面轮动因子...")

    df_factors = calc_rotation_factors(q_universe)

    # Label: T 日买入(收盘价) → T+1 日卖出(收盘价) 的收益
    df_with_label = df_factors.with_columns(
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1)
            .alias("fwd_ret_1d")
    )

    # 截面标准化
    df_normalized = cross_section_normalize(df_with_label, FACTOR_COLS)

    # 收集 (触发计算)
    print("⏳ [Step 2] Collecting... (这一步可能需要几分钟)")
    df_all = df_normalized.collect()
    print(f"✅ 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    return (df_all,)


@app.cell
def _(FACTOR_COLS, df_all, go, make_subplots, np, pl, stats):
    # ==============================================================================
    # Cell 3: 因子 IC 分析
    # ==============================================================================
    print("📊 [Step 3] 计算因子 IC (Spearman 截面相关系数)...")

    # 过滤掉没有 label 的行
    df_valid = df_all.filter(pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan())

    dates = df_valid["date"].unique().sort().to_list()
    ic_records = []

    for d in dates:
        daily = df_valid.filter(pl.col("date") == d)
        if len(daily) < 30:
            continue
        ret = daily["fwd_ret_1d"].to_numpy()
        row = {"date": d}
        for f in FACTOR_COLS:
            fvals = daily[f].to_numpy()
            mask = np.isfinite(fvals) & np.isfinite(ret)
            if mask.sum() < 30:
                row[f] = np.nan
            else:
                corr, _ = stats.spearmanr(fvals[mask], ret[mask])
                row[f] = corr
        ic_records.append(row)

    df_ic = pl.DataFrame(ic_records)
    print(f"✅ IC 计算完成: {len(df_ic)} 个交易日")

    # IC 汇总统计
    ic_summary = []
    for f in FACTOR_COLS:
        ic_series = df_ic[f].drop_nulls().drop_nans()
        if len(ic_series) == 0:
            continue
        ic_arr = ic_series.to_numpy()
        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos_ratio = np.mean(ic_arr > 0)
        ic_summary.append({
            "factor": f,
            "IC_mean": round(ic_mean, 4),
            "IC_std": round(ic_std, 4),
            "ICIR": round(icir, 4),
            "IC_pos_ratio": round(ic_pos_ratio, 4),
            "abs_ICIR": round(abs(icir), 4),
        })

    df_ic_summary = pl.DataFrame(ic_summary).sort("abs_ICIR", descending=True)
    print("\n" + "=" * 80)
    print("  因子 IC 排行榜 (按 |ICIR| 降序)")
    print("=" * 80)
    print(f"{'因子':<22} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'IC>0 比例':>10}")
    print("-" * 80)
    for row in df_ic_summary.iter_rows(named=True):
        print(f"{row['factor']:<22} {row['IC_mean']:>10.4f} {row['IC_std']:>10.4f} "
              f"{row['ICIR']:>10.4f} {row['IC_pos_ratio']:>10.1%}")
    print("-" * 80)

    # Top 6 因子的 IC 累积曲线
    top_factors = df_ic_summary["factor"].head(6).to_list()
    fig_ic = make_subplots(rows=1, cols=1)
    for f in top_factors:
        ic_cum = df_ic.select(["date", f]).drop_nulls().sort("date")
        fig_ic.add_trace(go.Scatter(
            x=ic_cum["date"].to_list(),
            y=ic_cum[f].cum_sum().to_list(),
            name=f,
            mode="lines",
        ))
    fig_ic.update_layout(
        title="Top 6 因子 — IC 累积曲线",
        xaxis_title="日期", yaxis_title="累积 IC",
        height=500, template="plotly_dark",
    )
    fig_ic.show()
    return df_ic, df_ic_summary


@app.cell
def _(df_all, df_ic_summary, np, pl):
    # ==============================================================================
    # Cell 4: 简单 Top-N 等权轮动回测
    # ==============================================================================
    TOP_N = 20
    COST_RATE = 0.002  # 双边千分之二

    # 选择 ICIR 最高的因子作为排序依据
    best_factor = df_ic_summary["factor"][0]
    # 根据 IC_mean 的方向决定排序方式
    best_ic_mean = df_ic_summary.filter(pl.col("factor") == best_factor)["IC_mean"][0]
    sort_descending = best_ic_mean > 0  # IC>0 → 因子越大收益越高 → 降序

    print(f"🎯 [Step 4] Top-{TOP_N} 轮动回测")
    print(f"   排序因子: {best_factor} (IC_mean={best_ic_mean:.4f}, {'降序' if sort_descending else '升序'})")
    print(f"   双边成本: {COST_RATE:.1%}")

    # 也构建一个多因子等权合成得分做对比
    # 取 ICIR 最高的 5 个因子, 按 IC 方向对齐后等权求和
    top5 = df_ic_summary.head(5)
    score_exprs = []
    for row in top5.iter_rows(named=True):
        f = row["factor"]
        direction = 1 if row["IC_mean"] > 0 else -1
        score_exprs.append(pl.col(f) * direction)

    df_scored = df_all.with_columns(
        (sum(score_exprs) / len(score_exprs)).alias("composite_score")
    )

    # 逐日回测
    df_valid_bt = df_scored.filter(
        pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan()
    )
    dates_bt = df_valid_bt["date"].unique().sort().to_list()

    results_single = []   # 单因子
    results_composite = []  # 多因子合成

    for d in dates_bt:
        daily = df_valid_bt.filter(pl.col("date") == d)
        if len(daily) < TOP_N:
            continue

        # 单因子 Top-N
        top_single = daily.sort(best_factor, descending=sort_descending).head(TOP_N)
        avg_ret_single = top_single["fwd_ret_1d"].mean()
        results_single.append({"date": d, "daily_ret": avg_ret_single - COST_RATE})

        # 多因子合成 Top-N
        top_composite = daily.sort("composite_score", descending=True).head(TOP_N)
        avg_ret_composite = top_composite["fwd_ret_1d"].mean()
        results_composite.append({"date": d, "daily_ret": avg_ret_composite - COST_RATE})

    df_bt_single = pl.DataFrame(results_single).sort("date")
    df_bt_composite = pl.DataFrame(results_composite).sort("date")

    # 全 A 等权基准
    results_bench = []
    for d in dates_bt:
        daily = df_valid_bt.filter(pl.col("date") == d)
        if len(daily) < TOP_N:
            continue
        avg_ret = daily["fwd_ret_1d"].mean()
        results_bench.append({"date": d, "daily_ret": avg_ret})
    df_bench = pl.DataFrame(results_bench).sort("date")

    # 计算净值
    def calc_metrics(df_ret, name):
        rets = df_ret["daily_ret"].to_numpy()
        nav = np.cumprod(1 + rets)
        total_ret = nav[-1] - 1
        n_years = len(rets) / 242
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        max_dd = np.max(1 - nav / np.maximum.accumulate(nav))
        sharpe = np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(242)
        avg_daily = np.mean(rets)
        win_rate = np.mean(rets > 0)
        skew = float(pl.Series(rets).skew()) if len(rets) > 2 else 0
        return {
            "name": name,
            "total_ret": total_ret,
            "ann_ret": ann_ret,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "avg_daily": avg_daily,
            "win_rate": win_rate,
            "skew": skew,
            "n_days": len(rets),
        }

    m_single = calc_metrics(df_bt_single, f"单因子({best_factor})")
    m_composite = calc_metrics(df_bt_composite, "多因子合成(Top5)")
    m_bench = calc_metrics(df_bench, "全A等权基准")

    print("\n" + "=" * 100)
    print(f"  Top-{TOP_N} 等权轮动回测 (双边成本 {COST_RATE:.1%})")
    print("=" * 100)
    print(f"{'策略':<22} {'年化':>8} {'累计':>8} {'最大回撤':>8} {'Sharpe':>8} {'日均':>8} {'胜率':>8} {'偏度':>8} {'天数':>6}")
    print("-" * 100)
    for m in [m_single, m_composite, m_bench]:
        print(f"{m['name']:<22} {m['ann_ret']:>7.1%} {m['total_ret']:>7.1%} "
              f"{m['max_dd']:>7.1%} {m['sharpe']:>8.2f} {m['avg_daily']:>7.3%} "
              f"{m['win_rate']:>7.1%} {m['skew']:>8.2f} {m['n_days']:>6d}")
    print("-" * 100)
    return (
        COST_RATE,
        TOP_N,
        best_factor,
        df_bench,
        df_bt_composite,
        df_bt_single,
    )


@app.cell
def _(
    COST_RATE,
    FACTOR_COLS,
    TOP_N,
    best_factor,
    df_bench,
    df_bt_composite,
    df_bt_single,
    df_ic,
    go,
    make_subplots,
    np,
    pl,
):
    # ==============================================================================
    # Cell 5: 结果可视化
    # ==============================================================================

    # --- 5.1 净值曲线 ---
    fig_nav = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=["净值曲线", "日收益率"])

    for df_r, name, color in [
        (df_bt_single, f"单因子({best_factor})", "#00d4aa"),
        (df_bt_composite, "多因子合成(Top5)", "#ff6b6b"),
        (df_bench, "全A等权基准", "#888888"),
    ]:
        rets = df_r["daily_ret"].to_numpy()
        nav = np.cumprod(1 + rets)
        dates_list = df_r["date"].to_list()
        fig_nav.add_trace(go.Scatter(
            x=dates_list, y=nav, name=name, mode="lines",
            line=dict(color=color, width=2),
        ), row=1, col=1)

    # 日收益率 (仅显示合成策略)
    fig_nav.add_trace(go.Bar(
        x=df_bt_composite["date"].to_list(),
        y=df_bt_composite["daily_ret"].to_list(),
        name="日收益率(合成)", marker_color="#ff6b6b", opacity=0.4,
        showlegend=False,
    ), row=2, col=1)

    fig_nav.update_layout(
        title=f"截面轮动 Top-{TOP_N} 回测 (成本 {COST_RATE:.1%})",
        height=700, template="plotly_dark",
    )
    fig_nav.update_yaxes(title_text="净值", row=1, col=1)
    fig_nav.update_yaxes(title_text="日收益率", row=2, col=1)
    fig_nav.show()

    # --- 5.2 年度收益拆解 ---
    def yearly_breakdown(df_r, name):
        df_y = df_r.with_columns(pl.col("date").dt.year().alias("year"))
        years = df_y["year"].unique().sort().to_list()
        rows = []
        for y in years:
            yr_data = df_y.filter(pl.col("year") == y)
            rets = yr_data["daily_ret"].to_numpy()
            nav = np.cumprod(1 + rets)
            total = nav[-1] - 1
            dd = np.max(1 - nav / np.maximum.accumulate(nav))
            rows.append({"year": y, "return": total, "max_dd": dd, "n_days": len(rets)})
        return rows

    print("\n" + "=" * 70)
    print("  年度收益拆解")
    print("=" * 70)
    print(f"{'年份':>6} | {'合成策略':>10} {'最大回撤':>10} | {'基准':>10} {'超额':>10} | {'交易日':>6}")
    print("-" * 70)
    yr_strat = {r["year"]: r for r in yearly_breakdown(df_bt_composite, "strat")}
    yr_bench = {r["year"]: r for r in yearly_breakdown(df_bench, "bench")}
    for y in sorted(yr_strat.keys()):
        s = yr_strat[y]
        b = yr_bench.get(y, {"return": 0, "max_dd": 0, "n_days": 0})
        excess = s["return"] - b["return"]
        print(f"{y:>6} | {s['return']:>9.1%} {s['max_dd']:>9.1%} | {b['return']:>9.1%} {excess:>9.1%} | {s['n_days']:>6}")
    print("-" * 70)

    # --- 5.3 IC 月度热力图 ---
    df_ic_monthly = (
        df_ic
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ])
    )

    # 取 Top 10 因子做热力图
    top10_factors = [f for f in FACTOR_COLS if f in df_ic.columns][:10]
    if len(top10_factors) > 0:
        ic_heatmap_data = []
        year_months = (
            df_ic_monthly
            .select(["year", "month"])
            .unique()
            .sort(["year", "month"])
        )
        for row in year_months.iter_rows(named=True):
            y, m = row["year"], row["month"]
            monthly = df_ic_monthly.filter(
                (pl.col("year") == y) & (pl.col("month") == m)
            )
            entry = {"period": f"{y}-{m:02d}"}
            for f in top10_factors:
                vals = monthly[f].drop_nulls().drop_nans()
                entry[f] = float(vals.mean()) if len(vals) > 0 else 0.0
            ic_heatmap_data.append(entry)

        df_heatmap = pl.DataFrame(ic_heatmap_data)
        periods = df_heatmap["period"].to_list()
        z_data = [df_heatmap[f].to_list() for f in top10_factors]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data, x=periods, y=top10_factors,
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.3f}" for v in row] for row in z_data],
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(
            title="因子月度 IC 均值热力图",
            height=400, template="plotly_dark",
            xaxis_title="月份", yaxis_title="因子",
        )
        fig_heat.show()
    return


if __name__ == "__main__":
    app.run()
