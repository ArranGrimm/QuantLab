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
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(conn)
        .join(st_blacklist_df, on="code", how="anti")
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

    os.environ['RUST_BACKTRACE']='full'
    # ==============================================================================
    # Cell 2: 因子计算 + 截面标准化 + Label
    # ==============================================================================
    print("⏳ [Step 2] 计算截面轮动因子...")

    df_factors = calc_rotation_factors(q_universe)

    # Label: T 日尾盘买入 → T+1 日尾盘卖出
    # 因子使用 T-1 及更早数据 (shift(1), 无前视), 信号在 T 日盘中计算完毕
    df_with_label = df_factors.with_columns(
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1)
            .alias("fwd_ret_1d")
    )

    # 截面标准化
    df_normalized = cross_section_normalize(df_with_label, FACTOR_COLS)

    # 只保留研究需要的列，避免把中间辅助列一并 collect
    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount", "close_raw", "market_cap_100m",
        "circulating_capital", "fwd_ret_1d",
        *FACTOR_COLS,
    ]

    # 收集 (触发计算)
    print("⏳ [Step 2] Collecting... (这一步可能需要几分钟)")
    df_all = df_normalized.select(final_cols).collect()
    print(f"✅ 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    return (df_all,)


@app.cell
def _(FACTOR_COLS, df_all, go, make_subplots, np, pl, stats):
    # ==============================================================================
    # Cell 3: 因子 IC 分析
    # ==============================================================================
    def run_ic_analysis():
        print("📊 [Step 3] 计算因子 IC (Spearman 截面相关系数)...")

        df_valid_local = df_all.filter(
            pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan()
        )

        trading_dates = df_valid_local["date"].unique().sort().to_list()
        ic_records_local = []

        for trade_date in trading_dates:
            daily_df = df_valid_local.filter(pl.col("date") == trade_date)
            if len(daily_df) < 30:
                continue

            ret_values = daily_df["fwd_ret_1d"].to_numpy()
            ic_row = {"date": trade_date}

            for factor_name in FACTOR_COLS:
                factor_values = daily_df[factor_name].to_numpy()
                valid_mask = np.isfinite(factor_values) & np.isfinite(ret_values)
                if valid_mask.sum() < 30:
                    ic_row[factor_name] = np.nan
                else:
                    corr, _ = stats.spearmanr(factor_values[valid_mask], ret_values[valid_mask])
                    ic_row[factor_name] = corr

            ic_records_local.append(ic_row)

        df_ic_local = pl.DataFrame(ic_records_local)
        print(f"✅ IC 计算完成: {len(df_ic_local)} 个交易日")

        ic_summary_local = []
        for factor_name in FACTOR_COLS:
            ic_series = df_ic_local[factor_name].drop_nulls().drop_nans()
            if len(ic_series) == 0:
                continue

            ic_arr = ic_series.to_numpy()
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_pos_ratio = np.mean(ic_arr > 0)
            ic_summary_local.append({
                "factor": factor_name,
                "IC_mean": round(ic_mean, 4),
                "IC_std": round(ic_std, 4),
                "ICIR": round(icir, 4),
                "IC_pos_ratio": round(ic_pos_ratio, 4),
                "abs_ICIR": round(abs(icir), 4),
            })

        df_ic_summary_local = pl.DataFrame(ic_summary_local).sort("abs_ICIR", descending=True)
        print("\n" + "=" * 80)
        print("  因子 IC 排行榜 (按 |ICIR| 降序)")
        print("=" * 80)
        print(f"{'因子':<22} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'IC>0 比例':>10}")
        print("-" * 80)
        for summary_row in df_ic_summary_local.iter_rows(named=True):
            print(
                f"{summary_row['factor']:<22} {summary_row['IC_mean']:>10.4f} "
                f"{summary_row['IC_std']:>10.4f} {summary_row['ICIR']:>10.4f} "
                f"{summary_row['IC_pos_ratio']:>10.1%}"
            )
        print("-" * 80)

        top_factors_local = df_ic_summary_local["factor"].head(6).to_list()
        fig_ic_local = make_subplots(rows=1, cols=1)
        for factor_name in top_factors_local:
            ic_cum = df_ic_local.select(["date", factor_name]).drop_nulls().sort("date")
            fig_ic_local.add_trace(go.Scatter(
                x=ic_cum["date"].to_list(),
                y=ic_cum[factor_name].cum_sum().to_list(),
                name=factor_name,
                mode="lines",
            ))

        fig_ic_local.update_layout(
            title="Top 6 因子 — IC 累积曲线",
            xaxis_title="日期",
            yaxis_title="累积 IC",
            height=500,
            template="plotly_dark",
        )
        fig_ic_local.show()
        return df_ic_local, df_ic_summary_local

    df_ic, df_ic_summary = run_ic_analysis()
    return df_ic, df_ic_summary


@app.cell
def _(df_all, df_ic_summary, np, pl):
    # ==============================================================================
    # Cell 4: 简单 Top-N 等权轮动回测
    # ==============================================================================
    def run_topn_backtest():
        top_n = 20
        cost_rate = 0.002  # 双边千分之二

        best_factor_local = df_ic_summary["factor"][0]
        best_ic_mean = df_ic_summary.filter(pl.col("factor") == best_factor_local)["IC_mean"][0]
        sort_desc = best_ic_mean > 0

        print(f"🎯 [Step 4] Top-{top_n} 轮动回测")
        print(
            f"   排序因子: {best_factor_local} "
            f"(IC_mean={best_ic_mean:.4f}, {'降序' if sort_desc else '升序'})"
        )
        print(f"   双边成本: {cost_rate:.1%}")

        top5_factors = df_ic_summary.head(5)
        score_exprs = []
        for factor_row in top5_factors.iter_rows(named=True):
            factor_name = factor_row["factor"]
            direction = 1 if factor_row["IC_mean"] > 0 else -1
            score_exprs.append(pl.col(factor_name) * direction)

        df_scored_local = df_all.with_columns(
            (sum(score_exprs) / len(score_exprs)).alias("composite_score")
        )

        df_valid_bt_local = df_scored_local.filter(
            pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan()
        )
        trading_dates = df_valid_bt_local["date"].unique().sort().to_list()

        results_single_local = []
        results_composite_local = []
        results_bench_local = []

        for trade_date in trading_dates:
            daily_df = df_valid_bt_local.filter(pl.col("date") == trade_date)
            if len(daily_df) < top_n:
                continue

            top_single = daily_df.sort(best_factor_local, descending=sort_desc).head(top_n)
            avg_ret_single = top_single["fwd_ret_1d"].mean()
            results_single_local.append({"date": trade_date, "daily_ret": avg_ret_single - cost_rate})

            top_composite = daily_df.sort("composite_score", descending=True).head(top_n)
            avg_ret_composite = top_composite["fwd_ret_1d"].mean()
            results_composite_local.append({"date": trade_date, "daily_ret": avg_ret_composite - cost_rate})

            avg_ret_bench = daily_df["fwd_ret_1d"].mean()
            results_bench_local.append({"date": trade_date, "daily_ret": avg_ret_bench})

        df_bt_single_local = pl.DataFrame(results_single_local).sort("date")
        df_bt_composite_local = pl.DataFrame(results_composite_local).sort("date")
        df_bench_local = pl.DataFrame(results_bench_local).sort("date")

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

        metrics_single = calc_metrics(df_bt_single_local, f"单因子({best_factor_local})")
        metrics_composite = calc_metrics(df_bt_composite_local, "多因子合成(Top5)")
        metrics_bench = calc_metrics(df_bench_local, "全A等权基准")

        print("\n" + "=" * 100)
        print(f"  Top-{top_n} 等权轮动回测 (双边成本 {cost_rate:.1%})")
        print("=" * 100)
        print(f"{'策略':<22} {'年化':>8} {'累计':>8} {'最大回撤':>8} {'Sharpe':>8} {'日均':>8} {'胜率':>8} {'偏度':>8} {'天数':>6}")
        print("-" * 100)
        for metric in [metrics_single, metrics_composite, metrics_bench]:
            print(
                f"{metric['name']:<22} {metric['ann_ret']:>7.1%} {metric['total_ret']:>7.1%} "
                f"{metric['max_dd']:>7.1%} {metric['sharpe']:>8.2f} {metric['avg_daily']:>7.3%} "
                f"{metric['win_rate']:>7.1%} {metric['skew']:>8.2f} {metric['n_days']:>6d}"
            )
        print("-" * 100)

        return (
            cost_rate,
            top_n,
            best_factor_local,
            df_bench_local,
            df_bt_composite_local,
            df_bt_single_local,
        )

    (
        COST_RATE,
        TOP_N,
        best_factor,
        df_bench,
        df_bt_composite,
        df_bt_single,
    ) = run_topn_backtest()
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


@app.cell
def _(FACTOR_COLS, df_all, go, make_subplots, np, pl):
    # ==============================================================================
    # Cell 6: LightGBM Walk-Forward 回测
    # ==============================================================================
    def run_lgbm_walkforward():
        import lightgbm as lgb
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        TRAIN_WINDOW = 480   # ~2 年训练窗口
        RETRAIN_FREQ = 20    # 每 20 个交易日重训一次
        TOP_N = 20
        HOLD_BUFFER = 150     # 持仓缓冲: 进入要 Top-20, 保留只需 Top-50
        COST = 0.002

        feature_cols = list(FACTOR_COLS)

        lgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 100,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        print(f"🤖 [Phase 2] LightGBM Walk-Forward 回测", flush=True)
        print(f"   训练窗口: {TRAIN_WINDOW}天, 重训频率: 每{RETRAIN_FREQ}天", flush=True)
        print(f"   特征数: {len(feature_cols)}, Top-{TOP_N}, 缓冲: Top-{HOLD_BUFFER}, 成本: {COST:.1%}", flush=True)

        df_valid_ml = (
            df_all
            .filter(pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan())
            .sort("date")
        )

        X_all_np = df_valid_ml.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
        y_all_np = df_valid_ml["fwd_ret_1d"].to_numpy().astype(np.float32)
        dates_np = df_valid_ml["date"].to_numpy()
        codes_np = df_valid_ml["code"].to_numpy()

        unique_dates_ml = np.unique(dates_np)
        unique_dates_ml.sort()
        n_dates = len(unique_dates_ml)

        date_start = np.searchsorted(dates_np, unique_dates_ml, side="left")
        date_end = np.searchsorted(dates_np, unique_dates_ml, side="right")

        results_ml = []
        model = None
        last_train_idx = -RETRAIN_FREQ
        prev_holdings = set()

        for i in range(TRAIN_WINDOW, n_dates):
            cur_date = unique_dates_ml[i]

            if i - last_train_idx >= RETRAIN_FREQ or model is None:
                ts = date_start[i - TRAIN_WINDOW]
                te = date_end[i - 1]
                X_tr = X_all_np[ts:te]
                y_tr = y_all_np[ts:te]

                valid = np.isfinite(y_tr)
                if valid.sum() < 1000:
                    continue

                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(
                    X_tr[valid], y_tr[valid],
                    feature_name=feature_cols,
                )
                last_train_idx = i

                pct = (i - TRAIN_WINDOW) / (n_dates - TRAIN_WINDOW) * 100
                print(
                    f"   [{cur_date}] 重训 ({pct:.0f}%), "
                    f"样本: {valid.sum():,}",
                    flush=True,
                )

            s, e = date_start[i], date_end[i]
            X_te = X_all_np[s:e]
            y_te = y_all_np[s:e]
            codes_te = codes_np[s:e]
            n_stocks = e - s

            if n_stocks < TOP_N:
                continue

            preds = model.predict(X_te)

            rank_order = np.argsort(-preds)
            ranks = np.empty(n_stocks, dtype=int)
            ranks[rank_order] = np.arange(n_stocks)

            if not prev_holdings:
                selected_idx = rank_order[:TOP_N]
            else:
                keep_idx = [
                    j for j in range(n_stocks)
                    if codes_te[j] in prev_holdings and ranks[j] < HOLD_BUFFER
                ]
                n_keep = len(keep_idx)

                if n_keep >= TOP_N:
                    keep_idx.sort(key=lambda j: ranks[j])
                    selected_idx = np.array(keep_idx[:TOP_N])
                else:
                    n_fill = TOP_N - n_keep
                    keep_set = set(keep_idx)
                    new_candidates = [j for j in rank_order if j not in keep_set]
                    selected_idx = np.array(keep_idx + new_candidates[:n_fill])

            cur_holdings = set(codes_te[selected_idx])

            if prev_holdings:
                n_new = len(cur_holdings - prev_holdings)
                turnover = n_new / TOP_N
            else:
                turnover = 1.0

            actual_cost = COST * turnover
            prev_holdings = cur_holdings

            avg_ret_gross = float(np.mean(y_te[selected_idx]))
            bench_ret = float(np.nanmean(y_te))

            results_ml.append({
                "date": cur_date.astype("datetime64[D]").item(),
                "daily_ret": avg_ret_gross - actual_cost,
                "daily_ret_full_cost": avg_ret_gross - COST,
                "gross": avg_ret_gross,
                "bench": bench_ret,
                "turnover": turnover,
                "n_stocks": n_stocks,
            })

        df_ml = pl.DataFrame(results_ml).sort("date")
        avg_turnover = df_ml["turnover"].mean()
        print(f"\n   📊 平均日换手率: {avg_turnover:.1%} (日均成本: {COST * avg_turnover:.3%})", flush=True)

        def calc_perf(rets_arr, name):
            nav = np.cumprod(1 + rets_arr)
            total = nav[-1] - 1
            n_y = len(rets_arr) / 242
            ann = (1 + total) ** (1 / max(n_y, 0.01)) - 1
            dd = np.max(1 - nav / np.maximum.accumulate(nav))
            sh = np.mean(rets_arr) / max(np.std(rets_arr), 1e-8) * np.sqrt(242)
            wr = np.mean(rets_arr > 0)
            sk = float(pl.Series(rets_arr).skew()) if len(rets_arr) > 2 else 0
            return {
                "name": name, "total": total, "ann": ann, "dd": dd,
                "sharpe": sh, "avg": np.mean(rets_arr), "wr": wr,
                "skew": sk, "n": len(rets_arr),
            }

        m_net = calc_perf(df_ml["daily_ret"].to_numpy(), f"LGB (真实成本)")
        m_full = calc_perf(df_ml["daily_ret_full_cost"].to_numpy(), f"LGB (100%换手成本)")
        m_gross = calc_perf(df_ml["gross"].to_numpy(), f"LGB (毛收益)")
        m_bench = calc_perf(df_ml["bench"].to_numpy(), "全A等权基准")

        print(f"\n{'=' * 120}")
        print(f"  LightGBM Walk-Forward Top-{TOP_N}  |  平均日换手率: {avg_turnover:.1%}")
        print(f"{'=' * 120}")
        hdr = f"{'策略':<28} {'年化':>8} {'累计':>8} {'最大回撤':>8} {'Sharpe':>8} {'日均':>8} {'胜率':>8} {'偏度':>8} {'天数':>6}"
        print(hdr)
        print("-" * 120)
        for m in [m_net, m_full, m_gross, m_bench]:
            print(
                f"{m['name']:<28} {m['ann']:>7.1%} {m['total']:>7.1%} "
                f"{m['dd']:>7.1%} {m['sharpe']:>8.2f} {m['avg']:>7.3%} "
                f"{m['wr']:>7.1%} {m['skew']:>8.2f} {m['n']:>6d}"
            )
        print("-" * 120)

        # Feature importance
        if model is not None:
            imp_vals = model.feature_importances_
            imp_max = max(imp_vals) if max(imp_vals) > 0 else 1
            imp_df = pl.DataFrame({
                "factor": feature_cols,
                "importance": imp_vals.tolist(),
            }).sort("importance", descending=True)

            print("\n" + "=" * 55)
            print("  LightGBM 特征重要性 Top 15")
            print("=" * 55)
            for imp_row in imp_df.head(15).iter_rows(named=True):
                bar_len = int(imp_row["importance"] / imp_max * 30)
                bar = "█" * bar_len
                print(f"  {imp_row['factor']:<22} {imp_row['importance']:>6} {bar}")
            print("=" * 55)

        # Yearly breakdown
        df_ml_yr = df_ml.with_columns(pl.col("date").dt.year().alias("year"))
        print(f"\n{'=' * 95}")
        print("  年度收益拆解 (LightGBM)")
        print(f"{'=' * 95}")
        print(f"{'年份':>6} | {'真实净收益':>10} {'毛收益':>10} {'最大回撤':>10} {'换手率':>8} | {'基准':>10} {'超额(净)':>10} | {'天数':>6}")
        print("-" * 95)
        for yr in sorted(df_ml_yr["year"].unique().to_list()):
            yr_data = df_ml_yr.filter(pl.col("year") == yr)
            net_rets = yr_data["daily_ret"].to_numpy()
            gross_rets = yr_data["gross"].to_numpy()
            bench_rets = yr_data["bench"].to_numpy()
            yr_turnover = yr_data["turnover"].mean()
            nav_n = np.cumprod(1 + net_rets)
            nav_g = np.cumprod(1 + gross_rets)
            nav_b = np.cumprod(1 + bench_rets)
            dd_n = np.max(1 - nav_n / np.maximum.accumulate(nav_n))
            print(
                f"{yr:>6} | {nav_n[-1]-1:>9.1%} {nav_g[-1]-1:>9.1%} {dd_n:>9.1%} {yr_turnover:>7.1%} "
                f"| {nav_b[-1]-1:>9.1%} {nav_n[-1]-1 - (nav_b[-1]-1):>9.1%} | {len(net_rets):>6}"
            )
        print("-" * 95)

        # NAV chart
        nav_net = np.cumprod(1 + df_ml["daily_ret"].to_numpy())
        nav_gross = np.cumprod(1 + df_ml["gross"].to_numpy())
        nav_bench = np.cumprod(1 + df_ml["bench"].to_numpy())

        fig_ml = make_subplots(
            rows=2, cols=1, row_heights=[0.7, 0.3],
            shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=["LightGBM 净值曲线", "日收益率"],
        )
        dates_list = df_ml["date"].to_list()
        fig_ml.add_trace(go.Scatter(
            x=dates_list, y=nav_net.tolist(),
            name="LGB Net", mode="lines",
            line=dict(color="#00d4aa", width=2),
        ), row=1, col=1)
        fig_ml.add_trace(go.Scatter(
            x=dates_list, y=nav_gross.tolist(),
            name="LGB Gross", mode="lines",
            line=dict(color="#ffaa00", width=1.5, dash="dot"),
        ), row=1, col=1)
        fig_ml.add_trace(go.Scatter(
            x=dates_list, y=nav_bench.tolist(),
            name="全A等权基准", mode="lines",
            line=dict(color="#888888", width=1),
        ), row=1, col=1)
        fig_ml.add_trace(go.Bar(
            x=dates_list, y=df_ml["daily_ret"].to_list(),
            marker_color="#00d4aa", opacity=0.3, showlegend=False,
        ), row=2, col=1)
        fig_ml.update_layout(
            title=f"LightGBM Walk-Forward Top-{TOP_N} (成本 {COST:.1%})",
            height=700, template="plotly_dark",
        )
        fig_ml.update_yaxes(title_text="净值", row=1, col=1)
        fig_ml.update_yaxes(title_text="日收益率", row=2, col=1)
        fig_ml.show()

        return df_ml

    df_lgbm_result = run_lgbm_walkforward()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
