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

    # Label: T 日尾盘买入 → T+N 日尾盘卖出 (多期限, 用于 Alpha Decay 分析)
    # 因子使用 T-1 及更早数据 (shift(1), 无前视), 信号在 T 日盘中计算完毕
    df_with_label = df_factors.with_columns([
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_1d"),
        (pl.col("close_adj").shift(-2).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_2d"),
        (pl.col("close_adj").shift(-3).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_3d"),
        (pl.col("close_adj").shift(-5).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_5d"),
    ])

    # 截面标准化
    df_normalized = cross_section_normalize(df_with_label, FACTOR_COLS)

    # 只保留研究需要的列，避免把中间辅助列一并 collect
    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount", "close_raw", "market_cap_100m",
        "circulating_capital",
        "fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d",
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
def _(df_all, df_ic_summary, go, make_subplots, np, stats):
    # ==============================================================================
    # Cell 3b: Alpha Decay 分析 — 因子预测力随持仓天数衰减
    # ==============================================================================
    def run_alpha_decay():
        horizons = ["fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d"]
        h_days = [1, 2, 3, 5]

        top_factors = df_ic_summary["factor"].head(15).to_list()
        print("📉 [Alpha Decay] Top-15 因子预测力随持仓天数衰减...", flush=True)

        df_sub = df_all.select(top_factors + horizons + ["date"]).sort("date")
        dates_arr = df_sub["date"].to_numpy()
        factor_np = {fn: df_sub[fn].to_numpy() for fn in top_factors}
        ret_np = {h: df_sub[h].to_numpy() for h in horizons}

        u_dates = np.unique(dates_arr)
        u_dates.sort()
        d_start = np.searchsorted(dates_arr, u_dates, side="left")
        d_end = np.searchsorted(dates_arr, u_dates, side="right")

        ic_lists = {h: {fn: [] for fn in top_factors} for h in horizons}

        for di in range(len(u_dates)):
            s, e = d_start[di], d_end[di]
            if e - s < 30:
                continue
            for h in horizons:
                rv = ret_np[h][s:e]
                vr = np.isfinite(rv)
                if vr.sum() < 30:
                    continue
                for fn in top_factors:
                    fv = factor_np[fn][s:e]
                    mask = np.isfinite(fv) & vr
                    if mask.sum() < 30:
                        continue
                    corr, _ = stats.spearmanr(fv[mask], rv[mask])
                    ic_lists[h][fn].append(corr)

            if di % 200 == 0:
                print(f"  进度: {di}/{len(u_dates)} ({di/len(u_dates)*100:.0f}%)", flush=True)

        decay = {}
        for h in horizons:
            decay[h] = {}
            for fn in top_factors:
                if ic_lists[h][fn]:
                    arr = np.array(ic_lists[h][fn])
                    decay[h][fn] = {
                        "ic_mean": float(np.mean(arr)),
                        "icir": float(np.mean(arr) / max(np.std(arr), 1e-8)),
                    }
                else:
                    decay[h][fn] = {"ic_mean": 0.0, "icir": 0.0}

        print("\n" + "=" * 100)
        print("  因子 IC 衰减对比 (Top 15, 按 1d |ICIR| 排序)")
        print("=" * 100)
        hdr = f"{'因子':<22}"
        for d in h_days:
            hdr += f"  {'IC_'+str(d)+'d':>8} {'ICIR_'+str(d)+'d':>8}"
        print(hdr)
        print("-" * 100)
        for fn in top_factors:
            row_str = f"{fn:<22}"
            for h in horizons:
                dd = decay[h][fn]
                row_str += f"  {dd['ic_mean']:>8.4f} {dd['icir']:>8.4f}"
            print(row_str)
        print("-" * 100)

        avg_icir = {}
        print("\n📊 平均 |ICIR| 衰减:")
        for h, d in zip(horizons, h_days):
            vals = [abs(decay[h][fn]["icir"]) for fn in top_factors]
            avg = float(np.mean(vals))
            avg_icir[d] = avg
            print(f"  {d}d: avg |ICIR| = {avg:.4f}")

        COST_REF = 0.002
        GROSS_REF = 0.00187
        print(f"\n💰 盈亏平衡 (假设日均毛alpha≈{GROSS_REF:.3%}, 成本={COST_REF:.1%}):")
        for d in h_days:
            cum = GROSS_REF * d
            net = cum - COST_REF
            daily_net = net / d
            print(f"  持仓 {d}d: 累积毛={cum:.3%}, 成本={COST_REF:.3%}, "
                  f"净={net:.3%}, 日均净={daily_net:.4%}")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Top-8 因子 |ICIR| 衰减", "平均 |ICIR| 衰减"],
        )

        for fn in top_factors[:8]:
            y_vals = [abs(decay[h][fn]["icir"]) for h in horizons]
            fig.add_trace(go.Scatter(
                x=h_days, y=y_vals, name=fn, mode="lines+markers",
            ), row=1, col=1)

        avg_y = [avg_icir[d] for d in h_days]
        fig.add_trace(go.Scatter(
            x=h_days, y=avg_y, name="平均",
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in avg_y],
            textposition="top center",
            line=dict(width=3, color="#00d4aa"),
        ), row=1, col=2)

        fig.update_layout(
            height=450, template="plotly_dark",
            xaxis_title="持仓天数", xaxis2_title="持仓天数",
            yaxis_title="|ICIR|", yaxis2_title="平均 |ICIR|",
        )
        fig.show()

        return decay, avg_icir

    decay_summary, avg_icir_decay = run_alpha_decay()
    return decay_summary, avg_icir_decay


@app.cell
def _():
    # Cell 4: (已移除 — 线性排名回测已被 LightGBM + Rust 架构替代)
    return


@app.cell
def _():
    # Cell 5: (已移除 — 旧可视化依赖线性回测, 回测逻辑已迁移至 Rust)
    return


@app.cell
def _(FACTOR_COLS, df_all, np, pl):
    # ==============================================================================
    # Cell 6: LightGBM Walk-Forward 打分 → Parquet 导出
    # 模型只负责打分, 回测交给 Rust ECS 引擎
    # ==============================================================================
    def run_lgbm_scoring():
        import lightgbm as lgb
        import warnings
        from utils.signal_export import export_rotation_scores
        warnings.filterwarnings("ignore", category=UserWarning)

        TRAIN_WINDOW = 480
        RETRAIN_FREQ = 20
        TOP_N = 20
        LABEL = "fwd_ret_1d"

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

        print("🤖 LightGBM Walk-Forward 打分 → Parquet", flush=True)
        print(f"   训练窗口: {TRAIN_WINDOW}天, 重训: 每{RETRAIN_FREQ}天, 标签: {LABEL}", flush=True)
        print(f"   特征数: {len(feature_cols)}, Top-{TOP_N}", flush=True)

        df_valid_ml = (
            df_all
            .filter(pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan())
            .sort("date")
        )

        X_all_np = df_valid_ml.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
        y_all_np = df_valid_ml[LABEL].to_numpy().astype(np.float32)
        dates_np = df_valid_ml["date"].to_numpy()
        codes_np = df_valid_ml["code"].to_numpy()

        unique_dates_ml = np.unique(dates_np)
        unique_dates_ml.sort()
        n_dates = len(unique_dates_ml)

        date_start = np.searchsorted(dates_np, unique_dates_ml, side="left")
        date_end = np.searchsorted(dates_np, unique_dates_ml, side="right")

        score_dates = []
        score_codes = []
        score_values = []
        model = None
        last_train_idx = -RETRAIN_FREQ

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
                model.fit(X_tr[valid], y_tr[valid], feature_name=feature_cols)
                last_train_idx = i

                pct = (i - TRAIN_WINDOW) / (n_dates - TRAIN_WINDOW) * 100
                print(f"   [{cur_date}] 重训 ({pct:.0f}%), 样本: {valid.sum():,}", flush=True)

            if model is None:
                continue

            s, e = date_start[i], date_end[i]
            X_te = X_all_np[s:e]
            codes_te = codes_np[s:e]
            n_stocks = e - s

            if n_stocks < TOP_N:
                continue

            preds = model.predict(X_te)
            cur_date_py = cur_date.astype("datetime64[D]").item()
            score_dates.extend([cur_date_py] * n_stocks)
            score_codes.extend(codes_te.tolist())
            score_values.extend(preds.tolist())

        print(f"\n   ✅ 打分完成: {len(score_values):,} 条记录", flush=True)

        # ── Build scores DataFrame + join price data ──
        df_scores = pl.DataFrame({
            "date": score_dates,
            "code": score_codes,
            "score": score_values,
        })

        price_cols = [
            "date", "code", "open_adj", "high_adj", "low_adj", "close_adj",
            "volume", "market_cap_100m",
        ]
        df_scores_full = df_scores.join(
            df_all.select([c for c in price_cols if c in df_all.columns]),
            on=["date", "code"],
            how="left",
        )

        # ── Export Parquet ──
        output_path = export_rotation_scores(df_scores_full, top_n=TOP_N)

        # ── Feature importance (最后一个模型) ──
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

        return df_scores_full, output_path

    df_rotation_scores, scores_path = run_lgbm_scoring()
    return df_rotation_scores, scores_path


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
