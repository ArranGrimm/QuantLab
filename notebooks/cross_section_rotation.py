import marimo

__generated_with = "0.21.1"
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
    MV_MIN = 80      # 最小流通市值 (亿)
    MV_MAX = 500     # 最大流通市值 (亿)
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

    print(f"✅ 参数: 流通市值 {MV_MIN}~{MV_MAX} 亿, 上市>{MIN_LIST_DAYS}天, 起始={START_DATE}")
    return (
        FACTOR_COLS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        calc_rotation_factors,
        cross_section_normalize,
        go,
        make_subplots,
        np,
        pl,
        q_full,
        stats,
    )


@app.cell
def _(
    FACTOR_COLS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    calc_rotation_factors,
    cross_section_normalize,
    pl,
    q_full,
):
    import os

    os.environ['RUST_BACKTRACE']='full'
    # ==============================================================================
    # Cell 2: 全量因子计算 → 市值过滤 → 截面标准化
    #
    # 关键流程:
    #   q_full (全量) → 因子计算 (连续序列) → forward return (连续序列)
    #   → 市值+上市天数过滤 (确定可交易 universe)
    #   → 截面标准化 (在 universe 内 z-score)
    #
    # 因子必须在连续序列上计算, 市值过滤只决定"哪些股票可交易"
    # ==============================================================================
    print("⏳ [Step 2] 计算截面轮动因子 (全量股票, 保证序列连续)...")

    df_factors = calc_rotation_factors(q_full)

    # Label: 在连续序列上计算 forward return, 避免因市值过滤产生的缺口
    df_with_label = df_factors.with_columns([
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_1d"),
        (pl.col("close_adj").shift(-2).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_2d"),
        (pl.col("close_adj").shift(-3).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_3d"),
        (pl.col("close_adj").shift(-5).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_5d"),
    ])

    # 市值 + 上市天数过滤 → 确定每日可交易 universe
    df_universe = (
        df_with_label
        .with_columns(
            pl.col("date").cum_count().over("code").alias("_list_days")
        )
        .filter(
            (pl.col("_list_days") >= MIN_LIST_DAYS) &
            (pl.col("market_cap_100m") >= MV_MIN) &
            (pl.col("market_cap_100m") <= MV_MAX)
        )
    )

    # 在可交易 universe 内做截面标准化
    df_normalized = cross_section_normalize(df_universe, FACTOR_COLS)

    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount", "close_raw", "market_cap_100m",
        "circulating_capital",
        "fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d",
        *FACTOR_COLS,
    ]

    print("⏳ [Step 2] Collecting... (全量因子计算, 可能需要更长时间)")
    df_all = df_normalized.select(final_cols).collect()
    print(f"✅ 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    return (df_all,)


@app.cell
def _(FACTOR_COLS, df_all, go, make_subplots, np, pl):
    # ==============================================================================
    # Cell 3: 因子 IC 分析 (Polars 原生加速)
    # ==============================================================================
    from utils.ic_analysis import calc_factor_ic

    def run_ic_analysis():
        LABEL = "fwd_ret_1d"

        df_valid_local = df_all.filter(
            pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan()
        )

        ic_results = calc_factor_ic(
            df_valid_local,
            factor_cols=list(FACTOR_COLS),
            label=LABEL,
            min_samples=30,
        )

        # 构建 df_ic_summary 供 Cell 3b 使用
        ic_summary_local = []
        for factor_name, r in ic_results.items():
            ic_summary_local.append({
                "factor": factor_name,
                "IC_mean": round(r["ic_mean"], 4),
                "IC_std": round(r["ic_std"], 4),
                "ICIR": round(r["icir"], 4),
                "IC_pos_ratio": 0.0,
                "abs_ICIR": round(abs(r["icir"]), 4),
            })
        df_ic_summary_local = pl.DataFrame(ic_summary_local).sort("abs_ICIR", descending=True)

        # 逐日 IC DataFrame (用于累积 IC 图)
        date_counts = df_valid_local.group_by("date").agg(pl.len().alias("n"))
        valid_dates = date_counts.filter(pl.col("n") >= 30)["date"]
        df_filtered = df_valid_local.filter(pl.col("date").is_in(valid_dates))

        available = [f for f in FACTOR_COLS if f in df_filtered.columns]
        df_ic_local = (
            df_filtered
            .group_by("date")
            .agg([
                pl.corr(f, LABEL, method="spearman").alias(f)
                for f in available
            ])
            .sort("date")
        )

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
        return df_ic_summary_local

    df_ic_summary = run_ic_analysis()
    return (df_ic_summary,)


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
    return


@app.cell
def _():
    # Cell 4: (已移除 — 线性排名回测已被 LightGBM + Rust 架构替代)
    return


@app.cell
def _():
    # Cell 5: (已移除 — 旧可视化依赖线性回测, 回测逻辑已迁移至 Rust)
    return


@app.cell
def _(FACTOR_COLS, df_all, np, pl, q_full):
    # ==============================================================================
    # Cell 6: LightGBM Walk-Forward 打分 → Parquet 导出
    # 模型只负责打分, 回测交给 Rust ECS 引擎
    #
    # 关键: Parquet 必须包含所有"曾被评分"股票在整个回测期间的价格数据,
    # 即使某天该股票不在 universe 内 (市值越界等), 也要保留价格行,
    # 否则 Rust 引擎无法对该仓位执行止损/排名退出等检查 → "幽灵仓位"
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
        EMA_ALPHA = 0.15  # Score 时序平滑 (1.0 = 不平滑)

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

        # ── Build scores DataFrame + EMA 平滑 ──
        df_scores = pl.DataFrame({
            "date": score_dates,
            "code": score_codes,
            "score": score_values,
        })

        df_scores_raw = df_scores.clone()

        if EMA_ALPHA < 1.0:
            df_scores = (
                df_scores
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score")
                      .ewm_mean(alpha=EMA_ALPHA)
                      .over("code")
                      .alias("score")
                )
            )
            print(f"   ⚡ Score EMA 平滑: α={EMA_ALPHA}", flush=True)

        # ── 补全: 用 q_full 获取所有曾评分股票的完整价格序列 ──
        # 当股票市值越界离开 universe 时, 依然保留其价格行供 Rust 做退出判断
        ever_scored_codes = df_scores["code"].unique().to_list()
        score_date_min = df_scores["date"].min()
        score_date_max = df_scores["date"].max()

        print(f"   📦 补全价格: {len(ever_scored_codes)} 只股票, "
              f"{score_date_min} ~ {score_date_max}", flush=True)

        price_cols = ["date", "code", "open_adj", "high_adj", "low_adj",
                      "close_adj", "volume", "market_cap_100m"]
        df_full_prices = (
            q_full
            .filter(pl.col("code").is_in(ever_scored_codes))
            .filter(pl.col("date") >= score_date_min)
            .filter(pl.col("date") <= score_date_max)
            .select([c for c in price_cols if c in q_full.collect_schema().names()])
            .collect()
        )

        # Left-join: 有评分的用真实分数, 无评分的 (离开 universe) score 填 -999
        df_expanded = df_full_prices.join(
            df_scores, on=["date", "code"], how="left"
        ).with_columns(
            pl.col("score").fill_null(-999.0),
        )

        n_scored = df_scores.height
        n_total = df_expanded.height
        n_padded = n_total - n_scored
        print(f"   评分行: {n_scored:,}, 补全行: {n_padded:,}, 总计: {n_total:,}", flush=True)

        # ── Export Parquet ──
        output_path = export_rotation_scores(df_expanded, top_n=TOP_N)

        # ── Feature importance (最后一个模型) ──
        if model is not None:
            imp_vals = model.feature_importances_
            imp_max = max(imp_vals) if max(imp_vals) > 0 else 1
            imp_df = pl.DataFrame({
                "factor": feature_cols,
                "importance": imp_vals.tolist(),
            }).sort("importance", descending=True)

            print("\n" + "=" * 55)
            print(f"  LightGBM 特征重要性 (全部 {len(feature_cols)} 个)")
            print("=" * 55)
            for imp_row in imp_df.iter_rows(named=True):
                bar_len = int(imp_row["importance"] / imp_max * 30)
                bar = "█" * bar_len
                print(f"  {imp_row['factor']:<22} {imp_row['importance']:>6} {bar}")
            print("=" * 55)

        return df_expanded, output_path, df_scores_raw

    df_rotation_scores, scores_path, df_scores_raw = run_lgbm_scoring()
    return (df_scores_raw,)


@app.cell
def _(df_all, df_scores_raw, go, make_subplots, np, pl, stats):
    # ==============================================================================
    # Cell 7: Signal Quality Analysis — 模型信号统计检验
    #
    # 基于 Cell 6 输出的原始分数 (df_scores_raw), 独立做 EMA 平滑.
    # 修改 EMA_ALPHA 只需重跑本 Cell, 无需重新训练模型.
    #
    # 包含:
    #   7a. OOS IC/ICIR + t检验 + 累积IC曲线
    #   7b. Quintile Long-Short 分层收益
    #   7c. Prediction Turnover (Top-20 日间重叠率)
    # ==============================================================================
    def run_signal_quality():
        EMA_ALPHA = 0.15  # Score 时序平滑 (1.0 = 不平滑, 仅影响分析, 不影响训练)

        # ── 合并原始 score 与 forward return ──
        df_signal = (
            df_scores_raw
            .select(["date", "code", "score"])
            .join(
                df_all.select(["date", "code", "fwd_ret_1d"]),
                on=["date", "code"],
                how="inner",
            )
            .filter(pl.col("fwd_ret_1d").is_not_null() & pl.col("fwd_ret_1d").is_not_nan())
            .sort(["date", "code"])
        )

        if EMA_ALPHA < 1.0:
            df_signal = (
                df_signal
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score")
                      .ewm_mean(alpha=EMA_ALPHA)
                      .over("code")
                      .alias("score")
                )
                .sort(["date", "code"])
            )
            print(f"   ⚡ Score EMA 平滑: α={EMA_ALPHA}")
        else:
            print("   ⚡ 无 EMA 平滑 (原始分数)")

        dates_np = df_signal["date"].to_numpy()
        scores_np = df_signal["score"].to_numpy().astype(np.float64)
        rets_np = df_signal["fwd_ret_1d"].to_numpy().astype(np.float64)
        codes_np = df_signal["code"].to_numpy()

        unique_dates = np.unique(dates_np)
        unique_dates.sort()
        n_days = len(unique_dates)
        date_start = np.searchsorted(dates_np, unique_dates, side="left")
        date_end = np.searchsorted(dates_np, unique_dates, side="right")

        print(f"📊 Signal Quality Analysis")
        print(f"   样本: {len(dates_np):,} 条, {n_days} 个交易日\n")

        # ================================================================
        # 7a. OOS IC / ICIR / t-test
        # ================================================================
        daily_ic = np.full(n_days, np.nan)
        daily_n_stocks = np.zeros(n_days, dtype=int)

        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            sc = scores_np[s:e]
            rt = rets_np[s:e]
            mask = np.isfinite(sc) & np.isfinite(rt)
            cnt = mask.sum()
            daily_n_stocks[i] = cnt
            if cnt >= 30:
                ic, _ = stats.spearmanr(sc[mask], rt[mask])
                if np.isfinite(ic):
                    daily_ic[i] = ic

        valid_ic = daily_ic[np.isfinite(daily_ic)]
        ic_mean = float(np.mean(valid_ic))
        ic_std = float(np.std(valid_ic))
        icir = ic_mean / max(ic_std, 1e-8)
        t_stat = ic_mean / max(ic_std, 1e-8) * np.sqrt(len(valid_ic))
        ic_pos_pct = float(np.mean(valid_ic > 0)) * 100
        cum_ic = np.nancumsum(daily_ic)

        print("=" * 65)
        print("  7a. OOS Model IC Analysis")
        print("=" * 65)
        print(f"  IC Mean:       {ic_mean:+.4f}")
        print(f"  IC Std:        {ic_std:.4f}")
        print(f"  ICIR:          {icir:+.4f}")
        print(f"  t-stat:        {t_stat:+.2f}  {'✅ 显著 (>2)' if abs(t_stat) > 2 else '❌ 不显著 (<2)'}")
        print(f"  IC > 0 占比:   {ic_pos_pct:.1f}%")
        print(f"  有效天数:      {len(valid_ic)} / {n_days}")
        print(f"  日均股票数:    {int(np.mean(daily_n_stocks))}")
        print("-" * 65)

        # ================================================================
        # 7b. Quintile Analysis (分层收益)
        # ================================================================
        N_Q = 5
        quintile_daily = {q: [] for q in range(1, N_Q + 1)}
        ls_daily = []

        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            sc = scores_np[s:e]
            rt = rets_np[s:e]
            mask = np.isfinite(sc) & np.isfinite(rt)
            if mask.sum() < N_Q * 10:
                continue

            sc_v = sc[mask]
            rt_v = rt[mask]
            order = np.argsort(-sc_v)
            n = len(order)
            group_size = n // N_Q

            for q in range(N_Q):
                start_idx = q * group_size
                end_idx = (q + 1) * group_size if q < N_Q - 1 else n
                grp_ret = float(np.mean(rt_v[order[start_idx:end_idx]]))
                quintile_daily[q + 1].append(grp_ret)

            q1_ret = float(np.mean(rt_v[order[:group_size]]))
            q5_ret = float(np.mean(rt_v[order[-(n - (N_Q - 1) * group_size):]]))
            ls_daily.append(q1_ret - q5_ret)

        ls_arr = np.array(ls_daily)
        ls_mean = float(np.mean(ls_arr))
        ls_std = float(np.std(ls_arr))
        ls_sharpe = ls_mean / max(ls_std, 1e-8) * np.sqrt(242)
        ls_t = ls_mean / max(ls_std, 1e-8) * np.sqrt(len(ls_arr))
        ls_hit = float(np.mean(ls_arr > 0)) * 100

        print("\n" + "=" * 65)
        print("  7b. Quintile Long-Short Analysis (Q1 做多 - Q5 做空)")
        print("=" * 65)
        for q in range(1, N_Q + 1):
            arr = np.array(quintile_daily[q])
            qm = float(np.mean(arr)) * 100
            print(f"  Q{q} 日均收益: {qm:+.3f}%")
        print(f"  ---")
        print(f"  L/S 日均收益:  {ls_mean * 100:+.4f}%")
        print(f"  L/S 年化Sharpe: {ls_sharpe:.2f}")
        print(f"  L/S t-stat:    {ls_t:+.2f}  {'✅ 显著' if abs(ls_t) > 2 else '❌ 不显著'}")
        print(f"  L/S 胜率:      {ls_hit:.1f}%")
        cum_ls = np.cumsum(ls_arr)
        ls_dd = float(np.max(np.maximum.accumulate(cum_ls) - cum_ls))
        print(f"  L/S 最大回撤:  {ls_dd * 100:.2f}%")
        print("-" * 65)

        # ================================================================
        # 7c. Prediction Turnover (Top-N 日间重叠率)
        # ================================================================
        for top_n in [20, 50]:
            prev_top = None
            overlaps = []
            for i in range(n_days):
                s, e = date_start[i], date_end[i]
                if e - s < top_n:
                    continue
                sc = scores_np[s:e]
                cd = codes_np[s:e]
                top_idx = np.argsort(-sc)[:top_n]
                top_codes = set(cd[top_idx])
                if prev_top is not None:
                    overlap = len(top_codes & prev_top) / top_n
                    overlaps.append(overlap)
                prev_top = top_codes

            ov_arr = np.array(overlaps)
            est_daily_turnover = (1 - np.mean(ov_arr)) * 2
            print(f"\n  Top-{top_n} 日均重叠率: {np.mean(ov_arr) * 100:.1f}%, "
                  f"日均双边换手: {est_daily_turnover * 100:.1f}%, "
                  f"年化换手: {est_daily_turnover * 242:.0f}x")

        # ================================================================
        # 7d. 分年分析
        # ================================================================
        print("\n" + "=" * 65)
        print("  7d. 分年 IC / L-S 统计")
        print("=" * 65)
        print(f"  {'年份':<6} {'IC_mean':>8} {'ICIR':>8} {'t-stat':>8} "
              f"{'L/S日均':>10} {'L/S Sharpe':>10} {'显著?':>6}")
        print("-" * 65)

        ic_dates = unique_dates[np.isfinite(daily_ic)]
        ic_valid_vals = daily_ic[np.isfinite(daily_ic)]

        years = sorted(set(d.astype("datetime64[Y]").item().year
                          for d in ic_dates))

        for yr in years:
            yr_mask_ic = np.array([d.astype("datetime64[Y]").item().year == yr
                                   for d in ic_dates])
            yr_ic = ic_valid_vals[yr_mask_ic]

            if len(yr_ic) < 20:
                continue

            yr_ic_mean = float(np.mean(yr_ic))
            yr_ic_std = float(np.std(yr_ic))
            yr_icir = yr_ic_mean / max(yr_ic_std, 1e-8)
            yr_t = yr_ic_mean / max(yr_ic_std, 1e-8) * np.sqrt(len(yr_ic))

            yr_ls = [ls_daily[j] for j in range(len(ls_daily))
                     if j < len(unique_dates) and
                     unique_dates[j].astype("datetime64[Y]").item().year == yr]
            if yr_ls:
                yr_ls_arr = np.array(yr_ls)
                yr_ls_mean = float(np.mean(yr_ls_arr))
                yr_ls_sharpe = yr_ls_mean / max(float(np.std(yr_ls_arr)), 1e-8) * np.sqrt(242)
            else:
                yr_ls_mean = 0.0
                yr_ls_sharpe = 0.0

            sig = "✅" if abs(yr_t) > 2 else "❌"
            print(f"  {yr:<6} {yr_ic_mean:>+8.4f} {yr_icir:>+8.4f} {yr_t:>+8.2f} "
                  f"{yr_ls_mean * 100:>+10.4f}% {yr_ls_sharpe:>10.2f} {sig:>6}")
        print("-" * 65)

        # ================================================================
        # Visualization
        # ================================================================
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "累积 IC 曲线",
                "分层日均收益 (Quintile)",
                "L/S 累积收益曲线",
                "Top-20 日间重叠率 (滚动20日均)",
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # 1. Cumulative IC
        valid_dates = unique_dates[np.isfinite(daily_ic)]
        cum_ic_valid = np.nancumsum(daily_ic[np.isfinite(daily_ic)])
        fig.add_trace(go.Scatter(
            x=valid_dates.astype("datetime64[D]").tolist(),
            y=cum_ic_valid.tolist(),
            name="累积IC", line=dict(color="#00d4aa"),
        ), row=1, col=1)

        # 2. Quintile bar chart
        q_means = [float(np.mean(quintile_daily[q])) * 100 for q in range(1, N_Q + 1)]
        colors = ["#00d4aa" if m > 0 else "#ff6b6b" for m in q_means]
        fig.add_trace(go.Bar(
            x=[f"Q{q}" for q in range(1, N_Q + 1)],
            y=q_means,
            marker_color=colors,
            name="日均收益%",
        ), row=1, col=2)

        # 3. L/S cumulative return
        cum_ls_arr = np.cumsum(ls_arr)
        fig.add_trace(go.Scatter(
            y=cum_ls_arr.tolist(),
            name="L/S累积收益", line=dict(color="#ffa500"),
        ), row=2, col=1)

        # 4. Top-20 overlap rolling mean
        prev_top = None
        overlap_series = []
        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            if e - s < 20:
                continue
            sc = scores_np[s:e]
            cd = codes_np[s:e]
            top_idx = np.argsort(-sc)[:20]
            top_codes = set(cd[top_idx])
            if prev_top is not None:
                overlap_series.append(len(top_codes & prev_top) / 20)
            prev_top = top_codes

        if len(overlap_series) > 20:
            ov_rolling = np.convolve(overlap_series,
                                     np.ones(20) / 20, mode="valid")
            fig.add_trace(go.Scatter(
                y=(ov_rolling * 100).tolist(),
                name="重叠率%(20日均)", line=dict(color="#9b59b6"),
            ), row=2, col=2)

        fig.update_layout(
            height=700, template="plotly_dark",
            showlegend=False,
            yaxis_title="累积IC", yaxis2_title="日均收益(%)",
            yaxis3_title="累积L/S收益", yaxis4_title="重叠率(%)",
        )
        fig.show()

        return {
            "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "t_stat": t_stat, "ic_pos_pct": ic_pos_pct,
            "ls_mean": ls_mean, "ls_sharpe": ls_sharpe, "ls_t": ls_t,
            "ls_hit": ls_hit,
        }

    signal_report = run_signal_quality()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
