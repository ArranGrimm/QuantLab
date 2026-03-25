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
    from plotly.subplots import make_subplots
    from scipy import stats

    from utils import load_daily_data_full
    from utils import get_st_blacklist_pl
    from utils import calc_b1_factors_wmacd
    from utils.signal_export import export_for_rust
    from utils.rotation_factors import calc_rotation_factors, FACTOR_COLS as ROTATION_FACTORS

    # ==============================================================================
    # Cell 1: 配置与数据加载
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    START_DATE = "2020-09-01"

    print("🚀 [Step 1] 加载全量日线数据...")
    st_blacklist = get_st_blacklist_pl("2026-03-17")
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
    )

    print(f"✅ 参数: 流通市值 {MV_MIN}~{MV_MAX}亿, 上市>{MIN_LIST_DAYS}天, 起始={START_DATE}")
    return (
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        ROTATION_FACTORS,
        calc_b1_factors_wmacd,
        calc_rotation_factors,
        export_for_rust,
        np,
        pl,
        q_full,
        stats,
    )


@app.cell
def _(
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    ROTATION_FACTORS,
    calc_rotation_factors,
    pl,
    q_full,
):
    # ==============================================================================
    # Cell 2: 因子计算 (通用因子 + B1 原始指标 + MFE 标签)
    #
    # 特征来源:
    #   A) rotation 42 因子 (通用量价微观结构)
    #   B) B1 原始连续值指标 (WL/YL/J/KDJ/缩量/MACD 等)
    # 标签:
    #   fwd_mfe_Nd = max(close[t+1:t+N]) / close[t] - 1  (前向最大有利偏移)
    # ==============================================================================
    print("⏳ [Step 2] 计算因子 (rotation通用 + B1指标)...")

    MFE_WINDOW = 10

    # ── A) rotation 通用因子 (连续序列上计算) ──
    df_factors = calc_rotation_factors(q_full)

    # ── B) B1 原始连续值指标 ──
    # 注意: rotation_factors 已经物化了 _c1, _o1, _h1, _l1, _v1, _c2 等 lag-1 列
    df_with_b1 = (
        df_factors

        # B1-1: 知行双线 (WL/YL) — Ztalk 体系核心
        .with_columns([
            pl.col("_c1").ewm_mean(span=10, adjust=False).over("code")
              .ewm_mean(span=10, adjust=False).over("code").alias("_wl"),
            ((pl.col("_c1").rolling_mean(14).over("code") +
              pl.col("_c1").rolling_mean(28).over("code") +
              pl.col("_c1").rolling_mean(57).over("code") +
              pl.col("_c1").rolling_mean(114).over("code")) / 4).alias("_yl"),
        ])
        .with_columns([
            ((pl.col("_c1") - pl.col("_wl")) / pl.max_horizontal(pl.col("_wl"), pl.lit(0.01)) * 100)
                .alias("b1_bias_c_wl"),
            ((pl.col("_c1") - pl.col("_yl")) / pl.max_horizontal(pl.col("_yl"), pl.lit(0.01)) * 100)
                .alias("b1_bias_c_yl"),
            ((pl.col("_wl") - pl.col("_yl")) / pl.max_horizontal(pl.col("_yl"), pl.lit(0.01)) * 100)
                .alias("b1_wl_yl_spread"),
        ])

        # B1-2: KDJ (9 周期)
        .with_columns([
            pl.col("_h1").rolling_max(9).over("code").alias("_h9"),
            pl.col("_l1").rolling_min(9).over("code").alias("_l9"),
        ])
        .with_columns(
            pl.when((pl.col("_h9") - pl.col("_l9")) == 0).then(50.0)
              .otherwise((pl.col("_c1") - pl.col("_l9")) / (pl.col("_h9") - pl.col("_l9")) * 100)
              .alias("_rsv")
        )
        .with_columns(pl.col("_rsv").ewm_mean(com=2, adjust=False).over("code").alias("b1_K"))
        .with_columns(pl.col("b1_K").ewm_mean(com=2, adjust=False).over("code").alias("b1_D"))
        .with_columns((3 * pl.col("b1_K") - 2 * pl.col("b1_D")).alias("b1_J"))
        .with_columns((pl.col("b1_J") - pl.col("b1_J").shift(1).over("code")).alias("b1_J_delta"))

        # B1-3: 量能结构
        .with_columns([
            pl.col("_v1").rolling_max(20).over("code").alias("_v_max_20"),
            pl.col("_v1").rolling_max(40).over("code").alias("_v_max_40"),
            pl.col("_v1").rolling_mean(5).over("code").alias("_v_ma5"),
            # 红量/绿量统计 (阳线量 vs 阴线量)
            pl.when(pl.col("_c1") >= pl.col("_o1")).then(pl.col("_v1")).otherwise(0)
              .rolling_sum(20).over("code").alias("_vol_yang_20"),
            pl.when(pl.col("_c1") < pl.col("_o1")).then(pl.col("_v1")).otherwise(0)
              .rolling_sum(20).over("code").alias("_vol_yin_20"),
        ])
        .with_columns([
            (pl.col("_v1") / pl.max_horizontal(pl.col("_v_max_20"), pl.lit(1.0)))
                .alias("b1_vol_shrink_20"),
            (pl.col("_v1") / pl.max_horizontal(pl.col("_v_max_40"), pl.lit(1.0)))
                .alias("b1_vol_shrink_40"),
            (pl.col("_v1") / pl.max_horizontal(pl.col("_v_ma5"), pl.lit(1.0)))
                .alias("b1_vol_rel_ma5"),
            (pl.col("_vol_yang_20") / pl.max_horizontal(pl.col("_vol_yin_20"), pl.lit(1.0)))
                .alias("b1_yang_yin_ratio"),
        ])

        # B1-4: K 线形态
        .with_columns([
            ((pl.col("_c1") - pl.col("_o1")).abs()
             / pl.max_horizontal(pl.col("_o1"), pl.lit(0.01))).alias("b1_body_pct"),
            ((pl.min_horizontal("_c1", "_o1") - pl.col("_l1"))
             / pl.max_horizontal(pl.col("_h1") - pl.col("_l1"), pl.lit(1e-8))).alias("b1_lower_shadow"),
        ])

        # B1-5: 回调结构
        .with_columns([
            pl.col("_h1").rolling_max(20).over("code").alias("_peak_20"),
            pl.col("_l1").rolling_min(20).over("code").alias("_trough_20"),
        ])
        .with_columns(
            ((pl.col("_peak_20") - pl.col("_c1"))
             / pl.max_horizontal(pl.col("_peak_20") - pl.col("_trough_20"), pl.lit(1e-8)))
                .alias("b1_retrace_ratio")
        )
    )

    # ── C) 前向 MFE 标签 ──
    fwd_cols = [
        pl.col("close_adj").shift(-i).over("code").alias(f"_fwd_{i}")
        for i in range(1, MFE_WINDOW + 1)
    ]
    df_with_label = (
        df_with_b1
        .with_columns(fwd_cols)
        .with_columns(
            pl.max_horizontal([f"_fwd_{i}" for i in range(1, MFE_WINDOW + 1)])
              .alias("_fwd_max")
        )
        .with_columns([
            (pl.col("_fwd_max") / pl.col("close_adj") - 1).alias("fwd_mfe_10d"),
            (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1)
                .alias("fwd_ret_1d"),
            (pl.col("close_adj").shift(-MFE_WINDOW).over("code") / pl.col("close_adj") - 1)
                .alias(f"fwd_ret_{MFE_WINDOW}d"),
        ])
    )

    # ── D) 市值过滤 + 截面标准化 ──
    B1_FACTORS = [
        "b1_bias_c_wl", "b1_bias_c_yl", "b1_wl_yl_spread",
        "b1_K", "b1_D", "b1_J", "b1_J_delta",
        "b1_vol_shrink_20", "b1_vol_shrink_40", "b1_vol_rel_ma5",
        "b1_yang_yin_ratio",
        "b1_body_pct", "b1_lower_shadow",
        "b1_retrace_ratio",
    ]
    ALL_FACTORS = list(ROTATION_FACTORS) + B1_FACTORS

    # 截面标准化 (在 universe 内 z-score)
    from utils.rotation_factors import cross_section_normalize

    df_universe = (
        df_with_label
        .with_columns(pl.col("date").cum_count().over("code").alias("_list_days"))
        .filter(
            (pl.col("_list_days") >= MIN_LIST_DAYS) &
            (pl.col("market_cap_100m") >= MV_MIN) &
            (pl.col("market_cap_100m") <= MV_MAX)
        )
    )
    df_normalized = cross_section_normalize(df_universe, ALL_FACTORS)

    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount", "market_cap_100m",
        "fwd_mfe_10d", "fwd_ret_1d", f"fwd_ret_{MFE_WINDOW}d",
        *ALL_FACTORS,
    ]

    print("⏳ [Step 2] Collecting...")
    df_all = df_normalized.select(final_cols).collect()
    print(f"✅ 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    print(f"   通用因子: {len(ROTATION_FACTORS)}, B1因子: {len(B1_FACTORS)}, 总计: {len(ALL_FACTORS)}")
    return ALL_FACTORS, df_all


@app.cell
def _(ALL_FACTORS, df_all, np, pl, stats):
    # ==============================================================================
    # Cell 3: 单因子 IC 分析 (vs MFE-10 标签)
    #
    # 哪些因子能预测未来 10 天的最大收益?
    # ==============================================================================
    def run_ic_analysis():
        LABEL = "fwd_mfe_10d"
        print(f"📊 [Step 3] 计算因子 IC (Spearman 截面相关 vs {LABEL})...")

        df_valid = df_all.filter(
            pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan()
        )

        dates_np = df_valid["date"].to_numpy()
        unique_dates = np.unique(dates_np)
        unique_dates.sort()

        ic_results = {}
        for factor in ALL_FACTORS:
            factor_np = df_valid[factor].to_numpy().astype(np.float64)
            ret_np = df_valid[LABEL].to_numpy().astype(np.float64)

            daily_ics = []
            for d in unique_dates:
                mask = dates_np == d
                f_day = factor_np[mask]
                r_day = ret_np[mask]
                valid = np.isfinite(f_day) & np.isfinite(r_day)
                if valid.sum() >= 30:
                    ic, _ = stats.spearmanr(f_day[valid], r_day[valid])
                    if np.isfinite(ic):
                        daily_ics.append(ic)

            if len(daily_ics) >= 20:
                arr = np.array(daily_ics)
                ic_mean = arr.mean()
                ic_std = arr.std()
                icir = ic_mean / ic_std if ic_std > 1e-8 else 0
                t = ic_mean / (ic_std / np.sqrt(len(arr))) if ic_std > 1e-8 else 0
                ic_results[factor] = {
                    "ic_mean": ic_mean, "ic_std": ic_std,
                    "icir": icir, "t_stat": t, "n_days": len(arr),
                }

        sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]["icir"]), reverse=True)

        print(f"\n{'因子':<25} {'IC_Mean':>8} {'IC_Std':>8} {'ICIR':>8} {'t-stat':>8} {'显著':>4}")
        print("-" * 75)
        for name, r in sorted_factors:
            sig = "✅" if abs(r["t_stat"]) > 2 else ""
            prefix = "🔵" if name.startswith("b1_") else "  "
            print(f"{prefix}{name:<23} {r['ic_mean']:>+8.4f} {r['ic_std']:>8.4f} "
                  f"{r['icir']:>+8.4f} {r['t_stat']:>+8.2f} {sig:>4}")
        print("-" * 75)

        b1_sig = sum(1 for n, r in sorted_factors if n.startswith("b1_") and abs(r["t_stat"]) > 2)
        rot_sig = sum(1 for n, r in sorted_factors if not n.startswith("b1_") and abs(r["t_stat"]) > 2)
        print(f"\n   B1 因子显著: {b1_sig}/{len([n for n,_ in sorted_factors if n.startswith('b1_')])}")
        print(f"   通用因子显著: {rot_sig}/{len([n for n,_ in sorted_factors if not n.startswith('b1_')])}")

        return ic_results

    # ic_results = run_ic_analysis()
    return


@app.cell
def _(ALL_FACTORS, df_all, np, pl):
    # ==============================================================================
    # Cell 4: LightGBM Walk-Forward 训练
    #
    # 全截面训练, MFE-10 回归标签
    # 推理时可过滤到 B1 候选 (本 cell 先做全量打分)
    # ==============================================================================
    from lightgbm import LGBMRegressor

    def run_lgbm_walkforward():
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        LABEL = "fwd_mfe_10d"
        TRAIN_WINDOW = 480
        RETRAIN_FREQ = 20
        EMA_ALPHA = 1.0

        feature_cols = list(ALL_FACTORS)

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

        print(f"🤖 LightGBM Walk-Forward (标签: {LABEL}, 窗口: {TRAIN_WINDOW}天)")
        print(f"   特征数: {len(feature_cols)}")

        df_valid = (
            df_all
            .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())
            .sort("date")
        )
        all_dates = df_valid["date"].unique().sort().to_list()
        n_dates = len(all_dates)

        X_all = df_valid.select(feature_cols).to_numpy().astype(np.float32)
        y_all = df_valid[LABEL].to_numpy().astype(np.float64)
        dates_all = df_valid["date"].to_numpy()
        codes_all = df_valid["code"].to_numpy()

        np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        score_dates, score_codes, score_values = [], [], []
        model = None
        last_train_idx = TRAIN_WINDOW - RETRAIN_FREQ

        for i in range(TRAIN_WINDOW, n_dates):
            cur_date = all_dates[i]

            if i - last_train_idx >= RETRAIN_FREQ or model is None:
                train_start = all_dates[max(0, i - TRAIN_WINDOW)]
                mask_tr = (dates_all >= np.datetime64(train_start)) & (dates_all < np.datetime64(cur_date))
                X_tr, y_tr = X_all[mask_tr], y_all[mask_tr]

                valid_mask = np.isfinite(y_tr)
                if valid_mask.sum() < 500:
                    continue
                X_tr, y_tr = X_tr[valid_mask], y_tr[valid_mask]

                model = LGBMRegressor(**lgb_params)
                model.fit(X_tr, y_tr, feature_name=feature_cols)
                last_train_idx = i

                pct = (i - TRAIN_WINDOW) / (n_dates - TRAIN_WINDOW) * 100
                print(f"   [{cur_date}] 重训 ({pct:.0f}%), 样本: {valid_mask.sum():,}", flush=True)

            if model is None:
                continue

            mask_te = dates_all == np.datetime64(cur_date)
            X_te = X_all[mask_te]
            codes_te = codes_all[mask_te]
            n_stocks = int(mask_te.sum())

            if n_stocks == 0:
                continue

            preds = model.predict(X_te)
            cur_date_py = cur_date
            score_dates.extend([cur_date_py] * n_stocks)
            score_codes.extend(codes_te.tolist())
            score_values.extend(preds.tolist())

        print(f"\n   ✅ 打分完成: {len(score_values):,} 条记录", flush=True)

        df_scores = pl.DataFrame({
            "date": score_dates,
            "code": score_codes,
            "score": score_values,
        })

        df_scores_raw = df_scores.clone()

        if EMA_ALPHA < 1.0:
            df_scores = (
                df_scores.sort(["code", "date"])
                .with_columns(
                    pl.col("score").ewm_mean(alpha=EMA_ALPHA).over("code").alias("score")
                )
            )
            print(f"   ⚡ Score EMA 平滑: α={EMA_ALPHA}")

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
            for row in imp_df.iter_rows(named=True):
                bar_len = int(row["importance"] / imp_max * 30)
                bar = "█" * bar_len
                prefix = "🔵" if row["factor"].startswith("b1_") else "  "
                print(f"{prefix}{row['factor']:<22} {row['importance']:>6} {bar}")
            print("=" * 55)

        return df_scores_raw

    df_scores_raw = run_lgbm_walkforward()
    return (df_scores_raw,)


@app.cell
def _(df_all, df_scores_raw, np, pl, stats):
    # ==============================================================================
    # Cell 5: 信号质量分析
    #
    # 基于原始分数, 独立 EMA 平滑 (修改 α 只需重跑本 Cell)
    # ==============================================================================
    def run_signal_quality():
        EMA_ALPHA = 1.0
        LABEL = "fwd_mfe_10d"

        df_signal = (
            df_scores_raw
            .select(["date", "code", "score"])
            .join(
                df_all.select(["date", "code", LABEL]),
                on=["date", "code"],
                how="inner",
            )
            .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())
            .sort(["date", "code"])
        )

        if EMA_ALPHA < 1.0:
            df_signal = (
                df_signal.sort(["code", "date"])
                .with_columns(
                    pl.col("score").ewm_mean(alpha=EMA_ALPHA).over("code").alias("score")
                )
                .sort(["date", "code"])
            )
            print(f"   ⚡ Score EMA 平滑: α={EMA_ALPHA}")
        else:
            print("   ⚡ 无 EMA 平滑 (原始分数)")

        dates_np = df_signal["date"].to_numpy()
        scores_np = df_signal["score"].to_numpy().astype(np.float64)
        rets_np = df_signal[LABEL].to_numpy().astype(np.float64)

        unique_dates = np.unique(dates_np)
        unique_dates.sort()
        n_days = len(unique_dates)
        date_start = np.searchsorted(dates_np, unique_dates, side="left")
        date_end = np.searchsorted(dates_np, unique_dates, side="right")

        # ── 7a. IC Analysis ──
        daily_ics = []
        for idx in range(n_days):
            s, e = date_start[idx], date_end[idx]
            sc, rt = scores_np[s:e], rets_np[s:e]
            valid = np.isfinite(sc) & np.isfinite(rt)
            if valid.sum() >= 30:
                ic, _ = stats.spearmanr(sc[valid], rt[valid])
                if np.isfinite(ic):
                    daily_ics.append(ic)

        ic_arr = np.array(daily_ics)
        ic_mean = ic_arr.mean()
        ic_std = ic_arr.std()
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 1e-8 else 0

        print(f"\n📊 Signal Quality Analysis (标签: {LABEL})")
        print(f"   样本: {len(scores_np):,} 条, {n_days} 个交易日\n")
        print("=" * 65)
        print("  Model IC Analysis (vs MFE-10)")
        print("=" * 65)
        print(f"  IC Mean:       {ic_mean:+.4f}")
        print(f"  IC Std:        {ic_std:.4f}")
        print(f"  ICIR:          {icir:+.4f}")
        t_sig = "✅ 显著 (>2)" if abs(t_stat) > 2 else "❌ 不显著"
        print(f"  t-stat:        {t_stat:+.2f}  {t_sig}")
        print(f"  IC > 0 占比:   {(ic_arr > 0).mean():.1%}")
        print(f"  有效天数:      {len(ic_arr)} / {n_days}")
        print("-" * 65)

        # ── 7b. Quintile Analysis ──
        # rankdata: rank 1 = 最小值, 所以 Q1=最低分, Q5=最高分
        print("\n" + "=" * 65)
        print("  Quintile Analysis (Q5 做多 - Q1 做空)")
        print("=" * 65)
        quintile_rets = {q: [] for q in range(1, 6)}
        for idx in range(n_days):
            s, e = date_start[idx], date_end[idx]
            sc, rt = scores_np[s:e], rets_np[s:e]
            valid = np.isfinite(sc) & np.isfinite(rt)
            if valid.sum() < 50:
                continue
            sc_v, rt_v = sc[valid], rt[valid]
            ranks = stats.rankdata(sc_v)
            n = len(ranks)
            for q in range(1, 6):
                lo = (q - 1) / 5 * n
                hi = q / 5 * n
                mask_q = (ranks > lo) & (ranks <= hi)
                if mask_q.sum() > 0:
                    quintile_rets[q].append(rt_v[mask_q].mean())

        q_labels = {1: "最低分", 2: "低分", 3: "中间", 4: "高分", 5: "最高分"}
        for q in range(1, 6):
            arr = np.array(quintile_rets[q])
            print(f"  Q{q} ({q_labels[q]}) 日均MFE: {arr.mean():+.3%}")
        ls_arr = np.array(quintile_rets[5]) - np.array(quintile_rets[1])
        ls_mean = ls_arr.mean()
        ls_std = ls_arr.std()
        ls_t = ls_mean / (ls_std / np.sqrt(len(ls_arr))) if ls_std > 1e-8 else 0
        ls_sig = "✅ 显著" if abs(ls_t) > 2 else "❌ 不显著"
        print(f"  ---")
        print(f"  L/S 日均 (Q5-Q1): {ls_mean:+.4%}")
        print(f"  L/S t-stat:       {ls_t:+.2f}  {ls_sig}")
        print("-" * 65)

        # ── 7c. Top-N Overlap ──
        codes_np = df_signal["code"].to_numpy()
        print()
        for top_n in [20, 50]:
            prev_top = None
            overlaps = []
            for idx in range(n_days):
                s, e = date_start[idx], date_end[idx]
                sc = scores_np[s:e]
                cd = codes_np[s:e]
                if len(sc) < top_n:
                    continue
                top_idx = np.argsort(sc)[-top_n:]
                cur_top = set(cd[top_idx].tolist())
                if prev_top is not None:
                    overlap = len(cur_top & prev_top) / top_n
                    overlaps.append(overlap)
                prev_top = cur_top

            if overlaps:
                mean_ol = np.mean(overlaps)
                turnover = 1 - mean_ol
                print(f"  Top-{top_n} 日均重叠率: {mean_ol:.1%}, 日均换手: {turnover*2:.1%}")

        print()
        # return ic_results

    run_signal_quality()
    return


@app.cell
def _(calc_b1_factors_wmacd, df_scores_raw, export_for_rust, pl, q_full):
    # ==============================================================================
    # Cell 6: 导出 ML 分数 → Rust B1 回测
    #
    # 流程:
    #   1. calc_b1_factors_wmacd() → 生成 b1_signal / WL / YL 等
    #   2. left-join ML scores → 每行附带模型分数
    #   3. export_for_rust() → 导出 parquet, Rust 用 score 排序候选
    # ==============================================================================
    def run_export():
        LOOSE_PERIODS = [
            ("2019-02-11", "2019-04-10"),
            ("2019-12-16", "2020-03-02"),
            ("2020-06-19", "2020-07-15"),
            ("2020-12-24", "2021-01-25"),
            ("2021-04-16", "2021-09-14"),
            ("2022-04-27", "2022-07-05"),
            ("2023-01-15", "2023-04-15"),
            ("2024-02-06", "2024-03-20"),
            ("2024-09-24", "2024-10-15"),
            ("2025-04-09", "2025-09-04"),
            ("2026-01-05", "2026-02-02"),
        ]

        print("⏳ [Step 6] 生成 B1 信号 (wmacd)...")
        df_b1 = calc_b1_factors_wmacd(q_full, {"MV_THRESHOLD": 40})

        print("⏳ [Step 6] 合并 ML 分数...")
        df_b1_scored = (
            df_b1
            .join(
                df_scores_raw.select(["date", "code", "score"]).lazy(),
                on=["date", "code"],
                how="left",
            )
            .with_columns(pl.col("score").fill_null(0.0))
        )

        n_signals = df_b1_scored.filter(pl.col("b1_signal")).select(pl.len()).collect().item()
        n_with_score = (
            df_b1_scored
            .filter(pl.col("b1_signal") & (pl.col("score") != 0.0))
            .select(pl.len()).collect().item()
        )
        print(f"   B1 信号总数: {n_signals:,}, 有 ML 分数: {n_with_score:,} ({n_with_score/max(n_signals,1):.1%})")

        output_path = export_for_rust(
            df_b1_scored,
            output_path="data/signals/market_data_b1ml.parquet",
            loose_periods=LOOSE_PERIODS,
            start_date="2023-01-01",
            extra_sort_cols=["score"],
        )
        print(f"\n   🎯 Rust 回测命令:")
        print(f'   cargo run -p bt-b1 --release -- --data ../../{output_path} --config crates/b1/config_wmacd.toml')
        print(f'   (记得把 config 里 sort_field 改为 "score", sort_ascending = false)')

    run_export()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
