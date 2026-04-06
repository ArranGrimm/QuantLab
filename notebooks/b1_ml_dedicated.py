import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import numpy as np
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
    calc_b1_factors_wmacd,
    calc_rotation_factors,
    pl,
    q_full,
):
    # ==============================================================================
    # Cell 2: 因子计算 + B1 信号标记 + MFE 标签
    #
    # 两条管线:
    #   A) rotation 42 因子 + B1 14 指标 + MFE 标签 (全市场)
    #   B) calc_b1_factors_wmacd() → b1_signal 列
    # 合并后: 每行标记是否为 B1 信号日
    # ==============================================================================
    print("⏳ [Step 2] 计算因子 + B1 信号标记...")

    MFE_WINDOW = 10

    # ── A) rotation 通用因子 ──
    df_factors = calc_rotation_factors(q_full)

    # ── B) B1 原始连续值指标 ──
    # B1 需要 T-1 数据 (shift(1)), rotation_factors 已改为 T 日数据, 此处自行计算
    df_with_b1 = (
        df_factors
        .with_columns([
            pl.col("close_adj").shift(1).over("code").alias("_c1"),
            pl.col("open_adj").shift(1).over("code").alias("_o1"),
            pl.col("high_adj").shift(1).over("code").alias("_h1"),
            pl.col("low_adj").shift(1).over("code").alias("_l1"),
            pl.col("volume").shift(1).over("code").alias("_v1"),
        ])

        # B1-1: 知行双线 (WL/YL)
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

    # ── D) B1 信号标记 ──
    print("⏳ [Step 2] 计算 B1 wmacd 信号...")
    df_b1_signals = (
        calc_b1_factors_wmacd(q_full, {"MV_THRESHOLD": MV_MIN})
        .select(["code", "date", "b1_signal"])
        .filter(pl.col("b1_signal"))
    )

    # ── E) 市值过滤 + 截面标准化 + 信号 join ──
    B1_FACTORS = [
        "b1_bias_c_wl", "b1_bias_c_yl", "b1_wl_yl_spread",
        "b1_K", "b1_D", "b1_J", "b1_J_delta",
        "b1_vol_shrink_20", "b1_vol_shrink_40", "b1_vol_rel_ma5",
        "b1_yang_yin_ratio",
        "b1_body_pct", "b1_lower_shadow",
        "b1_retrace_ratio",
    ]
    ALL_FACTORS = list(ROTATION_FACTORS) + B1_FACTORS

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

    # join b1_signal (left join, 大部分行 is_b1 = False)
    df_final = (
        df_normalized
        .select(final_cols)
        .join(df_b1_signals, on=["code", "date"], how="left")
        .with_columns(pl.col("b1_signal").fill_null(False).alias("is_b1"))
    )

    print("⏳ [Step 2] Collecting...")
    df_all = df_final.select([*final_cols, "is_b1"]).collect()

    n_total = df_all.shape[0]
    n_b1 = df_all.filter(pl.col("is_b1")).shape[0]
    b1_dates = df_all.filter(pl.col("is_b1"))["date"]
    print(f"✅ 数据集: {n_total:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    print(f"   通用因子: {len(ROTATION_FACTORS)}, B1因子: {len(B1_FACTORS)}, 总计: {len(ALL_FACTORS)}")
    print(f"   ────────────────────────────────")
    print(f"   🎯 B1 信号样本: {n_b1:,} 条 ({n_b1/n_total:.2%})")
    if n_b1 > 0:
        print(f"   B1 信号日期范围: {b1_dates.min()} ~ {b1_dates.max()}")
        n_b1_dates = df_all.filter(pl.col("is_b1"))["date"].n_unique()
        print(f"   B1 信号覆盖天数: {n_b1_dates}")
        print(f"   B1 日均信号数: {n_b1/n_b1_dates:.1f}")
    return ALL_FACTORS, df_all


@app.cell
def _(ALL_FACTORS, df_all, pl):
    # ==============================================================================
    # Cell 3: 单因子 IC 分析 (Polars 原生, 仅 B1 信号日样本 vs MFE-10)
    # ==============================================================================
    from utils.ic_analysis import calc_factor_ic, select_factors_by_ic

    LABEL = "fwd_mfe_10d"

    df_b1 = df_all.filter(
        pl.col("is_b1") &
        pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan()
    )
    print(f"   B1 有效样本: {df_b1.shape[0]:,} 条")

    ic_results = calc_factor_ic(
        df_b1,
        factor_cols=ALL_FACTORS,
        label=LABEL,
        min_samples=5,
        prefix_highlight="b1_",
    )

    selected_factors = select_factors_by_ic(ic_results, t_threshold=1.5)
    return (selected_factors,)


@app.cell
def _(ALL_FACTORS, df_all, np, pl, selected_factors):
    # ==============================================================================
    # Cell 4: LightGBM Walk-Forward (B1 专属训练)
    #
    # 训练: 仅用 is_b1=True 的样本
    # 推理: 对当日所有 is_b1=True 的股票打分
    # ==============================================================================
    from lightgbm import LGBMRegressor

    def run_lgbm_b1_dedicated():
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        LABEL = "fwd_mfe_10d"
        TRAIN_WINDOW = 720
        RETRAIN_FREQ = 20
        USE_SELECTED = True

        feature_cols = list(selected_factors) if USE_SELECTED else list(ALL_FACTORS)

        lgb_params = {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "min_child_samples": 20,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        print(f"🤖 LightGBM B1-Dedicated Walk-Forward")
        print(f"   标签: {LABEL}, 窗口: {TRAIN_WINDOW}天, 特征数: {len(feature_cols)}")
        print(f"   特征选择: {'IC筛选 ' + str(len(feature_cols)) + '个' if USE_SELECTED else '全部 ' + str(len(feature_cols)) + '个'}")

        # 全市场日期序列 (用于控制 walk-forward 节奏)
        df_valid = (
            df_all
            .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())
            .sort("date")
        )
        all_dates = df_valid["date"].unique().sort().to_list()
        n_dates = len(all_dates)

        # B1 子集: 用于训练
        df_b1 = df_valid.filter(pl.col("is_b1"))
        X_b1 = df_b1.select(feature_cols).to_numpy().astype(np.float32)
        y_b1 = df_b1[LABEL].to_numpy().astype(np.float64)
        dates_b1 = df_b1["date"].to_numpy()
        codes_b1 = df_b1["code"].to_numpy()
        np.nan_to_num(X_b1, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # 全市场: 用于推理 (只对 B1 信号日推理)
        X_all = df_valid.select(feature_cols).to_numpy().astype(np.float32)
        dates_all = df_valid["date"].to_numpy()
        codes_all = df_valid["code"].to_numpy()
        is_b1_all = df_valid["is_b1"].to_numpy()
        np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"   B1 训练样本总量: {len(X_b1):,}")
        print(f"   全市场推理样本: {len(X_all):,} (其中 B1: {is_b1_all.sum():,})")

        score_dates, score_codes, score_values = [], [], []
        model = None
        last_train_idx = TRAIN_WINDOW - RETRAIN_FREQ
        train_count = 0

        for i in range(TRAIN_WINDOW, n_dates):
            cur_date = all_dates[i]

            if i - last_train_idx >= RETRAIN_FREQ or model is None:
                train_start = all_dates[max(0, i - TRAIN_WINDOW)]
                mask_tr = (dates_b1 >= np.datetime64(train_start)) & (dates_b1 < np.datetime64(cur_date))
                X_tr, y_tr = X_b1[mask_tr], y_b1[mask_tr]

                valid_mask = np.isfinite(y_tr)
                n_valid = valid_mask.sum()
                if n_valid < 50:
                    continue
                X_tr, y_tr = X_tr[valid_mask], y_tr[valid_mask]

                model = LGBMRegressor(**lgb_params)
                model.fit(X_tr, y_tr, feature_name=feature_cols)
                last_train_idx = i
                train_count += 1

                pct = (i - TRAIN_WINDOW) / (n_dates - TRAIN_WINDOW) * 100
                print(f"   [{cur_date}] 重训 ({pct:.0f}%), B1样本: {n_valid:,}", flush=True)

            if model is None:
                continue

            # 推理: 只对当日 B1 信号股打分
            mask_te = (dates_all == np.datetime64(cur_date)) & is_b1_all
            if mask_te.sum() == 0:
                continue

            X_te = X_all[mask_te]
            codes_te = codes_all[mask_te]

            preds = model.predict(X_te)
            score_dates.extend([cur_date] * len(preds))
            score_codes.extend(codes_te.tolist())
            score_values.extend(preds.tolist())

        print(f"\n   ✅ 打分完成: {len(score_values):,} 条 B1 信号记录, 重训 {train_count} 次", flush=True)

        df_scores = pl.DataFrame({
            "date": score_dates,
            "code": score_codes,
            "score": score_values,
        })

        # 特征重要性
        if model is not None:
            imp_vals = model.feature_importances_
            imp_max = max(imp_vals) if max(imp_vals) > 0 else 1
            imp_df = pl.DataFrame({
                "factor": feature_cols,
                "importance": imp_vals.tolist(),
            }).sort("importance", descending=True)

            print("\n" + "=" * 55)
            print(f"  LightGBM 特征重要性 (B1专属, {len(feature_cols)} 个)")
            print("=" * 55)
            for row in imp_df.iter_rows(named=True):
                bar_len = int(row["importance"] / imp_max * 30)
                bar = "█" * bar_len
                prefix = "🔵" if row["factor"].startswith("b1_") else "  "
                print(f"{prefix}{row['factor']:<22} {row['importance']:>6} {bar}")
            print("=" * 55)

        return df_scores

    df_scores_b1 = run_lgbm_b1_dedicated()
    return (df_scores_b1,)


@app.cell
def _(df_all, df_scores_b1, np, pl, stats):
    # ==============================================================================
    # Cell 5: 信号质量分析 (B1 专属模型)
    #
    # 只在 B1 信号日样本上评估 IC / Quintile
    # ==============================================================================
    def run_signal_quality_b1():
        LABEL = "fwd_mfe_10d"

        df_signal = (
            df_scores_b1
            .select(["date", "code", "score"])
            .join(
                df_all.select(["date", "code", LABEL]),
                on=["date", "code"],
                how="inner",
            )
            .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())
            .sort(["date", "code"])
        )

        dates_np = df_signal["date"].to_numpy()
        scores_np = df_signal["score"].to_numpy().astype(np.float64)
        rets_np = df_signal[LABEL].to_numpy().astype(np.float64)
        codes_np = df_signal["code"].to_numpy()

        unique_dates = np.unique(dates_np)
        unique_dates.sort()
        n_days = len(unique_dates)
        date_start = np.searchsorted(dates_np, unique_dates, side="left")
        date_end = np.searchsorted(dates_np, unique_dates, side="right")

        # ── IC Analysis ──
        daily_ics = []
        for idx in range(n_days):
            s, e = date_start[idx], date_end[idx]
            sc, rt = scores_np[s:e], rets_np[s:e]
            valid = np.isfinite(sc) & np.isfinite(rt)
            if valid.sum() >= 5:
                ic, _ = stats.spearmanr(sc[valid], rt[valid])
                if np.isfinite(ic):
                    daily_ics.append(ic)

        ic_arr = np.array(daily_ics)
        ic_mean = ic_arr.mean()
        ic_std = ic_arr.std()
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 1e-8 else 0

        print(f"\n📊 B1 专属模型 Signal Quality (标签: {LABEL})")
        print(f"   样本: {len(scores_np):,} 条 B1 信号, {n_days} 个交易日\n")
        print("=" * 65)
        print("  Model IC Analysis (vs MFE-10, B1 subset)")
        print("=" * 65)
        print(f"  IC Mean:       {ic_mean:+.4f}")
        print(f"  IC Std:        {ic_std:.4f}")
        print(f"  ICIR:          {icir:+.4f}")
        t_sig = "✅ 显著 (>2)" if abs(t_stat) > 2 else "❌ 不显著"
        print(f"  t-stat:        {t_stat:+.2f}  {t_sig}")
        print(f"  IC > 0 占比:   {(ic_arr > 0).mean():.1%}")
        print(f"  有效天数:      {len(ic_arr)} / {n_days}")
        print("-" * 65)

        # ── Quintile Analysis ──
        # B1 信号日每天可能只有几只到几十只，最少 5 只才做 quintile
        print("\n" + "=" * 65)
        print("  Quintile Analysis (Q5 做多 - Q1 做空, B1 subset)")
        print("=" * 65)
        quintile_rets = {q: [] for q in range(1, 6)}
        for idx in range(n_days):
            s, e = date_start[idx], date_end[idx]
            sc, rt = scores_np[s:e], rets_np[s:e]
            valid = np.isfinite(sc) & np.isfinite(rt)
            if valid.sum() < 5:
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
            if quintile_rets[q]:
                arr = np.array(quintile_rets[q])
                print(f"  Q{q} ({q_labels[q]}) 日均MFE: {arr.mean():+.3%}  (n={len(arr)}天)")
        if quintile_rets[5] and quintile_rets[1]:
            min_len = min(len(quintile_rets[5]), len(quintile_rets[1]))
            ls_arr = np.array(quintile_rets[5][:min_len]) - np.array(quintile_rets[1][:min_len])
            ls_mean = ls_arr.mean()
            ls_std = ls_arr.std()
            ls_t = ls_mean / (ls_std / np.sqrt(len(ls_arr))) if ls_std > 1e-8 else 0
            ls_sig = "✅ 显著" if abs(ls_t) > 2 else "❌ 不显著"
            print(f"  ---")
            print(f"  L/S 日均 (Q5-Q1): {ls_mean:+.4%}")
            print(f"  L/S t-stat:       {ls_t:+.2f}  {ls_sig}")
        print("-" * 65)

        # ── Top-N Overlap ──
        print()
        for top_n in [3, 5]:
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

        # ── 每日 B1 信号数量分布 ──
        daily_counts = []
        for idx in range(n_days):
            s, e = date_start[idx], date_end[idx]
            daily_counts.append(e - s)
        dc = np.array(daily_counts)
        print(f"\n  📈 每日 B1 信号数量分布:")
        print(f"     均值: {dc.mean():.1f}, 中位: {np.median(dc):.0f}")
        print(f"     最小: {dc.min()}, 最大: {dc.max()}")
        print(f"     有信号天数: {(dc > 0).sum()} / {n_days}")
        print()

    run_signal_quality_b1()
    return


@app.cell
def _(calc_b1_factors_wmacd, df_scores_b1, export_for_rust, pl, q_full):
    # ==============================================================================
    # Cell 6: 导出 B1 专属 ML 分数 → Rust 回测
    # ==============================================================================
    def run_export_b1():
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

        print("⏳ [Step 6] 合并 B1 专属 ML 分数...")
        df_b1_scored = (
            df_b1
            .join(
                df_scores_b1.select(["date", "code", "score"]).lazy(),
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
            output_path="data/signals/market_data_b1ml_dedicated.parquet",
            loose_periods=LOOSE_PERIODS,
            start_date="2023-01-01",
            extra_sort_cols=["score"],
        )
        print(f"\n   🎯 Rust 回测命令:")
        print(f'   cargo run -p bt-b1 --release -- --data ../../{output_path} --config crates/b1/config_wmacd_ml.toml')

    run_export_b1()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
