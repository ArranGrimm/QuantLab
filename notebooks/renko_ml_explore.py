import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import polars as pl
    import numpy as np
    from scipy import stats

    from utils import load_daily_data_full
    from utils import get_st_blacklist_pl
    from utils import calc_renko_factors_wmacd
    from utils.signal_export import export_renko_scores
    from utils.rotation_factors import calc_rotation_factors, FACTOR_COLS as ROTATION_FACTORS
    from utils.ic_analysis import calc_factor_ic

    # ==============================================================================
    # Cell 1: 配置与数据加载
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    START_DATE = "2020-09-01"
    LABEL = "fwd_ret_open_2d"  # 可选: fwd_ret_open_2d / fwd_ret_close_2d / fwd_ret_close_3d
    ANALYSIS_EMA_ALPHAS = [1.0, 0.2, 0.1, 0.05]
    ANALYSIS_TOP_NS = [20, 50, 100]
    ANALYSIS_SCORE_QUANTILES = [0.99, 0.97, 0.95, 0.90]

    print("🚀 [Step 1] 加载全量日线数据...")
    st_blacklist = get_st_blacklist_pl()
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
    )

    print(f"✅ 参数: 流通市值 {MV_MIN}~{MV_MAX}亿, 上市>{MIN_LIST_DAYS}天, 起始={START_DATE}")
    return (
        ANALYSIS_EMA_ALPHAS,
        ANALYSIS_SCORE_QUANTILES,
        ANALYSIS_TOP_NS,
        LABEL,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        ROTATION_FACTORS,
        calc_factor_ic,
        calc_renko_factors_wmacd,
        calc_rotation_factors,
        export_renko_scores,
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
    calc_renko_factors_wmacd,
    calc_rotation_factors,
    pl,
    q_full,
):
    # ==============================================================================
    # Cell 2: 因子计算 (通用因子 + Renko 专属指标 + 入场对齐标签)
    #
    # 统一时钟:
    #   T 日收盘确认信号 / 计算特征
    #   T+1 日开盘买入
    #   标签一律以 T+1 open 为基准
    #
    # 两条管线:
    #   A) rotation 通用因子 (T 日) + Renko 10 专属指标 (T 日)
    #   B) calc_renko_factors_wmacd() → renko_signal + 中间变量
    # ==============================================================================
    print("⏳ [Step 2] 计算因子 (rotation通用 + Renko指标)...")

    MFE_WINDOW = 5

    # ── A) rotation 通用因子 ──
    df_factors = calc_rotation_factors(q_full)

    # ── B) Renko 中间变量 (砖型图值、周/月 MACD 等) ──
    print("⏳ [Step 2] 计算 Renko wmacd 信号...")
    df_renko = calc_renko_factors_wmacd(q_full, {"MV_THRESHOLD": MV_MIN})

    renko_cols = df_renko.select([
        "code", "date",
        "renko", "prev_renko",
        "renko_falling",
        "WL", "YL",
        "rw_dif", "rw_hist",
        "rm_hist",
        "renko_signal",
    ])

    # ── C) Renko 专属连续值指标 ──
    # 与交易时钟统一: 全部使用 T 日收盘可确认的数据
    df_with_renko = (
        df_factors
        .join(renko_cols, on=["code", "date"], how="left")

        .with_columns([
            pl.col("renko").fill_null(0.0).alias("rk_value"),
            (pl.col("renko").fill_null(0.0) - pl.col("prev_renko").fill_null(0.0))
                .alias("rk_delta"),
            # 连续绿砖天数 (rolling count of falling)
            pl.col("renko_falling").fill_null(False).cast(pl.Int32)
              .rolling_sum(20).over("code").alias("rk_green_days"),
        ])

        # WL/YL 相关
        .with_columns([
            ((pl.col("close_adj") - pl.col("WL").fill_null(pl.col("close_adj")))
             / pl.max_horizontal(pl.col("WL").fill_null(pl.col("close_adj")), pl.lit(0.01)) * 100)
                .alias("rk_bias_wl"),
            ((pl.col("WL").fill_null(pl.col("close_adj")) - pl.col("YL").fill_null(pl.col("close_adj")))
             / pl.max_horizontal(pl.col("YL").fill_null(pl.col("close_adj")), pl.lit(0.01)) * 100)
                .alias("rk_wl_yl_spread"),
        ])

        # K 线形态
        .with_columns(
            ((pl.col("close_adj") - pl.col("open_adj")).abs()
             / pl.max_horizontal(pl.col("open_adj"), pl.lit(0.01)))
                .alias("rk_shape"),
        )

        # 周/月 MACD
        .with_columns([
            (pl.col("rw_dif").fill_null(0.0)
             / pl.max_horizontal(pl.col("close_adj"), pl.lit(0.01)) * 100)
                .alias("rk_rw_dif_pct"),
            pl.col("rw_hist").fill_null(0.0).alias("rk_rw_hist"),
            pl.col("rm_hist").fill_null(0.0).alias("rk_rm_hist"),
        ])

        # 量能萎缩
        .with_columns(
            pl.col("volume").rolling_max(20).over("code").alias("_v_max_20_rk"),
        )
        .with_columns(
            (pl.col("volume") / pl.max_horizontal(pl.col("_v_max_20_rk"), pl.lit(1.0)))
                .alias("rk_vol_shrink"),
        )
    )

    # ── D) 入场对齐标签: 信号 T -> 买入 T+1 open ──
    fwd_high_cols = [
        pl.col("high_adj").shift(-i).over("code").alias(f"_fwd_high_{i}")
        for i in range(1, MFE_WINDOW + 1)
    ]
    df_with_label = (
        df_with_renko
        .with_columns([
            pl.col("open_adj").shift(-1).over("code").alias("buy_open_t1"),
            pl.col("open_adj").shift(-2).over("code").alias("open_t2"),
            pl.col("close_adj").shift(-1).over("code").alias("close_t1"),
            pl.col("close_adj").shift(-2).over("code").alias("close_t2"),
            pl.col("close_adj").shift(-3).over("code").alias("close_t3"),
            pl.col("close_adj").shift(-MFE_WINDOW).over("code").alias(f"close_t{MFE_WINDOW}"),
            *fwd_high_cols,
        ])
        .with_columns(
            pl.max_horizontal([f"_fwd_high_{i}" for i in range(1, MFE_WINDOW + 1)])
              .alias("_fwd_high_max")
        )
        .with_columns([
            (pl.col("_fwd_high_max") / pl.col("buy_open_t1") - 1).alias("fwd_mfe_5d"),
            (pl.col("close_t1") / pl.col("buy_open_t1") - 1)
                .alias("fwd_ret_1d"),
            (pl.col("open_t2") / pl.col("buy_open_t1") - 1)
                .alias("fwd_ret_open_2d"),
            (pl.col("close_t2") / pl.col("buy_open_t1") - 1)
                .alias("fwd_ret_close_2d"),
            (pl.col("close_t3") / pl.col("buy_open_t1") - 1)
                .alias("fwd_ret_close_3d"),
            (pl.col(f"close_t{MFE_WINDOW}") / pl.col("buy_open_t1") - 1)
                .alias(f"fwd_ret_{MFE_WINDOW}d"),
        ])
    )

    # ── E) 市值过滤 + 截面标准化 ──
    RENKO_FACTORS = [
        "rk_value", "rk_delta", "rk_green_days",
        "rk_bias_wl", "rk_wl_yl_spread",
        "rk_shape",
        "rk_rw_dif_pct", "rk_rw_hist", "rk_rm_hist",
        "rk_vol_shrink",
    ]
    ALL_FACTORS = list(ROTATION_FACTORS) + RENKO_FACTORS

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

    df_final = df_normalized.with_columns(
        pl.col("renko_signal").fill_null(False).alias("is_renko")
    )

    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount", "market_cap_100m",
        "buy_open_t1",
        "fwd_mfe_5d", "fwd_ret_1d", "fwd_ret_open_2d", "fwd_ret_close_2d", "fwd_ret_close_3d",
        f"fwd_ret_{MFE_WINDOW}d",
        *ALL_FACTORS,
        "is_renko",
        "renko_falling",
    ]

    print("⏳ [Step 2] Collecting...")
    df_all = df_final.select(final_cols).collect()

    n_total = df_all.shape[0]
    n_renko = df_all.filter(pl.col("is_renko")).shape[0]
    print(f"✅ 数据集: {n_total:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    print(f"   通用因子: {len(ROTATION_FACTORS)}, Renko因子: {len(RENKO_FACTORS)}, 总计: {len(ALL_FACTORS)}")
    print("   ────────────────────────────────")
    print(f"   🎯 Renko 信号样本: {n_renko:,} 条 ({n_renko/n_total:.2%})")
    if n_renko > 0:
        renko_dates = df_all.filter(pl.col("is_renko"))["date"]
        print(f"   Renko 信号日期范围: {renko_dates.min()} ~ {renko_dates.max()}")
        n_renko_dates = renko_dates.n_unique()
        print(f"   Renko 信号覆盖天数: {n_renko_dates}")
        print(f"   Renko 日均信号数: {n_renko/n_renko_dates:.1f}")
    return ALL_FACTORS, df_all


@app.cell
def _(ALL_FACTORS, LABEL, calc_factor_ic, df_all, pl):
    # ==============================================================================
    # Cell 3: 快速 IC 分析 (Polars 原生)
    # ==============================================================================
    df_valid = df_all.filter(
        pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan()
    )

    calc_factor_ic(
        df_valid,
        factor_cols=ALL_FACTORS,
        label=LABEL,
        min_samples=30,
        prefix_highlight="rk_",
    )
    return


@app.cell
def _(ALL_FACTORS, LABEL, df_all, np, pl):
    # ==============================================================================
    # Cell 4: LightGBM Walk-Forward 训练
    #
    # 全市场训练, 标签由 Cell 1 配置控制
    # 只输出原始分数; EMA 平滑交给 Cell 5 / Cell 6 独立处理
    # ==============================================================================
    from lightgbm import LGBMRegressor

    def run_lgbm_walkforward():
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        TRAIN_WINDOW = 480
        RETRAIN_FREQ = 20
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

        # 预计算每日的行范围 (避免逐日 mask)
        date_start = np.searchsorted(dates_all, np.array(all_dates, dtype=dates_all.dtype), side="left")
        date_end = np.searchsorted(dates_all, np.array(all_dates, dtype=dates_all.dtype), side="right")

        score_dates, score_codes, score_values = [], [], []
        model = None
        last_train_idx = TRAIN_WINDOW - RETRAIN_FREQ

        for i in range(TRAIN_WINDOW, n_dates):
            cur_date = all_dates[i]

            if i - last_train_idx >= RETRAIN_FREQ or model is None:
                ts = date_start[max(0, i - TRAIN_WINDOW)]
                te = date_end[i - 1]
                X_tr = X_all[ts:te]
                y_tr = y_all[ts:te]

                valid_mask = np.isfinite(y_tr)
                if valid_mask.sum() < 1000:
                    continue

                model = LGBMRegressor(**lgb_params)
                model.fit(X_tr[valid_mask], y_tr[valid_mask], feature_name=feature_cols)
                last_train_idx = i

                pct = (i - TRAIN_WINDOW) / (n_dates - TRAIN_WINDOW) * 100
                print(f"   [{cur_date}] 重训 ({pct:.0f}%), 样本: {valid_mask.sum():,}", flush=True)

            if model is None:
                continue

            s, e = date_start[i], date_end[i]
            X_te = X_all[s:e]
            codes_te = codes_all[s:e]
            n_stocks = e - s

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
                prefix = "🔵" if row["factor"].startswith("rk_") else "  "
                print(f"{prefix}{row['factor']:<22} {row['importance']:>6} {bar}")
            print("=" * 55)

        return df_scores

    df_scores_raw = run_lgbm_walkforward()
    return (df_scores_raw,)


@app.cell
def _(LABEL, df_all, df_scores_raw, np, pl, stats):
    # ==============================================================================
    # Cell 5: 信号质量分析
    # ==============================================================================
    def run_signal_quality():
        EMA_ALPHA = 1.0

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
        print("  Model IC Analysis")
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
        print("  ---")
        print(f"  L/S 日均 (Q5-Q1): {ls_mean:+.4%}")
        print(f"  L/S t-stat:       {ls_t:+.2f}  {ls_sig}")
        print("-" * 65)

        # ── Top-N Overlap ──
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

    run_signal_quality()
    return


@app.cell
def _(
    ANALYSIS_EMA_ALPHAS,
    ANALYSIS_SCORE_QUANTILES,
    ANALYSIS_TOP_NS,
    LABEL,
    df_all,
    df_scores_raw,
    np,
    pl,
    stats,
):
    # ==============================================================================
    # Cell 5b: Renko 实验面板
    #
    # 1. EMA 平滑实验
    # 2. Top-N 扩大实验
    # 3. 高分阈值过滤实验
    # ==============================================================================
    def build_signal(ema_alpha: float):
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
        if ema_alpha < 1.0:
            df_signal = (
                df_signal
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score").ewm_mean(alpha=ema_alpha).over("code").alias("score")
                )
                .sort(["date", "code"])
            )
        return df_signal

    def to_numpy_views(df_signal):
        dates_np = df_signal["date"].to_numpy()
        scores_np = df_signal["score"].to_numpy().astype(np.float64)
        rets_np = df_signal[LABEL].to_numpy().astype(np.float64)
        codes_np = df_signal["code"].to_numpy()
        unique_dates = np.unique(dates_np)
        unique_dates.sort()
        date_start = np.searchsorted(dates_np, unique_dates, side="left")
        date_end = np.searchsorted(dates_np, unique_dates, side="right")
        return dates_np, scores_np, rets_np, codes_np, unique_dates, date_start, date_end

    def calc_core_metrics(df_signal, top_n_for_overlap=20):
        _, scores_np, rets_np, codes_np, unique_dates, date_start, date_end = to_numpy_views(df_signal)
        n_days = len(unique_dates)

        daily_ics = []
        quintile_rets = {q: [] for q in range(1, 6)}
        prev_top = None
        overlaps = []

        for idx in range(n_days):
            s, e = date_start[idx], date_end[idx]
            sc = scores_np[s:e]
            rt = rets_np[s:e]
            cd = codes_np[s:e]
            valid = np.isfinite(sc) & np.isfinite(rt)

            if valid.sum() >= 30:
                ic, _ = stats.spearmanr(sc[valid], rt[valid])
                if np.isfinite(ic):
                    daily_ics.append(ic)

            if valid.sum() >= 50:
                sc_v, rt_v = sc[valid], rt[valid]
                order = np.argsort(sc_v)
                n = len(order)
                group_size = n // 5
                for q in range(5):
                    start_idx = q * group_size
                    end_idx = (q + 1) * group_size if q < 4 else n
                    grp = rt_v[order[start_idx:end_idx]]
                    if len(grp) > 0:
                        quintile_rets[q + 1].append(float(np.mean(grp)))

            if valid.sum() >= top_n_for_overlap:
                sc_v = sc[valid]
                cd_v = cd[valid]
                top_idx = np.argsort(sc_v)[-top_n_for_overlap:]
                cur_top = set(cd_v[top_idx].tolist())
                if prev_top is not None:
                    overlaps.append(len(cur_top & prev_top) / top_n_for_overlap)
                prev_top = cur_top

        ic_arr = np.array(daily_ics) if daily_ics else np.array([np.nan])
        ic_mean = float(np.nanmean(ic_arr))
        ic_std = float(np.nanstd(ic_arr))
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ics))) if ic_std > 1e-8 and daily_ics else 0.0

        q1 = float(np.mean(quintile_rets[1])) if quintile_rets[1] else np.nan
        q5 = float(np.mean(quintile_rets[5])) if quintile_rets[5] else np.nan
        ls_arr = np.array(quintile_rets[5]) - np.array(quintile_rets[1]) if quintile_rets[1] and quintile_rets[5] else np.array([])
        ls_mean = float(np.mean(ls_arr)) if len(ls_arr) > 0 else np.nan
        ls_t = float(ls_mean / (np.std(ls_arr) / np.sqrt(len(ls_arr)))) if len(ls_arr) > 1 and np.std(ls_arr) > 1e-8 else 0.0
        overlap = float(np.mean(overlaps)) if overlaps else np.nan
        turnover = (1 - overlap) * 2 if overlaps else np.nan

        return {
            "ic_mean": ic_mean,
            "icir": icir,
            "t_stat": t_stat,
            "q1": q1,
            "q5": q5,
            "ls_mean": ls_mean,
            "ls_t": ls_t,
            "overlap": overlap,
            "turnover": turnover,
        }

    def print_ema_experiment():
        print("\n" + "=" * 78)
        print(f"  实验 1: EMA 平滑 ({LABEL})")
        print("=" * 78)
        print(f"  {'EMA':<8} {'ICMean':>8} {'ICIR':>8} {'L/S日均':>10} {'L/S t':>8} {'Top20重叠':>10} {'换手':>8}")
        print("-" * 78)
        for alpha in ANALYSIS_EMA_ALPHAS:
            df_signal = build_signal(alpha)
            m = calc_core_metrics(df_signal, top_n_for_overlap=20)
            print(
                f"  {alpha:<8.2f} {m['ic_mean']:>+8.4f} {m['icir']:>+8.4f} "
                f"{m['ls_mean']*100:>+9.4f}% {m['ls_t']:>+8.2f} "
                f"{m['overlap']*100:>9.1f}% {m['turnover']*100:>7.1f}%"
            )
        print("-" * 78)

    def print_topn_experiment():
        ema_for_topn = 0.1 if 0.1 in ANALYSIS_EMA_ALPHAS else ANALYSIS_EMA_ALPHAS[0]
        df_signal = build_signal(ema_for_topn)
        _, scores_np, rets_np, codes_np, unique_dates, date_start, date_end = to_numpy_views(df_signal)

        print("\n" + "=" * 78)
        print(f"  实验 2: Top-N 扩大 ({LABEL}, EMA={ema_for_topn})")
        print("=" * 78)
        print(f"  {'TopN':<8} {'日均收益':>10} {'胜率':>8} {'重叠率':>10} {'换手':>8}")
        print("-" * 78)

        for top_n in ANALYSIS_TOP_NS:
            daily_rets = []
            prev_top = None
            overlaps = []
            for idx in range(len(unique_dates)):
                s, e = date_start[idx], date_end[idx]
                sc = scores_np[s:e]
                rt = rets_np[s:e]
                cd = codes_np[s:e]
                valid = np.isfinite(sc) & np.isfinite(rt)
                if valid.sum() < top_n:
                    continue
                sc_v = sc[valid]
                rt_v = rt[valid]
                cd_v = cd[valid]
                top_idx = np.argsort(sc_v)[-top_n:]
                daily_rets.append(float(np.mean(rt_v[top_idx])))
                cur_top = set(cd_v[top_idx].tolist())
                if prev_top is not None:
                    overlaps.append(len(cur_top & prev_top) / top_n)
                prev_top = cur_top

            arr = np.array(daily_rets)
            mean_ret = float(np.mean(arr)) if len(arr) > 0 else np.nan
            hit = float(np.mean(arr > 0)) if len(arr) > 0 else np.nan
            overlap = float(np.mean(overlaps)) if overlaps else np.nan
            turnover = (1 - overlap) * 2 if overlaps else np.nan
            print(
                f"  {top_n:<8} {mean_ret*100:>+9.4f}% {hit*100:>7.1f}% "
                f"{overlap*100:>9.1f}% {turnover*100:>7.1f}%"
            )
        print("-" * 78)

    def print_threshold_experiment():
        ema_for_threshold = 0.1 if 0.1 in ANALYSIS_EMA_ALPHAS else ANALYSIS_EMA_ALPHAS[0]
        df_signal = build_signal(ema_for_threshold)
        _, scores_np, rets_np, _, unique_dates, date_start, date_end = to_numpy_views(df_signal)

        print("\n" + "=" * 90)
        print(f"  实验 3: 高分阈值过滤 ({LABEL}, EMA={ema_for_threshold})")
        print("=" * 90)
        print(f"  {'分位阈值':<10} {'日均样本数':>10} {'日均收益':>10} {'胜率':>8} {'收益t值':>10}")
        print("-" * 90)

        for q in ANALYSIS_SCORE_QUANTILES:
            daily_counts = []
            daily_rets = []
            for idx in range(len(unique_dates)):
                s, e = date_start[idx], date_end[idx]
                sc = scores_np[s:e]
                rt = rets_np[s:e]
                valid = np.isfinite(sc) & np.isfinite(rt)
                if valid.sum() < 50:
                    continue
                sc_v = sc[valid]
                rt_v = rt[valid]
                threshold = np.quantile(sc_v, q)
                mask = sc_v >= threshold
                if mask.sum() == 0:
                    continue
                daily_counts.append(int(mask.sum()))
                daily_rets.append(float(np.mean(rt_v[mask])))

            arr = np.array(daily_rets)
            mean_count = float(np.mean(daily_counts)) if daily_counts else np.nan
            mean_ret = float(np.mean(arr)) if len(arr) > 0 else np.nan
            hit = float(np.mean(arr > 0)) if len(arr) > 0 else np.nan
            t_val = float(mean_ret / (np.std(arr) / np.sqrt(len(arr)))) if len(arr) > 1 and np.std(arr) > 1e-8 else 0.0
            print(
                f"  Top {int((1-q)*100):>2d}%     {mean_count:>10.1f} {mean_ret*100:>+9.4f}% "
                f"{hit*100:>7.1f}% {t_val:>+10.2f}"
            )
        print("-" * 90)

    print_ema_experiment()
    print_topn_experiment()
    print_threshold_experiment()
    return


@app.cell
def _(df_all, df_scores_raw, export_renko_scores, pl, q_full):
    # ==============================================================================
    # Cell 6: 导出 ML 分数 → Rust Renko 截面回测
    #
    # 时钟:
    #   T 日收盘得到 score/rank
    #   Rust 在 T+1 日开盘使用 pre_score/pre_rank 买入
    # ==============================================================================
    def run_export():
        TOP_N = 20
        EXPORT_EMA_ALPHA = 1.0  # 导出 parquet 用的分数平滑; 改这里仅需重跑 Cell 6
        df_scores_export = df_scores_raw

        if EXPORT_EMA_ALPHA < 1.0:
            df_scores_export = (
                df_scores_raw
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score").ewm_mean(alpha=EXPORT_EMA_ALPHA).over("code").alias("score")
                )
                .sort(["date", "code"])
            )
            print(f"⏳ [Step 6] 导出分数 EMA 平滑: α={EXPORT_EMA_ALPHA}")
        else:
            print("⏳ [Step 6] 导出使用原始分数 (无 EMA)")

        ever_scored_codes = df_scores_export["code"].unique().to_list()
        score_date_min = df_scores_export["date"].min()
        score_date_max = df_scores_export["date"].max()

        print("⏳ [Step 6] 补全完整价格序列...")
        print(
            f"   股票数: {len(ever_scored_codes)}, "
            f"{score_date_min} ~ {score_date_max}"
        )

        price_cols = [
            "date", "code", "open_adj", "high_adj", "low_adj", "close_adj",
            "volume", "market_cap_100m",
        ]
        df_full_prices = (
            q_full
            .filter(pl.col("code").is_in(ever_scored_codes))
            .filter(pl.col("date") >= score_date_min)
            .filter(pl.col("date") <= score_date_max)
            .select([c for c in price_cols if c in q_full.collect_schema().names()])
            .collect()
        )

        df_flags = (
            df_all
            .select(["date", "code", "is_renko"])
            .unique()
        )

        df_expanded = (
            df_full_prices
            .join(df_scores_export, on=["date", "code"], how="left")
            .join(df_flags, on=["date", "code"], how="left")
            .with_columns([
                pl.col("score").fill_null(-999.0),
                pl.col("is_renko").fill_null(False),
            ])
        )

        n_scored = df_scores_export.height
        n_total = df_expanded.height
        n_padded = n_total - n_scored
        print(f"   评分行: {n_scored:,}, 补全行: {n_padded:,}, 总计: {n_total:,}")

        output_path = export_renko_scores(df_expanded, top_n=TOP_N)
        return output_path

    scores_path = run_export()
    return


@app.cell
def _(df_all, df_scores_raw, pl):
    # ==============================================================================
    # Cell 7: 单笔交易动态退出 P&L 分析
    #
    # 退出规则 (信号日 T, 买入 T+1 open):
    #   1. T+1 收盘砖变绿 → T+2 开盘走 (持1天)
    #   2. T+2 收盘砖变绿 → T+3 开盘走 (持2天)
    #   3. 最大持仓 T+3 → T+3 收盘走 (持3天)
    # ==============================================================================
    def run_trade_analysis():
        print("=" * 70)
        print("  Renko 单笔交易动态退出分析")
        print("=" * 70)

        # ── 合并 ML 分数到信号 ──
        df_with_score = (
            df_all
            .join(
                df_scores_raw.select(["date", "code", "score"]),
                on=["date", "code"],
                how="left",
            )
            .with_columns(pl.col("score").fill_null(0.0))
            .sort(["code", "date"])
        )

        # ── 前向移位: 砖色 + 价格 ──
        df_shifted = df_with_score.with_columns([
            pl.col("renko_falling").shift(-1).over("code").alias("brick_t1"),
            pl.col("renko_falling").shift(-2).over("code").alias("brick_t2"),
            pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
            pl.col("open_adj").shift(-2).over("code").alias("exit_t2_open"),
            pl.col("open_adj").shift(-3).over("code").alias("exit_t3_open"),
            pl.col("close_adj").shift(-3).over("code").alias("exit_t3_close"),
        ])

        # ── 仅保留 Renko 信号日 ──
        df_signals = df_shifted.filter(pl.col("is_renko"))

        # ── 动态退出价 + 持仓天数 ──
        df_trades = (
            df_signals
            .with_columns([
                pl.when(pl.col("brick_t1").fill_null(True))
                  .then(pl.col("exit_t2_open"))
                  .when(pl.col("brick_t2").fill_null(True))
                  .then(pl.col("exit_t3_open"))
                  .otherwise(pl.col("exit_t3_close"))
                  .alias("exit_price"),
                pl.when(pl.col("brick_t1").fill_null(True))
                  .then(pl.lit(1))
                  .when(pl.col("brick_t2").fill_null(True))
                  .then(pl.lit(2))
                  .otherwise(pl.lit(3))
                  .alias("hold_days"),
            ])
            .with_columns(
                (pl.col("exit_price") / pl.col("buy_price") - 1).alias("trade_return")
            )
            .filter(
                pl.col("buy_price").is_not_null() &
                pl.col("exit_price").is_not_null() &
                pl.col("trade_return").is_not_nan()
            )
        )

        # ── ML 排名 (每日信号内排序) ──
        df_trades = df_trades.with_columns(
            pl.col("score")
              .rank(method="ordinal", descending=True)
              .over("date")
              .alias("ml_rank")
        )

        # ── 持仓天数分布 ──
        n_total = df_trades.height
        n_1d = df_trades.filter(pl.col("hold_days") == 1).height
        n_2d = df_trades.filter(pl.col("hold_days") == 2).height
        n_3d = df_trades.filter(pl.col("hold_days") == 3).height

        print(f"\n  总交易数: {n_total}")
        print(f"  持仓天数分布: 1天 {n_1d/n_total:.1%}, 2天 {n_2d/n_total:.1%}, 3天 {n_3d/n_total:.1%}")

        # ── 统计打印函数 ──
        def calc_metrics(df_sub, name):
            rets = df_sub["trade_return"].to_numpy()
            n = len(rets)
            if n == 0:
                return
            wins = rets[rets > 0]
            losses = rets[rets <= 0]
            win_rate = len(wins) / n
            avg_ret = rets.mean()
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-8
            odds = avg_win / max(avg_loss, 1e-8)
            expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
            print(f"  {name:<14} | {n:<6} | {win_rate:>6.1%} | {avg_ret:>+7.2%} | "
                  f"{odds:>6.2f}x | {expectancy:>+7.2%}")

        # ── 全局对比 ──
        header = f"  {'分组':<14} | {'信号数':<6} | {'胜率':>6} | {'均值':>7} | {'盈亏比':>6} | {'期望值':>7}"
        print(f"\n{header}")
        print("  " + "-" * 66)

        calc_metrics(df_trades, "全部信号")
        for top_n in [1, 2, 3]:
            df_top = df_trades.filter(pl.col("ml_rank") <= top_n)
            calc_metrics(df_top, f"ML Top-{top_n}")

        # ── 按年拆分 ──
        years = sorted(df_trades["date"].dt.year().unique().to_list())
        for year in years:
            df_year = df_trades.filter(pl.col("date").dt.year() == year)
            n_y = df_year.height
            if n_y == 0:
                continue
            print(f"\n  ── {year} 年 (信号: {n_y}) ──")
            print(f"  {'分组':<14} | {'信号数':<6} | {'胜率':>6} | {'均值':>7} | {'盈亏比':>6} | {'期望值':>7}")
            print("  " + "-" * 66)
            calc_metrics(df_year, "全部信号")
            for top_n in [1, 2, 3]:
                df_top_y = df_year.filter(pl.col("ml_rank") <= top_n)
                calc_metrics(df_top_y, f"ML Top-{top_n}")

        print("\n" + "=" * 70)

    run_trade_analysis()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
