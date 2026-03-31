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
    from utils import calc_renko_factors_wmacd
    from utils.signal_export import export_for_rust
    from utils.rotation_factors import calc_rotation_factors, FACTOR_COLS as ROTATION_FACTORS
    from utils.ic_analysis import calc_factor_ic, select_factors_by_ic

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
        calc_factor_ic,
        calc_renko_factors_wmacd,
        calc_rotation_factors,
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
    # Cell 2: 因子计算 (通用因子 + Renko 专属指标 + MFE-5 标签)
    #
    # 两条管线:
    #   A) rotation 42 通用因子 + Renko 10 专属指标 + MFE-5 标签
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
    # Renko 因子暂用 T-1 数据 (shift(1)), 时序对齐待后续讨论
    df_with_renko = (
        df_factors
        .with_columns([
            pl.col("close_adj").shift(1).over("code").alias("_c1"),
            pl.col("open_adj").shift(1).over("code").alias("_o1"),
            pl.col("volume").shift(1).over("code").alias("_v1"),
        ])
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
            ((pl.col("_c1") - pl.col("WL").fill_null(pl.col("_c1")))
             / pl.max_horizontal(pl.col("WL").fill_null(pl.col("_c1")), pl.lit(0.01)) * 100)
                .alias("rk_bias_wl"),
            ((pl.col("WL").fill_null(pl.col("_c1")) - pl.col("YL").fill_null(pl.col("_c1")))
             / pl.max_horizontal(pl.col("YL").fill_null(pl.col("_c1")), pl.lit(0.01)) * 100)
                .alias("rk_wl_yl_spread"),
        ])

        # K 线形态
        .with_columns(
            ((pl.col("_c1") - pl.col("_o1")).abs()
             / pl.max_horizontal(pl.col("_o1"), pl.lit(0.01)))
                .alias("rk_shape"),
        )

        # 周/月 MACD
        .with_columns([
            (pl.col("rw_dif").fill_null(0.0)
             / pl.max_horizontal(pl.col("_c1"), pl.lit(0.01)) * 100)
                .alias("rk_rw_dif_pct"),
            pl.col("rw_hist").fill_null(0.0).alias("rk_rw_hist"),
            pl.col("rm_hist").fill_null(0.0).alias("rk_rm_hist"),
        ])

        # 量能萎缩
        .with_columns(
            pl.col("_v1").rolling_max(20).over("code").alias("_v_max_20_rk"),
        )
        .with_columns(
            (pl.col("_v1") / pl.max_horizontal(pl.col("_v_max_20_rk"), pl.lit(1.0)))
                .alias("rk_vol_shrink"),
        )
    )

    # ── D) 前向 MFE-5 标签 ──
    fwd_cols = [
        pl.col("close_adj").shift(-i).over("code").alias(f"_fwd_{i}")
        for i in range(1, MFE_WINDOW + 1)
    ]
    df_with_label = (
        df_with_renko
        .with_columns(fwd_cols)
        .with_columns(
            pl.max_horizontal([f"_fwd_{i}" for i in range(1, MFE_WINDOW + 1)])
              .alias("_fwd_max")
        )
        .with_columns([
            (pl.col("_fwd_max") / pl.col("close_adj") - 1).alias("fwd_mfe_5d"),
            (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1)
                .alias("fwd_ret_1d"),
            (pl.col("close_adj").shift(-MFE_WINDOW).over("code") / pl.col("close_adj") - 1)
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
        "fwd_mfe_5d", "fwd_ret_1d", f"fwd_ret_{MFE_WINDOW}d",
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
    print(f"   ────────────────────────────────")
    print(f"   🎯 Renko 信号样本: {n_renko:,} 条 ({n_renko/n_total:.2%})")
    if n_renko > 0:
        renko_dates = df_all.filter(pl.col("is_renko"))["date"]
        print(f"   Renko 信号日期范围: {renko_dates.min()} ~ {renko_dates.max()}")
        n_renko_dates = renko_dates.n_unique()
        print(f"   Renko 信号覆盖天数: {n_renko_dates}")
        print(f"   Renko 日均信号数: {n_renko/n_renko_dates:.1f}")
    return ALL_FACTORS, df_all


@app.cell
def _(ALL_FACTORS, calc_factor_ic, df_all, pl):
    # ==============================================================================
    # Cell 3: 快速 IC 分析 (Polars 原生, vs MFE-5)
    # ==============================================================================
    LABEL = "fwd_mfe_5d"

    df_valid = df_all.filter(
        pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan()
    )

    ic_results = calc_factor_ic(
        df_valid,
        factor_cols=ALL_FACTORS,
        label=LABEL,
        min_samples=30,
        prefix_highlight="rk_",
    )
    return


@app.cell
def _(ALL_FACTORS, df_all, np, pl):
    # ==============================================================================
    # Cell 4: LightGBM Walk-Forward 训练
    #
    # 全市场训练, MFE-5 回归标签
    # ==============================================================================
    from lightgbm import LGBMRegressor

    def run_lgbm_walkforward():
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        LABEL = "fwd_mfe_5d"
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
                prefix = "🔵" if row["factor"].startswith("rk_") else "  "
                print(f"{prefix}{row['factor']:<22} {row['importance']:>6} {bar}")
            print("=" * 55)

        return df_scores_raw

    df_scores_raw = run_lgbm_walkforward()
    return (df_scores_raw,)


@app.cell
def _(df_all, df_scores_raw, np, pl, stats):
    # ==============================================================================
    # Cell 5: 信号质量分析
    # ==============================================================================
    def run_signal_quality():
        EMA_ALPHA = 1.0
        LABEL = "fwd_mfe_5d"

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
        print("  Model IC Analysis (vs MFE-5)")
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
        print(f"  ---")
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
def _(calc_renko_factors_wmacd, df_scores_raw, pl, q_full):
    # ==============================================================================
    # Cell 6: 导出 ML 分数 → Rust 回测 (预留)
    #
    # TODO: Rust 端暂无 Renko 回测引擎, 本 Cell 先做导出
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

        print("⏳ [Step 6] 生成 Renko 信号 (wmacd)...")
        df_renko = calc_renko_factors_wmacd(q_full, {"MV_THRESHOLD": 40})

        print("⏳ [Step 6] 合并 ML 分数...")
        df_renko_scored = (
            df_renko
            .join(
                df_scores_raw.select(["date", "code", "score"]).lazy(),
                on=["date", "code"],
                how="left",
            )
            .with_columns(pl.col("score").fill_null(0.0))
        )

        n_signals = df_renko_scored.filter(pl.col("renko_signal")).select(pl.len()).collect().item()
        n_with_score = (
            df_renko_scored
            .filter(pl.col("renko_signal") & (pl.col("score") != 0.0))
            .select(pl.len()).collect().item()
        )
        print(f"   Renko 信号总数: {n_signals:,}, 有 ML 分数: {n_with_score:,} ({n_with_score/max(n_signals,1):.1%})")
        print(f"\n   ⚠️ Rust Renko 回测引擎尚未实现, 导出功能预留")

    # run_export()
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
