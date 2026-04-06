import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats

    from utils import load_daily_data_full, add_price_limit_cols
    from utils import get_st_blacklist_pl
    from manifests.rotation_feature_sets import (
        describe_rotation_feature_set,
        get_rotation_feature_set,
    )
    from utils.rotation_factors import (
        calc_rotation_factors,
        cross_section_normalize,
        FACTOR_COLS,
    )
    from utils.alpha158_factors import (
        calc_alpha158_factors,
        resolve_alpha158_group_config,
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
    NORMALIZE_MODE = "zscore"  # 可选: zscore / rank_pct / rank_gauss
    LABEL = "fwd_ret_1d"  # 可选: fwd_ret_{1/2/3/5}d / fwd_ret_{1/2/3/5}d_excess / fwd_ret_{1/2/3/5}d_rank_pct
    FEATURE_SET = "core_plus_alpha158_kbar_shape"
    selected_feature_set = get_rotation_feature_set(FEATURE_SET)
    ALPHA158_GROUP_MODE = selected_feature_set.alpha158_group_mode

    print("🚀 [Step 1] 加载全量日线数据...")
    st_blacklist = get_st_blacklist_pl("2026-03-31")
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
    )

    print(f"✅ 参数: 流通市值 {MV_MIN}~{MV_MAX} 亿, 上市>{MIN_LIST_DAYS}天, 起始={START_DATE}")
    print(f"✅ Feature Set: {describe_rotation_feature_set(FEATURE_SET)}")
    return (
        ALPHA158_GROUP_MODE,
        FACTOR_COLS,
        FEATURE_SET,
        LABEL,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        NORMALIZE_MODE,
        add_price_limit_cols,
        calc_alpha158_factors,
        calc_rotation_factors,
        cross_section_normalize,
        go,
        make_subplots,
        np,
        pl,
        q_full,
        resolve_alpha158_group_config,
        selected_feature_set,
        stats,
    )


@app.cell
def _(
    ALPHA158_GROUP_MODE,
    FACTOR_COLS,
    FEATURE_SET,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    NORMALIZE_MODE,
    add_price_limit_cols,
    calc_alpha158_factors,
    calc_rotation_factors,
    cross_section_normalize,
    pl,
    q_full,
    resolve_alpha158_group_config,
    selected_feature_set,
):
    import os

    os.environ['RUST_BACKTRACE']='full'
    # ==============================================================================
    # Cell 2: 全量因子计算 → 市值过滤 → 截面标准化
    #
    # 关键流程:
    #   q_full (全量) → 因子计算 (连续序列) → forward return (连续序列)
    #   → 市值+上市天数过滤 (确定可交易 universe)
    #   → 截面标准化 (在 universe 内按 NORMALIZE_MODE 变换)
    #
    # 因子必须在连续序列上计算, 市值过滤只决定"哪些股票可交易"
    # ==============================================================================
    print("⏳ [Step 2] 计算训练特征 (全量股票, 保证序列连续)...")

    requested_feature_cols = list(selected_feature_set.feature_cols or ())
    if not requested_feature_cols:
        raise ValueError(
            f"Feature set '{FEATURE_SET}' 没有稳定的因子清单，不能作为训练入口。"
        )

    rotation_factor_set = set(FACTOR_COLS)
    need_rotation_factors = any(
        factor_name in rotation_factor_set for factor_name in requested_feature_cols
    )
    need_alpha158_factors = bool(ALPHA158_GROUP_MODE)

    df_factors = q_full
    active_alpha158_factor_cols = []
    available_feature_cols: set[str] = set()

    print(f"   Feature Set: {FEATURE_SET}")
    print(f"   状态: {selected_feature_set.status}")
    print(f"   描述: {selected_feature_set.description}")
    if need_rotation_factors:
        print(f"   计算 Rotation 因子: {len(FACTOR_COLS)} 个")
        df_factors = calc_rotation_factors(df_factors)
        available_feature_cols.update(FACTOR_COLS)
    else:
        print("   跳过 Rotation 因子计算")

    if need_alpha158_factors and ALPHA158_GROUP_MODE:
        alpha158_train_config = resolve_alpha158_group_config(ALPHA158_GROUP_MODE)
        active_alpha158_factor_cols = list(alpha158_train_config["factor_cols"])
        available_feature_cols.update(active_alpha158_factor_cols)
        print(f"   Alpha158 分组: {alpha158_train_config['group_mode_label']}")
        print(f"   计算 Alpha158 因子: {len(active_alpha158_factor_cols)} 个")
        df_factors = calc_alpha158_factors(
            df_factors,
            use_kbar=bool(alpha158_train_config["use_kbar"]),
            price_fields=alpha158_train_config["price_fields"],
            include=alpha158_train_config["include_ops"],
        )
    else:
        print("   跳过 Alpha158 因子计算")

    missing_feature_cols = [
        factor_name
        for factor_name in requested_feature_cols
        if factor_name not in available_feature_cols
    ]
    if missing_feature_cols:
        raise ValueError(
            f"Feature set '{FEATURE_SET}' 存在未准备好的因子: {missing_feature_cols}"
        )

    active_factor_cols = requested_feature_cols
    print(f"   合计待标准化: {len(active_factor_cols)} 个")

    # Label: 在连续序列上计算 forward return, 避免因市值过滤产生的缺口
    df_with_label = df_factors.with_columns([
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_1d"),
        (pl.col("close_adj").shift(-2).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_2d"),
        (pl.col("close_adj").shift(-3).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_3d"),
        (pl.col("close_adj").shift(-5).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_5d"),
    ])

    # 涨跌停标记 (不删行, 仅打标记; 与 Rust bt-core 判定逻辑一致)
    df_with_label = add_price_limit_cols(df_with_label)

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
    print(f"   截面归一化模式: {NORMALIZE_MODE}")
    df_normalized = cross_section_normalize(
        df_universe,
        active_factor_cols,
        mode=NORMALIZE_MODE,
    )

    # 超额收益标签: 截面去均值 (在 universe 内计算, 消除市场 beta)
    base_label_cols = ["fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d"]
    df_normalized = df_normalized.with_columns([
        (pl.col(label_name) - pl.col(label_name).mean().over("date")).alias(f"{label_name}_excess")
        for label_name in base_label_cols
    ])

    # 排序化标签: 将未来收益映射为当日截面分位数 [0, 1]
    rank_pct_exprs = []
    for label_name in base_label_cols:
        valid_mask = pl.col(label_name).is_not_null() & pl.col(label_name).is_not_nan()
        valid_count = (
            pl.when(valid_mask)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .sum()
            .over("date")
            .cast(pl.Float32)
        )
        rank_expr = pl.col(label_name).rank(method="average").over("date").cast(pl.Float32)
        rank_pct_exprs.append(
            pl.when(valid_mask & (valid_count > 1))
            .then((rank_expr - 1.0) / (valid_count - 1.0))
            .otherwise(pl.lit(None, dtype=pl.Float32))
            .alias(f"{label_name}_rank_pct")
        )
    df_normalized = df_normalized.with_columns(rank_pct_exprs)

    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj", "vwap_adj",
        "pre_close_adj",
        "volume", "amount", "close_raw", "vwap_raw", "market_cap_100m",
        "circulating_capital",
        "fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d",
        "fwd_ret_1d_excess", "fwd_ret_2d_excess", "fwd_ret_3d_excess", "fwd_ret_5d_excess",
        "fwd_ret_1d_rank_pct", "fwd_ret_2d_rank_pct", "fwd_ret_3d_rank_pct", "fwd_ret_5d_rank_pct",
        "is_limit_up", "is_limit_down",
        *active_factor_cols,
    ]

    print("⏳ [Step 2] Collecting... (全量因子计算, 可能需要更长时间)")
    df_all = df_normalized.select(final_cols).collect()
    n_limit_up = df_all.filter(pl.col("is_limit_up")).height
    n_limit_down = df_all.filter(pl.col("is_limit_down")).height
    print(f"✅ 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    print(f"   涨停样本: {n_limit_up:,} ({n_limit_up/df_all.shape[0]*100:.2f}%)")
    print(f"   跌停样本: {n_limit_down:,} ({n_limit_down/df_all.shape[0]*100:.2f}%)")
    return (df_all,)


@app.cell
def _(df_all, pl):
    # ==============================================================================
    # Cell 2b: amount / volume 单位校验
    # 口径说明:
    #   - stock_daily.volume 单位 = 手
    #   - vwap_raw = amount / (volume * 100)
    #   - turnover_rate(%) = volume * 100 / circulating_capital * 100
    # ==============================================================================
    unit_metrics = (
        df_all
        .filter(
            (pl.col("volume") > 0) &
            (pl.col("amount") > 0) &
            (pl.col("close_raw") > 0) &
            pl.col("vwap_raw").is_not_null()
        )
        .select([
            (
                (pl.col("amount") / pl.col("volume") - pl.col("close_raw")).abs()
                / pl.col("close_raw")
            ).median().alias("err_if_volume_is_share"),
            (
                (pl.col("amount") / (pl.col("volume") * 100.0) - pl.col("close_raw")).abs()
                / pl.col("close_raw")
            ).median().alias("err_if_volume_is_lot"),
            (
                (pl.col("vwap_raw") - pl.col("close_raw")).abs()
                / pl.col("close_raw")
            ).median().alias("err_vwap_raw"),
        ])
    )
    unit_metrics_row = unit_metrics.row(0, named=True)
    print("🔎 [Step 2b] amount / volume 单位校验")
    print(f"   假设 volume=股: 相对误差中位数 {unit_metrics_row['err_if_volume_is_share']:.2%}")
    print(f"   假设 volume=手: 相对误差中位数 {unit_metrics_row['err_if_volume_is_lot']:.2%}")
    print(f"   当前 vwap_raw: 相对误差中位数 {unit_metrics_row['err_vwap_raw']:.2%}")
    print("   结论: stock_daily.volume 按“手”解释，turnover_rate 也必须先乘 100 还原股数。")

    if unit_metrics_row["err_vwap_raw"] > 0.02:
        raise ValueError("vwap_raw 与 close_raw 数量级不匹配，请检查 amount / volume 单位口径。")

    unit_sample = (
        df_all
        .filter(
            (pl.col("volume") > 0) &
            (pl.col("amount") > 0) &
            (pl.col("close_raw") > 0)
        )
        .select([
            "code",
            "date",
            "close_raw",
            "volume",
            "amount",
            (pl.col("amount") / pl.col("volume")).alias("px_if_volume_is_share"),
            (pl.col("amount") / (pl.col("volume") * 100.0)).alias("px_if_volume_is_lot"),
            "vwap_raw",
        ])
        .head(5)
    )
    print(unit_sample)
    return


@app.cell
def _(FEATURE_SET, selected_feature_set):
    # ==============================================================================
    # Cell 3: 训练入口说明
    # ==============================================================================
    print("ℹ️ [Step 3] 因子分析 / 因子选择已迁移到 notebooks/rotation_factor_lab.py")
    print("   当前 notebook 只负责训练入口、raw score 导出与 artifact 落盘。")
    print(f"   当前训练特征集: {FEATURE_SET} ({selected_feature_set.feature_count} 个因子)")
    print(f"   特征集状态: {selected_feature_set.status}")
    return


@app.cell
def _():
    # Cell 5: (已移除 — 旧可视化依赖线性回测, 回测逻辑已迁移至 Rust)
    return


@app.cell
def _(
    ALPHA158_GROUP_MODE,
    FEATURE_SET,
    LABEL,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    NORMALIZE_MODE,
    df_all,
    np,
    pl,
    resolve_alpha158_group_config,
    selected_feature_set,
):
    # ==============================================================================
    # Cell 6: LightGBM Walk-Forward 打分
    # 模型只负责打分, 导出与回测解耦。
    # 本 notebook 只消费 manifest 中冻结的稳定特征集。
    # ==============================================================================
    def run_lgbm_scoring():
        from datetime import datetime
        import lightgbm as lgb
        import warnings
        from utils.signal_export import build_feature_hash, build_rotation_train_run_id, get_git_commit
        warnings.filterwarnings("ignore", category=UserWarning)

        TRAIN_WINDOW = 480
        RETRAIN_FREQ = 20
        TOP_N = 20
        feature_cols = list(selected_feature_set.feature_cols or ())
        if not feature_cols:
            raise ValueError(
                f"Feature set '{FEATURE_SET}' 没有稳定因子清单，不能直接训练。"
            )

        alpha158_group_config = None
        if ALPHA158_GROUP_MODE:
            alpha158_group_config = resolve_alpha158_group_config(ALPHA158_GROUP_MODE)

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

        mode_desc = (
            f"{FEATURE_SET} -> {selected_feature_set.feature_mode} "
            f"({len(feature_cols)})"
        )
        print("🤖 LightGBM Walk-Forward 打分", flush=True)
        print(f"   训练窗口: {TRAIN_WINDOW}天, 重训: 每{RETRAIN_FREQ}天, 标签: {LABEL}", flush=True)
        print(f"   特征集: {mode_desc}, Top-{TOP_N}", flush=True)
        print(f"   特征集状态: {selected_feature_set.status}", flush=True)
        if alpha158_group_config is not None:
            print(f"   Alpha158 分组: {alpha158_group_config['group_mode_label']}", flush=True)
        print(f"   截面归一化: {NORMALIZE_MODE}", flush=True)

        df_valid_ml = (
            df_all
            .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())
            .sort("date")
        )

        X_all_np = df_valid_ml.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
        y_all_np = df_valid_ml[LABEL].to_numpy().astype(np.float32)
        dates_np = df_valid_ml["date"].to_numpy()
        codes_np = df_valid_ml["code"].to_numpy()
        is_limit_up_np = df_valid_ml["is_limit_up"].fill_null(False).to_numpy()

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

                valid = np.isfinite(y_tr) & ~is_limit_up_np[ts:te]
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

        # ── Build raw scores DataFrame ──
        df_scores_raw = pl.DataFrame({
            "date": score_dates,
            "code": score_codes,
            "score": score_values,
        })

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

        train_timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
        trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feature_hash = build_feature_hash(feature_cols)
        rotation_train_meta = {
            "strategy": "rotation",
            "label": LABEL,
            "model_name": "lightgbm",
            "feature_mode": selected_feature_set.feature_mode,
            "alpha158_group_mode": alpha158_group_config["group_mode_label"]
            if alpha158_group_config is not None
            else None,
            "alpha158_analysis_group_mode": selected_feature_set.alpha158_analysis_group_mode,
            "alpha158_top1_factors": None,
            "normalize_mode": NORMALIZE_MODE,
            "feature_hash": feature_hash,
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "train_timestamp_token": train_timestamp_token,
            "train_run_id": build_rotation_train_run_id(
                LABEL, "lightgbm", train_timestamp_token, feature_hash
            ),
            "trained_at": trained_at,
            "git_commit": get_git_commit(),
            "notebook": "notebooks/cross_section_rotation.py",
            "model_params": lgb_params,
            "train_window": TRAIN_WINDOW,
            "retrain_freq": RETRAIN_FREQ,
            "universe": {
                "mv_min": MV_MIN,
                "mv_max": MV_MAX,
                "min_list_days": MIN_LIST_DAYS,
            },
        }

        return df_scores_raw, rotation_train_meta

    df_scores_raw, rotation_train_meta = run_lgbm_scoring()
    return df_scores_raw, rotation_train_meta


@app.cell
def _(df_scores_raw, pl, q_full, rotation_train_meta):
    # ==============================================================================
    # Cell 6b: 导出 Rotation 分数 → Rust 回测
    #
    # 基于 Cell 6 输出的原始分数 (df_scores_raw), 独立控制导出侧 EMA.
    # 修改 EXPORT_EMA_ALPHA 只需重跑本 Cell, 无需重新训练模型.
    #
    # 关键: Parquet 必须包含所有"曾被评分"股票在整个回测期间的价格数据,
    # 即使某天该股票不在 universe 内 (市值越界等), 也要保留价格行,
    # 否则 Rust 引擎无法对该仓位执行止损/排名退出等检查 → "幽灵仓位"
    # ==============================================================================
    from utils.signal_export import export_rotation_scores

    def run_export():
        EXPORT_TOP_N = 20
        EXPORT_EMA_ALPHA = 0.30  # 导出 parquet 用的分数平滑; 改这里仅需重跑本 Cell

        df_scores_export = df_scores_raw

        if EXPORT_EMA_ALPHA < 1.0:
            df_scores_export = (
                df_scores_raw
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score")
                      .ewm_mean(alpha=EXPORT_EMA_ALPHA)
                      .over("code")
                      .alias("score")
                )
                .sort(["date", "code"])
            )
            print(f"⏳ [Step 6b] 导出分数 EMA 平滑: α={EXPORT_EMA_ALPHA}", flush=True)
        else:
            print("⏳ [Step 6b] 导出使用原始分数 (无 EMA)", flush=True)

        ever_scored_codes = df_scores_export["code"].unique().to_list()
        score_date_min = df_scores_export["date"].min()
        score_date_max = df_scores_export["date"].max()

        print(f"   📦 补全价格: {len(ever_scored_codes)} 只股票, "
              f"{score_date_min} ~ {score_date_max}", flush=True)

        price_cols = ["date", "code", "open_adj", "high_adj", "low_adj",
                      "close_adj", "volume", "market_cap_100m"]
        q_full_cols = q_full.collect_schema().names()
        df_full_prices = (
            q_full
            .filter(pl.col("code").is_in(ever_scored_codes))
            .filter(pl.col("date") >= score_date_min)
            .filter(pl.col("date") <= score_date_max)
            .select([c for c in price_cols if c in q_full_cols])
            .collect()
        )

        df_expanded = df_full_prices.join(
            df_scores_export, on=["date", "code"], how="left"
        ).with_columns(
            pl.col("score").fill_null(-999.0),
        )

        n_scored = df_scores_export.height
        n_total = df_expanded.height
        n_padded = n_total - n_scored
        print(f"   评分行: {n_scored:,}, 补全行: {n_padded:,}, 总计: {n_total:,}", flush=True)

        export_meta = {
            **rotation_train_meta,
            "export_ema_alpha": EXPORT_EMA_ALPHA,
            "export_token": f"e{str(EXPORT_EMA_ALPHA).replace('.', 'p')}_t{EXPORT_TOP_N}",
        }
        scores_path = export_rotation_scores(
            df_expanded,
            top_n=EXPORT_TOP_N,
            raw_scores=df_scores_raw,
            artifact_metadata=export_meta,
        )
        return scores_path

    run_export()
    return


@app.cell
def _(LABEL, df_all, df_scores_raw, go, make_subplots, np, pl, stats):
    # ==============================================================================
    # Cell 7: Signal Quality Analysis — 模型信号统计检验
    #
    # 基于 Cell 6 输出的原始分数 (df_scores_raw), 独立做 EMA 平滑.
    # 修改 EMA_ALPHA 只需重跑本 Cell, 无需重新训练模型.
    # 评估口径拆分:
    #   - 目标标签诊断: 跟随 LABEL, 看模型是否学到训练目标
    #   - 经济效果评估: 固定使用 fwd_ret_1d, 保证不同训练目标之间可比
    #
    # 包含:
    #   7a. Target IC/ICIR + t检验 + 累积IC曲线
    #   7b. Economic Quintile Long-Short 分层收益 (固定 fwd_ret_1d)
    #   7c. Prediction Turnover (Top-20 日间重叠率)
    # ==============================================================================
    def run_signal_quality():
        EMA_ALPHA = 0.3  # Score 时序平滑 (1.0 = 不平滑, 仅影响分析, 不影响训练)
        target_label_col = LABEL
        eval_label_col = "fwd_ret_1d"
        label_cols = ["date", "code", target_label_col]
        if eval_label_col != target_label_col:
            label_cols.append(eval_label_col)

        # ── 合并原始 score 与目标标签/经济评估标签 ──
        df_signal = (
            df_scores_raw
            .select(["date", "code", "score"])
            .join(
                df_all.select(label_cols),
                on=["date", "code"],
                how="inner",
            )
            .filter(
                pl.col(target_label_col).is_not_null()
                & pl.col(target_label_col).is_not_nan()
                & pl.col(eval_label_col).is_not_null()
                & pl.col(eval_label_col).is_not_nan()
            )
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
        target_rets_np = df_signal[target_label_col].to_numpy().astype(np.float64)
        eval_rets_np = df_signal[eval_label_col].to_numpy().astype(np.float64)
        codes_np = df_signal["code"].to_numpy()

        unique_dates = np.unique(dates_np)
        unique_dates.sort()
        n_days = len(unique_dates)
        date_start = np.searchsorted(dates_np, unique_dates, side="left")
        date_end = np.searchsorted(dates_np, unique_dates, side="right")

        print("📊 Signal Quality Analysis")
        print(f"   目标标签: {target_label_col}")
        print(f"   经济评估标签: {eval_label_col}")
        print(f"   样本: {len(dates_np):,} 条, {n_days} 个交易日\n")

        # ================================================================
        # 7a. Target IC / ICIR / t-test
        # ================================================================
        daily_ic = np.full(n_days, np.nan)
        daily_n_stocks = np.zeros(n_days, dtype=int)

        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            sc = scores_np[s:e]
            rt = target_rets_np[s:e]
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
        print("=" * 65)
        print("  7a. Target Label IC Analysis")
        print("=" * 65)
        print(f"  标签列:        {target_label_col}")
        print(f"  IC Mean:       {ic_mean:+.4f}")
        print(f"  IC Std:        {ic_std:.4f}")
        print(f"  ICIR:          {icir:+.4f}")
        print(f"  t-stat:        {t_stat:+.2f}  {'✅ 显著 (>2)' if abs(t_stat) > 2 else '❌ 不显著 (<2)'}")
        print(f"  IC > 0 占比:   {ic_pos_pct:.1f}%")
        print(f"  有效天数:      {len(valid_ic)} / {n_days}")
        print(f"  日均股票数:    {int(np.mean(daily_n_stocks))}")
        print("-" * 65)

        # ================================================================
        # 7b. Economic Quintile Analysis (分层收益)
        # ================================================================
        N_Q = 5
        quintile_daily = {q: [] for q in range(1, N_Q + 1)}
        ls_daily = []

        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            sc = scores_np[s:e]
            rt = eval_rets_np[s:e]
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
        print("  7b. Economic Quintile Long-Short Analysis (Q1 做多 - Q5 做空)")
        print("=" * 65)
        print(f"  经济评估标签:  {eval_label_col}")
        for q in range(1, N_Q + 1):
            arr = np.array(quintile_daily[q])
            qm = float(np.mean(arr)) * 100
            print(f"  Q{q} 日均收益: {qm:+.3f}%")
        print("  ---")
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
        print("  7d. 分年 Target IC / Economic L-S 统计")
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
                f"累积 Target IC 曲线 ({target_label_col})",
                f"分层日均收益 (按 {eval_label_col} 评估)",
                f"L/S 累积收益曲线 ({eval_label_col})",
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
            name="累积Target IC", line=dict(color="#00d4aa"),
        ), row=1, col=1)

        # 2. Quintile bar chart
        q_means = [float(np.mean(quintile_daily[q])) * 100 for q in range(1, N_Q + 1)]
        colors = ["#00d4aa" if m > 0 else "#ff6b6b" for m in q_means]
        fig.add_trace(go.Bar(
            x=[f"Q{q}" for q in range(1, N_Q + 1)],
            y=q_means,
            marker_color=colors,
            name=f"{eval_label_col} 日均收益%",
        ), row=1, col=2)

        # 3. L/S cumulative return
        cum_ls_arr = np.cumsum(ls_arr)
        fig.add_trace(go.Scatter(
            y=cum_ls_arr.tolist(),
            name=f"{eval_label_col} L/S累积收益", line=dict(color="#ffa500"),
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
            yaxis_title="累积Target IC", yaxis2_title="日均收益(%)",
            yaxis3_title=f"{eval_label_col} 累积L/S收益", yaxis4_title="重叠率(%)",
        )
        fig.show()

        return {
            "target_label": target_label_col,
            "eval_label": eval_label_col,
            "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "t_stat": t_stat, "ic_pos_pct": ic_pos_pct,
            "ls_mean": ls_mean, "ls_sharpe": ls_sharpe, "ls_t": ls_t,
            "ls_hit": ls_hit,
        }

    run_signal_quality()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
