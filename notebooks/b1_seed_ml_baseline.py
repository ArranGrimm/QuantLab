import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import numpy as np
    import polars as pl

    from utils import (
        build_b1_research_frame,
        calc_b1_factors_wmacd,
        describe_b1_feature_set,
        export_for_rust,
        get_st_blacklist_pl,
        load_daily_data_full,
        resolve_b1_feature_set,
    )

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2019-01-01"
    END_DATE = "2026-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"
    EXPORT_START_DATE = "2021-06-16"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0
    SEED_COL = "seed_strict"
    USE_BULL_ONLY = False

    LABEL_COL = "fwd_mfe_risk_adj_10d"
    POSITIVE_LABEL_THRESHOLDS: dict[str, float] = {
        "fwd_mfe_10d": 0.08,
        "fwd_mae_10d": -0.03,
        "fwd_net_10d": 0.05,
        "fwd_mfe_risk_adj_10d": 0.07,
        "fwd_ret_10d": 0.07,
        "fwd_ret_5d": 0.06,
        "fwd_ret_3d": 0.05,
        "fwd_ret_2d": 0.04,
        "fwd_ret_1d": 0.03,
    }
    POSITIVE_LABEL_THRESHOLD = POSITIVE_LABEL_THRESHOLDS.get(LABEL_COL, 0.05)
    FEATURE_SET_NAME = "selected"
    TRAIN_WINDOW = 480
    RETRAIN_FREQ = 20
    EMA_ALPHA = 1.0
    SCORE_THRESHOLD_QUANTILES = (0.50, 0.70, 0.80, 0.90, 0.95)
    TOPK_LIST = (1, 3, 5)

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
        # ("2026-04-08", "2026-04-30"), # 暂时注释掉，不影响基线回测的结论
    ]

    FEATURE_COLS = list(resolve_b1_feature_set(FEATURE_SET_NAME))
    FEATURE_SET_DESC = describe_b1_feature_set(FEATURE_SET_NAME)
    return (
        DB_PATH,
        EMA_ALPHA,
        END_DATE,
        EXPORT_START_DATE,
        FEATURE_COLS,
        FEATURE_SET_DESC,
        FEATURE_SET_NAME,
        LABEL_COL,
        LOOSE_PERIODS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        POSITIVE_LABEL_THRESHOLD,
        RETRAIN_FREQ,
        SCORE_THRESHOLD_QUANTILES,
        SEED_COL,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        TOPK_LIST,
        TRAIN_WINDOW,
        USE_BULL_ONLY,
        build_b1_research_frame,
        calc_b1_factors_wmacd,
        duckdb,
        export_for_rust,
        get_st_blacklist_pl,
        load_daily_data_full,
        np,
        pl,
    )


@app.cell
def _(
    EMA_ALPHA,
    FEATURE_COLS,
    FEATURE_SET_DESC,
    FEATURE_SET_NAME,
    LABEL_COL,
    POSITIVE_LABEL_THRESHOLD,
    RETRAIN_FREQ,
    SCORE_THRESHOLD_QUANTILES,
    SEED_COL,
    TOPK_LIST,
    TRAIN_WINDOW,
    USE_BULL_ONLY,
):
    print("=" * 72)
    print("  B1 Seed Train / Export Entry")
    print("=" * 72)
    print(f"  seed_col:          {SEED_COL}")
    print(f"  bull_regime_only:  {USE_BULL_ONLY}")
    print(f"  label_col:         {LABEL_COL}")
    print(f"  positive_thresh:   {POSITIVE_LABEL_THRESHOLD:.2%}")
    print(f"  feature_set:       {FEATURE_SET_NAME}")
    print(f"  feature_count:     {len(FEATURE_COLS)}")
    print(f"  feature_desc:      {FEATURE_SET_DESC}")
    print(f"  train_window:      {TRAIN_WINDOW}")
    print(f"  retrain_freq:      {RETRAIN_FREQ}")
    print(f"  ema_alpha:         {EMA_ALPHA}")
    print(f"  score_threshold_q: {SCORE_THRESHOLD_QUANTILES}")
    print(f"  topk_list:         {TOPK_LIST}")
    return


@app.cell
def _(
    DB_PATH,
    END_DATE,
    START_DATE,
    ST_SNAPSHOT_DATE,
    duckdb,
    get_st_blacklist_pl,
    load_daily_data_full,
    pl,
):
    _conn = duckdb.connect(DB_PATH, read_only=True)
    st_blacklist = get_st_blacklist_pl(ST_SNAPSHOT_DATE)
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(_conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(
            pl.col("date").is_between(
                pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
                pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
            )
        )
    )
    print("\n" + "=" * 72)
    print("  Step 1. 数据范围")
    print("=" * 72)
    print(
        pl.DataFrame(
            [
                {"item": "date_range", "value": f"{START_DATE} ~ {END_DATE}"},
                {"item": "st_snapshot_date", "value": ST_SNAPSHOT_DATE},
                {"item": "st_excluded_count", "value": str(len(st_blacklist))},
            ]
        )
    )
    return (q_full,)


@app.cell
def _(
    FEATURE_COLS,
    FEATURE_SET_NAME,
    LOOSE_PERIODS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    SEED_COL,
    SEED_J_MAX,
    USE_BULL_ONLY,
    build_b1_research_frame,
    pl,
    q_full,
):
    df_all = build_b1_research_frame(
        q_full,
        mv_min=MV_MIN,
        mv_max=MV_MAX,
        min_list_days=MIN_LIST_DAYS,
        seed_j_max=SEED_J_MAX,
        loose_periods=LOOSE_PERIODS,
    )
    valid_feature_cols = [col for col in FEATURE_COLS if col in df_all.columns]

    _seed_filter = pl.col(SEED_COL)
    if USE_BULL_ONLY:
        _seed_filter = _seed_filter & pl.col("is_manual_bull")
    df_seed = df_all.filter(_seed_filter)

    print("\n" + "=" * 72)
    print("  Step 2. Seed 训练样本")
    print("=" * 72)
    print(
        pl.DataFrame(
            [
                {"item": "rows_all", "value": f"{df_all.height:,}"},
                {"item": "rows_seed", "value": f"{df_seed.height:,}"},
                {"item": "date_count_seed", "value": str(df_seed["date"].n_unique()) if df_seed.height else "0"},
                {"item": "code_count_seed", "value": str(df_seed["code"].n_unique()) if df_seed.height else "0"},
                {"item": "feature_set", "value": FEATURE_SET_NAME},
                {"item": "feature_count", "value": str(len(valid_feature_cols))},
            ]
        )
    )
    return df_all, df_seed, valid_feature_cols


@app.cell
def _(
    FEATURE_SET_NAME,
    LABEL_COL,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    RETRAIN_FREQ,
    SEED_COL,
    TRAIN_WINDOW,
    USE_BULL_ONLY,
    df_seed,
    np,
    pl,
    valid_feature_cols,
):
    df_scores_raw = pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8, "score": pl.Float64})
    b1_train_meta = None
    print("\n" + "=" * 72)
    print("  Step 3. LightGBM Walk-Forward")
    print("=" * 72)

    try:
        from lightgbm import LGBMRegressor
        import warnings
        from datetime import datetime
        from utils.signal_export import (
            build_b1_train_run_id,
            build_feature_hash,
            get_git_commit,
        )

        warnings.filterwarnings("ignore", category=UserWarning)
    except Exception as exc:
        print(f"⚠️ LightGBM 不可用，跳过训练: {exc}")
    else:
        if df_seed.is_empty():
            print("⚠️ 当前 seed 样本为空，跳过训练。")
        else:
            df_train = (
                df_seed.filter(pl.col(LABEL_COL).is_not_null())
                .filter(pl.all_horizontal(*[pl.col(col).is_not_null() for col in valid_feature_cols]))
                .sort(["date", "code"])
            )
            df_score = (
                df_seed
                .filter(pl.all_horizontal(*[pl.col(col).is_not_null() for col in valid_feature_cols]))
                .sort(["date", "code"])
            )
            all_dates = df_train["date"].unique().sort().to_list()
            score_universe_dates = df_score["date"].unique().sort().to_list()

            if len(all_dates) <= TRAIN_WINDOW:
                print(
                    f"⚠️ 交易日数量不足以跑 walk-forward: "
                    f"n_dates={len(all_dates)}, train_window={TRAIN_WINDOW}"
                )
            else:
                x_all = df_train.select(valid_feature_cols).to_numpy().astype(np.float32)
                y_all = df_train[LABEL_COL].to_numpy().astype(np.float64)
                dates_all = df_train["date"].to_numpy()
                codes_all = df_train["code"].to_numpy()
                x_score_all = df_score.select(valid_feature_cols).to_numpy().astype(np.float32)
                dates_score_all = df_score["date"].to_numpy()
                codes_score_all = df_score["code"].to_numpy()
                np.nan_to_num(x_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(x_score_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                lgb_params = {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "min_child_samples": 80,
                    "verbose": -1,
                    "n_jobs": -1,
                    "random_state": 42,
                }

                score_dates = []
                score_codes = []
                score_values = []
                model = None
                last_train_idx = TRAIN_WINDOW - RETRAIN_FREQ

                print("🤖 LightGBM Walk-Forward 打分", flush=True)
                print(
                    f"   训练窗口: {TRAIN_WINDOW}天, 重训: 每{RETRAIN_FREQ}天, 标签: {LABEL_COL}",
                    flush=True,
                )
                print(
                    f"   特征集: {FEATURE_SET_NAME} ({len(valid_feature_cols)} 个)",
                    flush=True,
                )
                print(
                    f"   有效样本: {df_train.height:,} 行, {len(all_dates)} 个交易日",
                    flush=True,
                )
                for i in range(TRAIN_WINDOW, len(all_dates)):
                    cur_date = all_dates[i]

                    if i - last_train_idx >= RETRAIN_FREQ or model is None:
                        train_start = all_dates[max(0, i - TRAIN_WINDOW)]
                        mask_tr = (dates_all >= np.datetime64(train_start)) & (dates_all < np.datetime64(cur_date))
                        x_tr = x_all[mask_tr]
                        y_tr = y_all[mask_tr]
                        if len(y_tr) < 500:
                            continue
                        model = LGBMRegressor(**lgb_params)
                        model.fit(x_tr, y_tr)
                        last_train_idx = i

                        pct = (i - TRAIN_WINDOW) / (len(all_dates) - TRAIN_WINDOW) * 100
                        print(
                            f"   [{cur_date}] 重训 ({pct:.0f}%), 样本: {len(y_tr):,}",
                            flush=True,
                        )

                    mask_te = dates_all == np.datetime64(cur_date)
                    if not mask_te.any() or model is None:
                        continue

                    preds = model.predict(x_all[mask_te])
                    score_dates.extend([cur_date] * int(mask_te.sum()))
                    score_codes.extend(codes_all[mask_te].tolist())
                    score_values.extend(preds.tolist())

                extra_dates = sorted(set(score_universe_dates) - set(all_dates))
                if model is not None and extra_dates:
                    print(f"   📌 补充打分 {len(extra_dates)} 个无标签日期 ...", flush=True)
                    for cur_date in extra_dates:
                        mask_te = dates_score_all == np.datetime64(cur_date)
                        if not mask_te.any():
                            continue
                        preds = model.predict(x_score_all[mask_te])
                        score_dates.extend([cur_date] * int(mask_te.sum()))
                        score_codes.extend(codes_score_all[mask_te].tolist())
                        score_values.extend(preds.tolist())

                if score_values:
                    df_scores_raw = pl.DataFrame({"date": score_dates, "code": score_codes, "score": score_values})
                    train_timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
                    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feature_hash = build_feature_hash(valid_feature_cols)
                    b1_train_meta = {
                        "strategy": "b1",
                        "label": LABEL_COL,
                        "model_name": "lightgbm",
                        "feature_set_name": FEATURE_SET_NAME,
                        "feature_mode": FEATURE_SET_NAME,
                        "feature_hash": feature_hash,
                        "features": valid_feature_cols,
                        "feature_count": len(valid_feature_cols),
                        "train_timestamp_token": train_timestamp_token,
                        "train_run_id": build_b1_train_run_id(
                            LABEL_COL,
                            SEED_COL,
                            "lightgbm",
                            train_timestamp_token,
                            feature_hash,
                        ),
                        "trained_at": trained_at,
                        "git_commit": get_git_commit(),
                        "notebook": "notebooks/b1_seed_ml_baseline.py",
                        "model_params": lgb_params,
                        "train_window": TRAIN_WINDOW,
                        "retrain_freq": RETRAIN_FREQ,
                        "seed_col": SEED_COL,
                        "use_bull_only": USE_BULL_ONLY,
                        "signal_source": SEED_COL,
                        "sort_field": "score",
                        "sort_ascending": False,
                        "universe": {
                            "mv_min": MV_MIN,
                            "mv_max": MV_MAX,
                            "min_list_days": MIN_LIST_DAYS,
                        },
                    }
                    print(f"\n   ✅ 打分完成: {df_scores_raw.height:,} 条", flush=True)
                    print(
                        f"   分数覆盖日期: {df_scores_raw['date'].min()} ~ {df_scores_raw['date'].max()} "
                        f"(输入最新日期: {score_universe_dates[-1]})",
                        flush=True,
                    )
                else:
                    print("⚠️ 没有生成任何预测分数。")
    return b1_train_meta, datetime, df_scores_raw


@app.cell
def _(
    EMA_ALPHA,
    LABEL_COL,
    POSITIVE_LABEL_THRESHOLD,
    SCORE_THRESHOLD_QUANTILES,
    TOPK_LIST,
    df_scores_raw,
    df_seed,
    np,
    pl,
):
    _label_mean_col = f"{LABEL_COL}_mean"
    _hit_label = f"hit_{POSITIVE_LABEL_THRESHOLD:.0%}"

    eval_summary = pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "value": pl.Utf8,
        }
    )
    quintile_table = pl.DataFrame(
        schema={
            "bucket": pl.Int64,
            "samples": pl.Int64,
            _label_mean_col: pl.Float64,
        }
    )
    score_quantile_table = pl.DataFrame(
        schema={
            "quantile": pl.Utf8,
            "score_cut": pl.Float64,
        }
    )
    threshold_table = pl.DataFrame(
        schema={
            "threshold_q": pl.Utf8,
            "score_cut": pl.Float64,
            "days_with_signal": pl.Int64,
            "signal_day_ratio": pl.Float64,
            "avg_candidates_signal_day": pl.Float64,
            _label_mean_col: pl.Float64,
            _hit_label: pl.Float64,
        }
    )
    topk_table = pl.DataFrame(
        schema={
            "top_k": pl.Int64,
            "rows": pl.Int64,
            "days": pl.Int64,
            "avg_candidates_per_day": pl.Float64,
            _label_mean_col: pl.Float64,
            _hit_label: pl.Float64,
        }
    )

    print("\n" + "=" * 72)
    print("  Step 4. 纯模型基线评估")
    print("=" * 72)

    if df_scores_raw.is_empty():
        print("  结论: 当前没有模型输出，无法评估。")
    else:
        df_eval = (
            df_seed.join(df_scores_raw, on=["date", "code"], how="inner")
            .filter(pl.col(LABEL_COL).is_not_null() & pl.col("score").is_not_null())
            .sort(["code", "date"])
        )

        if EMA_ALPHA < 1.0:
            df_eval = df_eval.with_columns(pl.col("score").ewm_mean(alpha=EMA_ALPHA).over("code").alias("score"))

        daily_ic = (
            df_eval.group_by("date")
            .agg(
                [
                    pl.len().alias("samples"),
                    pl.corr("score", LABEL_COL, method="spearman").alias("ic"),
                ]
            )
            .filter(pl.col("samples") >= 20)
            .filter(pl.col("ic").is_not_null())
            .sort("date")
        )

        ic_mean = float(daily_ic["ic"].mean()) if daily_ic.height else 0.0
        ic_std = float(daily_ic["ic"].std()) if daily_ic.height > 1 else 0.0
        icir = ic_mean / ic_std * np.sqrt(252.0) if ic_std > 0 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(max(daily_ic.height, 1))) if ic_std > 0 else 0.0

        df_bucket = (
            df_eval.with_columns(
                [
                    pl.len().over("date").alias("_n"),
                    pl.col("score").rank("ordinal").over("date").alias("_rank"),
                ]
            )
            .filter(pl.col("_n") >= 5)
            .with_columns(
                (
                    (((pl.col("_rank") - 1) * 5) / pl.col("_n"))
                    .floor()
                    .clip(0, 4)
                    .cast(pl.Int64)
                    .alias("bucket")
                )
            )
        )
        quintile_table = (
            df_bucket.group_by("bucket")
            .agg([pl.len().alias("samples"), pl.col(LABEL_COL).mean().round(4).alias(_label_mean_col)])
            .sort("bucket")
        )

        _top_mean = (
            quintile_table.filter(pl.col("bucket") == 4).select(_label_mean_col).item()
            if quintile_table.filter(pl.col("bucket") == 4).height
            else None
        )
        _bottom_mean = (
            quintile_table.filter(pl.col("bucket") == 0).select(_label_mean_col).item()
            if quintile_table.filter(pl.col("bucket") == 0).height
            else None
        )
        spread = (_top_mean - _bottom_mean) if _top_mean is not None and _bottom_mean is not None else None
        score_array = df_eval["score"].to_numpy().astype(np.float64)
        total_eval_days = max(df_eval["date"].n_unique(), 1)

        score_quantile_table = pl.DataFrame(
            [
                {
                    "quantile": f"p{int(q * 100)}",
                    "score_cut": round(float(np.quantile(score_array, q)), 4),
                }
                for q in SCORE_THRESHOLD_QUANTILES
            ]
        )

        threshold_rows = []
        for q in SCORE_THRESHOLD_QUANTILES:
            score_cut = float(np.quantile(score_array, q))
            df_cut = df_eval.filter(pl.col("score") >= score_cut)
            days_with_signal = df_cut["date"].n_unique() if df_cut.height else 0
            active_days = max(days_with_signal, 1)
            threshold_rows.append(
                {
                    "threshold_q": f"p{int(q * 100)}",
                    "score_cut": score_cut,
                    "days_with_signal": days_with_signal,
                    "signal_day_ratio": days_with_signal / total_eval_days,
                    "avg_candidates_signal_day": df_cut.height / active_days,
                    _label_mean_col: float(df_cut[LABEL_COL].mean()) if df_cut.height else 0.0,
                    _hit_label: float((df_cut[LABEL_COL] >= POSITIVE_LABEL_THRESHOLD).mean()) if df_cut.height else 0.0,
                }
            )
        threshold_table = (
            pl.DataFrame(threshold_rows)
            .with_columns(
                [
                    pl.col("score_cut").round(4),
                    pl.col("signal_day_ratio").round(4),
                    pl.col("avg_candidates_signal_day").round(2),
                    pl.col(_label_mean_col).round(4),
                    pl.col(_hit_label).round(4),
                ]
            )
        )

        df_topk = df_eval.with_columns(
            pl.col("score").rank("ordinal", descending=True).over("date").alias("_rank_desc")
        )
        topk_rows = []
        for top_k in TOPK_LIST:
            df_k = df_topk.filter(pl.col("_rank_desc") <= top_k)
            if df_k.is_empty():
                continue
            day_count = max(df_k["date"].n_unique(), 1)
            topk_rows.append(
                {
                    "top_k": top_k,
                    "rows": df_k.height,
                    "days": day_count,
                    "avg_candidates_per_day": df_k.height / day_count,
                    _label_mean_col: float(df_k[LABEL_COL].mean()),
                    _hit_label: float((df_k[LABEL_COL] >= POSITIVE_LABEL_THRESHOLD).mean()),
                }
            )
        if topk_rows:
            topk_table = (
                pl.DataFrame(topk_rows)
                .with_columns(
                    [
                        pl.col("avg_candidates_per_day").round(2),
                        pl.col(_label_mean_col).round(4),
                        pl.col(_hit_label).round(4),
                    ]
                )
                .sort("top_k")
            )

        eval_summary = pl.DataFrame(
            [
                {"metric": "rows_eval", "value": f"{df_eval.height:,}"},
                {"metric": "days_eval", "value": str(df_eval["date"].n_unique())},
                {"metric": "daily_ic_mean", "value": f"{ic_mean:+.4f}"},
                {"metric": "daily_icir", "value": f"{icir:+.4f}"},
                {"metric": "daily_ic_tstat", "value": f"{t_stat:+.4f}"},
                {"metric": f"q4_minus_q0_{LABEL_COL}", "value": "n/a" if spread is None else f"{spread:+.4f}"},
            ]
        )
        print(eval_summary)
        print("\n  分层结果:")
        print(quintile_table)
        print("\n  score 分位数:")
        print(score_quantile_table)
        print(f"\n  score threshold 表现 (hit 阈值: {LABEL_COL} >= {POSITIVE_LABEL_THRESHOLD:.2%}):")
        print(threshold_table)
        print("\n  top-k 表现:")
        print(topk_table)
    return


@app.cell
def _(
    EXPORT_START_DATE,
    FEATURE_SET_NAME,
    LOOSE_PERIODS,
    MV_MIN,
    SEED_COL,
    USE_BULL_ONLY,
    b1_train_meta,
    calc_b1_factors_wmacd,
    df_all,
    df_scores_raw,
    export_for_rust,
    pl,
    q_full,
):
    print("\n" + "=" * 72)
    print("  Step 5. 导出到 Rust")
    print("=" * 72)

    if df_scores_raw.is_empty():
        print("  结论: 当前没有分数可导出。")
    else:
        output_path = f"data/signals/b1_{SEED_COL}_{FEATURE_SET_NAME}_seed_signal_score.parquet"
        # 导出底座必须保留完整后续行情，避免已持仓股票因研究过滤而在回测里丢失价格。
        df_export_base = calc_b1_factors_wmacd(q_full, {"MV_THRESHOLD": MV_MIN})
        df_signal_flags = df_all.select(["date", "code", SEED_COL]).lazy()
        df_export = (
            df_export_base
            .join(df_signal_flags, on=["date", "code"], how="left")
            .join(df_scores_raw.lazy(), on=["date", "code"], how="left")
            .with_columns(
                [
                    pl.col(SEED_COL).fill_null(False).alias("b1_signal"),
                    pl.col("score").fill_null(-999.0),
                ]
            )
        )
        seed_signal_rows = df_all.filter(pl.col(SEED_COL)).height
        export_meta = {
            **(b1_train_meta or {}),
            "feature_set_name": FEATURE_SET_NAME,
            "seed_col": SEED_COL,
            "use_bull_only": USE_BULL_ONLY,
            "signal_source": SEED_COL,
            "sort_field": "score",
            "sort_ascending": False,
            "export_ema_alpha": 1.0,
        }
        print(f"  候选定义: b1_signal <- {SEED_COL}")
        print(f"  seed signal rows: {seed_signal_rows:,}")
        print("  排序字段: score (降序)")
        export_file = export_for_rust(
            df_export,
            output_path=output_path,
            loose_periods=LOOSE_PERIODS,
            start_date=EXPORT_START_DATE,
            extra_sort_cols=["score"],
            raw_scores=df_scores_raw,
            artifact_metadata=export_meta,
            write_latest_alias=True,
        )
        print(f"  导出完成: {export_file}")
        print("  Rust 回测命令:")
        print("  backtest-engine\\run_b1.bat")
        print(f'  cargo run -p bt-b1 --release -- --data ../../{export_file} --config crates/b1/config_ml.toml')
    return (df_export,)


@app.cell
def _(datetime, df_export, pl):
    df_feb = df_export.filter(
        pl.col("date") >= datetime(2026, 4, 10)
    ).sort('score', descending=True).collect()
    return (df_feb,)


@app.cell
def _(df_feb):
    df_feb["date", "code", "score"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
