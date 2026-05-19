import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import numpy as np
    import polars as pl

    from manifests import (
        B1_BASE_TEXTBOOK_CASES,
        B1_TEXTBOOK_CASES,
        B1_TEXTBOOK_CASES_VERSION,
        EXPANDED_TEXTBOOK_CASES,
        EXPANDED_TEXTBOOK_CASES_VERSION,
    )
    from utils import (
        b1_feature_set_requires_rotation_kbar,
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
    SEED_COL = "seed_mid"
    USE_BULL_ONLY = False

    TARGET_MODE = "two_stage_textbook"
    STRUCTURE_LABEL_COL = "is_textbook_b1"
    STRUCTURE_SCORE_COL = "textbook_b1_score"
    STAGE2_LABEL_COL = "fwd_mfe_risk_adj_10d"
    STAGE2_MODEL_MODE = "tail_classifier"
    STAGE2_TAIL_THRESHOLD = 0.15
    LABEL_BY_TARGET_MODE: dict[str, str] = {
        "single_stage_mfe": "fwd_mfe_10d",
        "two_stage_textbook": STAGE2_LABEL_COL,
    }
    LABEL_COL = LABEL_BY_TARGET_MODE[TARGET_MODE]
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
        "textbook_b1_score": 0.65,
        "is_textbook_b1": 0.5,
    }
    POSITIVE_LABEL_THRESHOLD = POSITIVE_LABEL_THRESHOLDS.get(LABEL_COL, 0.05)
    EVAL_HIT_THRESHOLDS_BY_LABEL: dict[str, tuple[float, ...]] = {
        "fwd_mfe_risk_adj_10d": (0.07, 0.10, 0.15, 0.18),
    }
    EVAL_HIT_THRESHOLDS = EVAL_HIT_THRESHOLDS_BY_LABEL.get(
        LABEL_COL,
        (POSITIVE_LABEL_THRESHOLD,),
    )
    FEATURE_SET_NAME = "selected_rotation_hybrid_v1"
    TRAIN_WINDOW = 480
    RETRAIN_FREQ = 20
    EMA_ALPHA = 1.0
    SCORE_THRESHOLD_QUANTILES = (0.50, 0.70, 0.80, 0.90, 0.95)
    TOPK_LIST = (1, 3, 5)
    STAGE2_MIN_LABEL_FOR_TRAIN = (
        0.10
        if TARGET_MODE == "two_stage_textbook" and STAGE2_MODEL_MODE == "regression"
        else None
    )
    ENABLE_SAMPLE_WEIGHT = True
    BASE_TEXTBOOK_SAMPLE_WEIGHT = 3.0
    EXPANDED_TEXTBOOK_SAMPLE_WEIGHT = 2.0
    RECENT_SAMPLE_WEIGHT_START_DATE = "2022-01-01"
    RECENT_SAMPLE_WEIGHT = 1.5
    TAIL_SAMPLE_WEIGHT_QUANTILE = 0.90
    TAIL_SAMPLE_WEIGHT = 2.0

    LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),
        ("2019-12-16", "2020-03-02"),
        ("2020-06-19", "2020-07-15"),
        ("2020-12-24", "2021-01-25"),
        ("2021-04-20", "2021-06-16"),
        ("2021-07-12", "2021-08-17"),
        ("2021-08-25", "2021-09-16"),
        ("2022-04-28", "2022-07-25"),
        ("2022-10-14", "2022-12-19"),
        ("2023-01-06", "2023-05-12"),
        ("2023-08-01", "2023-08-11"),
        ("2023-08-30", "2023-09-20"),
        ("2023-10-26", "2023-12-20"),
        ("2024-01-02", "2024-01-17"),
        ("2024-01-25", "2024-01-30"),
        ("2024-02-07", "2024-03-25"),
        ("2024-04-18", "2024-05-15"),
        ("2024-07-12", "2024-07-23"),
        ("2024-08-01", "2024-08-12"),
        ("2024-09-02", "2024-11-14"),
        ("2025-01-15", "2025-01-27"),
        ("2025-02-07", "2025-02-28"),
        ("2025-04-09", "2025-04-18"),
        ("2025-05-07", "2025-09-04"),
        ("2026-01-06", "2026-02-02"),
        # ("2026-04-08", "2026-04-30"), # 暂时注释掉，不影响基线回测的结论
    ]

    FEATURE_COLS = list(resolve_b1_feature_set(FEATURE_SET_NAME))
    FEATURE_SET_DESC = describe_b1_feature_set(FEATURE_SET_NAME)
    INCLUDE_ROTATION_KBAR_FEATURES = b1_feature_set_requires_rotation_kbar(FEATURE_SET_NAME)
    return (
        B1_BASE_TEXTBOOK_CASES,
        B1_TEXTBOOK_CASES,
        B1_TEXTBOOK_CASES_VERSION,
        BASE_TEXTBOOK_SAMPLE_WEIGHT,
        DB_PATH,
        EMA_ALPHA,
        ENABLE_SAMPLE_WEIGHT,
        EVAL_HIT_THRESHOLDS,
        END_DATE,
        EXPANDED_TEXTBOOK_CASES,
        EXPANDED_TEXTBOOK_SAMPLE_WEIGHT,
        EXPANDED_TEXTBOOK_CASES_VERSION,
        EXPORT_START_DATE,
        FEATURE_COLS,
        FEATURE_SET_DESC,
        FEATURE_SET_NAME,
        INCLUDE_ROTATION_KBAR_FEATURES,
        LABEL_COL,
        LOOSE_PERIODS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        POSITIVE_LABEL_THRESHOLD,
        RECENT_SAMPLE_WEIGHT,
        RECENT_SAMPLE_WEIGHT_START_DATE,
        RETRAIN_FREQ,
        SCORE_THRESHOLD_QUANTILES,
        SEED_COL,
        SEED_J_MAX,
        STAGE2_LABEL_COL,
        STAGE2_MODEL_MODE,
        STAGE2_TAIL_THRESHOLD,
        STAGE2_MIN_LABEL_FOR_TRAIN,
        START_DATE,
        STRUCTURE_LABEL_COL,
        STRUCTURE_SCORE_COL,
        ST_SNAPSHOT_DATE,
        TAIL_SAMPLE_WEIGHT,
        TAIL_SAMPLE_WEIGHT_QUANTILE,
        TARGET_MODE,
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
    BASE_TEXTBOOK_SAMPLE_WEIGHT,
    B1_BASE_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES_VERSION,
    EMA_ALPHA,
    ENABLE_SAMPLE_WEIGHT,
    EVAL_HIT_THRESHOLDS,
    EXPANDED_TEXTBOOK_CASES,
    EXPANDED_TEXTBOOK_SAMPLE_WEIGHT,
    EXPANDED_TEXTBOOK_CASES_VERSION,
    FEATURE_COLS,
    FEATURE_SET_DESC,
    FEATURE_SET_NAME,
    INCLUDE_ROTATION_KBAR_FEATURES,
    LABEL_COL,
    POSITIVE_LABEL_THRESHOLD,
    RECENT_SAMPLE_WEIGHT,
    RECENT_SAMPLE_WEIGHT_START_DATE,
    RETRAIN_FREQ,
    SCORE_THRESHOLD_QUANTILES,
    SEED_COL,
    STAGE2_LABEL_COL,
    STAGE2_MODEL_MODE,
    STAGE2_TAIL_THRESHOLD,
    STAGE2_MIN_LABEL_FOR_TRAIN,
    STRUCTURE_LABEL_COL,
    STRUCTURE_SCORE_COL,
    TAIL_SAMPLE_WEIGHT,
    TAIL_SAMPLE_WEIGHT_QUANTILE,
    TARGET_MODE,
    TOPK_LIST,
    TRAIN_WINDOW,
    USE_BULL_ONLY,
):
    print("=" * 72)
    print("  B1 Seed Train / Export Entry")
    print("=" * 72)
    print(f"  seed_col:          {SEED_COL}")
    print(f"  target_mode:       {TARGET_MODE}")
    print(f"  bull_regime_only:  {USE_BULL_ONLY}")
    print(f"  label_col:         {LABEL_COL}")
    if TARGET_MODE == "two_stage_textbook":
        print(f"  structure_label:   {STRUCTURE_LABEL_COL}")
        print(f"  structure_score:   {STRUCTURE_SCORE_COL}")
        print(f"  stage2_label:      {STAGE2_LABEL_COL}")
    print(f"  positive_thresh:   {POSITIVE_LABEL_THRESHOLD:.2%}")
    print(f"  feature_set:       {FEATURE_SET_NAME}")
    print(f"  feature_count:     {len(FEATURE_COLS)}")
    print(f"  ext_rotation_kbar: {INCLUDE_ROTATION_KBAR_FEATURES}")
    print(f"  feature_desc:      {FEATURE_SET_DESC}")
    print(f"  train_window:      {TRAIN_WINDOW}")
    print(f"  retrain_freq:      {RETRAIN_FREQ}")
    print(f"  ema_alpha:         {EMA_ALPHA}")
    print(f"  score_threshold_q: {SCORE_THRESHOLD_QUANTILES}")
    print(f"  topk_list:         {TOPK_LIST}")
    print(f"  eval_hit_thresh:   {EVAL_HIT_THRESHOLDS}")
    print(f"  stage2_model:      {STAGE2_MODEL_MODE}")
    print(f"  stage2_tail_th:    {STAGE2_TAIL_THRESHOLD:.2%}")
    print(f"  stage2_train_min:  {STAGE2_MIN_LABEL_FOR_TRAIN}")
    print(f"  sample_weight:     {ENABLE_SAMPLE_WEIGHT}")
    if ENABLE_SAMPLE_WEIGHT:
        _tail_weight_desc = (
            f">={STAGE2_TAIL_THRESHOLD:.2%}"
            if STAGE2_MODEL_MODE == "tail_classifier"
            else f"q{int(TAIL_SAMPLE_WEIGHT_QUANTILE * 100)}"
        )
        print(
            "  weighting_cfg:     "
            f"base={BASE_TEXTBOOK_SAMPLE_WEIGHT:.2f}x, "
            f"expanded={EXPANDED_TEXTBOOK_SAMPLE_WEIGHT:.2f}x, "
            f"recent={RECENT_SAMPLE_WEIGHT:.2f}x@{RECENT_SAMPLE_WEIGHT_START_DATE}, "
            f"tail={TAIL_SAMPLE_WEIGHT:.2f}x@{_tail_weight_desc}"
        )
    print(f"  textbook_base:     {len(B1_BASE_TEXTBOOK_CASES)}")
    print(f"  textbook_expanded: {len(EXPANDED_TEXTBOOK_CASES)}")
    print(f"  textbook_total:    {len(B1_TEXTBOOK_CASES)}")
    print(f"  textbook_version:  {B1_TEXTBOOK_CASES_VERSION}")
    print(f"  expanded_version:  {EXPANDED_TEXTBOOK_CASES_VERSION}")
    print("")
    print("  quick_start:")
    print('    1. 结构 Lab: `notebooks/b1_condition_mining.py` -> LABEL_COL = "textbook_b1_score"')
    print('    2. 收益 Lab: `notebooks/b1_condition_mining.py` -> LABEL_COL = "fwd_mfe_risk_adj_10d"')
    print('    3. 案例扩容: `notebooks/b1_case_expansion_mining.py` -> 按需写入 `manifests/b1_expanded_textbook_cases.py`')
    print('    4. 双阶段训练: 当前 notebook -> TARGET_MODE = "two_stage_textbook"')
    print("    5. 看 Step 4 的 structure 摘要和 top-k，再导出给 Rust")
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
    B1_BASE_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES_VERSION,
    EXPANDED_TEXTBOOK_CASES,
    FEATURE_COLS,
    FEATURE_SET_NAME,
    INCLUDE_ROTATION_KBAR_FEATURES,
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
        include_rotation_kbar_features=INCLUDE_ROTATION_KBAR_FEATURES,
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
                {"item": "textbook_base_cases", "value": str(len(B1_BASE_TEXTBOOK_CASES))},
                {"item": "textbook_expanded_cases", "value": str(len(EXPANDED_TEXTBOOK_CASES))},
                {"item": "textbook_total_cases", "value": str(len(B1_TEXTBOOK_CASES))},
                {"item": "textbook_cases_version", "value": B1_TEXTBOOK_CASES_VERSION},
            ]
        )
    )
    return df_all, df_seed, valid_feature_cols


@app.cell
def _(
    BASE_TEXTBOOK_SAMPLE_WEIGHT,
    B1_BASE_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES_VERSION,
    ENABLE_SAMPLE_WEIGHT,
    EVAL_HIT_THRESHOLDS,
    EXPANDED_TEXTBOOK_CASES,
    EXPANDED_TEXTBOOK_SAMPLE_WEIGHT,
    EXPANDED_TEXTBOOK_CASES_VERSION,
    FEATURE_SET_NAME,
    LABEL_COL,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    RECENT_SAMPLE_WEIGHT,
    RECENT_SAMPLE_WEIGHT_START_DATE,
    RETRAIN_FREQ,
    SEED_COL,
    STAGE2_LABEL_COL,
    STAGE2_MODEL_MODE,
    STAGE2_TAIL_THRESHOLD,
    STAGE2_MIN_LABEL_FOR_TRAIN,
    STRUCTURE_LABEL_COL,
    TAIL_SAMPLE_WEIGHT,
    TAIL_SAMPLE_WEIGHT_QUANTILE,
    TARGET_MODE,
    TRAIN_WINDOW,
    USE_BULL_ONLY,
    df_seed,
    np,
    pl,
    valid_feature_cols,
):
    df_scores_raw = pl.DataFrame(
        schema={
            "date": pl.Date,
            "code": pl.Utf8,
            "score": pl.Float64,
            "structure_score": pl.Float64,
            "payoff_score": pl.Float64,
        }
    )
    b1_train_meta = None
    print("\n" + "=" * 72)
    print("  Step 3. LightGBM Walk-Forward")
    print("=" * 72)

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
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
            def _build_case_flag_df(case_rows, flag_col):
                if not case_rows:
                    return pl.DataFrame(
                        schema={
                            "code": pl.Utf8,
                            "date": pl.Date,
                            flag_col: pl.Boolean,
                        }
                    )
                return (
                    pl.DataFrame(case_rows)
                    .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
                    .select(["code", "date"])
                    .unique()
                    .with_columns(pl.lit(True).alias(flag_col))
                )

            def _build_sample_weights(
                dates,
                base_flags,
                expanded_flags,
                *,
                label_values=None,
                enable_tail=False,
                tail_label_threshold=None,
            ):
                weights = np.ones(len(dates), dtype=np.float32)
                tail_cut = None
                if not ENABLE_SAMPLE_WEIGHT:
                    return weights, tail_cut
                weights *= np.where(base_flags, BASE_TEXTBOOK_SAMPLE_WEIGHT, 1.0).astype(np.float32)
                weights *= np.where(expanded_flags, EXPANDED_TEXTBOOK_SAMPLE_WEIGHT, 1.0).astype(np.float32)
                recent_start = np.datetime64(RECENT_SAMPLE_WEIGHT_START_DATE)
                weights *= np.where(dates >= recent_start, RECENT_SAMPLE_WEIGHT, 1.0).astype(np.float32)
                if enable_tail and label_values is not None and len(label_values):
                    if tail_label_threshold is None:
                        tail_cut = float(np.quantile(label_values, TAIL_SAMPLE_WEIGHT_QUANTILE))
                    else:
                        tail_cut = float(tail_label_threshold)
                    weights *= np.where(label_values >= tail_cut, TAIL_SAMPLE_WEIGHT, 1.0).astype(np.float32)
                return weights, tail_cut

            base_case_df = _build_case_flag_df(B1_BASE_TEXTBOOK_CASES, "is_base_textbook_case")
            expanded_case_df = _build_case_flag_df(
                EXPANDED_TEXTBOOK_CASES, "is_expanded_textbook_case"
            )
            df_seed_weighted = (
                df_seed.join(base_case_df, on=["code", "date"], how="left")
                .join(expanded_case_df, on=["code", "date"], how="left")
                .with_columns(
                    [
                        pl.col("is_base_textbook_case")
                        .fill_null(False)
                        .alias("is_base_textbook_case"),
                        pl.col("is_expanded_textbook_case")
                        .fill_null(False)
                        .alias("is_expanded_textbook_case"),
                    ]
                )
            )
            df_structure_train = (
                df_seed_weighted.filter(pl.col(STRUCTURE_LABEL_COL).is_not_null())
                .filter(pl.all_horizontal(*[pl.col(col).is_not_null() for col in valid_feature_cols]))
                .sort(["date", "code"])
            )
            df_train = (
                df_seed_weighted.filter(pl.col(LABEL_COL).is_not_null())
                .filter(pl.all_horizontal(*[pl.col(col).is_not_null() for col in valid_feature_cols]))
                .sort(["date", "code"])
            )
            if TARGET_MODE == "two_stage_textbook":
                df_train = df_train.filter(pl.col(STRUCTURE_LABEL_COL))
            if STAGE2_MIN_LABEL_FOR_TRAIN is not None:
                df_train = df_train.filter(pl.col(LABEL_COL) >= STAGE2_MIN_LABEL_FOR_TRAIN)
            df_score = (
                df_seed_weighted
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
                x_structure_all = df_structure_train.select(valid_feature_cols).to_numpy().astype(np.float32)
                y_structure_all = df_structure_train[STRUCTURE_LABEL_COL].cast(pl.Int8).to_numpy().astype(np.int32)
                dates_structure_all = df_structure_train["date"].to_numpy()
                is_base_structure_all = (
                    df_structure_train["is_base_textbook_case"].to_numpy().astype(bool)
                )
                is_expanded_structure_all = (
                    df_structure_train["is_expanded_textbook_case"].to_numpy().astype(bool)
                )
                x_all = df_train.select(valid_feature_cols).to_numpy().astype(np.float32)
                y_all_cont = df_train[LABEL_COL].to_numpy().astype(np.float64)
                if STAGE2_MODEL_MODE == "tail_classifier":
                    y_all = (y_all_cont >= STAGE2_TAIL_THRESHOLD).astype(np.int32)
                else:
                    y_all = y_all_cont
                dates_all = df_train["date"].to_numpy()
                is_base_all = df_train["is_base_textbook_case"].to_numpy().astype(bool)
                is_expanded_all = df_train["is_expanded_textbook_case"].to_numpy().astype(bool)
                x_score_all = df_score.select(valid_feature_cols).to_numpy().astype(np.float32)
                dates_score_all = df_score["date"].to_numpy()
                codes_score_all = df_score["code"].to_numpy()
                np.nan_to_num(x_structure_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
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
                lgb_cls_params = {
                    **lgb_params,
                    "objective": "binary",
                    "class_weight": "balanced",
                }

                score_dates = []
                score_codes = []
                score_values = []
                structure_values = []
                payoff_values = []
                model = None
                structure_model = None
                last_train_idx = TRAIN_WINDOW - RETRAIN_FREQ

                print("🤖 LightGBM Walk-Forward 打分", flush=True)
                print(
                    f"   训练窗口: {TRAIN_WINDOW}天, 重训: 每{RETRAIN_FREQ}天, 模式: {TARGET_MODE}",
                    flush=True,
                )
                print(f"   排序标签: {LABEL_COL}", flush=True)
                if TARGET_MODE == "two_stage_textbook":
                    print(
                        f"   结构标签: {STRUCTURE_LABEL_COL}, 排序收益标签: {STAGE2_LABEL_COL}",
                        flush=True,
                    )
                print(
                    f"   特征集: {FEATURE_SET_NAME} ({len(valid_feature_cols)} 个)",
                    flush=True,
                )
                print(
                    f"   textbook案例: base={len(B1_BASE_TEXTBOOK_CASES)}, "
                    f"expanded={len(EXPANDED_TEXTBOOK_CASES)}, "
                    f"total={len(B1_TEXTBOOK_CASES)}",
                    flush=True,
                )
                print(
                    f"   textbook版本: {B1_TEXTBOOK_CASES_VERSION} "
                    f"(expanded={EXPANDED_TEXTBOOK_CASES_VERSION})",
                    flush=True,
                )
                if STAGE2_MIN_LABEL_FOR_TRAIN is not None:
                    print(
                        f"   stage2训练门槛: {LABEL_COL} >= {STAGE2_MIN_LABEL_FOR_TRAIN:.2%}",
                        flush=True,
                    )
                print(
                    f"   有效样本: {df_train.height:,} 行, {len(all_dates)} 个交易日",
                    flush=True,
                )
                for i in range(TRAIN_WINDOW, len(all_dates)):
                    cur_date = all_dates[i]

                    if i - last_train_idx >= RETRAIN_FREQ or model is None or (
                        TARGET_MODE == "two_stage_textbook" and structure_model is None
                    ):
                        train_start = all_dates[max(0, i - TRAIN_WINDOW)]
                        mask_tr = (dates_all >= np.datetime64(train_start)) & (dates_all < np.datetime64(cur_date))
                        x_tr = x_all[mask_tr]
                        y_tr = y_all[mask_tr]
                        y_tr_cont = y_all_cont[mask_tr]
                        if TARGET_MODE == "two_stage_textbook":
                            mask_structure = (
                                (dates_structure_all >= np.datetime64(train_start))
                                & (dates_structure_all < np.datetime64(cur_date))
                            )
                            x_structure_tr = x_structure_all[mask_structure]
                            y_structure_tr = y_structure_all[mask_structure]
                            structure_sample_weight, _ = _build_sample_weights(
                                dates_structure_all[mask_structure],
                                is_base_structure_all[mask_structure],
                                is_expanded_structure_all[mask_structure],
                            )
                            if len(y_structure_tr) >= 500 and len(np.unique(y_structure_tr)) >= 2:
                                structure_model = LGBMClassifier(**lgb_cls_params)
                                structure_model.fit(
                                    x_structure_tr,
                                    y_structure_tr,
                                    sample_weight=structure_sample_weight,
                                )
                            else:
                                structure_model = None

                        if len(y_tr) < 500:
                            continue
                        stage2_sample_weight, stage2_tail_cut = _build_sample_weights(
                            dates_all[mask_tr],
                            is_base_all[mask_tr],
                            is_expanded_all[mask_tr],
                            label_values=y_tr_cont,
                            enable_tail=True,
                            tail_label_threshold=(
                                STAGE2_TAIL_THRESHOLD
                                if STAGE2_MODEL_MODE == "tail_classifier"
                                else None
                            ),
                        )
                        if STAGE2_MODEL_MODE == "tail_classifier":
                            if len(np.unique(y_tr)) < 2:
                                continue
                            model = LGBMClassifier(**lgb_cls_params)
                            model.fit(x_tr, y_tr, sample_weight=stage2_sample_weight)
                        else:
                            model = LGBMRegressor(**lgb_params)
                            model.fit(x_tr, y_tr, sample_weight=stage2_sample_weight)
                        last_train_idx = i

                        pct = (i - TRAIN_WINDOW) / (len(all_dates) - TRAIN_WINDOW) * 100
                        if TARGET_MODE == "two_stage_textbook":
                            structure_rows = int(mask_structure.sum())
                            structure_pos = int(y_structure_tr.sum()) if len(y_structure_tr) else 0
                            stage2_pos = int(y_tr.sum()) if STAGE2_MODEL_MODE == "tail_classifier" else 0
                            structure_weight_mean = (
                                float(structure_sample_weight.mean())
                                if len(y_structure_tr)
                                else 0.0
                            )
                            stage2_weight_mean = float(stage2_sample_weight.mean())
                            tail_cut_text = (
                                f", stage2_tail_cut: {stage2_tail_cut:.4f}"
                                if stage2_tail_cut is not None
                                else ""
                            )
                            print(
                                f"   [{cur_date}] 双阶段重训 ({pct:.0f}%), "
                                f"stage1样本: {structure_rows:,} / 正类: {structure_pos:,}, "
                                f"stage1_w_mean: {structure_weight_mean:.2f}, "
                                f"stage2样本: {len(y_tr):,}"
                                f"{f' / 正类: {stage2_pos:,}' if STAGE2_MODEL_MODE == 'tail_classifier' else ''}, "
                                f"stage2_w_mean: {stage2_weight_mean:.2f}"
                                f"{tail_cut_text}",
                                flush=True,
                            )
                        else:
                            stage2_weight_mean = float(stage2_sample_weight.mean())
                            tail_cut_text = (
                                f", tail_cut: {stage2_tail_cut:.4f}"
                                if stage2_tail_cut is not None
                                else ""
                            )
                            print(
                                f"   [{cur_date}] 重训 ({pct:.0f}%), 样本: {len(y_tr):,}, "
                                f"weight_mean: {stage2_weight_mean:.2f}{tail_cut_text}",
                                flush=True,
                            )

                    mask_te = dates_all == np.datetime64(cur_date)
                    mask_score = dates_score_all == np.datetime64(cur_date)
                    if not mask_score.any() or model is None:
                        continue

                    if STAGE2_MODEL_MODE == "tail_classifier":
                        payoff_preds = model.predict_proba(x_score_all[mask_score])[:, 1]
                    else:
                        payoff_preds = model.predict(x_score_all[mask_score])
                    if TARGET_MODE == "two_stage_textbook":
                        if structure_model is None:
                            continue
                        structure_preds = structure_model.predict_proba(x_score_all[mask_score])[:, 1]
                        final_preds = structure_preds * payoff_preds
                    else:
                        structure_preds = np.ones_like(payoff_preds)
                        final_preds = payoff_preds
                    score_dates.extend([cur_date] * int(mask_score.sum()))
                    score_codes.extend(codes_score_all[mask_score].tolist())
                    score_values.extend(final_preds.tolist())
                    structure_values.extend(structure_preds.tolist())
                    payoff_values.extend(payoff_preds.tolist())

                extra_dates = sorted(set(score_universe_dates) - set(all_dates))
                if model is not None and extra_dates:
                    print(f"   📌 补充打分 {len(extra_dates)} 个无标签日期 ...", flush=True)
                    for cur_date in extra_dates:
                        mask_te = dates_score_all == np.datetime64(cur_date)
                        if not mask_te.any():
                            continue
                        if STAGE2_MODEL_MODE == "tail_classifier":
                            payoff_preds = model.predict_proba(x_score_all[mask_te])[:, 1]
                        else:
                            payoff_preds = model.predict(x_score_all[mask_te])
                        if TARGET_MODE == "two_stage_textbook":
                            if structure_model is None:
                                continue
                            structure_preds = structure_model.predict_proba(x_score_all[mask_te])[:, 1]
                            final_preds = structure_preds * payoff_preds
                        else:
                            structure_preds = np.ones_like(payoff_preds)
                            final_preds = payoff_preds
                        score_dates.extend([cur_date] * int(mask_te.sum()))
                        score_codes.extend(codes_score_all[mask_te].tolist())
                        score_values.extend(final_preds.tolist())
                        structure_values.extend(structure_preds.tolist())
                        payoff_values.extend(payoff_preds.tolist())

                if score_values:
                    df_scores_raw = pl.DataFrame(
                        {
                            "date": score_dates,
                            "code": score_codes,
                            "score": score_values,
                            "structure_score": structure_values,
                            "payoff_score": payoff_values,
                        }
                    )
                    train_timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
                    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feature_hash = build_feature_hash(valid_feature_cols)
                    run_label = (
                        LABEL_COL
                        if TARGET_MODE == "single_stage_mfe"
                        else (
                            f"textbook_tail{int(STAGE2_TAIL_THRESHOLD * 100)}_{STAGE2_LABEL_COL}"
                            if STAGE2_MODEL_MODE == "tail_classifier"
                            else f"textbook_{STAGE2_LABEL_COL}"
                        )
                    )
                    b1_train_meta = {
                        "strategy": "b1",
                        "label": run_label,
                        "model_name": "lightgbm",
                        "feature_set_name": FEATURE_SET_NAME,
                        "feature_mode": FEATURE_SET_NAME,
                        "feature_hash": feature_hash,
                        "features": valid_feature_cols,
                        "feature_count": len(valid_feature_cols),
                        "train_timestamp_token": train_timestamp_token,
                        "train_run_id": build_b1_train_run_id(
                            run_label,
                            SEED_COL,
                            "lightgbm",
                            train_timestamp_token,
                            feature_hash,
                        ),
                        "trained_at": trained_at,
                        "git_commit": get_git_commit(),
                        "notebook": "notebooks/b1_seed_ml_baseline.py",
                        "model_params": (
                            lgb_cls_params
                            if STAGE2_MODEL_MODE == "tail_classifier"
                            else lgb_params
                        ),
                        "target_mode": TARGET_MODE,
                        "structure_label_col": STRUCTURE_LABEL_COL if TARGET_MODE == "two_stage_textbook" else None,
                        "stage2_label_col": STAGE2_LABEL_COL if TARGET_MODE == "two_stage_textbook" else LABEL_COL,
                        "stage2_model_mode": STAGE2_MODEL_MODE,
                        "stage2_tail_threshold": STAGE2_TAIL_THRESHOLD,
                        "eval_hit_thresholds": list(EVAL_HIT_THRESHOLDS),
                        "train_window": TRAIN_WINDOW,
                        "retrain_freq": RETRAIN_FREQ,
                        "seed_col": SEED_COL,
                        "use_bull_only": USE_BULL_ONLY,
                        "stage2_min_label_for_train": STAGE2_MIN_LABEL_FOR_TRAIN,
                        "signal_source": SEED_COL,
                        "sort_field": "score",
                        "sort_ascending": False,
                        "textbook_cases_version": B1_TEXTBOOK_CASES_VERSION,
                        "textbook_base_case_count": len(B1_BASE_TEXTBOOK_CASES),
                        "textbook_expanded_case_count": len(EXPANDED_TEXTBOOK_CASES),
                        "textbook_total_case_count": len(B1_TEXTBOOK_CASES),
                        "textbook_expanded_cases_version": EXPANDED_TEXTBOOK_CASES_VERSION,
                        "sample_weighting": {
                            "enabled": ENABLE_SAMPLE_WEIGHT,
                            "base_textbook_case_weight": BASE_TEXTBOOK_SAMPLE_WEIGHT,
                            "expanded_textbook_case_weight": EXPANDED_TEXTBOOK_SAMPLE_WEIGHT,
                            "recent_start_date": RECENT_SAMPLE_WEIGHT_START_DATE,
                            "recent_sample_weight": RECENT_SAMPLE_WEIGHT,
                            "tail_quantile": (
                                TAIL_SAMPLE_WEIGHT_QUANTILE
                                if STAGE2_MODEL_MODE != "tail_classifier"
                                else None
                            ),
                            "tail_label_threshold": (
                                STAGE2_TAIL_THRESHOLD
                                if STAGE2_MODEL_MODE == "tail_classifier"
                                else None
                            ),
                            "tail_sample_weight": TAIL_SAMPLE_WEIGHT,
                        },
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
    B1_BASE_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES,
    B1_TEXTBOOK_CASES_VERSION,
    EMA_ALPHA,
    EVAL_HIT_THRESHOLDS,
    EXPANDED_TEXTBOOK_CASES,
    LABEL_COL,
    POSITIVE_LABEL_THRESHOLD,
    SCORE_THRESHOLD_QUANTILES,
    STRUCTURE_LABEL_COL,
    TARGET_MODE,
    TOPK_LIST,
    df_scores_raw,
    df_seed,
    np,
    pl,
):
    _label_mean_col = f"{LABEL_COL}_mean"

    def _format_hit_col(threshold):
        return f"hit_{int(round(threshold * 100))}pct"

    _hit_threshold_cols = {
        threshold: _format_hit_col(threshold) for threshold in EVAL_HIT_THRESHOLDS
    }

    eval_summary = pl.DataFrame(
        schema={
            "score_source": pl.Utf8,
            "metric": pl.Utf8,
            "value": pl.Utf8,
        }
    )
    quintile_table = pl.DataFrame(
        schema={
            "score_source": pl.Utf8,
            "bucket": pl.Int64,
            "samples": pl.Int64,
            _label_mean_col: pl.Float64,
        }
    )
    score_quantile_table = pl.DataFrame(
        schema={
            "score_source": pl.Utf8,
            "quantile": pl.Utf8,
            "score_cut": pl.Float64,
        }
    )
    threshold_table = pl.DataFrame(
        schema={
            "score_source": pl.Utf8,
            "threshold_q": pl.Utf8,
            "score_cut": pl.Float64,
            "days_with_signal": pl.Int64,
            "signal_day_ratio": pl.Float64,
            "avg_candidates_signal_day": pl.Float64,
            _label_mean_col: pl.Float64,
            **{
                hit_col_name: pl.Float64
                for hit_col_name in _hit_threshold_cols.values()
            },
        }
    )
    topk_table = pl.DataFrame(
        schema={
            "score_source": pl.Utf8,
            "top_k": pl.Int64,
            "rows": pl.Int64,
            "days": pl.Int64,
            "avg_candidates_per_day": pl.Float64,
            _label_mean_col: pl.Float64,
            **{
                hit_col_name: pl.Float64
                for hit_col_name in _hit_threshold_cols.values()
            },
        }
    )

    print("\n" + "=" * 72)
    print("  Step 4. 纯模型基线评估")
    print("=" * 72)

    if df_scores_raw.is_empty():
        print("  结论: 当前没有模型输出，无法评估。")
    else:
        df_eval_base = (
            df_seed.join(df_scores_raw, on=["date", "code"], how="inner")
            .sort(["code", "date"])
        )

        def _evaluate_score_source(score_col, score_source):
            _summary = pl.DataFrame(schema=eval_summary.schema)
            _quintile = pl.DataFrame(schema=quintile_table.schema)
            _quantile = pl.DataFrame(schema=score_quantile_table.schema)
            _threshold = pl.DataFrame(schema=threshold_table.schema)
            _topk = pl.DataFrame(schema=topk_table.schema)

            df_eval = (
                df_eval_base
                .filter(pl.col(LABEL_COL).is_not_null() & pl.col(score_col).is_not_null())
                .with_columns(pl.col(score_col).alias("_eval_score"))
                .sort(["code", "date"])
            )
            if df_eval.is_empty():
                return _summary, _quintile, _quantile, _threshold, _topk

            if EMA_ALPHA < 1.0:
                df_eval = df_eval.with_columns(
                    pl.col("_eval_score").ewm_mean(alpha=EMA_ALPHA).over("code").alias("_eval_score")
                )

            daily_ic = (
                df_eval.group_by("date")
                .agg(
                    [
                        pl.len().alias("samples"),
                        pl.corr("_eval_score", LABEL_COL, method="spearman").alias("ic"),
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
                        pl.col("_eval_score").rank("ordinal").over("date").alias("_rank"),
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
            _quintile = (
                df_bucket.group_by("bucket")
                .agg(
                    [
                        pl.len().alias("samples"),
                        pl.col(LABEL_COL).mean().round(4).alias(_label_mean_col),
                    ]
                )
                .sort("bucket")
                .with_columns(pl.lit(score_source).alias("score_source"))
                .select(["score_source", "bucket", "samples", _label_mean_col])
            )

            _top_mean = (
                _quintile.filter(pl.col("bucket") == 4).select(_label_mean_col).item()
                if _quintile.filter(pl.col("bucket") == 4).height
                else None
            )
            _bottom_mean = (
                _quintile.filter(pl.col("bucket") == 0).select(_label_mean_col).item()
                if _quintile.filter(pl.col("bucket") == 0).height
                else None
            )
            spread = (_top_mean - _bottom_mean) if _top_mean is not None and _bottom_mean is not None else None
            score_array = df_eval["_eval_score"].to_numpy().astype(np.float64)
            total_eval_days = max(df_eval["date"].n_unique(), 1)

            _quantile = pl.DataFrame(
                [
                    {
                        "score_source": score_source,
                        "quantile": f"p{int(q * 100)}",
                        "score_cut": round(float(np.quantile(score_array, q)), 4),
                    }
                    for q in SCORE_THRESHOLD_QUANTILES
                ]
            )

            threshold_rows = []
            for q in SCORE_THRESHOLD_QUANTILES:
                score_cut = float(np.quantile(score_array, q))
                df_cut = df_eval.filter(pl.col("_eval_score") >= score_cut)
                days_with_signal = df_cut["date"].n_unique() if df_cut.height else 0
                active_days = max(days_with_signal, 1)
                threshold_payload = {
                    "score_source": score_source,
                    "threshold_q": f"p{int(q * 100)}",
                    "score_cut": score_cut,
                    "days_with_signal": days_with_signal,
                    "signal_day_ratio": days_with_signal / total_eval_days,
                    "avg_candidates_signal_day": df_cut.height / active_days,
                    _label_mean_col: float(df_cut[LABEL_COL].mean()) if df_cut.height else 0.0,
                }
                for hit_threshold, hit_col_name in _hit_threshold_cols.items():
                    threshold_payload[hit_col_name] = (
                        float((df_cut[LABEL_COL] >= hit_threshold).mean())
                        if df_cut.height
                        else 0.0
                    )
                threshold_rows.append(threshold_payload)
            _threshold = (
                pl.DataFrame(threshold_rows)
                .with_columns(
                    [
                        pl.col("score_cut").round(4),
                        pl.col("signal_day_ratio").round(4),
                        pl.col("avg_candidates_signal_day").round(2),
                        pl.col(_label_mean_col).round(4),
                        *[
                            pl.col(hit_col_name).round(4)
                            for hit_col_name in _hit_threshold_cols.values()
                        ],
                    ]
                )
            )

            df_topk = df_eval.with_columns(
                pl.col("_eval_score").rank("ordinal", descending=True).over("date").alias("_rank_desc")
            )
            topk_rows = []
            for top_k in TOPK_LIST:
                df_k = df_topk.filter(pl.col("_rank_desc") <= top_k)
                if df_k.is_empty():
                    continue
                day_count = max(df_k["date"].n_unique(), 1)
                topk_payload = {
                    "score_source": score_source,
                    "top_k": top_k,
                    "rows": df_k.height,
                    "days": day_count,
                    "avg_candidates_per_day": df_k.height / day_count,
                    _label_mean_col: float(df_k[LABEL_COL].mean()),
                }
                for hit_threshold, hit_col_name in _hit_threshold_cols.items():
                    topk_payload[hit_col_name] = float(
                        (df_k[LABEL_COL] >= hit_threshold).mean()
                    )
                topk_rows.append(topk_payload)
            if topk_rows:
                _topk = (
                    pl.DataFrame(topk_rows)
                    .with_columns(
                        [
                            pl.col("avg_candidates_per_day").round(2),
                            pl.col(_label_mean_col).round(4),
                            *[
                                pl.col(hit_col_name).round(4)
                                for hit_col_name in _hit_threshold_cols.values()
                            ],
                        ]
                    )
                    .sort("top_k")
                )

            overall_hit_rows = [
                {
                    "score_source": score_source,
                    "metric": hit_col_name,
                    "value": f"{float((df_eval[LABEL_COL] >= hit_threshold).mean()):.2%}",
                }
                for hit_threshold, hit_col_name in _hit_threshold_cols.items()
            ]
            _summary = pl.DataFrame(
                [
                    {
                        "score_source": score_source,
                        "metric": "rows_eval",
                        "value": f"{df_eval.height:,}",
                    },
                    {
                        "score_source": score_source,
                        "metric": "days_eval",
                        "value": str(df_eval["date"].n_unique()),
                    },
                    {
                        "score_source": score_source,
                        "metric": "daily_ic_mean",
                        "value": f"{ic_mean:+.4f}",
                    },
                    {
                        "score_source": score_source,
                        "metric": "daily_icir",
                        "value": f"{icir:+.4f}",
                    },
                    {
                        "score_source": score_source,
                        "metric": "daily_ic_tstat",
                        "value": f"{t_stat:+.4f}",
                    },
                    {
                        "score_source": score_source,
                        "metric": f"q4_minus_q0_{LABEL_COL}",
                        "value": "n/a" if spread is None else f"{spread:+.4f}",
                    },
                    *overall_hit_rows,
                ]
            )
            return _summary, _quintile, _quantile, _threshold, _topk

        score_sources = [("score", "final_score")]
        if "payoff_score" in df_scores_raw.columns:
            score_sources.append(("payoff_score", "payoff_score"))
        if TARGET_MODE == "two_stage_textbook" and "structure_score" in df_scores_raw.columns:
            score_sources.append(("structure_score", "structure_score"))

        summary_frames = []
        quintile_frames = []
        quantile_frames = []
        threshold_frames = []
        topk_frames = []
        for score_col, score_source in score_sources:
            (
                _summary_frame,
                _quintile_frame,
                _quantile_frame,
                _threshold_frame,
                _topk_frame,
            ) = _evaluate_score_source(score_col, score_source)
            if _summary_frame.height:
                summary_frames.append(_summary_frame)
            if _quintile_frame.height:
                quintile_frames.append(_quintile_frame)
            if _quantile_frame.height:
                quantile_frames.append(_quantile_frame)
            if _threshold_frame.height:
                threshold_frames.append(_threshold_frame)
            if _topk_frame.height:
                topk_frames.append(_topk_frame)

        if summary_frames:
            eval_summary = pl.concat(summary_frames, how="vertical")
        if quintile_frames:
            quintile_table = pl.concat(quintile_frames, how="vertical")
        if quantile_frames:
            score_quantile_table = pl.concat(quantile_frames, how="vertical")
        if threshold_frames:
            threshold_table = pl.concat(threshold_frames, how="vertical")
        if topk_frames:
            topk_table = pl.concat(topk_frames, how="vertical")

        print("\n  分数源汇总:")
        print(eval_summary)
        print("\n  分层结果 (score_source):")
        print(quintile_table)
        print("\n  score 分位数 (score_source):")
        print(score_quantile_table)
        print(
            "\n  score threshold 表现 "
            f"(score_source 对比, hit 阈值: {[f'{threshold:.0%}' for threshold in EVAL_HIT_THRESHOLDS]}):"
        )
        print(threshold_table)
        print("\n  top-k 表现 (score_source):")
        print(topk_table)
        if TARGET_MODE == "two_stage_textbook":
            df_structure_eval = (
                df_seed.join(df_scores_raw, on=["date", "code"], how="inner")
                .filter(pl.col("structure_score").is_not_null())
                .sort(["date", "code"])
            )
            if df_structure_eval.height:
                structure_cut = float(np.quantile(df_structure_eval["structure_score"].to_numpy(), 0.90))
                top_structure = df_structure_eval.filter(pl.col("structure_score") >= structure_cut)
                textbook_case_rows = (
                    df_structure_eval.filter(pl.col("is_textbook_case")).height
                    if "is_textbook_case" in df_structure_eval.columns
                    else 0
                )
                textbook_case_top_hit = (
                    float(
                        df_structure_eval
                        .filter(pl.col("is_textbook_case"))
                        .select((pl.col("structure_score") >= structure_cut).cast(pl.Float64).mean())
                        .item()
                    )
                    if textbook_case_rows
                    else 0.0
                )
                structure_summary = pl.DataFrame(
                    [
                        {
                            "metric": "structure_positive_rate",
                            "value": f"{float(df_structure_eval[STRUCTURE_LABEL_COL].mean()):.2%}",
                        },
                        {
                            "metric": "structure_top_decile_hit",
                            "value": f"{float(top_structure[STRUCTURE_LABEL_COL].mean()):.2%}" if top_structure.height else "0.00%",
                        },
                        {
                            "metric": "textbook_case_rows",
                            "value": str(textbook_case_rows),
                        },
                        {
                            "metric": "textbook_manifest_base_cases",
                            "value": str(len(B1_BASE_TEXTBOOK_CASES)),
                        },
                        {
                            "metric": "textbook_manifest_expanded_cases",
                            "value": str(len(EXPANDED_TEXTBOOK_CASES)),
                        },
                        {
                            "metric": "textbook_manifest_total_cases",
                            "value": str(len(B1_TEXTBOOK_CASES)),
                        },
                        {
                            "metric": "textbook_cases_version",
                            "value": B1_TEXTBOOK_CASES_VERSION,
                        },
                        {
                            "metric": "textbook_case_top_decile_hit",
                            "value": f"{textbook_case_top_hit:.2%}",
                        },
                    ]
                )
                print("\n  结构识别摘要:")
                print(structure_summary)
                component_cols = [
                    "textbook_trend_score",
                    "textbook_kbar_score",
                    "textbook_volume_score",
                    "textbook_trigger_score",
                ]
                available_component_cols = [
                    component_col
                    for component_col in component_cols
                    if component_col in df_structure_eval.columns
                ]
                if available_component_cols:
                    component_rows = []
                    for component_col in available_component_cols:
                        component_rows.extend(
                            [
                                {
                                    "slice": "all",
                                    "component": component_col,
                                    "mean_value": float(df_structure_eval[component_col].mean()),
                                },
                                {
                                    "slice": "top_structure_decile",
                                    "component": component_col,
                                    "mean_value": float(top_structure[component_col].mean()) if top_structure.height else 0.0,
                                },
                            ]
                        )
                    print("\n  结构子分画像:")
                    print(
                        pl.DataFrame(component_rows).with_columns(
                            pl.col("mean_value").round(4)
                        )
                    )
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
                    pl.col("structure_score").fill_null(0.0),
                    pl.col("payoff_score").fill_null(0.0),
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
            extra_sort_cols=["score", "structure_score", "payoff_score"],
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
        pl.col("date") == datetime(2026, 4, 15)
    ).sort('score', descending=True).collect()
    return (df_feb,)


@app.cell
def _(df_feb):
    df_feb["date", "code", "score"]
    return


if __name__ == "__main__":
    app.run()
