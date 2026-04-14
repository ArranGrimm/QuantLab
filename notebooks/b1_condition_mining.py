import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import numpy as np
    import polars as pl

    from utils import (
        B1_FEATURE_GROUP_LABELS,
        B1_FEATURE_GROUPS,
        build_b1_research_frame,
        build_ic_summary_frame,
        calc_factor_corr,
        calc_factor_ic,
        compute_factor_decay,
        describe_b1_feature_set,
        find_redundant_factors,
        get_st_blacklist_pl,
        load_daily_data_full,
        resolve_b1_feature_set,
        summarize_factor_groups,
    )

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2019-01-01"
    END_DATE = "2026-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0

    ACTIVE_SEED_COL = "seed_strict"
    USE_BULL_ONLY = True
    LABEL_COL = "textbook_b1_score"
    POSITIVE_LABEL_THRESHOLDS: dict[str, float] = {
        "fwd_mfe_10d": 0.08,
        "fwd_mae_10d": -0.03,
        "fwd_net_10d": 0.05,
        "fwd_mfe_risk_adj_10d": 0.07,
        "fwd_ret_10d": 0.07,
        "fwd_ret_5d": 0.05,
        "fwd_ret_3d": 0.04,
        "fwd_ret_2d": 0.003,
        "fwd_ret_1d": 0.002,
        "textbook_b1_score": 0.65,
        "is_textbook_b1": 0.5,
        "textbook_trend_score": 0.65,
        "textbook_kbar_score": 0.65,
        "textbook_volume_score": 0.65,
        "textbook_trigger_score": 0.65,
    }
    POSITIVE_LABEL_THRESHOLD = POSITIVE_LABEL_THRESHOLDS.get(LABEL_COL, 0.05)
    N_BINS = 5
    MIN_DAILY_SAMPLES = 30
    REVIEW_FEATURE = "range_pct"
    ANALYSIS_FEATURE_SET_NAME = "rotation_core12_kbar"
    TRAIN_FEATURE_SET_NAME = "selected"
    DECAY_HORIZONS = ("fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d")
    CORR_SAMPLE_N = 250_000
    CORR_THRESHOLD = 0.85
    RUN_CORR_DIAGNOSTICS = True

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
    ]

    analysis_feature_cols = list(resolve_b1_feature_set(ANALYSIS_FEATURE_SET_NAME))
    train_feature_cols = list(resolve_b1_feature_set(TRAIN_FEATURE_SET_NAME))
    analysis_feature_desc = describe_b1_feature_set(ANALYSIS_FEATURE_SET_NAME)
    train_feature_desc = describe_b1_feature_set(TRAIN_FEATURE_SET_NAME)
    return (
        ACTIVE_SEED_COL,
        ANALYSIS_FEATURE_SET_NAME,
        B1_FEATURE_GROUPS,
        B1_FEATURE_GROUP_LABELS,
        CORR_SAMPLE_N,
        CORR_THRESHOLD,
        DB_PATH,
        DECAY_HORIZONS,
        END_DATE,
        LABEL_COL,
        LOOSE_PERIODS,
        MIN_DAILY_SAMPLES,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        N_BINS,
        POSITIVE_LABEL_THRESHOLD,
        REVIEW_FEATURE,
        RUN_CORR_DIAGNOSTICS,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        TRAIN_FEATURE_SET_NAME,
        USE_BULL_ONLY,
        analysis_feature_cols,
        analysis_feature_desc,
        build_b1_research_frame,
        build_ic_summary_frame,
        calc_factor_corr,
        calc_factor_ic,
        compute_factor_decay,
        duckdb,
        find_redundant_factors,
        get_st_blacklist_pl,
        load_daily_data_full,
        np,
        pl,
        summarize_factor_groups,
        train_feature_cols,
        train_feature_desc,
    )


@app.cell
def _(
    ACTIVE_SEED_COL,
    ANALYSIS_FEATURE_SET_NAME,
    LABEL_COL,
    N_BINS,
    TRAIN_FEATURE_SET_NAME,
    USE_BULL_ONLY,
    analysis_feature_desc,
    train_feature_desc,
):
    print("=" * 72)
    print("  B1 Factor Lab")
    print("=" * 72)
    print("  这本 notebook 只负责统计学特征研究，不再承担规则收敛主线。")
    print("  当前默认流程: Seed 样本概览 -> IC -> 分组归纳 -> 分箱 -> 衰减 -> 相关性诊断")
    print("  quick_start:")
    print('    1. 看结构总分: LABEL_COL = "textbook_b1_score"')
    print('    2. 看结构子分: LABEL_COL = "textbook_trend_score" / "textbook_kbar_score" / "textbook_volume_score" / "textbook_trigger_score"')
    print('    3. 看结构二分类: LABEL_COL = "is_textbook_b1"')
    print('    4. 看结构合格后的收益排序: LABEL_COL = "fwd_mfe_risk_adj_10d"')
    print("")
    print(f"  active_seed:         {ACTIVE_SEED_COL}")
    print(f"  bull_regime_only:    {USE_BULL_ONLY}")
    print(f"  label_col:           {LABEL_COL}")
    print(f"  n_bins:              {N_BINS}")
    print(f"  analysis_feature_set:{ANALYSIS_FEATURE_SET_NAME}")
    print(f"  train_feature_set:   {TRAIN_FEATURE_SET_NAME}")
    print("")
    print(f"  analysis_desc: {analysis_feature_desc}")
    print(f"  train_desc:    {train_feature_desc}")
    print("=" * 72)
    return


@app.cell
def _(np, pl):
    def ordered_unique(values):
        return list(dict.fromkeys(str(value) for value in values if str(value).strip()))

    def build_seed_summary(df, positive_threshold):
        seed_cols = ("seed_loose", "seed_mid", "seed_strict")
        scope_defs = (
            ("all", pl.lit(True)),
            ("bull_only", pl.col("is_manual_bull")),
        )
        rows = []
        for scope_name, scope_expr in scope_defs:
            df_scope = df.filter(scope_expr)
            for seed_col in seed_cols:
                df_seed = df_scope.filter(pl.col(seed_col))
                if df_seed.is_empty():
                    continue
                date_count = max(df_seed["date"].n_unique(), 1)
                rows.append(
                    {
                        "scope": scope_name,
                        "seed": seed_col,
                        "rows": df_seed.height,
                        "dates": date_count,
                        "avg_candidates_per_day": df_seed.height / date_count,
                        "mfe10_mean": float(df_seed["fwd_mfe_10d"].mean()),
                        "mfe_hit_rate": float((df_seed["fwd_mfe_10d"] >= positive_threshold).mean()),
                        "ret1_mean": float(df_seed["fwd_ret_1d"].mean()),
                        "ret3_mean": float(df_seed["fwd_ret_3d"].mean()),
                        "ret5_mean": float(df_seed["fwd_ret_5d"].mean()),
                    }
                )

        if not rows:
            return pl.DataFrame(
                schema={
                    "scope": pl.String,
                    "seed": pl.String,
                    "rows": pl.Int64,
                    "dates": pl.Int64,
                    "avg_candidates_per_day": pl.Float64,
                    "mfe10_mean": pl.Float64,
                    "mfe_hit_rate": pl.Float64,
                    "ret1_mean": pl.Float64,
                    "ret3_mean": pl.Float64,
                    "ret5_mean": pl.Float64,
                }
            )

        return (
            pl.DataFrame(rows)
            .with_columns(
                [
                    pl.col("avg_candidates_per_day").round(2),
                    pl.col("mfe10_mean").round(4),
                    pl.col("mfe_hit_rate").round(4),
                    pl.col("ret1_mean").round(4),
                    pl.col("ret3_mean").round(4),
                    pl.col("ret5_mean").round(4),
                ]
            )
            .sort(["scope", "seed"])
        )

    def build_bin_scoreboard(df, feature_cols, label_col, positive_threshold, n_bins):
        base_mean = float(df[label_col].mean()) if df.height else 0.0
        base_hit = float((df[label_col] >= positive_threshold).mean()) if df.height else 0.0
        rows = []
        for feature in feature_cols:
            df_feature = (
                df.select([feature, label_col])
                .filter(pl.col(feature).is_not_null() & pl.col(label_col).is_not_null())
            )
            if df_feature.height < 200:
                continue

            x_vals = df_feature[feature].to_numpy().astype(np.float64)
            y_vals = df_feature[label_col].to_numpy().astype(np.float64)
            if not np.isfinite(x_vals).all() or np.nanstd(x_vals) < 1e-12:
                continue

            edges = np.unique(np.quantile(x_vals, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 3:
                continue

            bin_ids = np.digitize(x_vals, edges[1:-1], right=True)
            bin_rows = []
            for bin_idx in range(len(edges) - 1):
                mask = bin_ids == bin_idx
                count = int(mask.sum())
                if count < 30:
                    continue
                bin_rows.append(
                    {
                        "bin_idx": bin_idx + 1,
                        "count": count,
                        "mean_label": float(np.mean(y_vals[mask])),
                        "hit_rate": float(np.mean(y_vals[mask] >= positive_threshold)),
                        "left_edge": float(edges[bin_idx]),
                        "right_edge": float(edges[bin_idx + 1]),
                    }
                )

            if len(bin_rows) < 2:
                continue

            best_bin = max(bin_rows, key=lambda item: item["mean_label"])
            worst_bin = min(bin_rows, key=lambda item: item["mean_label"])
            mean_series = np.asarray([item["mean_label"] for item in bin_rows], dtype=np.float64)
            order_series = np.arange(1, len(bin_rows) + 1, dtype=np.float64)
            monotonicity = float(np.corrcoef(order_series, mean_series)[0, 1]) if len(bin_rows) >= 3 else 0.0

            rows.append(
                {
                    "feature": feature,
                    "best_bin_range": f"[{best_bin['left_edge']:.4f}, {best_bin['right_edge']:.4f}]",
                    "best_bin_mean": best_bin["mean_label"],
                    "best_bin_hit_rate": best_bin["hit_rate"],
                    "best_bin_lift": best_bin["mean_label"] - base_mean,
                    "worst_bin_range": f"[{worst_bin['left_edge']:.4f}, {worst_bin['right_edge']:.4f}]",
                    "worst_bin_lift": worst_bin["mean_label"] - base_mean,
                    "monotonicity": monotonicity,
                    "base_hit_rate": base_hit,
                }
            )

        if not rows:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "best_bin_range": pl.String,
                    "best_bin_mean": pl.Float64,
                    "best_bin_hit_rate": pl.Float64,
                    "best_bin_lift": pl.Float64,
                    "worst_bin_range": pl.String,
                    "worst_bin_lift": pl.Float64,
                    "monotonicity": pl.Float64,
                    "base_hit_rate": pl.Float64,
                }
            )

        return (
            pl.DataFrame(rows)
            .with_columns(
                [
                    pl.col("best_bin_mean").round(4),
                    pl.col("best_bin_hit_rate").round(4),
                    pl.col("best_bin_lift").round(4),
                    pl.col("worst_bin_lift").round(4),
                    pl.col("monotonicity").round(4),
                    pl.col("base_hit_rate").round(4),
                ]
            )
            .sort(["best_bin_lift", "monotonicity"], descending=[True, True])
        )

    def build_feature_bin_detail(df, feature, label_col, positive_threshold, n_bins):
        df_feature = (
            df.select([feature, label_col])
            .filter(pl.col(feature).is_not_null() & pl.col(label_col).is_not_null())
        )
        if df_feature.height < 200:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "bin": pl.Int64,
                    "range": pl.String,
                    "count": pl.Int64,
                    "label_mean": pl.Float64,
                    "hit_rate": pl.Float64,
                }
            )

        x_vals = df_feature[feature].to_numpy().astype(np.float64)
        y_vals = df_feature[label_col].to_numpy().astype(np.float64)
        edges = np.unique(np.quantile(x_vals, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "bin": pl.Int64,
                    "range": pl.String,
                    "count": pl.Int64,
                    "label_mean": pl.Float64,
                    "hit_rate": pl.Float64,
                }
            )

        bin_ids = np.digitize(x_vals, edges[1:-1], right=True)
        rows = []
        for bin_idx in range(len(edges) - 1):
            mask = bin_ids == bin_idx
            count = int(mask.sum())
            if count < 30:
                continue
            rows.append(
                {
                    "feature": feature,
                    "bin": bin_idx + 1,
                    "range": f"[{edges[bin_idx]:.4f}, {edges[bin_idx + 1]:.4f}]",
                    "count": count,
                    "label_mean": float(np.mean(y_vals[mask])),
                    "hit_rate": float(np.mean(y_vals[mask] >= positive_threshold)),
                }
            )

        if not rows:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "bin": pl.Int64,
                    "range": pl.String,
                    "count": pl.Int64,
                    "label_mean": pl.Float64,
                    "hit_rate": pl.Float64,
                }
            )

        return pl.DataFrame(rows).with_columns(
            [pl.col("label_mean").round(4), pl.col("hit_rate").round(4)]
        )

    def build_decay_table(decay_summary, factor_cols):
        horizons = list(decay_summary)
        rows = []
        for factor in factor_cols:
            row = {"factor": factor}
            for horizon in horizons:
                metric = decay_summary.get(horizon, {}).get(factor, {})
                row[f"{horizon}_ic_mean"] = round(float(metric.get("ic_mean", 0.0)), 4)
                row[f"{horizon}_icir"] = round(float(metric.get("icir", 0.0)), 4)
            rows.append(row)
        if not rows:
            return pl.DataFrame(schema={"factor": pl.String})
        return pl.DataFrame(rows)

    return (
        build_bin_scoreboard,
        build_decay_table,
        build_feature_bin_detail,
        build_seed_summary,
        ordered_unique,
    )


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
    conn = duckdb.connect(DB_PATH, read_only=True)
    st_blacklist = get_st_blacklist_pl(ST_SNAPSHOT_DATE)
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(
            pl.col("date").is_between(
                pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
                pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
            )
        )
    )

    data_scope = pl.DataFrame(
        [
            {"item": "date_range", "value": f"{START_DATE} ~ {END_DATE}"},
            {"item": "st_snapshot_date", "value": ST_SNAPSHOT_DATE},
            {"item": "st_excluded_count", "value": str(len(st_blacklist))},
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 1. 数据范围")
    print("=" * 72)
    print(data_scope)
    return (q_full,)


@app.cell
def _(
    LOOSE_PERIODS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    SEED_J_MAX,
    analysis_feature_cols,
    build_b1_research_frame,
    pl,
    q_full,
    train_feature_cols,
):
    df_all = build_b1_research_frame(
        q_full,
        mv_min=MV_MIN,
        mv_max=MV_MAX,
        min_list_days=MIN_LIST_DAYS,
        seed_j_max=SEED_J_MAX,
        loose_periods=LOOSE_PERIODS,
        include_rotation_kbar_features=True,
    )

    available_analysis_feature_cols = [col for col in analysis_feature_cols if col in df_all.columns]
    available_train_feature_cols = [col for col in train_feature_cols if col in df_all.columns]
    universe_summary = pl.DataFrame(
        [
            {"item": "rows", "value": f"{df_all.height:,}"},
            {"item": "date_min", "value": str(df_all["date"].min())},
            {"item": "date_max", "value": str(df_all["date"].max())},
            {"item": "codes", "value": str(df_all["code"].n_unique())},
            {"item": "analysis_feature_count", "value": str(len(available_analysis_feature_cols))},
            {"item": "train_feature_count", "value": str(len(available_train_feature_cols))},
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 2. 研究底表")
    print("=" * 72)
    print(universe_summary)
    return (
        available_analysis_feature_cols,
        available_train_feature_cols,
        df_all,
    )


@app.cell
def _(build_seed_summary, df_all):
    seed_summary = build_seed_summary(df_all, 0.08)
    print("\n" + "=" * 72)
    print("  Step 3. Seed Pool 概览")
    print("=" * 72)
    print(seed_summary)
    return


@app.cell
def _(
    ACTIVE_SEED_COL,
    LABEL_COL,
    POSITIVE_LABEL_THRESHOLD,
    USE_BULL_ONLY,
    available_analysis_feature_cols,
    available_train_feature_cols,
    df_all,
    pl,
):
    lab_mask = pl.col(ACTIVE_SEED_COL)
    if USE_BULL_ONLY:
        lab_mask = lab_mask & pl.col("is_manual_bull")

    df_lab = df_all.filter(
        lab_mask & pl.col(LABEL_COL).is_not_null() & pl.col(LABEL_COL).is_not_nan()
    )

    summary_rows = [
        {"item": "rows", "value": f"{df_lab.height:,}"},
        {"item": "dates", "value": str(df_lab["date"].n_unique()) if df_lab.height else "0"},
        {"item": "codes", "value": str(df_lab["code"].n_unique()) if df_lab.height else "0"},
        {"item": "label_mean", "value": f"{float(df_lab[LABEL_COL].mean()):+.4f}" if df_lab.height else "0.0000"},
        {
            "item": f"positive_rate (>= {POSITIVE_LABEL_THRESHOLD:.2%})",
            "value": f"{float((df_lab[LABEL_COL] >= POSITIVE_LABEL_THRESHOLD).mean()):.2%}" if df_lab.height else "0.00%",
        },
        {"item": "analysis_feature_count", "value": str(len(available_analysis_feature_cols))},
        {"item": "train_feature_count", "value": str(len(available_train_feature_cols))},
    ]
    if "textbook_b1_score" in df_lab.columns:
        summary_rows.append(
            {
                "item": "textbook_score_mean",
                "value": f"{float(df_lab['textbook_b1_score'].mean()):.4f}" if df_lab.height else "0.0000",
            }
        )
    for component_col in (
        "textbook_trend_score",
        "textbook_kbar_score",
        "textbook_volume_score",
        "textbook_trigger_score",
    ):
        if component_col in df_lab.columns:
            summary_rows.append(
                {
                    "item": f"{component_col}_mean",
                    "value": f"{float(df_lab[component_col].mean()):.4f}" if df_lab.height else "0.0000",
                }
            )
    if "is_textbook_b1" in df_lab.columns:
        summary_rows.append(
            {
                "item": "textbook_positive_rate",
                "value": f"{float(df_lab['is_textbook_b1'].mean()):.2%}" if df_lab.height else "0.00%",
            }
        )
    if "is_textbook_case" in df_lab.columns:
        summary_rows.append(
            {
                "item": "textbook_case_rows",
                "value": str(df_lab.filter(pl.col("is_textbook_case")).height) if df_lab.height else "0",
            }
        )
    lab_summary = pl.DataFrame(summary_rows)
    print("\n" + "=" * 72)
    print("  Step 4. Lab 样本")
    print("=" * 72)
    print(lab_summary)
    return (df_lab,)


@app.cell
def _(
    B1_FEATURE_GROUPS,
    B1_FEATURE_GROUP_LABELS,
    LABEL_COL,
    MIN_DAILY_SAMPLES,
    available_analysis_feature_cols,
    available_train_feature_cols,
    build_ic_summary_frame,
    calc_factor_ic,
    df_lab,
    ordered_unique,
    pl,
    summarize_factor_groups,
):
    print("\n" + "=" * 72)
    print("  Step 5. 因子 IC 画像")
    print("=" * 72)

    analysis_ic_results = calc_factor_ic(
        df_lab,
        available_analysis_feature_cols,
        label=LABEL_COL,
        min_samples=MIN_DAILY_SAMPLES,
    )
    analysis_ic_summary = build_ic_summary_frame(analysis_ic_results)
    comparison_feature_cols = ordered_unique(
        [*available_analysis_feature_cols, *available_train_feature_cols]
    )
    comparison_ic_results = (
        analysis_ic_results
        if comparison_feature_cols == available_analysis_feature_cols
        else calc_factor_ic(
            df_lab,
            comparison_feature_cols,
            label=LABEL_COL,
            min_samples=MIN_DAILY_SAMPLES,
            verbose=False,
        )
    )
    comparison_ic_summary = build_ic_summary_frame(comparison_ic_results)
    active_group_defs = {
        group_key: tuple(
            factor for factor in factor_cols if factor in available_analysis_feature_cols
        )
        for group_key, factor_cols in B1_FEATURE_GROUPS.items()
        if any(factor in available_analysis_feature_cols for factor in factor_cols)
    }
    active_group_labels = {
        group_key: B1_FEATURE_GROUP_LABELS.get(group_key, group_key)
        for group_key in active_group_defs
    }
    group_summary = summarize_factor_groups(
        analysis_ic_results,
        active_group_defs,
        active_group_labels,
    )
    frozen_train_ic = (
        comparison_ic_summary.filter(pl.col("factor").is_in(available_train_feature_cols))
        if comparison_ic_summary.height
        else pl.DataFrame(schema=comparison_ic_summary.schema)
    )

    print("\n  分组汇总:")
    print(group_summary)
    if set(available_analysis_feature_cols) != set(available_train_feature_cols):
        print("\n  说明: 当前为 analysis/train 分离模式，下面训练冻结集画像基于并集补算。")
    print("\n  冻结训练特征当前画像:")
    print(frozen_train_ic)
    return (
        analysis_ic_summary,
        comparison_ic_results,
        comparison_ic_summary,
        frozen_train_ic,
        group_summary,
    )


@app.cell
def _(
    LABEL_COL,
    N_BINS,
    POSITIVE_LABEL_THRESHOLD,
    available_analysis_feature_cols,
    build_bin_scoreboard,
    df_lab,
):
    bin_scoreboard = build_bin_scoreboard(
        df_lab,
        available_analysis_feature_cols,
        LABEL_COL,
        POSITIVE_LABEL_THRESHOLD,
        N_BINS,
    )
    print("\n" + "=" * 72)
    print("  Step 6. 单变量分箱得分榜")
    print("=" * 72)
    print(bin_scoreboard)
    return (bin_scoreboard,)


@app.cell
def _(
    LABEL_COL,
    N_BINS,
    POSITIVE_LABEL_THRESHOLD,
    REVIEW_FEATURE,
    available_analysis_feature_cols,
    bin_scoreboard,
    build_feature_bin_detail,
    df_lab,
):
    review_feature = REVIEW_FEATURE
    if review_feature not in available_analysis_feature_cols and bin_scoreboard.height:
        review_feature = bin_scoreboard["feature"][0]

    review_table = build_feature_bin_detail(
        df_lab,
        review_feature,
        LABEL_COL,
        POSITIVE_LABEL_THRESHOLD,
        N_BINS,
    )

    print("\n" + "=" * 72)
    print(f"  Step 7. 特征深挖: {review_feature}")
    print("=" * 72)
    print(review_table)
    return


@app.cell
def _(
    DECAY_HORIZONS,
    analysis_ic_summary,
    available_train_feature_cols,
    build_decay_table,
    compute_factor_decay,
    df_lab,
    ordered_unique,
    pl,
):
    decay_factor_cols = ordered_unique(
        [
            *available_train_feature_cols,
            *(
                analysis_ic_summary["factor"].head(8).to_list()
                if analysis_ic_summary.height
                else []
            ),
        ]
    )
    decay_summary, avg_abs_icir = compute_factor_decay(
        df_lab,
        decay_factor_cols,
        horizons=DECAY_HORIZONS,
    )
    decay_table = build_decay_table(decay_summary, decay_factor_cols)
    decay_curve = pl.DataFrame(
        [
            {"horizon_days": horizon_days, "avg_abs_icir": round(value, 4)}
            for horizon_days, value in sorted(avg_abs_icir.items())
        ]
    )

    print("\n" + "=" * 72)
    print("  Step 8. 多周期衰减")
    print("=" * 72)
    print(decay_table)
    print("\n  平均绝对 ICIR 曲线:")
    print(decay_curve)
    return


@app.cell
def _(
    CORR_SAMPLE_N,
    CORR_THRESHOLD,
    RUN_CORR_DIAGNOSTICS,
    analysis_ic_summary,
    available_train_feature_cols,
    calc_factor_corr,
    comparison_ic_results,
    df_lab,
    find_redundant_factors,
    group_summary,
    ordered_unique,
    pl,
):
    corr_candidates = ordered_unique(
        [
            *available_train_feature_cols,
            *(group_summary["top_factor"].to_list() if group_summary.height else []),
            *(
                analysis_ic_summary["factor"].head(10).to_list()
                if analysis_ic_summary.height
                else []
            ),
        ]
    )
    corr_keep_cols = list(corr_candidates)
    corr_drop_cols = []
    corr_decisions = pl.DataFrame(
        schema={
            "kept": pl.String,
            "dropped": pl.String,
            "corr": pl.Float64,
            "reason": pl.String,
        }
    )

    print("\n" + "=" * 72)
    print("  Step 9. 相关性与冗余诊断")
    print("=" * 72)
    if not RUN_CORR_DIAGNOSTICS or len(corr_candidates) < 2:
        print("  当前跳过相关性诊断。")
    else:
        corr_matrix, corr_factor_names = calc_factor_corr(
            df_lab,
            corr_candidates,
            method="spearman",
            sample_n=CORR_SAMPLE_N,
        )
        corr_keep_cols, corr_drop_cols, decisions = find_redundant_factors(
            corr_matrix,
            corr_factor_names,
            ic_results=comparison_ic_results,
            threshold=CORR_THRESHOLD,
        )
        corr_decisions = (
            pl.DataFrame(
                [
                    {"kept": kept, "dropped": dropped, "corr": round(corr, 4), "reason": reason}
                    for kept, dropped, corr, reason in decisions
                ]
            )
            if decisions
            else corr_decisions
        )
        print(f"  keep: {corr_keep_cols}")
        print(f"  drop: {corr_drop_cols}")
        print(corr_decisions)
    return (corr_drop_cols,)


@app.cell
def _(
    analysis_ic_summary,
    available_train_feature_cols,
    comparison_ic_summary,
    corr_drop_cols,
    frozen_train_ic,
    group_summary,
    ordered_unique,
    pl,
):
    frozen_train_cols = list(available_train_feature_cols)
    group_top_cols = ordered_unique(group_summary["top_factor"].to_list() if group_summary.height else [])
    watchlist = (
        analysis_ic_summary
        .filter(~pl.col("factor").is_in(frozen_train_cols))
        .head(6)
        if analysis_ic_summary.height
        else pl.DataFrame(schema={"factor": pl.String})
    )
    watchlist_cols = watchlist["factor"].to_list() if watchlist.height else []
    ic_abs_map = {
        row["factor"]: float(row["abs_ICIR"])
        for row in (
            comparison_ic_summary.select(["factor", "abs_ICIR"]).to_dicts()
            if comparison_ic_summary.height
            else []
        )
    }

    pruned_train_cols = [factor for factor in frozen_train_cols if factor not in corr_drop_cols]
    replacement_candidates = [
        factor
        for factor in ordered_unique([*watchlist_cols, *group_top_cols])
        if factor not in pruned_train_cols and factor not in corr_drop_cols
    ]

    suggested_next_freeze = list(pruned_train_cols)
    recommended_drops: list[str] = []
    recommended_adds: list[str] = []
    for candidate in replacement_candidates:
        weakest_factor = min(
            suggested_next_freeze,
            key=lambda factor: ic_abs_map.get(factor, 0.0),
            default=None,
        )
        if weakest_factor is None:
            break

        candidate_strength = ic_abs_map.get(candidate, 0.0)
        weakest_strength = ic_abs_map.get(weakest_factor, 0.0)
        if candidate_strength <= weakest_strength:
            continue

        suggested_next_freeze = [
            factor for factor in suggested_next_freeze if factor != weakest_factor
        ]
        suggested_next_freeze.append(candidate)
        recommended_drops.append(weakest_factor)
        recommended_adds.append(candidate)

    final_snapshot = [
        {"bucket": "frozen_train_cols", "value": ", ".join(frozen_train_cols)},
        {"bucket": "pruned_train_cols", "value": ", ".join(pruned_train_cols)},
        {"bucket": "group_top_cols", "value": ", ".join(group_top_cols)},
        {"bucket": "replacement_candidates", "value": ", ".join(replacement_candidates)},
        {"bucket": "recommended_drops", "value": ", ".join(recommended_drops)},
        {"bucket": "recommended_adds", "value": ", ".join(recommended_adds)},
        {"bucket": "suggested_next_freeze", "value": ", ".join(suggested_next_freeze)},
    ]

    print("\n" + "=" * 72)
    print("  Step 10. Lab 结论")
    print("=" * 72)
    for i in final_snapshot:
        print(i)
    print("\n  当前冻结训练集画像:")
    print(frozen_train_ic)
    print("\n  尚未冻结但值得观察的 watchlist:")
    print(watchlist)
    return


if __name__ == "__main__":
    app.run()
