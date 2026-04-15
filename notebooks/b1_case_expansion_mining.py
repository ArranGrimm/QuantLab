import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    from utils import build_b1_research_frame, get_st_blacklist_pl, load_daily_data_full
    from utils.b1_feature_pool import B1_TEXTBOOK_CASES, B1_TEXTBOOK_SCORE_FEATURE_COLS

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0
    ACTIVE_SEED_COL = "seed_mid"
    USE_BULL_ONLY = False
    INCLUDE_ROTATION_KBAR_FEATURES = False

    LOOKBACK_DAYS = 25
    CURVE_POINTS = 25
    FEATURE_WEIGHT = 0.65
    CURVE_WEIGHT = 0.35
    MIN_TOTAL_SIMILARITY = 0.72

    MIN_FWD_MFE_10D = 0.15
    MAX_FWD_MAE_10D = -0.08
    MIN_FWD_MFE_RISK_ADJ_10D = 0.09

    CASE_VECTOR_COLS = list(B1_TEXTBOOK_SCORE_FEATURE_COLS)

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
    return (
        ACTIVE_SEED_COL,
        CASE_VECTOR_COLS,
        CURVE_POINTS,
        CURVE_WEIGHT,
        DB_PATH,
        END_DATE,
        FEATURE_WEIGHT,
        INCLUDE_ROTATION_KBAR_FEATURES,
        LOOKBACK_DAYS,
        LOOSE_PERIODS,
        MAX_FWD_MAE_10D,
        MIN_FWD_MFE_10D,
        MIN_FWD_MFE_RISK_ADJ_10D,
        MIN_LIST_DAYS,
        MIN_TOTAL_SIMILARITY,
        MV_MAX,
        MV_MIN,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        USE_BULL_ONLY,
        B1_TEXTBOOK_CASES,
        build_b1_research_frame,
        duckdb,
        get_st_blacklist_pl,
        load_daily_data_full,
        mo,
        np,
        pl,
    )


@app.cell
def _(
    ACTIVE_SEED_COL,
    CURVE_POINTS,
    CURVE_WEIGHT,
    END_DATE,
    FEATURE_WEIGHT,
    INCLUDE_ROTATION_KBAR_FEATURES,
    LOOKBACK_DAYS,
    MAX_FWD_MAE_10D,
    MIN_FWD_MFE_10D,
    MIN_FWD_MFE_RISK_ADJ_10D,
    MIN_TOTAL_SIMILARITY,
    START_DATE,
    USE_BULL_ONLY,
):
    print("=" * 72)
    print("  B1 Case Expansion Mining")
    print("=" * 72)
    print("  目标: 从历史强势样本中扩容更像 textbook 的 B1 案例。")
    print("  方法: seed_mid 样本池 + 结果过滤 + 结构向量相似度 + 曲线形状相似度。")
    print("")
    print(f"  date_range:         {START_DATE} ~ {END_DATE}")
    print(f"  active_seed:        {ACTIVE_SEED_COL}")
    print(f"  bull_regime_only:   {USE_BULL_ONLY}")
    print(f"  include_rot_kbar:   {INCLUDE_ROTATION_KBAR_FEATURES}")
    print(f"  lookback_days:      {LOOKBACK_DAYS}")
    print(f"  curve_points:       {CURVE_POINTS}")
    print(f"  feature_weight:     {FEATURE_WEIGHT:.2f}")
    print(f"  curve_weight:       {CURVE_WEIGHT:.2f}")
    print(f"  min_total_similarity:{MIN_TOTAL_SIMILARITY:.2f}")
    print(f"  min_fwd_mfe_10d:    {MIN_FWD_MFE_10D:.2%}")
    print(f"  max_fwd_mae_10d:    {MAX_FWD_MAE_10D:.2%}")
    print(f"  min_risk_adj_10d:   {MIN_FWD_MFE_RISK_ADJ_10D:.2%}")
    return


@app.cell
def _(DB_PATH, END_DATE, START_DATE, ST_SNAPSHOT_DATE, duckdb, get_st_blacklist_pl, load_daily_data_full, pl):
    conn = duckdb.connect(DB_PATH, read_only=True)
    st_blacklist = get_st_blacklist_pl(ST_SNAPSHOT_DATE)
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()
    q_full = (
        load_daily_data_full(conn)
        .filter(
            (pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
            & (pl.col("date") <= pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
        )
        .join(st_blacklist_df, on="code", how="anti")
    )
    return (q_full,)


@app.cell
def _(
    ACTIVE_SEED_COL,
    B1_TEXTBOOK_CASES,
    INCLUDE_ROTATION_KBAR_FEATURES,
    LOOSE_PERIODS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
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
    seed_filter = pl.col(ACTIVE_SEED_COL)
    if USE_BULL_ONLY:
        seed_filter = seed_filter & pl.col("is_manual_bull")
    df_seed = df_all.filter(seed_filter)
    case_df = (
        pl.DataFrame(B1_TEXTBOOK_CASES)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .rename({"name": "case_name"})
    )
    df_cases = (
        df_all.join(case_df, on=["code", "date"], how="inner")
        .sort(["date", "code"])
    )

    print("\n" + "=" * 72)
    print("  Step 1. 历史样本池")
    print("=" * 72)
    print(
        pl.DataFrame(
            [
                {"item": "rows_all", "value": f"{df_all.height:,}"},
                {"item": "rows_seed", "value": f"{df_seed.height:,}"},
                {"item": "dates_seed", "value": str(df_seed["date"].n_unique()) if df_seed.height else "0"},
                {"item": "cases_found", "value": str(df_cases.height)},
            ]
        )
    )
    return case_df, df_all, df_cases, df_seed


@app.cell
def _(CASE_VECTOR_COLS, CURVE_POINTS, LOOKBACK_DAYS, np, pl):
    def _normalize_curve(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.zeros(CURVE_POINTS, dtype=np.float64)
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if max_val - min_val <= 1e-8:
            return np.zeros(len(values), dtype=np.float64)
        return (values - min_val) / (max_val - min_val)

    def _resample_curve(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.zeros(CURVE_POINTS, dtype=np.float64)
        if len(values) == CURVE_POINTS:
            return values.astype(np.float64)
        base_x = np.arange(len(values), dtype=np.float64)
        target_x = np.linspace(0, len(values) - 1, CURVE_POINTS)
        return np.interp(target_x, base_x, values).astype(np.float64)

    def build_curve_accessor(df_prices: pl.DataFrame):
        price_map: dict[str, tuple[list, np.ndarray, dict]] = {}
        for subdf in df_prices.partition_by("code", maintain_order=True):
            code = subdf["code"][0]
            dates = subdf["date"].to_list()
            close_vals = subdf["close_adj"].to_numpy().astype(np.float64)
            date_to_idx = {date_val: idx for idx, date_val in enumerate(dates)}
            price_map[code] = (dates, close_vals, date_to_idx)

        def get_curve(code: str, date_value):
            item = price_map.get(code)
            if item is None:
                return None
            _, close_vals, date_to_idx = item
            idx = date_to_idx.get(date_value)
            if idx is None:
                return None
            start_idx = max(0, idx - LOOKBACK_DAYS + 1)
            window = close_vals[start_idx : idx + 1]
            if len(window) < max(10, LOOKBACK_DAYS // 2):
                return None
            return _normalize_curve(_resample_curve(window))

        return get_curve

    def calc_curve_similarity(curve_a: np.ndarray | None, curve_b: np.ndarray | None) -> float:
        if curve_a is None or curve_b is None:
            return 0.0
        diff = curve_a - curve_b
        distance = float(np.sqrt(np.mean(diff * diff)))
        return max(0.0, 1.0 - distance)

    def build_feature_scales(df: pl.DataFrame) -> dict[str, float]:
        scales: dict[str, float] = {}
        for col_name in CASE_VECTOR_COLS:
            if col_name not in df.columns:
                continue
            series = df[col_name].drop_nulls().drop_nans()
            if series.is_empty():
                scales[col_name] = 1.0
                continue
            q1 = float(series.quantile(0.25, interpolation="linear"))
            q3 = float(series.quantile(0.75, interpolation="linear"))
            median = float(series.median())
            scale = max((q3 - q1) * 2.0, abs(median) * 0.35, 1e-4)
            scales[col_name] = scale
        return scales

    def calc_feature_similarity(candidate_row: dict, case_row: dict, scales: dict[str, float]) -> float:
        similarities = []
        for col_name in CASE_VECTOR_COLS:
            cand_val = candidate_row.get(col_name)
            case_val = case_row.get(col_name)
            if cand_val is None or case_val is None:
                continue
            if isinstance(cand_val, float) and np.isnan(cand_val):
                continue
            if isinstance(case_val, float) and np.isnan(case_val):
                continue
            scale = scales.get(col_name, 1.0)
            sim = max(0.0, 1.0 - abs(float(cand_val) - float(case_val)) / scale)
            similarities.append(sim)
        return float(np.mean(similarities)) if similarities else 0.0

    return build_curve_accessor, build_feature_scales, calc_curve_similarity, calc_feature_similarity


@app.cell
def _(
    MAX_FWD_MAE_10D,
    MIN_FWD_MFE_10D,
    MIN_FWD_MFE_RISK_ADJ_10D,
    df_seed,
    pl,
):
    df_candidates = (
        df_seed
        .filter(~pl.col("is_textbook_case"))
        .filter(pl.col("fwd_mfe_10d") >= MIN_FWD_MFE_10D)
        .filter(pl.col("fwd_mae_10d") >= MAX_FWD_MAE_10D)
        .filter(pl.col("fwd_mfe_risk_adj_10d") >= MIN_FWD_MFE_RISK_ADJ_10D)
        .sort(["date", "code"])
    )
    candidate_summary = pl.DataFrame(
        [
            {"item": "candidate_rows", "value": f"{df_candidates.height:,}"},
            {"item": "candidate_dates", "value": str(df_candidates["date"].n_unique()) if df_candidates.height else "0"},
            {
                "item": "candidate_mean_fwd_mfe_10d",
                "value": f"{float(df_candidates['fwd_mfe_10d'].mean()):.4f}" if df_candidates.height else "0.0000",
            },
            {
                "item": "candidate_mean_risk_adj_10d",
                "value": f"{float(df_candidates['fwd_mfe_risk_adj_10d'].mean()):.4f}" if df_candidates.height else "0.0000",
            },
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 2. 结果强样本过滤")
    print("=" * 72)
    print(candidate_summary)
    return candidate_summary, df_candidates


@app.cell
def _(
    CURVE_WEIGHT,
    FEATURE_WEIGHT,
    MIN_TOTAL_SIMILARITY,
    build_curve_accessor,
    build_feature_scales,
    calc_curve_similarity,
    calc_feature_similarity,
    df_all,
    df_candidates,
    df_cases,
    np,
    pl,
):
    price_accessor = build_curve_accessor(
        df_all.select(["code", "date", "close_adj"]).sort(["code", "date"])
    )
    feature_scales = build_feature_scales(df_candidates)

    case_rows = df_cases.to_dicts()
    candidate_rows = df_candidates.to_dicts()
    match_rows = []
    for candidate_row in candidate_rows:
        candidate_curve = price_accessor(candidate_row["code"], candidate_row["date"])
        best_total = -1.0
        best_payload = None
        for case_row in case_rows:
            case_curve = price_accessor(case_row["code"], case_row["date"])
            feature_sim = calc_feature_similarity(candidate_row, case_row, feature_scales)
            curve_sim = calc_curve_similarity(candidate_curve, case_curve)
            total_sim = FEATURE_WEIGHT * feature_sim + CURVE_WEIGHT * curve_sim
            if total_sim > best_total:
                best_total = total_sim
                best_payload = {
                    "best_match_case": case_row["case_name"],
                    "best_match_case_code": case_row["code"],
                    "best_match_case_date": case_row["date"],
                    "feature_similarity": feature_sim,
                    "curve_similarity": curve_sim,
                    "total_similarity": total_sim,
                }
        if best_payload is None:
            continue
        match_rows.append(
            {
                "date": candidate_row["date"],
                "code": candidate_row["code"],
                "best_match_case": best_payload["best_match_case"],
                "best_match_case_code": best_payload["best_match_case_code"],
                "best_match_case_date": best_payload["best_match_case_date"],
                "feature_similarity": best_payload["feature_similarity"],
                "curve_similarity": best_payload["curve_similarity"],
                "total_similarity": best_payload["total_similarity"],
                "expanded_textbook_candidate": best_payload["total_similarity"] >= MIN_TOTAL_SIMILARITY,
                "fwd_mfe_10d": candidate_row["fwd_mfe_10d"],
                "fwd_mae_10d": candidate_row["fwd_mae_10d"],
                "fwd_mfe_risk_adj_10d": candidate_row["fwd_mfe_risk_adj_10d"],
                "textbook_b1_score": candidate_row.get("textbook_b1_score"),
                "textbook_trend_score": candidate_row.get("textbook_trend_score"),
                "textbook_kbar_score": candidate_row.get("textbook_kbar_score"),
                "textbook_volume_score": candidate_row.get("textbook_volume_score"),
                "textbook_trigger_score": candidate_row.get("textbook_trigger_score"),
            }
        )

    if match_rows:
        df_matches = (
            pl.DataFrame(match_rows)
            .with_columns(
                [
                    pl.col("feature_similarity").round(4),
                    pl.col("curve_similarity").round(4),
                    pl.col("total_similarity").round(4),
                    pl.col("fwd_mfe_10d").round(4),
                    pl.col("fwd_mae_10d").round(4),
                    pl.col("fwd_mfe_risk_adj_10d").round(4),
                    pl.col("textbook_b1_score").round(4),
                    pl.col("textbook_trend_score").round(4),
                    pl.col("textbook_kbar_score").round(4),
                    pl.col("textbook_volume_score").round(4),
                    pl.col("textbook_trigger_score").round(4),
                ]
            )
            .sort(["total_similarity", "fwd_mfe_risk_adj_10d"], descending=[True, True])
        )
    else:
        df_matches = pl.DataFrame(
            schema={
                "date": pl.Date,
                "code": pl.String,
                "best_match_case": pl.String,
                "best_match_case_code": pl.String,
                "best_match_case_date": pl.Date,
                "feature_similarity": pl.Float64,
                "curve_similarity": pl.Float64,
                "total_similarity": pl.Float64,
                "expanded_textbook_candidate": pl.Boolean,
                "fwd_mfe_10d": pl.Float64,
                "fwd_mae_10d": pl.Float64,
                "fwd_mfe_risk_adj_10d": pl.Float64,
                "textbook_b1_score": pl.Float64,
                "textbook_trend_score": pl.Float64,
                "textbook_kbar_score": pl.Float64,
                "textbook_volume_score": pl.Float64,
                "textbook_trigger_score": pl.Float64,
            }
        )

    df_expanded = df_matches.filter(pl.col("expanded_textbook_candidate"))
    expansion_summary = pl.DataFrame(
        [
            {"item": "matched_rows", "value": f"{df_matches.height:,}"},
            {"item": "expanded_rows", "value": f"{df_expanded.height:,}"},
            {
                "item": "expanded_ratio",
                "value": f"{(df_expanded.height / max(df_matches.height, 1)):.2%}",
            },
            {
                "item": "expanded_mean_similarity",
                "value": f"{float(df_expanded['total_similarity'].mean()):.4f}" if df_expanded.height else "0.0000",
            },
            {
                "item": "expanded_mean_risk_adj_10d",
                "value": f"{float(df_expanded['fwd_mfe_risk_adj_10d'].mean()):.4f}" if df_expanded.height else "0.0000",
            },
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 3. 案例相似度扩容")
    print("=" * 72)
    print(expansion_summary)
    return df_expanded, df_matches


@app.cell
def _(df_expanded, df_matches, mo, pl):
    mo.md("## Top 相似候选")
    display_cols = [
        "date",
        "code",
        "best_match_case",
        "total_similarity",
        "feature_similarity",
        "curve_similarity",
        "fwd_mfe_10d",
        "fwd_mae_10d",
        "fwd_mfe_risk_adj_10d",
        "textbook_b1_score",
    ]
    top_rows = df_matches.select(display_cols).head(50)
    mo.ui.table(top_rows.to_pandas())
    return


@app.cell
def _(df_expanded, mo, pl):
    mo.md("## 扩容候选分组摘要")
    if df_expanded.is_empty():
        summary = pl.DataFrame(
            schema={
                "best_match_case": pl.String,
                "rows": pl.Int64,
                "mean_similarity": pl.Float64,
                "mean_mfe_10d": pl.Float64,
                "mean_mae_10d": pl.Float64,
                "mean_risk_adj_10d": pl.Float64,
            }
        )
    else:
        summary = (
            df_expanded.group_by("best_match_case")
            .agg(
                [
                    pl.len().alias("rows"),
                    pl.col("total_similarity").mean().round(4).alias("mean_similarity"),
                    pl.col("fwd_mfe_10d").mean().round(4).alias("mean_mfe_10d"),
                    pl.col("fwd_mae_10d").mean().round(4).alias("mean_mae_10d"),
                    pl.col("fwd_mfe_risk_adj_10d").mean().round(4).alias("mean_risk_adj_10d"),
                ]
            )
            .sort("rows", descending=True)
        )
    mo.ui.table(summary.to_pandas())
    return


@app.cell
def _(df_expanded, mo, pl):
    mo.md("## 扩容候选明细")
    detail_cols = [
        "date",
        "code",
        "best_match_case",
        "best_match_case_date",
        "total_similarity",
        "feature_similarity",
        "curve_similarity",
        "fwd_mfe_10d",
        "fwd_mae_10d",
        "fwd_mfe_risk_adj_10d",
        "textbook_trend_score",
        "textbook_kbar_score",
        "textbook_volume_score",
        "textbook_trigger_score",
    ]
    mo.ui.table(df_expanded.select(detail_cols).head(100).to_pandas())
    return


if __name__ == "__main__":
    app.run()
