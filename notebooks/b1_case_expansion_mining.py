import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl
    from pathlib import Path

    from manifests import B1_BASE_TEXTBOOK_CASES, B1_BASE_TEXTBOOK_CASES_VERSION
    from utils import build_b1_research_frame, get_st_blacklist_pl, load_daily_data_full
    from utils.b1_feature_pool import B1_TEXTBOOK_SCORE_FEATURE_COLS

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    CASE_MV_MIN = 0
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0
    ACTIVE_SEED_COL = "seed_mid"
    USE_BULL_ONLY = False
    INCLUDE_ROTATION_KBAR_FEATURES = False

    LOOKBACK_DAYS = 25
    CURVE_POINTS = 25
    TOP_K_MATCHES = 3
    FEATURE_WEIGHT = 0.65
    CURVE_WEIGHT = 0.35
    MIN_TOTAL_SIMILARITY = 0.72

    EXPANDED_CASES_VERSION = "case_expansion_top1_v2_conservative"
    EXPANDED_CASE_MIN_TOTAL_SIM = 0.84
    EXPANDED_CASE_MIN_RISK_ADJ_10D = 0.18
    EXPANDED_CASE_MIN_TEXTBOOK_SCORE = 0.75
    EXPANDED_CASES_PER_ARCHETYPE = 8
    EXPANDED_CASES_OUTPUT_PATH = Path("../manifests/b1_expanded_textbook_cases.py")
    WRITE_EXPANDED_CASES_MANIFEST = False

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
        B1_BASE_TEXTBOOK_CASES,
        B1_BASE_TEXTBOOK_CASES_VERSION,
        CASE_MV_MIN,
        CASE_VECTOR_COLS,
        CURVE_POINTS,
        CURVE_WEIGHT,
        DB_PATH,
        END_DATE,
        EXPANDED_CASES_OUTPUT_PATH,
        EXPANDED_CASES_PER_ARCHETYPE,
        EXPANDED_CASES_VERSION,
        EXPANDED_CASE_MIN_RISK_ADJ_10D,
        EXPANDED_CASE_MIN_TEXTBOOK_SCORE,
        EXPANDED_CASE_MIN_TOTAL_SIM,
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
        TOP_K_MATCHES,
        USE_BULL_ONLY,
        WRITE_EXPANDED_CASES_MANIFEST,
        Path,
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
    B1_BASE_TEXTBOOK_CASES,
    B1_BASE_TEXTBOOK_CASES_VERSION,
    CURVE_POINTS,
    CURVE_WEIGHT,
    END_DATE,
    EXPANDED_CASES_PER_ARCHETYPE,
    EXPANDED_CASES_VERSION,
    EXPANDED_CASE_MIN_RISK_ADJ_10D,
    EXPANDED_CASE_MIN_TEXTBOOK_SCORE,
    EXPANDED_CASE_MIN_TOTAL_SIM,
    FEATURE_WEIGHT,
    INCLUDE_ROTATION_KBAR_FEATURES,
    LOOKBACK_DAYS,
    MAX_FWD_MAE_10D,
    MIN_FWD_MFE_10D,
    MIN_FWD_MFE_RISK_ADJ_10D,
    MIN_TOTAL_SIMILARITY,
    START_DATE,
    TOP_K_MATCHES,
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
    print(f"  top_k_matches:      {TOP_K_MATCHES}")
    print(f"  base_case_count:    {len(B1_BASE_TEXTBOOK_CASES)}")
    print(f"  base_case_version:  {B1_BASE_TEXTBOOK_CASES_VERSION}")
    print(f"  feature_weight:     {FEATURE_WEIGHT:.2f}")
    print(f"  curve_weight:       {CURVE_WEIGHT:.2f}")
    print(f"  min_total_similarity:{MIN_TOTAL_SIMILARITY:.2f}")
    print(f"  min_fwd_mfe_10d:    {MIN_FWD_MFE_10D:.2%}")
    print(f"  max_fwd_mae_10d:    {MAX_FWD_MAE_10D:.2%}")
    print(f"  min_risk_adj_10d:   {MIN_FWD_MFE_RISK_ADJ_10D:.2%}")
    print("")
    print("  artifact:")
    print(f"    version:          {EXPANDED_CASES_VERSION}")
    print(f"    min_total_sim:    {EXPANDED_CASE_MIN_TOTAL_SIM:.2f}")
    print(f"    min_risk_adj_10d: {EXPANDED_CASE_MIN_RISK_ADJ_10D:.2%}")
    print(f"    min_textbook:     {EXPANDED_CASE_MIN_TEXTBOOK_SCORE:.2f}")
    print(f"    per_archetype:    {EXPANDED_CASES_PER_ARCHETYPE}")
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
    B1_BASE_TEXTBOOK_CASES,
    CASE_MV_MIN,
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
    df_case_source = build_b1_research_frame(
        q_full,
        mv_min=CASE_MV_MIN,
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
        pl.DataFrame(B1_BASE_TEXTBOOK_CASES)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .rename({"name": "case_name"})
    )
    df_cases = (
        df_case_source.join(case_df, on=["code", "date"], how="inner")
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
                {"item": "case_mv_min", "value": str(CASE_MV_MIN)},
                {"item": "base_cases_found", "value": str(df_cases.height)},
            ]
        )
    )
    return df_all, df_cases, df_seed


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

    return (
        build_curve_accessor,
        build_feature_scales,
        calc_curve_similarity,
        calc_feature_similarity,
    )


@app.cell
def _(MAX_FWD_MAE_10D, MIN_FWD_MFE_10D, MIN_FWD_MFE_RISK_ADJ_10D, df_seed, pl):
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
    return (df_candidates,)


@app.cell
def _(df_seed, df_candidates, np, pl):
    _has_tb = "is_textbook_b1" in df_seed.columns and "textbook_b1_score" in df_seed.columns

    print("\n" + "=" * 72)
    print("  Step 2b. B1 形态占比与平均表现概览")
    print("=" * 72)

    if not _has_tb:
        print("  [SKIP] df_seed 中缺少 textbook 相关列, 跳过。")
    else:
        def _safe_mean(df, col):
            if df.height == 0 or col not in df.columns:
                return float("nan")
            s = df[col].drop_nulls().drop_nans()
            return float(s.mean()) if s.len() > 0 else float("nan")

        _df_b1 = df_seed.filter(pl.col("is_textbook_b1"))
        _df_non_b1 = df_seed.filter(~pl.col("is_textbook_b1"))

        _perf_groups = [
            ("seed_mid 全体", df_seed),
            ("形态像B1 (is_textbook_b1)", _df_b1),
            ("非B1形态", _df_non_b1),
        ]
        _perf_rows = []
        for _label, _df_g in _perf_groups:
            _perf_rows.append({
                "group": _label,
                "rows": _df_g.height,
                "mean_mfe_10d": round(_safe_mean(_df_g, "fwd_mfe_10d"), 4),
                "mean_mae_10d": round(_safe_mean(_df_g, "fwd_mae_10d"), 4),
                "mean_risk_adj_10d": round(_safe_mean(_df_g, "fwd_mfe_risk_adj_10d"), 4),
                "hit_10pct": round(
                    _df_g.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.10).height / max(_df_g.height, 1), 4
                ),
                "hit_15pct": round(
                    _df_g.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height / max(_df_g.height, 1), 4
                ),
            })

        print("\n  [1] seed_mid 全体 vs B1 形态 — 平均前瞻表现:")
        print(pl.DataFrame(_perf_rows))

        _score_bins = [0.0, 0.3, 0.5, 0.65, 0.75, 0.85, 1.01]
        _bin_rows = []
        for _i in range(len(_score_bins) - 1):
            _lo, _hi = _score_bins[_i], _score_bins[_i + 1]
            _df_bin = df_seed.filter(
                (pl.col("textbook_b1_score") >= _lo) & (pl.col("textbook_b1_score") < _hi)
            )
            if _df_bin.height == 0:
                continue
            _bin_rows.append({
                "score_range": f"[{_lo:.2f}, {_hi:.2f})",
                "rows": _df_bin.height,
                "pct_of_seed": round(_df_bin.height / max(df_seed.height, 1), 4),
                "mean_mfe_10d": round(_safe_mean(_df_bin, "fwd_mfe_10d"), 4),
                "mean_risk_adj_10d": round(_safe_mean(_df_bin, "fwd_mfe_risk_adj_10d"), 4),
                "hit_15pct": round(
                    _df_bin.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height / max(_df_bin.height, 1), 4
                ),
            })
        if _bin_rows:
            print("\n  [2] textbook_b1_score 分段平均表现:")
            print(pl.DataFrame(_bin_rows))

        _seed_b1_ratio = _df_b1.height / max(df_seed.height, 1)

        _cand_total = df_candidates.height
        _cand_b1 = df_candidates.filter(pl.col("is_textbook_b1")).height if _cand_total else 0
        _cand_b1_ratio = _cand_b1 / max(_cand_total, 1)

        _median_ra = float(
            df_seed["fwd_mfe_risk_adj_10d"].drop_nulls().drop_nans().median()
        ) if df_seed.height else 0.0
        _df_above_med = df_seed.filter(pl.col("fwd_mfe_risk_adj_10d") > _median_ra)
        _am_total = _df_above_med.height
        _am_b1 = _df_above_med.filter(pl.col("is_textbook_b1")).height if _am_total else 0
        _am_b1_ratio = _am_b1 / max(_am_total, 1)

        _p90_ra = float(
            df_seed["fwd_mfe_risk_adj_10d"].drop_nulls().drop_nans().quantile(0.90, interpolation="linear")
        ) if df_seed.height else 0.0
        _df_top_dec = df_seed.filter(pl.col("fwd_mfe_risk_adj_10d") >= _p90_ra)
        _td_total = _df_top_dec.height
        _td_b1 = _df_top_dec.filter(pl.col("is_textbook_b1")).height if _td_total else 0
        _td_b1_ratio = _td_b1 / max(_td_total, 1)

        _enrich_rows = [
            {
                "sample": "seed_mid 全体 (baseline)",
                "total": df_seed.height,
                "b1_like": _df_b1.height,
                "b1_ratio": f"{_seed_b1_ratio:.2%}",
                "enrichment": "1.00x",
                "mean_risk_adj": round(_safe_mean(df_seed, "fwd_mfe_risk_adj_10d"), 4),
            },
            {
                "sample": f"高于中位数 risk_adj (>{_median_ra:.4f})",
                "total": _am_total,
                "b1_like": _am_b1,
                "b1_ratio": f"{_am_b1_ratio:.2%}",
                "enrichment": f"{_am_b1_ratio / max(_seed_b1_ratio, 1e-8):.2f}x",
                "mean_risk_adj": round(_safe_mean(_df_above_med, "fwd_mfe_risk_adj_10d"), 4),
            },
            {
                "sample": "结果强样本 (df_candidates)",
                "total": _cand_total,
                "b1_like": _cand_b1,
                "b1_ratio": f"{_cand_b1_ratio:.2%}",
                "enrichment": f"{_cand_b1_ratio / max(_seed_b1_ratio, 1e-8):.2f}x",
                "mean_risk_adj": round(_safe_mean(df_candidates, "fwd_mfe_risk_adj_10d"), 4),
            },
            {
                "sample": f"Top 10% risk_adj (>={_p90_ra:.4f})",
                "total": _td_total,
                "b1_like": _td_b1,
                "b1_ratio": f"{_td_b1_ratio:.2%}",
                "enrichment": f"{_td_b1_ratio / max(_seed_b1_ratio, 1e-8):.2f}x",
                "mean_risk_adj": round(_safe_mean(_df_top_dec, "fwd_mfe_risk_adj_10d"), 4),
            },
        ]

        print("\n  [3] 强表现样本中 B1 形态占比 (enrichment):")
        print(pl.DataFrame(_enrich_rows))
        print(f"\n  enrichment = 该组 B1 占比 / seed_mid 全体 B1 占比 ({_seed_b1_ratio:.2%})")
        if _seed_b1_ratio > 0 and not np.isnan(_cand_b1_ratio):
            if _cand_b1_ratio > _seed_b1_ratio * 1.5:
                print("  → B1 形态在强样本中显著富集, 结构信号有筛选价值。")
            elif _cand_b1_ratio > _seed_b1_ratio:
                print("  → B1 形态在强样本中有一定富集, 但幅度不大。")
            else:
                print("  → B1 形态在强样本中未见富集, 结构信号当前对收益筛选贡献有限。")
    return


@app.cell
def _(
    CURVE_WEIGHT,
    FEATURE_WEIGHT,
    MIN_TOTAL_SIMILARITY,
    TOP_K_MATCHES,
    build_curve_accessor,
    build_feature_scales,
    calc_curve_similarity,
    calc_feature_similarity,
    df_all,
    df_candidates,
    df_cases,
    pl,
):
    price_accessor = build_curve_accessor(
        df_all.select(["code", "date", "close_adj"]).sort(["code", "date"])
    )
    feature_scales = build_feature_scales(df_candidates)

    case_rows = df_cases.to_dicts()
    candidate_rows = df_candidates.to_dicts()
    topk_rows = []
    match_rows = []
    for candidate_row in candidate_rows:
        candidate_curve = price_accessor(candidate_row["code"], candidate_row["date"])
        candidate_case_scores = []
        for case_row in case_rows:
            case_curve = price_accessor(case_row["code"], case_row["date"])
            feature_sim = calc_feature_similarity(candidate_row, case_row, feature_scales)
            curve_sim = calc_curve_similarity(candidate_curve, case_curve)
            total_sim = FEATURE_WEIGHT * feature_sim + CURVE_WEIGHT * curve_sim
            candidate_case_scores.append(
                {
                    "match_case": case_row["case_name"],
                    "match_case_code": case_row["code"],
                    "match_case_date": case_row["date"],
                    "feature_similarity": feature_sim,
                    "curve_similarity": curve_sim,
                    "total_similarity": total_sim,
                }
            )
        if not candidate_case_scores:
            continue
        candidate_case_scores.sort(
            key=lambda item: (item["total_similarity"], item["feature_similarity"], item["curve_similarity"]),
            reverse=True,
        )
        best_payload = candidate_case_scores[0]
        match_rows.append(
            {
                "date": candidate_row["date"],
                "code": candidate_row["code"],
                "best_match_case": best_payload["match_case"],
                "best_match_case_code": best_payload["match_case_code"],
                "best_match_case_date": best_payload["match_case_date"],
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
        for rank_idx, case_score in enumerate(candidate_case_scores[:TOP_K_MATCHES], start=1):
            topk_rows.append(
                {
                    "date": candidate_row["date"],
                    "code": candidate_row["code"],
                    "match_rank": rank_idx,
                    "match_case": case_score["match_case"],
                    "match_case_code": case_score["match_case_code"],
                    "match_case_date": case_score["match_case_date"],
                    "feature_similarity": case_score["feature_similarity"],
                    "curve_similarity": case_score["curve_similarity"],
                    "total_similarity": case_score["total_similarity"],
                    "fwd_mfe_10d": candidate_row["fwd_mfe_10d"],
                    "fwd_mae_10d": candidate_row["fwd_mae_10d"],
                    "fwd_mfe_risk_adj_10d": candidate_row["fwd_mfe_risk_adj_10d"],
                    "textbook_b1_score": candidate_row.get("textbook_b1_score"),
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
        df_topk_matches = (
            pl.DataFrame(topk_rows)
            .with_columns(
                [
                    pl.col("feature_similarity").round(4),
                    pl.col("curve_similarity").round(4),
                    pl.col("total_similarity").round(4),
                    pl.col("fwd_mfe_10d").round(4),
                    pl.col("fwd_mae_10d").round(4),
                    pl.col("fwd_mfe_risk_adj_10d").round(4),
                    pl.col("textbook_b1_score").round(4),
                ]
            )
            .sort(["match_rank", "total_similarity", "fwd_mfe_risk_adj_10d"], descending=[False, True, True])
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
        df_topk_matches = pl.DataFrame(
            schema={
                "date": pl.Date,
                "code": pl.String,
                "match_rank": pl.Int64,
                "match_case": pl.String,
                "match_case_code": pl.String,
                "match_case_date": pl.Date,
                "feature_similarity": pl.Float64,
                "curve_similarity": pl.Float64,
                "total_similarity": pl.Float64,
                "fwd_mfe_10d": pl.Float64,
                "fwd_mae_10d": pl.Float64,
                "fwd_mfe_risk_adj_10d": pl.Float64,
                "textbook_b1_score": pl.Float64,
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
    return df_expanded, df_matches, df_topk_matches


@app.cell
def _(
    CURVE_WEIGHT,
    FEATURE_WEIGHT,
    build_curve_accessor,
    build_feature_scales,
    calc_curve_similarity,
    calc_feature_similarity,
    df_all,
    df_candidates,
    df_cases,
    pl,
):
    _price_accessor = build_curve_accessor(
        df_all.select(["code", "date", "close_adj"]).sort(["code", "date"])
    )
    case_feature_scales = build_feature_scales(df_candidates)
    _case_rows = df_cases.to_dicts()
    _pairwise_rows = []

    for _anchor_row in _case_rows:
        _anchor_curve = _price_accessor(_anchor_row["code"], _anchor_row["date"])
        for _peer_row in _case_rows:
            if _anchor_row["case_name"] == _peer_row["case_name"]:
                continue
            _peer_curve = _price_accessor(_peer_row["code"], _peer_row["date"])
            _feature_sim = calc_feature_similarity(_anchor_row, _peer_row, case_feature_scales)
            _curve_sim = calc_curve_similarity(_anchor_curve, _peer_curve)
            _total_sim = FEATURE_WEIGHT * _feature_sim + CURVE_WEIGHT * _curve_sim
            _pairwise_rows.append(
                {
                    "anchor_case": _anchor_row["case_name"],
                    "peer_case": _peer_row["case_name"],
                    "feature_similarity": _feature_sim,
                    "curve_similarity": _curve_sim,
                    "total_similarity": _total_sim,
                }
            )

    if _pairwise_rows:
        df_case_pairwise = (
            pl.DataFrame(_pairwise_rows)
            .with_columns(
                [
                    pl.col("feature_similarity").round(4),
                    pl.col("curve_similarity").round(4),
                    pl.col("total_similarity").round(4),
                ]
            )
            .sort(["anchor_case", "total_similarity"], descending=[False, True])
        )
    else:
        df_case_pairwise = pl.DataFrame(
            schema={
                "anchor_case": pl.String,
                "peer_case": pl.String,
                "feature_similarity": pl.Float64,
                "curve_similarity": pl.Float64,
                "total_similarity": pl.Float64,
            }
        )

    return (df_case_pairwise,)


@app.cell
def _(df_case_pairwise, mo):
    mo.md("## Archetype 两两相似度矩阵")
    if df_case_pairwise.is_empty():
        _pairwise_matrix_output = mo.md("当前没有可展示的案例相似度矩阵。")
    else:
        _pairwise_matrix = (
            df_case_pairwise.to_pandas()
            .pivot(index="anchor_case", columns="peer_case", values="total_similarity")
            .reset_index()
            .round(4)
        )
        _pairwise_matrix_output = mo.ui.table(_pairwise_matrix)
    _pairwise_matrix_output
    return


@app.cell
def _(df_matches, mo):
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
def _(TOP_K_MATCHES, df_topk_matches, mo, pl):
    mo.md("## Top-k Archetype 覆盖摘要")
    if df_topk_matches.is_empty():
        _topk_summary = pl.DataFrame(
            schema={
                "match_case": pl.String,
                "top1_rows": pl.Int64,
                "topk_rows": pl.Int64,
                "top1_share_in_topk": pl.Float64,
                "topk_mean_similarity": pl.Float64,
                "topk_mean_risk_adj_10d": pl.Float64,
            }
        )
    else:
        _topk_summary = (
            df_topk_matches.group_by("match_case")
            .agg(
                [
                    pl.when(pl.col("match_rank") == 1).then(1).otherwise(0).sum().alias("top1_rows"),
                    pl.len().alias("topk_rows"),
                    pl.col("total_similarity").mean().round(4).alias("topk_mean_similarity"),
                    pl.col("fwd_mfe_risk_adj_10d").mean().round(4).alias("topk_mean_risk_adj_10d"),
                ]
            )
            .with_columns(
                (
                    pl.col("top1_rows") / pl.max_horizontal(pl.col("topk_rows"), pl.lit(1))
                ).round(4).alias("top1_share_in_topk")
            )
            .sort(["topk_rows", "top1_rows"], descending=[True, True])
        )
    mo.md(f"这里展示每个案例进入候选 `Top-{TOP_K_MATCHES}` 匹配的次数。若 `topk_rows` 很多但 `top1_rows` 很少，说明它常排第二/第三。")
    mo.ui.table(_topk_summary.to_pandas())
    return


@app.cell
def _(df_expanded, mo, pl):
    mo.md("## 扩容候选分组摘要")
    if df_expanded.is_empty():
        _expanded_summary = pl.DataFrame(
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
        _expanded_summary = (
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
    mo.ui.table(_expanded_summary.to_pandas())
    return


@app.cell
def _(TOP_K_MATCHES, df_topk_matches, mo):
    mo.md("## Top-k 候选明细")
    _topk_detail_cols = [
        "date",
        "code",
        "match_rank",
        "match_case",
        "match_case_date",
        "total_similarity",
        "feature_similarity",
        "curve_similarity",
        "fwd_mfe_risk_adj_10d",
        "textbook_b1_score",
    ]
    mo.md(f"保留每个候选的前 `{TOP_K_MATCHES}` 个匹配案例，方便看哪些案例经常排第二/第三。")
    mo.ui.table(df_topk_matches.select(_topk_detail_cols).head(150).to_pandas())
    return


@app.cell
def _(df_expanded, mo):
    mo.md("## 扩容候选明细")
    _expanded_detail_cols = [
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
    mo.ui.table(df_expanded.select(_expanded_detail_cols).head(100).to_pandas())
    return


@app.cell
def _(
    EXPANDED_CASES_PER_ARCHETYPE,
    EXPANDED_CASE_MIN_RISK_ADJ_10D,
    EXPANDED_CASE_MIN_TEXTBOOK_SCORE,
    EXPANDED_CASE_MIN_TOTAL_SIM,
    df_matches,
    pl,
):
    if df_matches.is_empty():
        df_expanded_cases_artifact = pl.DataFrame(
            schema={
                "code": pl.String,
                "date": pl.Date,
                "name": pl.String,
                "source_archetype": pl.String,
                "source_case_date": pl.Date,
                "total_similarity": pl.Float64,
                "feature_similarity": pl.Float64,
                "curve_similarity": pl.Float64,
                "textbook_b1_score": pl.Float64,
                "fwd_mfe_risk_adj_10d": pl.Float64,
                "artifact_rank": pl.Int64,
            }
        )
    else:
        df_expanded_cases_artifact = (
            df_matches
            .filter(pl.col("total_similarity") >= EXPANDED_CASE_MIN_TOTAL_SIM)
            .filter(pl.col("fwd_mfe_risk_adj_10d") >= EXPANDED_CASE_MIN_RISK_ADJ_10D)
            .filter(pl.col("textbook_b1_score") >= EXPANDED_CASE_MIN_TEXTBOOK_SCORE)
            .sort(
                [
                    "best_match_case",
                    "total_similarity",
                    "fwd_mfe_risk_adj_10d",
                    "textbook_b1_score",
                    "date",
                    "code",
                ],
                descending=[False, True, True, True, False, False],
            )
            # 分析层保留同股多日期，最终训练 artifact 只保留每个 archetype 下该股票最优的一条。
            .unique(subset=["best_match_case", "code"], keep="first", maintain_order=True)
            .sort(
                [
                    "best_match_case",
                    "total_similarity",
                    "fwd_mfe_risk_adj_10d",
                    "textbook_b1_score",
                    "date",
                    "code",
                ],
                descending=[False, True, True, True, False, False],
            )
            .with_columns(
                (
                    pl.col("code").cum_count().over("best_match_case")
                ).alias("artifact_rank")
            )
            .filter(pl.col("artifact_rank") <= EXPANDED_CASES_PER_ARCHETYPE)
            .with_columns(
                pl.concat_str(
                    [
                        pl.lit("扩容/"),
                        pl.col("best_match_case"),
                        pl.lit("/"),
                        pl.col("code"),
                        pl.lit("/"),
                        pl.col("date").dt.strftime("%Y-%m-%d"),
                    ]
                ).alias("name")
            )
            .select(
                [
                    "code",
                    "date",
                    "name",
                    pl.col("best_match_case").alias("source_archetype"),
                    pl.col("best_match_case_date").alias("source_case_date"),
                    "total_similarity",
                    "feature_similarity",
                    "curve_similarity",
                    "textbook_b1_score",
                    "fwd_mfe_risk_adj_10d",
                    "artifact_rank",
                ]
            )
        )

    if df_expanded_cases_artifact.is_empty():
        expanded_cases_summary = pl.DataFrame(
            schema={
                "source_archetype": pl.String,
                "rows": pl.Int64,
                "mean_total_similarity": pl.Float64,
                "mean_risk_adj_10d": pl.Float64,
            }
        )
    else:
        expanded_cases_summary = (
            df_expanded_cases_artifact.group_by("source_archetype")
            .agg(
                [
                    pl.len().alias("rows"),
                    pl.col("total_similarity").mean().round(4).alias("mean_total_similarity"),
                    pl.col("fwd_mfe_risk_adj_10d").mean().round(4).alias("mean_risk_adj_10d"),
                ]
            )
            .sort(["rows", "mean_total_similarity"], descending=[True, True])
        )
    return df_expanded_cases_artifact, expanded_cases_summary


@app.cell
def _(expanded_cases_summary, mo):
    mo.md("## EXPANDED_TEXTBOOK_CASES 摘要")
    mo.ui.table(expanded_cases_summary.to_pandas())
    return


@app.cell
def _(df_expanded_cases_artifact, mo):
    mo.md("## EXPANDED_TEXTBOOK_CASES 预览")
    mo.ui.table(df_expanded_cases_artifact.to_pandas())
    return


@app.cell
def _(EXPANDED_CASES_VERSION, df_expanded_cases_artifact, pl):
    expanded_cases_export_rows = (
        df_expanded_cases_artifact
        .select(["code", "date", "name"])
        .with_columns(pl.col("date").dt.strftime("%Y-%m-%d").alias("date"))
        .to_dicts()
    )
    expanded_cases_meta_rows = (
        df_expanded_cases_artifact
        .with_columns(pl.col("date").dt.strftime("%Y-%m-%d").alias("date"))
        .with_columns(pl.col("source_case_date").dt.strftime("%Y-%m-%d").alias("source_case_date"))
        .to_dicts()
    )
    artifact_py_text = "\n".join(
        [
            "from __future__ import annotations",
            "",
            f'EXPANDED_TEXTBOOK_CASES_VERSION = "{EXPANDED_CASES_VERSION}"',
            "",
            f"EXPANDED_TEXTBOOK_CASES = {expanded_cases_export_rows!r}",
            "",
            f"EXPANDED_TEXTBOOK_CASES_META = {expanded_cases_meta_rows!r}",
            "",
        ]
    )
    return artifact_py_text, expanded_cases_export_rows, expanded_cases_meta_rows


@app.cell
def _(artifact_py_text, mo):
    mo.md("## EXPANDED_TEXTBOOK_CASES Manifest")
    mo.md(f"```python\n{artifact_py_text}\n```")
    return


@app.cell
def _(
    EXPANDED_CASES_OUTPUT_PATH,
    WRITE_EXPANDED_CASES_MANIFEST,
    artifact_py_text,
    mo,
):
    if WRITE_EXPANDED_CASES_MANIFEST:
        EXPANDED_CASES_OUTPUT_PATH.write_text(artifact_py_text, encoding="utf-8")
        _write_manifest_output = mo.md(
            f"`EXPANDED_TEXTBOOK_CASES` 已写入 `{EXPANDED_CASES_OUTPUT_PATH.as_posix()}`。"
        )
    else:
        _write_manifest_output = mo.md(
            "\n".join(
                [
                    "默认不写盘。",
                    f"如需覆盖 manifest，请把 `WRITE_EXPANDED_CASES_MANIFEST = True`，输出路径为 `{EXPANDED_CASES_OUTPUT_PATH.as_posix()}`。",
                ]
            )
        )
    _write_manifest_output
    return


if __name__ == "__main__":
    app.run()
