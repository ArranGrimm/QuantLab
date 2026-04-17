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
def _(CASE_VECTOR_COLS, df_cases, np, pl):
    print("\n" + "=" * 72)
    print("  Step 2c. Textbook centroid 自洽性诊断 (H1 验证)")
    print("=" * 72)
    print("  目的: 11 个基础案例自身是否被 'median centroid' textbook_b1_score 正确识别。")
    print("  解读:")
    print("    - 若 11 个案例自身得分都 >= threshold 且方差小 → centroid 合理")
    print("    - 若 11 个案例得分发散、甚至有低于 threshold → 多 archetype 把 centroid 拍扁")
    print("    - 两两相似度均值低 → 案例之间天然不像, 求中位数无意义")

    _has_case_cols = (
        df_cases.height > 0
        and "textbook_b1_score" in df_cases.columns
        and "case_name" in df_cases.columns
    )
    if not _has_case_cols:
        print("  [SKIP] df_cases 为空或缺少 textbook 列, 跳过诊断。")
    else:
        _threshold = (
            float(df_cases["textbook_b1_threshold"][0])
            if "textbook_b1_threshold" in df_cases.columns
            else 0.65
        )

        _score_cols = [
            "textbook_b1_score",
            "textbook_trend_score",
            "textbook_kbar_score",
            "textbook_volume_score",
            "textbook_trigger_score",
        ]
        _score_cols_present = [c for c in _score_cols if c in df_cases.columns]
        _self_scores = (
            df_cases
            .select(["case_name", "code", "date"] + _score_cols_present + (
                ["is_textbook_b1"] if "is_textbook_b1" in df_cases.columns else []
            ))
            .with_columns([pl.col(c).round(4).alias(c) for c in _score_cols_present])
            .sort("textbook_b1_score", descending=True)
        )
        print(f"\n  [1] 11 个基础案例自身得分 (threshold = {_threshold:.4f}):")
        print(_self_scores)

        _self_series = df_cases["textbook_b1_score"].drop_nulls().drop_nans()
        _below = df_cases.filter(pl.col("textbook_b1_score") < _threshold).height
        _self_std = float(_self_series.std()) if _self_series.len() > 1 else 0.0
        print("\n  自洽性统计:")
        print(f"    case_count           : {df_cases.height}")
        print(f"    self_score_min       : {float(_self_series.min()):.4f}")
        print(f"    self_score_max       : {float(_self_series.max()):.4f}")
        print(f"    self_score_mean      : {float(_self_series.mean()):.4f}")
        print(f"    self_score_std       : {_self_std:.4f}")
        print(f"    cases_below_threshold: {_below} / {df_cases.height}")
        if _below > 0:
            print(
                "    → 'median centroid' 没法把所有 11 个案例都拉进 is_textbook_b1=True"
            )

        _feature_cols_present = [c for c in CASE_VECTOR_COLS if c in df_cases.columns]
        if not _feature_cols_present:
            print("\n  [SKIP] df_cases 缺少 textbook 输入特征列, 跳过 [2]~[4]。")
        else:
            _case_features = (
                df_cases
                .select(["case_name"] + _feature_cols_present)
                .with_columns([pl.col(c).round(4).alias(c) for c in _feature_cols_present])
                .sort("case_name")
            )
            print("\n  [2] 11 个案例的 14 维 textbook 特征向量 (raw value):")
            print(_case_features)

            _spread_rows = []
            _scales: dict[str, float] = {}
            for _col in _feature_cols_present:
                _series = df_cases[_col].drop_nulls().drop_nans()
                if _series.len() == 0:
                    continue
                _med = float(_series.median())
                _q1 = float(_series.quantile(0.25, interpolation="linear"))
                _q3 = float(_series.quantile(0.75, interpolation="linear"))
                _mn = float(_series.min())
                _mx = float(_series.max())
                _scale = max(
                    (_q3 - _q1) * 2.0,
                    (_mx - _mn),
                    abs(_med) * 0.35,
                    1e-4,
                )
                _max_dist = max(abs(_mx - _med), abs(_med - _mn))
                _scales[_col] = _scale
                _spread_rows.append({
                    "feature": _col,
                    "case_min": round(_mn, 4),
                    "case_median": round(_med, 4),
                    "case_max": round(_mx, 4),
                    "scale": round(_scale, 4),
                    "max_dist_to_med": round(_max_dist, 4),
                    "min_self_sim": round(max(0.0, 1.0 - _max_dist / _scale), 4),
                })
            print(
                "\n  [3] 每维特征的 case 内分布 (min_self_sim = case 离 centroid 最远那个的相似度):"
            )
            print(pl.DataFrame(_spread_rows))

            _case_records = (
                df_cases.select(["case_name"] + _feature_cols_present).to_dicts()
            )
            _names = [r["case_name"] for r in _case_records]
            _matrix_rows = []
            for _i, _ri in enumerate(_case_records):
                _row: dict[str, object] = {"case_name": _names[_i]}
                for _j, _rj in enumerate(_case_records):
                    _sims = []
                    for _col in _feature_cols_present:
                        _vi = _ri.get(_col)
                        _vj = _rj.get(_col)
                        if _vi is None or _vj is None:
                            continue
                        if isinstance(_vi, float) and np.isnan(_vi):
                            continue
                        if isinstance(_vj, float) and np.isnan(_vj):
                            continue
                        _scale = _scales.get(_col, 1.0)
                        _sims.append(
                            max(0.0, 1.0 - abs(float(_vi) - float(_vj)) / _scale)
                        )
                    _row[_names[_j]] = (
                        round(float(np.mean(_sims)), 3) if _sims else 0.0
                    )
                _matrix_rows.append(_row)
            print(
                "\n  [4] 11 个案例两两特征相似度矩阵 (1.0 = 完全一致, 0.0 = 完全不像):"
            )
            print(pl.DataFrame(_matrix_rows))

            _off_diag = []
            for _i, _row_i in enumerate(_matrix_rows):
                for _j, _name_j in enumerate(_names):
                    if _i == _j:
                        continue
                    _val = _row_i.get(_name_j)
                    if isinstance(_val, (int, float)):
                        _off_diag.append(float(_val))
            if _off_diag:
                _arr = np.array(_off_diag)
                print("\n  pair-wise 相似度 (off-diagonal) 统计:")
                print(f"    mean : {_arr.mean():.4f}")
                print(f"    min  : {_arr.min():.4f}")
                print(f"    max  : {_arr.max():.4f}")
                print(f"    p25  : {float(np.quantile(_arr, 0.25)):.4f}")
                print(f"    p75  : {float(np.quantile(_arr, 0.75)):.4f}")
                if _arr.mean() < 0.70:
                    print(
                        "    → 案例之间平均相似度偏低, 'median centroid' 可能不是任何单一 archetype 的好近似"
                    )
                else:
                    print(
                        "    → 案例之间平均相似度较高, 'median centroid' 可能仍合理, 反向富集要从 H2/H3/H4 找原因"
                    )
    return


@app.cell
def _(
    CASE_VECTOR_COLS,
    MAX_FWD_MAE_10D,
    MIN_FWD_MFE_10D,
    MIN_FWD_MFE_RISK_ADJ_10D,
    df_cases,
    df_seed,
    np,
    pl,
):
    print("\n" + "=" * 72)
    print("  Step 2d. v2 max-archetype 模拟 (H1 修复实验)")
    print("=" * 72)
    print("  公式:")
    print("    per_archetype_sim[x, k] = mean_f clip(1 - |x[f] - case_k[f]| / scale[f], 0, 1)")
    print("    textbook_b1_score_v2[x] = max_k per_archetype_sim[x, k]")
    print("    threshold_v2            = clip(quantile(LOO_self_scores, 0.20), 0.55, 0.80)")
    print("    is_textbook_b1_v2[x]    = textbook_b1_score_v2[x] >= threshold_v2")
    print("  scale: 与 v1 完全一致 (基于 11 个 case 的 IQR/range/|median|)")

    _feat_cols = [
        _c for _c in CASE_VECTOR_COLS
        if _c in df_seed.columns and _c in df_cases.columns
    ]

    if df_cases.height == 0 or not _feat_cols:
        print("\n  [SKIP] df_cases 或特征列缺失, 跳过 v2 模拟。")
        df_seed_v2 = df_seed
        threshold_v2 = float("nan")
    else:
        _scale_per_feat: dict[str, float] = {}
        for _f in _feat_cols:
            _s = df_cases[_f].drop_nulls().drop_nans()
            if _s.is_empty():
                continue
            _med = float(_s.median())
            _q1 = float(_s.quantile(0.25, interpolation="linear"))
            _q3 = float(_s.quantile(0.75, interpolation="linear"))
            _mn = float(_s.min())
            _mx = float(_s.max())
            _scale_per_feat[_f] = max(
                (_q3 - _q1) * 2.0,
                (_mx - _mn),
                abs(_med) * 0.35,
                1e-4,
            )
        _feat_cols = [_f for _f in _feat_cols if _f in _scale_per_feat]
        _scale_vec = np.array([_scale_per_feat[_f] for _f in _feat_cols], dtype=np.float64)

        _case_matrix = df_cases.select(_feat_cols).to_numpy().astype(np.float64)
        _case_names_list = df_cases["case_name"].to_list()
        _sample_matrix = df_seed.select(_feat_cols).to_numpy().astype(np.float64)

        _n_samples = _sample_matrix.shape[0]
        _n_cases = _case_matrix.shape[0]
        _per_arc_sim = np.zeros((_n_samples, _n_cases), dtype=np.float64)
        for _k in range(_n_cases):
            _diff = np.abs(_sample_matrix - _case_matrix[_k][None, :]) / _scale_vec[None, :]
            _per_feat = np.clip(1.0 - _diff, 0.0, 1.0)
            with np.errstate(invalid="ignore"):
                _arc_mean = np.nanmean(_per_feat, axis=1)
            _per_arc_sim[:, _k] = np.where(np.isnan(_arc_mean), 0.0, _arc_mean)

        _v2_score = _per_arc_sim.max(axis=1)
        _v2_best_idx = _per_arc_sim.argmax(axis=1)
        _v2_best_archetype = [_case_names_list[_i] for _i in _v2_best_idx]

        _case_sim_matrix = np.zeros((_n_cases, _n_cases), dtype=np.float64)
        for _i in range(_n_cases):
            for _j in range(_n_cases):
                _diff = np.abs(_case_matrix[_i] - _case_matrix[_j]) / _scale_vec
                _per_feat = np.clip(1.0 - _diff, 0.0, 1.0)
                _val = float(np.nanmean(_per_feat))
                _case_sim_matrix[_i, _j] = 0.0 if np.isnan(_val) else _val
        _loo_scores = []
        for _i in range(_n_cases):
            _others = np.concatenate([_case_sim_matrix[_i, :_i], _case_sim_matrix[_i, _i + 1 :]])
            if _others.size > 0:
                _loo_scores.append(float(_others.max()))
        threshold_v2 = (
            float(np.clip(np.quantile(_loo_scores, 0.20), 0.55, 0.80))
            if _loo_scores
            else 0.65
        )

        df_seed_v2 = df_seed.with_columns([
            pl.Series("textbook_b1_score_v2", _v2_score, dtype=pl.Float64),
            pl.Series("textbook_best_archetype_v2", _v2_best_archetype, dtype=pl.String),
            pl.Series("is_textbook_b1_v2", _v2_score >= threshold_v2, dtype=pl.Boolean),
        ])

        print(f"\n  threshold_v2 = {threshold_v2:.4f}  (LOO q20, clipped to [0.55, 0.80])")
        print(
            f"  v2 通过样本数 = {df_seed_v2.filter(pl.col('is_textbook_b1_v2')).height:,} "
            f"/ {df_seed_v2.height:,}"
        )

        def _safe_mean_v2(_df_g, _col):
            if _df_g.height == 0 or _col not in _df_g.columns:
                return float("nan")
            _s = _df_g[_col].drop_nulls().drop_nans()
            return float(_s.mean()) if _s.len() > 0 else float("nan")

        _df_b1_v1 = df_seed_v2.filter(pl.col("is_textbook_b1"))
        _df_non_b1_v1 = df_seed_v2.filter(~pl.col("is_textbook_b1"))
        _df_b1_v2 = df_seed_v2.filter(pl.col("is_textbook_b1_v2"))
        _df_non_b1_v2 = df_seed_v2.filter(~pl.col("is_textbook_b1_v2"))

        _seed_mean_ra = _safe_mean_v2(df_seed_v2, "fwd_mfe_risk_adj_10d")
        _seed_hit15 = (
            df_seed_v2.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
            / max(df_seed_v2.height, 1)
        )

        _table_a_rows = [
            {
                "version": "baseline",
                "group": "seed_mid 全体",
                "rows": df_seed_v2.height,
                "mean_risk_adj": round(_seed_mean_ra, 4),
                "hit_15pct": round(_seed_hit15, 4),
            },
            {
                "version": "v1",
                "group": "is_textbook_b1=True",
                "rows": _df_b1_v1.height,
                "mean_risk_adj": round(_safe_mean_v2(_df_b1_v1, "fwd_mfe_risk_adj_10d"), 4),
                "hit_15pct": round(
                    _df_b1_v1.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
                    / max(_df_b1_v1.height, 1),
                    4,
                ),
            },
            {
                "version": "v1",
                "group": "is_textbook_b1=False",
                "rows": _df_non_b1_v1.height,
                "mean_risk_adj": round(_safe_mean_v2(_df_non_b1_v1, "fwd_mfe_risk_adj_10d"), 4),
                "hit_15pct": round(
                    _df_non_b1_v1.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
                    / max(_df_non_b1_v1.height, 1),
                    4,
                ),
            },
            {
                "version": "v2",
                "group": "is_textbook_b1_v2=True",
                "rows": _df_b1_v2.height,
                "mean_risk_adj": round(_safe_mean_v2(_df_b1_v2, "fwd_mfe_risk_adj_10d"), 4),
                "hit_15pct": round(
                    _df_b1_v2.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
                    / max(_df_b1_v2.height, 1),
                    4,
                ),
            },
            {
                "version": "v2",
                "group": "is_textbook_b1_v2=False",
                "rows": _df_non_b1_v2.height,
                "mean_risk_adj": round(_safe_mean_v2(_df_non_b1_v2, "fwd_mfe_risk_adj_10d"), 4),
                "hit_15pct": round(
                    _df_non_b1_v2.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
                    / max(_df_non_b1_v2.height, 1),
                    4,
                ),
            },
        ]
        print("\n  [A] is_textbook_b1 平均前瞻表现 (v1 vs v2):")
        print(pl.DataFrame(_table_a_rows))
        _v2_true_ra = _safe_mean_v2(_df_b1_v2, "fwd_mfe_risk_adj_10d")
        if not np.isnan(_v2_true_ra):
            if _v2_true_ra >= _seed_mean_ra * 1.05:
                print(f"  → [A 通过] v2 textbook 组 ({_v2_true_ra:.4f}) > baseline*1.05 ({_seed_mean_ra * 1.05:.4f})")
            elif _v2_true_ra >= _seed_mean_ra:
                print(f"  → [A 弱通过] v2 textbook 组 ({_v2_true_ra:.4f}) ≥ baseline ({_seed_mean_ra:.4f}), 但提升有限")
            else:
                print(f"  → [A 未通过] v2 textbook 组 ({_v2_true_ra:.4f}) < baseline ({_seed_mean_ra:.4f})")

        _bins = [0.0, 0.30, 0.50, 0.65, 0.75, 0.85, 1.01]
        _bin_rows = []
        for _i in range(len(_bins) - 1):
            _lo, _hi = _bins[_i], _bins[_i + 1]
            _df_v1_bin = df_seed_v2.filter(
                (pl.col("textbook_b1_score") >= _lo) & (pl.col("textbook_b1_score") < _hi)
            )
            _df_v2_bin = df_seed_v2.filter(
                (pl.col("textbook_b1_score_v2") >= _lo) & (pl.col("textbook_b1_score_v2") < _hi)
            )
            _bin_rows.append({
                "score_range": f"[{_lo:.2f}, {_hi:.2f})",
                "v1_rows": _df_v1_bin.height,
                "v1_mean_risk_adj": round(_safe_mean_v2(_df_v1_bin, "fwd_mfe_risk_adj_10d"), 4),
                "v1_hit_15pct": round(
                    _df_v1_bin.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
                    / max(_df_v1_bin.height, 1),
                    4,
                ),
                "v2_rows": _df_v2_bin.height,
                "v2_mean_risk_adj": round(_safe_mean_v2(_df_v2_bin, "fwd_mfe_risk_adj_10d"), 4),
                "v2_hit_15pct": round(
                    _df_v2_bin.filter(pl.col("fwd_mfe_risk_adj_10d") >= 0.15).height
                    / max(_df_v2_bin.height, 1),
                    4,
                ),
            })
        print("\n  [B] 6 档分箱平均表现 (v1 vs v2 单调性):")
        print(pl.DataFrame(_bin_rows))
        _v2_bin_ra = [_r["v2_mean_risk_adj"] for _r in _bin_rows if _r["v2_rows"] > 0]
        if len(_v2_bin_ra) >= 4:
            _is_decreasing = all(
                _v2_bin_ra[_i] >= _v2_bin_ra[_i + 1] for _i in range(len(_v2_bin_ra) - 1)
            )
            _is_increasing = all(
                _v2_bin_ra[_i] <= _v2_bin_ra[_i + 1] for _i in range(len(_v2_bin_ra) - 1)
            )
            if _is_increasing:
                print("  → [B 通过] v2 6 档严格单调递增 (高分 = 高 risk_adj)")
            elif _is_decreasing:
                print("  → [B 未通过] v2 6 档仍单调递减 (高分 = 低 risk_adj), 与 v1 同病")
            elif _v2_bin_ra[-1] > _v2_bin_ra[0]:
                print("  → [B 弱通过] v2 不再严格递减, 最高档 > 最低档")
            else:
                print("  → [B 未通过] v2 高低档次序仍偏负向 (最高档 ≤ 最低档)")

        _seed_b1_ratio_v1 = _df_b1_v1.height / max(df_seed_v2.height, 1)
        _seed_b1_ratio_v2 = _df_b1_v2.height / max(df_seed_v2.height, 1)

        _enrich_specs = [
            ("seed_mid 全体 (baseline)", df_seed_v2),
        ]
        _med_ra = float(
            df_seed_v2["fwd_mfe_risk_adj_10d"].drop_nulls().drop_nans().median()
        ) if df_seed_v2.height else 0.0
        _enrich_specs.append((
            f"高于中位数 risk_adj (>{_med_ra:.4f})",
            df_seed_v2.filter(pl.col("fwd_mfe_risk_adj_10d") > _med_ra),
        ))
        _df_cands_v2 = (
            df_seed_v2
            .filter(~pl.col("is_textbook_case"))
            .filter(pl.col("fwd_mfe_10d") >= MIN_FWD_MFE_10D)
            .filter(pl.col("fwd_mae_10d") >= MAX_FWD_MAE_10D)
            .filter(pl.col("fwd_mfe_risk_adj_10d") >= MIN_FWD_MFE_RISK_ADJ_10D)
        )
        _enrich_specs.append(("结果强样本 (df_candidates)", _df_cands_v2))
        _p90_ra = float(
            df_seed_v2["fwd_mfe_risk_adj_10d"].drop_nulls().drop_nans().quantile(0.90, interpolation="linear")
        ) if df_seed_v2.height else 0.0
        _df_top10 = df_seed_v2.filter(pl.col("fwd_mfe_risk_adj_10d") >= _p90_ra)
        _enrich_specs.append((f"Top 10% risk_adj (>={_p90_ra:.4f})", _df_top10))

        _enrich_rows = []
        for _label, _df_g in _enrich_specs:
            _t = _df_g.height
            _b1_v1 = _df_g.filter(pl.col("is_textbook_b1")).height if _t else 0
            _b1_v2 = _df_g.filter(pl.col("is_textbook_b1_v2")).height if _t else 0
            _r1 = _b1_v1 / max(_t, 1)
            _r2 = _b1_v2 / max(_t, 1)
            _enrich_rows.append({
                "sample": _label,
                "total": _t,
                "v1_b1_ratio": f"{_r1:.2%}",
                "v1_enrichment": f"{_r1 / max(_seed_b1_ratio_v1, 1e-8):.2f}x",
                "v2_b1_ratio": f"{_r2:.2%}",
                "v2_enrichment": f"{_r2 / max(_seed_b1_ratio_v2, 1e-8):.2f}x",
                "mean_risk_adj": round(_safe_mean_v2(_df_g, "fwd_mfe_risk_adj_10d"), 4),
            })
        print("\n  [C] 强表现样本中 B1 形态占比 (v1 vs v2 enrichment):")
        print(pl.DataFrame(_enrich_rows))

        _top10_v2_enrich = (
            _df_top10.filter(pl.col("is_textbook_b1_v2")).height
            / max(_df_top10.height, 1)
            / max(_seed_b1_ratio_v2, 1e-8)
        ) if _df_top10.height > 0 else 0.0
        _cand_v2_enrich = (
            _df_cands_v2.filter(pl.col("is_textbook_b1_v2")).height
            / max(_df_cands_v2.height, 1)
            / max(_seed_b1_ratio_v2, 1e-8)
        ) if _df_cands_v2.height > 0 else 0.0
        if _top10_v2_enrich >= 1.20 and _cand_v2_enrich >= 1.20:
            print(f"  → [C 通过] v2 enrichment Top10%={_top10_v2_enrich:.2f}x, df_candidates={_cand_v2_enrich:.2f}x, 都 ≥ 1.20x")
        elif _top10_v2_enrich >= 1.0 and _cand_v2_enrich >= 1.0:
            print(f"  → [C 弱通过] v2 enrichment 转正 (Top10%={_top10_v2_enrich:.2f}x, candidates={_cand_v2_enrich:.2f}x), 但未达 1.20x 阈值")
        elif _top10_v2_enrich > (
            _df_top10.filter(pl.col("is_textbook_b1")).height
            / max(_df_top10.height, 1)
            / max(_seed_b1_ratio_v1, 1e-8)
        ) if _df_top10.height > 0 else False:
            print(f"  → [C 未通过] v2 enrichment 仍 < 1.0x (Top10%={_top10_v2_enrich:.2f}x), 但相对 v1 有改善")
        else:
            print(f"  → [C 未通过] v2 enrichment 仍 < 1.0x (Top10%={_top10_v2_enrich:.2f}x), 反向富集没解决")

        _arch_rows = []
        if _df_top10.height > 0 and "textbook_best_archetype_v2" in _df_top10.columns:
            _arch_top = (
                _df_top10
                .group_by("textbook_best_archetype_v2")
                .agg([
                    pl.len().alias("rows"),
                    pl.col("fwd_mfe_risk_adj_10d").mean().alias("mean_risk_adj"),
                    pl.col("textbook_b1_score_v2").mean().alias("mean_v2_score"),
                ])
                .with_columns([
                    (pl.col("rows") / _df_top10.height).alias("pct_of_top10"),
                ])
                .sort("rows", descending=True)
                .with_columns([
                    pl.col("mean_risk_adj").round(4),
                    pl.col("mean_v2_score").round(4),
                    pl.col("pct_of_top10").round(4),
                ])
            )
            print("\n  [D] Top 10% risk_adj 中样本最匹配的 archetype 分布:")
            print(_arch_top)

            _missing = [
                _name for _name in _case_names_list
                if _name not in _arch_top["textbook_best_archetype_v2"].to_list()
            ]
            if _missing:
                print(f"  注: 在 Top 10% 强样本中无人匹配的 archetype: {_missing}")
    return df_seed_v2, threshold_v2


@app.cell
def _(df_cases, df_seed, np, pl):
    print("\n" + "=" * 72)
    print("  Step 2e. 完美案例自身前瞻收益现实检验 (基础假设验证)")
    print("=" * 72)
    print("  目的: 检验 11 个 '完美案例' 在 fwd_mfe_10d 等 10 日前瞻标签下到底强不强。")
    print("  解读:")
    print("    - 若案例均值 >> seed_mid 均值 → 标签和案例对齐, 之前的反向富集是别的原因")
    print("    - 若案例均值 ≈ seed_mid 均值 → 标签和案例没对齐, 案例的 '完美' 不在 10 日尺度上")
    print("    - 若案例均值 < seed_mid 均值 → 标签和案例完全错配, 之前所有诊断都要重新审视")

    _fwd_cols_priority = [
        "fwd_ret_1d",
        "fwd_ret_2d",
        "fwd_ret_3d",
        "fwd_ret_5d",
        "fwd_ret_10d",
        "fwd_mfe_10d",
        "fwd_mae_10d",
        "fwd_mfe_risk_adj_10d",
    ]
    _fwd_cols = [_c for _c in _fwd_cols_priority if _c in df_cases.columns]

    if df_cases.height == 0 or not _fwd_cols:
        print("\n  [SKIP] df_cases 为空或缺少 fwd_* 列。")
    else:
        _per_case_view = (
            df_cases
            .select(["case_name", "code", "date"] + _fwd_cols)
            .with_columns([pl.col(_c).round(4).alias(_c) for _c in _fwd_cols])
            .sort("fwd_mfe_10d", descending=True)
            if "fwd_mfe_10d" in _fwd_cols
            else df_cases.select(["case_name", "code", "date"] + _fwd_cols)
        )
        print("\n  [1] 11 个案例自身的前瞻收益 (按 fwd_mfe_10d 倒序):")
        print(_per_case_view)

        def _safe_mean_2e(_df_g, _col):
            if _df_g.height == 0 or _col not in _df_g.columns:
                return float("nan")
            _s = _df_g[_col].drop_nulls().drop_nans()
            return float(_s.mean()) if _s.len() > 0 else float("nan")

        def _safe_quantile_2e(_df_g, _col, _q):
            if _df_g.height == 0 or _col not in _df_g.columns:
                return float("nan")
            _s = _df_g[_col].drop_nulls().drop_nans()
            return float(_s.quantile(_q, interpolation="linear")) if _s.len() > 0 else float("nan")

        _p90_ra = (
            _safe_quantile_2e(df_seed, "fwd_mfe_risk_adj_10d", 0.90)
            if "fwd_mfe_risk_adj_10d" in df_seed.columns else float("nan")
        )
        _df_top10_seed = (
            df_seed.filter(pl.col("fwd_mfe_risk_adj_10d") >= _p90_ra)
            if not np.isnan(_p90_ra) else df_seed.head(0)
        )

        _compare_groups = [
            ("seed_mid 全体 (baseline)", df_seed),
            ("11 个完美案例", df_cases),
            (f"seed_mid Top 10% risk_adj (>= {_p90_ra:.4f})", _df_top10_seed),
        ]
        _compare_rows = []
        for _label, _df_g in _compare_groups:
            _row = {"group": _label, "rows": _df_g.height}
            for _c in _fwd_cols:
                _row[_c] = round(_safe_mean_2e(_df_g, _c), 4)
            _compare_rows.append(_row)
        print("\n  [2] 案例 vs seed_mid 全体 vs seed_mid Top10% 均值对比:")
        print(pl.DataFrame(_compare_rows))

        _seed_mfe = _safe_mean_2e(df_seed, "fwd_mfe_10d") if "fwd_mfe_10d" in df_cases.columns else float("nan")
        _case_mfe = _safe_mean_2e(df_cases, "fwd_mfe_10d") if "fwd_mfe_10d" in df_cases.columns else float("nan")
        _top10_mfe = _safe_mean_2e(_df_top10_seed, "fwd_mfe_10d") if "fwd_mfe_10d" in df_cases.columns else float("nan")
        _seed_ra = _safe_mean_2e(df_seed, "fwd_mfe_risk_adj_10d") if "fwd_mfe_risk_adj_10d" in df_cases.columns else float("nan")
        _case_ra = _safe_mean_2e(df_cases, "fwd_mfe_risk_adj_10d") if "fwd_mfe_risk_adj_10d" in df_cases.columns else float("nan")

        if "fwd_mfe_10d" in df_cases.columns and not np.isnan(_seed_mfe) and not np.isnan(_case_mfe):
            print("\n  关键诊断:")
            print(f"    case_mean_fwd_mfe_10d        : {_case_mfe:.4f}")
            print(f"    seed_mid_mean_fwd_mfe_10d    : {_seed_mfe:.4f}")
            print(f"    top10_seed_mean_fwd_mfe_10d  : {_top10_mfe:.4f}")
            print(f"    case / seed ratio (mfe_10d)  : {_case_mfe / max(_seed_mfe, 1e-8):.2f}x")
            print(f"    case_mean_fwd_mfe_risk_adj   : {_case_ra:.4f}")
            print(f"    seed_mid_mean_fwd_mfe_risk_adj: {_seed_ra:.4f}")
            print(f"    case / seed ratio (risk_adj) : {_case_ra / max(_seed_ra, 1e-8):.2f}x")

            if _case_mfe >= _top10_mfe * 0.90:
                print("\n  → [对齐] 案例 fwd_mfe_10d 与 seed_mid Top10% 同档, 标签和案例对齐良好")
                print("    反向富集的根因不在 '案例-标签时间错配', 应转去查特征语义 (H2)")
            elif _case_mfe >= _seed_mfe * 1.5:
                print("\n  → [部分对齐] 案例 fwd_mfe_10d 显著优于 baseline, 但远不如 Top10%")
                print("    案例确实强, 但10日窗口可能没完全捕获其爆发力, R1 (拉长窗口) 值得一试")
            elif _case_mfe >= _seed_mfe:
                print("\n  → [弱对齐] 案例 fwd_mfe_10d 仅略高于 baseline")
                print("    案例的 '完美' 主要不在 10 日尺度上, R1/R3 (改标签) 是首选")
            else:
                print("\n  → [反对齐] 案例 fwd_mfe_10d 低于 baseline!")
                print("    标签和案例完全错配, 之前所有 '反向富集' 的诊断都需要重新审视")
                print("    案例可能根本不该用 fwd_mfe_10d 评估, 或者案例日期定错了 (应是启动日而非买点日)")

        _below_baseline = (
            df_cases.filter(pl.col("fwd_mfe_10d") < _seed_mfe).height
            if "fwd_mfe_10d" in df_cases.columns and not np.isnan(_seed_mfe) else 0
        )
        _below_top10 = (
            df_cases.filter(pl.col("fwd_mfe_10d") < _top10_mfe).height
            if "fwd_mfe_10d" in df_cases.columns and not np.isnan(_top10_mfe) else 0
        )
        if "fwd_mfe_10d" in df_cases.columns:
            print("\n  分布观察:")
            print(f"    案例 fwd_mfe_10d 低于 seed_mid baseline 均值的有: {_below_baseline} / {df_cases.height}")
            print(f"    案例 fwd_mfe_10d 低于 Top10% 均值的有        : {_below_top10} / {df_cases.height}")
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
