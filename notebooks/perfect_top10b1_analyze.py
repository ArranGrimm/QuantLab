import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    """
    Perfect Top10 B1 — 探索 / 诊断专用 Notebook
    -------------------------------------------------
    迁移自 b1_case_expansion_mining.py 的 Step 2b ~ 2e，
    并新增 multi-horizon 标签、国轩高科 horizon 敏感性、
    Step 2f Cohen's d 特征重要性。
    所有探索性、临时性挖掘工作都放在这里，
    b1_case_expansion_mining.py 仅保留 mining 主流程。
    """
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    from manifests import B1_BASE_TEXTBOOK_CASES
    from utils import build_b1_research_frame, get_st_blacklist_pl, load_daily_data_full
    from utils.b1_feature_pool import (
        B1_FEATURE_TO_GROUP,
        B1_TEXTBOOK_HARD_RULES_V3,
        B1_TEXTBOOK_SCORE_FEATURE_COLS,
        B1_TEXTBOOK_SCORE_FEATURE_COLS_V3,
    )

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
    INCLUDE_ROTATION_KBAR_FEATURES = True

    # ── textbook 评分版本 ─────────────────────────────────────────
    # "v1" = 14 个原 textbook 特征 (已被 H2 证伪, 反向富集 0.74~0.79x)
    # "v3" = 10 个共线去重后的高 |Cohen's d| 特征 + bad_k_count==0/trigger_recent_10==1 hard rule
    TEXTBOOK_SCORE_VERSION = "v3"

    # ── 多 horizon 标签 ───────────────────────────────────────────
    HORIZONS = [5, 10, 15, 20, 30, 40]
    ACTIVE_HORIZON = 20  # B1 形态占比 / Cohen's d 等单 horizon 分析的默认 horizon
    SPOTLIGHT_CODE = "sz.002074"  # 国轩高科 — horizon 敏感性专项

    if TEXTBOOK_SCORE_VERSION == "v3":
        CASE_VECTOR_COLS = list(B1_TEXTBOOK_SCORE_FEATURE_COLS_V3)
    else:
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
        ACTIVE_HORIZON,
        ACTIVE_SEED_COL,
        B1_BASE_TEXTBOOK_CASES,
        B1_FEATURE_TO_GROUP,
        B1_TEXTBOOK_HARD_RULES_V3,
        CASE_MV_MIN,
        CASE_VECTOR_COLS,
        DB_PATH,
        END_DATE,
        HORIZONS,
        INCLUDE_ROTATION_KBAR_FEATURES,
        LOOSE_PERIODS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        SEED_J_MAX,
        SPOTLIGHT_CODE,
        START_DATE,
        ST_SNAPSHOT_DATE,
        TEXTBOOK_SCORE_VERSION,
        USE_BULL_ONLY,
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
    print(f"q_full prepared, st_blacklist size = {len(st_blacklist):,}")
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
    TEXTBOOK_SCORE_VERSION,
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
        textbook_score_version=TEXTBOOK_SCORE_VERSION,
    )
    df_case_source = build_b1_research_frame(
        q_full,
        mv_min=CASE_MV_MIN,
        mv_max=MV_MAX,
        min_list_days=MIN_LIST_DAYS,
        seed_j_max=SEED_J_MAX,
        loose_periods=LOOSE_PERIODS,
        include_rotation_kbar_features=INCLUDE_ROTATION_KBAR_FEATURES,
        textbook_score_version=TEXTBOOK_SCORE_VERSION,
    )
    print(f"build_b1_research_frame done. textbook_score_version = {TEXTBOOK_SCORE_VERSION!r}")
    seed_filter = pl.col(ACTIVE_SEED_COL)
    if USE_BULL_ONLY:
        seed_filter = seed_filter & pl.col("is_manual_bull")
    df_seed_base = df_all.filter(seed_filter)

    case_df_lookup = (
        pl.DataFrame(B1_BASE_TEXTBOOK_CASES)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .rename({"name": "case_name"})
    )
    df_cases_base = (
        df_case_source.join(case_df_lookup, on=["code", "date"], how="inner")
        .sort(["date", "code"])
    )

    print("=" * 72)
    print("  Step 0. 数据准备")
    print("=" * 72)
    print(
        pl.DataFrame(
            [
                {"item": "rows_all", "value": f"{df_all.height:,}"},
                {"item": "rows_seed", "value": f"{df_seed_base.height:,}"},
                {
                    "item": "dates_seed",
                    "value": str(df_seed_base["date"].n_unique()) if df_seed_base.height else "0",
                },
                {"item": "base_cases_found", "value": str(df_cases_base.height)},
            ]
        )
    )
    return df_all, df_cases_base, df_seed_base


@app.cell
def _(HORIZONS, df_all, df_cases_base, df_seed_base, pl, q_full):
    """
    Step 0b. 多 horizon 前瞻标签构建
    -------------------------------------------------
    build_b1_research_frame 内部硬编码了 10d 标签 (fwd_mfe_10d 等)；
    本 cell 在 notebook 层面基于 q_full 额外构建 5/15/20/30/40d 标签，
    然后 join 到 df_seed/df_cases，便于做 horizon 敏感性分析。
    """
    needed_horizons = sorted(set(HORIZONS) - {10})  # 10d 已在 df_all 中
    if not needed_horizons:
        df_seed = df_seed_base
        df_cases = df_cases_base
        extra_horizon_cols = []
    else:
        max_h = max(needed_horizons)
        future_high = [
            pl.col("high_adj").shift(-step).over("code").alias(f"_fh_{step}")
            for step in range(1, max_h + 1)
        ]
        future_low = [
            pl.col("low_adj").shift(-step).over("code").alias(f"_fl_{step}")
            for step in range(1, max_h + 1)
        ]
        df_labels_extra = (
            q_full.select(["code", "date", "close_adj", "high_adj", "low_adj"])
            .sort(["code", "date"])
            .with_columns(future_high + future_low)
        )
        extra_exprs: list[pl.Expr] = []
        for h in needed_horizons:
            high_names = [f"_fh_{step}" for step in range(1, h + 1)]
            low_names = [f"_fl_{step}" for step in range(1, h + 1)]
            extra_exprs.extend(
                [
                    (pl.max_horizontal(high_names) / pl.col("close_adj") - 1).alias(f"fwd_mfe_{h}d"),
                    (pl.min_horizontal(low_names) / pl.col("close_adj") - 1).alias(f"fwd_mae_{h}d"),
                    (pl.col("close_adj").shift(-h).over("code") / pl.col("close_adj") - 1).alias(
                        f"fwd_ret_{h}d"
                    ),
                ]
            )
        df_labels_extra = df_labels_extra.with_columns(extra_exprs)
        risk_adj_exprs = [
            (pl.col(f"fwd_mfe_{h}d") / (1 + pl.col(f"fwd_mae_{h}d").abs())).alias(
                f"fwd_mfe_risk_adj_{h}d"
            )
            for h in needed_horizons
        ]
        df_labels_extra = df_labels_extra.with_columns(risk_adj_exprs).select(
            ["code", "date"]
            + [f"fwd_mfe_{h}d" for h in needed_horizons]
            + [f"fwd_mae_{h}d" for h in needed_horizons]
            + [f"fwd_ret_{h}d" for h in needed_horizons]
            + [f"fwd_mfe_risk_adj_{h}d" for h in needed_horizons]
        ).collect()

        df_seed = df_seed_base.join(df_labels_extra, on=["code", "date"], how="left")
        df_cases = df_cases_base.join(df_labels_extra, on=["code", "date"], how="left")
        extra_horizon_cols = [
            c for c in df_labels_extra.columns if c not in ("code", "date")
        ]

    # 候选池仍按 10d 阈值过滤，保持与 mining 主流程口径一致。
    print("=" * 72)
    print("  Step 0b. multi-horizon 前瞻标签")
    print("=" * 72)
    print(f"  extra horizon cols added: {len(extra_horizon_cols)}  → {extra_horizon_cols}")
    print(f"  rows_seed (after join):  {df_seed.height:,}")
    print(f"  rows_cases (after join): {df_cases.height}")
    print("  df_all 仍保留 build_b1_research_frame 原 10d 标签, 不重复扩展。")
    _ = df_all  # 占位防 unused
    return df_cases, df_seed


@app.cell
def _(HORIZONS, SPOTLIGHT_CODE, df_cases, df_seed, np, pl):
    def _run():
        """
        Step A. 完美案例自身前瞻收益现实检验 (multi-horizon)
        -------------------------------------------------
        迁移自 b1_case_expansion_mining.py Step 2e，并扩展为多 horizon。
        重点验证: 10d 太短是否解释了 '国轩高科' 等案例的反例。
        """
        print("\n" + "=" * 72)
        print("  Step A. 完美案例 vs seed_mid baseline — 各 horizon 平均前瞻表现")
        print("=" * 72)

        metric_kinds = ["fwd_mfe", "fwd_mae", "fwd_ret", "fwd_mfe_risk_adj"]
        available = []
        for h in HORIZONS:
            for m in metric_kinds:
                col = f"{m}_{h}d"
                if col in df_seed.columns and col in df_cases.columns:
                    available.append((h, m, col))

        if df_cases.height == 0 or not available:
            print("  [SKIP] df_cases 为空或无可用 horizon 标签")
        else:
            def _safe_mean(_df_g, _col):
                if _df_g.height == 0 or _col not in _df_g.columns:
                    return float("nan")
                _s = _df_g[_col].drop_nulls().drop_nans()
                return float(_s.mean()) if _s.len() > 0 else float("nan")

            rows = []
            for h in HORIZONS:
                row = {"horizon": f"{h}d"}
                for m in metric_kinds:
                    col = f"{m}_{h}d"
                    if col in df_seed.columns:
                        row[f"seed_{m}"] = round(_safe_mean(df_seed, col), 4)
                    if col in df_cases.columns:
                        row[f"case_{m}"] = round(_safe_mean(df_cases, col), 4)
                mfe_seed = row.get("seed_fwd_mfe", float("nan"))
                mfe_case = row.get("case_fwd_mfe", float("nan"))
                if not np.isnan(mfe_seed) and not np.isnan(mfe_case) and mfe_seed > 0:
                    row["case/seed_ratio_mfe"] = round(mfe_case / mfe_seed, 2)
                rows.append(row)

            print("\n  [A1] 各 horizon 案例均值 vs seed_mid 均值 (mfe / mae / ret / mfe_risk_adj):")
            print(pl.DataFrame(rows))

            print(
                "\n  解读:\n"
                "    - case/seed_ratio_mfe 在哪个 horizon 最大 → 案例的 '完美' 主要在该窗口体现。\n"
                "    - 若 ratio 在 20d 才显著放大 → 10d 标签确实截断了案例的爆发力, R1 (拉长窗口) 值得做。\n"
                "    - 若 ratio 在所有 horizon 都接近 1 → 案例并不普遍优于 seed_mid, 反向富集与窗口无关。"
            )

            per_case_cols = (
                ["case_name", "code", "date"]
                + [f"fwd_mfe_{h}d" for h in HORIZONS if f"fwd_mfe_{h}d" in df_cases.columns]
                + [f"fwd_mae_{h}d" for h in HORIZONS if f"fwd_mae_{h}d" in df_cases.columns]
            )
            per_case = (
                df_cases.select(per_case_cols)
                .with_columns(
                    [pl.col(c).round(4) for c in per_case_cols if c not in ("case_name", "code", "date")]
                )
                .sort("fwd_mfe_10d", descending=True)
            )
            print("\n  [A2] 11 个案例自身在各 horizon 的 fwd_mfe / fwd_mae 明细:")
            print(per_case)

            if SPOTLIGHT_CODE:
                spotlight = df_cases.filter(pl.col("code") == SPOTLIGHT_CODE)
                if spotlight.height > 0:
                    spot_row = spotlight.to_dicts()[0]
                    print(
                        f"\n  [A3] Spotlight {spot_row.get('case_name', SPOTLIGHT_CODE)} "
                        f"({SPOTLIGHT_CODE}, {spot_row['date']}) horizon 演化:"
                    )
                    spot_rows = []
                    for h in HORIZONS:
                        spot_rows.append(
                            {
                                "horizon": f"{h}d",
                                "fwd_mfe": round(spot_row.get(f"fwd_mfe_{h}d") or float("nan"), 4),
                                "fwd_mae": round(spot_row.get(f"fwd_mae_{h}d") or float("nan"), 4),
                                "fwd_ret": round(spot_row.get(f"fwd_ret_{h}d") or float("nan"), 4),
                                "fwd_mfe_risk_adj": round(
                                    spot_row.get(f"fwd_mfe_risk_adj_{h}d") or float("nan"), 4
                                ),
                            }
                        )
                    print(pl.DataFrame(spot_rows))
                    print(
                        "\n    判读: 若 fwd_mfe 在 20d/30d 显著抬升, 则 '国轩高科' 不是坏案例, "
                        "而是 10d 窗口看不到趋势型案例的回报。"
                    )
                else:
                    print(f"\n  [A3] Spotlight code {SPOTLIGHT_CODE} 不在 df_cases 内。")

    _run()
    return


@app.cell
def _(HORIZONS, df_cases, df_seed, np, pl):
    def _run():
        """
        Step A4. 多 horizon 下案例 vs Top10% seed 的 '是否同档' 判读
        -------------------------------------------------
        与 [A1] 互补: 比对案例均值与各 horizon 自身 Top10% seed 的距离,
        更能判断案例是否仍属强样本; 也用于 '换 20d 后国轩高科是否回到强样本档' 的回答。
        """
        print("\n" + "=" * 72)
        print("  Step A4. 案例 vs seed_mid Top10% — 各 horizon 对照")
        print("=" * 72)

        if df_cases.height == 0:
            print("  [SKIP] df_cases 为空")
        else:
            def _safe_mean(_df_g, _col):
                if _df_g.height == 0 or _col not in _df_g.columns:
                    return float("nan")
                _s = _df_g[_col].drop_nulls().drop_nans()
                return float(_s.mean()) if _s.len() > 0 else float("nan")

            def _safe_q(_df_g, _col, _q):
                if _df_g.height == 0 or _col not in _df_g.columns:
                    return float("nan")
                _s = _df_g[_col].drop_nulls().drop_nans()
                return float(_s.quantile(_q, interpolation="linear")) if _s.len() > 0 else float("nan")

            rows = []
            per_case_below_top10 = []
            for h in HORIZONS:
                ra_col = f"fwd_mfe_risk_adj_{h}d"
                mfe_col = f"fwd_mfe_{h}d"
                if ra_col not in df_seed.columns or mfe_col not in df_cases.columns:
                    continue
                p90 = _safe_q(df_seed, ra_col, 0.90)
                df_top10 = (
                    df_seed.filter(pl.col(ra_col) >= p90) if not np.isnan(p90) else df_seed.head(0)
                )
                top10_mfe = _safe_mean(df_top10, mfe_col)
                case_mfe = _safe_mean(df_cases, mfe_col)
                seed_mfe = _safe_mean(df_seed, mfe_col)
                below_top10 = (
                    df_cases.filter(pl.col(mfe_col) < top10_mfe).height
                    if not np.isnan(top10_mfe)
                    else 0
                )
                below_baseline = (
                    df_cases.filter(pl.col(mfe_col) < seed_mfe).height
                    if not np.isnan(seed_mfe)
                    else 0
                )
                verdict = ""
                if not np.isnan(case_mfe) and not np.isnan(top10_mfe) and not np.isnan(seed_mfe):
                    if case_mfe >= top10_mfe * 0.90:
                        verdict = "对齐(≥ Top10*0.90)"
                    elif case_mfe >= seed_mfe * 1.5:
                        verdict = "部分对齐(≥ baseline*1.5)"
                    elif case_mfe >= seed_mfe:
                        verdict = "弱对齐(略 > baseline)"
                    else:
                        verdict = "反对齐(< baseline)"
                rows.append(
                    {
                        "horizon": f"{h}d",
                        "seed_p90_risk_adj": round(p90, 4) if not np.isnan(p90) else None,
                        "top10_mean_mfe": round(top10_mfe, 4) if not np.isnan(top10_mfe) else None,
                        "case_mean_mfe": round(case_mfe, 4) if not np.isnan(case_mfe) else None,
                        "seed_mean_mfe": round(seed_mfe, 4) if not np.isnan(seed_mfe) else None,
                        "case_below_baseline": f"{below_baseline}/{df_cases.height}",
                        "case_below_top10": f"{below_top10}/{df_cases.height}",
                        "verdict": verdict,
                    }
                )
                for case_row in df_cases.to_dicts():
                    cv = case_row.get(mfe_col)
                    if cv is None:
                        continue
                    if not np.isnan(top10_mfe) and float(cv) < top10_mfe:
                        per_case_below_top10.append(
                            {
                                "horizon": f"{h}d",
                                "case_name": case_row.get("case_name"),
                                "code": case_row.get("code"),
                                "fwd_mfe": round(float(cv), 4),
                                "top10_mfe": round(top10_mfe, 4),
                            }
                        )

            print("\n  [A4] 各 horizon 案例均值 vs Top10% seed 均值:")
            print(pl.DataFrame(rows))
            if per_case_below_top10:
                print("\n  [A5] 各 horizon 仍跑不过 Top10% seed 的案例明细 (是否 horizon 拉长后消失?):")
                print(
                    pl.DataFrame(per_case_below_top10).sort(["case_name", "horizon"])
                )

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_cases, df_seed, np, pl):
    def _run():
        """
        Step B. B1 形态占比与平均表现概览 (按 ACTIVE_HORIZON)
        -------------------------------------------------
        迁移自 b1_case_expansion_mining.py Step 2b，
        支持切换 ACTIVE_HORIZON 看不同窗口下 '反向富集' 是否仍成立。
        """
        h = ACTIVE_HORIZON
        mfe_col = f"fwd_mfe_{h}d"
        mae_col = f"fwd_mae_{h}d"
        ra_col = f"fwd_mfe_risk_adj_{h}d"

        has_tb = (
            "is_textbook_b1" in df_seed.columns
            and "textbook_b1_score" in df_seed.columns
            and ra_col in df_seed.columns
            and mfe_col in df_seed.columns
        )

        print("\n" + "=" * 72)
        print(f"  Step B. B1 形态占比与平均前瞻表现概览  (horizon = {h}d)")
        print("=" * 72)

        if not has_tb:
            print("  [SKIP] df_seed 缺少 textbook 列或所选 horizon 列")
        else:
            def _safe_mean(_df, _col):
                if _df.height == 0 or _col not in _df.columns:
                    return float("nan")
                _s = _df[_col].drop_nulls().drop_nans()
                return float(_s.mean()) if _s.len() > 0 else float("nan")

            df_b1 = df_seed.filter(pl.col("is_textbook_b1"))
            df_non_b1 = df_seed.filter(~pl.col("is_textbook_b1"))

            perf_groups = [
                ("seed_mid 全体", df_seed),
                ("形态像B1 (is_textbook_b1)", df_b1),
                ("非B1形态", df_non_b1),
            ]
            perf_rows = []
            for label, df_g in perf_groups:
                perf_rows.append(
                    {
                        "group": label,
                        "rows": df_g.height,
                        f"mean_mfe_{h}d": round(_safe_mean(df_g, mfe_col), 4),
                        f"mean_mae_{h}d": round(_safe_mean(df_g, mae_col), 4),
                        f"mean_risk_adj_{h}d": round(_safe_mean(df_g, ra_col), 4),
                        "hit_15pct": round(
                            df_g.filter(pl.col(ra_col) >= 0.15).height / max(df_g.height, 1), 4
                        ),
                    }
                )
            print(f"\n  [B1] seed_mid 全体 vs B1 形态 — 平均前瞻表现 ({h}d):")
            print(pl.DataFrame(perf_rows))

            score_bins = [0.0, 0.3, 0.5, 0.65, 0.75, 0.85, 1.01]
            bin_rows = []
            for i in range(len(score_bins) - 1):
                lo, hi = score_bins[i], score_bins[i + 1]
                df_bin = df_seed.filter(
                    (pl.col("textbook_b1_score") >= lo) & (pl.col("textbook_b1_score") < hi)
                )
                if df_bin.height == 0:
                    continue
                bin_rows.append(
                    {
                        "score_range": f"[{lo:.2f}, {hi:.2f})",
                        "rows": df_bin.height,
                        f"mean_mfe_{h}d": round(_safe_mean(df_bin, mfe_col), 4),
                        f"mean_risk_adj_{h}d": round(_safe_mean(df_bin, ra_col), 4),
                    }
                )
            if bin_rows:
                print(f"\n  [B2] textbook_b1_score 分段平均表现 ({h}d):")
                print(pl.DataFrame(bin_rows))

            seed_b1_ratio = df_b1.height / max(df_seed.height, 1)
            med_ra = _safe_mean(df_seed, ra_col) if False else float(
                df_seed[ra_col].drop_nulls().drop_nans().median()
            ) if df_seed.height else 0.0
            df_above_med = df_seed.filter(pl.col(ra_col) > med_ra)
            p90_ra = float(
                df_seed[ra_col].drop_nulls().drop_nans().quantile(0.90, interpolation="linear")
            ) if df_seed.height else 0.0
            df_top_dec = df_seed.filter(pl.col(ra_col) >= p90_ra)

            enrich_rows = [
                {
                    "sample": "seed_mid 全体 (baseline)",
                    "total": df_seed.height,
                    "b1_like": df_b1.height,
                    "b1_ratio": f"{seed_b1_ratio:.2%}",
                    "enrichment": "1.00x",
                    f"mean_risk_adj_{h}d": round(_safe_mean(df_seed, ra_col), 4),
                },
                {
                    "sample": f"高于中位数 risk_adj_{h}d (>{med_ra:.4f})",
                    "total": df_above_med.height,
                    "b1_like": df_above_med.filter(pl.col("is_textbook_b1")).height,
                    "b1_ratio": f"{df_above_med.filter(pl.col('is_textbook_b1')).height / max(df_above_med.height, 1):.2%}",
                    "enrichment": (
                        f"{(df_above_med.filter(pl.col('is_textbook_b1')).height / max(df_above_med.height, 1)) / max(seed_b1_ratio, 1e-8):.2f}x"
                    ),
                    f"mean_risk_adj_{h}d": round(_safe_mean(df_above_med, ra_col), 4),
                },
                {
                    "sample": f"Top 10% risk_adj_{h}d (>={p90_ra:.4f})",
                    "total": df_top_dec.height,
                    "b1_like": df_top_dec.filter(pl.col("is_textbook_b1")).height,
                    "b1_ratio": f"{df_top_dec.filter(pl.col('is_textbook_b1')).height / max(df_top_dec.height, 1):.2%}",
                    "enrichment": (
                        f"{(df_top_dec.filter(pl.col('is_textbook_b1')).height / max(df_top_dec.height, 1)) / max(seed_b1_ratio, 1e-8):.2f}x"
                    ),
                    f"mean_risk_adj_{h}d": round(_safe_mean(df_top_dec, ra_col), 4),
                },
            ]
            print(f"\n  [B3] 强表现样本中 B1 形态占比 (enrichment, horizon = {h}d):")
            print(pl.DataFrame(enrich_rows))
            print(f"\n  enrichment = 该组 B1 占比 / seed_mid 全体 B1 占比 ({seed_b1_ratio:.2%})")
            _ = (np, df_cases)

    _run()
    return


@app.cell
def _(CASE_VECTOR_COLS, df_cases, np, pl):
    def _run():
        """
        Step C. Textbook centroid 自洽性诊断 (H1 验证)
        -------------------------------------------------
        迁移自 b1_case_expansion_mining.py Step 2c。
        诊断 11 个 base case 自身得分是否被 'median centroid' 拉到 is_textbook_b1=True，
        以及 case 之间两两相似度有多分散。
        """
        print("\n" + "=" * 72)
        print("  Step C. Textbook centroid 自洽性诊断 (H1 验证)")
        print("=" * 72)

        has_case_cols = (
            df_cases.height > 0
            and "textbook_b1_score" in df_cases.columns
            and "case_name" in df_cases.columns
        )
        if not has_case_cols:
            print("  [SKIP] df_cases 为空或缺少 textbook 列")
        else:
            threshold = (
                float(df_cases["textbook_b1_threshold"][0])
                if "textbook_b1_threshold" in df_cases.columns
                else 0.65
            )
            score_cols = [
                "textbook_b1_score",
                "textbook_trend_score",
                "textbook_kbar_score",
                "textbook_volume_score",
                "textbook_trigger_score",
            ]
            score_cols_present = [c for c in score_cols if c in df_cases.columns]
            self_scores = (
                df_cases.select(
                    ["case_name", "code", "date"]
                    + score_cols_present
                    + (["is_textbook_b1"] if "is_textbook_b1" in df_cases.columns else [])
                )
                .with_columns([pl.col(c).round(4).alias(c) for c in score_cols_present])
                .sort("textbook_b1_score", descending=True)
            )
            print(f"\n  [C1] 11 个基础案例自身得分 (threshold = {threshold:.4f}):")
            print(self_scores)

            ss = df_cases["textbook_b1_score"].drop_nulls().drop_nans()
            below = df_cases.filter(pl.col("textbook_b1_score") < threshold).height
            std_v = float(ss.std()) if ss.len() > 1 else 0.0
            print("\n  自洽性统计:")
            print(f"    case_count           : {df_cases.height}")
            print(f"    self_score_min       : {float(ss.min()):.4f}")
            print(f"    self_score_max       : {float(ss.max()):.4f}")
            print(f"    self_score_mean      : {float(ss.mean()):.4f}")
            print(f"    self_score_std       : {std_v:.4f}")
            print(f"    cases_below_threshold: {below} / {df_cases.height}")

            feat_cols = [c for c in CASE_VECTOR_COLS if c in df_cases.columns]
            if feat_cols:
                scales: dict[str, float] = {}
                spread_rows = []
                for col in feat_cols:
                    s = df_cases[col].drop_nulls().drop_nans()
                    if s.len() == 0:
                        continue
                    med = float(s.median())
                    q1 = float(s.quantile(0.25, interpolation="linear"))
                    q3 = float(s.quantile(0.75, interpolation="linear"))
                    mn = float(s.min())
                    mx = float(s.max())
                    scale = max((q3 - q1) * 2.0, (mx - mn), abs(med) * 0.35, 1e-4)
                    max_dist = max(abs(mx - med), abs(med - mn))
                    scales[col] = scale
                    spread_rows.append(
                        {
                            "feature": col,
                            "case_min": round(mn, 4),
                            "case_median": round(med, 4),
                            "case_max": round(mx, 4),
                            "scale": round(scale, 4),
                            "max_dist_to_med": round(max_dist, 4),
                            "min_self_sim": round(max(0.0, 1.0 - max_dist / scale), 4),
                        }
                    )
                print("\n  [C2] 每维 textbook 特征的 case 内分布 (min_self_sim 越低越发散):")
                print(pl.DataFrame(spread_rows))

                recs = df_cases.select(["case_name"] + feat_cols).to_dicts()
                names = [r["case_name"] for r in recs]
                mat_rows = []
                for i, ri in enumerate(recs):
                    row: dict[str, object] = {"case_name": names[i]}
                    for j, rj in enumerate(recs):
                        sims = []
                        for col in feat_cols:
                            vi = ri.get(col)
                            vj = rj.get(col)
                            if vi is None or vj is None:
                                continue
                            if isinstance(vi, float) and np.isnan(vi):
                                continue
                            if isinstance(vj, float) and np.isnan(vj):
                                continue
                            sims.append(
                                max(0.0, 1.0 - abs(float(vi) - float(vj)) / scales.get(col, 1.0))
                            )
                        row[names[j]] = round(float(np.mean(sims)), 3) if sims else 0.0
                    mat_rows.append(row)
                print("\n  [C3] 11 个案例两两特征相似度矩阵 (1.0=完全一致, 0.0=完全不像):")
                print(pl.DataFrame(mat_rows))

                off_diag = []
                for i, row_i in enumerate(mat_rows):
                    for j, name_j in enumerate(names):
                        if i == j:
                            continue
                        val = row_i.get(name_j)
                        if isinstance(val, (int, float)):
                            off_diag.append(float(val))
                if off_diag:
                    arr = np.array(off_diag)
                    print("\n  pair-wise 相似度 (off-diagonal) 统计:")
                    print(f"    mean : {arr.mean():.4f}")
                    print(f"    min  : {arr.min():.4f}")
                    print(f"    max  : {arr.max():.4f}")
                    print(f"    p25  : {float(np.quantile(arr, 0.25)):.4f}")
                    print(f"    p75  : {float(np.quantile(arr, 0.75)):.4f}")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, CASE_VECTOR_COLS, df_cases, df_seed, np, pl):
    def _run():
        """
        Step D. v2 max-archetype 模拟 (H1 修复实验)
        -------------------------------------------------
        迁移自 b1_case_expansion_mining.py Step 2d。
        用 max_k 替代 median centroid，看是否能解决反向富集；
        horizon 跟随 ACTIVE_HORIZON。
        """
        h = ACTIVE_HORIZON
        ra_col = f"fwd_mfe_risk_adj_{h}d"

        print("\n" + "=" * 72)
        print(f"  Step D. v2 max-archetype 模拟 (H1 修复实验, horizon = {h}d)")
        print("=" * 72)
        print("  公式: textbook_b1_score_v2[x] = max_k mean_f clip(1 - |x[f]-case_k[f]|/scale[f], 0, 1)")

        feat_cols = [
            c for c in CASE_VECTOR_COLS if c in df_seed.columns and c in df_cases.columns
        ]
        if df_cases.height == 0 or not feat_cols or ra_col not in df_seed.columns:
            print(f"  [SKIP] df_cases / 特征列 / {ra_col} 缺失")
        else:
            scale_per_feat: dict[str, float] = {}
            for f in feat_cols:
                s = df_cases[f].drop_nulls().drop_nans()
                if s.is_empty():
                    continue
                med = float(s.median())
                q1 = float(s.quantile(0.25, interpolation="linear"))
                q3 = float(s.quantile(0.75, interpolation="linear"))
                mn = float(s.min())
                mx = float(s.max())
                scale_per_feat[f] = max((q3 - q1) * 2.0, (mx - mn), abs(med) * 0.35, 1e-4)
            feat_cols = [f for f in feat_cols if f in scale_per_feat]
            scale_vec = np.array([scale_per_feat[f] for f in feat_cols], dtype=np.float64)

            case_matrix = df_cases.select(feat_cols).to_numpy().astype(np.float64)
            case_names_list = df_cases["case_name"].to_list()
            sample_matrix = df_seed.select(feat_cols).to_numpy().astype(np.float64)

            n_samples = sample_matrix.shape[0]
            n_cases = case_matrix.shape[0]
            per_arc_sim = np.zeros((n_samples, n_cases), dtype=np.float64)
            for k in range(n_cases):
                diff = np.abs(sample_matrix - case_matrix[k][None, :]) / scale_vec[None, :]
                per_feat = np.clip(1.0 - diff, 0.0, 1.0)
                with np.errstate(invalid="ignore"):
                    arc_mean = np.nanmean(per_feat, axis=1)
                per_arc_sim[:, k] = np.where(np.isnan(arc_mean), 0.0, arc_mean)

            v2_score = per_arc_sim.max(axis=1)
            v2_best_idx = per_arc_sim.argmax(axis=1)
            v2_best_archetype = [case_names_list[i] for i in v2_best_idx]

            case_sim_matrix = np.zeros((n_cases, n_cases), dtype=np.float64)
            for i in range(n_cases):
                for j in range(n_cases):
                    diff = np.abs(case_matrix[i] - case_matrix[j]) / scale_vec
                    per_feat = np.clip(1.0 - diff, 0.0, 1.0)
                    val = float(np.nanmean(per_feat))
                    case_sim_matrix[i, j] = 0.0 if np.isnan(val) else val
            loo_scores = []
            for i in range(n_cases):
                others = np.concatenate([case_sim_matrix[i, :i], case_sim_matrix[i, i + 1 :]])
                if others.size > 0:
                    loo_scores.append(float(others.max()))
            threshold_v2 = (
                float(np.clip(np.quantile(loo_scores, 0.20), 0.55, 0.80)) if loo_scores else 0.65
            )

            df_seed_v2 = df_seed.with_columns(
                [
                    pl.Series("textbook_b1_score_v2", v2_score, dtype=pl.Float64),
                    pl.Series("textbook_best_archetype_v2", v2_best_archetype, dtype=pl.String),
                    pl.Series("is_textbook_b1_v2", v2_score >= threshold_v2, dtype=pl.Boolean),
                ]
            )

            print(f"\n  threshold_v2 = {threshold_v2:.4f}  (LOO q20, clipped to [0.55, 0.80])")
            print(
                f"  v2 通过样本数 = {df_seed_v2.filter(pl.col('is_textbook_b1_v2')).height:,} "
                f"/ {df_seed_v2.height:,}"
            )

            def _smean(_d, _c):
                if _d.height == 0 or _c not in _d.columns:
                    return float("nan")
                _s = _d[_c].drop_nulls().drop_nans()
                return float(_s.mean()) if _s.len() > 0 else float("nan")

            df_b1_v1 = df_seed_v2.filter(pl.col("is_textbook_b1"))
            df_b1_v2 = df_seed_v2.filter(pl.col("is_textbook_b1_v2"))
            seed_mean = _smean(df_seed_v2, ra_col)
            rows_a = [
                {
                    "version": "baseline",
                    "group": "seed_mid 全体",
                    "rows": df_seed_v2.height,
                    "mean_risk_adj": round(seed_mean, 4),
                },
                {
                    "version": "v1",
                    "group": "is_textbook_b1=True",
                    "rows": df_b1_v1.height,
                    "mean_risk_adj": round(_smean(df_b1_v1, ra_col), 4),
                },
                {
                    "version": "v2",
                    "group": "is_textbook_b1_v2=True",
                    "rows": df_b1_v2.height,
                    "mean_risk_adj": round(_smean(df_b1_v2, ra_col), 4),
                },
            ]
            print(f"\n  [D1] is_textbook_b1 平均 risk_adj_{h}d (v1 vs v2):")
            print(pl.DataFrame(rows_a))

            bins = [0.0, 0.30, 0.50, 0.65, 0.75, 0.85, 1.01]
            bin_rows = []
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                d_v1 = df_seed_v2.filter(
                    (pl.col("textbook_b1_score") >= lo) & (pl.col("textbook_b1_score") < hi)
                )
                d_v2 = df_seed_v2.filter(
                    (pl.col("textbook_b1_score_v2") >= lo) & (pl.col("textbook_b1_score_v2") < hi)
                )
                bin_rows.append(
                    {
                        "score_range": f"[{lo:.2f}, {hi:.2f})",
                        "v1_rows": d_v1.height,
                        "v1_mean_risk_adj": round(_smean(d_v1, ra_col), 4),
                        "v2_rows": d_v2.height,
                        "v2_mean_risk_adj": round(_smean(d_v2, ra_col), 4),
                    }
                )
            print(f"\n  [D2] 6 档分箱平均 risk_adj_{h}d (v1 vs v2 单调性):")
            print(pl.DataFrame(bin_rows))

            p90 = float(
                df_seed_v2[ra_col].drop_nulls().drop_nans().quantile(0.90, interpolation="linear")
            ) if df_seed_v2.height else 0.0
            df_top10 = df_seed_v2.filter(pl.col(ra_col) >= p90)
            seed_b1_ratio_v1 = df_b1_v1.height / max(df_seed_v2.height, 1)
            seed_b1_ratio_v2 = df_b1_v2.height / max(df_seed_v2.height, 1)
            enrich_rows = []
            for label, df_g in [
                ("seed_mid 全体 (baseline)", df_seed_v2),
                (f"Top 10% risk_adj_{h}d (>={p90:.4f})", df_top10),
            ]:
                t = df_g.height
                r1 = df_g.filter(pl.col("is_textbook_b1")).height / max(t, 1)
                r2 = df_g.filter(pl.col("is_textbook_b1_v2")).height / max(t, 1)
                enrich_rows.append(
                    {
                        "sample": label,
                        "total": t,
                        "v1_b1_ratio": f"{r1:.2%}",
                        "v1_enrichment": f"{r1 / max(seed_b1_ratio_v1, 1e-8):.2f}x",
                        "v2_b1_ratio": f"{r2:.2%}",
                        "v2_enrichment": f"{r2 / max(seed_b1_ratio_v2, 1e-8):.2f}x",
                    }
                )
            print(f"\n  [D3] 强表现样本中 B1 占比 (v1 vs v2 enrichment, horizon = {h}d):")
            print(pl.DataFrame(enrich_rows))

            if df_top10.height > 0:
                arch_top = (
                    df_top10.group_by("textbook_best_archetype_v2")
                    .agg(
                        [
                            pl.len().alias("rows"),
                            pl.col(ra_col).mean().alias("mean_risk_adj"),
                            pl.col("textbook_b1_score_v2").mean().alias("mean_v2_score"),
                        ]
                    )
                    .with_columns([(pl.col("rows") / df_top10.height).alias("pct_of_top10")])
                    .sort("rows", descending=True)
                    .with_columns(
                        [
                            pl.col("mean_risk_adj").round(4),
                            pl.col("mean_v2_score").round(4),
                            pl.col("pct_of_top10").round(4),
                        ]
                    )
                )
                print(f"\n  [D4] Top 10% risk_adj_{h}d 中样本最匹配的 archetype 分布:")
                print(arch_top)
                missing = [
                    n for n in case_names_list if n not in arch_top["textbook_best_archetype_v2"].to_list()
                ]
                if missing:
                    print(f"  注: 在 Top 10% 强样本中无人匹配的 archetype: {missing}")

    _run()
    return


@app.cell
def _(
    ACTIVE_HORIZON,
    B1_FEATURE_TO_GROUP,
    CASE_VECTOR_COLS,
    df_cases,
    df_seed,
    np,
    pl,
):
    def _run():
        """
        Step E. Cohen's d 特征重要性 (H2 验证 / Step 2f)
        -------------------------------------------------
        在全特征池上比较 '完美案例 vs seed_mid baseline' 的标准化效应量,
        回答: 真正把案例从 seed_mid 中区分开的特征有哪些?
        判读:
          - 若 |d| Top-N 中, 14 个 textbook 特征出现率高 → H2 不成立, 反向富集另有原因
          - 若 14 个 textbook 特征几乎不在 Top → H2 成立, 案例的 '完美' 由别的因子驱动
        """
        h = ACTIVE_HORIZON
        print("\n" + "=" * 72)
        print(f"  Step E. Cohen's d 特征效应量排序 (case vs seed baseline, horizon = {h}d)")
        print("=" * 72)

        if df_cases.height == 0:
            print("  [SKIP] df_cases 为空")
        else:
            excluded_prefixes = (
                "fwd_",
                "_",
                "is_",
                "textbook_",
                "case",
                "seed_",
            )
            excluded_exact = {
                "code",
                "date",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "volume",
                "amount",
                "market_cap_100m",
                "TRIGGER",
                "KEY_K",
                "KEY_K_EXIST",
                "PLRY_CNT",
                "GOOD28",
                "MAX28_OK",
                "YANGYIN_OK",
                "is_manual_bull",
            }

            numeric_cols = []
            for col, dtype in zip(df_seed.columns, df_seed.dtypes):
                if col in excluded_exact:
                    continue
                if any(col.startswith(p) for p in excluded_prefixes):
                    continue
                if col not in df_cases.columns:
                    continue
                if dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                    continue
                numeric_cols.append(col)

            textbook_set = set(CASE_VECTOR_COLS)

            rows = []
            for col in numeric_cols:
                seed_s = df_seed[col].drop_nulls().drop_nans()
                case_s = df_cases[col].drop_nulls().drop_nans()
                if seed_s.len() < 50 or case_s.len() < 3:
                    continue
                seed_mean = float(seed_s.mean())
                case_mean = float(case_s.mean())
                seed_std = float(seed_s.std(ddof=1)) if seed_s.len() > 1 else 0.0
                case_std = float(case_s.std(ddof=1)) if case_s.len() > 1 else 0.0
                n1, n2 = seed_s.len(), case_s.len()
                denom = max(n1 + n2 - 2, 1)
                pooled_var = ((n1 - 1) * seed_std**2 + (n2 - 1) * case_std**2) / denom
                pooled_std = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
                if pooled_std <= 0:
                    continue
                d = (case_mean - seed_mean) / pooled_std
                rows.append(
                    {
                        "feature": col,
                        "in_textbook14": col in textbook_set,
                        "group": B1_FEATURE_TO_GROUP.get(col, "other"),
                        "case_mean": round(case_mean, 4),
                        "seed_mean": round(seed_mean, 4),
                        "case_std": round(case_std, 4),
                        "seed_std": round(seed_std, 4),
                        "cohen_d": round(d, 4),
                        "abs_d": round(abs(d), 4),
                    }
                )

            if not rows:
                print("  [SKIP] 没有合格的数值特征 (样本量不足 / 方差为 0)")
            else:
                df_cohen = pl.DataFrame(rows).sort("abs_d", descending=True)
                print(
                    f"\n  scanned features: {len(rows)},  textbook14 hit "
                    f"{df_cohen.filter(pl.col('in_textbook14')).height} / {len(textbook_set)}"
                )

                print("\n  [E1] |Cohen's d| Top 30 — 真正区分案例和 baseline 的特征:")
                print(df_cohen.head(30))

                print("\n  [E2] 14 个 textbook 特征自身的 Cohen's d (按 |d| 降序):")
                print(
                    df_cohen.filter(pl.col("in_textbook14")).sort("abs_d", descending=True)
                )

                print("\n  [E3] 按 group 聚合 — 哪一组特征整体上更能区分案例:")
                group_summary = (
                    df_cohen.group_by("group")
                    .agg(
                        [
                            pl.len().alias("n_features"),
                            pl.col("abs_d").mean().alias("mean_abs_d"),
                            pl.col("abs_d").max().alias("max_abs_d"),
                            pl.col("in_textbook14").sum().alias("n_in_textbook14"),
                        ]
                    )
                    .with_columns(
                        [
                            pl.col("mean_abs_d").round(4),
                            pl.col("max_abs_d").round(4),
                        ]
                    )
                    .sort("mean_abs_d", descending=True)
                )
                print(group_summary)

                top30 = df_cohen.head(30)
                tb_in_top30 = int(top30.filter(pl.col("in_textbook14")).height)
                tb_total = len(textbook_set)
                print("\n  [E4] 自动判读:")
                print(f"    Top 30 中 textbook14 特征命中 = {tb_in_top30} / {tb_total}")
                if tb_in_top30 >= tb_total * 0.7:
                    print(
                        "    → [H2 不成立] 14 个 textbook 特征大部分都在 Top, 反向富集要从别的角度找"
                        " (聚合方式 H1 / 阈值 / 标签窗口 / 案例日期)。"
                    )
                elif tb_in_top30 >= tb_total * 0.4:
                    print(
                        "    → [H2 部分成立] textbook14 部分在 Top, 但缺失了能真正区分案例的关键因子, "
                        "建议在 Top 30 中挑出非 textbook 的高 |d| 特征做 v3 模板。"
                    )
                else:
                    print(
                        "    → [H2 强成立] textbook14 几乎不在 Top, 案例的 '完美' 由 14 维之外的因子驱动, "
                        "需要重新设计 textbook_b1_score 的输入特征。"
                    )

                # ── Step F. 共线诊断 (|d| Top 30 Pearson 矩阵) ──────────────
                print("\n" + "=" * 72)
                print(
                    f"  Step F. |Cohen's d| Top 30 共线诊断 (Pearson, on seed pool, horizon = {h}d)"
                )
                print("=" * 72)
                CORR_THR = 0.85

                top_features = top30["feature"].to_list()
                feat_to_d = dict(
                    zip(top30["feature"].to_list(), top30["abs_d"].to_list())
                )
                feat_to_in_tb = dict(
                    zip(top30["feature"].to_list(), top30["in_textbook14"].to_list())
                )

                mat_df = (
                    df_seed.select(top_features)
                    .fill_nan(None)
                    .drop_nulls()
                )
                if mat_df.height < 100 or len(top_features) < 2:
                    print(
                        f"  [SKIP] Top 30 在 seed 池上可用样本不足 (rows={mat_df.height})"
                    )
                else:
                    mat = mat_df.to_numpy()
                    corr = np.corrcoef(mat.T)
                    n_feat = len(top_features)

                    high_pairs = []
                    for i in range(n_feat):
                        for j in range(i + 1, n_feat):
                            c = float(corr[i, j])
                            if abs(c) >= CORR_THR:
                                high_pairs.append(
                                    {
                                        "feature_a": top_features[i],
                                        "feature_b": top_features[j],
                                        "corr": round(c, 4),
                                        "abs_corr": round(abs(c), 4),
                                        "abs_d_a": round(feat_to_d[top_features[i]], 4),
                                        "abs_d_b": round(feat_to_d[top_features[j]], 4),
                                    }
                                )

                    print(
                        f"\n  rows used = {mat_df.height},  features = {n_feat},  "
                        f"|corr| >= {CORR_THR} pair count = {len(high_pairs)}"
                    )

                    print("\n  [F1] |corr| >= 0.85 的高共线对 (按 |corr| 降序):")
                    if not high_pairs:
                        print("    无 — Top 30 之间共线性可控")
                    else:
                        df_pairs = (
                            pl.DataFrame(high_pairs)
                            .sort("abs_corr", descending=True)
                            .drop("abs_corr")
                        )
                        print(df_pairs)

                    parent = list(range(n_feat))

                    def _find(x: int) -> int:
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def _union(a: int, b: int) -> None:
                        ra, rb = _find(a), _find(b)
                        if ra != rb:
                            parent[ra] = rb

                    feat_idx = {f: i for i, f in enumerate(top_features)}
                    for pair in high_pairs:
                        _union(feat_idx[pair["feature_a"]], feat_idx[pair["feature_b"]])

                    clusters: dict[int, list[str]] = {}
                    for i, feat in enumerate(top_features):
                        root = _find(i)
                        clusters.setdefault(root, []).append(feat)

                    multi_clusters = [
                        members for members in clusters.values() if len(members) > 1
                    ]
                    singletons = [
                        members[0] for members in clusters.values() if len(members) == 1
                    ]

                    print(
                        f"\n  [F2] 共线簇 ({len(multi_clusters)} 簇, 另 {len(singletons)} 个独立特征):"
                    )
                    if not multi_clusters:
                        print("    无")
                    else:
                        for idx, members in enumerate(multi_clusters, 1):
                            members_sorted = sorted(
                                members, key=lambda f: feat_to_d[f], reverse=True
                            )
                            rep = members_sorted[0]
                            print(
                                f"    簇 {idx}: 代表 = {rep}  |d|={feat_to_d[rep]:.3f}  "
                                f"({len(members_sorted)} 个等价特征)"
                            )
                            for m in members_sorted:
                                star = " ★" if m == rep else ""
                                tb_tag = " [tb14]" if feat_to_in_tb[m] else ""
                                print(
                                    f"      - {m:30s} |d|={feat_to_d[m]:.3f}{tb_tag}{star}"
                                )

                    v3_candidates = []
                    for members in clusters.values():
                        rep = max(members, key=lambda f: feat_to_d[f])
                        v3_candidates.append(rep)
                    v3_candidates.sort(key=lambda f: feat_to_d[f], reverse=True)

                    print(
                        f"\n  [F3] 共线去重后的 V3 候选清单 ({len(v3_candidates)} 个, 按 |d| 降序):"
                    )
                    for i, feat in enumerate(v3_candidates, 1):
                        tb_tag = " ★ textbook14" if feat_to_in_tb[feat] else ""
                        print(f"    {i:2d}. {feat:30s} |d|={feat_to_d[feat]:.3f}{tb_tag}")

                    print(
                        "\n  说明: V3 草拟时, 每个共线簇只保留 |d| 最大的代表; "
                        "[tb14] 表示该特征当前在 textbook14 中。"
                    )

    _run()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 探索面板

    - **Step 0**: 数据加载 + multi-horizon 标签
    - **Step A / A4**: 完美案例自身前瞻收益 (multi-horizon, 含 spotlight `sz.002074` 国轩高科)
    - **Step B**: B1 形态占比与平均表现 (按 `ACTIVE_HORIZON` 切换)
    - **Step C**: Textbook centroid 自洽性诊断 (H1)
    - **Step D**: v2 max-archetype 模拟 (H1 修复实验, 跟随 `ACTIVE_HORIZON`)
    - **Step E**: Cohen's d 特征重要性 (H2 / Step 2f, 跟随 `ACTIVE_HORIZON`)
    - **Step F**: |d| Top 30 共线诊断 (Pearson 矩阵 + 共线簇代表 + V3 候选清单)

    切换 horizon: 修改 cell-0 的 `ACTIVE_HORIZON ∈ {5,10,15,20,30,40}`,
    其他全部分析自动重算。
    """)
    return


if __name__ == "__main__":
    app.run()
