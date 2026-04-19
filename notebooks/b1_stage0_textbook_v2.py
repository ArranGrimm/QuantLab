import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    from utils import build_b1_research_frame, get_st_blacklist_pl, load_daily_data_full

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0
    INCLUDE_ROTATION_KBAR_FEATURES = False

    ACTIVE_HORIZON = 20

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
        DB_PATH,
        END_DATE,
        INCLUDE_ROTATION_KBAR_FEATURES,
        LOOSE_PERIODS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        build_b1_research_frame,
        duckdb,
        get_st_blacklist_pl,
        load_daily_data_full,
        mo,
        np,
        pl,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # B1 Stage-0 教科书 v2 — 序列健康度 + 原版硬规则

        **基于**: z 哥 B1 完美图原版 (Downloads/十大B1完美图.pdf) 的 5 条硬规则:

        1. 量能 — 对比前期高点放量, 极致缩量
        2. 涨幅 — -2% ~ +1.8% (小阴小阳)
        3. 振幅 — ≤ 7%
        4. 均线 — 白线 > 黄线 (= WL > YL)
        5. 指标 — J 勾到大负值 (≤ 13, 放宽到 < 14)

        **对照 b1_stage0_J_interaction 的反直觉发现**: vol_shrink_40 单点的 V1 (无量) 是负 alpha,
        因为它包含大量"无人问津的烂股长期阴跌"; 教科书的"健康"必须先满足
        **前期 60 日内出现过放量启动 (大哥建仓痕迹)**, 缩量才有意义.

        **新引入特征** (b1_feature_pool 已加):

        - `prior_volume_surge_60d`: 过去 60 天内是否出现过 vol/vol_ma40 ≥ 2 的放量阳线
        - `peak_vol_shrink_60d`: 当日量 / 过去 60 天最大量 (越小越"对比前期高点缩量")
        - `pullback_vol_shrink_5_20`: 5日均量 / 20日均量 (越小越回调缩量)

        **Notebook 结构**:

        - Cell T0: q_full + df_all (含新特征)
        - Cell T1: 累积过滤 — L0 / L2 / L2+surge / L2+surge+今日企稳 / +J<14 各层 alpha
        - Cell T2: 在 L2_surge 内做 J × peak_vol_shrink_60d 2D, 验证 "前期有放量后, 极致缩量 + 小J" 是否真有 alpha
        - Cell T3: 6 个 case-stratum Bootstrap 95% CI 对比, 含原版完整 5 条规则
        """
    )
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
    print(f"q_full prepared, st_blacklist size = {len(st_blacklist):,}")
    return (q_full,)


@app.cell
def _(
    INCLUDE_ROTATION_KBAR_FEATURES,
    LOOSE_PERIODS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    SEED_J_MAX,
    build_b1_research_frame,
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
        textbook_score_version="v1",
    )
    print(f"df_all = {df_all.height:,} rows, {len(df_all.columns)} cols")
    for c in ["prior_volume_surge_60d", "peak_vol_shrink_60d", "pullback_vol_shrink_5_20"]:
        present = c in df_all.columns
        print(f"  含 {c} = {present}")
    return (df_all,)


@app.cell
def _(ACTIVE_HORIZON, df_all, pl):
    """注入 fwd_*_{horizon}d + 当日涨幅 + 振幅. 输出 df_all_h."""
    def _run():
        _h = ACTIVE_HORIZON

        future_high_cols = [
            pl.col("high_adj").shift(-step).over("code").alias(f"_fwd_high_{step}")
            for step in range(1, _h + 1)
        ]
        future_low_cols = [
            pl.col("low_adj").shift(-step).over("code").alias(f"_fwd_low_{step}")
            for step in range(1, _h + 1)
        ]
        future_high_names = [f"_fwd_high_{step}" for step in range(1, _h + 1)]
        future_low_names = [f"_fwd_low_{step}" for step in range(1, _h + 1)]

        df_all_h = (
            df_all.lazy()
            .with_columns(future_high_cols + future_low_cols)
            .with_columns([
                (pl.max_horizontal(future_high_names) / pl.col("close_adj") - 1).alias(f"fwd_mfe_{_h}d"),
                (pl.min_horizontal(future_low_names) / pl.col("close_adj") - 1).alias(f"fwd_mae_{_h}d"),
                (pl.col("close_adj").shift(-_h).over("code") / pl.col("close_adj") - 1).alias(f"fwd_ret_{_h}d"),
                (pl.col("close_adj") / pl.col("close_adj").shift(1).over("code") - 1).alias("today_ret"),
                (
                    (pl.col("high_adj") - pl.col("low_adj"))
                    / pl.max_horizontal(pl.col("close_adj").shift(1).over("code"), pl.lit(0.01))
                ).alias("today_amplitude"),
            ])
            .with_columns(
                (pl.col(f"fwd_mfe_{_h}d") / (1 + pl.col(f"fwd_mae_{_h}d").abs())).alias(
                    f"fwd_mfe_risk_adj_{_h}d"
                )
            )
            .drop(future_high_names + future_low_names)
            .filter(pl.col(f"fwd_mfe_risk_adj_{_h}d").is_not_null())
            .filter(pl.col(f"fwd_ret_{_h}d").is_not_null())
            .with_columns(
                pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"),
            )
            .select([
                "code", "date", "J", "WL", "YL", "close_adj",
                "vol_shrink_40", "red_green_ratio_20",
                "prior_volume_surge_60d", "peak_vol_shrink_60d", "pullback_vol_shrink_5_20",
                "today_ret", "today_amplitude", "is_manual_bull",
                "market_cap_100m", "amount_ma20",
                f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_ret_{_h}d", f"fwd_mfe_risk_adj_{_h}d",
            ])
            .collect()
        )
        print(f"df_all_h = {df_all_h.height:,} rows")
        return df_all_h

    df_all_h = _run()
    return (df_all_h,)


@app.cell
def _(df_all_h, pl):
    """Sanity check: 新特征的分布 + 在 L2 (WL>YL & close>YL) 上的覆盖率."""
    def _run():
        L2 = df_all_h.filter(
            (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        )
        n_L2 = L2.height
        L2_surge = L2.filter(pl.col("prior_volume_surge_60d"))
        n_surge = L2_surge.height

        print(f"L2 (WL>YL & close>YL)                       : {n_L2:>10,} rows")
        print(f"L2 + prior_volume_surge_60d (有过放量启动)   : {n_surge:>10,} rows ({n_surge/n_L2:.2%} of L2)")

        print("\npeak_vol_shrink_60d (在 L2_surge 内) 分位:")
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            print(f"  Q{int(q*100)} = {float(L2_surge['peak_vol_shrink_60d'].quantile(q) or 0.0):.4f}")

        print("\npullback_vol_shrink_5_20 (在 L2_surge 内) 分位:")
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            print(f"  Q{int(q*100)} = {float(L2_surge['pullback_vol_shrink_5_20'].quantile(q) or 0.0):.4f}")

        print(f"\n今日企稳 (today_ret ∈ [-2%, +1.8%]) 在 L2_surge 内占比: "
              f"{((L2_surge['today_ret'] >= -0.02) & (L2_surge['today_ret'] <= 0.018)).mean():.2%}")
        print(f"今日振幅 ≤ 7% 在 L2_surge 内占比: "
              f"{(L2_surge['today_amplitude'] <= 0.07).mean():.2%}")
        print(f"J < 14 在 L2_surge 内占比: "
              f"{(L2_surge['J'] < 14).mean():.2%}")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_all_h, pl):
    """Step T1: 累积过滤层层加压, 看每加一条原版规则 alpha 怎么变。
    L0 → L2 → +surge → +今日企稳 → +振幅 → +J<14 → +极致缩量 (peak_shrink ≤ Q25)
    """
    def _run():
        _h = ACTIVE_HORIZON
        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"
        ret = f"fwd_ret_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)
        baseline_ret = float(df_all_h[ret].mean() or 0.0)

        L2 = df_all_h.filter(
            (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        )
        peak_q25 = float(L2.filter(pl.col("prior_volume_surge_60d"))["peak_vol_shrink_60d"].quantile(0.25) or 0.0)

        cond_surge = pl.col("prior_volume_surge_60d")
        cond_today = (pl.col("today_ret") >= -0.02) & (pl.col("today_ret") <= 0.018)
        cond_amp = pl.col("today_amplitude") <= 0.07
        cond_jhook = pl.col("J") < 14
        cond_shrink = pl.col("peak_vol_shrink_60d") <= peak_q25

        layers = [
            ("L0 全市场 (baseline)",                          df_all_h),
            ("L2 (WL>YL & close>YL)",                         L2),
            ("L2 + 规则1: 前期放量启动 (surge_60d)",            L2.filter(cond_surge)),
            ("L2 + 规则1+2: + 今日企稳 (-2%~+1.8%)",            L2.filter(cond_surge & cond_today)),
            ("L2 + 规则1+2+3: + 振幅≤7%",                       L2.filter(cond_surge & cond_today & cond_amp)),
            ("L2 + 规则1+2+3+5: + J<14 (勾到大负值)",            L2.filter(cond_surge & cond_today & cond_amp & cond_jhook)),
            ("L2 + 全套5条规则 (+极致缩量 peak≤Q25)",            L2.filter(cond_surge & cond_today & cond_amp & cond_jhook & cond_shrink)),
        ]

        rows = []
        for label, sub in layers:
            if sub.height == 0:
                rows.append({"layer": label, "rows": 0, "note": "空"})
                continue
            mean_mfe = float(sub[mfe].mean() or 0.0)
            mean_ric = float(sub[ric].mean() or 0.0)
            mean_ret = float(sub[ret].mean() or 0.0)
            hit15 = float((sub[mfe] >= 0.15).mean() or 0.0)
            win = float((sub[ret] > 0).mean() or 0.0)
            rows.append({
                "layer": label,
                "rows": sub.height,
                f"mean_mfe": round(mean_mfe, 4),
                f"mean_ric": round(mean_ric, 4),
                f"mean_ret": round(mean_ret, 4),
                "hit_15pct": round(hit15, 4),
                "win_rate": f"{win:.2%}",
                "mfe_lift_vs_L0": f"{(mean_mfe - baseline_mfe) * 100:+.2f}pp",
                "ric_lift_vs_L0": f"{(mean_ric - baseline_ric) * 100:+.2f}pp",
                "ret_lift_vs_L0": f"{(mean_ret - baseline_ret) * 100:+.2f}pp",
                "hit_lift_vs_L0": f"{(hit15 - baseline_hit) * 100:+.2f}pp",
            })

        print("=" * 80)
        print(f"  [T1] 累积过滤 alpha lift  (horizon = {_h}d)")
        print(f"  baseline: mfe={baseline_mfe:.4f}, ric={baseline_ric:.4f}, hit15={baseline_hit:.4f}, ret={baseline_ret:.4f}")
        print(f"  peak_vol_shrink_60d Q25 (在 L2_surge 内) = {peak_q25:.4f}")
        print("=" * 80)
        print(pl.DataFrame(rows))

        return peak_q25

    peak_q25 = _run()
    return (peak_q25,)


@app.cell
def _(ACTIVE_HORIZON, df_all_h, pl):
    """Step T2: 在 L2_surge (WL>YL & close>YL & prior_surge_60d) 内做 J × peak_vol_shrink_60d 2D。
    这是真正"教科书定义下的健康样本"对 J 阈值的响应。"""
    def _run():
        _h = ACTIVE_HORIZON
        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)

        L2_surge = df_all_h.filter(
            (pl.col("WL") > pl.col("YL"))
            & (pl.col("close_adj") > pl.col("YL"))
            & pl.col("prior_volume_surge_60d")
            & pl.col("peak_vol_shrink_60d").is_not_null()
        )
        print(f"L2_surge = {L2_surge.height:,} rows  (vs L2_all-old ≈ 964k 中保留比例 {L2_surge.height/963833:.2%})")

        pvs = L2_surge["peak_vol_shrink_60d"]
        q25 = float(pvs.quantile(0.25) or 0.0)
        q50 = float(pvs.quantile(0.50) or 0.0)
        q75 = float(pvs.quantile(0.75) or 0.0)

        def _shrink_bin(col):
            return (
                pl.when(col <= q25).then(pl.lit("S1 ≤Q25 (极致缩量, 教科书最优)"))
                .when(col <= q50).then(pl.lit("S2 Q25~Q50 (温和缩量)"))
                .when(col <= q75).then(pl.lit("S3 Q50~Q75"))
                .otherwise(pl.lit("S4 >Q75 (近期重新放量)"))
            )

        def _j_bin(col):
            return (
                pl.when(col < 14).then(pl.lit("J1 J<14 (教科书标准)"))
                .when(col <= 20).then(pl.lit("J2 14≤J≤20"))
                .when(col <= 30).then(pl.lit("J3 20<J≤30"))
                .when(col <= 50).then(pl.lit("J4 30<J≤50"))
                .when(col <= 80).then(pl.lit("J5 50<J≤80"))
                .otherwise(pl.lit("J6 J>80"))
            )

        df = L2_surge.with_columns(
            _shrink_bin(pl.col("peak_vol_shrink_60d")).alias("_sbin"),
            _j_bin(pl.col("J")).alias("_jbin"),
        )

        agg = (
            df.group_by(["_sbin", "_jbin"])
            .agg(
                pl.len().alias("rows"),
                pl.col(mfe).mean().alias("mean_mfe"),
                pl.col(ric).mean().alias("mean_ric"),
                (pl.col(mfe) >= 0.15).mean().alias("hit_15pct"),
            )
            .with_columns(
                (pl.col("mean_mfe") - baseline_mfe).alias("mfe_lift"),
                (pl.col("mean_ric") - baseline_ric).alias("ric_lift"),
                (pl.col("hit_15pct") - baseline_hit).alias("hit_lift"),
            )
            .sort(["_sbin", "_jbin"])
        )

        def _print_pivot(value_col, fmt, title):
            print("\n" + "=" * 82)
            print(f"  {title}")
            print("=" * 82)
            piv = agg.pivot(values=value_col, index="_sbin", on="_jbin", sort_columns=True)
            piv_fmt = piv.select([
                pl.col("_sbin"),
                *[pl.col(c).map_elements(lambda x, _f=fmt: _f.format(x) if x is not None else "—", return_dtype=pl.Utf8) for c in piv.columns if c != "_sbin"],
            ])
            print(piv_fmt)

        rows_piv = agg.pivot(values="rows", index="_sbin", on="_jbin", sort_columns=True)
        print("=" * 82)
        print(f"  [T2.0] 样本量 pivot  (L2_surge 内, peak_shrink × J)")
        print(f"  peak_q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}, baseline_mfe={baseline_mfe:.4f}")
        print("=" * 82)
        print(rows_piv)

        _print_pivot("mfe_lift", "{:+.4f}", "[T2.1] mfe_lift_vs_L0  ← 关键, 看 S1+J1 是否爆发")
        _print_pivot("ric_lift", "{:+.4f}", "[T2.2] risk_adj_lift_vs_L0")
        _print_pivot("hit_lift", "{:+.4f}", "[T2.3] hit_15pct_lift_vs_L0")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_all_h, np, peak_q25, pl):
    """Step T3: 6 个 case-stratum 的 Bootstrap 95% CI 显著性检验。
    包含 "原版 5 条规则全开" 这个最严定义。
    """
    def _run():
        _h = ACTIVE_HORIZON
        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"
        ret = f"fwd_ret_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)
        baseline_ret = float(df_all_h[ret].mean() or 0.0)

        cond_L2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        cond_surge = pl.col("prior_volume_surge_60d")
        cond_today = (pl.col("today_ret") >= -0.02) & (pl.col("today_ret") <= 0.018)
        cond_amp = pl.col("today_amplitude") <= 0.07
        cond_jhook = pl.col("J") < 14
        cond_shrink = pl.col("peak_vol_shrink_60d") <= peak_q25

        cells = [
            ("a. L2 全集 (上版基线)",                          cond_L2),
            ("b. L2 + surge (有过启动)",                       cond_L2 & cond_surge),
            ("c. L2 + surge + 今日企稳",                       cond_L2 & cond_surge & cond_today),
            ("d. L2 + surge + 企稳 + 振幅≤7% + J<14",          cond_L2 & cond_surge & cond_today & cond_amp & cond_jhook),
            ("e. L2 + 教科书全 5 条 (含 peak_shrink≤Q25)",     cond_L2 & cond_surge & cond_today & cond_amp & cond_jhook & cond_shrink),
            ("f. (对照) L2 + surge + J<14 (不要今日企稳/缩量)", cond_L2 & cond_surge & cond_jhook),
            ("g. (对照) L2 + surge + 极致缩量 (不要 J<14)",     cond_L2 & cond_surge & cond_shrink),
        ]

        rng = np.random.default_rng(42)
        N_BOOT = 1000
        rows = []
        for label, cond in cells:
            sub = df_all_h.filter(cond)
            if sub.height < 50:
                rows.append({"cell": label, "rows": sub.height, "note": "样本太少"})
                continue
            sub_mfe = sub[mfe].to_numpy()
            sub_ric = sub[ric].to_numpy()
            sub_ret = sub[ret].to_numpy()
            n = sub.height

            mfe_lifts = np.empty(N_BOOT)
            ric_lifts = np.empty(N_BOOT)
            ret_lifts = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                mfe_lifts[b] = sub_mfe[idx].mean() - baseline_mfe
                ric_lifts[b] = sub_ric[idx].mean() - baseline_ric
                ret_lifts[b] = sub_ret[idx].mean() - baseline_ret

            mfe_ci = (np.quantile(mfe_lifts, 0.025), np.quantile(mfe_lifts, 0.975))
            ric_ci = (np.quantile(ric_lifts, 0.025), np.quantile(ric_lifts, 0.975))
            ret_ci = (np.quantile(ret_lifts, 0.025), np.quantile(ret_lifts, 0.975))

            hit15 = float((sub_mfe >= 0.15).mean())
            win = float((sub_ret > 0).mean())

            rows.append({
                "cell": label,
                "rows": sub.height,
                "mean_mfe": round(float(sub_mfe.mean()), 4),
                "mfe_lift": f"{mfe_lifts.mean()*100:+.3f}pp",
                "mfe_95%CI": f"[{mfe_ci[0]*100:+.3f}, {mfe_ci[1]*100:+.3f}]pp",
                "mfe_sig": "✓" if mfe_ci[0] > 0 else ("⚠反向" if mfe_ci[1] < 0 else "✗"),
                "ric_lift": f"{ric_lifts.mean()*100:+.3f}pp",
                "ret_lift": f"{ret_lifts.mean()*100:+.3f}pp",
                "ret_95%CI": f"[{ret_ci[0]*100:+.3f}, {ret_ci[1]*100:+.3f}]pp",
                "ret_sig": "✓" if ret_ci[0] > 0 else ("⚠反向" if ret_ci[1] < 0 else "✗"),
                "hit_15pct": f"{hit15:.4%}",
                "win_rate": f"{win:.2%}",
            })

        print("=" * 82)
        print(f"  [T3] 教科书规则累积  Bootstrap 95% CI ({N_BOOT} resamples, horizon={_h}d)")
        print(f"  baseline mfe={baseline_mfe:.4f}, ric={baseline_ric:.4f}, hit15={baseline_hit:.4f}, ret={baseline_ret:.4f}")
        print("=" * 82)
        print(pl.DataFrame(rows))

        print("\n  ───────────────────── 假设判读 ─────────────────────")
        print("  对比 a→b: 加 surge 是否显著放大 alpha → 验证 '前期放量启动' 是教科书的硬前提")
        print("  对比 b→c: 加 '今日小阴小阳' 是否进一步放大 → 验证企稳规则")
        print("  对比 c→d: 加 J<14 在已经企稳的样本上是否还有 alpha → 验证你的核心假设!")
        print("  对比 d→e: 再加极致缩量是否还有边际增量")
        print("  对比 d vs f: J<14 单独加在 surge 后, 不需要今日企稳, 是否就够了")
        print("  对比 e vs g: 教科书全规则 vs 仅 surge+缩量, 看 J 在末端是否多余")
        print("\n  关键看: e 的 ret_sig 是否 ✓ → 全规则下 close-to-close 是否真正翻正")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_all_h, np, peak_q25, pl):
    """Step T4: regime filter — 把 T1 累积过滤限制在 is_manual_bull == True 子集
    (LOOSE_PERIODS 框定的多头/宽松行情区间)。

    背景: 知乎同好观察 — z 哥 PDF 只讲了 '术' (怎么识别完美 B1 形态),
    没讲 '道' (什么市场状态才能开 B1). 本 cell 验证教科书 5 条规则
    在多头区间内是否能从负 alpha 翻成正 alpha。

    重要: 这里用 df_all_h 全样本算 baseline, 但子样本只取 is_manual_bull, 才能
    公平对比 'regime filter 是否独立贡献 alpha'。
    """
    def _run():
        _h = ACTIVE_HORIZON
        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"
        ret = f"fwd_ret_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)
        baseline_ret = float(df_all_h[ret].mean() or 0.0)

        bull = df_all_h.filter(pl.col("is_manual_bull"))
        bear = df_all_h.filter(~pl.col("is_manual_bull"))
        print(f"is_manual_bull 覆盖: {bull.height:,} / {df_all_h.height:,} = {bull.height/df_all_h.height:.2%}")

        # 多头自身的 baseline (regime alpha 独立贡献)
        bull_baseline_mfe = float(bull[mfe].mean() or 0.0)
        bull_baseline_ret = float(bull[ret].mean() or 0.0)
        bear_baseline_mfe = float(bear[mfe].mean() or 0.0)
        bear_baseline_ret = float(bear[ret].mean() or 0.0)
        print(f"  bull baseline: mfe={bull_baseline_mfe:.4f}, ret={bull_baseline_ret:.4f}, win={float((bull[ret]>0).mean() or 0.0):.2%}")
        print(f"  bear baseline: mfe={bear_baseline_mfe:.4f}, ret={bear_baseline_ret:.4f}, win={float((bear[ret]>0).mean() or 0.0):.2%}")
        print(f"  regime alpha (bull vs L0): mfe_lift={(bull_baseline_mfe-baseline_mfe)*100:+.2f}pp, "
              f"ret_lift={(bull_baseline_ret-baseline_ret)*100:+.2f}pp")

        cond_L2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        cond_surge = pl.col("prior_volume_surge_60d")
        cond_today = (pl.col("today_ret") >= -0.02) & (pl.col("today_ret") <= 0.018)
        cond_amp = pl.col("today_amplitude") <= 0.07
        cond_jhook = pl.col("J") < 14
        cond_shrink = pl.col("peak_vol_shrink_60d") <= peak_q25

        layers = [
            ("L0 全市场 (基线)",                                df_all_h),
            ("R0 manual_bull 全集 (regime baseline)",            bull),
            ("R0 manual_bear 全集 (对照)",                       bear),
            ("R+L2 bull + (WL>YL & close>YL)",                  bull.filter(cond_L2)),
            ("R+L2+surge",                                       bull.filter(cond_L2 & cond_surge)),
            ("R+L2+surge+today企稳",                              bull.filter(cond_L2 & cond_surge & cond_today)),
            ("R+L2+surge+企稳+振幅+J<14",                          bull.filter(cond_L2 & cond_surge & cond_today & cond_amp & cond_jhook)),
            ("R+L2+教科书全 5 条 (含极致缩量)",                    bull.filter(cond_L2 & cond_surge & cond_today & cond_amp & cond_jhook & cond_shrink)),
            ("(对照) bear+L2+教科书全 5 条",                      bear.filter(cond_L2 & cond_surge & cond_today & cond_amp & cond_jhook & cond_shrink)),
        ]

        rng = np.random.default_rng(42)
        N_BOOT = 1000
        rows = []
        for label, sub in layers:
            if sub.height < 50:
                rows.append({"layer": label, "rows": sub.height, "note": "样本太少"})
                continue
            sub_mfe = sub[mfe].to_numpy()
            sub_ret = sub[ret].to_numpy()
            n = sub.height

            mfe_lifts = np.empty(N_BOOT)
            ret_lifts = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                mfe_lifts[b] = sub_mfe[idx].mean() - baseline_mfe
                ret_lifts[b] = sub_ret[idx].mean() - baseline_ret

            mfe_ci = (np.quantile(mfe_lifts, 0.025), np.quantile(mfe_lifts, 0.975))
            ret_ci = (np.quantile(ret_lifts, 0.025), np.quantile(ret_lifts, 0.975))

            mean_mfe = float(sub_mfe.mean())
            mean_ret = float(sub_ret.mean())
            hit15 = float((sub_mfe >= 0.15).mean())
            win = float((sub_ret > 0).mean())

            rows.append({
                "layer": label,
                "rows": sub.height,
                "mean_mfe": round(mean_mfe, 4),
                "mfe_lift_vs_L0": f"{mfe_lifts.mean()*100:+.3f}pp",
                "mfe_95%CI": f"[{mfe_ci[0]*100:+.3f}, {mfe_ci[1]*100:+.3f}]pp",
                "mfe_sig": "✓" if mfe_ci[0] > 0 else ("⚠反向" if mfe_ci[1] < 0 else "✗"),
                "mean_ret": round(mean_ret, 4),
                "ret_lift_vs_L0": f"{ret_lifts.mean()*100:+.3f}pp",
                "ret_95%CI": f"[{ret_ci[0]*100:+.3f}, {ret_ci[1]*100:+.3f}]pp",
                "ret_sig": "✓" if ret_ci[0] > 0 else ("⚠反向" if ret_ci[1] < 0 else "✗"),
                "hit_15pct": f"{hit15:.4%}",
                "win_rate": f"{win:.2%}",
            })

        print("\n" + "=" * 82)
        print(f"  [T4] regime filter — manual_bull 子集上累积 alpha (Bootstrap 95% CI, horizon={_h}d)")
        print(f"  baseline (全样本 L0): mfe={baseline_mfe:.4f}, ret={baseline_ret:.4f}, hit15={baseline_hit:.4f}")
        print("=" * 82)
        print(pl.DataFrame(rows))

        print("\n  ───────────────────── 假设判读 ─────────────────────")
        print("  对比 R0 vs L0          : regime filter 自身能不能贡献 alpha")
        print("  对比 R+L2教科书5条 vs L+L2教科书5条 (T3 之 e 行) : regime 是否能挽救教科书")
        print("  对比 R+L2教科书5条 vs bear+L2教科书5条          : 教科书规则在 bull/bear 是否方向相反")
        print("  关键结论:")
        print("    若 R+L2教科书5条 翻成 ✓ 正 alpha → 同好/教科书都对, 缺的是 regime 这一道")
        print("    若仍是负 alpha → 教科书规则在任何 regime 都失效, B1 真的过时了")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_all_h, np, pl):
    """Step T5: 流动性过滤 (隐患 2) — 用户已确认 LOOSE_PERIODS 是 ex-ante 无 look-ahead。
    现在的问题是: R0 manual_bull baseline 的 +2.46% 是不是被微盘股的尾部脉冲虚高的?

    做 3 档流动性过滤, 在每档里对比:
    - R0 (manual_bull 全集 baseline)
    - R+L2 (+ WL>YL & close>YL)
    - R+L2+J<14 (前述最强形态 cell)
    看形态规则在 "实战可交易池" 里能否反超 baseline。
    """
    def _run():
        _h = ACTIVE_HORIZON
        mfe = f"fwd_mfe_{_h}d"
        ret = f"fwd_ret_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ret = float(df_all_h[ret].mean() or 0.0)

        cond_bull = pl.col("is_manual_bull")
        cond_L2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        cond_jhook = pl.col("J") < 14
        cond_surge = pl.col("prior_volume_surge_60d")
        cond_amp = pl.col("today_amplitude") <= 0.07
        cond_today = (pl.col("today_ret") >= -0.02) & (pl.col("today_ret") <= 0.018)

        # 3 档流动性 (mv 单位: 亿, amount_ma20 单位: 元)
        liquidity_filters = [
            ("L0 无流动性过滤 (df_all 默认 40~1500 亿)", pl.lit(True)),
            ("LF1 中档: mv≥50 + amt20≥3000万",          (pl.col("market_cap_100m") >= 50) & (pl.col("amount_ma20") >= 3e7)),
            ("LF2 严档: mv≥100 + amt20≥5000万",         (pl.col("market_cap_100m") >= 100) & (pl.col("amount_ma20") >= 5e7)),
        ]

        # 在每档流动性下评估的 cell
        formation_cells = [
            ("R0 manual_bull (regime only baseline)",        cond_bull),
            ("R+L2 (regime + WL>YL & close>YL)",             cond_bull & cond_L2),
            ("R+L2+J<14",                                     cond_bull & cond_L2 & cond_jhook),
            ("R+L2+surge+企稳+振幅+J<14 (T4 最强 cell)",       cond_bull & cond_L2 & cond_surge & cond_today & cond_amp & cond_jhook),
        ]

        rng = np.random.default_rng(42)
        N_BOOT = 1000

        for lf_label, lf_cond in liquidity_filters:
            df_lf = df_all_h.filter(lf_cond)
            lf_baseline_mfe = float(df_lf[mfe].mean() or 0.0)
            lf_baseline_ret = float(df_lf[ret].mean() or 0.0)
            lf_baseline_win = float((df_lf[ret] > 0).mean() or 0.0)
            lf_baseline_hit15 = float((df_lf[mfe] >= 0.15).mean() or 0.0)

            print("\n" + "=" * 86)
            print(f"  [T5] {lf_label}  (本档全市场 rows={df_lf.height:,})")
            print(f"  本档 baseline: mfe={lf_baseline_mfe:.4f}, ret={lf_baseline_ret:.4f}, win={lf_baseline_win:.2%}, hit15={lf_baseline_hit15:.4%}")
            print("=" * 86)

            rows = []
            for fc_label, fc_cond in formation_cells:
                sub = df_lf.filter(fc_cond)
                if sub.height < 50:
                    rows.append({"cell": fc_label, "rows": sub.height, "note": "样本太少"})
                    continue
                sub_mfe = sub[mfe].to_numpy()
                sub_ret = sub[ret].to_numpy()
                n = sub.height

                ret_lifts = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    ret_lifts[b] = sub_ret[idx].mean() - lf_baseline_ret
                ret_ci = (np.quantile(ret_lifts, 0.025), np.quantile(ret_lifts, 0.975))

                mean_mfe = float(sub_mfe.mean())
                mean_ret = float(sub_ret.mean())
                hit15 = float((sub_mfe >= 0.15).mean())
                win = float((sub_ret > 0).mean())

                rows.append({
                    "cell": fc_label,
                    "rows": sub.height,
                    "mean_mfe": round(mean_mfe, 4),
                    "mean_ret": round(mean_ret, 4),
                    "ret_lift_vs_本档baseline": f"{(mean_ret - lf_baseline_ret) * 100:+.3f}pp",
                    "ret_95%CI_lift": f"[{ret_ci[0]*100:+.3f}, {ret_ci[1]*100:+.3f}]pp",
                    "ret_sig_vs_本档": "✓" if ret_ci[0] > 0 else ("⚠反向" if ret_ci[1] < 0 else "✗"),
                    "hit_15pct": f"{hit15:.4%}",
                    "win_rate": f"{win:.2%}",
                    "win_lift_vs_本档": f"{(win - lf_baseline_win) * 100:+.2f}pp",
                })
            print(pl.DataFrame(rows))

        print("\n  ───────────────────── 关键判读 ─────────────────────")
        print("  1. 看 LF1/LF2 的 R0 manual_bull baseline ret 是否大幅缩水 (例如 +2.46% → +1.5%)")
        print("     → 缩水 = baseline +2.46% 含微盘脉冲, '随便买' 论点不成立")
        print("  2. 看 LF2 严档下 R+L2+J<14 的 ret_lift_vs_本档 是否仍 ✓ 显著正")
        print("     → 仍 ✓ = 形态规则在实战可交易池里独立有 alpha")
        print("  3. 看 win_rate 列, 形态规则胜率是否稳定高于本档 baseline")
        print("     → 高于 = 即使 mean_ret 一样, 实战策略可凭胜率优势 + 止盈跑赢")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_all_h, np, pl):
    """Step T6: 信号源独立拆解 — 把 3 个候选信号各自抽出来跟全市场比, 看哪些单独就有 alpha.
    回答用户问: "J<14 单独跟全市场比, 真的强吗?"
    在 LF2 严档池子里跑, 每行都是 "在严档池里, 加上某个单独条件 vs 严档全市场 baseline".
    """
    def _run():
        _h = ACTIVE_HORIZON
        ret = f"fwd_ret_{_h}d"
        mfe = f"fwd_mfe_{_h}d"

        cond_bull = pl.col("is_manual_bull")
        cond_L2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        cond_jhook = pl.col("J") < 14
        cond_lf2 = (pl.col("market_cap_100m") >= 100) & (pl.col("amount_ma20") >= 5e7)

        rng = np.random.default_rng(42)
        N_BOOT = 1000

        for pool_label, pool_cond in [
            ("全市场池 (L0, 默认 mv 40~1500 亿)", pl.lit(True)),
            ("严档可买池 (LF2, mv≥100亿+成交≥5000万)", cond_lf2),
        ]:
            df_pool = df_all_h.filter(pool_cond)
            base_ret = float(df_pool[ret].mean() or 0.0)
            base_win = float((df_pool[ret] > 0).mean() or 0.0)
            base_hit15 = float((df_pool[mfe] >= 0.15).mean() or 0.0)
            print("\n" + "=" * 86)
            print(f"  [T6] {pool_label}, 池内全市场 baseline: ret={base_ret:.4f}, win={base_win:.2%}, hit15={base_hit15:.4%}")
            print("=" * 86)

            signals = [
                ("仅 J<14",                         cond_jhook),
                ("仅 白>黄 且 收>黄",                cond_L2),
                ("仅 多头区间",                      cond_bull),
                ("多头 + 白>黄 + 收>黄",             cond_bull & cond_L2),
                ("多头 + 白>黄 + 收>黄 + J<14",      cond_bull & cond_L2 & cond_jhook),
            ]

            rows = []
            for sig_label, sig_cond in signals:
                sub = df_pool.filter(sig_cond)
                if sub.height < 50:
                    rows.append({"signal": sig_label, "rows": sub.height, "note": "样本太少"})
                    continue
                sub_ret = sub[ret].to_numpy()
                sub_mfe = sub[mfe].to_numpy()
                n = sub.height

                lifts = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    lifts[b] = sub_ret[idx].mean() - base_ret
                ci = (np.quantile(lifts, 0.025), np.quantile(lifts, 0.975))

                rows.append({
                    "signal": sig_label,
                    "rows": sub.height,
                    "mean_ret": f"{sub_ret.mean()*100:+.3f}%",
                    "ret_lift_vs_本池baseline": f"{(sub_ret.mean() - base_ret)*100:+.3f}pp",
                    "95%CI_lift": f"[{ci[0]*100:+.3f}, {ci[1]*100:+.3f}]pp",
                    "显著": "✓正" if ci[0] > 0 else ("⚠负" if ci[1] < 0 else "✗无"),
                    "win_rate": f"{(sub_ret>0).mean()*100:.2f}%",
                    "hit_15pct": f"{(sub_mfe>=0.15).mean()*100:.2f}%",
                })
            print(pl.DataFrame(rows))

        print("\n  ───── 怎么读这张表 ─────")
        print("  '✓正' = 这个信号单独使用就比池子均值强, 且 95% 置信区间不跨 0")
        print("  '⚠负' = 这个信号单独使用反而比池子均值弱 (负 alpha)")
        print("  '✗无' = 95% 置信区间跨 0, 没法说有没有 alpha")
        print("  对比 '仅 J<14' 这一行 vs '多头+白>黄+收>黄+J<14' 这一行, 就能看出 J<14 是独立干活还是依赖别的条件")

    _run()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 待回填: 关键判读

        - [ ] L2 → L2+surge 的 mfe_lift 增量: ___________
        - [ ] L2+surge → +企稳 的 mfe_lift 增量: __________
        - [ ] +J<14 是否还有显著边际 alpha: __________
        - [ ] 教科书全规则 (e) 的 ret_sig 是否翻正 (close-to-close 真实赚钱): ___________
        - [ ] T2 表格 S1+J1 (极致缩量+J<14) 是否是局部最大值: ___________

        **决策**:

        - 路 A: 教科书全规则在统计上完全成立 → 把 stage-0 改成新定义, 重跑 backtest 看 PnL
        - 路 B: 部分成立 (例如 surge+企稳已足够, J<14 多余) → 局部更新 stage-0
        - 路 C: 全部不成立 → 教科书 B1 在 21-25 年 A 股的 alpha 已经被市场吃掉
        """
    )
    return


if __name__ == "__main__":
    app.run()
