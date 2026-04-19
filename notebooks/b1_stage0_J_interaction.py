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
        # B1 Stage-0 J × 量价健康度 交互效应分析

        **目标**: 验证假设 — 教科书 B1 要求 J<14 是否在"量价健康"前提下成立?

        **背景** (来自 b1_stage0_alpha_proof.py):

        - L2 (WL>YL & close>YL) 是当前已证明的最大 stage-0 alpha 来源 (mfe_lift +1.26pp)
        - L3 seed_mid (= L2 + J<=20) 反而把 alpha 拉低到 +0.72pp
        - 结论: J<=20 **无条件** 看是负 alpha
        - **但**: 教科书 B1 要求量价健康 + N 型上抬 + J<14, 我们没测过"条件 alpha"

        **方法**: 在 L2 子集内做 2D 切片

        - 行: 量价健康度分位 (用 `vol_shrink_40` — 越小越无量, 教科书要求"回调点无量")
        - 列: J 分桶 (J<=14 / 14-20 / 20-30 / 30-50 / 50-80 / >80)
        - 单元格指标: rows, mean_mfe_20d, mean_risk_adj_20d, hit_15pct, mfe_lift_vs_L0

        **判读关键**:

        1. 健康度最高那一行 (vol_shrink_40 ≤ 25%), J 越小 alpha **越大** → 教科书 J<14 假设成立
        2. 健康度最低那一行 (vol_shrink_40 > 75%), J 越小 alpha **越小** → 解释为何无条件 J<=20 是负 alpha
        3. 出现单调或鞍点结构 → 给出"健康样本下的最佳 J 阈值"

        **辅助验证**: 再用 `red_green_ratio_20` (前 20 日阳线比例) 做一次 2D, 交叉验证.
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
    print(f"含 vol_shrink_40 = {'vol_shrink_40' in df_all.columns}")
    print(f"含 red_green_ratio_20 = {'red_green_ratio_20' in df_all.columns}")
    return (df_all,)


@app.cell
def _(ACTIVE_HORIZON, df_all, pl):
    """注入 fwd_*_{horizon}d + 健康度特征, 并直接过滤到 L2 = WL>YL & close>YL。
    返回 df_L2_h (后续 2D 切片的工作集) 和 df_all_h (用于全市场基线)。"""
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
            ])
            .with_columns(
                (pl.col(f"fwd_mfe_{_h}d") / (1 + pl.col(f"fwd_mae_{_h}d").abs())).alias(
                    f"fwd_mfe_risk_adj_{_h}d"
                )
            )
            .drop(future_high_names + future_low_names)
            .filter(pl.col(f"fwd_mfe_risk_adj_{_h}d").is_not_null())
            .filter(pl.col(f"fwd_ret_{_h}d").is_not_null())
            .select([
                "code", "date", "J", "WL", "YL", "close_adj",
                "vol_shrink_40", "red_green_ratio_20",
                f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_ret_{_h}d", f"fwd_mfe_risk_adj_{_h}d",
            ])
            .collect()
        )

        df_L2_h = df_all_h.filter(
            (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        )

        print(f"df_all_h = {df_all_h.height:,} rows  (全市场基线)")
        print(f"df_L2_h  = {df_L2_h.height:,} rows  (WL>YL & close>YL, 已证明的正 alpha 子集)")
        print(f"L2 share = {df_L2_h.height / df_all_h.height:.2%}")

        return df_all_h, df_L2_h

    df_all_h, df_L2_h = _run()
    return df_L2_h, df_all_h


@app.cell
def _(ACTIVE_HORIZON, df_L2_h, df_all_h, pl):
    """Step J1: 在 L2 内做 J × vol_shrink_40 (无量回调度) 2D 切片。
    vol_shrink_40 = 当日量 / 过去 40 天最大量, 越小越无量 → 越符合教科书"回调点缩量"。
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

        vs40 = df_L2_h["vol_shrink_40"].drop_nulls()
        q25 = float(vs40.quantile(0.25) or 0.0)
        q50 = float(vs40.quantile(0.50) or 0.0)
        q75 = float(vs40.quantile(0.75) or 0.0)

        def _vol_bin(col):
            return (
                pl.when(col <= q25).then(pl.lit("V1 ≤Q25 (最无量, 最健康)"))
                .when(col <= q50).then(pl.lit("V2 Q25~Q50"))
                .when(col <= q75).then(pl.lit("V3 Q50~Q75"))
                .otherwise(pl.lit("V4 >Q75 (最大量, 最不健康)"))
            )

        def _j_bin(col):
            return (
                pl.when(col <= 14).then(pl.lit("J1 J≤14"))
                .when(col <= 20).then(pl.lit("J2 14<J≤20"))
                .when(col <= 30).then(pl.lit("J3 20<J≤30"))
                .when(col <= 50).then(pl.lit("J4 30<J≤50"))
                .when(col <= 80).then(pl.lit("J5 50<J≤80"))
                .otherwise(pl.lit("J6 J>80"))
            )

        df = df_L2_h.filter(pl.col("vol_shrink_40").is_not_null()).with_columns(
            _vol_bin(pl.col("vol_shrink_40")).alias("_vbin"),
            _j_bin(pl.col("J")).alias("_jbin"),
        )

        agg = (
            df.group_by(["_vbin", "_jbin"])
            .agg(
                pl.len().alias("rows"),
                pl.col(mfe).mean().alias("mean_mfe"),
                pl.col(ric).mean().alias("mean_ric"),
                pl.col(ret).mean().alias("mean_ret"),
                (pl.col(mfe) >= 0.15).mean().alias("hit_15pct"),
            )
            .with_columns(
                (pl.col("mean_mfe") - baseline_mfe).alias("mfe_lift"),
                (pl.col("mean_ric") - baseline_ric).alias("ric_lift"),
                (pl.col("hit_15pct") - baseline_hit).alias("hit_lift"),
                (pl.col("mean_ret") - baseline_ret).alias("ret_lift"),
            )
            .sort(["_vbin", "_jbin"])
        )

        def _print_pivot(value_col, fmt, title):
            print("\n" + "=" * 78)
            print(f"  {title}")
            print(f"  baseline (L0 全市场) mfe={baseline_mfe:.4f}, ric={baseline_ric:.4f}, hit15={baseline_hit:.4f}, ret={baseline_ret:.4f}")
            print("=" * 78)
            piv = agg.pivot(values=value_col, index="_vbin", on="_jbin", sort_columns=True)
            piv_fmt = piv.select([
                pl.col("_vbin"),
                *[pl.col(c).map_elements(lambda x, _f=fmt: _f.format(x) if x is not None else "—", return_dtype=pl.Utf8) for c in piv.columns if c != "_vbin"],
            ])
            print(piv_fmt)

        rows_piv = agg.pivot(values="rows", index="_vbin", on="_jbin", sort_columns=True)
        print("\n" + "=" * 78)
        print(f"  [J1.0] 样本量 pivot (L2 内 vol_shrink_40 × J 切片, horizon={_h}d)")
        print("=" * 78)
        print(rows_piv)

        _print_pivot("mfe_lift", "{:+.4f}", "[J1.1] mfe_lift_vs_L0  ← 关键, 越大越好")
        _print_pivot("ric_lift", "{:+.4f}", "[J1.2] risk_adj_lift_vs_L0")
        _print_pivot("hit_lift", "{:+.4f}", "[J1.3] hit_15pct_lift_vs_L0")
        _print_pivot("ret_lift", "{:+.4f}", "[J1.4] ret_lift_vs_L0  (实际 close-to-close)")

        return agg

    j1_agg = _run()
    return (j1_agg,)


@app.cell
def _(ACTIVE_HORIZON, df_L2_h, df_all_h, pl):
    """Step J2: 用 red_green_ratio_20 (前 20 日阳线比例) 做交叉验证。
    阳线比例高 → 前期上涨结构健康 → 教科书 N 型上抬的"上涨段"。"""
    def _run():
        _h = ACTIVE_HORIZON
        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)

        rg = df_L2_h["red_green_ratio_20"].drop_nulls()
        q25 = float(rg.quantile(0.25) or 0.0)
        q50 = float(rg.quantile(0.50) or 0.0)
        q75 = float(rg.quantile(0.75) or 0.0)

        def _rg_bin(col):
            return (
                pl.when(col <= q25).then(pl.lit("R1 ≤Q25 (阳线少, 弱)"))
                .when(col <= q50).then(pl.lit("R2 Q25~Q50"))
                .when(col <= q75).then(pl.lit("R3 Q50~Q75"))
                .otherwise(pl.lit("R4 >Q75 (阳线多, 强)"))
            )

        def _j_bin(col):
            return (
                pl.when(col <= 14).then(pl.lit("J1 J≤14"))
                .when(col <= 20).then(pl.lit("J2 14<J≤20"))
                .when(col <= 30).then(pl.lit("J3 20<J≤30"))
                .when(col <= 50).then(pl.lit("J4 30<J≤50"))
                .when(col <= 80).then(pl.lit("J5 50<J≤80"))
                .otherwise(pl.lit("J6 J>80"))
            )

        df = df_L2_h.filter(pl.col("red_green_ratio_20").is_not_null()).with_columns(
            _rg_bin(pl.col("red_green_ratio_20")).alias("_rbin"),
            _j_bin(pl.col("J")).alias("_jbin"),
        )

        agg = (
            df.group_by(["_rbin", "_jbin"])
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
            .sort(["_rbin", "_jbin"])
        )

        def _print_pivot(value_col, fmt, title):
            print("\n" + "=" * 78)
            print(f"  {title}")
            print("=" * 78)
            piv = agg.pivot(values=value_col, index="_rbin", on="_jbin", sort_columns=True)
            piv_fmt = piv.select([
                pl.col("_rbin"),
                *[pl.col(c).map_elements(lambda x, _f=fmt: _f.format(x) if x is not None else "—", return_dtype=pl.Utf8) for c in piv.columns if c != "_rbin"],
            ])
            print(piv_fmt)

        rows_piv = agg.pivot(values="rows", index="_rbin", on="_jbin", sort_columns=True)
        print("\n" + "=" * 78)
        print(f"  [J2.0] 样本量 pivot (L2 内 red_green_ratio_20 × J 切片)")
        print("=" * 78)
        print(rows_piv)

        _print_pivot("mfe_lift", "{:+.4f}", "[J2.1] mfe_lift_vs_L0  ← 交叉验证")
        _print_pivot("ric_lift", "{:+.4f}", "[J2.2] risk_adj_lift_vs_L0")
        _print_pivot("hit_lift", "{:+.4f}", "[J2.3] hit_15pct_lift_vs_L0")

        return agg

    j2_agg = _run()
    return (j2_agg,)


@app.cell
def _(ACTIVE_HORIZON, df_L2_h, df_all_h, np, pl):
    """Step J3: 重点格子的 Bootstrap 95% CI 显著性检验。
    选 4 个关键对照:
      A. L2 全集 (基线 — 已证明 +alpha)
      B. L2 + V1 (健康) + J≤14   ← 假设最优
      C. L2 + V1 (健康) + J>80   ← 假设最差 (高位多头, 已涨上来)
      D. L2 + V4 (不健康) + J≤14 ← 假设负面 (下跌中继, 教科书反例)
    """
    def _run():
        _h = ACTIVE_HORIZON
        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"

        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)

        vs40 = df_L2_h["vol_shrink_40"].drop_nulls()
        q25 = float(vs40.quantile(0.25) or 0.0)
        q75 = float(vs40.quantile(0.75) or 0.0)

        cells = [
            ("A. L2 全集 (基线对照)",                    pl.lit(True)),
            ("B. L2 + V1健康 + J≤14 (假设最优)",          (pl.col("vol_shrink_40") <= q25) & (pl.col("J") <= 14)),
            ("C. L2 + V1健康 + 14<J≤20 (B1 教科书带)",    (pl.col("vol_shrink_40") <= q25) & (pl.col("J") > 14) & (pl.col("J") <= 20)),
            ("D. L2 + V1健康 + J>80 (假设差)",            (pl.col("vol_shrink_40") <= q25) & (pl.col("J") > 80)),
            ("E. L2 + V4不健康 + J≤14 (教科书反例)",      (pl.col("vol_shrink_40") > q75) & (pl.col("J") <= 14)),
            ("F. L2 + V4不健康 + J>80",                  (pl.col("vol_shrink_40") > q75) & (pl.col("J") > 80)),
        ]

        rng = np.random.default_rng(42)
        N_BOOT = 1000
        rows = []
        for label, cond in cells:
            sub = df_L2_h.filter(cond)
            if sub.height < 100:
                rows.append({"cell": label, "rows": sub.height, "note": "样本太少"})
                continue
            sub_mfe = sub[mfe].to_numpy()
            sub_ric = sub[ric].to_numpy()
            n = sub.height

            mfe_lifts = np.empty(N_BOOT)
            ric_lifts = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                mfe_lifts[b] = sub_mfe[idx].mean() - baseline_mfe
                ric_lifts[b] = sub_ric[idx].mean() - baseline_ric

            mfe_ci = (np.quantile(mfe_lifts, 0.025), np.quantile(mfe_lifts, 0.975))
            ric_ci = (np.quantile(ric_lifts, 0.025), np.quantile(ric_lifts, 0.975))
            mean_mfe = float(sub_mfe.mean())
            hit15 = float((sub_mfe >= 0.15).mean())

            rows.append({
                "cell": label,
                "rows": sub.height,
                "mean_mfe": round(mean_mfe, 4),
                "mfe_lift": f"{mfe_lifts.mean()*100:+.3f}pp",
                "mfe_95%_CI": f"[{mfe_ci[0]*100:+.3f}, {mfe_ci[1]*100:+.3f}]pp",
                "mfe_sig": "✓" if mfe_ci[0] > 0 else ("⚠ 反向" if mfe_ci[1] < 0 else "✗"),
                "ric_lift": f"{ric_lifts.mean()*100:+.3f}pp",
                "ric_95%_CI": f"[{ric_ci[0]*100:+.3f}, {ric_ci[1]*100:+.3f}]pp",
                "ric_sig": "✓" if ric_ci[0] > 0 else ("⚠ 反向" if ric_ci[1] < 0 else "✗"),
                "hit_15pct": f"{hit15:.4%}",
                "hit_lift_pp": f"{(hit15-baseline_hit)*100:+.2f}pp",
            })

        print("=" * 80)
        print(f"  [J3] 关键格子 Bootstrap 95% CI ({N_BOOT} resamples, horizon={_h}d)")
        print(f"  baseline mfe={baseline_mfe:.4f}, ric={baseline_ric:.4f}, hit15={baseline_hit:.4f}")
        print("=" * 80)
        print(pl.DataFrame(rows))

        print("\n  ───────────────────── 假设判读 ─────────────────────")
        print("  对比 B (健康+小J) vs C (健康+B1带): 看 J≤14 是否比 14<J≤20 显著更优")
        print("  对比 B (健康+小J) vs D (健康+大J): 看小 J 在健康样本里是否显著占优")
        print("  对比 B (健康+小J) vs E (不健康+小J): 看健康度是否扭转 J<=14 的方向")
        print("  对比 D (健康+大J) vs F (不健康+大J): 健康度对大 J 是否还有溢价")
        print("\n  若 B > C > D 单调成立 → 教科书 J<14 假设强成立, 应该改 stage-0 J 阈值")
        print("  若 B ≈ C ≈ D 平坦      → 在健康样本里 J 无意义, J 阈值可以彻底砍掉")
        print("  若 E < B 显著          → 解释了为何无条件 J<=20 是负 alpha (被 E 拖累)")

    _run()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 待回填: 跑完后的关键发现

        - [ ] 健康样本 (V1 行) 内 J 单调性: ___________
        - [ ] 不健康样本 (V4 行) 内 J 单调性: __________
        - [ ] B (健康+J≤14) 是否显著优于 C (健康+B1 带): __________
        - [ ] E (不健康+J≤14) 是否显著为负: ___________
        - [ ] red_green_ratio_20 交叉验证是否一致: ___________

        **决策**:

        - 选项 A: 砍掉 J<=20 (用 L2 替代 seed_mid)
        - 选项 B: 改 J<=14 + 加量价健康过滤 (vol_shrink_40 ≤ Q25)
        - 选项 C: 保留 J<=20, 但加量价健康过滤
        - 选项 D: 看数据再说
        """
    )
    return


if __name__ == "__main__":
    app.run()
