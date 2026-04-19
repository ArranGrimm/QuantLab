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
    INCLUDE_ROTATION_KBAR_FEATURES = True

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
        # B1 Stage-0 Setup Alpha — 统计学严格证明

        **目标**: 不依赖 backtest 引擎, 用纯统计学证明 B1 stage-0 setup
        (J<=20 / WL>YL / close>YL 三条件) 在波动结构维度上有显著 alpha.

        **论点**: B1 不是"方向 alpha" (mean_ret 上跑不赢全市场), 而是
        "波动结构 alpha" (mfe / risk_adj 显著高于全市场). backtest 正收益
        是这个统计 alpha 通过止损择时的实现.

        **结构**:

        - **Cell 数据加载**: q_full + df_all (全市场, 仅 mv/list_days/ST 过滤)
        - **Step H**: 在 df_all 上层层加压, 看 J/WL/close 三条件单独/组合的 mfe/ret lift
        - **Step I**: 4 个独立统计检验
            - I1 截面百分位 t-test
            - I2 hit_15pct 二项 z-test
            - I3 月度时序 alpha t-test (时间稳定性)
            - I4 Bootstrap 95% CI

        **判读**: 4 检验全 ✓ → B1 信号在统计学上严格证明 (波动结构维度).
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
    """构造 df_all (仅 mv/list_days/ST 过滤, 不限定 seed_mid)。
    textbook_score_version='v1' 仅为底表稳定, 本 notebook 不使用 textbook 列。"""
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
    return (df_all,)


@app.cell
def _(ACTIVE_HORIZON, df_all, pl):
    """Step H: stage-0 alpha 检验 — J/WL/close 三条件的边际贡献。
    在 df_all 上注入 fwd_*_{horizon}d, 然后切 8 个层级对比 mfe/ret/risk_adj。
    返回 df_all_h 供 Step I 复用。"""
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
                f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_ret_{_h}d", f"fwd_mfe_risk_adj_{_h}d",
            ])
            .collect()
        )
        print(f"  df_all_h 准备完成, rows = {df_all_h.height:,}")

        cond_J = pl.col("J") <= 20
        cond_WL = pl.col("WL") > pl.col("YL")
        cond_close = pl.col("close_adj") > pl.col("YL")

        strata = [
            ("L0 全市场 (df_all, 仅 mv/list/ST)",       pl.lit(True)),
            ("L1 +J<=20 only (KDJ 超跌)",                cond_J),
            ("L1 +WL>YL only (中长期多头)",               cond_WL),
            ("L1 +close>YL only (个股在多空线上方)",      cond_close),
            ("L2 +J<=20 & WL>YL (= seed_loose)",         cond_J & cond_WL),
            ("L2 +J<=20 & close>YL",                     cond_J & cond_close),
            ("L2 +WL>YL & close>YL",                     cond_WL & cond_close),
            ("L3 三条件全开 (= seed_mid)",                 cond_J & cond_WL & cond_close),
        ]

        rows = []
        n_total = df_all_h.height
        baseline_mfe = float(df_all_h[f"fwd_mfe_{_h}d"].mean() or 0.0)
        baseline_risk = float(df_all_h[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
        for label, cond in strata:
            sub = df_all_h.filter(cond)
            if sub.height == 0:
                continue
            mfe = float(sub[f"fwd_mfe_{_h}d"].mean() or 0.0)
            mae = float(sub[f"fwd_mae_{_h}d"].mean() or 0.0)
            risk = float(sub[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
            hit15 = float((sub[f"fwd_mfe_{_h}d"] >= 0.15).mean() or 0.0)
            rows.append({
                "stratum": label,
                "rows": sub.height,
                "share_of_total": f"{sub.height / n_total:.2%}",
                f"mean_mfe_{_h}d": round(mfe, 4),
                f"mean_mae_{_h}d": round(mae, 4),
                f"mean_risk_adj_{_h}d": round(risk, 4),
                "hit_15pct": round(hit15, 4),
                "mfe_lift_vs_L0": f"{(mfe - baseline_mfe)*100:+.2f}pp",
                "risk_lift_vs_L0": f"{(risk - baseline_risk)*100:+.2f}pp",
            })

        print("\n" + "=" * 72)
        print(f"  [H1] stage-0 alpha 层层加压 (horizon = {_h}d, MFE-based)")
        print("=" * 72)
        print(pl.DataFrame(rows))

        h2_rows = []
        for label, cond in strata:
            sub = df_all_h.filter(cond)
            if sub.height < 100:
                continue
            top10_thr = float(sub[f"fwd_mfe_risk_adj_{_h}d"].quantile(0.90) or 0.0)
            mean_top = float(sub.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") >= top10_thr)[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
            mean_bot = float(sub.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") < top10_thr)[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
            hit15_top = float((sub.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") >= top10_thr)[f"fwd_mfe_{_h}d"] >= 0.15).mean() or 0.0)
            h2_rows.append({
                "stratum": label,
                "rows": sub.height,
                "top10_thr": round(top10_thr, 4),
                "top10_mean_risk_adj": round(mean_top, 4),
                "top10_hit_15pct": round(hit15_top, 4),
                "bot90_mean_risk_adj": round(mean_bot, 4),
            })
        print("\n" + "=" * 72)
        print(f"  [H2] 各层级内部右尾 (Top 10% risk_adj_{_h}d) 强度")
        print("=" * 72)
        print(pl.DataFrame(h2_rows))

        h3_rows = []
        baseline_ret = float(df_all_h[f"fwd_ret_{_h}d"].mean() or 0.0)
        baseline_win = float((df_all_h[f"fwd_ret_{_h}d"] > 0).mean() or 0.0)
        for label, cond in strata:
            sub = df_all_h.filter(cond)
            if sub.height == 0:
                continue
            ret_mean = float(sub[f"fwd_ret_{_h}d"].mean() or 0.0)
            ret_median = float(sub[f"fwd_ret_{_h}d"].median() or 0.0)
            ret_std = float(sub[f"fwd_ret_{_h}d"].std() or 0.0)
            win_rate = float((sub[f"fwd_ret_{_h}d"] > 0).mean() or 0.0)
            mfe_mean = float(sub[f"fwd_mfe_{_h}d"].mean() or 0.0)
            mfe_to_ret = (ret_mean / mfe_mean) if mfe_mean else 0.0
            h3_rows.append({
                "stratum": label,
                "rows": sub.height,
                f"mean_ret_{_h}d": round(ret_mean, 4),
                f"median_ret_{_h}d": round(ret_median, 4),
                f"std_ret_{_h}d": round(ret_std, 4),
                "win_rate(>0)": f"{win_rate:.2%}",
                "ret_lift_vs_L0": f"{(ret_mean - baseline_ret)*100:+.2f}pp",
                "win_lift_vs_L0": f"{(win_rate - baseline_win)*100:+.2f}pp",
                "ret/mfe": f"{mfe_to_ret:.2f}",
            })
        print("\n" + "=" * 72)
        print(f"  [H3] 真实持有收益 fwd_ret_{_h}d (close-to-close), 不是 MFE!")
        print("=" * 72)
        print(pl.DataFrame(h3_rows))

        print("\n  ───────────────────── 判读 ─────────────────────")
        print("  • [H1/H2 是 MFE-based] mfe = 持有期内最大浮盈, 天然右偏, baseline=11% 不等于赚 11%")
        print(f"  • [H3 才是真实可交易收益] mean_ret_{_h}d 是实际 close-to-close 期望")
        print("  • H3 上 B1 setup ret_lift 普遍是负的 → '无脑持有 20 天' 跑不赢全市场")
        print("  • 但 backtest 实测 B1 是正收益 → 配合止损择时, mfe 可以被部分实现")
        print("  • 因此真正的 alpha 来源是 H1/H2 显示的 mfe 上行 + risk_adj 比值优势")
        print("  • Step I 用 4 个统计检验严格证明这种 alpha 存在")

        return df_all_h

    df_all_h = _run()
    return (df_all_h,)


@app.cell
def _(ACTIVE_HORIZON, df_all_h, np, pl):
    """Step I: 统计学严格证明 B1 信号 alpha (波动结构维度)
    4 个独立检验:
      I1. 截面百分位 alpha + t-test (是否显著 > 50%)
      I2. hit_15pct 二项 z-test (是否显著 > L0)
      I3. 时序月度 alpha t-test (alpha 是否时间稳定)
      I4. Bootstrap 95% CI (lift 是否显著 != 0)
    """
    def _run():
        _h = ACTIVE_HORIZON
        from scipy import stats as _stats

        cond_J = pl.col("J") <= 20
        cond_WL = pl.col("WL") > pl.col("YL")
        cond_close = pl.col("close_adj") > pl.col("YL")
        strata = {
            "L1 +J<=20":          cond_J,
            "L1 +WL>YL":          cond_WL,
            "L1 +close>YL":       cond_close,
            "L2 +WL & close":     cond_WL & cond_close,
            "L3 seed_mid":        cond_J & cond_WL & cond_close,
        }

        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"

        # ── I1. 截面百分位 alpha (排除"跨日期混合"嫌疑) ─────────
        df_pct = df_all_h.with_columns(
            pl.col(ric).rank(method="average").over("date").alias("_rank"),
            pl.len().over("date").alias("_n_day"),
        ).with_columns(
            (pl.col("_rank") / pl.col("_n_day")).alias("_pct")
        )

        i1_rows = []
        for label, cond in strata.items():
            sub = df_pct.filter(cond)
            if sub.height == 0:
                continue
            pcts = sub["_pct"].to_numpy()
            t_stat, p_val = _stats.ttest_1samp(pcts, 0.5)
            i1_rows.append({
                "stratum": label,
                "rows": sub.height,
                "mean_pct": round(float(pcts.mean()), 4),
                "lift_vs_50%": f"{(pcts.mean() - 0.5) * 100:+.2f}pp",
                "t_stat": round(float(t_stat), 2),
                "p_value": f"{p_val:.2e}",
                "verdict": "✓ 显著" if p_val < 0.01 and pcts.mean() > 0.5 else (
                    "⚠ 显著反向" if p_val < 0.01 else "✗ 不显著"),
            })
        print("=" * 72)
        print(f"  [I1] 截面百分位 alpha (在每天截面 {ric} 中的平均排名)")
        print("=" * 72)
        print(pl.DataFrame(i1_rows))

        # ── I2. hit_15pct 二项 z-test (右尾事件概率) ──────────────
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)
        i2_rows = []
        for label, cond in strata.items():
            sub = df_all_h.filter(cond)
            if sub.height == 0:
                continue
            n_sub = sub.height
            p_sub = float((sub[mfe] >= 0.15).mean() or 0.0)
            se = (baseline_hit * (1 - baseline_hit) / n_sub) ** 0.5
            z = (p_sub - baseline_hit) / se if se > 0 else 0.0
            p_val = 2 * (1 - _stats.norm.cdf(abs(z)))
            i2_rows.append({
                "stratum": label,
                "rows": sub.height,
                "hit_15pct": f"{p_sub:.4%}",
                "baseline_hit": f"{baseline_hit:.4%}",
                "lift_pp": f"{(p_sub - baseline_hit) * 100:+.2f}pp",
                "z_stat": round(z, 2),
                "p_value": f"{p_val:.2e}",
                "verdict": "✓ 显著" if p_val < 0.01 and p_sub > baseline_hit else (
                    "⚠ 显著反向" if p_val < 0.01 else "✗ 不显著"),
            })
        print("\n" + "=" * 72)
        print(f"  [I2] hit_15pct 二项 z-test (P(fwd_mfe_{_h}d >= 15%) 是否显著高于全市场)")
        print("=" * 72)
        print(pl.DataFrame(i2_rows))

        # ── I3. 月度时序 alpha t-test (时间稳定性, 最值钱的一个检验) ─
        df_month = df_all_h.with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("_ym")
        )
        baseline_by_month = df_month.group_by("_ym").agg(
            pl.col(ric).mean().alias("L0_mean")
        )

        i3_rows = []
        for label, cond in strata.items():
            sub_month = df_month.filter(cond).group_by("_ym").agg(
                pl.col(ric).mean().alias("sub_mean"),
                pl.len().alias("sub_n"),
            )
            joined = sub_month.join(baseline_by_month, on="_ym", how="inner").filter(
                pl.col("sub_n") >= 30
            ).with_columns(
                (pl.col("sub_mean") - pl.col("L0_mean")).alias("alpha")
            )
            if joined.height == 0:
                continue
            alphas = joined["alpha"].to_numpy()
            t_stat, p_val = _stats.ttest_1samp(alphas, 0.0)
            pos_months = int((alphas > 0).sum())
            i3_rows.append({
                "stratum": label,
                "n_months": joined.height,
                "mean_monthly_alpha": round(float(alphas.mean()), 5),
                "alpha_in_pp": f"{alphas.mean() * 100:+.3f}pp",
                "pos_month_ratio": f"{pos_months/joined.height:.0%}",
                "t_stat": round(float(t_stat), 2),
                "p_value": f"{p_val:.2e}",
                "verdict": "✓ 显著且稳定" if p_val < 0.05 and alphas.mean() > 0 and pos_months/joined.height > 0.55 else (
                    "✓ 显著但不稳定" if p_val < 0.05 and alphas.mean() > 0 else "✗ 不显著"),
            })
        print("\n" + "=" * 72)
        print(f"  [I3] 月度时序 alpha t-test (月度 stratum_mean - L0_mean 是否显著 > 0)")
        print("=" * 72)
        print(pl.DataFrame(i3_rows))

        # ── I4. Bootstrap 95% CI (估计精度) ───────────────────────
        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)

        rng = np.random.default_rng(42)
        N_BOOT = 1000
        i4_rows = []
        for label, cond in strata.items():
            sub = df_all_h.filter(cond)
            if sub.height < 100:
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
            i4_rows.append({
                "stratum": label,
                "rows": sub.height,
                "mfe_lift": f"{mfe_lifts.mean()*100:+.3f}pp",
                "mfe_95%_CI": f"[{mfe_ci[0]*100:+.3f}, {mfe_ci[1]*100:+.3f}]pp",
                "mfe_sig": "✓" if mfe_ci[0] > 0 or mfe_ci[1] < 0 else "✗",
                "risk_adj_lift": f"{ric_lifts.mean()*100:+.3f}pp",
                "risk_adj_95%_CI": f"[{ric_ci[0]*100:+.3f}, {ric_ci[1]*100:+.3f}]pp",
                "ric_sig": "✓" if ric_ci[0] > 0 or ric_ci[1] < 0 else "✗",
            })
        print("\n" + "=" * 72)
        print(f"  [I4] Bootstrap 95% CI ({N_BOOT} resamples, mfe_lift / risk_adj_lift)")
        print("=" * 72)
        print(pl.DataFrame(i4_rows))

        print("\n  ───────────────────── 总判读 ─────────────────────")
        print("  四个检验全 ✓ → B1 信号在 '波动结构 alpha' 维度上严格证明")
        print("  I1 显著 → 信号能把样本筛到截面 risk_adj 排名上端 (相对 alpha)")
        print("  I2 显著 → 信号能放大 '出现 15% 以上浮盈' 的概率 (right-tail alpha)")
        print("  I3 显著 → alpha 不是来自某段时间的偶然 (时间稳定性, 最值钱的检验)")
        print("  I4 CI 不跨 0 → mean lift 在 1000 次重采样下稳定 (估计精度)")

    _run()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 关键结论 (来自 2026-04-18 实测)

        **B1 信号 (seed_mid) 在 4 个独立统计检验上全部 ✓ 通过**:

        | 检验 | seed_mid | 显著性 |
        |---|---|---|
        | I1 截面百分位 | mean_pct = 0.5239 (lift +2.39pp) | t=34.6, p<<0.001 |
        | I2 hit_15pct z-test | 25.86% vs 23.59% (lift +2.28pp) | z=22.5, p≈0 |
        | I3 月度 alpha t-test | mean +0.770pp, 71% 月份正 | t=3.09, p=0.003 |
        | I4 Bootstrap CI | mfe_lift +0.720pp, CI=[+0.65, +0.79] | 不跨 0 |

        **附带的颠覆性发现**:

        - **L2 +WL & close (无 J<=20) 在所有 4 个检验上 alpha ~ seed_mid 的 1.5x**:
            - I1 +2.62 vs +2.39, I2 +3.44 vs +2.28, I3 +0.951 vs +0.770, I4 +1.261 vs +0.720
            - 样本量 964k vs 176k (5.5x 大)
        - **J<=20 是显著反向 alpha**: 4 检验里 3 个反向显著, 月度 alpha 仅 37% 月份正
        - **暗示**: 把 J<=20 这条从 stage-0 移除, B1 backtest PnL 大概率还有 50-75% 提升空间
        """
    )
    return


if __name__ == "__main__":
    app.run()
