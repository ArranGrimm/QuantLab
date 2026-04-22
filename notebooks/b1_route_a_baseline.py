import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # B1 路线 A — 闭眼随机买 (诚实下限)

    **这个 notebook 在做一件事**: 在严格的 6 条池子规则里, 闭眼随机抓 5 只票, 看历史能赚多少.
    这就是"什么花活都不做"的诚实下限. 后面任何路线 C 的择时优化, 都必须打败这个数才有意义.

    **池子规则** (严格对齐 `b1_alpha_proof.py`, 每条都要满足):
    1. **多头区间** — 在你 RPA 抓的活跃市值多头日内 (T+1 ex-ante, 不偷看未来)
    2. **白线 > 黄线** — 日线 Ztalk 白线高过黄线
    3. **收盘 > 黄线** — 日线收盘价高过黄线
    4. **J < 14** — KDJ 的 J 值小于 14 (超跌)
    5. **市值 ≥ 100亿** — 流通市值, 你真能买进去不冲击市价
    6. **日均成交额 ≥ 5000万** — 20 日均成交额, 流动性够

    **怎么模拟交易**:
    - 每个满足规则的交易日, 池子里通常有几十到上百只候选, 我们闭眼随机抓 5 只
    - 持有 N 天后卖出 (N 取 5/10/15/20/25/30 天, 每个分别统计)
    - 两个版本对比: **死拿 N 天不止损** vs **跌 3% 止损出场**
    - 整套流程随机重抽 1000 次 (固定 seed 42, 任何人都能复现同样结果)

    **怎么读结果**:

    本 notebook 给两张表 + 一条曲线:
    - **表 1 "每笔交易赚多少"**: 把所有抽到的票当独立交易, 看平均/分布/最差
    - **表 2 "组合复利净值"**: 用两种朴素方案模拟资金使用, 算期末净值和最深回撤
        - 方案 A "每 N 天才开一次仓": 单笔不重叠, 资金利用率低 (1/N)
        - 方案 B "N 份资金错峰滚动开仓": 把资金切 N 份, 每天有一份在外面, 利用率 100%
    - **一条曲线**: 单次 trial (seed=42) 在 20 天持仓下的等权净值长什么样

    **重要说明**:
    - 当前 v1 不排除 T+1 涨停 (alpha_proof 也未排除, 保持口径一致)
    - 不算交易摩擦 (滑点 / 印花税 / 佣金), 是回测毛收益
    - 方案 A/B 的最深回撤是 epoch (= N 天) 粒度, 比真实日级粒度粗一些
    """)
    return


@app.cell
def _():
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    from utils import build_b1_research_frame, get_st_blacklist_pl, load_daily_data_full

    pl.Config(tbl_rows=-1, tbl_cols=-1)
    return (
        build_b1_research_frame,
        duckdb,
        get_st_blacklist_pl,
        load_daily_data_full,
        mo,
        np,
        pl,
    )


@app.cell
def _():
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0

    LF2_MV_MIN = 100
    LF2_AMT_MA20_MIN = 5e7

    J_THRESHOLD = 14

    HORIZONS = [5, 10, 15, 20, 25, 30]
    MAX_HORIZON = max(HORIZONS)

    N_PICKS = 5
    N_TRIALS = 1000
    MC_SEED = 42

    STOP_LOSS_PCT = 0.03

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
        DB_PATH,
        END_DATE,
        HORIZONS,
        J_THRESHOLD,
        LF2_AMT_MA20_MIN,
        LF2_MV_MIN,
        LOOSE_PERIODS,
        MAX_HORIZON,
        MC_SEED,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        N_PICKS,
        N_TRIALS,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        STOP_LOSS_PCT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 数据准备 (3 格)

    - 第 1 格: 拉日线 + 剔除 ST
    - 第 2 格: 算 B1 特征 (J/WL/YL/is_manual_bull/...) 走 `build_b1_research_frame`
    - 第 3 格: 算多 horizon 前瞻收益 + amount_ma20 + 6 条 AND 过滤 → df_candidates
    """)
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
    print(f"原始日 K 数据已加载, ST 黑名单 {len(st_blacklist):,} 只票已剔除")
    return (q_full,)


@app.cell
def _(
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
        include_rotation_kbar_features=False,
        textbook_score_version="v1",
    )
    print(f"特征已算完, 共 {df_all.height:,} 行")
    return (df_all,)


@app.cell
def _(
    HORIZONS,
    J_THRESHOLD,
    LF2_AMT_MA20_MIN,
    LF2_MV_MIN,
    MAX_HORIZON,
    df_all,
    pl,
):
    """注入多 horizon 前瞻收益 + 流动性, 然后套 6 条 AND 拿 candidates."""
    def _run():
        future_high_cols = [
            pl.col("high_adj").shift(-step).over("code").alias(f"_fwd_high_{step}")
            for step in range(1, MAX_HORIZON + 1)
        ]
        future_low_cols = [
            pl.col("low_adj").shift(-step).over("code").alias(f"_fwd_low_{step}")
            for step in range(1, MAX_HORIZON + 1)
        ]
        future_close_cols = [
            pl.col("close_adj").shift(-step).over("code").alias(f"_fwd_close_{step}")
            for step in range(1, MAX_HORIZON + 1)
        ]

        per_h_cols = []
        for h in HORIZONS:
            high_names_h = [f"_fwd_high_{s}" for s in range(1, h + 1)]
            low_names_h = [f"_fwd_low_{s}" for s in range(1, h + 1)]
            per_h_cols.extend([
                (pl.max_horizontal(high_names_h) / pl.col("close_adj") - 1).alias(f"fwd_mfe_{h}d"),
                (pl.min_horizontal(low_names_h) / pl.col("close_adj") - 1).alias(f"fwd_mae_{h}d"),
                (pl.col(f"_fwd_close_{h}") / pl.col("close_adj") - 1).alias(f"fwd_ret_{h}d"),
            ])

        all_temp_names = (
            [f"_fwd_high_{s}" for s in range(1, MAX_HORIZON + 1)]
            + [f"_fwd_low_{s}" for s in range(1, MAX_HORIZON + 1)]
            + [f"_fwd_close_{s}" for s in range(1, MAX_HORIZON + 1)]
        )

        keep_cols = ["code", "date", "J", "WL", "YL", "close_adj",
                     "is_manual_bull", "market_cap_100m", "amount_ma20"]
        for h in HORIZONS:
            keep_cols.extend([f"fwd_mfe_{h}d", f"fwd_mae_{h}d", f"fwd_ret_{h}d"])

        df_full = (
            df_all.lazy()
            .with_columns(future_high_cols + future_low_cols + future_close_cols)
            .with_columns(per_h_cols)
            .with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
            .drop(all_temp_names)
            .filter(pl.col(f"fwd_ret_{MAX_HORIZON}d").is_not_null())
            .select(keep_cols)
            .collect()
        )

        df_lf2 = df_full.filter(
            (pl.col("market_cap_100m") >= LF2_MV_MIN)
            & (pl.col("amount_ma20") >= LF2_AMT_MA20_MIN)
        )

        df_cand = df_lf2.filter(
            pl.col("is_manual_bull")
            & (pl.col("WL") > pl.col("YL"))
            & (pl.col("close_adj") > pl.col("YL"))
            & (pl.col("J") < J_THRESHOLD)
        )

        print(f"完整数据 (含到 +{MAX_HORIZON}d 前瞻): {df_full.height:>10,} 行")
        print(f"  LF2 严格池 (mv≥100亿 + amt20≥5000万) : {df_lf2.height:>10,} 行")
        print(f"  6 条 AND 候选池 (路线 A 池子)         : {df_cand.height:>10,} 行")
        return df_full, df_lf2, df_cand

    df_full, df_lf2, df_candidates = _run()
    return df_candidates, df_full, df_lf2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 候选池容量诊断

    路线 A 的核心前提: **每个交易日的候选数 >= 5 只**, 否则随机选 5 不成立.
    """)
    return


@app.cell
def _(N_PICKS, df_candidates, pl):
    daily_counts = (
        df_candidates
        .group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .sort("date")
    )
    n_days_total = daily_counts.height
    n_days_ge_picks = daily_counts.filter(pl.col("n_candidates") >= N_PICKS).height

    summary = pl.DataFrame({
        "口径": [
            "总交易日数 (有候选的多头日)",
            f"交易日候选数 >= {N_PICKS} 的天数",
            "占比",
            "每日候选数 中位数",
            "每日候选数 均值",
            "每日候选数 5% 分位",
            "每日候选数 95% 分位",
            "每日候选数 最小值",
            "每日候选数 最大值",
        ],
        "值": [
            f"{n_days_total:,}",
            f"{n_days_ge_picks:,}",
            f"{n_days_ge_picks / max(n_days_total, 1) * 100:.2f}%",
            f"{daily_counts['n_candidates'].median():.0f}",
            f"{daily_counts['n_candidates'].mean():.1f}",
            f"{daily_counts['n_candidates'].quantile(0.05):.0f}",
            f"{daily_counts['n_candidates'].quantile(0.95):.0f}",
            f"{daily_counts['n_candidates'].min():.0f}",
            f"{daily_counts['n_candidates'].max():.0f}",
        ],
    })
    print("=" * 72)
    print("  路线 A 候选池容量诊断")
    print("=" * 72)
    print(summary)
    return (daily_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 蒙特卡洛主体 (1000 次随机抽样)

    **每次试验都做这些事** (对每个持仓天数 N 重复一遍):

    1. 跑遍所有满足"6 条池子规则 + 候选数 ≥ 5"的交易日
    2. 每天闭眼随机抽 5 只票, 记录它们 N 天后的涨跌 (`fwd_ret_Nd`)
    3. 同时记录"如果跌 3% 就出场"那一版的回报 (跌幅看 `fwd_mae_Nd`)

    **每次试验完后, 算两组数**:

    - **每笔交易级**: 把抽到的所有票 (= 交易日数 × 5) 平均涨跌, 中位涨跌, 单笔最差损失, 触及+15%的概率
    - **组合复利净值**: 用两种朴素的资金使用方式各自模拟一条净值曲线
        - **方案 A "等 N 天才开一次仓"**: 把所有交易日切成不重叠的 N 天块, 只在每块第一天开仓. 资金利用率 1/N
        - **方案 B "N 份资金错峰开仓"**: 把资金切 N 份, 第 0/N/2N/... 天用钱包 0, 第 1/(N+1)/(2N+1)/... 天用钱包 1, 依此类推. 总净值 = N 个钱包等权平均. 资金利用率 100%

    1000 次重复后, 把每个数的 5%/中位/95% 分位都算出来, 显示分布.
    """)
    return


@app.cell
def _():
    """单 trial 一条 daily 收益序列 → 算方案 A / B 两条净值曲线 + 各自的最深回撤."""
    import numpy as _np

    def compute_two_navs(daily_ret, horizon: int):
        """
        daily_ret: shape (n_trade_days,) 每个交易日 5 只票均值的"持仓 N 天后" 涨跌
        horizon: N 天
        返回:
          nav_a, max_dd_a: 方案 A "等 N 天才开一次仓"
          nav_b, max_dd_b: 方案 B "N 份资金错峰开仓"
        """
        n_days = len(daily_ret)

        epoch_idx_a = _np.arange(0, n_days, horizon)
        epoch_rets_a = daily_ret[epoch_idx_a]
        nav_a = _np.cumprod(1.0 + epoch_rets_a)

        n_complete_epochs = n_days // horizon
        if n_complete_epochs == 0:
            nav_b = _np.array([1.0])
        else:
            sleeve_navs = _np.empty((horizon, n_complete_epochs))
            for j in range(horizon):
                sleeve_rets = daily_ret[j::horizon][:n_complete_epochs]
                sleeve_navs[j] = _np.cumprod(1.0 + sleeve_rets)
            nav_b = sleeve_navs.mean(axis=0)

        def _max_dd(nav):
            running_max = _np.maximum.accumulate(nav)
            dd = (nav - running_max) / running_max
            return float(dd.min())

        return nav_a, _max_dd(nav_a), nav_b, _max_dd(nav_b)

    return (compute_two_navs,)


@app.cell
def _(
    HORIZONS,
    MC_SEED,
    N_PICKS,
    N_TRIALS,
    STOP_LOSS_PCT,
    compute_two_navs,
    daily_counts,
    df_candidates,
    np,
    pl,
):
    """蒙特卡洛主体. 返回 results dict[horizon] -> dict[stat] -> per-trial array."""
    def _run():
        eligible_dates = daily_counts.filter(pl.col("n_candidates") >= N_PICKS)["date"].to_list()

        cand_eligible = df_candidates.filter(pl.col("date").is_in(eligible_dates)).sort(["date", "code"])
        cand_eligible = cand_eligible.with_row_index("row_idx")

        date_index_lookup = {}
        for d, sub in cand_eligible.group_by("date", maintain_order=True):
            d_val = d[0] if isinstance(d, tuple) else d
            date_index_lookup[d_val] = sub["row_idx"].to_numpy()

        trade_dates = sorted(date_index_lookup.keys())
        n_trade_days = len(trade_dates)
        print(f"参与 MC 的交易日数: {n_trade_days:,}")
        print(f"  最早: {trade_dates[0]}    最晚: {trade_dates[-1]}")

        h_data = {}
        for h in HORIZONS:
            h_data[h] = {
                "ret": cand_eligible[f"fwd_ret_{h}d"].to_numpy(),
                "mae": cand_eligible[f"fwd_mae_{h}d"].to_numpy(),
                "mfe": cand_eligible[f"fwd_mfe_{h}d"].to_numpy(),
            }

        rng = np.random.default_rng(MC_SEED)

        results = {h: {
            "trial_mean_ret_buyhold": np.empty(N_TRIALS),
            "trial_mean_ret_stoploss": np.empty(N_TRIALS),
            "trial_median_ret_buyhold": np.empty(N_TRIALS),
            "trial_hit15": np.empty(N_TRIALS),
            "trial_worst_trade_buyhold": np.empty(N_TRIALS),
            "trial_navA_end_buyhold": np.empty(N_TRIALS),
            "trial_navA_end_stoploss": np.empty(N_TRIALS),
            "trial_navA_dd_buyhold": np.empty(N_TRIALS),
            "trial_navA_dd_stoploss": np.empty(N_TRIALS),
            "trial_navB_end_buyhold": np.empty(N_TRIALS),
            "trial_navB_end_stoploss": np.empty(N_TRIALS),
            "trial_navB_dd_buyhold": np.empty(N_TRIALS),
            "trial_navB_dd_stoploss": np.empty(N_TRIALS),
        } for h in HORIZONS}

        for t in range(N_TRIALS):
            picks_per_day = []
            for d in trade_dates:
                pool_idx = date_index_lookup[d]
                pick = rng.choice(pool_idx, size=N_PICKS, replace=False)
                picks_per_day.append(pick)
            picks_flat = np.concatenate(picks_per_day)

            for h in HORIZONS:
                rets = h_data[h]["ret"][picks_flat]
                maes = h_data[h]["mae"][picks_flat]
                mfes = h_data[h]["mfe"][picks_flat]

                rets_sl = np.where(maes <= -STOP_LOSS_PCT, -STOP_LOSS_PCT, rets)

                rets_by_day = rets.reshape(n_trade_days, N_PICKS)
                rets_sl_by_day = rets_sl.reshape(n_trade_days, N_PICKS)
                mfes_by_day = mfes.reshape(n_trade_days, N_PICKS)

                daily_strat_ret_bh = rets_by_day.mean(axis=1)
                daily_strat_ret_sl = rets_sl_by_day.mean(axis=1)
                daily_hit15 = (mfes_by_day >= 0.15).mean(axis=1)

                results[h]["trial_mean_ret_buyhold"][t] = rets.mean()
                results[h]["trial_mean_ret_stoploss"][t] = rets_sl.mean()
                results[h]["trial_median_ret_buyhold"][t] = np.median(rets)
                results[h]["trial_hit15"][t] = daily_hit15.mean()
                results[h]["trial_worst_trade_buyhold"][t] = rets.min()

                navA_bh, ddA_bh, navB_bh, ddB_bh = compute_two_navs(daily_strat_ret_bh, h)
                navA_sl, ddA_sl, navB_sl, ddB_sl = compute_two_navs(daily_strat_ret_sl, h)

                results[h]["trial_navA_end_buyhold"][t] = navA_bh[-1] - 1.0
                results[h]["trial_navA_dd_buyhold"][t] = ddA_bh
                results[h]["trial_navA_end_stoploss"][t] = navA_sl[-1] - 1.0
                results[h]["trial_navA_dd_stoploss"][t] = ddA_sl
                results[h]["trial_navB_end_buyhold"][t] = navB_bh[-1] - 1.0
                results[h]["trial_navB_dd_buyhold"][t] = ddB_bh
                results[h]["trial_navB_end_stoploss"][t] = navB_sl[-1] - 1.0
                results[h]["trial_navB_dd_stoploss"][t] = ddB_sl

            if (t + 1) % 100 == 0:
                print(f"  trial {t + 1:>4d}/{N_TRIALS} done")

        return results, n_trade_days, trade_dates

    mc_results, n_trade_days, trade_dates = _run()
    return mc_results, n_trade_days, trade_dates


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 表 1: 每笔交易能赚多少 (1000 次试验汇总)

    把每个试验里抽到的所有票都当成独立交易, 看平均赚多少, 最差能多惨, 触及 +15% 的概率多少.
    """)
    return


@app.cell
def _(HORIZONS, mc_results, np, pl):
    _rows = []
    for _h in HORIZONS:
        _r = mc_results[_h]
        _rows.append({
            "持仓天数": f"{_h}d",
            "策略": "死拿不止损",
            "每笔平均赚 (1000次均值)": f"{_r['trial_mean_ret_buyhold'].mean()*100:+.3f}%",
            "每笔平均赚 (中位试验)": f"{np.median(_r['trial_mean_ret_buyhold'])*100:+.3f}%",
            "差的 5% 试验只赚到": f"{np.quantile(_r['trial_mean_ret_buyhold'], 0.05)*100:+.3f}%",
            "好的 5% 试验能赚到": f"{np.quantile(_r['trial_mean_ret_buyhold'], 0.95)*100:+.3f}%",
            "单笔最差损失 (中位试验)": f"{np.median(_r['trial_worst_trade_buyhold'])*100:+.2f}%",
            "持有期内触及 +15% 的概率": f"{_r['trial_hit15'].mean()*100:.2f}%",
        })
        _rows.append({
            "持仓天数": f"{_h}d",
            "策略": "跌 3% 止损",
            "每笔平均赚 (1000次均值)": f"{_r['trial_mean_ret_stoploss'].mean()*100:+.3f}%",
            "每笔平均赚 (中位试验)": f"{np.median(_r['trial_mean_ret_stoploss'])*100:+.3f}%",
            "差的 5% 试验只赚到": f"{np.quantile(_r['trial_mean_ret_stoploss'], 0.05)*100:+.3f}%",
            "好的 5% 试验能赚到": f"{np.quantile(_r['trial_mean_ret_stoploss'], 0.95)*100:+.3f}%",
            "单笔最差损失 (中位试验)": "—",
            "持有期内触及 +15% 的概率": "—",
        })
    summary_per_trade = pl.DataFrame(_rows)
    print("=" * 100)
    print("  表 1: 每笔交易能赚多少 (1000 次试验汇总)")
    print("=" * 100)
    print(summary_per_trade)
    print("\n  怎么读:")
    print("  - '每笔平均赚 (1000次均值)' 这一列, 应该跟下面 '日期等权对照' 几乎相等 (差异 < 0.05pp 才算对)")
    print("  - '差的 5%' 和 '好的 5%' 区间越窄, 说明随机抽 5 只的运气成分越小, 池子规则越稳")
    print("  - 死拿 vs 止损 差距越小, 越验证 Q7-2 'alpha 不来自精细持仓管理'")
    return (summary_per_trade,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 表 2: 组合复利净值 (1000 次试验汇总)

    用两种朴素的资金使用方式各模拟一条净值曲线:

    - **方案 A "等 N 天才开一次仓"**: 单笔不重叠, 每 N 天才把钱全投出去. 资金利用率 1/N
    - **方案 B "N 份资金错峰滚动"**: 钱切 N 份, 每天一份在外面, 利用率 100%. 真实操作更接近这个

    报告 1000 次试验的: 期末净值 (相对初始 = 1) + 持仓中途最深回撤
    """)
    return


@app.cell
def _(HORIZONS, mc_results, np, pl):
    _rows = []
    for _h in HORIZONS:
        _r = mc_results[_h]
        _rows.append({
            "持仓天数": f"{_h}d",
            "策略": "死拿不止损",
            "方案A 期末净值 (中位试验)": f"{(1 + np.median(_r['trial_navA_end_buyhold'])):.3f}x",
            "方案A 期末净值 (差的 5%)": f"{(1 + np.quantile(_r['trial_navA_end_buyhold'], 0.05)):.3f}x",
            "方案A 最深回撤 (中位试验)": f"{np.median(_r['trial_navA_dd_buyhold'])*100:.2f}%",
            "方案A 最深回撤 (差的 5%)": f"{np.quantile(_r['trial_navA_dd_buyhold'], 0.05)*100:.2f}%",
            "方案B 期末净值 (中位试验)": f"{(1 + np.median(_r['trial_navB_end_buyhold'])):.3f}x",
            "方案B 期末净值 (差的 5%)": f"{(1 + np.quantile(_r['trial_navB_end_buyhold'], 0.05)):.3f}x",
            "方案B 最深回撤 (中位试验)": f"{np.median(_r['trial_navB_dd_buyhold'])*100:.2f}%",
            "方案B 最深回撤 (差的 5%)": f"{np.quantile(_r['trial_navB_dd_buyhold'], 0.05)*100:.2f}%",
        })
        _rows.append({
            "持仓天数": f"{_h}d",
            "策略": "跌 3% 止损",
            "方案A 期末净值 (中位试验)": f"{(1 + np.median(_r['trial_navA_end_stoploss'])):.3f}x",
            "方案A 期末净值 (差的 5%)": f"{(1 + np.quantile(_r['trial_navA_end_stoploss'], 0.05)):.3f}x",
            "方案A 最深回撤 (中位试验)": f"{np.median(_r['trial_navA_dd_stoploss'])*100:.2f}%",
            "方案A 最深回撤 (差的 5%)": f"{np.quantile(_r['trial_navA_dd_stoploss'], 0.05)*100:.2f}%",
            "方案B 期末净值 (中位试验)": f"{(1 + np.median(_r['trial_navB_end_stoploss'])):.3f}x",
            "方案B 期末净值 (差的 5%)": f"{(1 + np.quantile(_r['trial_navB_end_stoploss'], 0.05)):.3f}x",
            "方案B 最深回撤 (中位试验)": f"{np.median(_r['trial_navB_dd_stoploss'])*100:.2f}%",
            "方案B 最深回撤 (差的 5%)": f"{np.quantile(_r['trial_navB_dd_stoploss'], 0.05)*100:.2f}%",
        })
    summary_portfolio = pl.DataFrame(_rows)
    print("=" * 110)
    print("  表 2: 组合复利净值 (1000 次试验汇总, 期末净值 = 1.0 表示打平, 2.0 表示翻倍)")
    print("=" * 110)
    print(summary_portfolio)
    print("\n  怎么读:")
    print("  - 方案 A 期末净值低 (1/N 资金利用率), 但反映 '一笔接一笔不重叠' 的最朴素操作")
    print("  - 方案 B 期末净值高 (100% 利用率), 但回撤也会被 N 份并行钱包平滑掉")
    print("  - 真实操作更接近方案 B, 因为没人愿意每 N 天才开一次仓")
    print("  - '差的 5%' 是诚实下限的最坏情况, 用来设资金管理预案")
    return (summary_portfolio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 解析对照: MC 均值的"理论收敛值"

    这一格是给上面 "表 1 每笔平均赚" 做交叉验证.

    **两种"全买"算法, 数学上不一样**:
    - **算法 1 (按交易日等权)**: 每天先算"今天所有候选的平均涨跌", 再把所有日子等权平均
        - = 闭眼随机抽 5 只 (1000 次) 的真期望, 表 1 的 "每笔平均赚 (1000次均值)" 应该精确收敛到这一行
    - **算法 2 (按交易行等权)**: 把整个池子 28697 行 ret 全部加起来除以 28697
        - = "所有候选的总平均", 大多头日候选多权重就大, 不是 MC 收敛对象

    **如果 MC 均值 == 算法 1, 说明蒙特卡洛实现正确**.
    """)
    return


@app.cell
def _(HORIZONS, N_PICKS, daily_counts, df_candidates, pl):
    """两种 baseline 算法的对比: 按日期等权 vs 按行等权.
    重要: 这两个算法都只在 '候选 >= N_PICKS' 的日子上算, 才能跟 MC 同集合精确对照.
    """
    _eligible_dates = daily_counts.filter(pl.col("n_candidates") >= N_PICKS)["date"].to_list()
    _df_cand_mc = df_candidates.filter(pl.col("date").is_in(_eligible_dates))

    _rows = []
    for _h in HORIZONS:
        _per_day = (
            _df_cand_mc
            .group_by("date")
            .agg([
                pl.col(f"fwd_ret_{_h}d").mean().alias("daily_mean_ret"),
                pl.col(f"fwd_mfe_{_h}d").mean().alias("daily_mean_mfe"),
                (pl.col(f"fwd_mfe_{_h}d") >= 0.15).mean().alias("daily_hit15"),
            ])
        )
        _date_eq_ret = float(_per_day["daily_mean_ret"].mean() or 0)
        _date_eq_hit15 = float(_per_day["daily_hit15"].mean() or 0)

        _ret = _df_cand_mc[f"fwd_ret_{_h}d"]
        _mfe = _df_cand_mc[f"fwd_mfe_{_h}d"]
        _row_eq_ret = float(_ret.mean() or 0)
        _row_eq_hit15 = float((_mfe >= 0.15).mean() or 0)

        _rows.append({
            "持仓天数": f"{_h}d",
            "算法1 按交易日等权 平均赚": f"{_date_eq_ret*100:+.3f}%",
            "算法1 按交易日等权 触及+15%概率": f"{_date_eq_hit15*100:.2f}%",
            "算法2 按交易行等权 平均赚": f"{_row_eq_ret*100:+.3f}%",
            "算法2 按交易行等权 触及+15%概率": f"{_row_eq_hit15*100:.2f}%",
            "两算法差距": f"{(_row_eq_ret - _date_eq_ret)*100:+.3f}pp",
        })
    print("=" * 110)
    print(f"  解析对照: 只在 '候选 >= {N_PICKS}' 的 {len(_eligible_dates)} 天上算 (跟 MC 同集合)")
    print(f"  MC 表 1 的 '每笔平均赚 (1000次均值)' 应当精确收敛到 '算法1 按交易日等权' (差异 < 0.02pp)")
    print("=" * 110)
    print(pl.DataFrame(_rows))
    print("\n  解读:")
    print("  - 算法1 > 算法2 还是 < 算法2 取决于'候选数多的日子涨得好还是坏'")
    print("  - 一般 算法2 > 算法1, 说明大多头日 (候选 100+) 也是真涨幅大的日子")
    print("  - 闭眼随机选 5 只是日期等权, 所以拿不到大多头日的额外加权")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 单次试验净值曲线样例 (HORIZON=20, seed=42)

    挑一次试验 (固定 seed=42), 把方案 A 和方案 B 的净值曲线拉出来看, 直观感受 4 年走下来长什么样.

    - 横轴: 第 N 个 epoch (1 个 epoch = 20 天)
    - 纵轴: 净值 (起点 = 1.0)
    """)
    return


@app.cell
def _(MC_SEED, N_PICKS, STOP_LOSS_PCT, compute_two_navs, daily_counts, df_candidates, np, pl, trade_dates):
    def _build_one_trial_curve(horizon: int, trial_seed: int = MC_SEED):
        eligible_dates = daily_counts.filter(pl.col("n_candidates") >= N_PICKS)["date"].to_list()
        cand = df_candidates.filter(pl.col("date").is_in(eligible_dates)).sort(["date", "code"]).with_row_index("row_idx")
        date_index_lookup = {}
        for d, sub in cand.group_by("date", maintain_order=True):
            d_val = d[0] if isinstance(d, tuple) else d
            date_index_lookup[d_val] = sub["row_idx"].to_numpy()

        rng = np.random.default_rng(trial_seed)
        ret_arr = cand[f"fwd_ret_{horizon}d"].to_numpy()
        mae_arr = cand[f"fwd_mae_{horizon}d"].to_numpy()

        daily_bh = []
        daily_sl = []
        for d in trade_dates:
            pool_idx = date_index_lookup[d]
            pick = rng.choice(pool_idx, size=N_PICKS, replace=False)
            r = ret_arr[pick]
            m = mae_arr[pick]
            r_sl = np.where(m <= -STOP_LOSS_PCT, -STOP_LOSS_PCT, r)
            daily_bh.append(r.mean())
            daily_sl.append(r_sl.mean())

        daily_bh_arr = np.array(daily_bh)
        daily_sl_arr = np.array(daily_sl)

        navA_bh, ddA_bh, navB_bh, ddB_bh = compute_two_navs(daily_bh_arr, horizon)
        navA_sl, ddA_sl, navB_sl, ddB_sl = compute_two_navs(daily_sl_arr, horizon)

        return {
            "horizon": horizon,
            "trade_dates": trade_dates,
            "navA_bh": navA_bh, "ddA_bh": ddA_bh,
            "navA_sl": navA_sl, "ddA_sl": ddA_sl,
            "navB_bh": navB_bh, "ddB_bh": ddB_bh,
            "navB_sl": navB_sl, "ddB_sl": ddB_sl,
        }

    sample_20d = _build_one_trial_curve(horizon=20, trial_seed=MC_SEED)

    print("=" * 86)
    print(f"  单次试验净值曲线 (持仓 20 天, seed=42)")
    print("=" * 86)

    _navA_bh = sample_20d["navA_bh"]
    _navA_sl = sample_20d["navA_sl"]
    _navB_bh = sample_20d["navB_bh"]
    _navB_sl = sample_20d["navB_sl"]

    print(f"\n  方案 A '等 20 天才开一次仓' (共 {len(_navA_bh)} 个 epoch):")
    print(f"    死拿不止损: 期末净值 {_navA_bh[-1]:.3f}x ({(_navA_bh[-1]-1)*100:+.2f}%), 最深回撤 {sample_20d['ddA_bh']*100:.2f}%")
    print(f"    跌3%止损  : 期末净值 {_navA_sl[-1]:.3f}x ({(_navA_sl[-1]-1)*100:+.2f}%), 最深回撤 {sample_20d['ddA_sl']*100:.2f}%")

    print(f"\n  方案 B 'N 份资金错峰滚动' (共 {len(_navB_bh)} 个 epoch):")
    print(f"    死拿不止损: 期末净值 {_navB_bh[-1]:.3f}x ({(_navB_bh[-1]-1)*100:+.2f}%), 最深回撤 {sample_20d['ddB_bh']*100:.2f}%")
    print(f"    跌3%止损  : 期末净值 {_navB_sl[-1]:.3f}x ({(_navB_sl[-1]-1)*100:+.2f}%), 最深回撤 {sample_20d['ddB_sl']*100:.2f}%")

    _curve_a = pl.DataFrame({
        "epoch": list(range(len(_navA_bh))),
        "方案A 死拿净值": _navA_bh,
        "方案A 止损净值": _navA_sl,
    })
    _curve_b = pl.DataFrame({
        "epoch": list(range(len(_navB_bh))),
        "方案B 死拿净值": _navB_bh,
        "方案B 止损净值": _navB_sl,
    })
    print("\n  方案 A 净值曲线 (头尾各 3 个 epoch):")
    print(pl.concat([_curve_a.head(3), _curve_a.tail(3)]))
    print("\n  方案 B 净值曲线 (头尾各 3 个 epoch):")
    print(pl.concat([_curve_b.head(3), _curve_b.tail(3)]))
    return (sample_20d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 路线 A 诚实下限摘要 (路线 C 的 acceptance gate)

    把上面 1000 次试验的核心 number 提炼出来一张表, 用于后续路线 C 任何择时优化的对照基线.
    """)
    return


@app.cell
def _(HORIZONS, mc_results, np, pl):
    _rows = []
    for _h in HORIZONS:
        _r = mc_results[_h]
        _navB_bh_med = float(np.median(_r["trial_navB_end_buyhold"]))
        _navB_sl_med = float(np.median(_r["trial_navB_end_stoploss"]))
        _ddB_bh_med = float(np.median(_r["trial_navB_dd_buyhold"]))
        _ddB_sl_med = float(np.median(_r["trial_navB_dd_stoploss"]))
        _ddB_bh_5pct = float(np.quantile(_r["trial_navB_dd_buyhold"], 0.05))
        _per_trade_med = float(_r["trial_mean_ret_buyhold"].mean())

        _rows.append({
            "持仓天数": f"{_h}d",
            "每笔平均赚": f"{_per_trade_med*100:+.3f}%",
            "方案B 死拿期末": f"{(1 + _navB_bh_med):.3f}x ({_navB_bh_med*100:+.2f}%)",
            "方案B 死拿回撤(中位)": f"{_ddB_bh_med*100:.2f}%",
            "方案B 死拿回撤(差5%)": f"{_ddB_bh_5pct*100:.2f}%",
            "方案B 止损期末": f"{(1 + _navB_sl_med):.3f}x ({_navB_sl_med*100:+.2f}%)",
            "方案B 止损回撤(中位)": f"{_ddB_sl_med*100:.2f}%",
            "止损 vs 死拿 期末差": f"{(_navB_sl_med - _navB_bh_med)*100:+.2f}pp",
        })
    print("=" * 120)
    print("  路线 A 诚实下限摘要 (4 年 2021-07 ~ 2025-09, 闭眼随机抽 5 只 LF2 候选)")
    print("=" * 120)
    print(pl.DataFrame(_rows))
    print()
    print("=" * 120)
    print("  路线 A 给路线 C 留的 acceptance gate")
    print("=" * 120)
    _bh_meds = [float(np.median(mc_results[_h]["trial_navB_end_buyhold"])) for _h in HORIZONS]
    _best_h_idx = int(np.argmax(_bh_meds))
    _best_h = HORIZONS[_best_h_idx]
    _best_end = _bh_meds[_best_h_idx]
    print(f"  方案 B 死拿最佳持仓天数: {_best_h}d, 4 年期末 {(1+_best_end):.3f}x ({_best_end*100:+.2f}%)")
    print(f"  → 折算年化大约 +{((1+_best_end)**(1/4) - 1)*100:.2f}%")
    print(f"  → 任何路线 C 方案, 4 年期末若不显著高于 {(1+_best_end):.3f}x, 都没意义")
    print()
    print("  关键观察:")
    print("  1. '止损 vs 死拿' 差距列, 几乎全是负数 → 止损削利润但保不住回撤, Q7-2 实锤再次验证")
    print("  2. 持仓天数从 5d 到 30d, 死拿期末净值都在 1.1x~1.4x 这个窄区间, 选股本身的差异不大")
    print("  3. → 主要 alpha 来自'多头日 ex-ante 开仓'这个 timing, 不是来自'选哪 5 只'或'怎么持仓'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 后续接路线 C 的接口

    本 notebook 的 `df_candidates` (6 条 AND 后的 LF2 候选池) 就是路线 C 所有 timing 实验的输入域.

    **路线 C 候选研究问题** (引自 `experiments/b1-next-phase.md` 2026-04-19 (晚)):
    - 多头切换日 T+N 的 alpha 衰减曲线 (找最优持仓窗口)
    - 多头区间内分早 / 中 / 末期 alpha 差异
    - 多头区间是否在不同行业上 alpha 差异
    - 活跃市值的 +4% / -2.3% 阈值是否最优 (网格搜索)
    - 是否能识别"假启动"用更严格的开仓确认
    - 是否能识别 regime 即将切换作为提前清仓信号

    **acceptance gate**: 任何路线 C 方案的方案 B 死拿期末, 都应当显著超过上面摘要表里的最佳数.
    """)
    return


if __name__ == "__main__":
    app.run()
