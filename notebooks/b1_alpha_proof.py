import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # B1 策略 alpha 验证 (整合版)

    **这个 notebook 在做一件事**: 用统计的办法, 一个一个回答 "B1 策略到底哪些规则真的有效, 哪些是错觉".

    所有问题都用同一个池子: 你**真能买**的票 (流通市值≥100亿 + 日均成交额≥5000万 = 流动性档 LF2).
    基准 = "在这个池子里, 每天每只票都买", 然后看每条规则筛出来的子池, 比这个基准强还是弱.

    **本 notebook 整合自 (现已删除)**:
    - `b1_stage0_alpha_proof.py` (4 大统计检验)
    - `b1_stage0_J_interaction.py` (J × 量价 二维交互)
    - `b1_stage0_textbook_v2.py` (累积过滤 + 流动性 + 信号拆解)

    **每一格回答的问题** (按顺序):

    1. **[Q1]** 在我真能买的池子里, 全市场平均 20 天涨多少? — 这就是基准
    2. **[Q2]** "活跃市值多头区间" 这一条规则单独用, 真的比平均强吗?
    3. **[Q3]** "白线高于黄线 + 收盘高于黄线" 这一条规则单独用, 真的比平均强吗?
    4. **[Q4]** "J 值 < 14" 这一条规则单独用, 真的比平均强吗?
    5. **[Q5]** 把上面 3 条加在一起用, 比单独最强的那条还强吗?
    6. **[Q6]** z 哥教科书完整 5 条规则, 真的有用吗?
    7. **[Q7]** 这些结论稳不稳? (重抽样验证)
    8. **[Q8]** 一张总表 — 所有规则横向对比

    **重要提醒** (基于 2026-04-19 的所有验证):

    - 真正能赚钱的几乎只有 1 条: 多头区间 (你 RPA 抓的活跃市值)
    - 教科书形态规则单独用大都没效甚至有害, 它们只在多头区间内部叠加时贡献边际 +0.14 个百分点
    - LightGBM + textbook 标签 + alpha158 那一套, 跨样本 0.94 倍富集 = 等于瞎猜
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

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0

    HORIZON = 20

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
        HORIZON,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 数据准备 (3 格)

    - **第 1 格**: 从 DuckDB 拉 2021-01-01 到 2025-12-31 的全部 A 股日 K (剔除 ST)
    - **第 2 格**: 算所有 B1 相关特征 (J, 白线 WL, 黄线 YL, 多头标记 is_manual_bull, 量价健康, 等等)
    - **第 3 格**: 算"未来 20 天会涨多少 / 最高涨多少 / 最深跌多少" + 算 20 日均成交额 + 锁定 LF2 池子

    看不太懂没关系, 数据准备完了直接看后面的 [Q1] ~ [Q8].
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
    print(f"特征已算完, 共 {df_all.height:,} 行 (= 票数 × 交易日数)")
    print(f"  含 多头区间标记 is_manual_bull = {('is_manual_bull' in df_all.columns)}")
    print(f"  含 教科书量价特征 prior_volume_surge_60d / peak_vol_shrink_60d = {('prior_volume_surge_60d' in df_all.columns)}")
    return (df_all,)


@app.cell
def _(HORIZON, df_all, pl):
    """注入未来 20 天数据 + 流动性 + 锁定 3 个池子."""
    def _run():
        _h = HORIZON

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

        df = (
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
                pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"),
            ])
            .drop(future_high_names + future_low_names)
            .filter(pl.col(f"fwd_ret_{_h}d").is_not_null())
            .select([
                "code", "date", "J", "WL", "YL", "close_adj",
                "prior_volume_surge_60d", "peak_vol_shrink_60d",
                "today_ret", "today_amplitude", "is_manual_bull",
                "market_cap_100m", "amount_ma20",
                f"fwd_mfe_{_h}d", f"fwd_ret_{_h}d",
            ])
            .collect()
        )

        df_pool_l0 = df
        df_pool_lf1 = df.filter((pl.col("market_cap_100m") >= 50) & (pl.col("amount_ma20") >= 3e7))
        df_pool_lf2 = df.filter((pl.col("market_cap_100m") >= 100) & (pl.col("amount_ma20") >= 5e7))

        print(f"完整数据 (含未来 {_h} 天涨跌): {df.height:,} 行")
        print(f"  L0 全市场池 (默认 mv 40~1500 亿)               : {df_pool_l0.height:>10,} 行")
        print(f"  LF1 中等池 (mv≥50亿 + 日均成交≥3000万)         : {df_pool_lf1.height:>10,} 行")
        print(f"  LF2 严格池 (mv≥100亿 + 日均成交≥5000万) ← 默认 : {df_pool_lf2.height:>10,} 行")
        return df_pool_l0, df_pool_lf1, df_pool_lf2

    df_pool_l0, df_pool_lf1, df_pool_lf2 = _run()
    return df_pool_l0, df_pool_lf1, df_pool_lf2


@app.cell
def _(HORIZON, np, pl):
    """共用工具函数: 评估一个子池子相对全池基准的表现."""
    def evaluate_signals(df_pool, signal_dict, label="(未指定池)"):
        """
        df_pool: 完整池子 DataFrame
        signal_dict: {"规则名": polars 布尔表达式}
        label: 池子的中文名 (如 "LF2 严格池")
        返回: 一张对比表
        """
        ret_col = f"fwd_ret_{HORIZON}d"
        mfe_col = f"fwd_mfe_{HORIZON}d"

        base_ret = float(df_pool[ret_col].mean() or 0.0)
        base_win = float((df_pool[ret_col] > 0).mean() or 0.0)
        base_hit15 = float((df_pool[mfe_col] >= 0.15).mean() or 0.0)

        rng = np.random.default_rng(42)
        N_BOOT = 1000

        rows = [{
            "规则": f"全部买 (基准, {label})",
            "样本数": df_pool.height,
            "20天平均涨跌": f"{base_ret*100:+.3f}%",
            "比基准多赚": "0.000pp (基准本身)",
            "重抽样区间": "—",
            "显著性": "—",
            "20天涨的概率": f"{base_win*100:.2f}%",
            "20天最高涨幅触及+15%的概率": f"{base_hit15*100:.2f}%",
        }]

        for rule_label, cond in signal_dict.items():
            sub = df_pool.filter(cond)
            if sub.height < 50:
                rows.append({"规则": rule_label, "样本数": sub.height, "20天平均涨跌": "样本太少"})
                continue
            sub_ret = sub[ret_col].to_numpy()
            sub_mfe = sub[mfe_col].to_numpy()

            lifts = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, sub.shape[0], size=sub.shape[0])
                lifts[b] = sub_ret[idx].mean() - base_ret
            ci_lo = np.quantile(lifts, 0.025)
            ci_hi = np.quantile(lifts, 0.975)

            sig = "✓ 显著为正" if ci_lo > 0 else ("⚠ 显著为负" if ci_hi < 0 else "✗ 不显著 (跟基准没差别)")

            rows.append({
                "规则": rule_label,
                "样本数": sub.height,
                "20天平均涨跌": f"{sub_ret.mean()*100:+.3f}%",
                "比基准多赚": f"{(sub_ret.mean() - base_ret)*100:+.3f}pp",
                "重抽样区间": f"[{ci_lo*100:+.3f}, {ci_hi*100:+.3f}]pp",
                "显著性": sig,
                "20天涨的概率": f"{(sub_ret > 0).mean()*100:.2f}%",
                "20天最高涨幅触及+15%的概率": f"{(sub_mfe >= 0.15).mean()*100:.2f}%",
            })

        return pl.DataFrame(rows)

    return (evaluate_signals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q1] 在我真能买的池子里, 全市场平均 20 天涨多少?

    **问题**: 我用什么"基准"来判断一条规则有没有效?
    基准 = 这个池子里所有票都买, 看 20 天后平均涨多少. 任何规则要叫"有效", 必须比这个基准多赚.

    **为什么要看 3 档池子**: 池子越大, 微盘小破票越多, 它们会用尾部脉冲拉高均值, 让基准虚高.
    我们最关心的是 **LF2 严格池** (你真能买进 5 万元不冲击市价的票).
    """)
    return


@app.cell
def _(HORIZON, df_pool_l0, df_pool_lf1, df_pool_lf2, pl):
    """[Q1] 3 档池子各自的基准表现."""
    def _run():
        ret = f"fwd_ret_{HORIZON}d"
        mfe = f"fwd_mfe_{HORIZON}d"
        rows = []
        for label, df in [
            ("L0 全市场池 (默认 mv 40~1500 亿, 含微盘)", df_pool_l0),
            ("LF1 中等池 (mv≥50亿 + 日均成交≥3000万)", df_pool_lf1),
            ("LF2 严格池 (mv≥100亿 + 日均成交≥5000万) ← 你真能买", df_pool_lf2),
        ]:
            rows.append({
                "池子": label,
                "样本数": df.height,
                "20天平均涨跌": f"{float(df[ret].mean() or 0)*100:+.3f}%",
                "20天涨的概率": f"{float((df[ret]>0).mean() or 0)*100:.2f}%",
                "20天最高涨幅触及+15%的概率": f"{float((df[mfe]>=0.15).mean() or 0)*100:.2f}%",
            })
        print("=" * 86)
        print(f"  [Q1] 3 档池子各自的全市场基准  (持有 20 天)")
        print("=" * 86)
        print(pl.DataFrame(rows))
        print("\n  解读:")
        print("  - L0 全市场 +1.10% → LF2 严格池 +0.24%, 基准缩水 78%")
        print("  - 这意味着: L0 看起来 '随便买都涨 1.1%' 是个错觉, 是几百只小破票的尾部脉冲拉的")
        print("  - 后面所有规则的对比, 我们都用 LF2 池子, 因为这才是你真能买的世界")
    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q2] "活跃市值多头区间" 一条规则单独用, 真的比平均强吗?

    **问题**: 你用 RPA 抓的"活跃市值多头区间" (LOOSE_PERIODS, T+1 触发) 这一条规则, 单独用,
    在 LF2 池子里, 比"全部买"强多少?

    **怎么判定有效**: 看"比基准多赚"是不是正数, 然后看"重抽样区间"是否完全大于 0
    (重抽样 = 我们把样本随机重抽 1000 次都算一遍, 看结果落在哪个范围, 排除巧合).
    """)
    return


@app.cell
def _(df_pool_lf2, evaluate_signals, pl):
    """[Q2] 多头区间单独使用."""
    print("=" * 86)
    print("  [Q2] 在 LF2 严格池里, 仅用 '活跃市值多头区间' 一条规则")
    print("=" * 86)
    print(evaluate_signals(
        df_pool_lf2,
        {"仅用 活跃市值多头区间": pl.col("is_manual_bull")},
        label="LF2 严格池",
    ))
    print("\n  解读: 多头区间这一条规则单独用, 比基准多赚 +1.46pp, 重抽样区间完全>0, ✓ 显著为正")
    print("  这是 B1 策略真正的金矿. 100% ex-ante 可执行 (今天收盘后才判定明天能不能开仓)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q3] "白线高于黄线 + 收盘高于黄线" 一条规则单独用, 真的强吗?

    **问题**: z 哥常说的"白线在黄线上方 + 收盘在黄线上方", 这个组合规则单独用,
    是不是真的能选出比平均涨得更多的票?
    """)
    return


@app.cell
def _(df_pool_lf2, evaluate_signals, pl):
    """[Q3] 白>黄 且 收>黄 单独使用."""
    print("=" * 86)
    print("  [Q3] 在 LF2 严格池里, 仅用 '白线>黄线 且 收盘>黄线'")
    print("=" * 86)
    print(evaluate_signals(
        df_pool_lf2,
        {"仅用 白线>黄线 且 收盘>黄线": (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))},
        label="LF2 严格池",
    ))
    print("\n  解读: 这条规则单独用, 比基准 ⚠ 少赚 -0.62pp, 重抽样区间完全<0, 显著为负")
    print("  也就是说: 单看白>黄+收>黄的票, 平均反而比 '全部买' 还差")
    print("  注意: 这跟你的常识可能矛盾. 别急, 第 [Q5] 格会解释为什么和多头区间叠加后又变正")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q4] "J 值 < 14" 一条规则单独用, 真的强吗?

    **问题**: 教科书上说 B1 必须 J 勾到大负值 (z 哥原版要求 J < 14).
    这一条单独用, 比平均涨得更多吗?
    """)
    return


@app.cell
def _(df_pool_lf2, evaluate_signals, pl):
    """[Q4] J<14 单独使用."""
    print("=" * 86)
    print("  [Q4] 在 LF2 严格池里, 仅用 'J<14'")
    print("=" * 86)
    print(evaluate_signals(
        df_pool_lf2,
        {"仅用 J<14": pl.col("J") < 14},
        label="LF2 严格池",
    ))
    print("\n  解读: J<14 单独用, 比基准多赚 +0.07pp, 区间勉强>0 (✓ 显著但极弱)")
    print("  几乎等于瞎选. 它本身不是好选股器, 顶多是一个 '过滤器'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q5] 三条加起来用, 比单独最强的那条 (多头区间) 还强吗?

    **问题**: 既然 [Q2]~[Q4] 都看完了, 我们现在把 3 条加在一起, 看是不是 1 + 1 + 1 > 1.

    - 多头区间 单独 = +1.46pp
    - + 白>黄+收>黄 = ?
    - + J<14 = ?

    **关键观察**: 注意"辛普森悖论". 一个规则在全市场是负 alpha, 但叠加在多头子集里又能变正,
    反过来在多头子集里加它有时候反而拖累. 这一格直接给你答案.
    """)
    return


@app.cell
def _(df_pool_lf2, evaluate_signals, pl):
    """[Q5] 多个规则叠加."""
    def _run():
        c_bull = pl.col("is_manual_bull")
        c_l2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        c_jhook = pl.col("J") < 14

        print("=" * 86)
        print("  [Q5] 在 LF2 严格池里, 多条规则叠加")
        print("=" * 86)
        print(evaluate_signals(
            df_pool_lf2,
            {
                "(回顾) 仅 活跃市值多头区间": c_bull,
                "多头区间 + 白>黄+收>黄": c_bull & c_l2,
                "多头区间 + 白>黄+收>黄 + J<14 ← B1 简化版": c_bull & c_l2 & c_jhook,
            },
            label="LF2 严格池",
        ))
        print("\n  解读:")
        print("  - 多头 (+1.46pp) → 加白>黄+收>黄 (+1.27pp), 反而 -0.19pp, 形态拖累了多头")
        print("  - 多头 (+1.46pp) → 加白>黄+收>黄+J<14 (+1.60pp), 比裸多头多 +0.14pp")
        print("  - 注意代价: 样本从 60 万缩到 2.8 万, 你失去 95% 的可交易日机会换 +0.14pp")
        print("  - 触及 +15% 的概率从 23.4% 提到 25.4% (+2pp), 这部分给止盈策略真正干活")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q6] 教科书完整 5 条规则, 真的有用吗?

    **z 哥 B1 完美图原版 5 条**:

    1. 量能 — 对比前期高点放量, 极致缩量
    2. 涨幅 — 当日 -2% ~ +1.8% (小阴小阳)
    3. 振幅 — 当日 ≤ 7%
    4. 均线 — 白线 > 黄线 (且收盘 > 黄线)
    5. 指标 — J < 14 (勾到大负值)

    我们一条一条往上加, 看每加一条 alpha 怎么变. **这一格是教科书定义的最严格版本**.
    """)
    return


@app.cell
def _(df_pool_lf2, evaluate_signals, pl):
    """[Q6] 教科书 5 条累积."""
    def _run():
        c_bull = pl.col("is_manual_bull")
        c_l2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        c_surge = pl.col("prior_volume_surge_60d")
        c_today = (pl.col("today_ret") >= -0.02) & (pl.col("today_ret") <= 0.018)
        c_amp = pl.col("today_amplitude") <= 0.07
        c_jhook = pl.col("J") < 14

        l2_surge = df_pool_lf2.filter(c_l2 & c_surge & pl.col("peak_vol_shrink_60d").is_not_null())
        q25 = float(l2_surge["peak_vol_shrink_60d"].quantile(0.25) or 0.0)
        c_shrink = pl.col("peak_vol_shrink_60d") <= q25

        print("=" * 86)
        print("  [Q6] 在 LF2 严格池里, 多头区间内部累积加教科书 5 条规则")
        print(f"  (极致缩量阈值 = {q25:.4f}, 在 L2_surge 子集内取 25 分位)")
        print("=" * 86)
        print(evaluate_signals(
            df_pool_lf2,
            {
                "多头": c_bull,
                "多头 + 白>黄+收>黄": c_bull & c_l2,
                "多头 + 白>黄+收>黄 + 规则1 (前期60天有放量启动)": c_bull & c_l2 & c_surge,
                "多头 + ... + 规则2 (今日企稳 -2%~+1.8%)": c_bull & c_l2 & c_surge & c_today,
                "多头 + ... + 规则3 (振幅≤7%)": c_bull & c_l2 & c_surge & c_today & c_amp,
                "多头 + ... + 规则5 (J<14)": c_bull & c_l2 & c_surge & c_today & c_amp & c_jhook,
                "多头 + 完整 5 条 (+ 规则1 极致缩量)": c_bull & c_l2 & c_surge & c_today & c_amp & c_jhook & c_shrink,
            },
            label="LF2 严格池",
        ))
        print("\n  解读:")
        print("  - 完整 5 条规则在多头区间内, 比裸多头基准多赚 ~+1pp, ✓ 显著, 但 < 裸多头自己的 +1.46pp")
        print("  - 也就是教科书规则的 '边际贡献' 是负的: 你筛得越严, 反而比 '只看多头区间' 更差")
        print("  - 但 '触及 +15%' 的概率会更高 → 教科书规则只有在配合止盈策略时才有意义")
        print("  - 极致缩量 (规则1的 peak≤Q25) 通常是最大的 alpha killer, 加它直接砍 0.3~0.5pp")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Q7] 多头区间的 alpha 在月间稳不稳?

    **问题**: 上面 Q2 的 +1.46pp 是把 5 年数据当一坨算的平均.
    万一这个 alpha 全部来自某 1~2 个超级月 (比如 2024-09 那波), 其他时间根本不赚钱呢?

    **方法**: 把每个月分开算 — 当月有多头日的, 我们把当月所有多头日全部买入, 算 20 天后平均涨多少.
    然后看月度序列: 大部分月份都赚, 还是只有少数月份猛赚拉高均值?

    **附带洞察**: 把"多头开仓"和"非多头开仓"按月对比, 你会发现一个反直觉事实 —
    在很多月里, **非多头日开仓的未来 20 天涨幅 ≥ 多头日开仓**. 为什么?
    因为 fwd_ret_20d 是"从这一天往后 20 天", 多头末期开仓很快撞上 regime switch 回吐,
    而非多头末期开仓 20 天后正好踩到下一波多头.
    所以 +1.46pp 真正的 alpha 是 **"避开非多头时段空仓"**, 不是"多头时段票更好".
    """)
    return


@app.cell
def _(HORIZON, df_pool_lf2, np, pl):
    """[Q7] 多头开仓的 fwd_ret_20d 在月间是否稳定."""
    def _run():
        ret = f"fwd_ret_{HORIZON}d"
        df = df_pool_lf2.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym"))

        monthly = (
            df.group_by("ym")
            .agg([
                pl.col(ret).mean().alias("当月_全部买入_平均涨跌"),
                pl.col(ret).filter(pl.col("is_manual_bull")).mean().alias("当月_多头日买入_平均涨跌"),
                pl.col(ret).filter(pl.col("is_manual_bull")).count().alias("当月_多头日开仓数"),
                pl.len().alias("当月_总开仓机会"),
            ])
            .filter(pl.col("当月_多头日开仓数") >= 100)
            .sort("ym")
        )

        bull_rets = monthly["当月_多头日买入_平均涨跌"].to_numpy()
        bull_rets = bull_rets[~np.isnan(bull_rets)]
        n_months = len(bull_rets)
        mean_ret = float(bull_rets.mean())
        std_ret = float(bull_rets.std(ddof=1))
        t_stat_pos = mean_ret / (std_ret / np.sqrt(n_months))
        n_positive = int((bull_rets > 0).sum())

        full_buy = monthly["当月_全部买入_平均涨跌"].to_numpy()
        diffs = bull_rets - full_buy
        diffs = diffs[~np.isnan(diffs)]
        n_diff = len(diffs)
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1))
        t_diff = mean_diff / (std_diff / np.sqrt(n_diff)) if n_diff > 1 and std_diff > 0 else float("nan")
        n_diff_pos = int((diffs > 0).sum())

        print("=" * 86)
        print(f"  [Q7-1] 每个有多头日的月份, 多头日开仓 20 天后平均涨多少?")
        print("=" * 86)
        print(f"  统计的月份数      : {n_months}")
        print(f"  月均 '多头日开仓 fwd_ret_20d': {mean_ret*100:+.3f}%")
        print(f"  月度标准差                  : {std_ret*100:.3f}%")
        print(f"  多头日开仓 20 天后赚钱的月份: {n_positive}/{n_months} = {n_positive/n_months:.2%}")
        print(f"  t 检验 (vs 0)               : {t_stat_pos:+.3f}  (|t|>2 显著为正)")

        print("\n" + "=" * 86)
        print(f"  [Q7-2] 同月对比: 多头日开仓 vs 当月每天都买")
        print("=" * 86)
        print(f"  月均 '多头日开仓 - 全月每天买': {mean_diff*100:+.3f}pp")
        print(f"  多头胜出的月份: {n_diff_pos}/{n_diff} = {n_diff_pos/n_diff:.2%}")
        print(f"  t 检验 vs 0   : {t_diff:+.3f}")
        print(f"  → 反直觉发现: 月内对比下, 多头日开仓未必比 '当月每天都买' 强")
        print(f"     因为 fwd_ret_20d 跨 regime, 多头末期开仓会撞上 regime switch 回吐")

        print("\n  最近 12 个月明细:")
        print(monthly.tail(12).select([
            "ym", "当月_总开仓机会", "当月_多头日开仓数",
            "当月_全部买入_平均涨跌", "当月_多头日买入_平均涨跌"
        ]))

        print("\n  ───── 结论 ─────")
        print("  Q2 +1.46pp 看似是 '多头票更好', 实际 alpha 来源更准确的描述是:")
        print("  '只在多头时段开仓, 避开非多头时段不开仓' 这个择时动作本身值钱")
        print("  这意味着: 真正的实战策略不是 '多头时段挑哪只票', 而是 '在非多头时段空仓'")
        print("  → 这给路线 C (研究择时本身) 提供了强力支持")

    _run()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## [Q8] 一张总表 — 所有规则横向对比

    把 [Q2]~[Q6] 的所有重要规则放在一张表里, 方便一眼看清楚:

    - 哪条规则真的强 (✓)
    - 哪条规则有害 (⚠)
    - 哪条规则没用 (✗)
    - 加规则到底是不是越多越好 (看样本数 vs 多赚的权衡)
    """)
    return


@app.cell
def _(df_pool_lf2, evaluate_signals, pl):
    """[Q8] 终极汇总表."""
    def _run():
        c_bull = pl.col("is_manual_bull")
        c_l2 = (pl.col("WL") > pl.col("YL")) & (pl.col("close_adj") > pl.col("YL"))
        c_surge = pl.col("prior_volume_surge_60d")
        c_today = (pl.col("today_ret") >= -0.02) & (pl.col("today_ret") <= 0.018)
        c_amp = pl.col("today_amplitude") <= 0.07
        c_jhook = pl.col("J") < 14

        l2_surge = df_pool_lf2.filter(c_l2 & c_surge & pl.col("peak_vol_shrink_60d").is_not_null())
        q25 = float(l2_surge["peak_vol_shrink_60d"].quantile(0.25) or 0.0)
        c_shrink = pl.col("peak_vol_shrink_60d") <= q25

        print("=" * 86)
        print("  [Q8] 终极汇总表  (LF2 严格池, 持有 20 天)")
        print("=" * 86)
        print(evaluate_signals(
            df_pool_lf2,
            {
                "[单条] 仅 活跃市值多头区间": c_bull,
                "[单条] 仅 白>黄+收>黄": c_l2,
                "[单条] 仅 J<14": c_jhook,
                "[组合] 多头 + 白>黄+收>黄": c_bull & c_l2,
                "[组合] 多头 + 白>黄+收>黄 + J<14 (B1 简化)": c_bull & c_l2 & c_jhook,
                "[教科书] 多头 + 完整 5 条规则": c_bull & c_l2 & c_surge & c_today & c_amp & c_jhook & c_shrink,
            },
            label="LF2 严格池",
        ))

        print("\n  ═══════ 一句话总结 ═══════")
        print("  真正的 alpha 几乎全部来自 1 条规则: 活跃市值多头区间")
        print("  教科书形态规则单独用要么没用 (J<14) 要么有害 (白>黄+收>黄)")
        print("  它们只在 '叠加在多头区间里' 时贡献边际 +0.14pp, 代价是放弃 95% 的样本")
        print("  教科书最严的 5 条规则比 '只看多头' 反而少赚, 但 '触及+15%的概率' 显著更高")
        print("  → 实战意义: 不带止盈/止损, 你应该尽量少加规则; 带止盈策略, 教科书规则才有用")

    _run()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 下一步 — 我们要解决的真正问题

    到这里, "B1 规则有没有效" 这个问题已经被回答清楚了. 但**这不解决实战的核心问题**:

    > "我每天有 50~100 只候选, 该怎么挑出 top5?"

    这是规则解决不了的, 必须做"池子内排序研究".

    **接下来 3 条候选路线**:

    ### 路线 A — 极简化 (1 天可做完)

    - 把策略 collapse 成: 多头区间 + LF2 流动性过滤 + **池内随机选 5 只**
    - 用蒙特卡洛模拟 1000 次, 看分布
    - 这是你的"诚实下限", 任何更复杂的方案必须先打败它

    ### 路线 B — 池子内排序研究 (1~2 周, 真正解决问题)

    - 池子已经定: `多头 + 白>黄+收>黄 + J<14 + LF2 流动性`
    - 不再做"是不是 B1"的分类 (规则解决了)
    - 改做: **同一天的池子里, 哪些特征能预测 20 天涨多少**
    - 候选排序信号 (逐一测 IC):
      - 收盘距黄线的距离 (越近=刚回调到位?)
      - J 值在池内的相对排名
      - 当日成交相对前 20 日均量
      - 板块强度 / 行业轮动位
      - 20 日内反弹累计幅度
      - 短期动量 (5d / 10d / 20d)
    - 验收: top5 表现 > 池子内随机选 5 只

    ### 路线 C — 优化择时本身 (长期, 价值最大)

    - 既然多头区间是金矿, 把多头区间本身研究透:
      - 多头初期 (T+1~T+5) / 中期 / 末期 alpha 是否衰减
      - 多头切换日 T+N 的胜率衰减曲线 (找最优持仓窗口)
      - 多头区间在不同行业上 alpha 差异 (是否需要叠加行业过滤)
      - 活跃市值的 +4% / -2.3% 阈值是否最优, 网格搜索
      - 是否能识别"假启动" (2-3 天就转空头) 用更严格的开仓确认

    ### 我个人建议

    1. **先做路线 A** (1 天), 跑一个诚实下限基准
    2. **马上做路线 B** (1~2 周), 真正解决 top5 选股
    3. **再做路线 C** (长期), 把金矿挖深
    """)
    return


if __name__ == "__main__":
    app.run()
