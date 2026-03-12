import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb
    import datetime

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"

    PE_THRESHOLD = 20.0
    YIELD_THRESHOLD = 3.0
    MA120_DISCOUNT = 0.12

    CANDIDATE_POOL = {
        "格力电器": ("sz.000651", "红队"),
        "美的集团": ("sz.000333", "红队"),
        "伊利股份": ("sh.600887", "红队"),
        "云天化":   ("sh.600096", "红队"),
        "南山铝业": ("sh.600219", "红队"),
        "皖通高速": ("sh.600012", "红队"),
        "宁沪高速": ("sh.600377", "红队"),
        "江苏金租": ("sh.600901", "红队"),
        "传音控股": ("sh.688036", "黄队"),
        "中国移动": ("sh.600941", "黄队"),
        "中国神华": ("sh.601088", "黄队"),
        "桐昆集团": ("sh.601233", "黄队"),
        "温氏股份": ("sz.300498", "黄队"),
        "中国中车": ("sh.601766", "黄队"),
        "浙能电力": ("sh.600023", "黄队"),
        "招商银行": ("sh.600036", "黄队"),
        "爱玛科技": ("sh.603529", "黄队"),
        "思维列控": ("sh.603508", "黄队"),
        "双汇发展": ("sz.000895", "蓝队"),
        "中国海油": ("sh.600938", "蓝队"),
        "冀中能源": ("sz.000937", "蓝队"),
        "嘉化能源": ("sh.600273", "蓝队"),
        "京基智农": ("sz.000048", "蓝队"),
        "国光股份": ("sz.002749", "蓝队"),
        "中国平安": ("sh.601318", "蓝队"),
        "上海银行": ("sh.601229", "蓝队"),
        "浦发银行": ("sh.600000", "蓝队"),
        "嘉益股份": ("sh.603826", "蓝队"),
    }

    mo.md("# 点金术筛选器")
    return (
        CANDIDATE_POOL,
        DB_PATH,
        MA120_DISCOUNT,
        PE_THRESHOLD,
        YIELD_THRESHOLD,
        datetime,
        duckdb,
        mo,
        pl,
    )


@app.cell
def _(datetime, pl):
    def compute_indicators(conn):
        """
        批量计算全市场点金术三大指标: PE(TTM), 股息率(TTM), MA120距离(%)
        返回 Polars DataFrame, 每行一只股票
        """

        # ── A. 前复权因子 (全量, qmt_factors 表很小) ──
        adj_ratio = (
            conn.sql(
                "SELECT code, date, "
                "CAST(COALESCE(dr, 1.0) AS DOUBLE) AS dr "
                "FROM qmt_factors ORDER BY code, date"
            ).pl().lazy()
            .sort(["code", "date"])
            .with_columns(pl.col("dr").cum_prod().over("code").alias("cum_dr"))
            .with_columns(
                pl.col("cum_dr").last().over("code").alias("latest_cum_dr")
            )
            .with_columns(
                (pl.col("cum_dr") / pl.col("latest_cum_dr")).alias("adj_ratio")
            )
            .select(["code", "date", "adj_ratio"])
        )

        # ── B. MA120 (前复权, 只加载近 300 天日线数据) ──
        df_daily = (
            conn.sql(
                "SELECT code, date, CAST(close AS DOUBLE) AS close "
                "FROM stock_daily "
                "WHERE date >= CURRENT_DATE - INTERVAL '300 days' "
                "ORDER BY code, date"
            ).pl().lazy()
        )

        df_ma = (
            df_daily
            .sort(["code", "date"])
            .join_asof(
                adj_ratio.sort(["code", "date"]),
                on="date", by="code", strategy="backward"
            )
            .with_columns(pl.col("adj_ratio").fill_null(1.0))
            .with_columns(
                (pl.col("close") * pl.col("adj_ratio")).alias("adj_close")
            )
            .sort(["code", "date"])
            .with_columns(
                pl.col("adj_close")
                .rolling_mean(window_size=120, min_periods=120)
                .over("code")
                .alias("ma120")
            )
            .filter(pl.col("date") == pl.col("date").max().over("code"))
            .select("code", "date", "close", "adj_close", "ma120")
            .with_columns(
                ((pl.col("adj_close") - pl.col("ma120"))
                 / pl.col("ma120") * 100).alias("ma120_dist_pct")
            )
            .collect()
        )

        # ── C. PE (TTM): 最新利润年化 ──
        df_latest_income = (
            conn.sql(
                "SELECT code, date AS report_date, pub_date, "
                "CAST(net_profit AS DOUBLE) AS net_profit "
                "FROM finance_income ORDER BY code, pub_date"
            ).pl()
            .sort(["code", "pub_date"])
            .group_by("code").last()
            .with_columns(
                pl.when(pl.col("report_date").dt.month() == 3).then(4.0)
                .when(pl.col("report_date").dt.month() == 6).then(2.0)
                .when(pl.col("report_date").dt.month() == 9).then(1.333333)
                .otherwise(1.0)
                .alias("annual_factor")
            )
            .with_columns(
                (pl.col("net_profit") * pl.col("annual_factor"))
                .alias("annualized_profit")
            )
            .select("code", "annualized_profit")
        )

        # ── D. 最新股本 ──
        df_latest_cap = (
            conn.sql(
                "SELECT code, pub_date, "
                "CAST(total_capital AS DOUBLE) AS total_capital "
                "FROM finance_capital ORDER BY code, pub_date"
            ).pl()
            .filter(pl.col("pub_date").is_not_null())
            .sort(["code", "pub_date"])
            .group_by("code").last()
            .select("code", "total_capital")
        )

        # ── E. 股息率 (TTM): 最近 365 天 interest 求和 ──
        cutoff = df_ma["date"].max() - datetime.timedelta(days=365)
        df_div_ttm = (
            conn.sql(
                "SELECT code, date, CAST(interest AS DOUBLE) AS interest "
                "FROM qmt_factors "
                "WHERE interest IS NOT NULL AND interest > 0 "
                "ORDER BY code, date"
            ).pl()
            .filter(pl.col("date") >= cutoff)
            .group_by("code")
            .agg(pl.col("interest").sum().alias("div_ttm"))
        )

        # ── F. 合并 ──
        result = (
            df_ma
            .join(df_latest_income, on="code", how="left")
            .join(df_latest_cap, on="code", how="left")
            .join(df_div_ttm, on="code", how="left")
            .with_columns(pl.col("div_ttm").fill_null(0.0))
            .with_columns([
                (pl.col("close") * pl.col("total_capital")
                 / pl.col("annualized_profit")).alias("pe_ttm"),
                (pl.col("div_ttm") / pl.col("close") * 100)
                .alias("yield_pct"),
            ])
            .select(
                "code", "date", "close",
                "pe_ttm", "yield_pct",
                "ma120", "ma120_dist_pct",
            )
        )
        return result

    return (compute_indicators,)


@app.cell
def _(CANDIDATE_POOL, DB_PATH, compute_indicators, duckdb, mo):
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        df_all = compute_indicators(conn)
    except Exception as e:
        mo.stop(True, mo.md(f"**Error**: {e}"))
    finally:
        conn.close()

    candidate_codes = [v[0] for v in CANDIDATE_POOL.values()]
    return candidate_codes, df_all


@app.cell
def _(
    CANDIDATE_POOL,
    MA120_DISCOUNT,
    PE_THRESHOLD,
    YIELD_THRESHOLD,
    candidate_codes,
    df_all,
    mo,
    pl,
):
    # ── Feature 1: 候选池检测 ──

    pool_rows = [
        {"code": info[0], "name": name, "team": info[1]}
        for name, info in CANDIDATE_POOL.items()
    ]
    df_pool_info = pl.DataFrame(pool_rows)

    ma120_thr = -MA120_DISCOUNT * 100  # -12.0

    df_check = (
        df_all
        .filter(pl.col("code").is_in(candidate_codes))
        .join(df_pool_info, on="code", how="inner")
        .with_columns([
            ((pl.col("pe_ttm") > 0) & (pl.col("pe_ttm") < PE_THRESHOLD))
            .fill_null(False).alias("pass_pe"),
            (pl.col("yield_pct") > YIELD_THRESHOLD)
            .fill_null(False).alias("pass_yield"),
            (pl.col("ma120_dist_pct") < ma120_thr)
            .fill_null(False).alias("pass_ma120"),
        ])
        .with_columns(
            (pl.col("pass_pe") & pl.col("pass_yield") & pl.col("pass_ma120"))
            .alias("all_pass")
        )
        .with_columns([
            pl.col("close").round(2),
            pl.col("pe_ttm").round(2),
            pl.col("yield_pct").round(2),
            pl.col("ma120_dist_pct").round(2),
        ])
        .sort(["team", "code"])
    )

    def _build_reason(row):
        reasons = []
        pe = row["pe_ttm"]
        yld = row["yield_pct"]
        dist = row["ma120_dist_pct"]
        if not row["pass_pe"]:
            s = f"{pe:.1f}" if pe is not None else "N/A"
            reasons.append(f"PE={s}")
        if not row["pass_yield"]:
            s = f"{yld:.1f}%" if yld is not None else "N/A"
            reasons.append(f"股息={s}")
        if not row["pass_ma120"]:
            s = f"{dist:+.1f}%" if dist is not None else "N/A"
            reasons.append(f"MA120距离={s}")
        return " | ".join(reasons) if reasons else "全部达标"

    reasons = [
        _build_reason(df_check.row(i, named=True))
        for i in range(df_check.height)
    ]
    df_check = df_check.with_columns(pl.Series("原因", reasons))

    df_display = df_check.select([
        pl.col("team").alias("队伍"),
        pl.col("name").alias("名称"),
        pl.col("code").alias("代码"),
        pl.col("close").alias("现价"),
        pl.col("pe_ttm").alias("PE(TTM)"),
        pl.col("yield_pct").alias("股息率%"),
        pl.col("ma120_dist_pct").alias("MA120距离%"),
        pl.when(pl.col("all_pass")).then(pl.lit("达标"))
        .otherwise(pl.lit("不达标")).alias("状态"),
        pl.col("原因"),
    ])

    pass_count = df_check.filter(pl.col("all_pass")).height
    total_count = df_check.height

    mo.vstack([
        mo.md(
            f"### 候选池检测\n"
            f"条件: PE > 0 且 < {PE_THRESHOLD} & 股息率 > {YIELD_THRESHOLD}% & "
            f"低于 MA120 {MA120_DISCOUNT*100:.0f}%\n\n"
            f"**{pass_count}/{total_count}** 只达标"
        ),
        mo.ui.table(df_display, selection=None),
    ])
    return


@app.cell
def _(MA120_DISCOUNT, PE_THRESHOLD, YIELD_THRESHOLD, df_all, mo, pl):
    # ── Feature 2: 全市场扫描 ──

    _ma120_thr = -MA120_DISCOUNT * 100

    df_scan = (
        df_all
        .filter(
            (pl.col("pe_ttm") > 0)
            & (pl.col("pe_ttm") < PE_THRESHOLD)
            & (pl.col("yield_pct") > YIELD_THRESHOLD)
            & (pl.col("ma120_dist_pct") < _ma120_thr)
            & pl.col("ma120").is_not_null()
        )
        .with_columns([
            pl.col("close").round(2),
            pl.col("pe_ttm").round(2),
            pl.col("yield_pct").round(2),
            pl.col("ma120_dist_pct").round(2),
        ])
        .sort("yield_pct", descending=True)
    )

    df_scan_display = df_scan.select([
        pl.col("code").alias("代码"),
        pl.col("close").alias("现价"),
        pl.col("pe_ttm").alias("PE(TTM)"),
        pl.col("yield_pct").alias("股息率%"),
        pl.col("ma120_dist_pct").alias("MA120距离%"),
    ])

    mo.vstack([
        mo.md(
            f"### 全市场扫描\n"
            f"条件: PE(TTM) > 0 且 < {PE_THRESHOLD} & 股息率 > {YIELD_THRESHOLD}% & "
            f"低于 MA120 {MA120_DISCOUNT*100:.0f}%\n\n"
            f"共找到 **{df_scan.height}** 只达标股票"
        ),
        mo.ui.table(df_scan_display, selection=None),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
