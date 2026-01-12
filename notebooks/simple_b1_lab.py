import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    from datetime import datetime
    from utils.b1_factors import calc_b1_factors_tg
    from utils.baostock_utils import get_st_blacklist_pl

    # ==============================================================================
    # 1. 配置与数据加载 (Clean Version)
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"

    # ==============================================================================
    # Ztalk 体系核心：只在“活跃市值”强势期开仓
    MANUAL_LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),  # 春季躁动
        ("2019-12-16", "2020-03-02"),  # 疫情反弹
        ("2020-06-19", "2020-07-15"),  # 证券带头的疯牛
        ("2020-12-24", "2021-01-25"),  # 新能源抱团主升
        ("2021-04-16", "2021-09-14"),  # 锂电光伏大主升
        ("2022-04-27", "2022-07-05"),  # 427大反弹
        ("2023-01-15", "2023-04-15"),  # ChatGPT/CPO 狂潮
        ("2024-02-06", "2024-03-20"),  # 救市后AI反弹
        # ("2024-09-24", "2024-10-15"),  # 924 史诗级暴涨
        ("2025-04-09", "2025-09-04"),  # 2025年慢牛行情
        ("2026-01-05", "2026-03-31"),  # 2025年慢牛行情延续
    ]


    print("🚀 [Step 1] 加载原始行情数据...")

    # (A) 加载复前权行情
    q_adj = (
        pl.scan_parquet(
            os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
            include_file_paths="file_path"
        )
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ])
        .select(["code", "date", "open", "high", "low", "close", "volume", "amount"])
        .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})
        .filter(pl.col("volume") > 0)
    )

    # (B) 加载 Raw (不复权) 和 Capital (股本)
    q_raw = (
        pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_raw", "*.parquet"), include_file_paths="file_path")
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ])
        .select(["code", "date", "close"]).rename({"close": "close_raw"})
    )

    q_cap = (
        pl.scan_parquet(os.path.join(DATA_ROOT, "finance_capital", "*.parquet"), include_file_paths="file_path")
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.col("m_anntime").str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("circulating_capital").cast(pl.Float64)
        ])
        .select(["code", "date", "circulating_capital"]).sort(["code", "date"])
    )



    # ==============================================================================
    # ⚙️ 策略参数配置 V3.0 (Based on 10 Golden Cases)
    # ==============================================================================
    CONFIG = {
        # === 基础门槛 ===
        "J_THRESHOLD": 13.8,          # 放宽至 13.8
        "X": 10
    }
    st_blacklist = get_st_blacklist_pl('2025-01-09') # 获取ST列表

    # 过滤掉 ST 股
    # (C) 合并数据 (移除了所有市场指数相关代码)
    print("🔗 [Step 2] 合并基础数据...")
    q_full = (
        q_adj
        .join(q_raw, on=["code", "date"])
        .sort(["code", "date"])
        .join_asof(q_cap, on="date", by="code", strategy="backward")
        .with_columns([
            (pl.col("close_raw") * pl.col("circulating_capital") / 1e8).alias("market_cap_100m")
        ]).filter(
            ~pl.col("code").is_in(st_blacklist)
        )
    )
    return (
        CONFIG,
        MANUAL_LOOSE_PERIODS,
        calc_b1_factors_tg,
        datetime,
        pl,
        q_full,
    )


@app.cell
def _(CONFIG, calc_b1_factors_tg, q_full):
    # 3. 执行计算
    print("⏳ 计算原始 B1 信号...")
    df_signals = calc_b1_factors_tg(q_full, CONFIG)
    return (df_signals,)


@app.cell
def _(MANUAL_LOOSE_PERIODS, datetime, df_signals, pl):
    # ==============================================================================
    # 4. 回测引擎：实战派 (动态技术止损版 - K线最低价风控)
    # ==============================================================================
    def run_strategy_realistic_dynamic_stop(df_signals: pl.LazyFrame, return_days: list) -> pl.DataFrame:
        print("🛠️ [Step 4] 启动实战回测：开盘突击 + 动态技术止损 (K线最低价下浮2%)...")

        # 1. 择时条件构建 (纯表达式，无需 collect)
        loose_conditions = []
        for s_str, e_str in MANUAL_LOOSE_PERIODS:
            try:
                s = datetime.strptime(s_str, "%Y-%m-%d").date()
                e = datetime.strptime(e_str, "%Y-%m-%d").date()
                loose_conditions.append(pl.col("date").is_between(s, e))
            except: pass

        # 用 any_horizontal 合并所有区间条件
        is_loose_expr = pl.any_horizontal(loose_conditions) if loose_conditions else pl.lit(False)

        expr_list = [
            # --- 择时标记 ---
            is_loose_expr.cast(pl.Int32).alias("is_loose"),
            # --- 进攻视角：T+1 开盘就买 ---
            pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
            pl.col("low_adj").shift(-1).over("code").alias("buy_price_low"),
            # --- 辅助指标：冷却与排序 ---
            # 标记是否处于冷却期
            (pl.col("b1_signal").cast(pl.Int32).shift(1).rolling_max(10).over("code").fill_null(0) == 0).alias("is_cool"),
        ]

        return_expr_list = []

        for rd in return_days:
            # --- 上帝视角：预取未来 N 天的数据 ---
            expr_list.append(
                pl.col("close_adj").shift(-rd).over("code").alias(f"close_{rd}d"),
            )
            # 最低价 (用于判断是否触发止损)
            expr_list.append(
                pl.col("close_adj").rolling_min(rd).shift(-rd).over("code").alias(f"low_min_{rd}d"),
            )
            # x日收益
            return_expr_list.append(
                pl.when(pl.col(f"low_min_{rd}d") <= pl.col("stop_price_tech"))
                  .then(pl.col("risk_pct")) # 触发止损，亏损额度即为 risk_pct
                  .otherwise((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1)
                  .alias(f"ret_{rd}d")
            )
            # 对照组：无止损死拿
            return_expr_list.append(
                ((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1).alias(f"ret_{rd}d_raw")
            )

        # 2. 核心交易逻辑
        return (
            df_signals
            .sort(["code", "date"])
            # ==============================================================================
            # 🔥 关键修改：在过滤之前，在全量数据上计算所有价格和指标
            # ==============================================================================
            .with_columns(expr_list)
            .with_columns(
                # --- 防守视角：动态技术止损 (Dynamic Technical Stop) ---
                # 逻辑：取信号当日(T)的最低价，向下浮动 2% 作为硬防守线
                # 这比固定 7% 更贴合个股走势
                (pl.col("buy_price_low") * 0.98).alias("stop_price_tech"),   
            )
            # ==============================================================================
            # 🔥 核心过滤 (Filtering) - 必须在 shift 计算之后
            # ==============================================================================
            .filter(pl.col("b1_signal"))        # 必须是信号
            .filter(pl.col("is_loose") == 1)    # 必须是活跃市值多头
            # .filter(pl.col("is_cool") == True)  # 必须是新鲜信号(非重复)
            # 不再过滤 buy_price，让最新一天信号保留，收益自然为 null
            # ==============================================================================
            # 🔥 每日排序 (Ranking) - 可选多种排序因子
            # ==============================================================================
            # 排序因子选项 (descending=False 表示值越小越优先):
            #   1. "volume"    - 缩量优先 (量小=控盘强)
            #   2. "amplitude" - 低振幅优先 (波动小=稳定)
            #   3. "J"         - J值低优先 (超卖=安全边际)
            #   4. "zx_dist"   - 距知行线近优先 (贴线=支撑强)
            #   5. "random"    - 随机排序 (对照组)
            # ==============================================================================
            .with_columns([
                # 距离知行多空线的相对距离 (越小越好)
                ((pl.col("close_adj") - pl.col("zx_long")) / pl.col("zx_long")).abs().alias("zx_dist"),
                # 随机因子
                pl.lit(1).alias("random_factor"),  # 占位，后续用 sample 或 hash
            ])
            .with_columns([
                # 🎯 修改这里切换排序因子
                pl.col("zx_dist").rank("ordinal", descending=False).over("date").alias("daily_rank")
            ])
            .filter(pl.col("daily_rank") <= 3) # 每天只买前N个
            # ==============================================================================
            # 🔥 收益结算 (Settlement)
            # ==============================================================================
            .with_columns([
                # 计算这一单的实际风险比例 (Stop / Buy - 1)
                # 比如：买入日最低价很低，导致止损线在买入价下方 5%，则 risk_pct = -0.05
                ((pl.col("stop_price_tech") / pl.col("buy_price")) - 1).alias("risk_pct")
            ])
            .with_columns(return_expr_list)
            .collect()
        )

    # ==============================================================================
    # 5. 执行并打印报告 (使用新函数)
    # ==============================================================================
    # 注意：传入的是 df_signals (LazyFrame)，函数内部会自动处理全量计算和过滤
    return_days = [5, 10, 15, 20, 25, 30]

    df_result_dynamic = run_strategy_realistic_dynamic_stop(df_signals, return_days)

    print(f"\n====== ⚔️ Ztalk 实战回测 (动态技术止损版) ======")
    total_trades_dynamic = df_result_dynamic.height
    print(f"✅ 交易信号总数: {total_trades_dynamic}")

    if total_trades_dynamic > 0:
        print("-" * 100)
        print(f"{'策略模式':<12} | {'胜率':<8} | {'均值':<8} | {'盈亏比(Odds)':<10} | {'期望值(Exp)':<10}")
        print("-" * 100)

        # 辅助打印函数 (逻辑不变)
        def print_metric_dynamic(name, col_name, df_res):
            df_valid = df_res.filter(pl.col(col_name).is_not_null())
            cnt = df_valid.height
            if cnt == 0: return

            win_cnt = df_valid.filter(pl.col(col_name) > 0).height
            win_rate = win_cnt / cnt

            avg_ret = df_valid.select(pl.col(col_name).mean()).item()

            avg_win = df_valid.filter(pl.col(col_name) > 0).select(pl.col(col_name).mean()).item()
            avg_loss = df_valid.filter(pl.col(col_name) <= 0).select(pl.col(col_name).mean()).item()

            if avg_loss == 0 or avg_loss is None: 
                odds = 99.9 
            else:
                odds = abs(avg_win / avg_loss)

            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

            print(f"{name:<12} | {win_rate*100:>6.1f}% | {avg_ret*100:>6.2f}% | {odds:>10.2f}x  | {expectancy*100:>8.2f}%")

        for rd in return_days:
            print_metric_dynamic(f"持仓{rd}天", f"ret_{rd}d", df_result_dynamic)
        print("-" * 100)
        for rd in return_days:
            print_metric_dynamic(f"死拿{rd}天(对照)", f"ret_{rd}d_raw", df_result_dynamic)
        print("-" * 100)
    return (df_result_dynamic,)


@app.cell
def _(df_result_dynamic, pl):
    # ==============================================================================
    # 6. 附录：年度交易频率压力测试 (Stress Test)
    # ==============================================================================
    def analyze_yearly_intensity(df_result: pl.DataFrame, target_year: int):
        print(f"\n====== 📊 {target_year} 年度交易强度分析 ======")

        # 1. 提取年份并过滤
        # 注意：需确认 date 是 Date 类型还是 String 类型，这里做了兼容处理
        try:
            # 尝试作为 Date 类型处理
            df_year = df_result.filter(pl.col("date").dt.year() == target_year)
        except:
            # 如果报错，说明是 String 类型，按字符串切片处理
            df_year = df_result.filter(pl.col("date").str.slice(0, 4) == str(target_year))

        total_signals = df_year.height

        if total_signals == 0:
            print(f"⚠️ {target_year} 年没有交易信号 (可能是数据未包含或择时全空)。")
            return

        # 2. 按日期聚合，统计每天的信号数量
        df_daily_counts = (
            df_year
            .group_by("date")
            .agg(pl.len().alias("trade_count"))
            .sort("trade_count", descending=True)
        )

        # 3. 计算统计指标
        active_days = df_daily_counts.height
        avg_trades = df_daily_counts.select(pl.col("trade_count").mean()).item()
        median_trades = df_daily_counts.select(pl.col("trade_count").median()).item()
        max_trades = df_daily_counts.select(pl.col("trade_count").max()).item()

        # 4. 打印报告
        print(f"📅 交易天数: {active_days} 天 (资金活跃度)")
        print(f"🔫 总开枪数: {total_signals} 次")
        print("-" * 40)
        print(f"📉 平均每天: {avg_trades:.1f} 只")
        print(f"⚖️ 中位每天: {median_trades:.1f} 只 (最常见的情况)")
        print(f"🔥 爆发极值: {max_trades} 只 (那天你忙得过来吗？)")
        print("-" * 40)

        # 5. 打印最忙碌的 Top 3 日子，看看发生了什么
        print("🥵 最忙碌的 3 天:")
        for row in df_daily_counts.head(3).iter_rows(named=True):
            print(f"   {row['date']}: {row['trade_count']} 只")

    # ==============================================================================
    # 运行分析
    # ==============================================================================
    # 统计 2024 或 2025 年的数据 (取决于你的数据源到哪一天)
    analyze_yearly_intensity(df_result_dynamic, 2024) 
    analyze_yearly_intensity(df_result_dynamic, 2025)
    analyze_yearly_intensity(df_result_dynamic, 2026)
    return


@app.cell
def _(datetime, df_result_dynamic, pl):
    df_result_dynamic.filter(
        (pl.col("date") == datetime(2026,1,5)) &
        (pl.col("b1_signal") == 1) 
    ).select(["date","code"])
    return


@app.cell
def _(df_signals, pl):
    PERFECT_CASES_CONFIG = [
        {"code": "688799_SH", "date": "2025-05-12", "name": "华纳药厂(标准)"},
        {"code": "300689_SZ", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
        {"code": "600601_SH", "date": "2025-07-23", "name": "方正科技(蓄势)"},
        {"code": "688321_SH", "date": "2025-06-20", "name": "微芯生物(双底)"},
        {"code": "002940_SZ", "date": "2025-07-11", "name": "昂利康(压轴)"},
        {"code": "301076_SZ", "date": "2025-08-01", "name": "新瀚新材(激进)"},
        {"code": "600184_SH", "date": "2025-07-10", "name": "光电股份(回踩)"},
        {"code": "002074_SZ", "date": "2025-08-01", "name": "国轩高科(趋势)"},
        {"code": "605378_SH", "date": "2025-07-31", "name": "野马电池(突破)"},
        {"code": "600366_SH", "date": "2025-08-06", "name": "宁波韵升(反包)"}
    ]
    def verify_perfect_cases(df_signals: pl.LazyFrame, cases_config: list):
        print("🔍 [Audit] 启动十大完美案例专项验证...")

        # 1. 构造案例查询表
        # 注意：这里我们假设 df_signals 包含了全量计算数据
        # 我们需要重新计算未来 30 天的数据，因为之前的 df_result 可能被 filter 掉了

        # 构造 Polars DataFrame 用于 Join
        target_df = pl.DataFrame(cases_config).with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # 2. 扩展计算未来 30 天的收益数据
        # 我们直接在 df_signals 上通过 code 和 date 关联，不需要全量重算
        # 但为了获取未来数据，我们需要先在 df_signals 里 shift

        print("⏳ 正在回溯历史行情 (T+1 -> T+30)...")

        # 定义评估周期
        horizons = [5, 10, 15, 20, 25, 30]

        # 核心验证逻辑
        audit_df = (
            df_signals
            .sort(["code", "date"])
            .with_columns([
                # 买入价：T+1 开盘价 (实战标准)
                pl.col("open_adj").shift(-1).over("code").alias("audit_buy_price"),

                # 择时状态 (复用之前的逻辑)
                # 这里简单起见，我们直接检查 b1_signal 字段
            ])
        )

        # 动态生成不同周期的收益列
        exprs = []
        for h in horizons:
            # 持有涨幅: (Close_T+N - Buy) / Buy
            exprs.append(
                ((pl.col("close_adj").shift(-h).over("code") / pl.col("audit_buy_price")) - 1).alias(f"ret_{h}d")
            )
            # 最高涨幅: (Max_High_T+1_to_T+N - Buy) / Buy
            # rolling_max(h) 往前看，shift(-h) 移到现在
            exprs.append(
                ((pl.col("high_adj").rolling_max(h).shift(-h).over("code") / pl.col("audit_buy_price")) - 1).alias(f"max_{h}d")
            )

        # 执行计算并关联目标
        result = (
            audit_df
            .with_columns(exprs)
            .join(target_df.lazy(), on=["code", "date"], how="inner") # 只保留这10个
            .collect()
        )

        # 3. 输出报表
        print("\n====== ✨ 十大完美案例验证报告 ✨ ======")
        print(f"{'名称':<10} | {'日期':<9} | {'代码':<8} | {'信号?':<5} | {'买入价':<6} | {'5日最高':<8} | {'10日最高':<8} | {'20日最高':<8} | {'30日最高':<8} | {'30日持有':<8}")
        print("-" * 120)

        for row in result.iter_rows(named=True):
            name = row['name']
            code = row['code']
            date_str = row['date']
            is_signal = "✅" if row['b1_signal'] else "❌"
            buy = row['audit_buy_price']

            # 格式化涨幅
            def fmt(val): return f"{val*100:>6.2f}%" if val is not None else "   N/A"

            # 打印核心行
            print(f"{name:<10} | {date_str}  | {code:<8} | {is_signal:<5} | {buy:<6.2f} | {fmt(row['max_5d']):<8} | {fmt(row['max_10d']):<8} | {fmt(row['max_20d']):<8} | {fmt(row['max_30d']):<8} | {fmt(row['ret_30d']):<8}")

            # 如果没选出来，打印原因
            if not row['b1_signal']:
                # 简单诊断一下原因
                reasons = []
                if not row['J_OK']: reasons.append(f"J值({row['J']:.1f})>13")
                if not row['MAX28_OK']: reasons.append("有天量阴")
                if not row['GOOD28']: reasons.append("有坏K线")
                if not row['YANGYIN_OK']: reasons.append(f"红绿比不足, p1: {row["vol_yang_p1"]/row["vol_yin_p1"]}, p2: {row["vol_yang_p2"]/row["vol_yin_p2"]}")
                if not row['TRIGGER']: reasons.append("无关键K")
                print(f"   ⚠️ 落选原因: {', '.join(reasons)}")

        print("-" * 120)


    # ==============================================================================
    # 执行验证
    # ==============================================================================
    # 假设 df_signals 依然在内存中 (即 run_strategy_b_with_manual_regime 的输入)
    if 'df_signals' in locals():
        verify_perfect_cases(df_signals, PERFECT_CASES_CONFIG)
    else:
        print("⚠️ df_signals 不在内存中，请先运行 Step 3 的 calc_b1_factors")
    return


@app.cell
def _(datetime, df_result_dynamic, pl):
    hn_df = df_result_dynamic.filter(
        (pl.col("code") == "688799_SH") &
        (pl.col("date") >= datetime(2025, 5, 1))
    )

    hn_df.select([
        'date',
        'ret_5d_raw',
        'ret_10d_raw',
        'ret_15d_raw',
        'ret_20d_raw',
        'ret_25d_raw',
        'ret_30d_raw',
    ])
    return


if __name__ == "__main__":
    app.run()
