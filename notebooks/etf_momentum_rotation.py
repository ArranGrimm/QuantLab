import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # -*- coding: utf-8 -*-
    import vectorbt as vbt
    import numpy as np
    import pandas as pd
    import numba
    from loguru import logger
    import sys
    import os
    import akshare as ak # <--- 新增akshare导入

    # --- 全局配置 ---
    strategy_name = "ETF_Momentum_Rotation_Akshare" # <--- 名称稍作修改以示区分
    etf_symbols_original = ['518880.SH', '513100.SH', '513130.SH', '159915.SZ'] # 黄金, 纳指, 恒生科技, 创业板
    # ETF代码给akshare使用 (不带后缀)
    etf_codes_for_akshare = [s.split('.')[0] for s in etf_symbols_original]

    start_date = '2020-01-01'
    end_date = '2026-01-29'



    data_directory = "Data_Akshare" # 如果需要本地缓存akshare数据 (本脚本直接在线获取)
    output_dir = "results"

    # --- 日志配置 ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except OSError as e:
            print(f"创建目录 {output_dir} 失败: {e}")
            sys.exit(1)

    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, level="INFO", format=log_format)
    log_file_path = os.path.join(output_dir, f"{strategy_name}_backtest.log")
    logger.add(log_file_path, level="DEBUG", rotation="10 MB", format=log_format)

    logger.info(f"------ {strategy_name} 回测开始 ------")
    logger.info(f"策略名称: {strategy_name}")
    logger.info(f"原始ETF池: {etf_symbols_original}")

    # --- 策略参数 ---
    lookback_period = 25
    annual_trading_days = 250
    initial_cash = 100000
    commission_rate = 0.0002
    slippage_rate = 0.0005

    # --- 辅助函数：动量计算 (Numba优化) ---
    @numba.njit
    def nb_calc_momentum_score(i, col, y_arr, annual_days): # 签名保持不变，接收vectorbt传入的参数
        n = len(y_arr)
        if np.isnan(y_arr).any() or n < 2: 
            return np.nan

        log_y = np.log(y_arr)
        x = np.arange(n)

        # --- 开始修改：引入线性递增权重 ---
        weights = np.linspace(1.0, 2.0, n) # 权重从1线性增加到2

        # 加权线性回归计算 (参考Numpy的polyfit对于加权w的实现思路，或直接实现加权最小二乘法)
        # Numba环境下直接用np.polyfit(x, y, 1, w=weights)可能不支持或效率不高
        # 我们需要手动实现加权最小二乘法的斜率和截距计算

        w_sum = np.sum(weights)
        w_x_sum = np.sum(weights * x)
        w_y_sum = np.sum(weights * log_y)
        w_x2_sum = np.sum(weights * x**2)
        w_xy_sum = np.sum(weights * x * log_y)

        denominator = w_sum * w_x2_sum - w_x_sum**2
        if denominator == 0: 
            return np.nan

        slope = (w_sum * w_xy_sum - w_x_sum * w_y_sum) / denominator
        intercept = (w_y_sum - slope * w_x_sum) / w_sum # 也可以是 (w_x2_sum * w_y_sum - w_x_sum * w_xy_sum) / denominator

        # 计算加权R²
        y_pred = slope * x + intercept
        weighted_residuals_sq = weights * (log_y - y_pred)**2

        # 加权均值
        weighted_mean_y = np.sum(weights * log_y) / w_sum
        weighted_ss_tot = np.sum(weights * (log_y - weighted_mean_y)**2)

        if weighted_ss_tot == 0: # 如果加权总平方和为0 (例如加权后的y值恒定)
            r_squared = 0.0 # 或1.0，取决于定义。如果预测完美，残差为0，R^2应为1。若y本身无波动，R^2通常无意义或为0。
                            # MarioC代码中是 np.sum(weights * (y - np.mean(y))**2)，这里用加权均值更一致。
                            # 如果 y 值本身恒定（即使加权后），weighted_ss_tot 会是0，导致除零。
                            # 如果 y 值恒定，log_y也恒定，那么slope会是0，y_pred也是恒定等于log_y。
                            # 此时 weighted_residuals_sq 会是0。如果 weighted_ss_tot 也是0，R^2可以定义为1（完美拟合常数）或0（无趋势）。
                            # 我们参考MarioC的实现，如果 y - np.mean(y) 部分加权后为0，则R^2可能出问题。
                            # 安全起见，如果 weighted_ss_tot 为0，且 weighted_residuals_sq 也为0，说明完美拟合常数，R^2=1。
                            # 如果 weighted_ss_tot 为0，但 weighted_residuals_sq 不为0（理论上不太可能），则R^2未定义或为0。
                            # 简单处理：若 weighted_ss_tot 为0，则 r_squared 为0 (表示没有可解释的方差，或趋势不明显)
            r_squared = 0.0
        else:
            r_squared = 1.0 - (np.sum(weighted_residuals_sq) / weighted_ss_tot)
            r_squared = max(0.0, r_squared) # 确保R²不为负

        # --- 修改结束 ---

        daily_factor = np.exp(slope)
        annualized_returns = daily_factor**annual_days - 1.0
        score = annualized_returns * r_squared

        if not np.isfinite(score): return np.nan
        return score

    return (
        ak,
        annual_trading_days,
        commission_rate,
        end_date,
        etf_codes_for_akshare,
        etf_symbols_original,
        initial_cash,
        logger,
        lookback_period,
        nb_calc_momentum_score,
        np,
        pd,
        slippage_rate,
        start_date,
        sys,
        vbt,
    )


@app.cell
def _(
    ak,
    annual_trading_days,
    commission_rate,
    end_date,
    etf_codes_for_akshare,
    etf_symbols_original,
    initial_cash,
    logger,
    lookback_period,
    nb_calc_momentum_score,
    np,
    pd,
    slippage_rate,
    start_date,
    sys,
    vbt,
):
    # --- 1. 加载数据 (使用 Akshare) ---
    logger.info(f"开始使用 akshare 加载数据，日期范围: {start_date} to {end_date}")

    # 转换日期格式为akshare需要的 'YYYYMMDD'
    ak_start_date = start_date.replace('-', '')
    ak_end_date = end_date.replace('-', '')

    all_etf_close_prices = {} # 存储每个ETF的收盘价Series，键为原始带后缀的symbol

    for i, ak_code in enumerate(etf_codes_for_akshare):
        original_symbol = etf_symbols_original[i] # 获取对应的原始带后缀的名称
        logger.debug(f"尝试从 akshare 加载 {original_symbol} (代码: {ak_code})...")
        try:
            etf_hist_df = ak.fund_etf_hist_em(
                symbol=ak_code,
                period="daily",
                start_date=ak_start_date,
                end_date=ak_end_date,
                adjust="hfq"  # 使用后复权数据
            )

            if etf_hist_df.empty:
                logger.warning(f"akshare 未返回 {original_symbol} (代码: {ak_code}) 的数据。")
                continue

            # 重命名列
            etf_hist_df.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume'
                # 其他列如 '成交额', '振幅', '涨跌幅', '涨跌额', '换手率' 可按需保留或重命名
            }, inplace=True)

            # 将'Date'列转换为datetime对象并设为索引
            etf_hist_df['Date'] = pd.to_datetime(etf_hist_df['Date'])
            etf_hist_df.set_index('Date', inplace=True)

            # 确保关键列是数值类型
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in etf_hist_df.columns:
                    etf_hist_df[col] = pd.to_numeric(etf_hist_df[col], errors='coerce')
                elif col == 'Volume': # 如果akshare没返回成交量，则补0
                     etf_hist_df[col] = 0
                else: # O,H,L,C是必须的
                    logger.error(f"ETF {original_symbol} 的akshare数据缺少关键列: {col}")
                    raise ValueError(f"ETF {original_symbol} 数据不完整")


            # 仅保留我们需要的 'Close' 列，并以原始带后缀的symbol命名
            if 'Close' in etf_hist_df.columns:
                all_etf_close_prices[original_symbol] = etf_hist_df['Close']
                logger.success(f"成功加载并处理了 {original_symbol} (代码: {ak_code}) 的数据。")
            else:
                logger.warning(f"ETF {original_symbol} 处理后缺少 'Close' 列。")

        except Exception as e_ak:
            logger.error(f"使用 akshare 加载 {original_symbol} (代码: {ak_code}) 时出错: {e_ak}")

    if not all_etf_close_prices:
        logger.error("未能从 akshare 加载任何有效的ETF数据。请检查ETF代码、日期范围和网络连接。")
        sys.exit(1)

    # 合并所有ETF的收盘价到一个DataFrame
    price_df = pd.DataFrame(all_etf_close_prices)

    # 数据清洗：填充周末等缺失的交易日数据，然后删除完全没有数据的行/列
    # 先基于第一个有效ETF的索引重新索引所有列，然后填充
    if not price_df.empty:
        base_index = price_df.index # 使用合并后的索引，它应该包含了所有交易日
        price_df = price_df.reindex(base_index).ffill() # 向前填充
        price_df = price_df.dropna(axis=0, how='all') # 删除所有列都为NaN的行（通常不会发生在此处）
        price_df = price_df.dropna(axis=1, how='all') # 删除数据加载后完全是NaN的ETF列
        # 进一步确保开始阶段没有NaN（动量计算对NaN敏感）
        # price_df = price_df.dropna(axis=0, how='any', subset=price_df.columns[:1]) # 基于第一个ETF确保起始数据完整

    if price_df.empty or price_df.shape[1] < 2: # 需要至少两个ETF才能轮动
        logger.error(f"数据加载或处理后，有效的ETF数量 ({price_df.shape[1]}) 少于2个，无法进行轮动。")
        sys.exit(1)

    # 更新实际参与轮动的ETF列表名称
    final_etf_symbols = price_df.columns.tolist()
    logger.info(f"最终使用的数据范围: {price_df.index.min()} 到 {price_df.index.max()}. 共 {len(price_df)} 行.")
    logger.info(f"最终参与轮动的ETF: {final_etf_symbols}")

    # --- 2. 计算动量得分 ---
    logger.info(f"开始计算动量得分，回顾期: {lookback_period} 天...")
    try:
        momentum_scores = price_df.vbt.rolling_apply(
            lookback_period,              # window (positional)
            nb_calc_momentum_score,       # apply_func_nb (positional)
            annual_trading_days           # *args (positional)
            # Removed args=, use_raw=, cache_func= based on documentation
        )
        logger.success("动量得分计算完成。")
    except Exception as e:
        logger.error(f"计算动量得分时出错: {e}")
        sys.exit(1)

    # --- 3. 生成交易信号
    logger.info("根据动量得分生成目标权重 (安全摸狗逻辑)...")
    try:
        # --- 开始修改：加入安全区间过滤 ---
        # 1. 复制一份动量得分用于操作，避免修改原始的 momentum_scores
        filtered_momentum_scores = momentum_scores.copy()

        # 2. 应用安全区间过滤条件: (0, 5]
        # 将不符合条件的得分设为负无穷小或NaN，这样它们在后续排名中会排在最后或被忽略
        # MarioC的逻辑是直接筛选DataFrame，我们这里通过修改得分来实现类似效果，以便后续排名
        condition = (filtered_momentum_scores > 0) & (filtered_momentum_scores <= 5)
        # 对于不满足条件的，我们给一个非常低的值，确保它们不会被选中
        # 或者，可以直接将它们设为NaN，rank函数会处理NaN
        filtered_momentum_scores[~condition] = np.nan 

        logger.info(f"应用安全区间 (0, 5] 过滤后的动量得分 (部分显示NaN为不合格):\n{filtered_momentum_scores.tail()}")

        # 3. 对过滤后的得分进行排名
        #   如果所有ETF都被过滤掉（即filtered_momentum_scores该行全是NaN），则ranks对应行也会是NaN
        ranks = filtered_momentum_scores.rank(axis=1, ascending=False, method='first')

        # 4. 生成目标权重：选择排名第一的（如果存在）
        #   如果某行ranks全是NaN，那么np.where(ranks == 1,...) 的结果在该行仍将是False(0.0)
        target_weights = pd.DataFrame(np.where(ranks == 1, 1.0, 0.0),
                                      index=price_df.index, 
                                      columns=price_df.columns)

        # 检查是否有任何一天选出了ETF (即是否有权重为1的情况)
        # 如果某天 target_weights.sum(axis=1) == 0，则表示当天无ETF可选，即空仓
        days_with_selection = target_weights.sum(axis=1) > 0
        logger.info(f"在 {days_with_selection.sum()} 天中选出了ETF进行持有 (共 {len(days_with_selection)} 个交易日)。")
        if days_with_selection.sum() == 0:
            logger.warning("警告：在整个回测期间，根据安全摸狗逻辑，没有选出任何ETF持有！请检查过滤条件或市场状况。")

        # --- 修改结束 ---
        logger.success("目标权重生成完成。")
    except Exception as e:
        logger.error(f"生成目标权重时出错: {e}")
        sys.exit(1)

    # 以黄金ETF (518880.SH) 为基准
    if '518880.SH' in price_df.columns:
        gold_etf_prices = price_df['518880.SH'].copy()
        # 确保基准价格序列与投资组合的索引对齐 (通常在price_df层面已经对齐)
        # gold_etf_prices = gold_etf_prices.reindex(portfolio.wrapper.index, method='ffill').fillna(method='bfill') # 对齐索引并填充
        benchmark_gold_returns = gold_etf_prices.vbt.to_returns()
        logger.info("已准备黄金ETF (518880.SH) 作为基准的收益率数据。")
    else:
        logger.warning("ETF池中未找到黄金ETF (518880.SH)，无法将其设为基准。")
        benchmark_gold_returns = None

    # 以纳指ETF (513100.SH) 为基准
    if '513100.SH' in price_df.columns:
        nasdaq_etf_prices = price_df['513100.SH'].copy()
        # nasdaq_etf_prices = nasdaq_etf_prices.reindex(portfolio.wrapper.index, method='ffill').fillna(method='bfill') # 对齐索引并填充
        benchmark_nasdaq_returns = nasdaq_etf_prices.vbt.to_returns()
        logger.info("已准备纳指ETF (513100.SH) 作为基准的收益率数据。")
    else:
        logger.warning("ETF池中未找到纳指ETF (513100.SH)，无法将其设为基准。")
        benchmark_nasdaq_returns = None


    vbt.settings['portfolio']['stats']['settings']['benchmark_rets'] = benchmark_nasdaq_returns

    # --- 4. 执行组合回测 ---
    logger.info(f"开始执行VectorBT组合回测，初始资金: {initial_cash:.2f}...")
    try:
        portfolio = vbt.Portfolio.from_orders(
            close=price_df, size=target_weights, size_type='targetpercent',
            group_by=True, cash_sharing=True, fees=commission_rate,
            slippage=slippage_rate, freq='D'
        )
        logger.success("组合回测执行完毕。")
    except Exception as e:
        logger.error(f"执行组合回测时出错: {e}")
        sys.exit(1)
    # 在 portfolio 回测完成之后，生成统计和绘图之前
    return (portfolio,)


@app.cell
def _(portfolio):
    portfolio.stats()
    return


@app.cell
def _(portfolio):
    orders_df = portfolio.orders.records_readable
    orders_df[orders_df['Timestamp'] >= '2025-12-01 15:00:00']
    return


@app.function
def analyze_year_return(portfolio):
    # =========================
    # 深度分析：年度收益 & 关键时期分析
    # =========================

    print("=" * 60)
    print("          ETF轮动策略深度分析报告")
    print("=" * 60)

    # 1. 每年年底的真实收益率分析
    print("\n📊 1. 年度收益率分析 (每年12月31日)")
    print("-" * 50)

    # 获取组合价值时间序列
    portfolio_value = portfolio.value()
    yearly_returns = {}

    # 定义年份和对应的年底日期
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    year_end_dates = ['2020-12-31', '2021-12-31', '2022-12-30', '2023-12-29', '2024-12-31', '2025-12-31', '2026-01-28']  # 最后一个是当前

    start_value = portfolio_value.iloc[0]
    print(f"初始投资金额: ${start_value:,.2f}")
    print()

    for i, year in enumerate(years):
        try:
            if i == 0:  # 2020年
                year_start_value = start_value
                year_end_date = year_end_dates[i]
            else:
                year_start_date = year_end_dates[i-1]
                year_end_date = year_end_dates[i]
                # 获取年初价值
                year_start_value = portfolio_value.loc[portfolio_value.index <= year_start_date].iloc[-1]

            # 获取年末价值
            if year == 2025:
                year_end_value = portfolio_value.iloc[-1]  # 当前最新值
                print(f"{year}年 (截至6月27日):")
            else:
                year_end_value = portfolio_value.loc[portfolio_value.index <= year_end_date].iloc[-1]
                print(f"{year}年:")

            annual_return = (year_end_value / year_start_value - 1) * 100
            yearly_returns[year] = annual_return

            print(f"  起始价值: ${year_start_value:,.2f}")
            print(f"  结束价值: ${year_end_value:,.2f}")
            print(f"  年度收益率: {annual_return:+.2f}%")
            print()

        except Exception as e:
            print(f"{year}年数据获取失败: {e}")

    # 2. 2024年9-10月暴涨分析
    print("\n🚀 2. 2024年9-10月暴涨期分析")
    print("-" * 50)

    # 分析2024年9月到10月的情况
    sept_oct_start = '2024-09-01'
    sept_oct_end = '2024-11-30'

    # 获取这个时期的组合价值
    sept_oct_value = portfolio_value.loc[sept_oct_start:sept_oct_end]
    if len(sept_oct_value) > 0:
        period_start_value = sept_oct_value.iloc[0]
        period_end_value = sept_oct_value.iloc[-1]
        period_return = (period_end_value / period_start_value - 1) * 100

        print(f"时期: {sept_oct_start} 到 {sept_oct_end}")
        print(f"期初价值: ${period_start_value:,.2f}")
        print(f"期末价值: ${period_end_value:,.2f}")
        print(f"期间收益率: {period_return:+.2f}%")
        print()

    # 获取这个时期的交易记录
    orders_records = portfolio.orders.records_readable
    sept_oct_trades = orders_records[
        (orders_records['Timestamp'] >= sept_oct_start) & 
        (orders_records['Timestamp'] <= sept_oct_end)
    ]

    print("📈 期间交易记录:")
    if len(sept_oct_trades) > 0:
        for _, trade in sept_oct_trades.iterrows():
            print(f"  {trade['Timestamp'].strftime('%Y-%m-%d')}: {trade['Side']} {trade['Column']} - "
                  f"价格: ¥{trade['Price']:.2f}, 数量: {trade['Size']:.0f}")
    else:
        print("  无交易记录")

    # 获取持仓信息
    print(f"\n📊 期间持仓分析:")
    try:
        # 从交易记录推断主要持仓
        if len(sept_oct_trades) > 0:
            print("期间主要交易的ETF:")
            for etf in sept_oct_trades['Column'].unique():
                etf_trades = sept_oct_trades[sept_oct_trades['Column'] == etf]
                buy_trades = etf_trades[etf_trades['Side'] == 'Buy']
                sell_trades = etf_trades[etf_trades['Side'] == 'Sell']

                etf_name = {
                    '518880.SH': '黄金ETF',
                    '513100.SH': '纳指ETF', 
                    '513130.SH': '恒生科技ETF',
                    '159915.SZ': '创业板ETF'
                }.get(etf, etf)

                net_shares = buy_trades['Size'].sum() - sell_trades['Size'].sum()
                total_value = (buy_trades['Size'] * buy_trades['Price']).sum()

                print(f"  {etf_name}: 净持仓变化 {net_shares:.0f}股, 交易金额 ¥{total_value:,.0f}")
        else:
            print("期间无交易，持仓无变化")

        # 尝试获取期间的组合权重
        try:
            asset_value = portfolio.asset_value()
            total_value_series = portfolio.value()

            period_asset_value = asset_value.loc[sept_oct_start:sept_oct_end]
            period_total_value = total_value_series.loc[sept_oct_start:sept_oct_end]

            if len(period_asset_value) > 0 and len(period_total_value) > 0:
                # 计算权重
                period_weights = period_asset_value.div(period_total_value, axis=0)

                print(f"\n期间平均持仓权重:")
                for col in period_weights.columns:
                    col_weights = period_weights[col]
                    non_zero_weights = col_weights[col_weights > 0.01]  # 权重大于1%
                    if len(non_zero_weights) > 0:
                        avg_weight = non_zero_weights.mean()
                        etf_name = {
                            '518880.SH': '黄金ETF',
                            '513100.SH': '纳指ETF', 
                            '513130.SH': '恒生科技ETF',
                            '159915.SZ': '创业板ETF'
                        }.get(col, col)
                        print(f"  {etf_name}: {avg_weight:.1%}")
        except Exception as weight_e:
            print(f"权重计算失败: {weight_e}")

    except Exception as e:
        print(f"持仓分析出错: {e}")

    # 3. 2025年持仓分析 - 黄金和纳指的占比
    print("\n🥇 3. 2025年持仓分析 (黄金 vs 纳指)")
    print("-" * 50)

    # 分析2025年的持仓情况
    year_2025_start = '2025-01-01'
    year_2025_trades = orders_records[orders_records['Timestamp'] >= year_2025_start]

    print("🔄 2025年交易记录:")
    if len(year_2025_trades) > 0:
        # 按ETF分类统计
        etf_trades_count = year_2025_trades.groupby('Column')['Side'].count()
        etf_trade_value = year_2025_trades.groupby('Column').apply(
            lambda x: (x['Size'] * x['Price']).sum()
        )

        print("各ETF交易次数和金额:")
        for etf in etf_trades_count.index:
            trade_count = etf_trades_count[etf]
            trade_value = etf_trade_value[etf]
            etf_name = {
                '518880.SH': '黄金ETF',
                '513100.SH': '纳指ETF', 
                '513130.SH': '恒生科技ETF',
                '159915.SZ': '创业板ETF'
            }.get(etf, etf)
            print(f"  {etf_name} ({etf}): {trade_count}次交易, 总金额: ¥{trade_value:,.0f}")

        print(f"\n📅 2025年详细交易时间线:")
        for _, trade in year_2025_trades.iterrows():
            etf_name = {
                '518880.SH': '黄金ETF',
                '513100.SH': '纳指ETF', 
                '513130.SH': '恒生科技ETF',
                '159915.SZ': '创业板ETF'
            }.get(trade['Column'], trade['Column'])
            print(f"  {trade['Timestamp'].strftime('%Y-%m-%d')}: {trade['Side']} {etf_name} - "
                  f"价格: ¥{trade['Price']:.2f}")
    else:
        print("  2025年无交易记录")

    # 分析当前持仓
    print(f"\n💼 当前持仓状况 (截至 {portfolio_value.index[-1].strftime('%Y-%m-%d')}):")
    try:
        # 获取最新的持仓
        current_positions = portfolio.positions.records_readable
        if len(current_positions) > 0:
            active_positions = current_positions[current_positions['Status'] == 'Open']
            if len(active_positions) > 0:
                for _, pos in active_positions.iterrows():
                    etf_name = {
                        '518880.SH': '黄金ETF',
                        '513100.SH': '纳指ETF', 
                        '513130.SH': '恒生科技ETF',
                        '159915.SZ': '创业板ETF'
                    }.get(pos['Column'], pos['Column'])
                    print(f"  持有 {etf_name}: 数量 {pos['Size']:.0f}, 当前价值 ¥{pos['Size'] * pos['Exit Price']:.0f}")
            else:
                print("  当前无持仓")
        else:
            print("  无持仓记录")
    except Exception as e:
        print(f"当前持仓分析出错: {e}")

    # 总结分析
    print(f"\n📋 4. 关键发现总结")
    print("-" * 50)
    print("• 年度收益率显示策略的表现波动性")
    if 2024 in yearly_returns and yearly_returns[2024] > 50:
        print("• 2024年确实是表现突出的一年，收益率超过50%")
    print("• 从交易记录可以看出策略的轮动特征")
    print("• 2025年的交易主要集中在哪些ETF上")

    print("\n" + "=" * 60)


@app.cell
def _(portfolio):
    analyze_year_return(portfolio)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
