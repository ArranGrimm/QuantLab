# -*- coding: utf-8 -*-
import vectorbt as vbt
import numpy as np
import pandas as pd
import numba
from loguru import logger
import sys
import os
from datetime import datetime
import akshare as ak # <--- 新增akshare导入

# --- 全局配置 ---
strategy_name = "ETF_Momentum_Rotation_Akshare" # <--- 名称稍作修改以示区分
etf_symbols_original = ['518880.SH', '513100.SH', '513130.SH', '159915.SZ'] # 黄金, 纳指, 恒生科技, 创业板
# ETF代码给akshare使用 (不带后缀)
etf_codes_for_akshare = [s.split('.')[0] for s in etf_symbols_original]

start_date = '2025-01-01'
end_date = '2025-05-12'

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
slippage_rate = 0.000

# --- 辅助函数：动量计算 (Numba优化) ---
@numba.njit
def nb_calc_momentum_score(i, col, y_arr, annual_days):
    n = len(y_arr)
    if np.isnan(y_arr).any() or n < 2: return np.nan
    log_y = np.log(y_arr)
    x = np.arange(n)
    sum_x, sum_y, sum_x2, sum_xy = np.sum(x), np.sum(log_y), np.sum(x**2), np.sum(x * log_y)
    denominator = (n * sum_x2 - sum_x**2)
    if denominator == 0: return np.nan
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    mean_y = np.mean(log_y)
    ss_tot = np.sum((log_y - mean_y)**2)
    if ss_tot == 0: r_squared = 0.0
    else:
        intercept = (sum_y - slope * sum_x) / n
        y_pred = slope * x + intercept
        ss_res = np.sum((log_y - y_pred)**2)
        r_squared = 1.0 - (ss_res / ss_tot)
        r_squared = max(0.0, r_squared)
    daily_factor = np.exp(slope)
    annualized_returns = daily_factor**annual_days - 1.0
    score = annualized_returns * r_squared
    if not np.isfinite(score): return np.nan
    return score

# --- 主执行程序 ---
if __name__ == "__main__":
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
        price_df = price_df.dropna(axis=0, how='any', subset=price_df.columns[:1]) # 基于第一个ETF确保起始数据完整

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

    # --- 3. 生成交易信号 (目标权重) ---
    logger.info("根据动量得分生成目标权重 (持有Top 1)...")
    try:
        ranks = momentum_scores.rank(axis=1, ascending=False, method='first')
        target_weights = pd.DataFrame(np.where(ranks == 1, 1.0, 0.0),
                                      index=price_df.index,
                                      columns=price_df.columns) # 列名应与price_df一致
        logger.success("目标权重生成完成。")
    except Exception as e:
        logger.error(f"生成目标权重时出错: {e}")
        sys.exit(1)

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

    # 以黄金ETF (518880.SH) 为基准
    if '518880.SH' in price_df.columns:
        gold_etf_prices = price_df['518880.SH'].copy()
        # 确保基准价格序列与投资组合的索引对齐 (通常在price_df层面已经对齐)
        gold_etf_prices = gold_etf_prices.reindex(portfolio.wrapper.index, method='ffill').fillna(method='bfill') # 对齐索引并填充
        benchmark_gold_returns = gold_etf_prices.vbt.to_returns()
        logger.info("已准备黄金ETF (518880.SH) 作为基准的收益率数据。")
    else:
        logger.warning("ETF池中未找到黄金ETF (518880.SH)，无法将其设为基准。")
        benchmark_gold_returns = None

    # 以纳指ETF (513100.SH) 为基准
    if '513100.SH' in price_df.columns:
        nasdaq_etf_prices = price_df['513100.SH'].copy()
        nasdaq_etf_prices = nasdaq_etf_prices.reindex(portfolio.wrapper.index, method='ffill').fillna(method='bfill') # 对齐索引并填充
        benchmark_nasdaq_returns = nasdaq_etf_prices.vbt.to_returns()
        logger.info("已准备纳指ETF (513100.SH) 作为基准的收益率数据。")
    else:
        logger.warning("ETF池中未找到纳指ETF (513100.SH)，无法将其设为基准。")
        benchmark_nasdaq_returns = None

    # --- 5. 显示与保存结果 ---
    logger.info("------ 回测结果 ------")
    stats_df = portfolio.stats()
    print(stats_df)
    stats_output_path = os.path.join(output_dir, f"{strategy_name}_stats.txt")
    try:
        with open(stats_output_path, 'w', encoding='utf-8') as f: # 添加utf-8编码
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Symbols: {final_etf_symbols}\n") # 使用最终的ETF列表
            f.write(f"Date Range: {price_df.index.min()} to {price_df.index.max()}\n")
            f.write(f"Lookback Period: {lookback_period}\n")
            f.write("-" * 30 + "\n")
            f.write(stats_df.to_string())
        logger.info(f"回测统计数据已保存到: {stats_output_path}")
    except Exception as e:
        logger.error(f"保存统计数据文件时出错: {e}")

    trades_df = portfolio.orders.records_readable
    trades_output_path = os.path.join(output_dir, f"{strategy_name}_trades_vectorbt.csv")
    try:
        trades_df.to_csv(trades_output_path, index=False, encoding='utf-8-sig') #确保CSV中文正确显示
        logger.info(f"交易记录已保存到: {trades_output_path}")
    except Exception as e:
        logger.error(f"保存交易记录时出错: {e}")

    report_path = os.path.join(output_dir, f"{strategy_name}_vectorbt_report.html")
    logger.info(f"正在生成交互式HTML报告到: {report_path} ...")
    try:
        fig = portfolio.plot()
        fig.write_html(report_path)
        logger.success(f"交互式报告已生成: {report_path}")
    except Exception as e:
        logger.error(f"生成HTML报告时出错: {e}")

    logger.info(f"------ {strategy_name} 回测结束 ------")

    if benchmark_gold_returns is not None:
        logger.info("\n------ 回测结果 (对比 黄金ETF 518880.SH) ------")
        stats_vs_gold = portfolio.stats(settings=dict(benchmark_rets=benchmark_gold_returns))
        print(stats_vs_gold)
        stats_gold_path = os.path.join(output_dir, f"{strategy_name}_stats_vs_Gold.txt")
        with open(stats_gold_path, 'w', encoding='utf-8') as f:
            f.write(stats_vs_gold.to_string())
        logger.info(f"对比黄金ETF的统计数据已保存到: {stats_gold_path}")

        # plot_vs_gold_path = os.path.join(output_dir, f"{strategy_name}_plot_vs_Gold.html")
        # fig_vs_gold = portfolio.plot(settings=dict(benchmark_rets=benchmark_gold_returns))
        # fig_vs_gold.write_html(plot_vs_gold_path)
        # logger.info(f"对比黄金ETF的交互式报告已生成: {plot_vs_gold_path}")

    if benchmark_nasdaq_returns is not None:
        logger.info("\n------ 回测结果 (对比 纳指ETF 513100.SH) ------")
        stats_vs_nasdaq = portfolio.stats(settings=dict(benchmark_rets=benchmark_nasdaq_returns))
        print(stats_vs_nasdaq)
        stats_nasdaq_path = os.path.join(output_dir, f"{strategy_name}_stats_vs_Nasdaq.txt")
        with open(stats_nasdaq_path, 'w', encoding='utf-8') as f:
            f.write(stats_vs_nasdaq.to_string())
        logger.info(f"对比纳指ETF的统计数据已保存到: {stats_nasdaq_path}")

        # plot_vs_nasdaq_path = os.path.join(output_dir, f"{strategy_name}_plot_vs_Nasdaq.html")
        # fig_vs_nasdaq = portfolio.plot(settings=dict(benchmark_rets=benchmark_nasdaq_returns))
        # fig_vs_nasdaq.write_html(plot_vs_nasdaq_path)
        # logger.info(f"对比纳指ETF的交互式报告已生成: {plot_vs_nasdaq_path}")