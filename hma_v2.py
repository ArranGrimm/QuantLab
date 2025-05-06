import pandas as pd
import talib
import math
import numpy as np
from loguru import logger
import sys
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, FractionalBacktest # backtesting.py提供的有用工具，虽然此策略未使用
# from bokeh.plotting import save, output_file # 如果保存绘图需要特定的 Bokeh 配置，请取消注释

# --- 全局配置 ---
target_name = "BTC"  # 例如: "CSI300", "AAPL", "BTC" 等
data_directory = "data" # 数据目录
csv_file_path = os.path.join(data_directory, f"{target_name}.csv") # CSV文件完整路径
output_dir = "results" # 保存报告和交易记录的目录

# --- 日志配置 ---
logger.remove()
logger.add(sys.stderr, level="INFO")
# 如果输出目录不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"已创建输出目录: {output_dir}")
# 添加日志文件输出
logger.add(os.path.join(output_dir, f"{target_name}_backtest.log"), level="DEBUG", rotation="10 MB")


# --- 策略参数 ---
# 这些参数稍后将传递给策略类
HULL_LEN = 55                 # Hull MA 周期
SMA_LEN = 130                 # SMA 周期
TRAIL_PCT = 0.05              # 追踪止损百分比 (例如 0.05 代表 5%)
EQUITY_FRACTION_PER_TRADE = 0.95 # 每笔交易使用的权益比例 (例如 0.95 代表 95%)
COMMISSION_PCT = 0.001        # 每笔交易的手续费率 (backtesting.py 会在进出场时应用)
# SLIPPAGE_PCT = 0.0005       # 滑点百分比 - backtesting.py 没有直接的滑点参数，可通过手续费模拟或假设成交价
INITIAL_EQUITY = 100000       # 初始资金

# --- HMA 计算辅助函数 ---
# 这个函数将在 backtesting.py 策略内部使用
def HMA_func(values, n):
    """
    计算赫尔移动平均线 (Hull Moving Average, HMA).
    需要 TA-Lib 库.
    输入: pandas Series 或 numpy 数组 (价格序列), 周期 n
    输出: HMA 的 numpy 数组
    """
    # 检查是否有足够的数据进行计算
    if len(values) < n + int(math.sqrt(n)):
        # 对于 backtesting.py，返回与输入值形状相同的 NaN 数组
        return np.full_like(values, np.nan)
    try:
        period_half = int(n / 2)      # HMA 参数: n/2 周期
        period_sqrt = int(math.sqrt(n)) # HMA 参数: sqrt(n) 周期
        # 计算 WMA
        wma_half = talib.WMA(values, timeperiod=period_half)
        wma_full = talib.WMA(values, timeperiod=n)
        # 确保中间计算不会因 WMA 初始的 NaN 值而失败
        hma_intermediate = 2 * wma_half - wma_full
        # 在进行最终 WMA 计算前，移除前导 NaN 以防止问题
        valid_indices = ~np.isnan(hma_intermediate) # 获取非 NaN 值的索引
        if not np.any(valid_indices):
             # 如果中间结果全是 NaN，则返回 NaN 数组
             return np.full_like(values, np.nan)
        # 仅对有效的中间值应用最终的 WMA
        hma_result = np.full_like(values, np.nan) # 用 NaN 初始化结果数组
        # 将计算结果填充回对应位置
        hma_result[valid_indices] = talib.WMA(hma_intermediate[valid_indices], timeperiod=period_sqrt)
        return hma_result
    except Exception as e:
        logger.error(f"使用周期 {n} 计算 HMA 时出错: {e}")
        # 出错时返回 NaN 数组
        return np.full_like(values, np.nan)


# --- Backtesting.py 策略定义 ---
class HullMARibbonStrategy(Strategy):
    # 将策略参数定义为类变量
    # 如果之后需要优化，bt.optimize() 可以调整这些值
    hull_len = HULL_LEN
    sma_len = SMA_LEN
    trail_pct = TRAIL_PCT
    equity_fraction = EQUITY_FRACTION_PER_TRADE

    def init(self):
        # 使用 self.I 包装器预计算指标
        # `self.I` 会自动处理对齐并返回 numpy 数组
        self.sma = self.I(talib.SMA, self.data.Close, self.sma_len, name="SMA")
        # 使用辅助函数计算 HMA
        self.hma = self.I(HMA_func, self.data.Close, self.hull_len, name="HMA")

        # 用于追踪止损的变量
        self.trailing_stop_price = 0.0 # 当前追踪止损价格
        self.peak_price = 0.0          # 入场后的最高价
        logger.debug("策略已初始化。指标已计算。")

    def next(self):
        # `next` 方法会在数据预热期后为每个数据点（K线）调用
        current_price = self.data.Close[-1] # 获取最近的收盘价

        # --- 1. 追踪止损逻辑 (仅当持有多头仓位时) ---
        if self.position.is_long:
            # 更新入场后的最高价
            self.peak_price = max(self.peak_price, current_price)
            # 计算新的潜在追踪止损价
            new_trailing_stop = self.peak_price * (1 - self.trail_pct)
            # 仅当新的止损价更高时才更新 (止损只上移不下移)
            self.trailing_stop_price = max(self.trailing_stop_price, new_trailing_stop)

            # 检查当前价格是否跌破追踪止损价
            if current_price < self.trailing_stop_price:
                logger.debug(f"{self.data.index[-1]}: 触发追踪止损 @ {current_price:.4f} (止损位: {self.trailing_stop_price:.4f})")
                self.position.close() # 平仓
                self.trailing_stop_price = 0.0 # 为下次交易重置
                self.peak_price = 0.0          # 为下次交易重置
                return # 当前 K 线不再执行其他检查

        # --- 2. HMA 退出逻辑 (仅当持有多头仓位时) ---
        # 确保有足够的数据点进行回看 (h[-1] vs h[-4])
        if self.position.is_long and len(self.hma) >= 4:
             # 退出信号: 当前 HMA 下穿 3 周期前的 HMA (对应原始逻辑 h < h_3)
            if self.hma[-1] < self.hma[-4]:
                logger.debug(f"{self.data.index[-1]}: 触发 HMA 退出信号 @ {current_price:.4f} (HMA: {self.hma[-1]:.4f} < HMA_3周期前: {self.hma[-4]:.4f})")
                self.position.close() # 平仓
                self.trailing_stop_price = 0.0 # 重置追踪止损
                self.peak_price = 0.0          # 重置最高价
                return # 当前 K 线不再检查入场条件

        # --- 3. 入场逻辑 (仅当空仓时) ---
        # 确保有足够的数据点满足所有条件
        # 需要至少6个点用于 h[-6], 5个点用于 h[-5], 3个点用于 h[-3], 2个点用于 sma[-2]/close[-2]
        if not self.position and len(self.hma) >= 6 and len(self.sma) >= 2 and len(self.data.Close) >= 2:
            # 条件基于信号产生 K 线的前一根 K 线的数据
            # 我们在 K 线 `i-1` 收盘时检查条件，然后在 K 线 `i` 开盘时交易
            # backtesting.py 通过 `trade_on_close=False` 处理这种时序

            # 趋势过滤 = 前一根 K 线收盘价 > 前一根 K 线的 SMA 值
            # trend_filter = current_row['close_1'] > current_row['sma_1']
            trend_filter = self.data.Close[-2] > self.sma[-2]

            # HMA 上升 = 前一根 K 线的 HMA > 4 周期前的 HMA
            # hma_rising = current_row['h_1'] > current_row['h_4']
            hma_rising = self.hma[-2] > self.hma[-5]

            # HMA 之前平缓或下降 = 2 周期前的 HMA <= 5 周期前的 HMA
            # hma_was_flat_down = current_row['h_2'] <= current_row['h_5']
            hma_was_flat_down = self.hma[-3] <= self.hma[-6]

            # 如果所有入场条件都满足
            if trend_filter and hma_rising and hma_was_flat_down:
                # === 修改开始 ===
                # 不再手动计算单位数量 size
                # 直接使用 equity_fraction (0到1之间的小数) 调用 self.buy

                # 确保权益为正且分数有效
                if self.equity > 0 and 0 < self.equity_fraction <= 1:
                    logger.success(
                        f"{self.data.index[-1]}: 触发入场信号。 使用 {self.equity_fraction*100:.1f}% 的权益 ({self.equity:.2f}) 发出买入指令"
                    )
                    # 使用权益分数直接调用 buy
                    # backtesting.py 会根据此比例和当前价格计算能买的最大整数或小数单位（取决于资产）
                    # 并确保不超过可用现金
                    self.buy(size=self.equity_fraction) # <--- !!! 这里是关键改动 !!!

                    # 入场后立即初始化追踪止损 (注意: 实际成交价可能略有不同，但用当前价近似)
                    # 在发出买入指令后设置，此时 self.position 可能还未更新，用当前价格估算
                    current_price_for_ts = self.data.Close[-1]
                    self.peak_price = current_price_for_ts
                    self.trailing_stop_price = current_price_for_ts * (1 - self.trail_pct)
                    logger.debug(f"    买入指令已发出，基于当前价 {current_price_for_ts:.4f} 设置初始追踪止损位为: {self.trailing_stop_price:.4f}")

                elif self.equity <= 0:
                     logger.warning(f"{self.data.index[-1]}: 权益为零或负数 ({self.equity:.2f})，无法买入。")
                else: # equity_fraction 无效的情况 (例如 > 1 或 <= 0)
                     logger.warning(f"{self.data.index[-1]}: 配置的 equity_fraction ({self.equity_fraction}) 无效 (必须在 0 和 1 之间)。")
                # === 修改结束 ===


# --- 主执行程序 ---
if __name__ == "__main__":

    if target_name == "YOUR_ASSET_NAME": # 提醒用户设置目标名称
        logger.error("请在脚本顶部设置 'target_name' 变量为您要回测的标的名称！")
        sys.exit(1)

    # 检查数据目录和文件是否存在
    if not os.path.exists(data_directory):
        logger.error(f"错误: 数据目录 '{data_directory}' 不存在.")
        sys.exit(1)
    if not os.path.exists(csv_file_path):
        logger.error(f"错误: 在 '{data_directory}' 目录下未找到数据文件 '{target_name}.csv'.")
        sys.exit(1)

    logger.info(f"开始加载数据文件: {csv_file_path}")
    try:
        # 加载数据，确保列名是 backtesting.py 期望的标准名称
        df = pd.read_csv(
            csv_file_path,
            index_col='Date',       # 使用 Date 列作为索引
            parse_dates=True,     # 解析日期
            thousands=','         # 处理数字中可能存在的逗号
        )
        # 重命名列以匹配 backtesting.py 标准: Open, High, Low, Close, Volume (可选)
        # 用户的原始数据包含: Price, Open, High, Low
        df.rename(columns={
            'Price': 'Close', # 假设 'Price' 是收盘价
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low'
        }, inplace=True)

        # 确保必需的列存在
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"错误: 加载的数据缺少必要的列: {missing_cols}. 需要 'Open', 'High', 'Low', 'Close'.")
            sys.exit(1)

        # 将 OHLC 列转换为数值类型，无法转换的设为 NaN
        for col in required_columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除在关键 OHLC 列中包含 NaN 的行
        df.dropna(subset=required_columns, inplace=True)

        # 按日期排序，以防万一数据是乱序的
        df.sort_index(inplace=True)

        # 可选：如果数据中有 Volume 列则使用，否则如果策略需要可以添加虚拟数据（本策略不需要）
        if 'Volume' not in df.columns:
            df['Volume'] = 0 # 如果 backtesting.py 的某些功能需要 Volume 列，则添加虚拟数据

        logger.info(f"{target_name} 数据加载完毕. 数据范围: {df.index.min()} 到 {df.index.max()}. 共 {len(df)} 行.")

        if df.empty: # 如果处理后数据为空
            logger.error("数据加载并清理后 DataFrame 为空，无法继续回测。")
            sys.exit(1)

    except Exception as e:
        logger.error(f"加载或处理数据时发生错误: {e}")
        sys.exit(1)

    # --- 配置并运行回测 ---
    logger.info(f"开始为 {target_name} 配置并运行 backtesting.py 回测...")

    # 实例化 Backtest 对象
    # - `df`: 你的 OHLCV 数据 (Pandas DataFrame)
    # - `HullMARibbonStrategy`: 你定义的策略类
    # - `cash`: 初始资金
    # - `commission`: 每笔交易的手续费率 (买入和卖出都会应用)
    # - `exclusive_orders=True`: 确保每个 K 线只执行一次订单/交易逻辑，简化 `next()` 中的状态管理
    # - `trade_on_close=False`: 在 K 线 `i` 收盘时产生的信号，在 K 线 `i+1` 的开盘价执行交易。匹配原始逻辑。
    # - `hedging=False`: 标准的非对冲模式。
    bt = FractionalBacktest(df, HullMARibbonStrategy,
                  cash=INITIAL_EQUITY,
                  commission=COMMISSION_PCT,
                  exclusive_orders=True,
                  trade_on_close=False, # 在下一根K线的开盘价交易
                  hedging=False
                 )

    # 运行回测
    # 策略类中定义的参数可以在这里被覆盖，例如用于优化：
    # stats = bt.run(hull_len=50, sma_len=120) # 示例：覆盖参数运行
    stats = bt.run() # 使用策略类中定义的默认参数运行

    logger.info(f"回测执行完毕 for {target_name}.")

    # --- 显示结果 ---
    logger.info("--- 回测结果摘要 ---")
    print(stats) # 打印关键性能指标的摘要

    # --- 显示交易列表 (可选) ---
    logger.info("--- 交易记录 ---")
    trades_df = stats['_trades'] # 从返回的 stats 对象中获取交易记录 DataFrame
    print(trades_df)

    # 将交易记录保存到 CSV 文件
    trades_output_path = os.path.join(output_dir, f"{target_name}_trades_backtestingpy.csv")
    try:
        trades_df.to_csv(trades_output_path, index=False)
        logger.info(f"交易记录已保存到: {trades_output_path}")
    except Exception as e:
        logger.error(f"保存交易记录文件时出错: {e}")


    # --- 生成交互式 HTML 报告 ---
    report_path = os.path.join(output_dir, f"{target_name}_backtesting_report.html")
    logger.info(f"正在生成交互式 HTML 报告到: {report_path}")
    try:
        # `plot` 函数生成 HTML 文件，并可以选择在浏览器中打开它
        # 如果数据频率很高且想看到每个交易标记，可以使用 `plot(resample=False)`
        bt.plot(filename=report_path, open_browser=False) # 设置 open_browser=True 可自动打开报告
        logger.info(f"报告已生成。")
    except Exception as e:
        # 常见问题可能与 Bokeh 版本或环境有关
        logger.error(f"生成 HTML 报告时出错: {e}")
        logger.error("请确保已正确安装 backtesting.py 及其依赖项 (包括 bokeh)。")

    logger.info("------ 回测结束 ------")