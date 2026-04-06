# -*- coding: utf-8 -*-
"""
混合频率ETF动量轮动策略（纯 polars 实现）
结合日线数据（历史动量）+ 5分钟数据（实时价格）
在11:00和14:30两个时点触发交易信号
输出列：date | signal | price | symbol
"""
import polars as pl
import numpy as np
from loguru import logger
import sys
import os
from datetime import time, datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# --- 全局配置 ---
STRATEGY_NAME = "Intraday_ETF_Momentum_Rotation"
ETF_SYMBOLS = ['518880.SH', '513100.SH', '513130.SH', '159915.SZ']  # 黄金, 纳指, 恒生科技, 创业板
DATA_DIR = "Data"
OUTPUT_DIR = "results"

# 策略参数
LOOKBACK_PERIOD = 25
ANNUAL_TRADING_DAYS = 250
COMMISSION_RATE = 0.0002
SLIPPAGE_RATE = 0.0005

# 触发时间点
TRIGGER_TIMES = ['11:00', '14:30']

# 动量计算安全区间
MOMENTUM_MIN = 0.0
MOMENTUM_MAX = 5.0

# --- 日志配置 ---
def setup_logging():
    """配置日志系统"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, level="INFO", format=log_format)
    
    log_file = os.path.join(OUTPUT_DIR, f"{STRATEGY_NAME}.log")
    logger.add(log_file, level="DEBUG", rotation="10 MB", format=log_format)
    
    logger.info(f"------ {STRATEGY_NAME} 策略开始 ------")
    logger.info(f"ETF标的: {ETF_SYMBOLS}")
    logger.info(f"触发时间: {TRIGGER_TIMES}")

def weighted_linear_regression(y_values: np.ndarray) -> Tuple[float, float]:
    """
    加权线性回归计算 - 严格按照原有逻辑实现
    
    Args:
        y_values: 价格序列（原始价格，非对数）
        
    Returns:
        (annualized_return, r_squared): 年化收益率和R²
    """
    n = len(y_values)
    if n < 2 or np.isnan(y_values).any():
        return np.nan, np.nan

    # 对数变换
    log_y = np.log(y_values)
    x = np.arange(n)

    # 线性递增权重 1.0 -> 2.0
    weights = np.linspace(1.0, 2.0, n)

    # 加权最小二乘
    w_sum = np.sum(weights)
    w_x_sum = np.sum(weights * x)
    w_y_sum = np.sum(weights * log_y)
    w_x2_sum = np.sum(weights * x**2)
    w_xy_sum = np.sum(weights * x * log_y)

    denominator = w_sum * w_x2_sum - (w_x_sum * w_x_sum)
    if denominator == 0:
        return np.nan, np.nan

    slope = (w_sum * w_xy_sum - w_x_sum * w_y_sum) / denominator
    intercept = (w_y_sum - slope * w_x_sum) / w_sum

    # 加权R²
    y_pred = slope * x + intercept
    weighted_residuals_sq = weights * (log_y - y_pred) ** 2
    weighted_mean_y = w_y_sum / w_sum
    weighted_ss_tot = np.sum(weights * (log_y - weighted_mean_y) ** 2)

    if weighted_ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - (np.sum(weighted_residuals_sq) / weighted_ss_tot)
        r_squared = max(0.0, r_squared)

    # 年化收益率
    daily_factor = np.exp(slope)
    annualized_return = daily_factor**ANNUAL_TRADING_DAYS - 1.0

    return annualized_return, r_squared

def calculate_momentum_score(historical_prices: List[float], current_price: float) -> float:
    """
    计算动量得分 - 混合历史日线数据和当前分钟级价格
    
    Args:
        historical_prices: 历史日线收盘价序列（25天）
        current_price: 当前时刻的价格
        
    Returns:
        动量得分
    """
    if len(historical_prices) < LOOKBACK_PERIOD:
        return np.nan

    # 组合历史+当前价格
    tail_hist = np.array(historical_prices[-(LOOKBACK_PERIOD - 1):], dtype=float)
    combined_prices = np.concatenate([tail_hist, np.array([current_price], dtype=float)])

    annualized_return, r_squared = weighted_linear_regression(combined_prices)

    if np.isnan(annualized_return) or np.isnan(r_squared):
        return np.nan

    score = annualized_return * r_squared
    return score if np.isfinite(score) else np.nan

class IntradayMomentumStrategy:
    """混合频率ETF动量轮动策略"""
    
    def __init__(self):
        self.daily_data = {}      # 日线数据
        self.intraday_data = {}   # 5分钟数据
        self.current_position = None  # 当前持仓
        self.traded_dates = set() # 已交易日期
        
    def load_daily_data(self) -> Dict[str, pl.DataFrame]:
        """加载日线数据（用于历史动量计算），来自 Data/{symbol}.csv"""
        logger.info("开始加载日线数据...")

        daily_data: Dict[str, pl.DataFrame] = {}
        for symbol in ETF_SYMBOLS:
            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            if not os.path.exists(file_path):
                logger.warning(f"缺少日线数据文件: {file_path}")
                continue
            try:
                df = pl.read_csv(file_path)
                if "Date" not in df.columns or "Close" not in df.columns:
                    logger.error(f"日线数据文件格式不正确: {file_path}")
                    continue
                df = df.with_columns([
                    pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("date"),
                    pl.col("Close").cast(pl.Float64).alias("close")
                ]).select(["date", "close"]).sort("date")
                daily_data[symbol] = df
                logger.info(f"加载日线数据: {symbol} 共 {df.height} 行")
            except Exception as e:
                logger.error(f"加载日线数据失败 {symbol}: {e}")
        return daily_data
    
    def load_intraday_data(self) -> Dict[str, pl.DataFrame]:
        """加载5分钟级数据"""
        logger.info("开始加载5分钟级数据...")
        
        intraday_data = {}
        
        for symbol in ETF_SYMBOLS:
            file_path = os.path.join(DATA_DIR, f"{symbol}_5m.csv")
            
            if not os.path.exists(file_path):
                logger.error(f"未找到数据文件: {file_path}")
                continue
            
            try:
                # 使用polars加载数据
                df = pl.read_csv(file_path)
                
                # 转换时间戳
                df = df.with_columns([
                    pl.from_epoch(pl.col("time"), time_unit="ms")
                    .dt.convert_time_zone("Asia/Shanghai")
                    .alias("datetime")
                ])
                
                # 添加日期和时间列
                df = df.with_columns([
                    pl.col("datetime").dt.date().alias("date"),
                    pl.col("datetime").dt.time().alias("time")
                ])
                
                # 过滤出触发时间点的数据
                trigger_times = [time.fromisoformat(t) for t in TRIGGER_TIMES]
                df = df.filter(pl.col("time").is_in(trigger_times))
                
                intraday_data[symbol] = df
                logger.info(f"成功加载 {symbol} 数据: {len(df)} 条记录")
                
            except Exception as e:
                logger.error(f"加载 {symbol} 数据失败: {e}")
                continue
        
        return intraday_data
    
    def get_trading_dates(self) -> List:
        """获取所有可交易日期"""
        all_dates = set()
        
        for symbol, df in self.intraday_data.items():
            dates = df.select("date").unique().to_series().to_list()
            all_dates.update(dates)
        
        return sorted(list(all_dates))
    
    def get_historical_prices(self, symbol: str, end_date, lookback_days: int) -> Optional[List[float]]:
        """获取历史日线价格（从CSV日线数据）"""
        if symbol not in self.daily_data:
            return None

        df = self.daily_data[symbol]
        historical_df = (
            df.filter(pl.col("date") < end_date)
              .sort("date")
              .tail(lookback_days)
        )
        if historical_df.height < lookback_days:
            return None
        return historical_df.select("close").to_series().to_list()
    
    def calculate_signals_for_datetime(self, target_datetime) -> Optional[Dict]:
        """计算指定时间点的交易信号"""
        target_date = target_datetime.date() if hasattr(target_datetime, 'date') else target_datetime
        target_time = target_datetime.time() if hasattr(target_datetime, 'time') else time(11, 0)
        
        # 检查是否已经交易过
        date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
        if date_str in self.traded_dates:
            return None
        
        logger.debug(f"计算 {target_date} {target_time} 的交易信号")
        
        momentum_scores = {}
        current_prices = {}
        
        # 为每个ETF计算动量得分
        for symbol in ETF_SYMBOLS:
            if symbol not in self.intraday_data:
                continue
            
            # 获取当前时刻的价格
            current_data = (
                self.intraday_data[symbol]
                .filter(
                    (pl.col("date") == target_date) & 
                    (pl.col("time") == target_time)
                )
            )
            
            if len(current_data) == 0:
                logger.debug(f"{symbol} 在 {target_date} {target_time} 无数据")
                continue
            
            current_price = current_data.select("close").item()
            current_prices[symbol] = current_price
            
            # 获取历史价格
            historical_prices = self.get_historical_prices(symbol, target_date, LOOKBACK_PERIOD)
            
            if historical_prices is None:
                logger.debug(f"{symbol} 历史数据不足")
                continue
            
            # 计算动量得分
            score = calculate_momentum_score(historical_prices, current_price)
            
            if not (isinstance(score, float) and np.isnan(score)):
                momentum_scores[symbol] = score
                logger.debug(f"{symbol} 动量得分: {score:.4f}")
        
        if not momentum_scores:
            logger.debug(f"{target_date} {target_time} 无有效动量得分")
            return None
        
        # 应用安全区间过滤
        filtered_scores = {
            symbol: score for symbol, score in momentum_scores.items()
            if MOMENTUM_MIN < score <= MOMENTUM_MAX
        }
        
        if not filtered_scores:
            logger.debug(f"{target_date} {target_time} 无符合安全区间的标的")
            return None
        
        # 选择得分最高的ETF
        best_symbol = max(filtered_scores.keys(), key=lambda x: filtered_scores[x])
        best_score = filtered_scores[best_symbol]
        
        logger.info(f"{target_date} {target_time} 选中 {best_symbol}, 得分: {best_score:.4f}")
        
        # 生成交易信号
        signal_data = {
            'datetime': target_datetime,
            'date': target_date,
            'time': target_time,
            'selected_symbol': best_symbol,
            'momentum_score': best_score,
            'execution_price': current_prices[best_symbol],
            'previous_position': self.current_position
        }
        
        # 标记已交易
        self.traded_dates.add(date_str)
        
        return signal_data
    
    def generate_trade_signals(self) -> pl.DataFrame:
        """生成所有交易信号"""
        logger.info("开始生成交易信号...")
        
        # 加载数据（先日线，再分钟线）
        self.daily_data = self.load_daily_data()
        self.intraday_data = self.load_intraday_data()
        
        if not self.intraday_data:
            logger.error("无可用数据")
            return pl.DataFrame()
        
        # 获取所有交易日
        trading_dates = self.get_trading_dates()
        logger.info(f"共找到 {len(trading_dates)} 个交易日")
        
        # 生成所有检查时点
        check_points = []
        for trade_date in trading_dates:
            for time_str in TRIGGER_TIMES:
                trigger_time = time.fromisoformat(time_str)
                # 创建完整的datetime对象用于计算
                dt = datetime.combine(trade_date, trigger_time)
                check_points.append(dt)
        
        logger.info(f"共 {len(check_points)} 个检查点")
        
        # 计算每个时点的信号
        signals = []
        for checkpoint in check_points:
            signal = self.calculate_signals_for_datetime(checkpoint)
            if signal:
                signals.append(signal)
        
        if not signals:
            logger.warning("未生成任何交易信号")
            return pl.DataFrame()
        
        # 转换为交易记录
        trades = []
        
        for i, signal in enumerate(signals):
            current_symbol = signal['selected_symbol']
            previous_symbol = signal['previous_position']
            
            # 如果有持仓变化，生成交易记录
            if current_symbol != previous_symbol:
                
                # 卖出之前的持仓
                if previous_symbol and previous_symbol != current_symbol:
                    trades.append({
                        'timestamp': signal['datetime'],
                        'date': signal['date'],
                        'time': signal['time'],
                        'action': 'SELL',
                        'symbol': previous_symbol,
                        'price': signal['execution_price'],  # 这里简化处理，实际应该是不同的价格
                        'reason': f'Switch from {previous_symbol} to {current_symbol}'
                    })
                
                # 买入新持仓
                trades.append({
                    'timestamp': signal['datetime'],
                    'date': signal['date'], 
                    'time': signal['time'],
                    'action': 'BUY',
                    'symbol': current_symbol,
                    'price': signal['execution_price'],
                    'reason': f'Momentum score: {signal["momentum_score"]:.4f}'
                })
                
                # 更新当前持仓
                self.current_position = current_symbol
        
        if not trades:
            logger.warning("未生成任何交易记录")
            return pl.DataFrame()
        
        # 转换为DataFrame并输出期望列
        trades_df = pl.DataFrame(trades)
        result_df = trades_df.select([
            pl.col("date").alias("date"),
            pl.col("action").alias("signal"),
            pl.col("price").alias("price"),
            pl.col("symbol").alias("symbol")
        ])
        logger.info(f"生成 {result_df.height} 条交易记录")
        return result_df

def main():
    """主函数"""
    setup_logging()
    
    try:
        # 创建策略实例
        strategy = IntradayMomentumStrategy()
        
        # 生成交易信号
        trades_df = strategy.generate_trade_signals()
        
        if len(trades_df) > 0:
            # 显示结果
            logger.info("交易信号预览:")
            print(trades_df.head(10))
            
            # 保存结果
            output_file = os.path.join(OUTPUT_DIR, f"{STRATEGY_NAME}_signals.csv")
            trades_df.write_csv(output_file)
            logger.info(f"交易信号已保存到: {output_file}")
            
            # 统计信息（基于 signal 列）
            buy_count = trades_df.filter(pl.col("signal") == "BUY").height
            sell_count = trades_df.filter(pl.col("signal") == "SELL").height
            unique_dates = trades_df.select("date").unique().height
            
            logger.info("统计信息:")
            logger.info(f"  买入次数: {buy_count}")
            logger.info(f"  卖出次数: {sell_count}")
            logger.info(f"  交易日数: {unique_dates}")
            
        else:
            logger.warning("未生成任何交易信号")
        
    except Exception as e:
        logger.error(f"策略执行失败: {e}")
        raise
    
    logger.info(f"------ {STRATEGY_NAME} 策略结束 ------")

if __name__ == "__main__":
    main()
