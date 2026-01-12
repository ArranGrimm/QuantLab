# B1 选股因子模块
from .b1_factors import calc_b1_factors_tg
from .b1_factors_opt import calc_b1_factors_opt, DEFAULT_CONFIG

# 回测引擎模块
from .backtest import (
    run_backtest, 
    print_backtest_report, 
    analyze_yearly_intensity,
    run_backtest_realistic,
    print_realistic_report,
)

# 数据工具
from .baostock_utils import *
from .data_utils import *

__all__ = [
    # 因子计算
    "calc_b1_factors_tg",
    "calc_b1_factors_opt",
    "DEFAULT_CONFIG",
    # 回测 (简单版)
    "run_backtest",
    "print_backtest_report",
    "analyze_yearly_intensity",
    # 回测 (非重叠版)
    "run_backtest_realistic",
    "print_realistic_report",
]
