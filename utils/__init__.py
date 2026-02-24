# B1 选股因子模块
from .b1_factors import calc_b1_factors_tg
from .b1_factors_opt import calc_b1_factors_base, calc_b1_factors_opt, calc_b1_factors_dynamic_j, calc_b1_factors_wmacd, DEFAULT_CONFIG

# 回测引擎模块
from .backtest import (
    run_backtest, 
    print_backtest_report, 
    analyze_yearly_intensity
)

# 数据工具
from .baostock_utils import get_st_blacklist_pl
from .akshare_utils import generate_sector_file
from .duckdb_utils import get_adj_factor_frame, load_daily_data_full, load_60m_data_adj, load_daily_data_single

# 信号导出 (for Rust)
from .signal_export import export_for_rust, validate_export

__all__ = [
    # duckdb utils
    "get_adj_factor_frame",
    "load_daily_data_full",
    "load_60m_data_adj",
    "load_daily_data_single",
    # 因子计算
    "calc_b1_factors_tg",
    "calc_b1_factors_base",
    "calc_b1_factors_opt",
    "calc_b1_factors_dynamic_j",
    "calc_b1_factors_wmacd",
    "DEFAULT_CONFIG",
    # 回测 (简单版)
    "run_backtest",
    "print_backtest_report",
    "analyze_yearly_intensity",
    # Rust 导出
    "export_for_rust",
    "validate_export",
    # baostock utils
    "get_st_blacklist_pl",
    # akshare utils
    "generate_sector_file",
]
