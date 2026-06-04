# B1 选股因子模块
from .b1_factors import calc_b1_factors_tg
from .b1_factors_opt import calc_b1_factors_base, calc_b1_factors_opt, calc_b1_factors_dynamic_j, calc_b1_factors_wmacd, DEFAULT_CONFIG
from .b1_feature_pool import (
    B1_FEATURE_GROUPS,
    B1_FEATURE_GROUP_LABELS,
    B1_FEATURE_SET_REGISTRY,
    B1_FEATURE_TO_GROUP,
    B1_MINING_CORE_FEATURE_COLS,
    B1_MINING_FEATURE_COLS,
    B1_MINING_SECOND_BATCH_FEATURE_COLS,
    B1_SELECTED_FEATURE_COLS,
    B1_SELECTED_ROTATION_HYBRID_V1_FEATURE_COLS,
    b1_feature_set_requires_rotation_kbar,
    build_b1_research_frame,
    describe_b1_feature_set,
    resolve_b1_feature_set,
)

# 砖型图反转因子模块
from .renko_factors import calc_renko_factors_base, calc_renko_factors, calc_renko_factors_wmacd

# 截面轮动因子模块
from .rotation_factors import (
    calc_rotation_factors,
    cross_section_normalize,
    FACTOR_COLS,
    FACTOR_GROUPS,
    FACTOR_GROUP_LABELS,
    FACTOR_TO_GROUP,
)
from .alpha158_factors import (
    calc_alpha158_factors,
    ALPHA158_FACTOR_COLS,
    ALPHA158_FACTOR_GROUPS,
    ALPHA158_FACTOR_GROUP_LABELS,
    ALPHA158_FACTOR_TO_GROUP,
    resolve_alpha158_group_config,
)

# 回测引擎模块
from .backtest import (
    run_backtest,
    run_backtest_short,
    print_backtest_report, 
    analyze_yearly_intensity
)

# 数据工具
from .baostock_utils import get_st_blacklist_pl
from .baostock_utils import get_stock_industry
from .data_source import (
    DEFAULT_QMT_DB,
    DEFAULT_TDX_DB,
    DataSourceSettings,
    daily_reader,
    open_daily_reader,
    resolve_data_source,
)
from .duckdb_utils import get_adj_factor_frame, load_daily_data_full, load_60m_data_adj, load_daily_data_single, add_price_limit_cols

# IC 分析 + 因子相关性工具
from .ic_analysis import calc_factor_ic, select_factors_by_ic, calc_factor_corr, print_corr_clusters, find_redundant_factors
from .factor_analysis import (
    build_daily_ic_frame,
    build_ic_summary_frame,
    compute_factor_decay,
    empty_group_summary_frame,
    empty_ic_summary_frame,
    extract_group_top_factor_cols,
    resolve_decay_factor_cols,
    summarize_factor_groups,
)

# 信号导出 (for Rust)
from .signal_export import export_for_rust, export_rotation_scores, export_renko_scores, validate_export

__all__ = [
    # data source
    "DEFAULT_QMT_DB",
    "DEFAULT_TDX_DB",
    "DataSourceSettings",
    "daily_reader",
    "open_daily_reader",
    "resolve_data_source",
    # duckdb utils
    "get_adj_factor_frame",
    "load_daily_data_full",
    "load_60m_data_adj",
    "load_daily_data_single",
    "add_price_limit_cols",
    # 因子计算
    "calc_b1_factors_tg",
    "calc_b1_factors_base",
    "calc_b1_factors_opt",
    "calc_b1_factors_dynamic_j",
    "calc_b1_factors_wmacd",
    "DEFAULT_CONFIG",
    "B1_FEATURE_GROUPS",
    "B1_FEATURE_GROUP_LABELS",
    "B1_FEATURE_SET_REGISTRY",
    "B1_FEATURE_TO_GROUP",
    "B1_MINING_CORE_FEATURE_COLS",
    "B1_MINING_FEATURE_COLS",
    "B1_MINING_SECOND_BATCH_FEATURE_COLS",
    "B1_SELECTED_FEATURE_COLS",
    "B1_SELECTED_ROTATION_HYBRID_V1_FEATURE_COLS",
    "b1_feature_set_requires_rotation_kbar",
    "build_b1_research_frame",
    "describe_b1_feature_set",
    "resolve_b1_feature_set",
    # 砖型图反转因子
    "calc_renko_factors_base",
    "calc_renko_factors",
    "calc_renko_factors_wmacd",
    # 截面轮动因子
    "calc_rotation_factors",
    "cross_section_normalize",
    "FACTOR_COLS",
    "FACTOR_GROUPS",
    "FACTOR_GROUP_LABELS",
    "FACTOR_TO_GROUP",
    "calc_alpha158_factors",
    "ALPHA158_FACTOR_COLS",
    "ALPHA158_FACTOR_GROUPS",
    "ALPHA158_FACTOR_GROUP_LABELS",
    "ALPHA158_FACTOR_TO_GROUP",
    "resolve_alpha158_group_config",
    # 回测
    "run_backtest",
    "run_backtest_short",
    "print_backtest_report",
    "analyze_yearly_intensity",
    # IC 分析 + 因子相关性
    "calc_factor_ic",
    "select_factors_by_ic",
    "calc_factor_corr",
    "print_corr_clusters",
    "find_redundant_factors",
    "build_daily_ic_frame",
    "build_ic_summary_frame",
    "compute_factor_decay",
    "empty_group_summary_frame",
    "empty_ic_summary_frame",
    "extract_group_top_factor_cols",
    "resolve_decay_factor_cols",
    "summarize_factor_groups",
    # Rust 导出
    "export_for_rust",
    "export_rotation_scores",
    "export_renko_scores",
    "validate_export",
    # baostock utils
    "get_st_blacklist_pl",
    # akshare utils
    "get_stock_industry",
]
