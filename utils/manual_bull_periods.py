"""手动多头区间 (RPA 抓取的活跃市值多头区间) 单点事实.

跨策略共享的市场层 timing 信号:
- B1 策略: 作为 6 条 AND 候选池中的 `is_manual_bull` 子条件 (per-row 广播)
- 截面轮动策略: 作为 `is_bull_regime` 开仓 gate (`require_bull_regime` 配置)

两个策略的"多头"语义来源于同一份事实 (用户 RPA 标记), 但应用方式不同:
- B1: 跟其他 5 条结构条件 AND, 形成强择股池
- 截面: 单独作为开仓时间窗, 截面 ranking 逻辑完全不变
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import polars as pl


LOOSE_PERIODS: list[tuple[str, str]] = [
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
"""市场层多头区间, 闭区间 (start, end), 格式 "YYYY-MM-DD".

当某个交易日落在任意一个区间内时, 视为整个市场处于多头状态.
所有股票当日的"多头标记"为同一个值 (市场级 daily flag).
"""


# 截面轮动策略对外暴露的别名: 强调这是"截面层 timing 信号", 跟 B1 池子语义解耦.
ROTATION_BULL_REGIME: list[tuple[str, str]] = LOOSE_PERIODS


def is_in_bull_regime_expr(
    date_col: str = "date",
    periods: Sequence[tuple[str, str]] | None = None,
) -> pl.Expr:
    """生成 polars 表达式: 判断 date_col 是否落在任意一个多头区间内.

    Args:
        date_col: 日期列名 (默认 "date")
        periods: 自定义区间列表, 默认使用 LOOSE_PERIODS

    Returns:
        polars boolean Expr (与 date_col 行级对齐)

    Example:
        df.with_columns(is_in_bull_regime_expr().alias("is_bull_regime"))
    """
    chosen = periods if periods is not None else LOOSE_PERIODS
    expr = pl.lit(False)
    for start, end in chosen:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        expr = expr | pl.col(date_col).is_between(start_date, end_date, closed="both")
    return expr
