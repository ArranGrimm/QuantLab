"""AMV 策略规则注册表。

规则是对 Ranker 输出做后处理的独立函数，分为三类：
- penalty: 修改候选分数（扣分/加权）
- gate: 过滤信号日（整日不开仓）
- rerank: 重新排序（改变 TopN 入选）

每项规则的 known_compatible / known_incompatible 是经过 Rust 验证的结论，
不是猜测。新增策略-规则组合时，必须通过 Rust raw execution 复核后再更新兼容性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuleType = Literal["penalty", "gate", "rerank"]


@dataclass(frozen=True)
class RuleSpec:
    """一条可复用的策略规则。"""

    id: str
    type: RuleType
    description: str
    known_compatible: tuple[str, ...] = ()
    known_incompatible: tuple[str, ...] = ()

    @property
    def applicable_strategies(self) -> list[str]:
        return list(self.known_compatible)


RULES: dict[str, RuleSpec] = {
    "sector-tailwind": RuleSpec(
        id="sector-tailwind",
        type="penalty",
        description="行业 10/20 日收益排名底部扣分 + 个股相对行业弱势确认。离散 bucket 版已被 cadence 否决，当前使用连续 linear + mix10/20 窗口。",
        known_compatible=("trend-p2", "trend-p3", "trend-p3-enhanced"),
        known_incompatible=("pullback-pb3", "event-firstboard"),
    ),
    "medium-trend-quality": RuleSpec(
        id="medium-trend-quality",
        type="penalty",
        description="128 日中期结构 + 趋势质量扣分。对 structure_score_128d < 0.5 且 trend_quality_score_128d < 0.5 的候选做 linear penalty。p0.03 是参数邻域峰值。",
        known_compatible=("trend-p2", "trend-p3", "trend-p3-enhanced"),
        known_incompatible=("pullback-pb3", "event-firstboard"),
    ),
    "amv-regime-gate": RuleSpec(
        id="amv-regime-gate",
        type="gate",
        description="AMV 活跃市值内部阶段风控：(aged + 非加速) OR (neg_streak >= 3 & 振幅 > 2.5%)。跳过 18.5% 信号日，Rust 验证收益+10.1pp/MaxDD -4.5pp，但不年年稳定。",
        known_compatible=("pullback-pb3",),
        known_incompatible=("trend-p2", "trend-p3", "trend-p3-enhanced"),
    ),
    "event-weakgate": RuleSpec(
        id="event-weakgate",
        type="gate",
        description="首板后回踩弱窗口过滤：7 维市场/候选/AMV 状态评分 >= 3.0 时跳过当日。base MaxDD 45.38% → weakgate 34.12%，但仍未 allocation-ready。",
        known_compatible=("event-firstboard",),
        known_incompatible=("trend-p2", "trend-p3", "trend-p3-enhanced", "pullback-pb3"),
    ),
}
