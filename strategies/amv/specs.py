"""AMV 策略体系的纯类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FactorDirection = Literal["higher", "lower"]
FactorRole = Literal["alpha", "risk", "gate", "context", "diagnostic"]


@dataclass(frozen=True)
class FactorSpec:
    """AMV 因子元数据。因子实现放在 factors/，这里只描述如何解释它。"""

    name: str
    label: str
    direction: FactorDirection
    role: FactorRole = "alpha"
    family: str = "base"
    description: str = ""

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "higher"


@dataclass(frozen=True)
class ScoreComponent:
    """一个 ranker 中的单个打分组件。"""

    factor: str
    weight: float = 1.0
    direction: FactorDirection = "higher"

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "higher"

    def to_dict(self) -> dict[str, object]:
        return {
            "factor": self.factor,
            "higher_is_better": self.higher_is_better,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class RankerSpec:
    """候选池排序器定义。"""

    id: str
    label: str
    group: str
    components: tuple[ScoreComponent, ...] = ()
    factor: str | None = None
    descending: bool | None = None
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        if self.components:
            result: dict[str, object] = {
                "id": self.id,
                "label": self.label,
                "group": self.group,
                "components": [component.to_dict() for component in self.components],
            }
            if self.weights:
                result["weights"] = dict(self.weights)
            return result

        if self.factor is None or self.descending is None:
            raise ValueError(f"ranker {self.id} must define either components or factor/descending")
        return {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "factor": self.factor,
            "descending": self.descending,
        }
