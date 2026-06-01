"""
AMV 策略注册表 —— 从 configs/*.json 动态加载策略定义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strategies.amv.specs import RankerSpec, ScoreComponent

CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass(frozen=True)
class Preset:
    name: str
    config: str


@dataclass(frozen=True)
class Strategy:
    """一个已知 AMV 子策略 = Ranker + Rules + Preset。"""

    name: str
    family: str
    label: str
    ranker: RankerSpec
    preset: Preset
    rules: tuple[str, ...] = ()
    rule_params: dict[str, Any] = field(default_factory=dict)
    status: str = "research"
    description: str = ""


# === JSON loader ===


def _resolve_weight(weight_template: str, params: dict[str, float]) -> float:
    result = weight_template
    for k, v in params.items():
        result = result.replace(f"{{{k}}}", str(v))
    return float(result)


def _build_ranker(name: str, ranker_config: dict, family: str, label: str) -> RankerSpec:
    rtype = ranker_config["type"]

    if rtype == "components":
        templates = json.loads((CONFIGS_DIR / "_rankers.json").read_text(encoding="utf-8"))
        template = templates[ranker_config["template"]]
        params = {**template.get("default_params", {}), **ranker_config.get("params", {})}
        components = tuple(
            ScoreComponent(
                factor=c["factor"],
                direction=c["direction"],
                weight=_resolve_weight(str(c["weight"]), params),
            )
            for c in template["components"]
        )
        return RankerSpec(
            id=f"{name}-ranker",
            label=template["label"],
            group=template["group"],
            components=components,
            weights=params,
        )

    if rtype == "custom":
        return RankerSpec(
            id=f"{name}-ranker",
            label=label,
            group=family,
            factor=ranker_config["function"],
            descending=True,
            weights={},
        )

    raise ValueError(f"unknown ranker type: {rtype}")


def load_strategy(name: str) -> Strategy:
    config = json.loads((CONFIGS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    ranker = _build_ranker(name, config["ranker"], config["family"], config["label"])

    preset_raw = config["preset"]
    preset = Preset(name=preset_raw["name"], config=preset_raw["config"]) if isinstance(preset_raw, dict) else Preset(name=preset_raw, config="")

    rules = tuple(r["id"] for r in config.get("rules", []))
    rule_params: dict[str, Any] = {}
    for r in config.get("rules", []):
        if r.get("params"):
            rule_params.update(r["params"])

    return Strategy(
        name=config["name"],
        family=config["family"],
        label=config["label"],
        ranker=ranker,
        preset=preset,
        rules=rules,
        rule_params=rule_params,
        status=config["status"],
        description=config.get("description", ""),
    )


def load_all_strategies() -> dict[str, Strategy]:
    strategies: dict[str, Strategy] = {}
    for path in sorted(CONFIGS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        strategy = load_strategy(path.stem)
        strategies[strategy.name] = strategy
    return strategies


KNOWN_STRATEGIES = load_all_strategies()


def resolve_project_path(root: Path, relative_path: str) -> Path:
    return (root / relative_path).resolve()
