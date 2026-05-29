from __future__ import annotations

import math
from typing import Any, cast

from strategies.amv.specs import FactorDirection, RankerSpec, ScoreComponent


def _weight_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _component(factor: str, direction: str, weight: float) -> ScoreComponent:
    if direction not in {"higher", "lower"}:
        raise ValueError(f"unknown factor direction: {direction}")
    return ScoreComponent(factor=factor, direction=cast(FactorDirection, direction), weight=weight)


def _reference_ranker() -> RankerSpec:
    return RankerSpec(
        id="ref_p2_k0p5_r0",
        label="当前基线 P2/K0.5/R0",
        group="reference",
        components=(
            _component("price_pos_20d", "higher", 2.0),
            _component("close_to_high_20d", "lower", 2.0),
            _component("KLEN", "lower", 0.5),
            _component("KMID2", "higher", 0.5),
        ),
    )


def _single_factor_rankers() -> list[RankerSpec]:
    return [
        RankerSpec("single_price_pos_20d", "单因子 20日高位", "single_price", factor="price_pos_20d", descending=True),
        RankerSpec(
            "single_near_high_20d",
            "单因子 接近20日新高",
            "single_price",
            factor="close_to_high_20d",
            descending=False,
        ),
        RankerSpec("single_klen_contract", "单因子 K线振幅收缩", "single_kbar", factor="KLEN", descending=False),
        RankerSpec("single_kmid2_strong", "单因子 实体占比偏强", "single_kbar", factor="KMID2", descending=True),
        RankerSpec("single_ret_5d", "单因子 5日动量", "single_momentum", factor="ret_5d", descending=True),
        RankerSpec("single_ret_20d", "单因子 20日动量", "single_momentum", factor="ret_20d", descending=True),
        RankerSpec("single_atr_14_pct_low", "单因子 ATR低风险", "single_risk", factor="atr_14_pct", descending=False),
        RankerSpec(
            "single_panic_vol_low",
            "单因子 卖压低",
            "single_risk",
            factor="panic_vol_ratio_20d",
            descending=False,
        ),
    ]


def _pkm_grid_rankers() -> list[RankerSpec]:
    rankers: list[RankerSpec] = []
    price_weights = (0.0, 1.0, 2.0, 3.0)
    kbar_weights = (0.0, 0.5, 1.0, 2.0)
    momentum_weights = (0.0, 0.5, 1.0, 2.0)
    seen_ratios: set[tuple[int, int, int]] = set()

    for price_weight in price_weights:
        for kbar_weight in kbar_weights:
            for momentum_weight in momentum_weights:
                if price_weight == 0 and kbar_weight == 0 and momentum_weight == 0:
                    continue
                ratio = (
                    int(round(price_weight * 2)),
                    int(round(kbar_weight * 2)),
                    int(round(momentum_weight * 2)),
                )
                divisor = math.gcd(math.gcd(ratio[0], ratio[1]), ratio[2]) or 1
                ratio = tuple(item // divisor for item in ratio)
                if ratio in seen_ratios:
                    continue
                seen_ratios.add(ratio)

                components: list[ScoreComponent] = []
                if price_weight > 0:
                    components.extend(
                        [
                            _component("price_pos_20d", "higher", price_weight),
                            _component("close_to_high_20d", "lower", price_weight),
                        ]
                    )
                if kbar_weight > 0:
                    components.extend(
                        [
                            _component("KLEN", "lower", kbar_weight),
                            _component("KMID2", "higher", kbar_weight),
                        ]
                    )
                if momentum_weight > 0:
                    components.extend(
                        [
                            _component("ret_5d", "higher", momentum_weight),
                            _component("ret_20d", "higher", momentum_weight),
                        ]
                    )

                rankers.append(
                    RankerSpec(
                        id=(
                            "grid_pkm"
                            f"_p{_weight_token(price_weight)}"
                            f"_k{_weight_token(kbar_weight)}"
                            f"_m{_weight_token(momentum_weight)}"
                        ),
                        label=(
                            "P/K/M网格 "
                            f"P{price_weight:g}/K{kbar_weight:g}/M{momentum_weight:g}"
                        ),
                        group="grid_pkm",
                        weights={
                            "price": price_weight,
                            "kbar": kbar_weight,
                            "momentum": momentum_weight,
                            "risk": 0.0,
                        },
                        components=tuple(components),
                    )
                )

    return rankers


def _legacy_pkr_rankers() -> list[RankerSpec]:
    rankers: list[RankerSpec] = []
    price_weights = (1.0, 2.0, 3.0)
    kbar_weights = (0.5, 1.0, 2.0)
    risk_weights = (0.0, 0.5, 1.0, 1.5)

    for price_weight in price_weights:
        for kbar_weight in kbar_weights:
            for risk_weight in risk_weights:
                components = [
                    _component("price_pos_20d", "higher", price_weight),
                    _component("close_to_high_20d", "lower", price_weight),
                    _component("KLEN", "lower", kbar_weight),
                    _component("KMID2", "higher", kbar_weight),
                ]
                if risk_weight > 0:
                    components.extend(
                        [
                            _component("atr_14_pct", "lower", risk_weight),
                            _component("panic_vol_ratio_20d", "lower", risk_weight),
                        ]
                    )

                rankers.append(
                    RankerSpec(
                        id=(
                            "grid_high_pos_kbar"
                            f"_p{_weight_token(price_weight)}"
                            f"_k{_weight_token(kbar_weight)}"
                            f"_r{_weight_token(risk_weight)}"
                        ),
                        label=(
                            "P/K/R网格 "
                            f"P{price_weight:g}/K{kbar_weight:g}/R{risk_weight:g}"
                        ),
                        group="grid_pkr",
                        weights={
                            "price": price_weight,
                            "kbar": kbar_weight,
                            "momentum": 0.0,
                            "risk": risk_weight,
                        },
                        components=tuple(components),
                    )
                )
    return rankers


def build_ranker_specs() -> list[RankerSpec]:
    return [
        _reference_ranker(),
        *_single_factor_rankers(),
        *_legacy_pkr_rankers(),
        *_pkm_grid_rankers(),
    ]


def build_rankers() -> list[dict[str, Any]]:
    return [ranker.to_dict() for ranker in build_ranker_specs()]
