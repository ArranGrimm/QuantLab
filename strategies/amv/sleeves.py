from __future__ import annotations

from strategies.amv.specs import RankerSpec, ScoreComponent, SleeveSpec


def _component(factor: str, direction: str, weight: float) -> ScoreComponent:
    if direction == "higher":
        return ScoreComponent(factor=factor, direction="higher", weight=weight)
    if direction == "lower":
        return ScoreComponent(factor=factor, direction="lower", weight=weight)
    raise ValueError(f"unknown factor direction: {direction}")


REF_RANKER = RankerSpec(
    id="reference_p2_k0p5_b0_c0_r0",
    label="Reference P2/K0.5",
    group="canonical",
    components=(
        _component("price_pos_20d", "higher", 2.0),
        _component("close_to_high_20d", "lower", 2.0),
        _component("KLEN", "lower", 0.5),
        _component("KMID2", "higher", 0.5),
    ),
    weights={"price": 2.0, "kbar": 0.5, "bias": 0.0, "close_pullback": 0.0, "risk": 0.0},
)

P3_RANKER = RankerSpec(
    id="candidate_p3_k0p5_b0_c0_r0",
    label="Candidate P3/K0.5",
    group="canonical",
    components=(
        _component("price_pos_20d", "higher", 3.0),
        _component("close_to_high_20d", "lower", 3.0),
        _component("KLEN", "lower", 0.5),
        _component("KMID2", "higher", 0.5),
    ),
    weights={"price": 3.0, "kbar": 0.5, "bias": 0.0, "close_pullback": 0.0, "risk": 0.0},
)

PB3_GATED_RANKER = RankerSpec(
    id="pullback_p0_k0_pb3_cp1_rv0",
    label="PB3/CP1/RV0 gated",
    group="pullback",
    components=(
        _component("ma_bias_20", "lower", 3.0),
        _component("disp_bias_20", "lower", 3.0),
        _component("KSFT", "lower", 1.0),
        _component("intraday_pos", "lower", 1.0),
    ),
    weights={"price": 0.0, "kbar": 0.0, "bias": 3.0, "close_pullback": 1.0, "risk": 0.0},
)


SLEEVE_SPECS: dict[str, SleeveSpec] = {
    "ref": SleeveSpec(
        id="reference_p2_k0p5_b0_c0_r0",
        label="Reference P2/K0.5 静态 Top3",
        family="ref",
        ranker=REF_RANKER,
        export_target="ref",
        export_args=("--sleeves", "reference_p2_k0p5_b0_c0_r0"),
        description="当前命名 reference baseline。",
    ),
    "p3": SleeveSpec(
        id="candidate_p3_k0p5_b0_c0_r0",
        label="Candidate P3/K0.5 静态 Top3",
        family="p3",
        ranker=P3_RANKER,
        export_target="p3",
        export_args=("--sleeves", "candidate_p3_k0p5_b0_c0_r0"),
        description="当前核心替换候选。",
    ),
    "pb3-gated": SleeveSpec(
        id="pullback_p0_k0_pb3_cp1_rv0",
        label="PB3 gated rolling 代表",
        family="pullback",
        ranker=PB3_GATED_RANKER,
        default_preset="6td-rolling",
        export_target="pb3-gated",
        export_args=(
            "--sleeves",
            "pullback_p0_k0_pb3_cp1_rv0",
            "--pb3-regime-gate",
            "aged_non_accel_or_chaos",
        ),
        description="当前 pullback 代表，进入组合权重前需 raw-execution allocation 复核。",
    ),
    "context": SleeveSpec(
        id="p3_context_combo",
        label="P3 板块顺风 + medium128 上下文组合",
        family="context",
        ranker=P3_RANKER,
        export_target="context",
        export_args=("--sector-penalties", "0.02", "--medium-penalties", "0.03"),
        description="当前最强静态 challenger，仍处于 forward monitor。",
    ),
    "limit-weakgate": SleeveSpec(
        id="limit_first_board_pullback_weakgate",
        label="首板后回踩 weak-window gate",
        family="limit_ecology",
        default_preset="5td-static",
        export_target="limit-weakgate",
        export_args=("--sleeves", "limit_first_board_pullback_weakgate"),
        description="涨停生态研究候选，尚未 allocation-ready。",
    ),
}


def get_sleeve(alias: str) -> SleeveSpec:
    try:
        return SLEEVE_SPECS[alias]
    except KeyError as exc:
        raise KeyError(f"unknown AMV sleeve alias: {alias}") from exc
