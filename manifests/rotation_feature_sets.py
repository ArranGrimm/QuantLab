from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from utils.alpha158_factors import (
    ALPHA158_FACTOR_COLS,
    ALPHA158_FACTOR_TO_GROUP,
    resolve_alpha158_group_config,
)
from utils.rotation_factors import FACTOR_COLS

FeatureSetStatus = Literal["active", "archived", "experimental"]
_ROTATION_FACTOR_COL_SET = frozenset(FACTOR_COLS)
_ALPHA158_FACTOR_COL_SET = frozenset(ALPHA158_FACTOR_COLS)

CORE_12_FEATURES: tuple[str, ...] = (
    "ret_max_5d",
    "vol_60d",
    "turnover_rate",
    "atr_14_pct",
    "amplitude",
    "intraday_ret_ma5",
    "disp_bias_20",
    "high_open_pct",
    "vol_std_20d",
    "abnormal_vol",
    "intraday_pos",
    "vol_price_corr_20d",
)


@dataclass(frozen=True)
class RotationFeatureSetSpec:
    name: str
    status: FeatureSetStatus
    feature_mode: str
    feature_cols: tuple[str, ...] | None
    description: str
    note: str = ""
    alpha158_group_mode: str | None = None
    alpha158_analysis_group_mode: str | None = None
    selectable: bool = True

    @property
    def feature_count(self) -> int:
        return len(self.feature_cols or ())


def _tuple_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def split_rotation_feature_cols(
    feature_cols: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    ordered_cols = _tuple_unique(feature_cols)
    rotation_cols = tuple(
        factor_name for factor_name in ordered_cols if factor_name in _ROTATION_FACTOR_COL_SET
    )
    alpha158_cols = tuple(
        factor_name for factor_name in ordered_cols if factor_name in _ALPHA158_FACTOR_COL_SET
    )
    unknown_cols = tuple(
        factor_name
        for factor_name in ordered_cols
        if factor_name not in _ROTATION_FACTOR_COL_SET
        and factor_name not in _ALPHA158_FACTOR_COL_SET
    )
    return rotation_cols, alpha158_cols, unknown_cols


def infer_alpha158_group_mode_from_features(
    feature_cols: Sequence[str],
) -> str | tuple[str, ...] | None:
    _, alpha158_cols, unknown_cols = split_rotation_feature_cols(feature_cols)
    if unknown_cols:
        raise ValueError(
            "自定义特征集包含未知因子: "
            + ", ".join(unknown_cols)
        )

    if not alpha158_cols:
        return None

    alpha158_group_keys = tuple(
        dict.fromkeys(ALPHA158_FACTOR_TO_GROUP[factor_name] for factor_name in alpha158_cols)
    )
    if len(alpha158_group_keys) == 1:
        return alpha158_group_keys[0]
    return alpha158_group_keys


def build_custom_rotation_feature_set(
    feature_cols: Sequence[str],
    *,
    name: str = "custom",
    status: FeatureSetStatus = "experimental",
    description: str = "任意自定义 Rotation/Alpha158 因子组合。",
    note: str = "运行时动态指定，仅用于当前实验。",
    selectable: bool = True,
) -> RotationFeatureSetSpec:
    ordered_cols = _tuple_unique(feature_cols)
    if not ordered_cols:
        raise ValueError("自定义特征集不能为空，请至少提供一个因子。")

    rotation_cols, alpha158_cols, unknown_cols = split_rotation_feature_cols(ordered_cols)
    if unknown_cols:
        raise ValueError(
            "自定义特征集包含未知因子: "
            + ", ".join(unknown_cols)
        )

    if rotation_cols and alpha158_cols:
        feature_mode = "custom"
    elif rotation_cols:
        feature_mode = "core" if ordered_cols == CORE_12_FEATURES else "custom"
    elif alpha158_cols:
        feature_mode = "alpha158"
    else:
        raise ValueError("自定义特征集未识别到任何可训练因子。")

    return RotationFeatureSetSpec(
        name=name,
        status=status,
        feature_mode=feature_mode,
        feature_cols=ordered_cols,
        description=description,
        note=note,
        alpha158_group_mode=infer_alpha158_group_mode_from_features(ordered_cols),
        selectable=selectable,
    )


def _build_registry() -> dict[str, RotationFeatureSetSpec]:
    alpha158_kbar_shape = tuple(
        resolve_alpha158_group_config("kbar_shape")["factor_cols"]
    )
    alpha158_all = tuple(resolve_alpha158_group_config("all")["factor_cols"])

    registry = {
        "core_12": RotationFeatureSetSpec(
            name="core_12",
            status="active",
            feature_mode="core",
            feature_cols=CORE_12_FEATURES,
            description="冻结的 Rotation 核心 12 因子，对标锚点。",
            note="用于早期日频对标锚点与跨设备稳定比较。",
        ),
        "all_rotation": RotationFeatureSetSpec(
            name="all_rotation",
            status="archived",
            feature_mode="all",
            feature_cols=tuple(FACTOR_COLS),
            description="全量 Rotation 因子集。",
            note="统计层可用，但组合层已明显弱于当前主线。",
        ),
        "alpha158_kbar_shape": RotationFeatureSetSpec(
            name="alpha158_kbar_shape",
            status="archived",
            feature_mode="alpha158",
            feature_cols=alpha158_kbar_shape,
            description="Alpha158 的 kbar_shape 子集单跑。",
            note="更像交互增强器，不作为独立主线。",
            alpha158_group_mode="kbar_shape",
        ),
        "core_plus_alpha158_kbar_shape": RotationFeatureSetSpec(
            name="core_plus_alpha158_kbar_shape",
            status="active",
            feature_mode="core_plus_alpha158",
            feature_cols=_tuple_unique([*CORE_12_FEATURES, *alpha158_kbar_shape]),
            description="当前主线候选：core_12 + Alpha158(kbar_shape)。",
            note="当前优先推进方案，兼顾 gross/net/drawdown。",
            alpha158_group_mode="kbar_shape",
        ),
        "all_plus_alpha158_kbar_shape": RotationFeatureSetSpec(
            name="all_plus_alpha158_kbar_shape",
            status="experimental",
            feature_mode="all_plus_alpha158",
            feature_cols=_tuple_unique([*FACTOR_COLS, *alpha158_kbar_shape]),
            description="全量 Rotation + Alpha158(kbar_shape)。",
            note="保留作扩容实验，不作为默认主线。",
            alpha158_group_mode="kbar_shape",
        ),
        "alpha158_all": RotationFeatureSetSpec(
            name="alpha158_all",
            status="experimental",
            feature_mode="alpha158",
            feature_cols=alpha158_all,
            description="Alpha158 全量因子单跑。",
            note="仅作探索，不建议直接作为训练入口默认选项。",
            alpha158_group_mode="all",
        ),
        "core_plus_alpha158_top1": RotationFeatureSetSpec(
            name="core_plus_alpha158_top1",
            status="experimental",
            feature_mode="core_plus_alpha158_top1",
            feature_cols=None,
            description="core_12 + Alpha158 各组 top1 的动态实验入口。",
            note="该集合依赖分析 notebook 现场产物，已退出训练入口默认路径。",
            alpha158_analysis_group_mode="all",
            selectable=False,
        ),
        "pruned_rotation": RotationFeatureSetSpec(
            name="pruned_rotation",
            status="archived",
            feature_mode="pruned",
            feature_cols=None,
            description="基于相关性剪枝得到的 Rotation 子集。",
            note="依赖分析阶段即时结果，当前不再作为稳定训练入口。",
            selectable=False,
        ),
    }
    return registry


ROTATION_FEATURE_SET_REGISTRY = _build_registry()


def list_rotation_feature_sets(
    *,
    include_unselectable: bool = True,
) -> list[RotationFeatureSetSpec]:
    specs = list(ROTATION_FEATURE_SET_REGISTRY.values())
    if not include_unselectable:
        specs = [spec for spec in specs if spec.selectable]
    return specs


def resolve_rotation_feature_set(
    name: str,
    *,
    require_selectable: bool = True,
    custom_feature_cols: Sequence[str] | None = None,
    custom_name: str = "custom",
    custom_status: FeatureSetStatus = "experimental",
    custom_description: str = "任意自定义 Rotation/Alpha158 因子组合。",
    custom_note: str = "运行时动态指定，仅用于当前实验。",
) -> RotationFeatureSetSpec:
    if name == "custom":
        spec = build_custom_rotation_feature_set(
            custom_feature_cols or (),
            name=custom_name,
            status=custom_status,
            description=custom_description,
            note=custom_note,
            selectable=True,
        )
        if require_selectable and not spec.selectable:
            raise ValueError(
                f"Rotation feature set '{name}' 不是稳定训练入口。"
                f"请改用 manifest 中 selectable=True 的集合。"
            )
        return spec

    try:
        spec = ROTATION_FEATURE_SET_REGISTRY[name]
    except KeyError as exc:
        valid_names = ", ".join([*ROTATION_FEATURE_SET_REGISTRY, "custom"])
        raise ValueError(
            f"Unsupported Rotation feature set: {name}. Expected one of: {valid_names}"
        ) from exc

    if require_selectable and not spec.selectable:
        raise ValueError(
            f"Rotation feature set '{name}' 不是稳定训练入口。"
            f"请改用 manifest 中 selectable=True 的集合。"
        )
    return spec


def get_rotation_feature_set(
    name: str,
    *,
    require_selectable: bool = True,
) -> RotationFeatureSetSpec:
    return resolve_rotation_feature_set(name, require_selectable=require_selectable)


def describe_rotation_feature_set_spec(spec: RotationFeatureSetSpec) -> str:
    parts = [
        f"name={spec.name}",
        f"status={spec.status}",
        f"feature_mode={spec.feature_mode}",
        f"feature_count={spec.feature_count}",
    ]
    if spec.alpha158_group_mode:
        parts.append(f"alpha158_group_mode={spec.alpha158_group_mode}")
    if spec.alpha158_analysis_group_mode:
        parts.append(f"alpha158_analysis_group_mode={spec.alpha158_analysis_group_mode}")
    return ", ".join(parts)


def describe_rotation_feature_set(name: str) -> str:
    spec = get_rotation_feature_set(name, require_selectable=False)
    return describe_rotation_feature_set_spec(spec)
