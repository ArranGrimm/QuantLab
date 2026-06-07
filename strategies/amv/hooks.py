"""AMV Rule Hook — pluggable rule components for the ranker pipeline.

Each hook controls three extension points:
  - lazy_features: add columns BEFORE collect (pure Polars lazy)
  - penalty:        score penalty expression (merged into ONE with_columns)
  - gate:           filter signal rows AFTER TopN (tiny DataFrame, ~2000 rows)

Adding a new rule = write a Hook subclass + add one line to strategy JSON.
Pipeline.py never needs to be touched for new rules.
"""

from __future__ import annotations

import polars as pl

from strategies.amv.regime import build_amv_regime_gate_frame


class RuleHook:
    """Base class for rule hooks. Every method defaults to no-op.

    Subclass and override only the methods your rule needs.
    """

    # ── lazy phase (collect 前) ──

    def lazy_features(self, lf: pl.LazyFrame, params: dict) -> pl.LazyFrame:
        """Add feature columns. Called BEFORE collect. Pure Polars lazy."""
        return lf

    def penalty_columns(self) -> list[str]:
        """Column names that penalty() reads. Collected for projection pushdown."""
        return []

    # ── eager score phase (collect 后, 合并到一个 with_columns) ──

    def penalty(self, params: dict) -> pl.Expr:
        """Score penalty expression. Return pl.lit(0.0) for no penalty.

        All hook penalties are summed with pl.sum_horizontal() in a single
        with_columns call — no intermediate DataFrame copies.
        """
        return pl.lit(0.0)

    # ── eager gate phase (TopN 后, signal_rows ~2000 rows) ──

    def gate_columns(self) -> list[str]:
        """Column names that gate() reads from signal_rows."""
        return []

    def gate(self, signals: pl.DataFrame, params: dict) -> pl.DataFrame:
        """Filter/modify signal rows AFTER TopN. Return unchanged for no gate."""
        return signals


# ═══════════════════════════════════════════════════════════════════════════
# Built-in hooks
# ═══════════════════════════════════════════════════════════════════════════


class MediumTrendQualityHook(RuleHook):
    """128d medium trend structure/quality penalty.

    Feature generation is inlined here — no separate file.
    """

    def lazy_features(self, lf: pl.LazyFrame, params: dict) -> pl.LazyFrame:
        W = 128

        def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
            return num / pl.when(den.abs() > 1e-12).then(den).otherwise(None)

        lf = lf.with_columns([
            (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("_ret1d"),
            (pl.col("close_adj") > pl.col("pre_close_adj")).alias("_upday"),
            (
                (pl.col("close_adj") - pl.col("open_adj")).abs()
                / pl.max_horizontal((pl.col("high_adj") - pl.col("low_adj")).abs(), pl.lit(1e-12))
            ).alias("_body_eff"),
        ]).with_columns(pl.col("_ret1d").abs().alias("_abs_ret"))

        lf = lf.with_columns([
            (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).alias(f"_ret_{W}d"),
            _safe_div(
                pl.col("close_adj") - pl.col("close_adj").rolling_min(W).over("code"),
                pl.col("close_adj").rolling_max(W).over("code") - pl.col("close_adj").rolling_min(W).over("code"),
            ).alias(f"_pos_{W}d"),
            _safe_div(
                (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).abs(),
                pl.col("_abs_ret").rolling_sum(W).over("code"),
            ).alias(f"_trend_eff_{W}d"),
            pl.col("_upday").rolling_mean(W).over("code").alias(f"_up_ratio_{W}d"),
            _safe_div(
                pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0,
                pl.col("_ret1d").rolling_std(W).over("code"),
            ).alias(f"_ret_vol_{W}d"),
            (
                _safe_div(
                    pl.col("close_adj").rolling_mean(W).over("code"),
                    pl.col("close_adj").rolling_mean(W).over("code").shift(20).over("code"),
                ) - 1.0
            ).alias(f"_ma_slope_{W}d"),
            pl.col("_body_eff").rolling_mean(W).over("code").alias(f"_body_eff_{W}d"),
        ])

        rank_specs = [
            (f"_ret_{W}d", True), (f"_pos_{W}d", True), (f"_trend_eff_{W}d", True),
            (f"_up_ratio_{W}d", True), (f"_ret_vol_{W}d", True), (f"_ma_slope_{W}d", True),
            (f"_body_eff_{W}d", True),
        ]
        for col_name, higher_is_better in rank_specs:
            lf = lf.with_columns(
                (
                    pl.col(col_name)
                    .rank("average", descending=not higher_is_better)
                    .over("date")
                    / pl.len().over("date")
                ).alias(f"{col_name}_rank_pct")
            )

        structure_score = (
            pl.col(f"_ret_{W}d_rank_pct")
            + pl.col(f"_pos_{W}d_rank_pct")
            + pl.col(f"_ma_slope_{W}d_rank_pct")
        ) / 3.0

        quality_score = (
            pl.col(f"_trend_eff_{W}d_rank_pct")
            + pl.col(f"_up_ratio_{W}d_rank_pct")
            + pl.col(f"_ret_vol_{W}d_rank_pct")
            + pl.col(f"_body_eff_{W}d_rank_pct")
        ) / 4.0

        return lf.with_columns([
            structure_score.alias("_structure_score_128d"),
            quality_score.alias("_quality_score_128d"),
        ])

    def penalty_columns(self) -> list[str]:
        return ["_structure_score_128d", "_quality_score_128d"]

    def penalty(self, params: dict) -> pl.Expr:
        penalty_val = params.get("medium_penalty", 0.03)
        weak_threshold = params.get("medium_weak_threshold", 0.50)

        structure = pl.col("_structure_score_128d").fill_null(1.0)
        quality = pl.col("_quality_score_128d").fill_null(1.0)

        medium_weak = (structure < weak_threshold) & (quality < weak_threshold)
        s_sh = (weak_threshold - structure) / weak_threshold
        q_sh = (weak_threshold - quality) / weak_threshold
        strength = (
            pl.when(medium_weak)
            .then(((s_sh + q_sh) / 2.0).clip(0.0, 1.0))
            .otherwise(0.0)
        )

        return strength * penalty_val


class AmvRegimeGateHook(RuleHook):
    """AMV internal-phase gate: skip signals on aged+non-accelerating or chaos days.

    Gate frame is computed once from DuckDB and cached on the hook instance.
    """

    def __init__(self) -> None:
        self._gate_frame: pl.DataFrame | None = None

    def gate(self, signals: pl.DataFrame, params: dict) -> pl.DataFrame:
        gate = self._build_gate_frame(params)
        gated = signals.join(gate, on="signal_date", how="left")
        return gated.filter(
            ~pl.col("gate_skip").fill_null(False)
        ).select(signals.columns)

    def _build_gate_frame(self, params: dict) -> pl.DataFrame:
        if self._gate_frame is not None:
            return self._gate_frame
        self._gate_frame = build_amv_regime_gate_frame(
            bull_trigger_pct=params.get("bull_trigger_pct", 4.0),
            bull_lookback_days=params.get("bull_lookback_days", 2),
            bear_trigger_1d_pct=params.get("bear_trigger_1d_pct", -2.3),
            effective_lag_days=params.get("effective_lag_days", 1),
        )
        return self._gate_frame


class EventWeakgateHook(RuleHook):
    """7-dimension weak-window gate for first-board pullback events."""

    def gate(self, signals: pl.DataFrame, params: dict) -> pl.DataFrame:
        from strategies.amv.factors.limit_ecology import filter_weakgate

        return filter_weakgate(signals, params)


# ═══════════════════════════════════════════════════════════════════════════
# Hook registry
# ═══════════════════════════════════════════════════════════════════════════

_HOOK_REGISTRY: dict[str, type[RuleHook]] = {
    "medium-trend-quality": MediumTrendQualityHook,
    "amv-regime-gate": AmvRegimeGateHook,
    "event-weakgate": EventWeakgateHook,
}


def resolve_hooks(rule_ids: tuple[str, ...] | list[str]) -> list[tuple[RuleHook, str]]:
    """Instantiate hooks from rule IDs.

    Returns list of (hook_instance, rule_id) pairs for pipeline use.
    """
    hooks: list[tuple[RuleHook, str]] = []
    seen: set[str] = set()
    for rid in rule_ids:
        if rid in seen:
            continue
        seen.add(rid)
        hook_cls = _HOOK_REGISTRY.get(rid)
        if hook_cls is None:
            raise ValueError(
                f"Unknown rule: {rid!r}. Available: {list(_HOOK_REGISTRY)}"
            )
        hooks.append((hook_cls(), rid))
    return hooks
