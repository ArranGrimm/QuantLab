"""Canonical AMV strategy metadata and assembly helpers."""

from strategies.amv.export import write_signal_artifact
from strategies.amv.regime import build_amv_phase_frame, build_amv_regime_gate_frame
from strategies.amv.workflows import WorkflowExportConfig, export_strategy
from strategies.amv.rules import RULES, RuleSpec
from strategies.amv.registry import KNOWN_STRATEGIES, Strategy
from strategies.amv.signals import (
    AMV_SIGNAL_EXPORT_COLUMNS,
    SignalAssemblyConfig,
    assemble_ranker_signal,
    base_candidate_expr,
    build_backtest_signal_frame,
    ranker_required_columns,
    ranker_score_expr,
)

__all__ = [
    "AMV_SIGNAL_EXPORT_COLUMNS",
    "KNOWN_STRATEGIES",
    "RULES",
    "RuleSpec",
    "SignalAssemblyConfig",
    "Strategy",
    "WorkflowExportConfig",
    "assemble_ranker_signal",
    "base_candidate_expr",
    "build_backtest_signal_frame",
    "build_amv_phase_frame",
    "build_amv_regime_gate_frame",
    "export_strategy",
    "ranker_required_columns",
    "ranker_score_expr",
    "write_signal_artifact",
]
