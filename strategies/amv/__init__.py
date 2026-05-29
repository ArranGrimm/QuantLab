"""Canonical AMV strategy metadata and assembly helpers."""

from strategies.amv.export import (
    SignalArtifact,
    SignalArtifactConfig,
    build_signal_meta,
    git_commit,
    relative_path,
    timestamp_token,
    write_signal_artifact,
)
from strategies.amv.regime import build_amv_phase_frame, build_pb3_regime_gate_frame
from strategies.amv.workflows import (
    WorkflowExportConfig,
    export_context_sleeve,
    export_limit_weakgate_sleeve,
    export_ranker_sleeve,
)
from strategies.amv.signals import (
    AMV_SIGNAL_EXPORT_COLUMNS,
    SignalAssemblyConfig,
    assemble_ranker_signal,
    base_candidate_expr,
    build_backtest_signal_frame,
    ranker_required_columns,
    ranker_score_expr,
    select_signal_rows,
    shift_signal_rows_to_execution,
    with_signal_scores,
)

__all__ = [
    "AMV_SIGNAL_EXPORT_COLUMNS",
    "SignalArtifact",
    "SignalArtifactConfig",
    "SignalAssemblyConfig",
    "WorkflowExportConfig",
    "assemble_ranker_signal",
    "base_candidate_expr",
    "build_backtest_signal_frame",
    "build_amv_phase_frame",
    "build_pb3_regime_gate_frame",
    "build_signal_meta",
    "export_context_sleeve",
    "export_limit_weakgate_sleeve",
    "export_ranker_sleeve",
    "git_commit",
    "ranker_required_columns",
    "ranker_score_expr",
    "relative_path",
    "select_signal_rows",
    "shift_signal_rows_to_execution",
    "timestamp_token",
    "with_signal_scores",
    "write_signal_artifact",
]
