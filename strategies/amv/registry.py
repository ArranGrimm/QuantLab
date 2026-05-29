from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from strategies.amv.sleeves import SLEEVE_SPECS


@dataclass(frozen=True)
class ExportTarget:
    """由 scripts/qlab.py 包装的稳定导出目标。"""

    description: str
    script: str | None
    args: tuple[str, ...]


@dataclass(frozen=True)
class BacktestPreset:
    description: str
    config: str


@dataclass(frozen=True)
class ReportAlias:
    description: str
    run_name: str


EXPORT_TARGETS: dict[str, ExportTarget] = {
    "ref": ExportTarget(
        description=SLEEVE_SPECS["ref"].label,
        script=None,
        args=SLEEVE_SPECS["ref"].export_args,
    ),
    "p3": ExportTarget(
        description=SLEEVE_SPECS["p3"].label,
        script=None,
        args=SLEEVE_SPECS["p3"].export_args,
    ),
    "context": ExportTarget(
        description=SLEEVE_SPECS["context"].label,
        script=None,
        args=SLEEVE_SPECS["context"].export_args,
    ),
    "pb3-gated": ExportTarget(
        description=SLEEVE_SPECS["pb3-gated"].label,
        script=None,
        args=SLEEVE_SPECS["pb3-gated"].export_args,
    ),
    "limit-weakgate": ExportTarget(
        description=SLEEVE_SPECS["limit-weakgate"].label,
        script=None,
        args=SLEEVE_SPECS["limit-weakgate"].export_args,
    ),
}


BACKTEST_PRESETS: dict[str, BacktestPreset] = {
    "6td-static": BacktestPreset(
        description="持有 6 个交易日，static strict Top3，无止损",
        config="backtest-engine/crates/amv-topn/config_6td_static_strict_top3_no_stop.toml",
    ),
    "5td-static": BacktestPreset(
        description="持有 5 个交易日，static strict Top3，无止损",
        config="backtest-engine/crates/amv-topn/config_5td_static_strict_top3_no_stop.toml",
    ),
    "3td-static": BacktestPreset(
        description="持有 3 个交易日，static strict Top3，无止损",
        config="backtest-engine/crates/amv-topn/config_3td_static_strict_top3_no_stop.toml",
    ),
    "6td-rolling": BacktestPreset(
        description="持有 6 个交易日，rolling21 refill Top10，无止损",
        config="backtest-engine/crates/amv-topn/config_6td_rolling21_refill_top10_no_stop.toml",
    ),
}


REPORT_ALIASES: dict[str, ReportAlias] = {
    "ref": ReportAlias("raw 口径 Reference P2 6td static", "ref_p2_6td"),
    "p3": ReportAlias("raw 口径 Candidate P3 6td static", "p3_6td"),
    "context": ReportAlias("raw 口径 P3 上下文组合 6td static", "context_combo_6td"),
    "pb3-gated": ReportAlias("raw 口径 PB3 gated 6td rolling", "pb3_gated_rolling"),
    "limit-base": ReportAlias("raw 口径 涨停首板 base 5td static", "limit_first_board_5td"),
    "limit-weakgate": ReportAlias("raw 口径 涨停首板 weakgate 5td static", "limit_weakgate_5td"),
}


def resolve_project_path(root: Path, relative_path: str) -> Path:
    return (root / relative_path).resolve()
