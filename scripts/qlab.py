from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_ENGINE_DIR = ROOT / "backtest-engine"
INTERPRETATION_TEXTS = {
    "core": "raw execution 显著压低静态 Ref/P3/context 收益，但 context combo 仍是当前最强核心静态 challenger。",
    "pb3": "PB3 gated rolling 在 raw execution 下收益低于旧 adjusted 诊断输入，进入组合权重前需要重新做 allocation 评估。",
    "limit_ecology": "首板后回踩事件 sleeve 在 raw execution 下吸引力下降；weak-window 变体仍未达到 allocation-ready。",
}
INTERPRETATION_LABELS = {
    "core": "核心静态",
    "pb3": "PB3",
    "limit_ecology": "涨停生态",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.amv.attribution import write_trade_attribution  # noqa: E402
from strategies.amv.registry import (  # noqa: E402
    BACKTEST_PRESETS,
    EXPORT_TARGETS,
    REPORT_ALIASES,
    resolve_project_path,
)
from strategies.amv.sleeves import SLEEVE_SPECS  # noqa: E402
from strategies.amv.status import P3_RAW_VS_ADJUSTED_ATTRIBUTION, RAW_EXECUTION_STATUS  # noqa: E402
from strategies.amv.workflows import (  # noqa: E402
    WorkflowExportConfig,
    export_context_sleeve,
    export_limit_weakgate_sleeve,
    export_ranker_sleeve,
)


DEFAULT_QMT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_SIGNAL_OUTPUT_ROOT = ROOT / "artifacts" / "amv_static_sleeve_signals"
DEFAULT_SECTOR_MAP = ROOT / "data" / "sector_map_em.csv"
NATIVE_EXPORT_TARGETS = {"context", "limit-weakgate", "pb3-gated", "ref", "p3"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(value: Any, *, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.2f}%"


def price_basis_label(value: Any) -> str:
    labels = {
        "raw_ohlc_pre_close for all new artifacts": "新 artifact 统一使用 raw OHLC / raw pre-close",
        "raw_ohlc_pre_close": "raw OHLC / raw pre-close",
        "adjusted_ohlc_fallback": "旧 artifact 回退 adjusted OHLC",
    }
    return labels.get(str(value), str(value))


def run_command(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    print("将执行命令:")
    print("  " + " ".join(cmd))
    if cwd is not None:
        print(f"工作目录: {display_path(cwd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def build_native_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SIGNAL_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    parser.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    parser.add_argument("--sector-start-date", default="2019-01-01")
    parser.add_argument("--refresh-sector-map", action="store_true")
    parser.add_argument("--sector-map-request-sleep", type=float, default=0.35)
    parser.add_argument("--rank-source", choices=["5d", "10d", "20d", "mix_10_20"], default="mix_10_20")
    parser.add_argument("--sector-penalty-mode", choices=["bucket", "linear"], default="linear")
    parser.add_argument(
        "--relative-confirm",
        choices=["none", "rel5_under0", "rel10_under0", "rel20_under0"],
        default="rel20_under0",
    )
    parser.add_argument("--bottom-rank-threshold", type=float, default=0.40)
    parser.add_argument("--sector-penalties", "--sector-penalty", dest="sector_penalty", type=float, default=0.02)
    parser.add_argument("--medium-penalty-mode", choices=["bucket", "linear"], default="linear")
    parser.add_argument("--weak-threshold", type=float, default=0.50)
    parser.add_argument("--medium-penalties", "--medium-penalty", dest="medium_penalty", type=float, default=0.03)
    parser.add_argument(
        "--pb3-regime-gate",
        choices=["none", "aged_non_accel_or_chaos"],
        default="aged_non_accel_or_chaos",
    )
    parser.add_argument("--price-limit-tolerance", type=float, default=0.001)
    return parser


def command_export_native(args: argparse.Namespace, target: Any) -> int:
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    native_parser = build_native_export_parser()
    native_args, unknown = native_parser.parse_known_args(extra_args)
    if unknown:
        raise ValueError(f"native export 无法识别的参数: {' '.join(unknown)}")

    sleeve = SLEEVE_SPECS[args.target]
    config = WorkflowExportConfig(
        qmt_db=native_args.qmt_db,
        output_root=native_args.output_root,
        start_date=native_args.start_date,
        end_date=native_args.end_date,
        st_snapshot_date=native_args.st_snapshot_date,
        mv_min=native_args.mv_min,
        amount_ma20_min=native_args.amount_ma20_min,
        top_n=native_args.top_n,
        amv_bull_trigger_pct=native_args.amv_bull_trigger_pct,
        amv_bull_lookback_days=native_args.amv_bull_lookback_days,
        amv_bear_trigger_1d_pct=native_args.amv_bear_trigger_1d_pct,
        amv_effective_lag_days=native_args.amv_effective_lag_days,
        sector_map=native_args.sector_map,
        sector_start_date=native_args.sector_start_date,
        refresh_sector_map=native_args.refresh_sector_map,
        sector_map_request_sleep=native_args.sector_map_request_sleep,
        rank_source=native_args.rank_source,
        sector_penalty_mode=native_args.sector_penalty_mode,
        relative_confirm=native_args.relative_confirm,
        bottom_rank_threshold=native_args.bottom_rank_threshold,
        sector_penalty=native_args.sector_penalty,
        medium_penalty_mode=native_args.medium_penalty_mode,
        weak_threshold=native_args.weak_threshold,
        medium_penalty=native_args.medium_penalty,
        pb3_regime_gate=native_args.pb3_regime_gate,
        price_limit_tolerance=native_args.price_limit_tolerance,
    )

    print(f"导出目标: {args.target} - {target.description}")
    print("导出实现: native strategies/amv")
    print(f"QMT 数据库: {display_path(config.qmt_db)}")
    print(f"输出根目录: {display_path(config.output_root)}")
    print(f"日期范围: {config.start_date} ~ {config.end_date}")
    if args.dry_run:
        return 0

    if args.target == "context":
        artifact = export_context_sleeve(sleeve, config, repo_root=ROOT)
    elif args.target == "limit-weakgate":
        artifact = export_limit_weakgate_sleeve(sleeve, config, repo_root=ROOT)
    else:
        artifact = export_ranker_sleeve(sleeve, config, repo_root=ROOT)
    print(f"Saved signal meta: {display_path(artifact.meta_path)}")
    print(f"Signal parquet: {display_path(artifact.signal_path)}")
    return 0


def load_ground_truth_runs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = RAW_EXECUTION_STATUS
    runs = {run["name"]: run for run in report.get("runs", [])}
    return report, runs


def command_status(args: argparse.Namespace) -> int:
    report, runs = load_ground_truth_runs()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("QuantLab 当前状态")
    print(f"- 成交价格口径: {price_basis_label(report.get('price_basis', 'unknown'))}")
    print(f"- 报告生成时间: {report.get('generated_at', 'unknown')}")
    print(f"- 第一阅读入口: {display_path(ROOT / 'CURRENT_STATE.md')}")
    print("")

    for alias in ["ref", "p3", "context", "pb3-gated", "limit-base", "limit-weakgate"]:
        report_alias = REPORT_ALIASES[alias]
        run = runs.get(report_alias.run_name)
        if run is None:
            continue
        print(
            f"{alias:14} {pct(run.get('total_return_pct')):>9} "
            f"MaxDD {pct(run.get('max_drawdown_pct'), signed=False):>8} "
            f"交易 {run.get('total_trades', 'n/a'):>4}  {report_alias.description}"
        )

    if report.get("interpretation"):
        print("")
        print("当前解读")
        for key, value in report.get("interpretation", {}).items():
            label = INTERPRETATION_LABELS.get(key, key)
            text = INTERPRETATION_TEXTS.get(key, value)
            print(f"- {label}: {text}")
    return 0


def command_export(args: argparse.Namespace) -> int:
    target = EXPORT_TARGETS[args.target]
    if args.target in NATIVE_EXPORT_TARGETS:
        return command_export_native(args, target)

    if target.script is None:
        raise ValueError(f"导出目标 {args.target} 没有关联旧脚本，也未接入 native workflow")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    script_path = resolve_project_path(ROOT, target.script)
    cmd = [sys.executable, str(script_path), *target.args, *extra_args]
    print(f"导出目标: {args.target} - {target.description}")
    return run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


def resolve_signal_input(path_text: str) -> tuple[Path, Path]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()

    if path.is_dir():
        meta_path = path / "signal.meta.json"
        parquet_path = path / "signal.parquet"
    elif path.suffix.lower() == ".json":
        meta_path = path
        meta = read_json(meta_path)
        parquet_rel = meta.get("signal_path") or meta.get("canonical_signal_path") or "signal.parquet"
        parquet_path = (meta_path.parent / parquet_rel).resolve()
    else:
        parquet_path = path
        meta_path = parquet_path.with_name("signal.meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"signal.meta.json 不存在: {meta_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"signal.parquet 不存在: {parquet_path}")
    return meta_path, parquet_path


def command_backtest(args: argparse.Namespace) -> int:
    preset = BACKTEST_PRESETS[args.preset]
    config_path = resolve_project_path(ROOT, preset.config)
    if not config_path.exists():
        raise FileNotFoundError(f"回测 config 不存在: {config_path}")

    meta_path, parquet_path = resolve_signal_input(args.signal)
    output_dir = meta_path.parent / "backtests" / f"{args.preset}_{timestamp_token()}"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["cargo", "run", "-p", "bt-amv-topn"]
    if not args.debug:
        cmd.append("--release")
    cmd.extend(
        [
            "--",
            "--data",
            str(parquet_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    print(f"回测 preset: {args.preset} - {preset.description}")
    print(f"信号 meta: {display_path(meta_path)}")
    print(f"信号文件: {display_path(parquet_path)}")
    print(f"输出目录: {display_path(output_dir)}")
    return run_command(cmd, cwd=BACKTEST_ENGINE_DIR, dry_run=args.dry_run)


def command_compare(args: argparse.Namespace) -> int:
    _, runs = load_ground_truth_runs()
    left_alias = REPORT_ALIASES[args.left]
    right_alias = REPORT_ALIASES[args.right]
    left = runs[left_alias.run_name]
    right = runs[right_alias.run_name]

    total_delta = right["total_return_pct"] - left["total_return_pct"]
    dd_delta = right["max_drawdown_pct"] - left["max_drawdown_pct"]
    result = {
        "left": {"alias": args.left, **left},
        "right": {"alias": args.right, **right},
        "delta_total_return_pp": round(total_delta, 2),
        "delta_max_drawdown_pp": round(dd_delta, 2),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"对比: {args.left} -> {args.right}")
    print(
        f"- 总收益: {pct(left['total_return_pct'])} -> "
        f"{pct(right['total_return_pct'])} ({total_delta:+.2f}pp)"
    )
    print(
        f"- MaxDD: {pct(left['max_drawdown_pct'], signed=False)} -> "
        f"{pct(right['max_drawdown_pct'], signed=False)} ({dd_delta:+.2f}pp)"
    )
    print(f"- 交易数: {left.get('total_trades', 'n/a')} -> {right.get('total_trades', 'n/a')}")
    print(f"- 左侧报告: {left.get('backtest_dir')}")
    print(f"- 右侧报告: {right.get('backtest_dir')}")
    return 0


def command_attribution(args: argparse.Namespace) -> int:
    if args.target == "p3-raw-vs-adjusted":
        report = P3_RAW_VS_ADJUSTED_ATTRIBUTION
        summary = report.get("summary", {})
        adjusted = summary.get("adjusted", {})
        raw = summary.get("raw", {})
        delta = summary.get("delta_raw_minus_adjusted", {})
        overlap = summary.get("overlap", {})
        matched = summary.get("matched_trade_sums", {})
        print("P3 raw-vs-adjusted 归因报告")
        print(f"- 报告摘要: strategies/amv/status.py::{report.get('report_name')}")
        print(
            f"- 总收益: adjusted {pct(adjusted.get('total_return_pct'))} -> "
            f"raw {pct(raw.get('total_return_pct'))} ({delta.get('total_return_pp', 0.0):+.2f}pp)"
        )
        print(
            f"- MaxDD: adjusted {pct(adjusted.get('max_drawdown_pct'), signed=False)} -> "
            f"raw {pct(raw.get('max_drawdown_pct'), signed=False)} ({delta.get('max_drawdown_pp', 0.0):+.2f}pp)"
        )
        print(
            f"- 交易重合: {overlap.get('common_entry_code', 'n/a')}/"
            f"{overlap.get('adjusted_trades', 'n/a')}，same exit date/reason "
            f"{overlap.get('same_exit_date', 'n/a')}/{overlap.get('same_exit_reason', 'n/a')}"
        )
        print(
            f"- PnL 差异: {matched.get('pnl_delta_raw_minus_adjusted', 0.0):+.2f}，"
            f"notional effect {matched.get('notional_effect', 0.0):+.2f}，"
            f"return effect {matched.get('return_effect', 0.0):+.2f}"
        )
        return 0

    missing = [
        name
        for name, value in {
            "--left-backtest": args.left_backtest,
            "--right-backtest": args.right_backtest,
            "--out": args.out,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"attribution trade 缺少必要参数: {', '.join(missing)}")

    if args.dry_run:
        print("将生成交易归因:")
        print(f"  left: {args.left_backtest} ({args.left_label})")
        print(f"  right: {args.right_backtest} ({args.right_label})")
        print(f"  out: {args.out}")
        print(f"  top_n: {args.top_n}")
        return 0

    attribution = write_trade_attribution(
        args.left_backtest,
        args.right_backtest,
        left_label=args.left_label,
        right_label=args.right_label,
        out=args.out,
        top_n=args.top_n,
    )
    delta = attribution["summary"]["delta_right_minus_left"]
    print(f"Wrote {args.out}")
    print(
        "return_delta={:.2f}pp maxdd_delta={:.2f}pp exact_overlap={}".format(
            delta.get("total_return_pct", 0.0),
            delta.get("max_drawdown_pct", 0.0),
            attribution["overlap"]["exact_overlap_count"],
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantLab 日常工作流 CLI", add_help=False)
    parser._positionals.title = "位置参数"
    parser._optionals.title = "可选参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="查看当前 raw-execution ground truth 状态", add_help=False)
    status._positionals.title = "位置参数"
    status._optionals.title = "可选参数"
    status.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    status.add_argument("--json", action="store_true", help="输出原始状态 JSON")
    status.set_defaults(func=command_status)

    export = subparsers.add_parser("export", help="导出 canonical 信号目标", add_help=False)
    export._positionals.title = "位置参数"
    export._optionals.title = "可选参数"
    export.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    export.add_argument("target", choices=sorted(EXPORT_TARGETS))
    export.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    export.set_defaults(func=command_export)

    backtest = subparsers.add_parser("backtest", help="用命名 preset 运行 bt-amv-topn", add_help=False)
    backtest._positionals.title = "位置参数"
    backtest._optionals.title = "可选参数"
    backtest.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    backtest.add_argument("signal", help="signal 目录、signal.meta.json 或 signal.parquet")
    backtest.add_argument("--preset", choices=sorted(BACKTEST_PRESETS), default="6td-static")
    backtest.add_argument("--debug", action="store_true", help="使用 debug 构建而不是 release")
    backtest.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    backtest.set_defaults(func=command_backtest)

    compare = subparsers.add_parser("compare", help="对比当前 raw-execution 报告别名", add_help=False)
    compare._positionals.title = "位置参数"
    compare._optionals.title = "可选参数"
    compare.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    compare.add_argument("left", choices=sorted(REPORT_ALIASES))
    compare.add_argument("right", choices=sorted(REPORT_ALIASES))
    compare.add_argument("--json", action="store_true", help="输出对比 JSON")
    compare.set_defaults(func=command_compare)

    attribution = subparsers.add_parser("attribution", help="运行 canonical 或通用交易归因", add_help=False)
    attribution._positionals.title = "位置参数"
    attribution._optionals.title = "可选参数"
    attribution.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    attribution.add_argument("target", choices=["p3-raw-vs-adjusted", "trade"])
    attribution.add_argument("--left-backtest", type=Path)
    attribution.add_argument("--right-backtest", type=Path)
    attribution.add_argument("--left-label", default="left")
    attribution.add_argument("--right-label", default="right")
    attribution.add_argument("--out", type=Path)
    attribution.add_argument("--top-n", type=int, default=12)
    attribution.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    attribution.set_defaults(func=command_attribution)

    return parser


def main() -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    if args.command == "export":
        args.extra_args = extra_args
    elif extra_args:
        parser.error(f"无法识别的参数: {' '.join(extra_args)}")
    try:
        return args.func(args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
