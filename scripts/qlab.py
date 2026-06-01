"""QuantLab 日常工作流 CLI — AMV 策略体系的唯一入口。"""

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
ARTIFACTS_DIR = ROOT / "artifacts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.amv.registry import KNOWN_STRATEGIES, Strategy, resolve_project_path  # noqa: E402
from strategies.amv.workflows import WorkflowExportConfig, export_strategy  # noqa: E402

DEFAULT_QMT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("  " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


def _is_canonical(result: dict) -> bool:
    return result.get("is_canonical", False)


def _load_canonical_results() -> dict[str, dict[str, Any]]:
    """扫描 artifacts/*/backtests/*.json，返回 {strategy_name: latest_canonical_result}"""
    canonical: dict[str, Any] = {}
    sig_dir = ARTIFACTS_DIR
    if not sig_dir.exists():
        return canonical

    latest: dict[str, tuple[str, dict]] = {}
    for bt_dir in sorted(sig_dir.glob("*/backtests/*/")):
        result_json = bt_dir / "result.json"
        if not result_json.exists():
            continue
        result = read_json(result_json)
        if not _is_canonical(result):
            continue
        name = result.get("strategy", bt_dir.parent.parent.name)
        ts = bt_dir.name
        if name not in latest or ts > latest[name][0]:
            latest[name] = (ts, result)

    return {name: data[1] for name, data in latest.items()}


# === status ===


def command_status(args: argparse.Namespace) -> int:
    canonical = _load_canonical_results()
    if not canonical:
        print("没有 canonical 回测结果。运行 qlab run <strategy> 生成。")
        return 0

    families: dict[str, list[tuple[str, dict]]] = {}
    for name, result in canonical.items():
        strategy = KNOWN_STRATEGIES.get(name)
        family = strategy.family if strategy else "unknown"
        families.setdefault(family, []).append((name, result))

    print("QuantLab 当前状态")
    print("")

    family_labels = {"trend": "趋势突破家族", "pullback": "回调反弹家族", "event": "事件驱动家族"}
    for family, items in families.items():
        print(f"── {family_labels.get(family, family)} ──")
        for name, result in items:
            strategy = KNOWN_STRATEGIES.get(name)
            m = result.get("from_report", {})
            status_mark = {"baseline": "[基线]", "challenger": "[挑战]", "complementary": "[互补]", "research": "[研究]"}.get(
                strategy.status if strategy else "", ""
            )
            print(
                f"  {name:24} {pct(m.get('total_return_pct')):>9} "
                f"MaxDD {pct(m.get('max_drawdown_pct')):>8} "
                f"交易 {m.get('total_trades', 0):>4}  {status_mark}"
            )
            if strategy and strategy.rules:
                print(f"  {'':24} 规则: {' → '.join(strategy.rules)}")
            if strategy and strategy.description:
                print(f"  {'':24} {strategy.description}")
        print("")
    return 0


# === export ===


def command_export(args: argparse.Namespace) -> int:
    strategy = KNOWN_STRATEGIES[args.strategy]
    config = WorkflowExportConfig(qmt_db=Path(args.qmt_db))
    output_dir = ARTIFACTS_DIR / strategy.name

    print(f"导出策略: {strategy.name} — {strategy.label}")
    print(f"数据库: {display_path(config.qmt_db)}")
    print(f"输出: {display_path(output_dir)}")

    artifact = export_strategy(strategy, config, output_dir)
    print(f"Done: {display_path(artifact.signal_path)}")
    return 0


# === backtest ===


def _compute_yearly_from_equity(equity_csv: Path) -> dict[str, float]:
    """从 daily_equity.csv 计算年度权益收益率。"""
    yearly: dict[str, float] = {}
    try:
        with open(equity_csv) as f:
            f.readline()  # skip header
            first_row = f.readline()
            if not first_row:
                return yearly
            parts = first_row.strip().split(",")
            prev_value = float(parts[-1])
            prev_year = parts[0][:4]
            yearly[prev_year] = 0.0

            for line in f:
                parts = line.strip().split(",")
                year = parts[0][:4]
                total_value = float(parts[-1])
                if year != prev_year:
                    yearly[prev_year] = round((total_value / prev_value - 1) * 100, 2)
                    prev_value = total_value
                    prev_year = year
            if prev_year not in yearly:
                final_value = float(parts[-1])
                yearly[prev_year] = round((final_value / prev_value - 1) * 100, 2)
    except (OSError, ValueError, IndexError):
        pass
    return yearly


def _synthesize_result(
    report_json: Path, equity_csv: Path, strategy: Strategy, preset_name: str, is_canonical: bool
) -> None:
    """从 Rust 输出合成 result.json"""
    report = read_json(report_json)
    metrics = report.get("metrics", {})

    result = {
        "strategy": strategy.name,
        "ran_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "preset": {"name": preset_name, "config": strategy.preset.config},
        "is_canonical": is_canonical,
        "from_report": {
            "total_trades": metrics.get("total_trades"),
            "win_rate_pct": metrics.get("win_rate_pct"),
            "total_return_pct": metrics.get("total_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "gross_return_pct": metrics.get("gross_return_pct"),
            "total_costs": metrics.get("total_costs"),
            "trading_days": metrics.get("trading_days"),
        },
        "yearly": _compute_yearly_from_equity(equity_csv),
    }
    result_path = report_json.parent / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def command_backtest(args: argparse.Namespace) -> int:
    strategy = KNOWN_STRATEGIES[args.strategy]
    signal_dir = ARTIFACTS_DIR / strategy.name
    signal_path = signal_dir / "signal.parquet"

    if not signal_path.exists():
        print(f"信号文件不存在: {display_path(signal_path)}")
        print(f"先运行: qlab export {strategy.name}")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = signal_dir / "backtests" / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_project_path(ROOT, strategy.preset.config)

    # 合成临时 TOML（如有参数覆盖）
    if args.top_n or args.max_hold or args.max_positions:
        toml_path = output_dir / "_backtest.toml"
        _write_tmp_toml(toml_path, config_path, args)
        config_path = toml_path

    cmd = [
        "cargo", "run", "-p", "bt-amv-topn", "--release", "--",
        "--data", str(signal_path),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]

    print(f"回测: {strategy.name} → {ts}")
    print(f"Preset: {strategy.preset.name}")
    rc = run_cmd(cmd, cwd=BACKTEST_ENGINE_DIR)
    if rc != 0:
        return rc

    # 合成 result.json
    report_json = output_dir / "report.json"
    equity_csv = output_dir / "daily_equity.csv"
    if report_json.exists() and equity_csv.exists():
        is_canonical = not (args.top_n or args.max_hold or args.max_positions)
        _synthesize_result(report_json, equity_csv, strategy, strategy.preset.name, is_canonical)
        print(f"Result: {display_path(output_dir / 'result.json')}")

    return 0


def _write_tmp_toml(toml_path: Path, base_toml: Path, args: argparse.Namespace) -> None:
    """复制基础 TOML 并覆盖 CLI 指定的参数。"""
    lines = base_toml.read_text(encoding="utf-8").split("\n")
    overrides = {}
    if args.top_n:
        overrides["top_n"] = args.top_n
    if args.max_hold:
        overrides["max_hold_trading_days"] = args.max_hold
    if args.max_positions:
        overrides["max_positions"] = args.max_positions

    result_lines = []
    for line in lines:
        written = False
        for key, value in overrides.items():
            if line.strip().startswith(key):
                result_lines.append(f"{key} = {value}")
                written = True
                break
        if not written:
            result_lines.append(line)
    toml_path.write_text("\n".join(result_lines), encoding="utf-8")


# === results ===


def command_results(args: argparse.Namespace) -> int:
    bt_root = ARTIFACTS_DIR / args.strategy / "backtests"
    if not bt_root.exists():
        print(f"没有回测结果: {args.strategy}")
        return 0

    results = []
    for bt_dir in sorted(bt_root.iterdir()):
        if not bt_dir.is_dir():
            continue
        result_json = bt_dir / "result.json"
        if not result_json.exists():
            continue
        result = read_json(result_json)
        results.append((bt_dir.name, result))

    if not results:
        print(f"没有 result.json: {args.strategy}")
        return 0

    print(f"{'Time':20} {'Return':>9} {'MaxDD':>8} {'Trades':>6} {'Status':12} {'Yearly'}")
    print("-" * 85)
    for ts, r in results:
        m = r.get("from_report", {})
        status = "canonical" if r.get("is_canonical") else "custom"
        yearly = r.get("yearly", {})
        y2026 = yearly.get("2026", 0)
        print(
            f"{ts:20} {pct(m.get('total_return_pct')):>9} "
            f"{pct(m.get('max_drawdown_pct')):>8} {m.get('total_trades', 0):>6} "
            f"{status:12} 2026: {pct(y2026)}"
        )

    if args.diff and len(results) >= 2:
        canonical_results = [(ts, r) for ts, r in results if r.get("is_canonical")]
        if len(canonical_results) >= 2:
            prev_ts, prev = canonical_results[-2]
            curr_ts, curr = canonical_results[-1]
            prev_m = prev.get("from_report", {})
            curr_m = curr.get("from_report", {})
            delta_return = (curr_m.get("total_return_pct", 0) or 0) - (prev_m.get("total_return_pct", 0) or 0)
            delta_dd = (curr_m.get("max_drawdown_pct", 0) or 0) - (prev_m.get("max_drawdown_pct", 0) or 0)
            print(f"\n── diff: {prev_ts} → {curr_ts} ──")
            print(f"  Return: {delta_return:+.2f}pp")
            print(f"  MaxDD:  {delta_dd:+.2f}pp")
            for year in sorted(set(list(prev.get("yearly", {})) + list(curr.get("yearly", {})))):
                py = prev.get("yearly", {}).get(year, 0)
                cy = curr.get("yearly", {}).get(year, 0)
                if py or cy:
                    print(f"  {year}: {py:+.2f}% → {cy:+.2f}%  ({cy - py:+.2f}pp)")

    return 0


# === run ===


def command_run(args: argparse.Namespace) -> int:
    export_args = argparse.Namespace(strategy=args.strategy, qmt_db=args.qmt_db)
    rc = command_export(export_args)
    if rc != 0:
        return rc
    return command_backtest(args)


# === CLI ===


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantLab 日常工作流 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p = subparsers.add_parser("status", help="展示所有策略最新 canonical 指标")
    p.set_defaults(func=command_status)

    # export
    p = subparsers.add_parser("export", help="导出信号文件")
    p.add_argument("strategy", choices=sorted(KNOWN_STRATEGIES.keys()), help="策略名")
    p.add_argument("--qmt-db", default=str(DEFAULT_QMT_DB), help="QMT 数据库路径")
    p.set_defaults(func=command_export)

    # backtest
    p = subparsers.add_parser("backtest", help="运行 Rust 回测")
    p.add_argument("strategy", choices=sorted(KNOWN_STRATEGIES.keys()), help="策略名")
    p.add_argument("--top-n", type=int)
    p.add_argument("--max-hold", type=int, help="最大持仓天数")
    p.add_argument("--max-positions", type=int, help="最大持仓数")
    p.set_defaults(func=command_backtest)

    # results
    p = subparsers.add_parser("results", help="查看历史回测结果")
    p.add_argument("strategy", choices=sorted(KNOWN_STRATEGIES.keys()), help="策略名")
    p.add_argument("--diff", action="store_true", help="最新两次 canonical 差异")
    p.set_defaults(func=command_results)

    # run
    p = subparsers.add_parser("run", help="export + backtest 一步完成")
    p.add_argument("strategy", choices=sorted(KNOWN_STRATEGIES.keys()), help="策略名")
    p.add_argument("--qmt-db", default=str(DEFAULT_QMT_DB), help="QMT 数据库路径")
    p.add_argument("--top-n", type=int)
    p.add_argument("--max-hold", type=int)
    p.add_argument("--max-positions", type=int)
    p.set_defaults(func=command_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
