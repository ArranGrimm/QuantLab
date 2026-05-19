from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_ENGINE_DIR = ROOT / "backtest-engine"
DEFAULT_BASE_CONFIG = ROOT / "backtest-engine" / "crates" / "amv-topn" / "config_10d.toml"


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "baseline_10d",
        "label": "baseline 10d",
        "overrides": {},
    },
    {
        "id": "hold_7d",
        "label": "hold 7td",
        "overrides": {"backtest.max_hold_trading_days": 7},
    },
    {
        "id": "hold_6d",
        "label": "hold 6td",
        "overrides": {"backtest.max_hold_trading_days": 6},
    },
    {
        "id": "hold_15d",
        "label": "hold 15td",
        "overrides": {"backtest.max_hold_trading_days": 15},
    },
    {
        "id": "stop_off",
        "label": "stop off",
        "overrides": {"stop_loss.enabled": False},
    },
    {
        "id": "stop_3pct",
        "label": "stop 3%",
        "overrides": {"stop_loss.enabled": True, "stop_loss.pct": 0.03},
    },
    {
        "id": "stop_8pct",
        "label": "stop 8%",
        "overrides": {"stop_loss.enabled": True, "stop_loss.pct": 0.08},
    },
    {
        "id": "bear_exit",
        "label": "bear exit",
        "overrides": {"exit.sell_on_bear_regime": True},
    },
    {
        "id": "gap_3pct",
        "label": "skip open gap >3%",
        "overrides": {"entry.max_open_gap_pct": 0.03},
    },
    {
        "id": "gap_5pct",
        "label": "skip open gap >5%",
        "overrides": {"entry.max_open_gap_pct": 0.05},
    },
    {
        "id": "bear_exit_gap_3pct",
        "label": "bear exit + gap 3%",
        "overrides": {"exit.sell_on_bear_regime": True, "entry.max_open_gap_pct": 0.03},
    },
    {
        "id": "hold_7d_gap_3pct",
        "label": "hold 7td + gap 3%",
        "overrides": {"backtest.max_hold_trading_days": 7, "entry.max_open_gap_pct": 0.03},
    },
    {
        "id": "hold_15d_bear_exit",
        "label": "hold 15td + bear exit",
        "overrides": {"backtest.max_hold_trading_days": 15, "exit.sell_on_bear_regime": True},
    },
    {
        "id": "stop_off_hold_7d",
        "label": "stop off + hold 7td",
        "overrides": {"stop_loss.enabled": False, "backtest.max_hold_trading_days": 7},
    },
    {
        "id": "stop_off_hold_5d",
        "label": "stop off + hold 5td",
        "overrides": {"stop_loss.enabled": False, "backtest.max_hold_trading_days": 5},
    },
    {
        "id": "stop_off_hold_6d",
        "label": "stop off + hold 6td",
        "overrides": {"stop_loss.enabled": False, "backtest.max_hold_trading_days": 6},
    },
    {
        "id": "stop_off_hold_15d",
        "label": "stop off + hold 15td",
        "overrides": {"stop_loss.enabled": False, "backtest.max_hold_trading_days": 15},
    },
    {
        "id": "stop_off_gap_3pct",
        "label": "stop off + gap 3%",
        "overrides": {"stop_loss.enabled": False, "entry.max_open_gap_pct": 0.03},
    },
    {
        "id": "stop_off_gap_5pct",
        "label": "stop off + gap 5%",
        "overrides": {"stop_loss.enabled": False, "entry.max_open_gap_pct": 0.05},
    },
    {
        "id": "stop_off_bear_exit",
        "label": "stop off + bear exit",
        "overrides": {"stop_loss.enabled": False, "exit.sell_on_bear_regime": True},
    },
    {
        "id": "stop_off_trailing_10_5",
        "label": "stop off + trailing 10/5",
        "overrides": {
            "stop_loss.enabled": False,
            "trailing_stop.enabled": True,
            "trailing_stop.activation_pct": 0.10,
            "trailing_stop.trailing_pct": 0.05,
        },
    },
    {
        "id": "stop_off_trailing_8_4",
        "label": "stop off + trailing 8/4",
        "overrides": {
            "stop_loss.enabled": False,
            "trailing_stop.enabled": True,
            "trailing_stop.activation_pct": 0.08,
            "trailing_stop.trailing_pct": 0.04,
        },
    },
]


def timestamp_ms_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"不支持的 TOML 值类型: {type(value).__name__}")


def resolve_signal_input(path_str: str) -> tuple[Path, Path]:
    path = Path(path_str).expanduser()
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


def apply_overrides(config: dict[str, Any], overrides: dict[str, object]) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    effective.setdefault("exit", {})
    for dotted_key, value in overrides.items():
        section, key = dotted_key.split(".", 1)
        if section not in effective:
            raise KeyError(f"配置缺少 section: {section}")
        effective[section][key] = value
    return effective


def write_config(path: Path, config: dict[str, Any], variant: dict[str, Any]) -> None:
    section_order = ["backtest", "entry", "exit", "stop_loss", "trailing_stop", "costs"]
    lines = [
        "# Auto-generated by scripts/amv_topn_enhancement_sweep.py",
        f"# Variant: {variant['id']} ({variant['label']})",
    ]
    overrides = variant["overrides"]
    if overrides:
        lines.append("# Overrides:")
        for dotted_key, value in overrides.items():
            lines.append(f"#   {dotted_key} = {value}")
    else:
        lines.append("# Overrides: none")
    lines.append("")

    for section in section_order:
        values = config.get(section)
        if not values:
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {format_toml_value(value)}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_variant(
    signal_parquet_path: Path,
    config_path: Path,
    output_dir: Path,
    release: bool,
) -> None:
    cmd = ["cargo", "run", "-p", "bt-amv-topn"]
    if release:
        cmd.append("--release")
    cmd.extend(
        [
            "--",
            "--data",
            str(signal_parquet_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    result = subprocess.run(cmd, cwd=BACKTEST_ENGINE_DIR, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"回测失败: {config_path}")


def summarize_report(variant: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    metrics = report["metrics"]
    cfg = report["backtest_config"]
    extra = report.get("extra") or {}
    return {
        "variant_id": variant["id"],
        "label": variant["label"],
        "overrides": variant["overrides"],
        "max_hold_trading_days": cfg.get("max_hold_trading_days"),
        "stop_loss_enabled": cfg.get("stop_loss_enabled"),
        "stop_loss_pct": cfg.get("stop_loss_pct"),
        "sell_on_bear_regime": cfg.get("sell_on_bear_regime"),
        "max_open_gap_pct": cfg.get("max_open_gap_pct"),
        "net_return_pct": metrics["total_return_pct"],
        "gross_return_pct": metrics["gross_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "win_rate_pct": metrics["win_rate_pct"],
        "total_trades": metrics["total_trades"],
        "avg_trades_per_day": metrics["avg_trades_per_day"],
        "total_costs": metrics["total_costs"],
        "final_portfolio": metrics["final_portfolio"],
        "limit_up_blocked": extra.get("limit_up_blocked"),
        "open_gap_blocked": extra.get("open_gap_blocked"),
        "bull_regime_blocked_signals": extra.get("bull_regime_blocked_signals"),
        "report_json_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV TopN enhancement sweep")
    parser.add_argument("signal", help="signal.meta.json / signal.parquet / signal directory")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--debug", action="store_true", help="使用 debug 构建而非 release")
    parser.add_argument(
        "--variant",
        action="append",
        help="只运行指定 variant_id；可重复传入。不传则运行默认小网格。",
    )
    args = parser.parse_args()

    signal_meta_path, signal_parquet_path = resolve_signal_input(args.signal)
    base_config_path = args.base_config.expanduser()
    if not base_config_path.is_absolute():
        base_config_path = (ROOT / base_config_path).resolve()
    else:
        base_config_path = base_config_path.resolve()

    selected_variants = VARIANTS
    if args.variant:
        wanted = set(args.variant)
        selected_variants = [variant for variant in VARIANTS if variant["id"] in wanted]
        missing = wanted - {variant["id"] for variant in selected_variants}
        if missing:
            raise ValueError(f"未知 variant: {sorted(missing)}")

    sweep_dir = signal_meta_path.parent / "backtests" / f"enhancement_{timestamp_ms_token()}"
    config_dir = sweep_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    base_config = read_toml(base_config_path)
    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(selected_variants, start=1):
        print(f"\n[{idx}/{len(selected_variants)}] {variant['id']} - {variant['label']}")
        effective_config = apply_overrides(base_config, variant["overrides"])
        config_path = config_dir / f"{variant['id']}.toml"
        output_dir = sweep_dir / variant["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config(config_path, effective_config, variant)
        run_variant(
            signal_parquet_path=signal_parquet_path,
            config_path=config_path,
            output_dir=output_dir,
            release=not args.debug,
        )
        rows.append(summarize_report(variant, output_dir / "report.json"))

    rows.sort(key=lambda row: (row["net_return_pct"], -row["max_drawdown_pct"]), reverse=True)
    summary = {
        "sweep_id": sweep_dir.name,
        "signal_meta_path": str(signal_meta_path),
        "signal_parquet_path": str(signal_parquet_path),
        "base_config_path": str(base_config_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
    }
    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nsummary: {summary_path}")
    print("Top variants:")
    for row in rows[:5]:
        print(
            f"  {row['variant_id']}: net={row['net_return_pct']:+.2f}% "
            f"mdd={row['max_drawdown_pct']:.2f}% trades={row['total_trades']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
