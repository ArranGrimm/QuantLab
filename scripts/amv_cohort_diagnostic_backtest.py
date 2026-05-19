from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_ENGINE_DIR = ROOT / "backtest-engine"
DEFAULT_CONFIGS = [
    ROOT / "backtest-engine" / "crates" / "amv-cohort-diagnostic" / "config_5td_rolling18.toml",
    ROOT / "backtest-engine" / "crates" / "amv-cohort-diagnostic" / "config_6td_rolling21.toml",
]


def timestamp_ms_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def run_one(signal_meta_path: Path, signal_parquet_path: Path, config_path: Path, release: bool) -> int:
    suffix = config_path.stem.removeprefix("config_")
    output_dir = signal_meta_path.parent / "backtests" / f"{suffix}_{timestamp_ms_token()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["cargo", "run", "-p", "bt-amv-cohort-diagnostic"]
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

    print("\n即将运行 close-to-close cohort diagnostic:")
    print(f"  Signal Meta: {signal_meta_path}")
    print(f"  Signal File: {signal_parquet_path}")
    print(f"  Config:      {config_path}")
    print(f"  Output Dir:  {output_dir}")

    result = subprocess.run(cmd, cwd=BACKTEST_ENGINE_DIR)
    if result.returncode == 0:
        print("\n诊断回测完成:")
        print(f"  {output_dir / 'report.txt'}")
        print(f"  {output_dir / 'report.json'}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV close-to-close cohort diagnostic helper")
    parser.add_argument("signal", help="signal.meta.json / signal.parquet / signal directory")
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="指定 config；可重复传入。不传则依次跑 5td rolling18 / 6td rolling21。",
    )
    parser.add_argument("--debug", action="store_true", help="使用 debug 构建而非 release")
    args = parser.parse_args()

    signal_meta_path, signal_parquet_path = resolve_signal_input(args.signal)
    configs = args.config or DEFAULT_CONFIGS
    for config in configs:
        config_path = config.expanduser()
        if not config_path.is_absolute():
            config_path = (ROOT / config_path).resolve()
        else:
            config_path = config_path.resolve()
        code = run_one(signal_meta_path, signal_parquet_path, config_path, release=not args.debug)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
