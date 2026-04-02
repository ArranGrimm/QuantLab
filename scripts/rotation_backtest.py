from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "rotation"
BACKTEST_ENGINE_DIR = ROOT / "backtest-engine"
DEFAULT_CONFIG = ROOT / "backtest-engine" / "crates" / "rotation" / "config.toml"


def timestamp_ms_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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


def choose_index(items: list[str], title: str) -> int:
    print(f"\n{title}")
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item}")
    while True:
        choice = input("输入编号: ").strip()
        if choice.isdigit():
            value = int(choice)
            if 1 <= value <= len(items):
                return value - 1
        print("无效输入，请重试。")


def pick_train_run() -> Path:
    train_dirs = sorted(
        [p for p in ARTIFACT_ROOT.iterdir() if p.is_dir() and (p / "signals.jsonl").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not train_dirs:
        raise FileNotFoundError("未找到任何包含 signals.jsonl 的 train run 目录")

    display = []
    for train_dir in train_dirs:
        signals = read_jsonl(train_dir / "signals.jsonl")
        display.append(f"{train_dir.name}  ({len(signals)} signals)")
    return train_dirs[choose_index(display, "选择 train run")]


def pick_signal_meta(train_dir: Path) -> Path:
    signals_index = train_dir / "signals.jsonl"
    rows = read_jsonl(signals_index)
    if not rows:
        raise FileNotFoundError(f"{signals_index} 中没有 signal 记录")

    rows = sorted(rows, key=lambda r: r.get("signal_id", ""), reverse=True)
    display = []
    resolved_meta_paths = []
    for row in rows:
        signal_id = row.get("signal_id", "unknown")
        label = row.get("label", "na")
        ema = row.get("export_ema_alpha", "na")
        top_n = row.get("top_n", "na")
        signal_meta_path = train_dir / row["signal_meta_path"]
        display.append(f"{signal_id}  (label={label}, ema={ema}, top_n={top_n})")
        resolved_meta_paths.append(signal_meta_path.resolve())
    return resolved_meta_paths[choose_index(display, f"选择 signal ({train_dir.name})")]


def run_rotation_backtest(signal_meta_path: Path, signal_parquet_path: Path, config_path: Path, release: bool) -> int:
    signal_dir = signal_meta_path.parent
    backtest_dir = signal_dir / "backtests" / timestamp_ms_token()
    backtest_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["cargo", "run", "-p", "bt-rotation"]
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
            str(backtest_dir),
        ]
    )

    print("\n即将运行:")
    print(f"  Signal Meta: {signal_meta_path}")
    print(f"  Signal File: {signal_parquet_path}")
    print(f"  Output Dir:  {backtest_dir}")
    print(f"  Config:      {config_path}")

    result = subprocess.run(cmd, cwd=BACKTEST_ENGINE_DIR)
    if result.returncode == 0:
        print("\n回测完成:")
        print(f"  {backtest_dir / 'report.txt'}")
        print(f"  {backtest_dir / 'report.json'}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotation signal backtest helper")
    parser.add_argument("signal", nargs="?", help="signal.meta.json / signal.parquet / signal directory")
    parser.add_argument("--pick", action="store_true", help="交互式选择 train run 和 signal")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Rotation config.toml 路径")
    parser.add_argument("--debug", action="store_true", help="使用 debug 构建而非 release")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()

    if args.signal:
        signal_meta_path, signal_parquet_path = resolve_signal_input(args.signal)
    else:
        train_dir = pick_train_run()
        signal_meta_path = pick_signal_meta(train_dir)
        signal_meta_path, signal_parquet_path = resolve_signal_input(str(signal_meta_path))

    return run_rotation_backtest(
        signal_meta_path=signal_meta_path,
        signal_parquet_path=signal_parquet_path,
        config_path=config_path,
        release=not args.debug,
    )


if __name__ == "__main__":
    raise SystemExit(main())
