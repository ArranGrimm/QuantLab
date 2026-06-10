"""权重网格扫描 — 批量生成策略 JSON → qlab run → 收集结果。

用法: uv run python research/scan_weights.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "strategies" / "amv" / "configs"
ARTIFACTS = ROOT / "artifacts"

# ═══════════════════════════════════════════════════════════════════════════
# 配置 — 改这里
# ═══════════════════════════════════════════════════════════════════════════

TEMPLATE = "trend-terrified"  # 扫描哪个策略模板
SCAN_PARAMS = {
    "P":  [1.0, 2.0, 3.0],
    "K":  [0.0, 0.5],
    "TF": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
}
BASELINE = "trend-p3"         # 对比基线


def compress(combo: dict) -> str:
    """P3_K0p5_TF1p5 → p3_k0p5_tf1p5"""
    return "_".join(f"{k.lower()}{str(v).replace('.','p')}" for k, v in combo.items())


def _run_strategy(strat_name: str) -> dict | None:
    """Export + backtest a strategy, return result.json contents."""
    from strategies.amv.registry import load_strategy
    from strategies.amv.pipeline import export_ranker_strategy, PipelineConfig
    from strategies.amv.export import write_signal_artifact
    from strategies.amv.data import resolve_end_date
    from utils.data_source import resolve_data_source

    strategy = load_strategy(strat_name)
    ds = resolve_data_source()
    cfg = PipelineConfig(data_source=ds)

    # 1. export
    artifact = export_ranker_strategy(strategy, cfg, ARTIFACTS / strat_name)

    # 2. backtest
    result = subprocess.run(
        ["cargo", "run", "-p", "bt-amv-topn", "--release", "--",
         "--data", str(artifact.signal_path),
         "--config", str(ROOT / strategy.preset.config),
         "--output-dir", str(ARTIFACTS / strat_name / "backtests" / datetime.now().strftime("%Y%m%d_%H%M%S")),
         ],
        capture_output=True, text=True, timeout=180,
        cwd=str(ROOT / "backtest-engine"),
    )
    if result.returncode != 0:
        return None

    # 3. read report.json (Rust output)
    bt_dir = ARTIFACTS / strat_name / "backtests"
    if not bt_dir.exists():
        return None
    latest = max(bt_dir.glob("*"), key=os.path.getmtime)
    rj = latest / "report.json"
    if not rj.exists():
        return None
    data = json.loads(rj.read_text())
    m = data.get("metrics", data)  # Rust writes metrics directly
    return {"from_report": {
        "total_return_pct": m.get("total_return_pct", 0.0),
        "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
        "total_trades": m.get("total_trades", 0),
        "win_rate_pct": m.get("win_rate_pct", 0.0),
    }}


def main() -> None:
    t0 = time.perf_counter()

    # generate param grid
    keys = list(SCAN_PARAMS)
    values = [SCAN_PARAMS[k] for k in keys]

    def _product(idx: int = 0, combo: dict | None = None) -> list[dict]:
        if combo is None: combo = {}
        if idx == len(keys):
            return [combo]
        results = []
        for v in values[idx]:
            results.extend(_product(idx + 1, {**combo, keys[idx]: v}))
        return results

    combos = _product()
    logger.info(f"扫描 {len(combos)} 个权重组合 ({TEMPLATE})")

    # read template
    template = json.loads((CONFIGS / f"{TEMPLATE}.json").read_text())

    results = []
    for i, combo in enumerate(combos):
        name = f"_scan_{compress(combo)}"
        cfg = json.loads(json.dumps(template))
        cfg["name"] = name
        cfg["ranker"]["params"] = combo

        tpath = CONFIGS / f"{name}.json"
        tpath.write_text(json.dumps(cfg, indent=2))

        res = _run_strategy(name)
        tpath.unlink(missing_ok=True)

        if res and "from_report" in res:
            m = res["from_report"]
            row = {**combo, "return": m["total_return_pct"], "maxdd": m["max_drawdown_pct"],
                   "trades": m["total_trades"], "win": m["win_rate_pct"]}
            results.append(row)
            print(f"  [{i+1}/{len(combos)}] {compress(combo):30s}  {row['return']:>+8.2f}%  maxdd={row['maxdd']:.1f}%")
        else:
            print(f"  [{i+1}/{len(combos)}] {compress(combo):30s}  FAILED")

    # clean temp artifact dirs
    for d in ARTIFACTS.glob("_scan_*"):
        shutil.rmtree(d, ignore_errors=True)

    if not results:
        logger.error("无有效结果")
        return

    # ── top 10 ──
    df = pl.DataFrame(results).sort("return", descending=True)
    print(f"\n  Top 10 (按 Return 排序):")
    print(f"  {'Rank':>4}  {'P':>3} {'K':>4} {'TF':>5}  {'Return':>9} {'MaxDD':>7} {'Win':>6}")
    print(f"  {'-'*45}")
    for i, row in enumerate(df.head(10).iter_rows(named=True)):
        print(f"  {i+1:>4}  {row['P']:>3.0f} {row['K']:>4.1f} {row['TF']:>5.2f}  "
              f"{row['return']:>+8.2f}% {row['maxdd']:>7.2f}% {row['win']:>5.1f}%")

    # ── vs baseline ──
    print(f"\n  vs baseline ({BASELINE})...")
    bl = _run_strategy(BASELINE)
    if bl and "from_report" in bl:
        bm = bl["from_report"]
        print(f"  baseline: {bm['total_return_pct']:+.2f}% / {bm['max_drawdown_pct']:.2f}% / {bm['total_trades']} trades")
        for i, row in enumerate(df.head(5).iter_rows(named=True)):
            delta = row["return"] - bm["total_return_pct"]
            print(f"  #{i+1}: {compress({k:row[k] for k in keys}):30s}  Δ={delta:+.2f}pp")

    logger.info(f"总耗时: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
