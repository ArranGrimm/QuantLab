"""
评审结果输出模块 — 终端格式化打印 + JSON 持久化。
"""
import json
from pathlib import Path
from datetime import datetime


def print_report(results: list[dict], min_score: float = 0.0):
    """将评审结果按 BUY / WATCH / SKIP 分组打印到终端。"""
    print()
    print("=" * 78)
    print(f"  B1 AI Agent 评审报告  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    if not results:
        print("\n  没有评审结果。\n")
        return

    buy = [r for r in results if r.get("verdict") == "BUY"]
    watch = [r for r in results if r.get("verdict") == "WATCH"]
    skip = [r for r in results if r.get("verdict") == "SKIP"]

    if buy:
        print(f"\n  BUY ({len(buy)})")
        print("-" * 78)
        _print_table(buy)

    if watch:
        print(f"\n  WATCH ({len(watch)})")
        print("-" * 78)
        _print_table(watch)

    if skip:
        print(f"\n  SKIP ({len(skip)})")
        print("-" * 78)
        _print_table(skip)

    print()
    print("=" * 78)
    print(
        f"  合计: {len(results)} 只  |  "
        f"BUY: {len(buy)}  |  WATCH: {len(watch)}  |  SKIP: {len(skip)}"
    )
    print("=" * 78)


def _print_table(items: list[dict]):
    header = (
        f"  {'代码':<14} {'总分':>5}  "
        f"{'趋势':>5} {'量价':>5} {'形态':>5} {'爆发':>5}  "
        f"{'类型':<10} {'点评'}"
    )
    print(header)
    print("  " + "-" * 74)
    for r in items:
        s = r.get("scores", {})
        print(
            f"  {r.get('code', '?'):<14} "
            f"{r.get('total_score', 0):>5.2f}  "
            f"{s.get('trend', 0):>5.1f} "
            f"{s.get('volume', 0):>5.1f} "
            f"{s.get('shape', 0):>5.1f} "
            f"{s.get('explosion', 0):>5.1f}  "
            f"{r.get('signal_type', '?'):<10} "
            f"{(r.get('comment', '') or '')[:28]}"
        )


def save_results(results: list[dict], output_dir: str | Path, target_date: str):
    """保存汇总结果到 suggestion.json。"""
    out = Path(output_dir) / target_date
    out.mkdir(parents=True, exist_ok=True)

    suggestion = {
        "date": target_date,
        "generated_at": datetime.now().isoformat(),
        "total": len(results),
        "buy_count": sum(1 for r in results if r.get("verdict") == "BUY"),
        "watch_count": sum(1 for r in results if r.get("verdict") == "WATCH"),
        "results": results,
    }

    path = out / "suggestion.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suggestion, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存: {path}")
