"""
B1 AI Agent — 主入口

用法:
  python -m agent.run                        # 使用最新交易日
  python -m agent.run --date 2026-02-24      # 指定日期
  python -m agent.run --skip-review          # 只生成图表, 跳过 AI 评审
  python -m agent.run --config agent/config.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml
import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import load_daily_data_full, calc_b1_factors_wmacd  # noqa: E402
from agent.chart import export_chart  # noqa: E402
from agent.context import build_context  # noqa: E402
from agent.report import print_report, save_results  # noqa: E402


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_candidates(
    df_signals: pl.LazyFrame,
    target_date: str,
    config: dict,
) -> pl.DataFrame:
    """筛选当日 B1 信号, 按 rank_by 排序后取 top_n。"""
    from datetime import datetime

    top_n = config.get("top_n", 20)
    rank_by = config.get("rank_by", "rw_dif_pct")
    rank_ascending = config.get("rank_ascending", False)

    target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

    return (
        df_signals
        .filter(
            (pl.col("date") == target_dt)
            & pl.col("b1_signal")
        )
        .sort(rank_by, descending=not rank_ascending)
        .head(top_n)
        .collect()
    )


def main():
    parser = argparse.ArgumentParser(description="B1 AI Agent — 每日选股 + AI 评审")
    parser.add_argument("--date", type=str, default=None, help="目标日期 YYYY-MM-DD (默认: 最新交易日)")
    parser.add_argument("--config", type=str, default=str(ROOT / "agent" / "config.yaml"))
    parser.add_argument("--skip-review", action="store_true", help="跳过 AI 评审, 只生成图表和指标")
    args = parser.parse_args()

    config = load_config(args.config)

    # ─── Step 1: 数据加载 ─────────────────────────────────────────
    db_path = config["data"]["db_path"]
    if not Path(db_path).is_absolute():
        db_path = str(ROOT / db_path)

    print(f"[1/5] 连接数据库: {db_path}")
    conn = duckdb.connect(db_path, read_only=True)

    print("[1/5] 加载日线数据...")
    df = load_daily_data_full(conn)

    # ─── Step 2: 因子计算 ─────────────────────────────────────────
    b1_config = config.get("b1_config", {})
    print("[2/5] 计算 B1 因子 (wmacd + 周线 WL>YL)...")
    df_signals = calc_b1_factors_wmacd(df, config=b1_config)

    # 确定目标日期
    if args.date:
        target_date = args.date
    else:
        latest = df_signals.select(pl.col("date").max()).collect().item()
        target_date = latest.strftime("%Y-%m-%d")

    print(f"[2/5] 目标日期: {target_date}")

    # ─── Step 3: 筛选候选 ─────────────────────────────────────────
    selection_config = config.get("selection", {})
    candidates = select_candidates(df_signals, target_date, selection_config)

    n = len(candidates)
    print(f"[3/5] 当日 B1 信号: {n} 只")

    if n == 0:
        print("没有找到 B1 信号, 退出。")
        conn.close()
        return

    for row in candidates.iter_rows(named=True):
        print(f"  - {row['code']}  J={row.get('J', '?'):.1f}  rw_dif_pct={row.get('rw_dif_pct', '?'):.2f}")

    # 收集候选股票的完整历史 (用于绘图)
    candidate_codes = candidates["code"].to_list()
    df_collected = (
        df_signals
        .filter(pl.col("code").is_in(candidate_codes))
        .collect()
    )

    # ─── Step 4: 图表导出 + 结构化上下文 ────────────────────────────
    chart_dir = Path(config["data"]["chart_output"]) / target_date
    bars = selection_config.get("bars", 90)

    charts: dict[str, Path] = {}
    contexts: dict[str, str] = {}

    print(f"[4/5] 导出 K 线图 -> {chart_dir}")
    for row in candidates.iter_rows(named=True):
        code = row["code"]
        try:
            chart_path = export_chart(df_collected, code, target_date, chart_dir, bars=bars)
            charts[code] = chart_path
            contexts[code] = build_context(row)
            print(f"  [OK] {code}")
        except Exception as e:
            print(f"  [ERR] {code} — {e}")

    if args.skip_review:
        print(f"\n图表导出完成 ({len(charts)} 只), 跳过 AI 评审 (--skip-review)")
        conn.close()
        return

    # ─── Step 5: AI 评审 ─────────────────────────────────────────
    reviewer_config = config.get("reviewer", {})
    reviewer_config["target_date"] = target_date
    reviewer_config["review_output"] = config["data"]["review_output"]

    prompt_path = ROOT / "agent" / "prompt.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    provider = reviewer_config.get("provider", "gemini")
    print(f"[5/5] AI 评审 ({provider}, model={reviewer_config.get('model', '?')})...")

    if provider == "gemini":
        from agent.reviewers.gemini import GeminiReviewer
        reviewer = GeminiReviewer(reviewer_config, prompt)
    else:
        raise ValueError(f"未知的 reviewer provider: {provider}")

    candidates_list = [
        {"code": row["code"]}
        for row in candidates.iter_rows(named=True)
        if row["code"] in charts
    ]

    results = reviewer.run(candidates_list, charts, contexts)

    # ─── 输出 ────────────────────────────────────────────────────
    print_report(results)
    save_results(results, config["data"]["review_output"], target_date)

    conn.close()
    print("\nAgent 完成。")


if __name__ == "__main__":
    main()
