"""What-if simulation: early stop-loss + extended holding for P3 static trades.

Simulates two exit rule enhancements on existing Rust backtest trades:
  1. Early stop: exit at d2 (3rd bar) if cum_ret < -3%
  2. Extended hold: if d6 cum_ret > 10% and near high, extend 3 more days with trailing stop

Uses actual daily close/high data to trace the price path for each trade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
DB = Path("D:/WorkSpace/Tinkering/QuantData/Ashare/qmt_data.duckdb")

P3_TRADES = (
    ROOT / "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0"
    "/backtests/6td_static_strict_top3_no_stop_20260520_092208_801/trades.csv"
)


def load_data() -> tuple[pl.DataFrame, dict]:
    trades = pl.read_csv(
        str(P3_TRADES),
        schema_overrides={
            "pnl": pl.Float64,
            "pnl_pct": pl.Float64,
            "entry_date": pl.Utf8,
            "exit_date": pl.Utf8,
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
            "code": pl.Utf8,
            "hold_trading_days": pl.Int64,
        },
    ).with_columns([
        pl.col("entry_date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("exit_date").str.strptime(pl.Date, "%Y-%m-%d"),
    ]).with_row_index("id")

    all_codes = trades["code"].unique().to_list()
    min_d = trades["entry_date"].min()
    max_d = trades["exit_date"].max()

    conn = duckdb.connect(str(DB), read_only=True)
    code_str = "', '".join(all_codes)
    daily = conn.execute(
        f"SELECT code, date, close, high FROM v_stock_daily_qfq_qmt "
        f"WHERE code IN ('{code_str}') AND date >= '{min_d}' AND date <= '{max_d}' "
        f"ORDER BY code, date"
    ).pl()
    conn.close()

    # Build per-code sorted price lists
    price_data: dict[str, list[tuple]] = {}
    for row in daily.iter_rows():
        code, d, c, h = row
        if c is None or h is None:
            continue
        price_data.setdefault(code, []).append((str(d), float(c), float(h)))

    return trades, price_data


def simulate(trades: pl.DataFrame, price_data: dict) -> pl.DataFrame:
    results = []

    for row in trades.iter_rows(named=True):
        tid = row["id"]
        code = row["code"]
        entry_d = row["entry_date"]
        exit_d = row["exit_date"]
        entry_px = row["entry_price"]
        final_pnl = row["pnl"]
        final_pnl_pct = row["pnl_pct"]

        bars = price_data.get(code, [])
        if not bars:
            continue

        # Collect bars from entry to exit+5 (for extended hold simulation)
        path = []
        for date_str, close_px, high_px in bars:
            if date_str < str(entry_d):
                continue
            if date_str > str(exit_d) and len([p for p in path if p["phase"] == "post"]) >= 5:
                break
            cum_ret = (close_px / entry_px - 1) * 100
            phase = "hold" if date_str <= str(exit_d) else "post"
            path.append({
                "date": date_str, "close": close_px, "high": high_px,
                "cum_ret": cum_ret, "phase": phase,
            })

        # (1) Early stop: d2 (index 2) cum_ret < -3%
        stop_triggered = False
        stop_day = None
        stop_pnl_pct = None
        for i, p in enumerate(path):
            if p["phase"] == "hold" and i >= 2 and p["cum_ret"] < -3.0:
                stop_triggered = True
                stop_day = i
                stop_pnl_pct = p["cum_ret"]
                break

        # (2) Extended hold: d6 cum_ret > 10% and near high, extend with trailing -5%
        ext_triggered = False
        ext_extra_pct = 0.0
        hold_bars = [p for p in path if p["phase"] == "hold"]
        if len(hold_bars) >= 6:
            d5 = hold_bars[4]
            d6 = hold_bars[5]
            if d6["cum_ret"] > 10:
                # near high: d5 and d6 close >= respective high * 0.98
                near_high = (
                    d5["close"] >= d5["high"] * 0.98
                    and d6["close"] >= d6["high"] * 0.98
                )
                if near_high:
                    ext_triggered = True
                    ext_high = d6["high"]
                    post_bars = [p for p in path if p["phase"] == "post"]
                    for p in post_bars:
                        if p["high"] > ext_high:
                            ext_high = p["high"]
                        if p["close"] <= ext_high * 0.95:  # trailing stop hit
                            ext_extra_pct = p["cum_ret"] - d6["cum_ret"]
                            break
                    else:
                        if post_bars:
                            ext_extra_pct = post_bars[-1]["cum_ret"] - d6["cum_ret"]

        results.append({
            "id": tid,
            "code": code,
            "entry_date": str(entry_d),
            "final_pnl": final_pnl,
            "final_pnl_pct": final_pnl_pct,
            "stop_triggered": stop_triggered,
            "stop_day": stop_day,
            "stop_pnl_pct": stop_pnl_pct,
            "ext_triggered": ext_triggered,
            "ext_extra_pct": ext_extra_pct,
            "d0_ret": path[0]["cum_ret"] if path else None,
            "d3_ret": hold_bars[3]["cum_ret"] if len(hold_bars) > 3 else None,
            "d6_ret": hold_bars[5]["cum_ret"] if len(hold_bars) > 5 else None,
        })

    return pl.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 exit logic what-if")
    parser.add_argument("--dont-record-progress", action="store_true")
    args = parser.parse_args()

    logger.info("Loading trades and daily data ...")
    trades, price_data = load_data()

    logger.info(f"Simulating {trades.height} trades ...")
    sim = simulate(trades, price_data)

    total_pnl = float(sim["final_pnl"].sum())
    total_n = sim.height

    # ═══ (1) Early stop ═══════════════════════════════════════════════
    print("=" * 60)
    print("(1) EARLY STOP: exit at d2 (3rd bar) if cum_ret < -3%")
    print("=" * 60)
    stopped = sim.filter(pl.col("stop_triggered"))
    print(f"  Trades affected: {stopped.height}/{total_n}")
    print(f"  Original PnL:    {float(stopped['final_pnl'].sum()):>+10,.0f}")
    # Approximate PnL if stopped: use final_pnl / final_pnl_pct as rough notional
    stop_savings = 0.0
    for row in stopped.iter_rows(named=True):
        if row["final_pnl_pct"] != 0 and row["stop_pnl_pct"] is not None:
            notional = row["final_pnl"] / row["final_pnl_pct"]
            stop_pnl = notional * row["stop_pnl_pct"] / 100
            stop_savings += (row["final_pnl"] - stop_pnl)
    print(f"  Approx savings:  {stop_savings:>+10,.0f}")
    print(f"  Without stop they'd go to: {float(stopped['final_pnl_pct'].mean())*100:.1f}% avg")
    print(f"  Stopped at ~:              {float(stopped['stop_pnl_pct'].mean()):.1f}% avg")
    print(f"\n  All stopped trades:")
    for row in stopped.sort("final_pnl").iter_rows(named=True):
        print(
            f"  {row['code']:<12} {row['entry_date']} "
            f"d0={row['d0_ret']:>+5.1f}% d3={str(row['d3_ret']):>6} "
            f"final={row['final_pnl_pct']:>+6.1%} stop~{row['stop_pnl_pct']:>+5.1f}%"
        )

    # ═══ (2) Extended hold ═════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("(2) EXTENDED HOLD: d6 cum_ret > 10% & near high → extend 3d trailing -5%")
    print("=" * 60)
    ext = sim.filter(pl.col("ext_triggered"))
    ext_extra = 0.0
    for row in ext.iter_rows(named=True):
        if row["final_pnl_pct"] != 0:
            notional = row["final_pnl"] / row["final_pnl_pct"]
            ext_extra += notional * row["ext_extra_pct"] / 100
    print(f"  Trades extended: {ext.height}/{total_n}")
    print(f"  d6 avg cum_ret:  {float(ext['d6_ret'].mean()):.1f}%")
    print(f"  Extra avg pct:   {float(ext['ext_extra_pct'].mean()):.1f}%")
    print(f"  Approx extra PnL:{ext_extra:>+10,.0f}")
    worse = ext.filter(pl.col("ext_extra_pct") < 0)
    if worse.height > 0:
        print(f"  Would lose more: {worse.height}t")
        for row in worse.iter_rows(named=True):
            print(f"    {row['code']} {row['entry_date']}: extra={row['ext_extra_pct']:+.1f}%")
    print(f"\n  All extended trades:")
    for row in ext.sort("ext_extra_pct", descending=True).iter_rows(named=True):
        print(
            f"  {row['code']:<12} {row['entry_date']} "
            f"d6={row['d6_ret']:>+5.1f}% extra={row['ext_extra_pct']:>+5.1f}% "
            f"final={row['final_pnl_pct']:>+6.1f}"
        )

    # ═══ (3) Combined ══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("(3) COMBINED")
    print("=" * 60)
    combined_delta = stop_savings + ext_extra
    print(f"  Stop savings:     {stop_savings:>+10,.0f}")
    print(f"  Extend extra:     {ext_extra:>+10,.0f}")
    print(f"  Combined delta:   {combined_delta:>+10,.0f}")
    print(f"  Current total:    {total_pnl:>10,.0f}")
    print(f"  New total (est):  {total_pnl + combined_delta:>10,.0f}")

    # ═══ progress.md ═══════════════════════════════════════════════════
    if not args.dont_record_progress:
        from datetime import date
        entry = (
            f"\n## {date.today().isoformat()}\n\n"
            f"### [AMV] P3 exit logic what-if\n\n"
            f"- 脚本: `scripts/amv_exit_logic_whatif.py`\n"
            f"- 目标: 模拟早期止损 + 延长持有对 P3 的改善效果\n"
            f"- 止损规则: d2 时 cum_ret < -3% → 提前卖出\n"
            f"- 延长规则: d6 时 cum_ret > 10% 且贴近高点 → 延长 3d trailing -5%\n"
            f"- 止损近似节省: {stop_savings:+,.0f}\n"
            f"- 延长近似增收: {ext_extra:+,.0f}\n"
            f"- 合并近似影响: {combined_delta:+,.0f}\n"
        )
        with open(ROOT / "progress.md", "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Appended to progress.md")


if __name__ == "__main__":
    main()
