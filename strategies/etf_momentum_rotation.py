"""ETF 行业轮动策略 — 活跃ETF动量排名 + AMV择时。
纯 Python 原型：TDX qfq 价格做动量 → bfq raw 价格做成交 → 信号切换时换仓。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
from loguru import logger

from utils.data_source import DEFAULT_TDX_DB

# ── 全局配置 ──────────────────────────────────────────────────────────
TDX_DB = DEFAULT_TDX_DB
LOOKBACK = 25
ANNUAL_DAYS = 250
SCORE_MIN, SCORE_MAX = 0.0, 5.0  # 安全区间
START_DATE = "2019-01-01"
END_DATE = "2026-06-03"

ETF_POOL: list[str] = [
    "sh510050",  # 上证50
    "sh510210",  # 上证指数
    "sh510300",  # 沪深300
    "sh512000",  # 券商
    "sh512010",  # 医药
    "sh512070",  # 证券保险
    "sh512100",  # 中证1000
    "sh512170",  # 医疗
    "sh512400",  # 有色金属
    "sh512480",  # 半导体
    "sh512690",  # 酒
    "sh512760",  # 芯片
    "sh512800",  # 银行
    "sh512880",  # 证券
    "sh512890",  # 红利低波
    "sh513050",  # 中概互联
    "sh513100",  # 纳指
    "sh513880",  # 日经

    "sz159915",  # 创业板
    # "sh588000",  # 科创50 (暂注释)
    "sz159941",  # 纳指(dup)
    "sz159949",  # 创业板50
    "sz159967",  # 创业板成长
]

OUTPUT_DIR = Path("artifacts/etf-momentum-rotation")

# ── 1. 数据加载 ──────────────────────────────────────────────────────

def load_etf_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    etf_str = "', '".join(ETF_POOL)
    conn = duckdb.connect(str(TDX_DB), read_only=True)
    try:
        qfq = conn.execute(f"""
            SELECT symbol, date, close FROM v_etf_qfq
            WHERE symbol IN ('{etf_str}')
            AND date >= '{START_DATE}' AND date <= '{END_DATE}'
            AND close > 0 ORDER BY symbol, date
        """).pl()
        bfq = conn.execute(f"""
            SELECT symbol, date, open, close FROM v_etf_bfq
            WHERE symbol IN ('{etf_str}')
            AND date >= '{START_DATE}' AND date <= '{END_DATE}'
            AND open > 0 AND close > 0 ORDER BY symbol, date
        """).pl()
    finally:
        conn.close()
    logger.info(f"ETF qfq: {qfq.height:,} rows, {qfq['symbol'].n_unique()} symbols")
    logger.info(f"ETF bfq: {bfq.height:,} rows, {bfq['symbol'].n_unique()} symbols")
    return qfq, bfq


# ── 2. 动量计算 (numba OLS) ──────────────────────────────────────────

def compute_momentum(qfq: pl.DataFrame) -> pl.DataFrame:
    import numba

    @numba.njit
    def _momentum_1d(y: np.ndarray) -> float:
        n = len(y)
        if n < 2 or np.any(np.isnan(y)):
            return np.nan
        log_y = np.log(y)
        x = np.arange(n, dtype=np.float64)
        weights = np.linspace(1.0, 2.0, n)
        w_sum = np.sum(weights)
        w_x = np.sum(weights * x)
        w_y = np.sum(weights * log_y)
        w_xx = np.sum(weights * x * x)
        w_xy = np.sum(weights * x * log_y)
        denom = w_sum * w_xx - w_x * w_x
        if denom == 0.0:
            return np.nan
        slope = (w_sum * w_xy - w_x * w_y) / denom
        intercept = (w_y - slope * w_x) / w_sum
        y_pred = slope * x + intercept
        residuals = weights * (log_y - y_pred) ** 2
        weighted_mean_y = w_y / w_sum
        ss_tot = np.sum(weights * (log_y - weighted_mean_y) ** 2)
        r2 = 1.0 - np.sum(residuals) / ss_tot if ss_tot > 0 else 0.0
        if r2 < 0.0:
            r2 = 0.0
        daily_factor = np.exp(slope)
        annualized = daily_factor ** ANNUAL_DAYS - 1.0
        score = annualized * r2
        if not np.isfinite(score):
            return np.nan
        return score

    @numba.njit
    def _momentum_matrix(n_t: int, n_m: int, close_mat: np.ndarray) -> np.ndarray:
        scores = np.full((n_t, n_m), np.nan)
        for j in range(n_m):
            for i in range(LOOKBACK - 1, n_t):
                scores[i, j] = _momentum_1d(close_mat[i - LOOKBACK + 1 : i + 1, j])
        return scores

    pivot = qfq.pivot(values="close", index="date", columns="symbol").sort("date")
    dates = pivot["date"].to_list()
    symbols = [c for c in pivot.columns if c != "date"]
    close_mat = pivot.select(symbols).to_numpy()

    scores_mat = _momentum_matrix(len(dates), len(symbols), close_mat)

    rows = []
    for j, sym in enumerate(symbols):
        for i, d in enumerate(dates):
            if i >= LOOKBACK - 1 and not np.isnan(scores_mat[i, j]):
                rows.append({"date": d, "symbol": sym, "momentum": float(scores_mat[i, j])})

    logger.info(f"Computed {len(rows):,} momentum scores for {len(symbols)} ETFs")
    return pl.DataFrame(rows).sort(["date", "symbol"])


# ── 3. 信号生成 ──────────────────────────────────────────────────────

def generate_signals(momentum: pl.DataFrame) -> pl.DataFrame:
    from strategies.amv.regime import build_amv_phase_frame

    amv = build_amv_phase_frame().with_columns(pl.col("date").cast(pl.Utf8))
    amv_bull = amv.filter(pl.col("is_bull_regime")).select(pl.col("date").alias("sig_date"))

    signals = (
        momentum.with_columns(pl.col("date").cast(pl.Utf8))
        .filter((pl.col("momentum") > SCORE_MIN) & (pl.col("momentum") <= SCORE_MAX))
        .sort(["date", "momentum"], descending=[False, True])
        .group_by("date")
        .first()
        .sort("date")
    )

    signals = signals.with_columns(pl.col("symbol").shift(1).alias("_prev"))
    signals = signals.with_columns(
        (pl.col("symbol") != pl.col("_prev")).fill_null(True).alias("changed")
    )

    switch = (
        signals.filter(pl.col("changed"))
        .join(amv_bull, left_on="date", right_on="sig_date", how="inner")
        .select(pl.col("date").alias("signal_date"), "symbol", "momentum")
        .sort("signal_date")
    )

    logger.info(f"{switch.height} switch signals on {amv_bull.height} bull days")
    return switch


# ── 4. 回测模拟 ──────────────────────────────────────────────────────

def simulate_trades(
    signals: pl.DataFrame, bfq: pl.DataFrame, initial_cap: float = 500_000.0, min_hold_days: int = 5
) -> dict[str, Any]:
    px: dict[tuple[str, str], tuple[float, float]] = {}
    for r in bfq.iter_rows():
        px[(r[0], str(r[1]))] = (float(r[2]), float(r[3]))

    from strategies.amv.regime import build_amv_phase_frame
    amv = build_amv_phase_frame().with_columns(pl.col("date").cast(pl.Utf8))
    bear_dates = set(amv.filter(~pl.col("is_bull_regime"))["date"].to_list())

    all_dates = sorted(set(str(d) for d in bfq["date"].to_list()))
    sig_list = signals.sort("signal_date").to_dicts()

    trades: list[dict] = []
    equity: list[dict] = []
    cash = initial_cap
    pos: dict | None = None
    last_entry_idx = -999
    si = 0

    for di, cd in enumerate(all_dates):
        is_bear = cd in bear_dates

        if is_bear and pos is not None:
            ep = px.get((pos["symbol"], cd), (0.0, 0.0))[1]
            if ep > 0:
                gross = pos["shares"] * ep
                net = gross - gross * 0.00025 - gross * 0.001 - gross * 0.001
                pnl = net - pos["cost"]
                trades.append({
                    "code": pos["symbol"], "entry_date": pos["entry_date"], "exit_date": cd,
                    "entry_price": pos["entry_price"], "exit_price": ep,
                    "shares": pos["shares"], "cost": pos["cost"], "exit_value": net,
                    "pnl": pnl, "pnl_pct": pnl / pos["cost"],
                    "hold_trading_days": di - all_dates.index(pos["entry_date"]),
                    "exit_reason": "BearRegime",
                })
                cash += net
            pos = None

        if si < len(sig_list) and sig_list[si]["signal_date"] == cd and not is_bear:
            if pos is not None and (di - last_entry_idx) < min_hold_days:
                si += 1
            else:
                new_sym = sig_list[si]["symbol"]
                if pos is not None:
                    ep = px.get((pos["symbol"], cd), (0.0, 0.0))[1]
                    if ep > 0:
                        gross = pos["shares"] * ep
                        net = gross - gross * 0.00025 - gross * 0.001 - gross * 0.001
                        pnl = net - pos["cost"]
                        trades.append({
                            "code": pos["symbol"], "entry_date": pos["entry_date"], "exit_date": cd,
                            "entry_price": pos["entry_price"], "exit_price": ep,
                            "shares": pos["shares"], "cost": pos["cost"], "exit_value": net,
                            "pnl": pnl, "pnl_pct": pnl / pos["cost"],
                            "hold_trading_days": di - all_dates.index(pos["entry_date"]),
                            "exit_reason": "SignalSwitch",
                        })
                        cash += net
                    pos = None
                ep = px.get((new_sym, cd), (0.0, 0.0))[0]
                if ep > 0:
                    eff = ep * (1 + 0.00025 + 0.001)
                    shares = int(cash / eff / 100) * 100
                    if shares > 0:
                        cost = shares * eff
                        cash -= cost
                        pos = {"symbol": new_sym, "entry_date": cd, "entry_price": ep,
                               "shares": shares, "cost": cost}
                        last_entry_idx = di
                si += 1

        pv = 0.0
        if pos is not None:
            ep = px.get((pos["symbol"], cd), (0.0, 0.0))[1]
            if ep > 0:
                pv = pos["shares"] * ep
        equity.append({"date": cd, "cash": cash, "positions_value": pv, "total_value": cash + pv})

    if pos is not None:
        ld = all_dates[-1]
        ep = px.get((pos["symbol"], ld), (0.0, 0.0))[1]
        if ep > 0:
            gross = pos["shares"] * ep
            net = gross - gross * 0.00025 - gross * 0.001 - gross * 0.001
            pnl = net - pos["cost"]
            trades.append({
                "code": pos["symbol"], "entry_date": pos["entry_date"], "exit_date": ld,
                "entry_price": pos["entry_price"], "exit_price": ep,
                "shares": pos["shares"], "cost": pos["cost"], "exit_value": net,
                "pnl": pnl, "pnl_pct": pnl / pos["cost"],
                "hold_trading_days": len(all_dates) - 1 - all_dates.index(pos["entry_date"]),
                "exit_reason": "EndOfData",
            })

    trades_df = pl.DataFrame(trades).sort("entry_date") if trades else pl.DataFrame()
    eq_df = pl.DataFrame(equity).sort("date")

    final_val = eq_df["total_value"][-1]
    total_ret = (final_val / initial_cap - 1) * 100
    eq_vals = eq_df["total_value"].to_numpy().astype(float)
    peak = np.maximum.accumulate(eq_vals)
    max_dd = float(np.min((eq_vals - peak) / peak * 100))
    daily_r = np.diff(eq_vals) / eq_vals[:-1]
    sharpe = float(np.mean(daily_r) / np.std(daily_r) * np.sqrt(252)) if np.std(daily_r) > 0 else 0

    yearly = {}
    for y in range(2019, 2027):
        ye = eq_df.filter(pl.col("date").str.starts_with(str(y)))
        if ye.height < 2:
            continue
        yearly[str(y)] = round((ye["total_value"][-1] / ye["total_value"][0] - 1) * 100, 2)

    return {
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "total_trades": trades_df.height,
        "win_rate_pct": round(trades_df.filter(pl.col("pnl") > 0).height / trades_df.height * 100, 1) if trades_df.height > 0 else 0,
        "yearly": yearly,
        "final_value": round(float(final_val), 2),
        "trades": trades_df,
        "equity": eq_df,
    }


# ── 5. 主流程 ──────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("1/4 Load ETF data ...")
    qfq, bfq = load_etf_data()

    logger.info("2/4 Compute momentum ...")
    mom = compute_momentum(qfq)

    logger.info("3/4 Generate signals ...")
    sig = generate_signals(mom)

    logger.info("4/4 Simulate ...")
    r = simulate_trades(sig, bfq)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = OUTPUT_DIR / ts
    rd.mkdir(parents=True, exist_ok=True)
    r["trades"].write_csv(rd / "trades.csv")
    r["equity"].write_csv(rd / "daily_equity.csv")
    report = {
        "strategy": "etf-momentum-rotation",
        "ran_at": ts,
        "total_return_pct": r["total_return_pct"],
        "max_drawdown_pct": r["max_drawdown_pct"],
        "sharpe": r["sharpe"],
        "total_trades": r["total_trades"],
        "win_rate_pct": r["win_rate_pct"],
        "yearly": r["yearly"],
    }
    (rd / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*50}")
    print("ETF Momentum Rotation — Results")
    print(f"{'='*50}")
    print(f"Total Return: {r['total_return_pct']:+.2f}%  MaxDD: {r['max_drawdown_pct']:.2f}%  Sharpe: {r['sharpe']:.2f}")
    print(f"Trades: {r['total_trades']}  WR: {r['win_rate_pct']:.1f}%")
    for y, yr in r["yearly"].items():
        print(f"  {y}: {yr:>+8.2f}%")
    print(f"\nSaved to {rd}")


if __name__ == "__main__":
    main()
