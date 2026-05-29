---
name: amv-trade-attribution
description: Standardizes AMV backtest comparison, trade attribution, signal swap explanation, cost drag, and sleeve complementarity analysis in QuantLab. Use when analyzing bt-amv-topn artifacts, AMV strategy attribution, P3 vs Ref, pullback sleeves, yearly/monthly PnL, overlap trades, or signal replacement mechanisms.
---

# AMV Trade Attribution

Use this skill for QuantLab AMV strategy diagnostics that compare backtest artifacts, explain trade PnL differences, or analyze sleeve complementarity.

## Core Rules

- Run Python with `uv run python ...`; do not rely on shell scripts for reusable analysis.
- Treat `bt-amv-topn` Rust outputs as the execution source of truth for tradable results.
- Keep generated reusable logic in repo `scripts/`, not inside this skill. This skill defines workflow and analysis standards.
- Write outputs under `reports/` when the result is durable; use `artifacts/` for large or intermediate run outputs.
- Update `progress.md` for new experiments. Update `project-status.md` only when a stable decision or current priority changes.
- If a visualization is useful, create or update a tracked canvas under `reports/canvases/`.

## Required Attribution Checklist

For strategy A vs B, inspect:

- Summary metrics: total return, MaxDD, win rate, trade count, costs, blocked limit-up count if available.
- Yearly and monthly deltas from equity and from realized trade PnL.
- Trade overlap by exact `(entry_date, code)` and code-level overlap.
- Unique winners and losers on both sides, sorted by PnL and return.
- Common-trade PnL deltas if both strategies hold the same code/entry.
- Cost drag when turnover or rolling/refill behavior differs.
- Daily return correlation when comparing sleeves for complementarity.

## Signal Swap Explanation

When explaining why one AMV ranker beats another:

- Do not stop at after-the-fact PnL.
- Go back to the signal date that produced the trade entry.
- Compare TopN ranks before and after the weight/ranker change.
- Decompose the score into the relevant components, for example P-block, K-block, momentum, pullback, CP, or RV.
- Identify whether the winning replacement is a repeatable mechanism or a concentrated lucky trade.
- Record both success and failure samples.

## AMV Naming

- `Ref` usually means `reference_p2_k0p5_b0_c0_r0`.
- `P3` usually means `candidate_p3_k0p5_b0_c0_r0`.
- Pullback naming uses `PB/CP/RV`:
  - `PB`: `ma_bias_20 + disp_bias_20`
  - `CP`: `KSFT + intraday_pos`
  - `RV`: `atr_14_pct + panic_vol_ratio_20d`

## Preferred Script Entrypoints

If these scripts exist, prefer them before writing one-off code:

- `scripts/backtest_trade_attribution.py`
- `scripts/amv_explain_signal_swaps.py`
- `scripts/amv_strategy_correlation.py`

If they do not exist yet, create reusable scripts under `scripts/` when the same analysis pattern is likely to recur.

## Output Summary

When reporting to the user, lead with the mechanism and decision impact, then include only the most important numbers. Mention any generated JSON, canvas, or document updates.
