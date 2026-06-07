---
name: amv-trade-attribution
description: Standardizes AMV bt-amv-topn comparison, trade attribution, signal replacement explanation, cost drag, and sleeve complementarity analysis in QuantLab. Use when analyzing AMV backtest artifacts, P3 vs Ref/context/PB3, raw-vs-adjusted differences, yearly/monthly PnL, trade overlap, unique winners/losers, or sleeve correlation.
---

# AMV Trade Attribution

Use this skill for QuantLab AMV strategy diagnostics that compare backtest artifacts, explain trade PnL differences, or analyze sleeve complementarity.

## Core Rules

- Run Python with `uv run python ...`; do not rely on shell scripts for reusable analysis.
- Treat `bt-amv-topn` Rust outputs as the execution source of truth for tradable results.
- Prefer `scripts/qlab.py` before writing one-off code.
- Keep reusable implementation under `strategies/amv/`; `scripts/` is reserved for `qlab.py`.
- Write generated attribution JSON under `artifacts/attribution/` unless the user explicitly asks for a tracked durable artifact.
- `reports/` now keeps only a small set of core canvases; do not recreate `reports/*.json`.
- Update `CURRENT_STATE.md` only when a stable decision, current priority, or active risk changes. Historical experiment flow is recovered from git history, not a live progress file.

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

Strategies use `family-variant` convention:
- `trend-p2`: trend continuation, P=2 weight (baseline)
- `trend-p3`: trend continuation, P=3 weight (challenger)
- `trend-p3-enhanced`: P3 + sector tailwind + medium trend quality penalties
- `pullback-pb3`: pullback PB3/CP1/RV0 + AMV regime gate
- `event-firstboard`: first board pullback + weak window gate

Ranker component naming uses `PB/CP/RV`:
- `PB`: `ma_bias_20 + disp_bias_20` (pullback bias)
- `CP`: `KSFT + intraday_pos` (close pullback)
- `RV`: `atr_14_pct + panic_vol_ratio_20d` (risk/volatility)

## Preferred Script Entrypoints

Prefer the canonical CLI before writing one-off code:

- `scripts/qlab.py results <strategy>` — list history
- `scripts/qlab.py results <strategy> --diff` — latest two canonical compared
- `scripts/qlab.py backtest <strategy> --top-n 5` — parameter override

Deep trade attribution: read `artifacts/<strategy>/backtests/<ts>/trades.csv` directly with Polars.

## Output Summary

When reporting to the user, lead with the mechanism and decision impact, then include only the most important numbers. Mention any generated JSON, canvas, or document updates.
