---
name: quantlab-qlab-workflow
description: Guides QuantLab daily AMV operations through the canonical qlab CLI. Use when exporting AMV signals, running bt-amv-topn backtests, comparing strategies, checking current status, reproducing raw-execution baselines, or when the user mentions qlab, ref, p3, context, pb3-gated, limit-weakgate, presets, or AMV artifacts.
---

# QuantLab QLab Workflow

Use this skill when the task is to operate the current AMV workflow, not to invent a new experiment.

## First Checks

- Read `CURRENT_STATE.md` for current baseline, challenger, sleeves, and daily commands.
- Read `AGENTS.md` if changing workflow code or docs.
- Prefer `scripts/qlab.py`; do not resurrect deleted one-off scripts.

## Canonical Commands

```bash
uv run python scripts/qlab.py status
uv run python scripts/qlab.py export p3
uv run python scripts/qlab.py export context
uv run python scripts/qlab.py export pb3-gated
uv run python scripts/qlab.py export limit-weakgate
uv run python scripts/qlab.py backtest artifacts/amv_static_sleeve_signals/<signal_id> --preset 6td-static
uv run python scripts/qlab.py compare p3 context
uv run python scripts/qlab.py attribution p3-raw-vs-adjusted
```

Stable export targets: `ref`, `p3`, `context`, `pb3-gated`, `limit-weakgate`.

Stable presets: `6td-static`, `5td-static`, `3td-static`, `6td-rolling`.

## Source Of Truth

- Tradable results come from Rust `bt-amv-topn` artifacts.
- `qlab status` reads stable summaries from `strategies/amv/status.py`.
- `reports/*.json` has been intentionally removed; do not depend on it.
- Use `artifacts/` for run outputs and large/intermediate files.

## When Editing

- Keep CLI orchestration thin in `scripts/qlab.py`.
- Put reusable strategy logic in `strategies/amv/`.
- Put generic data/price helpers in `utils/`.
- After changing workflow code, validate with focused `py_compile`, `qlab --dry-run` commands, and lints for edited files.
