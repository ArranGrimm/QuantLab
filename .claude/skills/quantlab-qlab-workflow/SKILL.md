---
name: quantlab-qlab-workflow
description: Guides QuantLab daily AMV operations through the canonical qlab CLI. Use when exporting AMV signals, running bt-amv-topn backtests, comparing strategies, checking current status, reproducing raw-execution baselines, or when the user mentions qlab, AMV strategies, presets, or AMV artifacts.
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
uv run python scripts/qlab.py export trend-p3
uv run python scripts/qlab.py export trend-p3-enhanced
uv run python scripts/qlab.py export pullback-pb3
uv run python scripts/qlab.py export event-firstboard
uv run python scripts/qlab.py backtest trend-p3
uv run python scripts/qlab.py backtest trend-p3 --top-n 5
uv run python scripts/qlab.py results trend-p3
uv run python scripts/qlab.py results trend-p3 --diff
uv run python scripts/qlab.py run trend-p3
```

Stable export targets: `trend-p2`, `trend-p3`, `trend-p3-enhanced`, `pullback-pb3`, `event-firstboard`, `event-firstboard-base`.

## Naming Convention

- Strategies: `family-variant`. See `strategies/amv/configs/*.json`.
- A Strategy = Ranker template + params + list of Rules + Preset.

## Source Of Truth

- Tradable results come from Rust `bt-amv-topn` artifacts.
- `qlab status` scans `artifacts/*/backtests/*/result.json` for latest canonical results.
- `qlab results` lists all historical backtest results for a strategy.
- Strategy definitions in `strategies/amv/configs/*.json`.

## When Editing

- Keep CLI orchestration thin in `scripts/qlab.py`.
- Put reusable strategy logic in `strategies/amv/`.
- Put generic data/price helpers in `utils/`.
- New strategy: add a JSON file in `strategies/amv/configs/`.
- New rule: implement in `pipeline.py` / `pipeline_event.py`, register in `specs.py::RULES`.
- `end_date` auto-detects from DuckDB; set explicitly only for backtesting historical ranges.
