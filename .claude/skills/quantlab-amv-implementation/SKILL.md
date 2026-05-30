---
name: quantlab-amv-implementation
description: Guides AMV strategy implementation changes in QuantLab. Use when editing strategies/amv, qlab native exports, AMV rules, factors, registry entries, raw-execution fields, PB3 gates, context combo, limit weakgate, or when converting old scripts into reusable AMV modules.
---

# QuantLab AMV Implementation

Use this skill for code changes to the current AMV implementation layer.

## Layering Rules

- `scripts/qlab.py`: CLI parsing, command orchestration, user-facing output only.
- `strategies/amv/registry.py`: stable targets, presets, aliases, and metadata.
- `strategies/amv/workflows.py`: native export workflow orchestration.
- `strategies/amv/rules/`: strategy-specific gates and rerank rules.
- `strategies/amv/factors/`: reusable AMV factor construction.
- `strategies/amv/attribution.py`: reusable backtest attribution.
- `strategies/amv/status.py`: stable status summaries used by `qlab status`.
- `utils/`: generic data, price, filesystem, ST, industry, and Polars helpers.

## Execution Invariants

- Tradable results must be validated through Rust `bt-amv-topn`.
- Use raw OHLC / raw pre-close for execution, capital, shares, fees, and limit checks.
- Adjusted prices may still be used for factor/ranker calculations.
- Preserve A-share lot rules and `allow_duplicate_positions = false` default semantics.
- Do not treat Python label or close-to-close results as final tradable evidence.

## Editing Workflow

1. Read the nearest existing module before editing.
2. Prefer extending existing registries/rules over adding a new script.
3. Keep behavior changes scoped to the named target or rule.
4. Validate with `py_compile`, targeted lints, `qlab export <target> --dry-run`, and a focused backtest only when needed.
5. Update docs only when the change affects current commands, stable conclusions, or cleanup policy.
