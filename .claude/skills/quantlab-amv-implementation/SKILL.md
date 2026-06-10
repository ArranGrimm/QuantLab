---
name: quantlab-amv-implementation
description: Guides AMV strategy implementation changes in QuantLab. Use when editing strategies/amv, qlab native exports, AMV rules, factors, registry entries, raw-execution fields, AMV regime gates, context combo, event weakgate, or when converting old scripts into reusable AMV modules.
---

# QuantLab AMV Implementation

Use this skill for code changes to the current AMV implementation layer.

## Layering Rules

- `scripts/qlab.py`: CLI parsing, command orchestration, user-facing output only.
- `strategies/amv/registry.py`: Strategy loader from `configs/*.json`.
- `strategies/amv/configs/`: JSON strategy definitions (single source of truth).
- `strategies/amv/data.py`: market data layer (DuckDB → Polars LazyFrame).
- `factors/registry.py`: 顶层因子注册表（strategy 和 research 共享），按需计算，按 status 分 active/experimental/dead
- `strategies/amv/hooks.py`: RuleHook 基类 + 内置 hook（MediumTrendQualityHook/AmvRegimeGateHook/EventWeakgateHook）
- `strategies/amv/pipeline.py`: unified ranker export（trend/pullback），Hook 驱动，无 if 分支，4 阶段流程
- `strategies/amv/pipeline_event.py`: event strategy export（first-board pullback + weakgate），合并双数据源为单 lazy chain
- `strategies/amv/export.py`: signal.parquet write only (no metadata files).
- `utils/`: generic data, price, filesystem, ST, industry, and Polars helpers.

## Execution Invariants

- Tradable results must be validated through Rust `bt-amv-topn`.
- `qlab status` reads `artifacts/*/backtests/*/result.json` (auto-discovery, no hardcoding).
- Signal export writes ONLY `signal.parquet` (no `selected_signals.csv`, no `signal.meta.json`).
- Use raw OHLC / raw pre-close for execution; adjusted prices for factor calculations.
- Preserve A-share lot rules and `allow_duplicate_positions = false` default semantics.
- `end_date` defaults to empty → auto-detected from DuckDB.

## Editing Workflow

1. Read the nearest existing module before editing.
2. Prefer extending existing pipeline helpers/factors over adding a new script.
3. New strategy: add JSON to `configs/`. New ranker template: add to `_rankers.json`. New rule: register in `hooks.py::_HOOK_REGISTRY`. New factor: add to `factors/registry.py::FACTOR_REGISTRY`.
4. Validate with `ruff check`, `qlab export <strategy>`, follow with `qlab backtest <strategy>`.
5. Update docs only when the change affects current commands, stable conclusions, or cleanup policy.
