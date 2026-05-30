---
name: quantlab-cleanup-maintenance
description: Applies QuantLab cleanup rules for deleting, migrating, or documenting obsolete scripts, reports, canvases, old strategy lines, agent files, and generated artifacts. Use when the user asks to clean directories, remove legacy B1/B3/rotation/agent code, prune reports or canvases, update script inventory, or continue project productization cleanup.
---

# QuantLab Cleanup Maintenance

Use this skill when reducing project surface area or deciding whether files should remain.

## Cleanup Principles

- Judge by future functional value, not import dependency alone.
- Keep only current workflow entrypoints; `scripts/` should stay centered on `scripts/qlab.py`.
- Do not preserve old strategy lines as runnable code unless the user explicitly wants them.
- Do not delete user changes outside the requested cleanup scope.
- Do not recreate `reports/*.json`; stable summaries belong in code/docs, and core visual summaries live in `reports/canvases/`.

## Before Deleting

- Inventory candidate files with `Glob`.
- Search references with `rg`.
- If a file still provides current behavior, migrate the reusable part into `strategies/amv/` or `utils/` before deleting the old entrypoint.
- Keep `.claude/skills/` project skills when they serve current AMV work; old `.agents/` references are obsolete.

## After Deleting

- Remove empty directories and cache leftovers such as `__pycache__`.
- Update only relevant docs:
  - `CURRENT_STATE.md` for current status, stable decisions, and active risks.
  - `AGENTS.md` for cleanup policy, document ownership, and workflow rules.
  - `strategies/archive-index.md` when a strategy route is archived.
- Validate with targeted `Glob`/`rg`, focused lints, and the smallest relevant `qlab` command.

## User Preference

During iterative cleanup, do not show `git status` or `git diff` after every small change unless the user asks for commit/PR work or the final state is unclear.
