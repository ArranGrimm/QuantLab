# Progress

## 2026-02-24

- 已定位 `cross_section_rotation` 的 Step 2 panic 根因。
- 触发点不是 `utils/rotation_factors.py` 的单个因子，而是 notebook 中 `~pl.col("code").is_in(st_blacklist)` 与后续复杂 lazy plan 的组合。
- 已将 ST 过滤方案改为 `anti-join`，并在 `collect()` 前仅保留研究所需列，减少中间辅助列参与最终执行计划。
