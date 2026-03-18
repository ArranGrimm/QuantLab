# Project Status

## 当前状态

- 截面轮动 Phase 1 已完成基础实现：
- `utils/rotation_factors.py`：31 个日线截面因子
- `notebooks/cross_section_rotation.py`：Universe、因子、IC、Top-N 回测、可视化

## 最新修复

- 修复 `Step 2 Collecting` 阶段的 Polars panic。
- 原因：`is_in(st_blacklist)` 在当前环境下会触发 Polars lazy 优化器异常。
- 方案：改用 ST 黑名单 `anti-join`，并在 `collect()` 前显式 `select` 最终输出列。

## 待验证

- 用户重新运行 notebook，确认 Step 2 能正常完成并进入 IC 分析与回测阶段。
