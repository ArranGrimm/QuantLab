# 清理计划

这是第二阶段删除 / 迁移计划。第一阶段只建立新入口和脚本状态标签。

## 第二阶段目标

- 将日常入口收敛到 `scripts/qlab.py`。
- 把可复用 AMV 策略定义从 ad hoc script 中迁出。
- 删除或归档已被 CLI、reports、Canvas 或 git history 覆盖的脚本。
- 删除历史代码前，确保当前 raw-execution 结论已经稳定沉淀。

## 不可跳过的门禁

删除或移动脚本前，必须满足所有适用门禁：

- 已有 `scripts/qlab.py` 替代命令，或者该脚本不属于任何当前 workflow。
- 最新结论已经进入 `CURRENT_STATE.md`、稳定 `reports/*.json`、Canvas、`experiments/` 说明，或已经在既有 `project-status.md` / `progress.md` 中沉淀。
- raw execution 状态清楚；仅 adjusted-execution 的结果必须先标为 historical。
- 没有 canonical script import 这个候选脚本。
- 用 `rg "<module_or_filename>"` 快速确认没有当前依赖。
- 删除动作要和行为改动分开 review / commit。

## 整理期记录节奏

- 不要把 `progress.md` 当整理操作日志。
- 不要每次删除引用、改文档、调脚本标签都更新 `project-status.md`。
- 只有阶段性里程碑才更新：例如 CLI 稳定可用、第一批 deprecated 脚本删除完成、`strategies/amv/` 迁移完成。
- 如果只是为了避免未来遗忘，优先更新 `AGENTS.md`、`docs/script-inventory.md` 或本文件。

## 迁移目标

### 策略逻辑

迁到 `strategies/amv/`：

- canonical sleeve id 和别名
- Ref、P3、PB3、context combo、limit ecology 的 score expression
- 可复用参数 preset
- signal target 元数据

### 工具逻辑

保留或迁到 `utils/`：

- DuckDB / QMT 数据读取
- ST 和上市状态过滤
- raw / adjusted price 处理
- 行业映射底层能力
- 通用 Polars helper

### 命令逻辑

保留在 `scripts/`：

- `qlab.py`
- 产出稳定 artifact 的薄 wrapper
- 仍会反复回看的诊断 runner

## 第一批删除候选

这些脚本已在 `docs/script-inventory.md` 标为 `deprecated`：

- `scripts/b1_backtest.py`
- `scripts/rotation_backtest.py`

建议顺序：

1. 确认没有文档继续推荐后，再归档或删除非 AMV 入口 `b1_backtest.py` 和 `rotation_backtest.py`。
2. 更老的 AMV exploratory 脚本，必须先确认结论已在现有记录中沉淀；若没有，优先写入 `experiments/` 归档说明，而不是继续膨胀 `progress.md`。

暂缓项：

- `scripts/b1_backtest.py` / `scripts/rotation_backtest.py` 仍有 `backtest-engine/run_b1.bat` / `run_rotation.bat` 调用，且 B1 历史路线文档仍有引用；先维持 `deprecated`，不直接删除。

已完成：

- `scripts/amv_topn_backtest.py` 已由 `qlab backtest` 替代并删除。
- `scripts/amv_limit_refill_rolling_nav.py` 已作为不可交易 Python rolling NAV 上限诊断删除。
- `scripts/amv_ltr_signal_export.py` 已随 LTR 路线降级删除。
- `scripts/amv_topn_enhancement_sweep.py` 已作为早期 enhancement sweep 删除。
- `scripts/amv_context_combo_signal_export.py` 已由 `qlab export context` native workflow 复现 raw ground truth 后删除。
- `qlab export pb3-gated` 已接入 native workflow，并复现 PB3 gated rolling raw ground truth；`scripts/amv_static_sleeve_signal_export.py` 不再作为 Ref/P3/PB3 当前入口。
- `qlab export limit-weakgate` 已接入 native workflow，并复现 limit weakgate 5td raw ground truth；`scripts/amv_limit_ecology_signal_export.py` 不再作为当前 `limit-weakgate` 入口。

## 迁移候选

这些脚本包含可复用逻辑，不应第一批删除：

- `scripts/amv_bull_pool_export_signals.py`
- `scripts/b1_executable_base_lab.py`
- `scripts/amv_regime_phase_diagnostic.py`

迁移方向：

1. 将 score definition 和 sleeve id 抽到 `strategies/amv/`。
2. 将通用数据加载和过滤逻辑抽到 `utils/`。
3. 在 `qlab export` 能直接调用抽出的策略模块前，原脚本先保留为薄 runner。

## 历史归档候选

删除前应先沉淀为 `experiments/` 说明：

- 早期 AMV grid 和 ranker lab
- oracle / horizon 上限 lab
- executable broad scan
- standalone sector / sentiment / medium export candidate
- B3 和 B1 exploratory lab

归档说明至少回答：

- 这条路线当时在验证什么问题？
- 哪个 report 或 Canvas 已沉淀结论？
- 为什么它不属于当前 workflow？
- 如果有替代入口，当前命令是什么？

## 停止条件

出现以下情况时，停止清理并重新评估：

- 删除脚本会移除当前报告的唯一可复现路径
- 当前 Canvas 或 JSON 无法再生成
- `qlab.py` 开始堆积重因子逻辑
- raw execution 和 adjusted execution 结论再次混在一起

## 第二阶段完成定义

满足以下条件时，第二阶段才算完成：

- 日常用户可以从 `CURRENT_STATE.md` 和 `scripts/qlab.py` 开始工作
- `scripts/` 里大部分文件都是薄命令入口
- AMV score 和 preset 元数据位于 `strategies/amv/`
- deprecated 脚本已经删除或归档
- `docs/script-inventory.md` 没有过期状态标签
