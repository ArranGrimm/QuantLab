# 活跃市值自动化 (Active Market Value Automation)

> 目标: 把指南针客户端的 `0AMV (活跃市值)` 指标从"手工标记 25 段 `LOOSE_PERIODS`"升级为"自动化抓取 + 解析 + 入库 + 日更"的完整数据管道。
> 当前状态: **`Phase 1 Capture` PoC 已通过 (`2026-04-21`)**, 进入历史回填 + Parse 阶段。

---

## 一、为什么要做这件事

### 前期发现回顾

#### 发现 1: 项目唯一稳定 alpha 来自活跃市值

- `B1` 策略 5 年回测的主要 alpha 来源: **手工标记的 25 段 `LOOSE_PERIODS`** (即活跃市值多头区间)
  - 详见 `experiments/b1-next-phase.md` 教科书路线证伪结论
  - `b1_alpha_proof.py Q5`: 全市场基线 +0.24% / 仅多头区间开仓 **+1.46pp**
  - `simple_b1_lab.py` 6 年回测复核: 总体 +2.60% (持仓 20 天)
- `Rotation` 截面策略叠加 `is_bull_regime` 后提升有限, 也间接说明 alpha 在时序而非截面
- 教科书形态特征 (J<14 / 白>黄+收>黄 / 量价健康 / N 字结构) 单独全部不显著

#### 发现 2: alpha 的本质是 timing 不是 selection

- `b1_alpha_proof.py Q7-2` 反直觉发现:
  - 多头日开仓 vs 当月每天买, 月均差仅 -0.21pp, t = -0.636 不显著
  - 多头日胜出当月每天买的月份只有 7/41 = 17.07%
- 真正的 alpha 描述: **"只在多头时段开仓, 避开非多头时段不开仓"** 这个择时动作本身值钱
- 不是"多头时段票更好" (cross-section selection alpha), 而是"非多头时段空仓" (timing alpha)

#### 发现 3: 当前手工 `LOOSE_PERIODS` 是单点依赖, 阻塞了所有后续路线

- 25 段 hindsight 圈定的多头期, 存在四个根本性问题:
  - **可执行性差**: 每段都是事后看图圈出, 实盘不能用
  - **无法日更**: 必须人盯盘判断 regime 是否切换, 频率受限
  - **机械替代失败**: 用"单根 ≥+4% / -2.3%"等机械规则替代后, 交易笔数翻倍, 胜率从 50% 降到 36%, 长窗 MDD 抬升到 35%~40% (见 `b1-next-phase.md` 活跃市值口径更新)
  - **无法验证**: 我们手上没有完整历史的活跃市值序列, 无法对账, 无法做阈值优化, 无法跑网格搜索
- 这条单点依赖直接卡住了:
  - `B1 路线 C` (优化择时本身, 已升级为主线候选, 见 `b1-next-phase.md`)
  - `Rotation` 是否值得叠加 timing 的最终判定
  - 整个项目长期对标的"多策略 + regime 切换"框架 (见 `target-strategy-evolution.md` 八)

#### 发现 4: 活跃市值是指南针专利, 没有公开 API / 数据接口

- 客户端 (`指南针全赢系统`) 是唯一来源
- 没有导出功能, 不开放协议
- 闲鱼上有人卖 "0AMV 全历史 1993~2026 Excel 200 元", 90% 概率也是 RPA 抓的, 但缺乏:
  - 可追溯性 (字段口径未声明)
  - 日更服务 (买完就完)
  - 数据质量报告
  - 校准能力 (无法自检)
- **结论: 必须自建 RPA 管道**, 是项目长期可持续的必要基础设施

### 战略判断

把活跃市值从"手工资源"升级为"自动化数据源", 是当前项目**单笔 ROI 最高**的工程投入:

- 一次性成本: ~3 小时 PoC + 后续 OCR + 入库
- 长期收益:
  - 解锁 `B1 路线 C` 的所有择时优化研究 (阈值网格 / 切换日衰减 / 区间内子段差异)
  - 让 `Rotation` 能用活跃市值做 regime feature 而非 hindsight 标签
  - 为"多策略 + regime 切换"长期目标提供机械可执行的 regime 输入
  - 沉淀 RPA 框架, 后续可扩展抓 D股价活跃度 / 资金主力 / 大盘资金等其他指南针指标

---

## 二、架构设计

### 两阶段解耦

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1: Capture (Windows 端, ✅ PoC 已通过)                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  指南针客户端 (0AMV K线 + readout 小窗)                    │  │
│  │      │                                                     │  │
│  │      ▼ pywinauto 拉前台 + 发 → 方向键                      │  │
│  │      ▼ mss 区域截图                                        │  │
│  │      ▼                                                     │  │
│  │  rpa_capture/shots/seq_NNNNN.png  +  manifest.jsonl        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│           PNG + manifest.jsonl (跨设备共享文件夹)                │
│                              │                                   │
│                              ▼                                   │
│  Phase 2: Parse (Mac 端, 待实现)                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  PaddleOCR (中文小字体 SOTA)                               │  │
│  │      ▼ 11 字段抽取 (date/OHLC/幅/量/额/盘/率/振)          │  │
│  │  polars 校验 (字段类型 / 日期连续 / 重复检测)              │  │
│  │      ▼                                                     │  │
│  │  DuckDB active_market_value 表                             │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  消费侧                                                          │
│  - utils/active_market_value.py  (规则引擎: 多/空头判定)          │
│  - utils/manual_bull_periods.py  (旧版 25 段, 废弃路径)           │
│  - notebooks/b1_*  +  notebooks/cross_section_rotation.py        │
└──────────────────────────────────────────────────────────────────┘
```

### 解耦原则

- Capture / Parse 通过 PNG + manifest.jsonl 通信, **OCR 方案升级不需要重抓**
- Parse 阶段 OCR 失败/低置信度可单独重跑, 不影响截图
- 入库后旧规则引擎和新规则引擎都从 DuckDB 取数, 不再分裂

### 依赖纪律

- Capture 阶段 (Windows VM): 仅 `pywinauto + mss` 两个包, 不挑环境
- Parse 阶段 (Mac 主力): 走项目现有 `uv` 环境 (polars / duckdb 已有, 仅新增 `paddleocr`)

### 部署形态演进

| 阶段 | Capture 跑在哪 | Parse 跑在哪 | 数据传输 |
|---|---|---|---|
| 当前 PoC | Windows 物理机 | (未跑) | (未传) |
| 短期 | Windows 物理机或 Mac+PD VM | Mac 主力 | 共享文件夹 / Git LFS / scp |
| 长期 | Mac+PD VM (固定) | Mac 主力 | Parallels 共享文件夹 |

---

## 三、Phase 1: Capture (✅ 已完成)

### PoC 验收 (`2026-04-21`)

| 验收项 | 结果 |
|---|---|
| 起点准确 | 手动选 2019-01-02, 截图显示 `20190102 周三`, 不漂 |
| 方向正确 | 按 → 后日期严格 +1 交易日 |
| 周末跳过 | `20190104(周五) → 20190107(周一)`, 指南针自己处理 |
| 单图清晰度 | 11 字段全部可读, 字号约 12px, OCR 难度极低 |
| 单图体积 | ~9 KB (区域裁剪 1284×111), 全屏方案的 1/300 |
| 性能 | 246 ms/张, 推算 1700 天 ≈ 7 分钟 |

### 关键技术决策

| 决策 | 取舍 | 选择 |
|---|---|---|
| 拉前台方式 | `pywinauto.Application` vs Win32 `SetForegroundWindow` | **Win32**, 避免 pywinauto 的"合成点击"导致 cursor 漂移 |
| 鼠标位置 | 不管 vs 截图前停到 (2,2) | **停到 (2,2)**, 防止主图 hover 干扰 cursor |
| 截图范围 | 全屏 vs 区域裁剪 | **区域裁剪**, 单图 ~9 KB vs ~3 MB |
| 时间方向 | seq=0 最新 vs seq=0 最早 | **seq=0 最早**, 顺序与日期单调一致, 入库无需 reverse |
| DPI 处理 | 默认 vs `SetProcessDpiAwarenessContext` | **显式 DPI 感知**, 高分屏坐标不偏 |

### 输出文件

- `rpa_capture/run_capture.py`: 主入口, 支持 `--days / --resume / --no-focus / --precount / --region` 等参数
- `rpa_capture/calibrate_region.py`: 交互式 readout 区域标定 (tkinter, stdlib 无新依赖)
- `rpa_capture/requirements.txt`: 仅 2 个包
- `rpa_capture/README.md`: 完整使用说明 + cursor 漂移问题 troubleshooting

---

## 四、Phase 2: Parse (🟡 待实现)

### 目标

把 `rpa_capture/shots/*.png + manifest.jsonl` 转成结构化 DataFrame, 写入 DuckDB。

### 依赖建议

- **PaddleOCR**: 中文小字体准确率最高的开源 OCR (相比 EasyOCR / Tesseract), 数字 / 日期识别接近 100%
- **polars**: 类型校验 + 缺失字段处理
- **duckdb**: 已是项目主存储

### 11 个 readout 字段

| 字段 | 类型 | 示例 | 备注 |
|---|---|---|---|
| `trade_date` | DATE | `2019-01-02` | 主键 |
| `weekday` | TEXT | `周三` | 校验用, 入库后丢弃 |
| `open` | DOUBLE | `2497.88` | |
| `high` | DOUBLE | `2538.88` | |
| `low` | DOUBLE | `2488.66` | |
| `close` | DOUBLE | `2533.66` | |
| `chg_pct` | DOUBLE | `1.43` | 涨幅 % |
| `volume` | DOUBLE | `26.54e8` | 量 (亿) |
| `amount` | DOUBLE | `2853e8` | 额 (亿) |
| `position` | DOUBLE | `-` | 盘 (留空, 部分日期没有) |
| `turnover` | DOUBLE | `-` | 率 (留空) |
| `amplitude` | DOUBLE | `2.04` | 振 % |

### Schema 草案

```sql
CREATE TABLE active_market_value (
  trade_date    DATE PRIMARY KEY,
  open          DOUBLE,
  high          DOUBLE,
  low           DOUBLE,
  close         DOUBLE,
  chg_pct       DOUBLE,
  volume        DOUBLE,
  amount        DOUBLE,
  position      DOUBLE,
  turnover      DOUBLE,
  amplitude     DOUBLE,
  captured_at   TIMESTAMP,
  source        TEXT,         -- 'rpa_v1' / 'manual' / 未来 'rpa_v2'
  ocr_confidence DOUBLE       -- min(每个字段 OCR 置信度)
);
```

### 校验规则

- 日期连续 (允许跳过周末和节假日, 但不允许跳工作日)
- `low <= open / close <= high`
- `chg_pct = (close - prev_close) / prev_close * 100` ±0.05 容差
- `amplitude = (high - low) / prev_close * 100` ±0.05 容差
- 任何一个字段 OCR 置信度 < 0.85 → 标记为 `needs_review`, 不直接入主表

### Parse 模块文件结构 (草案)

```
rpa_parse/
├── ocr_extract.py        # PaddleOCR 封装, 单张 PNG → dict
├── validate.py           # 11 字段类型 / 范围 / 连续性校验
├── ingest.py             # 写 DuckDB, 处理 upsert / conflict
├── reconcile.py          # 跟手工 LOOSE_PERIODS 交叉对账
└── requirements.txt      # paddleocr (paddle paddlepaddle 已有 GPU 版本)
```

### Parse 阶段验收门槛

- [ ] 1700 天历史数据全部入库, 无字段缺失
- [ ] 校验规则通过率 ≥ 99.5%
- [ ] 抽样 50 天人工对账, 字段值 100% 一致
- [ ] 跟手工 25 段 `LOOSE_PERIODS` 对账, 起止日重合度 ≥ 95%

---

## 五、Phase 3: 规则引擎 + 历史复算 (🔵 待规划)

### 任务

把"活跃市值 → 是否多头" 的判定规则**机械化**, 跑出全历史的 `is_bull_regime` 序列, 再回头跟手工 25 段对账。

### 候选规则清单 (按复杂度递增)

| 规则名 | 定义 | 来源 |
|---|---|---|
| R0 | 单根 chg_pct ≥ +4% 入场, 单根 chg_pct ≤ -2.3% 退场 | `b1-next-phase.md` 已有定义 |
| R1 | 2 根累计 ≥ +4% 入场, 退场同上 | 同上 |
| R2 | 3 根累计 ≥ +4% 入场, 退场同上 | 同上 |
| R3 | 引入 close 相对 MA13/34 位置作为辅助判定 | 待设计 |
| R4 | 引入 amount / volume 协同放量判定 | 待设计 |

### 验收方式

- 对每个规则, 计算与手工 25 段的:
  - 起止日精确重合率
  - 总日数差异
  - 多/空切换日数差异
  - 用各规则跑 `b1_alpha_proof.py Q2`, 看 `ret_lift` 是否仍 ≥ +1.0pp
- 期望: **找到一个机械规则, 既能复现手工 LOOSE_PERIODS 的多数区间 (≥80%), 又能保住 +1.0pp 以上的 alpha**
- 若所有候选规则都失败, 则说明手工 LOOSE_PERIODS 含**不可机械化的主观成分**, 需要重新评估 alpha 是否真的稳健

### 阈值网格搜索

数据有了之后, 可以网格搜索:

- 入场阈值: chg_pct ∈ {+3%, +3.5%, +4%, +4.5%, +5%}
- 退场阈值: chg_pct ∈ {-1.5%, -2%, -2.3%, -2.5%, -3%}
- 入场窗口: {1根, 2根累计, 3根累计}
- 共 5 × 5 × 3 = 75 个组合, 看哪一组在 1700 天历史上 alpha 最稳

---

## 六、Phase 4: 集成到 B1 / Rotation (🔵 远期)

### B1 集成

- `notebooks/simple_b1_lab.py`: 用机械 regime 替换 `LOOSE_PERIODS`, 跑相同 6314 个信号回测
- `notebooks/b1_alpha_proof.py`: Q2/Q5/Q7 全部用机械 regime 重跑, 确认 alpha 不缩水
- `notebooks/b1_seed_ml_baseline.py`: `is_manual_bull` → `is_active_bull` 重命名 + 重跑 ML
- 验收: ML B1 在机械 regime 下 MDD 控制在 15% 以内, 胜率回到 45%+ (当前是 36%~45%)

### Rotation 集成

- `notebooks/cross_section_rotation.py`: `is_bull_regime` 字段从手工源切到 active_market_value 派生
- 探索新维度:
  - 用 `chg_pct` / `amplitude` 作为 regime 强度连续值, 而非 0/1 开关
  - 用 `close vs MA34` 作为长期 regime, `close vs MA5` 作为短期 regime
- 验收: rotation 在机械 regime 下年化与手工版差距 ≤ 2pp

### 多策略框架

- 一旦 active_market_value 成为可日更的机械数据源, 就为 `target-strategy-evolution.md` 长期目标的"多策略 + 状态切换"铺好了 regime feature 基础
- 后续可基于此设计:
  - regime A (active_bull + 高波动): 跑 B1
  - regime B (active_bull + 低波动): 跑 Rotation top10
  - regime C (active_bear): 全部空仓
  - regime 切换日: 仓位 rebalance 触发

---

## 七、Phase 5: 日更 + 运维 (🔵 远期)

### 日更触发

- Windows 计划任务: 每天 16:00 (收盘后) 自动启动指南针
- Capture 脚本扩展: 增加 `--incremental` 模式, 只截最新 1~3 天 (容错重跑)
- Parse 脚本: 跟踪 `manifest.jsonl` 增量

### 监控

- 每天日更后写一份 `daily_health.json`:
  - 当日 OCR 成功率
  - 当日字段校验通过率
  - 当日 regime 是否切换 (推送告警)

### 灾备

- `rpa_capture/shots/` 全量打包到 NAS / 云盘, 每月一次
- DuckDB 表 dump 进 git 跟踪 (体积小, ~1MB)
- 指南针客户端版本 / 屏幕分辨率变更时, 重跑 `calibrate_region.py`

---

## 八、Phase 6: 扩展到其他指南针指标 (🔵 极远期)

如果 active_market_value 管道跑通, 同一套 RPA 框架可以扩展到:

- `D股价活跃度` (类似 0AMV 的另一个择时指标)
- `资金主力` (大单流向)
- `大盘资金` (主力动向)
- `板块活跃度` (行业 regime)

每加一个指标只需要:

1. 在指南针里手动定位
2. 跑一次 `calibrate_region.py` 标定新区域
3. 复制 `run_capture.py` 改 region 配置
4. Parse 阶段加新表

---

## 九、风险与不确定性

| 风险 | 影响 | 缓解 |
|---|---|---|
| 指南针客户端 UI 改版 | region 坐标失效 | 标定脚本可快速重跑 |
| OCR 字段误识 | 入库脏数据 | 校验规则 + ocr_confidence 阈值 + 抽样人工对账 |
| 机械规则跟手工 LOOSE_PERIODS 偏差大 | alpha 缩水 | Phase 3 网格搜索 + 兜底保留手工版作为对照 |
| 活跃市值未来停止更新 | 整个数据源失效 | 同步备份历史数据 + 探索 D股价活跃度作为替代 |
| 指南针服务条款限制 RPA | 法律 / 账号风险 | 仅本地使用, 不公开分享数据, 不做付费替代品 |

---

## 十、当前 TODO 清单

- [x] Phase 1 Capture PoC (`2026-04-21`)
- [ ] Phase 1 历史回填: PD VM 内跑 `--days 1700`, 输出全量 PNG (用户在家执行)
- [ ] Phase 2 Parse: `rpa_parse/ocr_extract.py` 用 PaddleOCR 实现 11 字段抽取
- [ ] Phase 2 Parse: `rpa_parse/validate.py` 实现校验规则
- [ ] Phase 2 Parse: `rpa_parse/ingest.py` 入 DuckDB
- [ ] Phase 3 对账: 跟手工 25 段 `LOOSE_PERIODS` 交叉对账, 评估 R0/R1/R2 三个候选机械规则
- [ ] Phase 3 网格搜索: 75 组阈值组合, 找出 alpha 最稳定的一组
- [ ] Phase 4 集成: 把 `simple_b1_lab.py` 切到机械 regime, 验证 alpha 不缩水
- [ ] Phase 5 日更: Windows 计划任务 + 增量 Capture + 监控
- [ ] Phase 6 扩展: 评估抓取 D股价活跃度的可行性

---

## 十一、相关文档

- `progress.md` `2026-04-21`: 本次 Capture PoC 验收记录
- `project-status.md` 四 → `活跃市值 RPA 管道`: 当前状态摘要
- `experiments/b1-next-phase.md`: B1 路线 C (优化择时本身) 的研究问题清单
- `experiments/target-strategy-evolution.md` 八: 长期"多策略 + 状态切换"框架
- `rpa_capture/README.md`: Capture 脚本使用说明
- `utils/manual_bull_periods.py` (待重命名): 当前手工 25 段 `LOOSE_PERIODS` 定义, Phase 4 集成后保留为对照
