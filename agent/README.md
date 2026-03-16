# B1 AI Agent — 每日智能选股工作流

基于 B1 超跌反转策略的 AI 辅助选股 Agent，采用**结构化指标 + K 线图双通道**多模态分析，替代人工盯盘筛选。

## 工作流

```
DuckDB 日线数据
    ↓
calc_b1_factors_wmacd (因子计算 + 周月MACD + 周线WL>YL)
    ↓
筛选当日 B1 信号 → 按 rw_dif_pct 排序 → 取 top_n
    ↓
┌───────────────────┐     ┌───────────────────┐
│  Plotly K线图      │     │  结构化指标文本     │
│  日K + 成交量      │     │  J/KDJ/WL偏离      │
│  WL/YL 叠加       │     │  周月MACD状态       │
│  B1 信号标记      │     │  红绿量比等         │
└────────┬──────────┘     └────────┬──────────┘
         └──────────┬──────────────┘
                    ↓
           Gemini 多模态评审
           (图表 + 指标 → 4维度打分)
                    ↓
         终端输出 + JSON 保存
         BUY / WATCH / SKIP 分类
```

## 使用方法

```bash
# 完整流程: 计算 → 图表 → AI评审 → 推荐
python -m agent.run

# 指定日期
python -m agent.run --date 2026-02-24

# 只生成图表, 跳过 AI 评审
python -m agent.run --skip-review

# 指定配置文件
python -m agent.run --config agent/config.yaml
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `GEMINI_API_KEY` | Google Gemini API Key (使用 Gemini 评审时必需) |

## 模块说明

| 文件 | 职责 |
|------|------|
| `run.py` | 主入口，编排 5 个步骤: 数据加载 → 因子计算 → 候选筛选 → 图表导出 → AI评审 |
| `chart.py` | Plotly K 线图生成 (双行: 日K+WL/YL / 成交量)，导出高分辨率 PNG |
| `context.py` | 从信号行提取结构化指标文本 (J值、KDJ、触发类型、MACD 状态等) |
| `prompt.md` | B1 专用评审提示词 (4 维度加权打分 + JSON 输出约束) |
| `config.yaml` | 全局配置 (数据库路径、模型选择、筛选参数等) |
| `report.py` | 终端格式化输出 + JSON 结果保存 |
| `reviewers/base.py` | BaseReviewer 抽象基类 (模板方法 + 断点续跑 + JSON 解析) |
| `reviewers/gemini.py` | GeminiReviewer 实现 (多模态: 图片 + 文本 → JSON) |

## 配置 (config.yaml)

```yaml
data:
  db_path: "../QuantData/Ashare/qmt_data.duckdb"
  chart_output: "data/charts"       # K线图输出目录
  review_output: "data/review"      # AI评审结果目录

selection:
  top_n: 20                         # 最多评审股票数
  rank_by: "rw_dif_pct"             # 排序字段 (周线DIF强度)
  rank_ascending: false
  bars: 90                          # K线图显示天数

reviewer:
  provider: "gemini"                # 评审模型 (gemini / claude 待实现)
  model: "gemini-2.5-flash"
  request_delay: 3                  # API 调用间隔 (秒), 避免限流
  skip_existing: true               # 断点续跑: 已有结果跳过
  suggest_min_score: 4.0            # BUY 推荐最低分

b1_config:
  WEEKLY_WL_YL_FILTER: true         # 启用周线 WL>YL 过滤
```

## AI 评审维度

| 维度 | 权重 | 评价内容 |
|------|------|---------|
| 趋势健康度 | 20% | 均线排列、WL/YL 关系、周月 MACD 共振 |
| 量价质量 | 30% | 缩量回调程度、红绿量比、量能窒息度 |
| 形态结构 | 30% | K 线形态收敛、实体大小、影线特征 |
| 爆发潜力 | 20% | 触发类型强度、J 值超卖深度、均线乖离空间 |

### 判定标准

| 总分 | 判定 | 说明 |
|------|------|------|
| >= 4.0 | **BUY** | 推荐买入，信号质量高 |
| 3.0 - 4.0 | **WATCH** | 关注观察，等待更好入场点 |
| < 3.0 | **SKIP** | 跳过，信号质量不足 |

## 设计亮点

- **双通道输入**: K 线图 (视觉模式识别) + 结构化指标 (精确数值)，比纯图表分析更可靠
- **可插拔 Reviewer**: 继承 `BaseReviewer` 即可对接任意 LLM (Gemini / Claude / GPT 等)
- **断点续跑**: `skip_existing=true` 时，已评审过的股票自动跳过，中断后可无缝续跑
- **直接函数调用**: 全流程用 Python 函数编排，无 subprocess 调用，数据流清晰
- **复用 QuantLab 核心**: 直接导入 `utils/` 中的因子计算和数据加载模块
