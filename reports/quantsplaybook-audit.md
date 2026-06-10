# QuantsPlaybook 审计报告

**日期**: 2026-06-08  
**仓库**: 与 QuantLab 同级目录（`../QuantsPlaybook`）  
**方法**: 6 agent 并行扫描 B-因子构建类 / C-择时类 / D-组合优化 / A-量化基本面 / SignalMaker，结合 QuantLab 当前状态分类

---

## 分类汇总

| 等级 | 数量 | 含义 |
|------|------|------|
| Tier 1 | 10 | 数据满足，立即值得探索 |
| Tier 2 | 18 | 有价值但数据/条件不够 |
| Tier 3 | 23 | 数据满足但暂无明显价值 |
| Tier 4 | 1 | 不值得探索 |

---

## Tier 1: 立即值得探索（10 个）

### 方向 A：增强现有 trend 策略

#### 1. 高质量动量因子 (B#24)
- **核心**: `r_60 - 3000·σ²`，风险调整动量。与市值相关性仅 0.085（不是伪市值因子）
- **实现**: 单表达式 Polars 因子 → factor registry 新条目 → MediumTrendQualityHook 增强或新建 trend-qm 策略 JSON
- **预期**: 比现有 P-block 更干净的趋势强度度量

#### 2. 上下影线因子 (B#3)
- **核心**: 上影线 = `High - Max(Close,Open)`，下影线 = `Min(Close,Open) - Low`。长上影=卖压/趋势衰竭信号
- **实现**: 纯 OHLC 表达式，直接扩展现有 KLEN/KMID2 → 新 Hook 或集成进 MediumTrendQualityHook
- **预期**: KLEN/KMID2 最自然的延伸，K 线形态分析的第三维度

#### 3. 均线收敛 PCF (B#16)
- **核心**: 多周期 MA(5/10/20/60/120) 标准差 → `-log(1+std)` 作为收敛度。高 PCF + 上升趋势 = 突破前兆
- **实现**: 新因子 + MAConvergenceHook（penalty 型），trend-p3-enhanced 中试用
- **预期**: 帮助我们识别趋势中的"蓄力"阶段，减少假突破

#### 4. 处置效应 CGO (B#13)
- **核心**: 换手率衰减的参考价格计算未实现盈亏。高 CGO=大量浮盈=卖压积累=趋势衰竭
- **实现**: factor registry 新条目 + CGOGateHook（gate 型，post-ranking 过滤）
- **预期**: 填补系统空白——我们有 entry 信号但缺 exit/avoid 信号

### 方向 B：改进市场择时 gate

#### 5. CSVC 熊牛指标 (C#2)
- **核心**: `BullBear = 换手率均值 / 收益率标准差`，4 状态（牛/熊/反弹/震荡）。低频(N=120-250)、低噪音、PBO 验证
- **实现**: BullBearRegimeGateHook（gate 型，指数级计算，缓存模式同 AmvRegimeGateHook）
- **预期**: 直接替代/增强现有 AMV regime 判断

#### 6. RSRS 阻力支撑强度 (C#1 + SignalMaker)
- **核心**: N=18 的 OLS(high~low) β 的 z-score × R²，M=600 标准化。5+ 变体已验证
- **实现**: RSRSRegimeGateHook（gate 型），SignalMaker qrs.py 可参考
- **预期**: 与 CSVC 互补——RSRS 测价格结构，CSVC 测量/波动率

#### 7. 扩散指标 Breadth (C#16 + C#24)
- **核心**: `%股票在MA20之上` + `(52w新高-52w新低)/总数`，阈值 ±20%
- **实现**: BreadthRegimeGateHook（gate 型，全市场日频计算，~1行/天）
- **预期**: 与 RSRS/CSVC 三 gate 互补——breadth 测参与度，RSRS 测价格结构，CSVC 测量/波

### 方向 C：新独立 sleeve

#### 8. STR/STV 凸显反转 (B#7)
- **核心**: 与趋势反向——低关注度(低 salience)股票被低估，高关注度被高估。Rank IC ~0.046，top-50 年化超额 ~17.4%
- **实现**: 新因子 + SalienceHook + 新策略 JSON（reversal-str 或 pullback 家族扩展）
- **预期**: 与 trend 负相关，做组合 sleeve 可压 MaxDD

#### 9. 球队硬币因子 (B#4)
- **核心**: 日内收益拆为 overnight/intraday/interday，波动率+换手率分 coin(均值回归) vs team(趋势)。Rank IC ~0.043
- **实现**: CoinTeamHook（lazy_features + penalty + gate）
- **预期**: coin/team 分类给 trend 策略提供"选股信心度"维度

#### 10. 隔夜-日间网络因子 (B#11)
- **核心**: 隔夜收益与日间收益的交叉相关网络。paper 年化 32.11%/Sharpe 2.37
- **实现**: factor registry + ODNetworkHook（N×N 相关矩阵，需懒加载）
- **预期**: 独立信息流维度，与价格趋势完全正交

---

## Tier 2: 有价值但数据不够（18 个）

| 因子/策略 | 缺失数据 |
|-----------|---------|
| APM 因子模型 (B#1) | 1分钟 K 线 |
| CPV 高频价量相关性 (B#25) | 分钟级 OHLCV |
| 聪明钱因子 2.0 (B#20) | 分钟级 + 订单方向 |
| 买卖压力 APB (B#10) | 30分钟 K 线 |
| A股反转微观来源 W-Factor (B#15) | 逐笔成交数据 |
| C-VIX 中国版 VIX (C#3) | 50ETF 期权链 |
| 企业生命周期 (B#5) | 季报财务数据 |
| 基金重仓超配 (B#12) | 季报基金持仓 |
| 基金经理超额收益 (B#18) | 基金经理履历/持仓 |
| 金股增强策略 (B#23) | 分析师推荐数据库 |
| 因子择时 (B#9) | 12+因子×60月历史 |
| 投资者情绪 GSISI (C#18) | 31 申万行业日收益聚合 |
| 价量共振/龙虎榜 (C#15) | 龙虎榜席位数据 |
| 日内 ETF 动量 (C#8) | ETF 分钟 K 线 |
| SignalMaker noise_area | 分钟 bar + pivot 表 |
| 华泰 FFScore (A#1) | 财报数据 |
| 申万大师系列 (A#2) | 财报+估值数据 |
| 北向资金 (C#7) | 陆股通日净流向 |

---

## Tier 3: 数据满足但暂无明显价值（23 个）

主要是：
- 与现有趋势信号高度相关的趋势跟随型（ICU均线、低延迟趋势线、均线交叉+通道突破、鳄鱼线等）
- 实现成本过高但增量不明确的（筹码因子 CYQ、HHT+分类、小波+SVM）
- 方法论不匹配的（MTL-TSMOM 设计给 4 ETF 轮动、DE 优化给季频调仓、多因子指数增强是组合构建框架非因子）

详见 workflow 输出文件。

---

## Tier 4: 不值得探索（1 个）

- **Trader-Company 遗传规划 (C#26)**: 策略生成元方法，无法验证，无 pipeline 集成路径

---

## 建议实施顺序

1. **方向 B 先做（择时 gate）**——RSRS 或 CSVC 熊牛指标纯计算，不需要新数据，直接改善 "2026 trend 连续亏损"
2. **方向 A #1 高质量动量**——一个表达式，替代或增强 P-block，qpilot 验证成本最低
3. **方向 A #2 上下影线**——KLEN/KMID2 的自然延伸，OHLC 即可
4. **方向 C #8 STR 反转**——独立 sleeve，与 trend 负相关，组合后可能压 MaxDD
