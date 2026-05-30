# rpa_parse: 活跃市值截图解析

本目录负责把 `rpa_capture/shots/seq_*.png` 解析成结构化的 `active_market_value` 数据。

Capture 侧为什么存在、如何在指南针里截图, 见 [`../rpa_capture/README.md`](../rpa_capture/README.md)。

设计上与 `rpa_capture/` 解耦:

- `rpa_capture/`: Windows 端只截图, 不做 OCR
- `rpa_parse/`: macOS 端用原生 Vision Framework 做 OCR、字段解析、校验、落盘

## 推荐运行方式

当前默认使用 **macOS Vision Framework**。PyObjC 绑定已写入项目依赖, 第一次使用前同步环境:

```bash
uv sync
```

先跑 1 张冒烟:

```bash
uv run python rpa_parse/parse_active_market_value.py \
  --input rpa_parse/shots \
  --output data/active_market_value_test \
  --raw-json \
  --limit 1 \
  --progress-every 1
```

确认 `active_market_value_test/active_market_value.csv` 字段正常后, 跑全量:

```bash
uv run python rpa_parse/parse_active_market_value.py \
  --input rpa_parse/shots \
  --output data/active_market_value \
  --raw-json
```

后续日更只解析新增截图:

```bash
uv run python rpa_parse/parse_active_market_value.py \
  --input rpa_parse/shots \
  --output data/active_market_value \
  --raw-json \
  --incremental \
  --progress-every 1
```

增量模式会读取已有 `active_market_value.parquet`, 按 `seq` 跳过已入表图片, 只 OCR 新增 `seq_*.png`。输出仍会重写完整的 `active_market_value.csv / parquet / review.csv`。

写入 DuckDB 由独立 ingest 脚本负责。默认写入 `../QuantData/Ashare/active_market_value.duckdb`, 按 `trade_date` upsert, 已存在日期会覆盖:

```bash
uv run python rpa_parse/ingest_active_market_value.py \
  --input data/active_market_value/active_market_value.parquet \
  --mode upsert
```

如果确认要全表替换:

```bash
uv run python rpa_parse/ingest_active_market_value.py \
  --input data/active_market_value/active_market_value.parquet \
  --mode replace
```

活跃市值不是 QMT 数据源, 因此单独维护 `active_market_value.duckdb`。需要和 QMT 行情联表时, 在研究 SQL 里同时 `ATTACH` 两个库。

## 输出文件

```text
data/active_market_value/
├── active_market_value.parquet
├── active_market_value.csv
├── active_market_value_review.csv
└── raw_ocr/
    ├── seq_00000.json
    └── ...
```

字段:

- `seq`: 截图序号
- `filename`: 原始截图文件名
- `trade_date`: 交易日
- `weekday`: 截图中的中文星期
- `open/high/low/close`: 活跃市值 OHLC
- `chg_pct`: 幅, 单位 %
- `volume`: 量, 单位 亿
- `amount`: 额, 单位 亿
- `position`: 盘, 单位 亿
- `turnover`: 率, 单位 %
- `amplitude`: 振, 单位 %
- `ocr_min_confidence`: 单张图 OCR 最低置信度
- `review_reason`: 需要人工复核的原因
- `ocr_text`: OCR 合并文本, 便于排查

备注: Vision 对短字段的 `confidence` 经常是 `0.5 / 1.0` 这类粗粒度值, 默认复核阈值设为 `0.5`。如果你想更严格, 可手动传 `--confidence-threshold 0.85`。

## 人工复核流程

优先打开:

```bash
data/active_market_value/active_market_value_review.csv
```

常见复核原因:

- `missing:*`: 有字段没识别出来
- `price_range`: OHLC 不满足 `low <= open/close <= high`
- `low_confidence`: OCR 最低置信度低于阈值

复核时打开对应 `filename`, 手工修正 CSV 后, 再重新写入 DuckDB。

## DuckDB 表结构

`ingest_active_market_value.py` 会创建:

- `active_market_value`: 主表, `trade_date DATE PRIMARY KEY`
- `active_market_value_qc`: 质量检查 view, 用前一日 `amv_close` 复算 `chg_abs_pct / amplitude_pct`

主表字段使用 `amv_` 前缀:

- `amv_open / amv_high / amv_low / amv_close`
- `chg_abs_pct`: 截图里的「幅」, 是绝对涨跌幅
- `volume_100m / amount_100m / position_100m`: 单位均为「亿」
- `turnover_pct / amplitude_pct`
- `source_seq / source_filename / raw_ocr_text / quality_flags`: OCR 追溯信息

## 解析策略

截图格式固定为 11 行:

```text
20260105 周一
开:169063.1
高:174667.0
低:169063.1
收:174652.1
幅:4.37%
量:1427.34亿
额:25461.01亿
盘:71544.55亿
率:2.00%
振:3.35%
```

脚本先用 macOS Vision 提取文本行, 再用字段名和数字正则解析。
只要字段名稳定, 后续 OCR 后端细节不会影响表结构。

## 后续工作

- [ ] 全量历史入库后, 抽样人工对账 OCR 字段
- [ ] 基于 `active_market_value.duckdb` 设计机械化的多/空头 regime 规则, 替代手工 `LOOSE_PERIODS`
- [ ] 与 AMV 主线 (`strategies/amv/`, `qlab export`) 联调 regime feature 消费路径
- [ ] 收盘后日更: capture 增量截图 → `--incremental` 解析 → `ingest ... --mode upsert`

长期系统目标 (多策略 + regime 切换) 见 `strategies/target-strategy-evolution.md`; 本目录只负责把活跃市值变成可查询的数据源。

## 风险备忘

| 风险 | 缓解 |
|---|---|
| OCR 误识 | `active_market_value_review.csv` + OHLC 范围校验 |
| 日期跳变 / 重复 | 解析阶段检查; 必要时回到 capture 重抓 |
| 数据源中断 | 定期备份 `shots/` 和 DuckDB dump |
