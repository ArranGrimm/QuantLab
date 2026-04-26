# rpa_parse: 活跃市值截图解析

本目录负责把 `rpa_capture/shots/seq_*.png` 解析成结构化的 `active_market_value` 数据。

设计上与 `rpa_capture/` 解耦:

- `rpa_capture/`: Windows 端只截图, 不做 OCR
- `rpa_parse/`: Mac/Windows 解析端做 OCR、字段解析、校验、落盘

## 推荐运行方式

PaddleOCR / PaddlePaddle 对 Python 版本比较敏感, 不建议直接加入主项目 `pyproject.toml`。
推荐用 `uv` 临时创建 Python 3.11 运行环境:

```bash
uv python install 3.11

uv run --python 3.11 \
  --with paddleocr \
  --with paddlepaddle \
  --with polars \
  --with duckdb \
  python rpa_parse/parse_active_market_value.py \
  --input rpa_capture/shots \
  --output data/active_market_value \
  --raw-json
```

如需直接写入 DuckDB:

```bash
uv run --python 3.11 \
  --with paddleocr \
  --with paddlepaddle \
  --with polars \
  --with duckdb \
  python rpa_parse/parse_active_market_value.py \
  --input rpa_capture/shots \
  --output data/active_market_value \
  --duckdb ../QuantData/Ashare/qmt_data.duckdb
```

第一次全量导入时如果确认要覆盖旧表:

```bash
uv run --python 3.11 \
  --with paddleocr \
  --with paddlepaddle \
  --with polars \
  --with duckdb \
  python rpa_parse/parse_active_market_value.py \
  --input rpa_capture/shots \
  --output data/active_market_value \
  --duckdb ../QuantData/Ashare/qmt_data.duckdb \
  --replace-duckdb
```

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

脚本先用 PaddleOCR 提取文本行, 再用字段名和数字正则解析。
只要字段名稳定, 后续 OCR 模型升级不会影响表结构。
