from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "active_market_value" / "active_market_value.parquet"
DEFAULT_DB = ROOT.parent / "QuantData" / "Ashare" / "active_market_value.duckdb"


TARGET_COLUMNS = [
    "trade_date",
    "amv_open",
    "amv_high",
    "amv_low",
    "amv_close",
    "chg_abs_pct",
    "volume_100m",
    "amount_100m",
    "position_100m",
    "turnover_pct",
    "amplitude_pct",
    "source",
    "source_seq",
    "source_filename",
    "source_captured_at",
    "weekday_text",
    "ocr_min_confidence",
    "quality_flags",
    "raw_ocr_text",
    "ingested_at",
]


def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"非法 SQL identifier: {name}")
    return f'"{name}"'


def create_table_sql(table: str) -> str:
    t = quote_ident(table)
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    trade_date          DATE PRIMARY KEY,

    amv_open            DOUBLE NOT NULL,
    amv_high            DOUBLE NOT NULL,
    amv_low             DOUBLE NOT NULL,
    amv_close           DOUBLE NOT NULL,

    -- 指南针 readout 的「幅」是绝对涨跌幅, 不是 signed return.
    chg_abs_pct         DOUBLE NOT NULL,

    -- 指南针 readout 的「量 / 额 / 盘」单位均为「亿」.
    volume_100m         DOUBLE NOT NULL,
    amount_100m         DOUBLE NOT NULL,
    position_100m       DOUBLE NOT NULL,

    turnover_pct        DOUBLE NOT NULL,
    amplitude_pct       DOUBLE NOT NULL,

    source              TEXT NOT NULL,
    source_seq          INTEGER,
    source_filename     TEXT,
    source_captured_at  TIMESTAMP,
    weekday_text        TEXT,
    ocr_min_confidence  DOUBLE,
    quality_flags       TEXT,
    raw_ocr_text        TEXT,

    ingested_at         TIMESTAMP NOT NULL
);
"""


def create_stage_sql(input_path: Path) -> tuple[str, list[str]]:
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        reader = "read_parquet(?)"
    elif suffix == ".csv":
        reader = "read_csv_auto(?)"
    else:
        raise ValueError(f"只支持 .parquet / .csv 输入: {input_path}")

    sql = f"""
CREATE OR REPLACE TEMP TABLE amv_stage AS
SELECT
    CAST(trade_date AS DATE) AS trade_date,
    CAST(open AS DOUBLE) AS amv_open,
    CAST(high AS DOUBLE) AS amv_high,
    CAST(low AS DOUBLE) AS amv_low,
    CAST(close AS DOUBLE) AS amv_close,
    CAST(chg_pct AS DOUBLE) AS chg_abs_pct,
    CAST(volume AS DOUBLE) AS volume_100m,
    CAST(amount AS DOUBLE) AS amount_100m,
    CAST(position AS DOUBLE) AS position_100m,
    CAST(turnover AS DOUBLE) AS turnover_pct,
    CAST(amplitude AS DOUBLE) AS amplitude_pct,
    'rpa_vision_v1' AS source,
    CAST(seq AS INTEGER) AS source_seq,
    CAST(filename AS TEXT) AS source_filename,
    TRY_CAST(captured_at AS TIMESTAMP) AS source_captured_at,
    CAST(weekday AS TEXT) AS weekday_text,
    CAST(ocr_min_confidence AS DOUBLE) AS ocr_min_confidence,
    NULLIF(CAST(review_reason AS TEXT), '') AS quality_flags,
    CAST(ocr_text AS TEXT) AS raw_ocr_text,
    current_timestamp AS ingested_at
FROM {reader};
"""
    return sql, [str(input_path)]


def create_qc_view_sql(table: str) -> str:
    t = quote_ident(table)
    view = quote_ident(f"{table}_qc")
    return f"""
CREATE OR REPLACE VIEW {view} AS
WITH x AS (
    SELECT
        *,
        lag(amv_close) OVER (ORDER BY trade_date) AS prev_close
    FROM {t}
)
SELECT
    *,
    abs((amv_close - prev_close) / prev_close * 100) AS calc_chg_abs_pct,
    (amv_high - amv_low) / prev_close * 100 AS calc_amplitude_pct,
    abs(chg_abs_pct - abs((amv_close - prev_close) / prev_close * 100)) AS chg_abs_pct_err,
    abs(amplitude_pct - ((amv_high - amv_low) / prev_close * 100)) AS amplitude_pct_err
FROM x;
"""


def validate_stage(conn) -> None:
    duplicate_count = conn.execute(
        """
        SELECT count(*)
        FROM (
            SELECT trade_date
            FROM amv_stage
            GROUP BY trade_date
            HAVING count(*) > 1
        );
        """
    ).fetchone()[0]
    if duplicate_count:
        raise ValueError(f"amv_stage 存在重复 trade_date: {duplicate_count} 个日期")

    null_count = conn.execute(
        """
        SELECT count(*)
        FROM amv_stage
        WHERE trade_date IS NULL
           OR amv_open IS NULL
           OR amv_high IS NULL
           OR amv_low IS NULL
           OR amv_close IS NULL
           OR chg_abs_pct IS NULL
           OR volume_100m IS NULL
           OR amount_100m IS NULL
           OR position_100m IS NULL
           OR turnover_pct IS NULL
           OR amplitude_pct IS NULL;
        """
    ).fetchone()[0]
    if null_count:
        raise ValueError(f"amv_stage 存在核心字段 NULL 行: {null_count}")


def upsert_sql(table: str) -> str:
    t = quote_ident(table)
    columns = ", ".join(TARGET_COLUMNS)
    excluded_sets = ",\n        ".join(
        f"{col} = EXCLUDED.{col}" for col in TARGET_COLUMNS if col != "trade_date"
    )
    return f"""
INSERT INTO {t} ({columns})
SELECT {columns}
FROM amv_stage
ON CONFLICT (trade_date) DO UPDATE SET
        {excluded_sets};
"""


def replace_sql(table: str) -> str:
    t = quote_ident(table)
    columns = ", ".join(TARGET_COLUMNS)
    return f"""
DELETE FROM {t};
INSERT INTO {t} ({columns})
SELECT {columns}
FROM amv_stage;
"""


def qc_error_count(conn, table: str, max_error_pp: float) -> int:
    view = quote_ident(f"{table}_qc")
    return conn.execute(
        f"""
        SELECT count(*)
        FROM {view}
        WHERE prev_close IS NOT NULL
          AND (
              chg_abs_pct_err > ?
              OR amplitude_pct_err > ?
          );
        """,
        [max_error_pp, max_error_pp],
    ).fetchone()[0]


def main() -> None:
    import duckdb

    parser = argparse.ArgumentParser(description="导入活跃市值 OCR 结果到 DuckDB")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="active_market_value parquet/csv")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="目标 DuckDB 路径")
    parser.add_argument("--table", default="active_market_value", help="目标表名")
    parser.add_argument(
        "--mode",
        choices=["upsert", "replace"],
        default="upsert",
        help="upsert: 按 trade_date 覆盖; replace: 清空表后重建数据",
    )
    parser.add_argument("--max-error-pp", type=float, default=0.08, help="QC 允许误差, 单位百分点")
    parser.add_argument("--allow-qc-errors", action="store_true", help="即使 QC 有异常也不失败")
    args = parser.parse_args()

    input_path = args.input.resolve()
    db_path = args.db.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    conn.execute("BEGIN TRANSACTION")
    try:
        if args.mode == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {quote_ident(args.table)}")
        conn.execute(create_table_sql(args.table))
        stage_sql, params = create_stage_sql(input_path)
        conn.execute(stage_sql, params)
        validate_stage(conn)
        if args.mode == "replace":
            conn.execute(replace_sql(args.table))
        else:
            conn.execute(upsert_sql(args.table))
        conn.execute(create_qc_view_sql(args.table))
        qc_errors = qc_error_count(conn, args.table, args.max_error_pp)
        if qc_errors and not args.allow_qc_errors:
            raise ValueError(
                f"{args.table}_qc 发现 {qc_errors} 行超过 {args.max_error_pp}pp 误差; "
                "请先排查或加 --allow-qc-errors"
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    summary = conn.execute(
        f"""
        SELECT
            count(*) AS rows,
            min(trade_date) AS min_date,
            max(trade_date) AS max_date,
            count(*) FILTER (WHERE quality_flags IS NOT NULL) AS flagged_rows
        FROM {quote_ident(args.table)};
        """
    ).fetchone()
    print(f"db: {db_path}")
    print(f"table: {args.table}")
    print(f"mode: {args.mode}")
    print(f"rows: {summary[0]}")
    print(f"date: {summary[1]} -> {summary[2]}")
    print(f"flagged_rows: {summary[3]}")
    print(f"qc_errors: {qc_errors}")
    print(f"qc_view: {args.table}_qc")


if __name__ == "__main__":
    main()
