"""半自动补录活跃市值 — Windows / Mac 均可运行, 不依赖 Vision OCR.

人工从指南针复制 11 行 readout, 校验后直接 upsert 到 active_market_value.duckdb。
不写 parquet, 不影响 Mac 批量 OCR parse。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from rpa_parse.ingest_active_market_value import (
    DEFAULT_DB,
    create_qc_view_sql,
    create_table_sql,
    qc_error_count,
    quote_ident,
    upsert_sql,
    validate_stage,
)
from rpa_parse.parse_active_market_value import (
    OCRLine,
    REQUIRED_FIELDS,
    normalize_text,
    parse_lines,
)

MANUAL_SOURCE = "manual_entry"
MANUAL_CORE_FIELDS = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
]


def parse_readout_text(text: str) -> dict[str, Any]:
    lines = [
        OCRLine(text=normalize_text(line), confidence=1.0)
        for line in text.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        raise ValueError("readout 为空")
    return parse_lines(lines)


def _field_missing(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if value is None:
        return True
    if field in {"position", "turnover"} and value == 0:
        return True
    return False


def fill_missing_fields(
    row: dict[str, Any],
    *,
    db_path: Path,
    table: str,
) -> tuple[dict[str, Any], list[str]]:
    """补全手机 App 常缺的「盘 / 率」, 以及可选的「幅」."""
    import duckdb

    notes: list[str] = []
    trade_date = row["trade_date"]
    if not db_path.exists():
        return row, notes

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        prev = conn.execute(
            f"""
            SELECT trade_date, amv_close, position_100m
            FROM {quote_ident(table)}
            WHERE trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 1
            """,
            [trade_date],
        ).fetchone()
    finally:
        conn.close()

    if prev is None:
        return row, notes

    prev_date, prev_close, prev_position = prev

    if _field_missing(row, "chg_pct") and prev_close:
        row["chg_pct"] = abs(row["close"] / prev_close - 1) * 100
        notes.append(f"inferred:chg_pct<=prev_close({prev_date})")

    if _field_missing(row, "position") and prev_position:
        row["position"] = prev_position
        notes.append(f"inferred:position<=prev_row({prev_date})")

    if _field_missing(row, "turnover") and row.get("position") and row.get("volume"):
        row["turnover"] = row["volume"] / row["position"] * 100
        notes.append("inferred:turnover<=volume/position*100")

    return row, notes


def validate_parsed_row(row: dict[str, Any], *, core_only: bool = False) -> list[str]:
    issues: list[str] = []
    required = MANUAL_CORE_FIELDS if core_only else REQUIRED_FIELDS
    missing: list[str] = []
    for field in required:
        if field == "trade_date":
            if row.get(field) is None:
                missing.append(field)
        elif _field_missing(row, field):
            missing.append(field)
    if missing:
        issues.append(f"missing:{','.join(missing)}")

    open_px = row.get("open")
    high_px = row.get("high")
    low_px = row.get("low")
    close_px = row.get("close")
    if None not in (open_px, high_px, low_px, close_px):
        if not (low_px <= open_px <= high_px and low_px <= close_px <= high_px):
            issues.append("price_range")

    return issues


def read_input_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8")
    if args.text is not None:
        return args.text
    print("粘贴指南针 0AMV readout (手机 App 通常 9 行), 空行结束:")
    print("  开/高/低/收/幅/量/额/振 — 盘/率 可省略, 脚本自动补")
    print("")
    chunks: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip() and chunks:
            break
        if line.strip() or chunks:
            chunks.append(line)
    text = "\n".join(chunks).strip()
    if not text:
        raise ValueError("未输入 readout 文本")
    return text


def show_gap(db_path: Path, *, compare_tdx: bool) -> int:
    import duckdb

    if not db_path.exists():
        print(f"AMV 数据库不存在: {db_path}")
        return 1

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            """
            SELECT count(*), max(trade_date), min(trade_date)
            FROM active_market_value
            """
        ).fetchone()
        count, max_date, min_date = row
        print(f"db: {db_path}")
        print(f"rows: {count}")
        print(f"date range: {min_date} -> {max_date}")

        latest = conn.execute(
            """
            SELECT trade_date, amv_close, chg_abs_pct, source
            FROM active_market_value
            ORDER BY trade_date DESC
            LIMIT 3
            """
        ).fetchall()
        print("latest rows:")
        for item in latest:
            print(f"  {item[0]} close={item[1]} chg={item[2]}% source={item[3]}")

        if max_date is None:
            return 0

        next_day = max_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        print(f"next weekday after AMV max: {next_day}")

        if compare_tdx:
            from utils.data_source import DEFAULT_TDX_DB

            if not DEFAULT_TDX_DB.exists():
                print(f"TDX 数据库不存在, 跳过对比: {DEFAULT_TDX_DB}")
            else:
                tdx_max = duckdb.connect(str(DEFAULT_TDX_DB), read_only=True).execute(
                    "SELECT max(date) FROM v_stock_qfq"
                ).fetchone()[0]
                print(f"TDX latest trading day: {tdx_max}")
                if tdx_max and max_date and tdx_max > max_date:
                    gap_days = (tdx_max - max_date).days
                    print(
                        f"gap: AMV 落后 TDX {gap_days} 日历日 "
                        f"(missing through {tdx_max} if each gap day is a trading day)"
                    )
    finally:
        conn.close()
    return 0


def upsert_manual_row(
    *,
    db_path: Path,
    table: str,
    row: dict[str, Any],
    quality_flags: str,
    dry_run: bool,
    max_error_pp: float,
    allow_qc_errors: bool,
) -> None:
    import duckdb

    trade_date = row["trade_date"]
    if not isinstance(trade_date, date):
        raise TypeError(f"trade_date 类型异常: {trade_date!r}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    preview = {
        "trade_date": str(trade_date),
        "amv_open": row["open"],
        "amv_high": row["high"],
        "amv_low": row["low"],
        "amv_close": row["close"],
        "chg_abs_pct": row["chg_pct"],
        "volume_100m": row["volume"],
        "amount_100m": row["amount"],
        "position_100m": row["position"],
        "turnover_pct": row["turnover"],
        "amplitude_pct": row["amplitude"],
        "source": MANUAL_SOURCE,
        "quality_flags": quality_flags or None,
    }
    print("parsed row:")
    for key, value in preview.items():
        print(f"  {key}: {value}")

    if dry_run:
        print("dry-run: 未写入 DuckDB")
        return

    conn = duckdb.connect(str(db_path))
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(create_table_sql(table))
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE amv_stage AS
            SELECT
                ?::DATE AS trade_date,
                ?::DOUBLE AS amv_open,
                ?::DOUBLE AS amv_high,
                ?::DOUBLE AS amv_low,
                ?::DOUBLE AS amv_close,
                ?::DOUBLE AS chg_abs_pct,
                ?::DOUBLE AS volume_100m,
                ?::DOUBLE AS amount_100m,
                ?::DOUBLE AS position_100m,
                ?::DOUBLE AS turnover_pct,
                ?::DOUBLE AS amplitude_pct,
                ?::TEXT AS source,
                NULL::INTEGER AS source_seq,
                ?::TEXT AS source_filename,
                NULL::TIMESTAMP AS source_captured_at,
                ?::TEXT AS weekday_text,
                NULL::DOUBLE AS ocr_min_confidence,
                ?::TEXT AS quality_flags,
                ?::TEXT AS raw_ocr_text,
                current_timestamp AS ingested_at
            """,
            [
                trade_date,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["chg_pct"],
                row["volume"],
                row["amount"],
                row["position"],
                row["turnover"],
                row["amplitude"],
                MANUAL_SOURCE,
                f"manual:{trade_date.isoformat()}",
                row.get("weekday"),
                quality_flags or None,
                row.get("ocr_text"),
            ],
        )
        validate_stage(conn)
        conn.execute(upsert_sql(table))
        conn.execute(create_qc_view_sql(table))
        qc_errors = qc_error_count(conn, table, max_error_pp)
        if qc_errors and not allow_qc_errors:
            raise ValueError(
                f"{table}_qc 发现 {qc_errors} 行超过 {max_error_pp}pp 误差; "
                "请核对 readout 或加 --allow-qc-errors"
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        summary = conn.execute(
            f"""
            SELECT count(*), min(trade_date), max(trade_date)
            FROM {quote_ident(table)}
            """
        ).fetchone()
        qc_row = conn.execute(
            f"""
            SELECT chg_abs_pct_err, amplitude_pct_err
            FROM {quote_ident(f'{table}_qc')}
            WHERE trade_date = ?
            """,
            [trade_date],
        ).fetchone()
        conn.close()

    print(f"wrote: {db_path} ({table})")
    print(f"rows: {summary[0]}  date: {summary[1]} -> {summary[2]}")
    if qc_row and qc_row[0] is not None:
        print(f"qc err (chg/amplitude pp): {qc_row[0]:.4f} / {qc_row[1]:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="人工补录指南针活跃市值到 DuckDB (Windows / Mac, 不写 parquet)"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="active_market_value.duckdb")
    parser.add_argument("--table", default="active_market_value", help="目标表名")
    parser.add_argument("--text-file", type=Path, help="readout 文本文件")
    parser.add_argument("--text", help="readout 文本 (单行用 \\n 分隔)")
    parser.add_argument(
        "--show-gap",
        action="store_true",
        help="查看 AMV 最新日期, 与 TDX 最大交易日对比",
    )
    parser.add_argument("--no-tdx", action="store_true", help="--show-gap 时不对比 TDX")
    parser.add_argument("--dry-run", action="store_true", help="只解析校验, 不写库")
    parser.add_argument("--max-error-pp", type=float, default=0.08, help="QC 允许误差 (百分点)")
    parser.add_argument("--allow-qc-errors", action="store_true", help="QC 超差仍写入")
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使存在 price_range / missing 也写入 (quality_flags 会保留原因)",
    )
    args = parser.parse_args()

    if args.show_gap:
        return show_gap(args.db.resolve(), compare_tdx=not args.no_tdx)

    try:
        text = read_input_text(args)
        db_path = args.db.resolve()
        row = parse_readout_text(text)
        core_issues = validate_parsed_row(row, core_only=True)
        if core_issues and not args.force:
            raise ValueError(f"核心字段未齐: {';'.join(core_issues)}")

        row, inferred = fill_missing_fields(row, db_path=db_path, table=args.table)
        if inferred:
            print("auto-fill:")
            for note in inferred:
                print(f"  {note}")

        issues = validate_parsed_row(row, core_only=False)
        quality_parts = [*core_issues, *issues, *inferred]
        quality_flags = ";".join(part for part in quality_parts if part)
        if issues and not args.force:
            raise ValueError(f"校验未通过: {';'.join(issues)}")
        upsert_manual_row(
            db_path=db_path,
            table=args.table,
            row=row,
            quality_flags=quality_flags,
            dry_run=args.dry_run,
            max_error_pp=args.max_error_pp,
            allow_qc_errors=args.allow_qc_errors,
        )
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
