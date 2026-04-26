from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


FIELD_SPECS = {
    "开": ("open", "price"),
    "高": ("high", "price"),
    "低": ("low", "price"),
    "收": ("close", "price"),
    "幅": ("chg_pct", "pct"),
    "量": ("volume", "yi"),
    "额": ("amount", "yi"),
    "盘": ("position", "yi"),
    "率": ("turnover", "pct"),
    "振": ("amplitude", "pct"),
}

REQUIRED_FIELDS = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "chg_pct",
    "volume",
    "amount",
    "position",
    "turnover",
    "amplitude",
]


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None
    box: Any = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return (
        text.replace("：", ":")
        .replace("﹕", ":")
        .replace("％", "%")
        .replace("，", ".")
        .replace("。", ".")
        .replace(",", "")
        .replace(" ", "")
        .strip()
    )


def seq_from_path(path: Path) -> int:
    match = re.search(r"seq_(\d+)", path.stem)
    if not match:
        raise ValueError(f"无法从文件名解析 seq: {path.name}")
    return int(match.group(1))


def box_top(box: Any) -> float:
    try:
        if box and isinstance(box[0], (list, tuple)):
            return min(float(point[1]) for point in box)
        if box and len(box) >= 4:
            return float(box[1])
    except Exception:
        return 0.0
    return 0.0


def box_left(box: Any) -> float:
    try:
        if box and isinstance(box[0], (list, tuple)):
            return min(float(point[0]) for point in box)
        if box and len(box) >= 4:
            return float(box[0])
    except Exception:
        return 0.0
    return 0.0


def build_ocr():
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(lang="ch", use_angle_cls=False)


def result_to_jsonable(result: Any) -> Any:
    if hasattr(result, "json"):
        return result.json
    if isinstance(result, list):
        return [result_to_jsonable(item) for item in result]
    return result


def flatten_ocr_lines(payload: Any) -> list[OCRLine]:
    lines: list[OCRLine] = []

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if hasattr(obj, "json"):
            walk(obj.json)
            return
        if isinstance(obj, dict):
            res = obj.get("res", obj)
            if isinstance(res, dict):
                texts = res.get("rec_texts") or res.get("texts")
                scores = res.get("rec_scores") or res.get("scores") or []
                boxes = res.get("rec_boxes") or res.get("dt_polys") or res.get("rec_polys") or []
                if texts:
                    for idx, text in enumerate(texts):
                        score = scores[idx] if idx < len(scores) else None
                        box = boxes[idx] if idx < len(boxes) else None
                        lines.append(OCRLine(str(text), float(score) if score is not None else None, box))
                    return
                if "rec_text" in res:
                    lines.append(
                        OCRLine(
                            str(res["rec_text"]),
                            float(res["rec_score"]) if res.get("rec_score") is not None else None,
                            res.get("rec_box"),
                        )
                    )
                    return
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, list):
            if len(obj) == 2 and isinstance(obj[1], (list, tuple)) and obj[1] and isinstance(obj[1][0], str):
                text = obj[1][0]
                score = obj[1][1] if len(obj[1]) > 1 else None
                lines.append(OCRLine(str(text), float(score) if score is not None else None, obj[0]))
                return
            for item in obj:
                walk(item)

    walk(payload)
    return sorted(lines, key=lambda line: (box_top(line.box), box_left(line.box)))


def ocr_image(ocr: Any, image_path: Path) -> tuple[list[OCRLine], Any]:
    if hasattr(ocr, "predict"):
        try:
            result = ocr.predict(input=str(image_path), batch_size=1)
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            result = ocr.predict(input=str(image_path))
    else:
        result = ocr.ocr(str(image_path), cls=False)
    jsonable = result_to_jsonable(result)
    return flatten_ocr_lines(result), jsonable


def parse_lines(lines: list[OCRLine]) -> dict[str, Any]:
    texts = [normalize_text(line.text) for line in lines if normalize_text(line.text)]
    joined = "\n".join(texts)
    compact = "".join(texts)

    row: dict[str, Any] = {"ocr_text": joined}
    confidences = [line.confidence for line in lines if line.confidence is not None]
    row["ocr_min_confidence"] = min(confidences) if confidences else None

    date_match = re.search(r"((?:19|20)\d{6})(周[一二三四五六日天])?", compact)
    if date_match:
        raw_date = date_match.group(1)
        row["trade_date"] = datetime.strptime(raw_date, "%Y%m%d").date()
        row["weekday"] = date_match.group(2)

    for label, (column, _kind) in FIELD_SPECS.items():
        match = re.search(rf"{label}:?([-+]?\d+(?:\.\d+)?)", compact)
        if match:
            row[column] = float(match.group(1))

    return row


def validate_rows(df: Any, confidence_threshold: float) -> Any:
    import polars as pl

    df = df.with_columns(pl.lit("").alias("review_reason"))

    missing_expr = pl.concat_str(
        [
            pl.when(pl.col(col).is_null()).then(pl.lit(col)).otherwise(pl.lit(""))
            for col in REQUIRED_FIELDS
        ],
        separator=",",
    ).str.replace_all(r"(^,+|,+$)", "")

    df = df.with_columns(
        pl.when(missing_expr != "")
        .then(pl.concat_str([pl.lit("missing:"), missing_expr]))
        .otherwise(pl.col("review_reason"))
        .alias("review_reason")
    )

    price_bad = (
        pl.col("low").is_not_null()
        & pl.col("open").is_not_null()
        & pl.col("close").is_not_null()
        & pl.col("high").is_not_null()
        & ~((pl.col("low") <= pl.col("open")) & (pl.col("open") <= pl.col("high")) & (pl.col("low") <= pl.col("close")) & (pl.col("close") <= pl.col("high")))
    )
    low_conf = pl.col("ocr_min_confidence").is_not_null() & (pl.col("ocr_min_confidence") < confidence_threshold)

    return df.with_columns(
        pl.when(price_bad)
        .then(pl.concat_str([pl.col("review_reason"), pl.lit(";price_range")]))
        .when(low_conf)
        .then(pl.concat_str([pl.col("review_reason"), pl.lit(";low_confidence")]))
        .otherwise(pl.col("review_reason"))
        .alias("review_reason")
    )


def load_manifest(shots_dir: Path) -> dict[int, dict[str, Any]]:
    manifest_path = shots_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows[int(item["seq"])] = item
    return rows


def write_duckdb(parquet_path: Path, duckdb_path: Path, replace: bool) -> None:
    import duckdb

    conn = duckdb.connect(str(duckdb_path))
    if replace:
        conn.execute(
            "CREATE OR REPLACE TABLE active_market_value AS SELECT * FROM read_parquet(?)",
            [str(parquet_path)],
        )
        return
    conn.execute(
        "CREATE TABLE IF NOT EXISTS active_market_value AS "
        "SELECT * FROM read_parquet(?) WHERE false",
        [str(parquet_path)],
    )
    conn.execute(
        "DELETE FROM active_market_value USING read_parquet(?) src "
        "WHERE active_market_value.trade_date = src.trade_date",
        [str(parquet_path)],
    )
    conn.execute("INSERT INTO active_market_value SELECT * FROM read_parquet(?)", [str(parquet_path)])


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser(description="解析指南针 0AMV 活跃市值截图为结构化数据")
    parser.add_argument("--input", required=True, type=Path, help="截图目录, 例如 rpa_capture/shots")
    parser.add_argument("--output", default=Path("data/active_market_value"), type=Path, help="输出目录")
    parser.add_argument("--duckdb", type=Path, help="可选: 写入 DuckDB 路径")
    parser.add_argument("--replace-duckdb", action="store_true", help="覆盖 DuckDB active_market_value 表")
    parser.add_argument("--raw-json", action="store_true", help="保存每张图的 OCR 原始 JSON")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    args = parser.parse_args()

    shots_dir = args.input
    image_paths = sorted(shots_dir.glob("seq_*.png"), key=seq_from_path)
    if not image_paths:
        raise FileNotFoundError(f"未找到 seq_*.png: {shots_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "raw_ocr"
    if args.raw_json:
        raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(shots_dir)
    ocr = build_ocr()
    rows: list[dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        seq = seq_from_path(image_path)
        lines, raw_payload = ocr_image(ocr, image_path)
        row = parse_lines(lines)
        row.update(
            {
                "seq": seq,
                "filename": image_path.name,
                "captured_at": manifest.get(seq, {}).get("captured_at"),
            }
        )
        rows.append(row)
        if args.raw_json:
            (raw_dir / f"{image_path.stem}.json").write_text(
                json.dumps(raw_payload, ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )
        if idx % 100 == 0 or idx == len(image_paths):
            print(f"parsed {idx}/{len(image_paths)}")

    df = pl.DataFrame(rows).sort("seq")
    df = validate_rows(df, args.confidence_threshold)

    parquet_path = args.output / "active_market_value.parquet"
    csv_path = args.output / "active_market_value.csv"
    review_path = args.output / "active_market_value_review.csv"

    df.write_parquet(parquet_path)
    df.write_csv(csv_path)
    df.filter(pl.col("review_reason") != "").write_csv(review_path)

    if args.duckdb:
        write_duckdb(parquet_path, args.duckdb, args.replace_duckdb)

    print(f"rows: {df.height}")
    print(f"review rows: {df.filter(pl.col('review_reason') != '').height}")
    print(f"wrote: {parquet_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {review_path}")


if __name__ == "__main__":
    main()
