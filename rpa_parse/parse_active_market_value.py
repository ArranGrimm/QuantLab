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

FIELD_ORDER = ["开", "高", "低", "收", "幅", "量", "额", "盘", "率", "振"]

FIELD_ALIASES = {
    "汁": "开",
    "升": "开",
    "帽": "幅",
    "日": "幅",
    "$": "率",
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
        .replace("（", "")
        .replace("）", "")
        .replace("(", "")
        .replace(")", "")
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


def _call_optional_float(obj: Any, method_name: str) -> float | None:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return float(method())
    except Exception:
        return None


def _configure_vision_request(request: Any, languages: list[str]) -> None:
    # Vision 的 language correction 会试图改写短字段/数字, 对固定 readout 反而有害.
    if hasattr(request, "setRecognitionLevel_"):
        request.setRecognitionLevel_(0)  # VNRequestTextRecognitionLevelAccurate
    if hasattr(request, "setUsesLanguageCorrection_"):
        request.setUsesLanguageCorrection_(False)
    if languages and hasattr(request, "setRecognitionLanguages_"):
        request.setRecognitionLanguages_(languages)
    if hasattr(request, "setCustomWords_"):
        request.setCustomWords_(
            [
                "开",
                "高",
                "低",
                "收",
                "幅",
                "量",
                "额",
                "盘",
                "率",
                "振",
                "亿",
                "周一",
                "周二",
                "周三",
                "周四",
                "周五",
            ]
        )


def ocr_image_with_vision(image_path: Path, languages: list[str]) -> tuple[list[OCRLine], Any]:
    try:
        import Quartz
        import Vision
        import objc
        from Cocoa import NSURL
        from Foundation import NSDictionary
    except ImportError as exc:
        raise RuntimeError(
            "macOS Vision OCR 需要 PyObjC 依赖: "
            "uv add pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-Cocoa"
        ) from exc

    with objc.autorelease_pool():
        input_url = NSURL.fileURLWithPath_(str(image_path.resolve()))
        input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)
        if input_image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        options = NSDictionary.dictionaryWithDictionary_({})
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(input_image, options)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        _configure_vision_request(request, languages)

        ret = handler.performRequests_error_([request], None)
        if isinstance(ret, tuple):
            ok, error = ret
        else:
            ok, error = bool(ret), None
        if not ok or error is not None:
            raise RuntimeError(f"Vision OCR 失败: {error}")

        lines: list[OCRLine] = []
        raw_lines: list[dict[str, Any]] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            recognized = candidates[0]
            text = str(recognized.string())
            confidence = _call_optional_float(recognized, "confidence") or _call_optional_float(
                observation,
                "confidence",
            )
            bbox = observation.boundingBox()
            x = float(bbox.origin.x)
            y = float(bbox.origin.y)
            width = float(bbox.size.width)
            height = float(bbox.size.height)
            # Vision 的 y 原点在左下角; 转成 top-origin, 便于按阅读顺序排序.
            box = [x, 1.0 - y - height, width, height]
            lines.append(OCRLine(text=text, confidence=confidence, box=box))
            raw_lines.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                }
            )

    lines = sorted(lines, key=lambda line: (box_top(line.box), box_left(line.box)))
    raw_lines = sorted(raw_lines, key=lambda line: (box_top(line["box"]), box_left(line["box"])))
    return lines, {"backend": "vision", "lines": raw_lines}


def build_text_rows(lines: list[OCRLine], row_tolerance: float = 0.035) -> list[str]:
    if not any(line.box is not None for line in lines):
        return [normalize_text(line.text) for line in lines if normalize_text(line.text)]

    grouped: list[list[OCRLine]] = []
    for line in sorted(lines, key=lambda item: (box_top(item.box), box_left(item.box))):
        text = normalize_text(line.text)
        if not text:
            continue
        top = box_top(line.box)
        if not grouped or abs(top - box_top(grouped[-1][0].box)) > row_tolerance:
            grouped.append([line])
        else:
            grouped[-1].append(line)

    rows = []
    for group in grouped:
        fragments = [normalize_text(line.text) for line in sorted(group, key=lambda item: box_left(item.box))]
        row = ""
        for fragment in fragments:
            if (
                ":" in row
                and "." not in row.split(":", 1)[1]
                and re.match(r"^\d\.\d+%?亿?$", fragment)
            ):
                row += fragment[1:]
            elif (
                ":" in row
                and "." not in row.split(":", 1)[1]
                and re.match(r"^\d+%?亿?$", fragment)
            ):
                row += f".{fragment}"
            else:
                row += fragment
        if row:
            rows.append(row)
    return rows


def parse_lines(lines: list[OCRLine]) -> dict[str, Any]:
    texts = build_text_rows(lines)
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

    field_index = 0
    pending_label: str | None = None
    pending_value = ""
    for text in texts:
        if re.search(r"(?:19|20)\d{6}", text):
            continue
        if ":" in text:
            raw_label, raw_value_text = text.split(":", 1)
        else:
            raw_label, raw_value_text = "", text
        value_match = re.match(r"^([-+]?\d*(?:\.\d*)?)%?亿?$", raw_value_text)
        if value_match:
            raw_value = value_match.group(1)
            label = FIELD_ALIASES.get(raw_label, raw_label)
            if label not in FIELD_SPECS and raw_value not in {"", ".", "+", "-"} and field_index < len(FIELD_ORDER):
                label = FIELD_ORDER[field_index]
            if label in FIELD_SPECS:
                column, _kind = FIELD_SPECS[label]
                if raw_value not in {"", ".", "+", "-"}:
                    if label == "额" and raw_label.isdigit() and raw_value.startswith("0"):
                        raw_value = f"{raw_label}{raw_value}"
                    row[column] = float(raw_value)
                    pending_label = None
                    pending_value = ""
                    field_index = max(field_index + 1, FIELD_ORDER.index(label) + 1)
                else:
                    pending_label = label
                    pending_value = raw_value
            continue
        if pending_label is not None:
            tail_match = re.match(r"^(\d+(?:\.\d+)?)%?亿?$", text)
            if tail_match:
                column, _kind = FIELD_SPECS[pending_label]
                row[column] = float(f"{pending_value}{tail_match.group(1)}")
                field_index = max(field_index + 1, FIELD_ORDER.index(pending_label) + 1)
                pending_label = None
                pending_value = ""

    # Fallback: 如果 Vision 把字段和值合并成非标准片段, 再用全量 compact 文本兜底.
    for label, (column, _kind) in FIELD_SPECS.items():
        if column in row:
            continue
        aliases = [label, *[alias for alias, canonical in FIELD_ALIASES.items() if canonical == label]]
        alias_pattern = "|".join(re.escape(alias) for alias in aliases)
        match = re.search(rf"(?:{alias_pattern}):?([-+]?\d+(?:\.\d+)?)", compact)
        if match:
            row[column] = float(match.group(1))

    return row


def validate_rows(df: Any, confidence_threshold: float) -> Any:
    import polars as pl

    df = df.with_columns(pl.lit("").alias("review_reason"))

    def append_review(reason: str, condition: Any) -> Any:
        return df.with_columns(
            pl.when(condition)
            .then(
                pl.when(pl.col("review_reason") == "")
                .then(pl.lit(reason))
                .otherwise(pl.concat_str([pl.col("review_reason"), pl.lit(f";{reason}")]))
            )
            .otherwise(pl.col("review_reason"))
            .alias("review_reason")
        )

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
    duplicate_trade_date = pl.col("trade_date").is_not_null() & pl.col("trade_date").is_duplicated()

    df = append_review("price_range", price_bad)
    df = append_review("low_confidence", low_conf)
    return append_review("duplicate_trade_date", duplicate_trade_date)


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


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser(description="解析指南针 0AMV 活跃市值截图为结构化数据")
    parser.add_argument("--input", required=True, type=Path, help="截图目录, 例如 rpa_capture/shots")
    parser.add_argument("--output", default=Path("data/active_market_value"), type=Path, help="输出目录")
    parser.add_argument("--raw-json", action="store_true", help="保存每张图的 OCR 原始 JSON")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量模式: 读取已有 active_market_value.parquet, 只 OCR 尚未入表的 seq",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="只解析前 N 张, 用于首次冒烟测试")
    parser.add_argument("--progress-every", type=int, default=10, help="每 N 张打印一次进度")
    parser.add_argument(
        "--vision-languages",
        nargs="*",
        default=["zh-Hans", "en-US"],
        help="传给 macOS Vision 的识别语言, 默认 zh-Hans en-US",
    )
    args = parser.parse_args()

    shots_dir = args.input
    image_paths = sorted(shots_dir.glob("seq_*.png"), key=seq_from_path)
    if not image_paths:
        raise FileNotFoundError(f"未找到 seq_*.png: {shots_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output / "active_market_value.parquet"
    csv_path = args.output / "active_market_value.csv"
    review_path = args.output / "active_market_value_review.csv"
    raw_dir = args.output / "raw_ocr"
    if args.raw_json:
        raw_dir.mkdir(parents=True, exist_ok=True)

    existing_df = None
    existing_seqs: set[int] = set()
    if args.incremental and parquet_path.exists():
        existing_df = pl.read_parquet(parquet_path)
        existing_seqs = set(existing_df["seq"].drop_nulls().cast(pl.Int64).to_list())
        image_paths = [path for path in image_paths if seq_from_path(path) not in existing_seqs]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    manifest = load_manifest(shots_dir)
    print(f"input: {shots_dir}", flush=True)
    if args.incremental:
        print(f"existing rows: {0 if existing_df is None else existing_df.height:,}", flush=True)
        print(f"new images: {len(image_paths):,}", flush=True)
    else:
        print(f"images: {len(image_paths):,}", flush=True)
    print(f"ocr backend: macOS Vision ({', '.join(args.vision_languages)})", flush=True)
    rows: list[dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        if idx == 1 or idx % args.progress_every == 0:
            print(f"parsing {idx}/{len(image_paths)}: {image_path.name}", flush=True)
        seq = seq_from_path(image_path)
        lines, raw_payload = ocr_image_with_vision(image_path, args.vision_languages)
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
        if idx % args.progress_every == 0 or idx == len(image_paths):
            print(f"parsed {idx}/{len(image_paths)}", flush=True)

    if rows:
        new_df = pl.DataFrame(rows)
    else:
        new_df = None

    if existing_df is not None and new_df is not None:
        df = pl.concat([existing_df, new_df], how="diagonal_relaxed")
    elif existing_df is not None:
        df = existing_df
    elif new_df is not None:
        df = new_df
    else:
        raise RuntimeError("没有可写入的数据: 既没有新增 OCR 行, 也没有已有 parquet")

    df = (
        df.sort("seq")
        .unique(subset=["seq"], keep="last", maintain_order=True)
        .sort("seq")
    )
    df = validate_rows(df, args.confidence_threshold)

    df.write_parquet(parquet_path)
    df.write_csv(csv_path)
    df.filter(pl.col("review_reason") != "").write_csv(review_path)

    print(f"rows: {df.height}")
    if args.incremental:
        print(f"new rows: {0 if new_df is None else new_df.height}")
    print(f"review rows: {df.filter(pl.col('review_reason') != '').height}")
    print(f"wrote: {parquet_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {review_path}")


if __name__ == "__main__":
    main()
