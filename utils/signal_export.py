"""
Signal Export Tool - Export stock signals for Rust backtesting

支持两种策略:
1. B1 超跌反转 — 事件驱动信号 (b1_signal)
2. 截面轮动模型 — 日频截面打分 (score + rank)

设计理念：
- 导出完整数据, Rust 端负责回测逻辑
- B1: 标记信号日, Rust 做买入/止损/止盈
- 截面模型: 导出每日全 universe 打分, Rust 做 Top-N 选股/持仓管理
"""
import polars as pl
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess


def _json_default(value):
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
        f.write("\n")


def _sidecar_meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def _slug_token(text: str) -> str:
    cleaned = []
    for ch in str(text).lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "na"


def _float_token(value: float, precision: int = 4) -> str:
    s = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def _timestamp_ms_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _rel_path(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace("\\", "/")


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_feature_hash(features: list[str]) -> str:
    joined = "\n".join(sorted(features))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def build_rotation_train_run_id(label: str, model_name: str, train_timestamp_token: str, feature_hash: str) -> str:
    return f"rot_{_slug_token(label)}_{_slug_token(model_name)}_{train_timestamp_token}_{feature_hash}"


def build_b1_train_run_id(
    label: str,
    seed_col: str,
    model_name: str,
    train_timestamp_token: str,
    feature_hash: str,
) -> str:
    return (
        f"b1_{_slug_token(seed_col)}_{_slug_token(label)}_"
        f"{_slug_token(model_name)}_{train_timestamp_token}_{feature_hash}"
    )


def export_for_rust(
    df_full: pl.LazyFrame,
    output_path: str = "data/signals/market_data.parquet",
    loose_periods: list = None,
    start_date: str = None,
    extra_sort_cols: list = None,
    raw_scores: pl.DataFrame | None = None,
    artifact_metadata: dict | None = None,
    artifact_root: str = "artifacts/b1",
    write_latest_alias: bool = False,
) -> str:
    """
    Export complete market data with B1 signals for Rust backtesting

    Args:
        df_full: LazyFrame from calc_b1_factors_opt (contains ALL rows, not just signals)
        output_path: Output parquet file path
        loose_periods: Active period list, e.g. [("2025-04-09", "2025-09-04"), ...]
        stop_loss_pct: Stop loss percentage for calculating stop price
        start_date: Only export data >= this date, e.g. "2025-01-01"
        extra_sort_cols: Extra columns to export for sorting, e.g. ["Bias_C_WL", "J"]

    Returns:
        Exported file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Filter by start_date if provided
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        df_full = df_full.filter(pl.col("date") >= pl.lit(start))
        print(f"Filtering data >= {start_date}")

    # Build is_loose marker
    if loose_periods:
        loose_exprs = []
        for s_str, e_str in loose_periods:
            s = datetime.strptime(s_str, "%Y-%m-%d").date()
            e = datetime.strptime(e_str, "%Y-%m-%d").date()
            loose_exprs.append(pl.col("date").is_between(pl.lit(s), pl.lit(e)))
        is_loose_expr = pl.any_horizontal(*loose_exprs)
    else:
        is_loose_expr = pl.lit(True)

    print("Processing data...")
    df_export = (
        df_full.sort(["code", "date"])
        .with_columns([
            pl.col("volume").rolling_mean(40).over("code").alias("vol_40_mean"),
        ])
        .with_columns(
            [
                # 标记：昨天收盘价
                pl.col("close_adj").shift(1).over("code").fill_null(pl.col("close_adj")).alias("pre_close_adj"),
                # 标记：昨天是否是信号 (T日信号 → T+1日买入)
                pl.col("b1_signal").shift(1).over("code").fill_null(False).alias("pre_b1_signal"),
                # 标记：是否在活跃期
                is_loose_expr.alias("is_loose"),
                # 计算 vol_ratio (用于排序的默认选项)
                (pl.col("volume") / pl.col("vol_40_mean")).alias("vol_ratio")
            ]
        )
    )

    # Select columns needed by Rust
    required_cols = [
        "code",
        "date",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "pre_close_adj",
        "volume",
        "WL",
        "YL",
        "J",
        "b1_signal",
        "pre_b1_signal",
        "is_loose",
        "vol_ratio"
    ]
    
    # Add extra sort columns if specified
    if extra_sort_cols:
        for col in extra_sort_cols:
            if col not in required_cols:
                required_cols.append(col)

    # Collect and select
    df_collected = df_export.collect()
    available_cols = [c for c in required_cols if c in df_collected.columns]

    if len(available_cols) < len(required_cols):
        missing = set(required_cols) - set(available_cols)
        print(f"Warning: Missing columns: {missing}")

    df_final = df_collected.select(available_cols)

    total_rows = df_final.height
    signal_rows = df_final.filter(pl.col("b1_signal")).height
    loose_signal_rows = df_final.filter(pl.col("b1_signal") & pl.col("is_loose")).height
    unique_codes = df_final.select(pl.col("code").n_unique()).item()
    unique_dates = df_final.select(pl.col("date").n_unique()).item()
    date_range = (
        df_final.select(pl.col("date").min()).item(),
        df_final.select(pl.col("date").max()).item(),
    )

    if artifact_metadata:
        train_run_id = artifact_metadata.get("train_run_id")
        if not train_run_id:
            raise ValueError("artifact_metadata must contain train_run_id")

        artifact_root_path = Path(artifact_root).resolve()
        train_run_dir = (artifact_root_path / train_run_id).resolve()
        signal_id = artifact_metadata.get("signal_id") or _timestamp_ms_token()
        signal_run_id = f"{train_run_id}_{signal_id}"
        signal_dir = train_run_dir / "signals" / signal_id
        signal_path = signal_dir / "signal.parquet"
        train_meta_path = train_run_dir / "train.meta.json"
        signal_meta_path = signal_dir / "signal.meta.json"
        raw_scores_path = train_run_dir / "raw_scores.parquet"
        signals_index_path = train_run_dir / "signals.jsonl"
        backtest_index_path = train_run_dir / "backtest.jsonl"
        latest_alias_path = None

        train_run_dir.mkdir(parents=True, exist_ok=True)
        signal_dir.mkdir(parents=True, exist_ok=True)
        signal_path = signal_path.resolve()
        train_meta_path = train_meta_path.resolve()
        signal_meta_path = signal_meta_path.resolve()
        raw_scores_path = raw_scores_path.resolve()
        signals_index_path = signals_index_path.resolve()
        backtest_index_path = backtest_index_path.resolve()

        if write_latest_alias:
            latest_alias_path = Path(output_path).resolve()
            latest_alias_path.parent.mkdir(parents=True, exist_ok=True)

        if raw_scores is not None and not raw_scores_path.exists():
            raw_scores.write_parquet(raw_scores_path)

        label = artifact_metadata.get("label")
        model_name = artifact_metadata.get("model_name")
        feature_mode = (
            artifact_metadata.get("feature_mode")
            or artifact_metadata.get("feature_set_name")
        )
        features = list(artifact_metadata.get("features", []))
        feature_hash = artifact_metadata.get("feature_hash") or build_feature_hash(features)
        signal_source = artifact_metadata.get("signal_source")
        seed_col = artifact_metadata.get("seed_col")
        sort_field = artifact_metadata.get("sort_field")
        sort_ascending = artifact_metadata.get("sort_ascending")

        train_meta = {
            "artifact_version": 2,
            "artifact_kind": "train_run",
            "strategy": "b1",
            "train_run_id": train_run_id,
            "label": label,
            "model_name": model_name,
            "feature_mode": feature_mode,
            "feature_set_name": artifact_metadata.get("feature_set_name"),
            "feature_hash": feature_hash,
            "feature_count": len(features),
            "features": features,
            "trained_at": artifact_metadata.get("trained_at"),
            "train_timestamp_token": artifact_metadata.get("train_timestamp_token"),
            "git_commit": artifact_metadata.get("git_commit"),
            "notebook": artifact_metadata.get("notebook"),
            "model_params": artifact_metadata.get("model_params"),
            "train_window": artifact_metadata.get("train_window"),
            "retrain_freq": artifact_metadata.get("retrain_freq"),
            "seed_col": seed_col,
            "use_bull_only": artifact_metadata.get("use_bull_only"),
            "signal_source": signal_source,
            "sort_field": sort_field,
            "sort_ascending": sort_ascending,
            "universe": artifact_metadata.get("universe"),
            "raw_scores_path": "raw_scores.parquet",
            "signals_index_path": "signals.jsonl",
            "backtest_index_path": "backtest.jsonl",
        }
        signal_meta = {
            "artifact_version": 2,
            "artifact_kind": "signal_export",
            "strategy": "b1",
            "train_run_id": train_run_id,
            "signal_id": signal_id,
            "signal_run_id": signal_run_id,
            "label": label,
            "model_name": model_name,
            "feature_mode": feature_mode,
            "feature_set_name": artifact_metadata.get("feature_set_name"),
            "feature_hash": feature_hash,
            "feature_count": len(features),
            "features": features,
            "seed_col": seed_col,
            "use_bull_only": artifact_metadata.get("use_bull_only"),
            "signal_source": signal_source,
            "sort_field": sort_field,
            "sort_ascending": sort_ascending,
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_timestamp": signal_id,
            "export_ema_alpha": artifact_metadata.get("export_ema_alpha"),
            "signal_rows": signal_rows,
            "trading_days": unique_dates,
            "unique_codes": unique_codes,
            "date_min": str(date_range[0]),
            "date_max": str(date_range[1]),
            "signal_path": "signal.parquet",
            "canonical_signal_path": "signal.parquet",
            "backtests_dir": "backtests",
            "latest_alias_path": _rel_path(latest_alias_path, signal_dir) if latest_alias_path else None,
            "train_meta_path": _rel_path(train_meta_path, signal_dir),
            "raw_scores_path": _rel_path(raw_scores_path, signal_dir),
            "signals_index_path": _rel_path(signals_index_path, signal_dir),
            "backtest_index_path": _rel_path(backtest_index_path, signal_dir),
            "git_commit": artifact_metadata.get("git_commit"),
            "notebook": artifact_metadata.get("notebook"),
        }

        df_final.write_parquet(signal_path)
        _write_json(train_meta_path, train_meta)
        _write_json(signal_meta_path, signal_meta)
        _append_jsonl(
            signals_index_path,
            {
                "record_type": "signal_export",
                "exported_at": signal_meta["exported_at"],
                "strategy": "b1",
                "train_run_id": train_run_id,
                "signal_id": signal_id,
                "signal_run_id": signal_run_id,
                "label": label,
                "model_name": model_name,
                "feature_mode": feature_mode,
                "feature_set_name": artifact_metadata.get("feature_set_name"),
                "feature_hash": feature_hash,
                "feature_count": len(features),
                "seed_col": seed_col,
                "use_bull_only": artifact_metadata.get("use_bull_only"),
                "signal_source": signal_source,
                "sort_field": sort_field,
                "sort_ascending": sort_ascending,
                "signal_dir": _rel_path(signal_dir, train_run_dir),
                "signal_path": _rel_path(signal_path, train_run_dir),
                "signal_meta_path": _rel_path(signal_meta_path, train_run_dir),
                "train_meta_path": _rel_path(train_meta_path, train_run_dir),
                "git_commit": artifact_metadata.get("git_commit"),
            },
        )

        if latest_alias_path:
            df_final.write_parquet(latest_alias_path)
            latest_meta = dict(signal_meta)
            latest_meta["artifact_kind"] = "signal_alias"
            latest_meta["canonical_signal_meta_path"] = str(signal_meta_path)
            _write_json(_sidecar_meta_path(latest_alias_path), latest_meta)

        print("\n=== B1 Signals Export ===")
        print(f"Run ID: {signal_run_id}")
        print(f"Canonical File: {signal_path}")
        print(f"Train Meta: {train_meta_path}")
        print(f"Signal Meta: {signal_meta_path}")
        print(f"Signals Index: {signals_index_path}")
        print(f"Backtest Index: {backtest_index_path}")
        if latest_alias_path:
            print(f"Latest Alias: {latest_alias_path}")
        print(f"Total rows: {total_rows:,}")
        print(f"Trading days: {unique_dates}")
        print(f"Unique stocks: {unique_codes}")
        print(f"Date range: {date_range[0]} ~ {date_range[1]}")
        print(f"B1 signals: {signal_rows}")
        print(f"B1 signals in loose periods: {loose_signal_rows}")
        print("B1 旧回测入口已归档，当前不再提供 run_b1.bat 示例。")
        return str(signal_path)

    # Write to parquet
    df_final.write_parquet(output_file)

    print("\n=== Export Summary ===")
    print(f"File: {output_file}")
    print(f"Total rows: {total_rows:,}")
    print(f"Trading days: {unique_dates}")
    print(f"Unique stocks: {unique_codes}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"B1 signals: {signal_rows}")
    print(f"B1 signals in loose periods: {loose_signal_rows}")

    return str(output_file)


def export_rotation_scores(
    df_scores: pl.DataFrame,
    output_path: str | None = None,
    top_n: int = 20,
    raw_scores: pl.DataFrame | None = None,
    artifact_metadata: dict | None = None,
    artifact_root: str = "artifacts/rotation",
    write_latest_alias: bool = False,
) -> str:
    """
    Export cross-section rotation model scores for Rust backtesting.

    Python 端只负责打分, Rust 端负责:
      - 每日读取 Top-N 候选
      - 买入/卖出决策 (止损/止盈/排名退出)
      - 仓位管理、交易成本

    Args:
        df_scores: DataFrame, 必须包含:
            date, code, score, open_adj, high_adj, low_adj, close_adj
            可选: volume, market_cap_100m
        output_path: 输出 Parquet 路径
        top_n: 每日 Top-N 标记 (is_top_n 列, 供 Rust 参考)

    Returns:
        Exported file path

    Output schema:
        date, code, score, rank, is_top_n,
        open_adj, high_adj, low_adj, close_adj, pre_close_adj,
        [volume, market_cap_100m]
    """
    required = ["date", "code", "score", "open_adj", "high_adj", "low_adj", "close_adj"]
    missing = [c for c in required if c not in df_scores.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Processing rotation scores...")

    df_valid = df_scores.filter(
        pl.col("score").is_not_null() & pl.col("score").is_not_nan()
    )

    df_export = (
        df_valid
        .sort(["code", "date"])
        .with_columns(
            pl.col("close_adj").shift(1).over("code")
                .fill_null(pl.col("close_adj"))
                .alias("pre_close_adj"),
        )
        .with_columns(
            pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("date")
                .cast(pl.UInt16)
                .alias("rank"),
        )
        .with_columns(
            (pl.col("rank") <= top_n).alias("is_top_n"),
        )
    )

    out_cols = [
        "date", "code", "score", "rank", "is_top_n",
        "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    ]
    # 可选 passthrough 列: 上游若提供则带入 parquet, 不提供则跳过 (向后兼容).
    # is_bull_regime: 市场层 timing 信号, 供 Rust engine 的 require_bull_regime 开仓 gate 使用.
    for opt_col in ["volume", "market_cap_100m", "is_bull_regime"]:
        if opt_col in df_scores.columns:
            out_cols.append(opt_col)

    df_final = df_export.select(out_cols)

    total_rows = df_final.height
    unique_dates = df_final.select(pl.col("date").n_unique()).item()
    unique_codes = df_final.select(pl.col("code").n_unique()).item()
    top_n_rows = df_final.filter(pl.col("is_top_n")).height
    date_range = (
        df_final.select(pl.col("date").min()).item(),
        df_final.select(pl.col("date").max()).item(),
    )

    if artifact_metadata:
        train_run_id = artifact_metadata.get("train_run_id")
        if not train_run_id:
            raise ValueError("artifact_metadata must contain train_run_id")

        artifact_root_path = Path(artifact_root).resolve()
        train_run_dir = (artifact_root_path / train_run_id).resolve()
        signal_id = artifact_metadata.get("signal_id") or _timestamp_ms_token()
        signal_run_id = f"{train_run_id}_{signal_id}"
        signal_dir = train_run_dir / "signals" / signal_id
        signal_path = signal_dir / "signal.parquet"
        train_meta_path = train_run_dir / "train.meta.json"
        signal_meta_path = signal_dir / "signal.meta.json"
        raw_scores_path = train_run_dir / "raw_scores.parquet"
        signals_index_path = train_run_dir / "signals.jsonl"
        backtest_index_path = train_run_dir / "backtest.jsonl"
        latest_alias_path = None

        train_run_dir.mkdir(parents=True, exist_ok=True)
        signal_dir.mkdir(parents=True, exist_ok=True)
        signal_path = signal_path.resolve()
        train_meta_path = train_meta_path.resolve()
        signal_meta_path = signal_meta_path.resolve()
        raw_scores_path = raw_scores_path.resolve()
        signals_index_path = signals_index_path.resolve()
        backtest_index_path = backtest_index_path.resolve()

        if write_latest_alias:
            if not output_path:
                raise ValueError("output_path is required when write_latest_alias=True")
            latest_alias_path = Path(output_path).resolve()
            latest_alias_path.parent.mkdir(parents=True, exist_ok=True)

        if raw_scores is not None and not raw_scores_path.exists():
            raw_scores.write_parquet(raw_scores_path)

        label = artifact_metadata.get("label")
        model_name = artifact_metadata.get("model_name")
        feature_mode = artifact_metadata.get("feature_mode")
        normalize_mode = artifact_metadata.get("normalize_mode")
        features = list(artifact_metadata.get("features", []))
        feature_hash = artifact_metadata.get("feature_hash") or build_feature_hash(features)
        train_meta = {
            "artifact_version": 2,
            "artifact_kind": "train_run",
            "strategy": "rotation",
            "train_run_id": train_run_id,
            "label": label,
            "model_name": model_name,
            "feature_mode": feature_mode,
            "normalize_mode": normalize_mode,
            "feature_hash": feature_hash,
            "feature_count": len(features),
            "features": features,
            "trained_at": artifact_metadata.get("trained_at"),
            "train_timestamp_token": artifact_metadata.get("train_timestamp_token"),
            "git_commit": artifact_metadata.get("git_commit"),
            "notebook": artifact_metadata.get("notebook"),
            "model_params": artifact_metadata.get("model_params"),
            "train_window": artifact_metadata.get("train_window"),
            "retrain_freq": artifact_metadata.get("retrain_freq"),
            "train_regime_mode": artifact_metadata.get("train_regime_mode"),
            "amv_bull_sample_weight": artifact_metadata.get("amv_bull_sample_weight"),
            "amv_bull_trigger_pct": artifact_metadata.get("amv_bull_trigger_pct"),
            "amv_bull_lookback_days": artifact_metadata.get("amv_bull_lookback_days"),
            "amv_bear_trigger_1d_pct": artifact_metadata.get("amv_bear_trigger_1d_pct"),
            "amv_effective_lag_days": artifact_metadata.get("amv_effective_lag_days"),
            "universe": artifact_metadata.get("universe"),
            "raw_scores_path": "raw_scores.parquet",
            "signals_index_path": "signals.jsonl",
            "backtest_index_path": "backtest.jsonl",
        }
        signal_meta = {
            "artifact_version": 2,
            "artifact_kind": "signal_export",
            "strategy": "rotation",
            "train_run_id": train_run_id,
            "signal_id": signal_id,
            "signal_run_id": signal_run_id,
            "label": label,
            "model_name": model_name,
            "feature_mode": feature_mode,
            "normalize_mode": normalize_mode,
            "feature_hash": feature_hash,
            "feature_count": len(features),
            "features": features,
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_timestamp": signal_id,
            "export_ema_alpha": artifact_metadata.get("export_ema_alpha"),
            "top_n": top_n,
            "signal_rows": total_rows,
            "trading_days": unique_dates,
            "unique_codes": unique_codes,
            "date_min": str(date_range[0]),
            "date_max": str(date_range[1]),
            "bull_regime_source": artifact_metadata.get("bull_regime_source"),
            "bull_regime_rows_pct": artifact_metadata.get("bull_regime_rows_pct"),
            "amv_bull_trigger_pct": artifact_metadata.get("amv_bull_trigger_pct"),
            "amv_bull_lookback_days": artifact_metadata.get("amv_bull_lookback_days"),
            "amv_bear_trigger_1d_pct": artifact_metadata.get("amv_bear_trigger_1d_pct"),
            "amv_effective_lag_days": artifact_metadata.get("amv_effective_lag_days"),
            "signal_path": "signal.parquet",
            "canonical_signal_path": "signal.parquet",
            "backtests_dir": "backtests",
            "latest_alias_path": _rel_path(latest_alias_path, signal_dir) if latest_alias_path else None,
            "train_meta_path": _rel_path(train_meta_path, signal_dir),
            "raw_scores_path": _rel_path(raw_scores_path, signal_dir),
            "signals_index_path": _rel_path(signals_index_path, signal_dir),
            "backtest_index_path": _rel_path(backtest_index_path, signal_dir),
            "git_commit": artifact_metadata.get("git_commit"),
            "notebook": artifact_metadata.get("notebook"),
        }

        df_final.write_parquet(signal_path)
        _write_json(train_meta_path, train_meta)
        _write_json(signal_meta_path, signal_meta)
        _append_jsonl(
            signals_index_path,
            {
                "record_type": "signal_export",
                "exported_at": signal_meta["exported_at"],
                "strategy": "rotation",
                "train_run_id": train_run_id,
                "signal_id": signal_id,
                "signal_run_id": signal_run_id,
                "label": label,
                "model_name": model_name,
                "feature_mode": feature_mode,
                "normalize_mode": normalize_mode,
                "feature_hash": feature_hash,
                "feature_count": len(features),
                "export_ema_alpha": artifact_metadata.get("export_ema_alpha"),
                "top_n": top_n,
                "signal_dir": _rel_path(signal_dir, train_run_dir),
                "signal_path": _rel_path(signal_path, train_run_dir),
                "signal_meta_path": _rel_path(signal_meta_path, train_run_dir),
                "train_meta_path": _rel_path(train_meta_path, train_run_dir),
                "git_commit": artifact_metadata.get("git_commit"),
            },
        )

        if latest_alias_path:
            df_final.write_parquet(latest_alias_path)
            latest_meta = dict(signal_meta)
            latest_meta["artifact_kind"] = "signal_alias"
            latest_meta["canonical_signal_meta_path"] = str(signal_meta_path)
            _write_json(_sidecar_meta_path(latest_alias_path), latest_meta)

        print("\n=== Rotation Scores Export ===")
        print(f"Run ID: {signal_run_id}")
        print(f"Canonical File: {signal_path}")
        print(f"Train Meta: {train_meta_path}")
        print(f"Signal Meta: {signal_meta_path}")
        print(f"Signals Index: {signals_index_path}")
        print(f"Backtest Index: {backtest_index_path}")
        if latest_alias_path:
            print(f"Latest Alias: {latest_alias_path}")
        print(f"Size: {signal_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"Total rows: {total_rows:,}")
        print(f"Trading days: {unique_dates}")
        print(f"Unique stocks: {unique_codes}")
        print(f"Date range: {date_range[0]} ~ {date_range[1]}")
        print(f"Top-{top_n} signals: {top_n_rows:,} ({top_n_rows/unique_dates:.1f}/day avg)")
        print("Rotation 旧回测入口已归档，当前不再提供 run_rotation.bat 示例。")
        return str(signal_path)

    if not output_path:
        raise ValueError("output_path is required when artifact_metadata is not provided")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(output_file)
    print("\n=== Rotation Scores Export ===")
    print(f"File: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Total rows: {total_rows:,}")
    print(f"Trading days: {unique_dates}")
    print(f"Unique stocks: {unique_codes}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"Top-{top_n} signals: {top_n_rows:,} ({top_n_rows/unique_dates:.1f}/day avg)")

    return str(output_file)


def export_renko_scores(
    df_scores: pl.DataFrame,
    output_path: str = "data/signals/renko_scores.parquet",
    top_n: int = 20,
) -> str:
    """
    Export full-universe Renko model scores for T+1 open execution.

    时钟:
      - T 日收盘后得到 score/rank
      - Rust 在 T+1 日开盘使用 pre_score/pre_rank 买入
      - T+1 日收盘使用当日 rank 做持仓管理

    Required columns:
        date, code, score, open_adj, high_adj, low_adj, close_adj

    Optional columns:
        volume, market_cap_100m, renko_signal / is_renko
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    required = ["date", "code", "score", "open_adj", "high_adj", "low_adj", "close_adj"]
    missing = [c for c in required if c not in df_scores.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Processing renko scores...")

    df_export = (
        df_scores
        .sort(["code", "date"])
        .with_columns(
            pl.col("close_adj").shift(1).over("code")
                .fill_null(pl.col("close_adj"))
                .alias("pre_close_adj"),
        )
        .with_columns(
            pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("date")
                .cast(pl.UInt16)
                .alias("rank"),
        )
        .with_columns(
            (pl.col("rank") <= top_n).alias("is_top_n"),
        )
        .with_columns([
            pl.col("score").shift(1).over("code").fill_null(-999.0).alias("pre_score"),
            pl.col("rank").shift(1).over("code").fill_null(9999).cast(pl.UInt16).alias("pre_rank"),
            pl.col("is_top_n").shift(1).over("code").fill_null(False).alias("pre_is_top_n"),
        ])
    )

    out_cols = [
        "date", "code",
        "score", "rank", "is_top_n",
        "pre_score", "pre_rank", "pre_is_top_n",
        "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    ]
    for opt_col in ["volume", "market_cap_100m", "is_renko", "renko_signal"]:
        if opt_col in df_scores.columns:
            out_cols.append(opt_col)

    df_final = df_export.select(out_cols)
    df_final.write_parquet(output_file)

    total_rows = df_final.height
    unique_dates = df_final.select(pl.col("date").n_unique()).item()
    unique_codes = df_final.select(pl.col("code").n_unique()).item()
    top_n_rows = df_final.filter(pl.col("is_top_n")).height
    pre_top_n_rows = df_final.filter(pl.col("pre_is_top_n")).height
    date_range = (
        df_final.select(pl.col("date").min()).item(),
        df_final.select(pl.col("date").max()).item(),
    )

    print("\n=== Renko Scores Export ===")
    print(f"File: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Total rows: {total_rows:,}")
    print(f"Trading days: {unique_dates}")
    print(f"Unique stocks: {unique_codes}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"Top-{top_n} signals: {top_n_rows:,} ({top_n_rows/unique_dates:.1f}/day avg)")
    print(f"Pre Top-{top_n} signals: {pre_top_n_rows:,} ({pre_top_n_rows/unique_dates:.1f}/day avg)")

    return str(output_file)


def validate_export(filepath: str) -> dict:
    """Validate exported parquet file"""
    df = pl.read_parquet(filepath)

    result = {
        "filepath": filepath,
        "rows": df.height,
        "columns": df.columns,
        "date_range": (
            df.select(pl.col("date").min()).item(),
            df.select(pl.col("date").max()).item(),
        ),
        "unique_codes": df.select(pl.col("code").n_unique()).item(),
        "signal_count": df.filter(pl.col("b1_signal")).height,
        "loose_signal_count": df.filter(pl.col("b1_signal") & pl.col("is_loose")).height,
    }

    print("\n=== Validation ===")
    print(f"Rows: {result['rows']:,}")
    print(f"Date Range: {result['date_range']}")
    print(f"Unique Codes: {result['unique_codes']}")
    print(f"B1 Signals: {result['signal_count']}")
    print(f"Loose B1 Signals: {result['loose_signal_count']}")

    return result
