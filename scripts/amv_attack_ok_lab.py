from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from scripts.amv_bull_pool_listwise_ranker_lab import (
    DEFAULT_QMT_DB,
    STATE_FEATURES,
    default_folds,
)
from scripts.amv_constrained_oracle_lab import DEFAULT_OUTPUT_ROOT as CONSTRAINED_OUTPUT_ROOT
from scripts.amv_horizon_oracle_explainability import (
    DEFAULT_OUTPUT_ROOT as EXPLAIN_OUTPUT_ROOT,
    build_state_frame,
)

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked.*")


DEFAULT_CONSTRAINED_DIR = CONSTRAINED_OUTPUT_ROOT / "20260517_131848"
DEFAULT_STATE_PANEL = EXPLAIN_OUTPUT_ROOT / "20260517_131130" / "oracle_choices_with_state.csv"
DEFAULT_OUTPUT_ROOT = CONSTRAINED_OUTPUT_ROOT.parent / "amv_attack_ok"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("thresholds must not be empty")
    if any((threshold <= 0 or threshold >= 1) for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be in (0, 1)")
    return thresholds


def read_constrained_choices(path: Path) -> pl.DataFrame:
    if path.is_dir():
        path = path / "daily_constrained_choices.csv"
    if not path.exists():
        raise FileNotFoundError(f"daily_constrained_choices.csv not found: {path}")
    return pl.read_csv(path, try_parse_dates=True)


def read_state_panel(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"state panel not found: {path}")
    state = pl.read_csv(path, try_parse_dates=True)
    required = {"date", "year", *STATE_FEATURES}
    missing = sorted(required - set(state.columns))
    if missing:
        raise ValueError(f"state panel missing columns: {', '.join(missing)}")
    return (
        state.group_by(["date", "year"])
        .agg([pl.col(col).first().alias(col) for col in STATE_FEATURES])
        .sort("date")
    )


def select_attack_labels(
    choices: pl.DataFrame,
    *,
    target_metric: str,
    margin: float,
    allow_cash: bool,
) -> pl.DataFrame:
    filtered = choices.filter(
        (pl.col("target_metric") == target_metric)
        & ((pl.col("margin") - margin).abs() < 1e-12)
        & (pl.col("allow_cash") == allow_cash)
    )
    if filtered.is_empty():
        raise ValueError(
            "no constrained choices matched "
            f"target_metric={target_metric} margin={margin} allow_cash={allow_cash}"
        )
    return (
        filtered.with_columns(
            [
                (pl.col("choice_type") == "attack").cast(pl.Int8).alias("attack_ok"),
                (pl.col("choice_type") == "base").cast(pl.Int8).alias("base_ok"),
                (pl.col("choice_type") == "cash").cast(pl.Int8).alias("cash_ok"),
            ]
        )
        .select(
            [
                "date",
                "year",
                "attack_ok",
                "base_ok",
                "cash_ok",
                "choice_type",
                "base_candidate_id",
                "attack_candidate_id",
                "attack_sleeve_id",
                "attack_horizon",
                "chosen_candidate_id",
                "base_ret",
                "base_dailyized_ret",
                "attack_ret",
                "attack_dailyized_ret",
                "chosen_ret",
                "chosen_dailyized_ret",
                "lift_vs_base",
                "dailyized_lift_vs_base",
                "target_metric",
                "margin",
                "allow_cash",
            ]
        )
        .sort("date")
    )


def split_by_year(df: pl.DataFrame, start_year: int, end_year: int) -> pl.DataFrame:
    return df.filter((pl.col("year") >= start_year) & (pl.col("year") <= end_year)).sort("date")


def to_numpy(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df.select(feature_cols).to_numpy().astype(np.float32)


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    if positives == 0:
        return None
    order = np.argsort(-score)
    y_sorted = y_true[order]
    precision_at_k = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    return float((precision_at_k * y_sorted).sum() / positives)


def classification_at_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = score >= threshold
    y_bool = y_true.astype(bool)
    tp = int((pred & y_bool).sum())
    fp = int((pred & ~y_bool).sum())
    tn = int((~pred & ~y_bool).sum())
    fn = int((~pred & y_bool).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "pred_attack_days": int(pred.sum()),
        "true_attack_days": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "pred_positive_rate": float(pred.mean()) if len(pred) else 0.0,
    }


def economic_at_threshold(
    panel: pl.DataFrame,
    score: np.ndarray,
    threshold: float,
    *,
    strategy_id: str,
    fold_id: str,
) -> dict[str, Any]:
    pred_attack = score >= threshold
    base_ret = panel["base_ret"].to_numpy()
    attack_ret = panel["attack_ret"].to_numpy()
    base_dailyized = panel["base_dailyized_ret"].to_numpy()
    attack_dailyized = panel["attack_dailyized_ret"].to_numpy()
    chosen_ret = np.where(pred_attack, attack_ret, base_ret)
    chosen_dailyized = np.where(pred_attack, attack_dailyized, base_dailyized)
    lift = chosen_ret - base_ret
    dailyized_lift = chosen_dailyized - base_dailyized
    y_true = panel["attack_ok"].to_numpy().astype(np.int32)
    cls = classification_at_threshold(y_true, score, threshold)
    return {
        "strategy_id": strategy_id,
        "fold_id": fold_id,
        "test_year": int(panel["year"].max()) if panel.height else None,
        "days": panel.height,
        "threshold": threshold,
        "mean_base_ret": float(base_ret.mean()),
        "mean_chosen_ret": float(chosen_ret.mean()),
        "mean_lift_vs_base": float(lift.mean()),
        "mean_base_dailyized_ret": float(base_dailyized.mean()),
        "mean_chosen_dailyized_ret": float(chosen_dailyized.mean()),
        "mean_dailyized_lift_vs_base": float(dailyized_lift.mean()),
        "positive_lift_ratio": float((lift > 0).mean()),
        **cls,
    }


def choose_threshold_by_valid_f1(
    valid: pl.DataFrame,
    valid_score: np.ndarray,
    grid: list[float],
) -> float:
    y_valid = valid["attack_ok"].to_numpy().astype(np.int32)
    scored = [classification_at_threshold(y_valid, valid_score, threshold) for threshold in grid]
    best = sorted(
        scored,
        key=lambda row: (row["f1"], row["precision"], -abs(row["pred_positive_rate"] - row["positive_rate"])),
        reverse=True,
    )[0]
    return float(best["threshold"])


def train_attack_model(
    panel: pl.DataFrame,
    *,
    feature_cols: list[str],
    thresholds: list[float],
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    threshold_grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05).tolist()]
    classification_rows: list[dict[str, Any]] = []
    economic_rows: list[dict[str, Any]] = []
    prediction_frames: list[pl.DataFrame] = []
    importance_frames: list[pl.DataFrame] = []

    for fold in default_folds():
        fold_id = str(fold["id"])
        train = split_by_year(panel, fold["train_start"], fold["train_end"])
        valid = split_by_year(panel, fold["valid_year"], fold["valid_year"])
        test = split_by_year(panel, fold["test_year"], fold["test_year"])
        if train.is_empty() or valid.is_empty() or test.is_empty():
            continue

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            scale_pos_weight=args.scale_pos_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                to_numpy(train, feature_cols),
                train["attack_ok"].to_numpy().astype(np.int32),
                eval_set=[
                    (
                        to_numpy(valid, feature_cols),
                        valid["attack_ok"].to_numpy().astype(np.int32),
                    )
                ],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

        valid_score = model.predict_proba(to_numpy(valid, feature_cols))[:, 1]
        test_score = model.predict_proba(to_numpy(test, feature_cols))[:, 1]
        y_test = test["attack_ok"].to_numpy().astype(np.int32)
        chosen_threshold = choose_threshold_by_valid_f1(valid, valid_score, threshold_grid)

        classification_rows.append(
            {
                "fold_id": fold_id,
                "test_year": fold["test_year"],
                "days": test.height,
                "positive_rate": float(y_test.mean()),
                "auc": binary_auc(y_test, test_score),
                "average_precision": average_precision(y_test, test_score),
                "valid_best_f1_threshold": chosen_threshold,
                "mean_score": float(test_score.mean()),
            }
        )

        baseline_zero = np.zeros(test.height, dtype=np.float64)
        oracle_score = y_test.astype(np.float64)
        economic_rows.append(
            economic_at_threshold(
                test,
                baseline_zero,
                0.5,
                strategy_id="always_base",
                fold_id=fold_id,
            )
        )
        economic_rows.append(
            economic_at_threshold(
                test,
                oracle_score,
                0.5,
                strategy_id="label_oracle_attack_ok",
                fold_id=fold_id,
            )
        )
        economic_rows.append(
            economic_at_threshold(
                test,
                np.ones(test.height, dtype=np.float64),
                0.5,
                strategy_id="always_attack",
                fold_id=fold_id,
            )
        )
        for threshold in thresholds:
            economic_rows.append(
                economic_at_threshold(
                    test,
                    test_score,
                    threshold,
                    strategy_id=f"model_threshold_{threshold:.2f}",
                    fold_id=fold_id,
                )
            )
        economic_rows.append(
            economic_at_threshold(
                test,
                test_score,
                chosen_threshold,
                strategy_id="model_valid_best_f1",
                fold_id=fold_id,
            )
        )

        prediction_frames.append(
            test.select(
                [
                    "date",
                    "year",
                    "attack_ok",
                    "choice_type",
                    "base_ret",
                    "attack_ret",
                    "base_dailyized_ret",
                    "attack_dailyized_ret",
                    "attack_candidate_id",
                    "chosen_candidate_id",
                ]
            ).with_columns(
                [
                    pl.lit(fold_id).alias("fold_id"),
                    pl.Series("attack_score", test_score),
                    pl.lit(chosen_threshold).alias("valid_best_f1_threshold"),
                ]
            )
        )
        importance_frames.append(
            pl.DataFrame(
                {
                    "fold_id": [fold_id] * len(feature_cols),
                    "test_year": [fold["test_year"]] * len(feature_cols),
                    "feature": feature_cols,
                    "importance_gain": model.booster_.feature_importance(
                        importance_type="gain"
                    ).tolist(),
                    "importance_split": model.booster_.feature_importance(
                        importance_type="split"
                    ).tolist(),
                }
            )
        )

    return (
        pl.DataFrame(classification_rows),
        pl.DataFrame(economic_rows),
        pl.concat(prediction_frames, how="vertical") if prediction_frames else pl.DataFrame(),
        pl.concat(importance_frames, how="vertical") if importance_frames else pl.DataFrame(),
    )


def summarize_economics(economic: pl.DataFrame) -> pl.DataFrame:
    return (
        economic.group_by("strategy_id")
        .agg(
            [
                pl.col("days").sum().alias("days"),
                pl.col("mean_base_ret").mean().alias("avg_base_ret"),
                pl.col("mean_chosen_ret").mean().alias("avg_chosen_ret"),
                pl.col("mean_lift_vs_base").mean().alias("avg_lift_vs_base"),
                pl.col("mean_chosen_dailyized_ret").mean().alias("avg_chosen_dailyized_ret"),
                pl.col("mean_dailyized_lift_vs_base").mean().alias("avg_dailyized_lift_vs_base"),
                pl.col("precision").mean().alias("avg_precision"),
                pl.col("recall").mean().alias("avg_recall"),
                pl.col("f1").mean().alias("avg_f1"),
                pl.col("pred_attack_days").sum().alias("pred_attack_days"),
                pl.col("true_attack_days").sum().alias("true_attack_days"),
            ]
        )
        .sort("avg_dailyized_lift_vs_base", descending=True)
    )


def summarize_importance(importance: pl.DataFrame) -> pl.DataFrame:
    return (
        importance.group_by("feature")
        .agg(
            [
                pl.col("importance_gain").mean().alias("avg_importance_gain"),
                pl.col("importance_split").mean().alias("avg_importance_split"),
            ]
        )
        .sort("avg_importance_gain", descending=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV constrained oracle attack_ok binary lab")
    parser.add_argument("--constrained-dir", type=Path, default=DEFAULT_CONSTRAINED_DIR)
    parser.add_argument("--state-panel", type=Path, default=DEFAULT_STATE_PANEL)
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-metric", default="top_ret_dailyized")
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--allow-cash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--state-horizon", type=int, default=6)
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--exclude-limit-up-entry", action="store_true")
    parser.add_argument("--price-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--relevance-top-k", type=int, default=100)
    parser.add_argument("--state-top-n", type=int, default=20)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    parser.add_argument("--thresholds", type=parse_thresholds, default=[0.30, 0.40, 0.50, 0.60])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--scale-pos-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Loading constrained oracle choices...")
    choices = read_constrained_choices(args.constrained_dir)
    labels = select_attack_labels(
        choices,
        target_metric=args.target_metric,
        margin=args.margin,
        allow_cash=args.allow_cash,
    )
    if args.state_panel.exists():
        print(f"Loading state panel: {args.state_panel}")
        state = read_state_panel(args.state_panel)
    else:
        print("State panel missing; rebuilding pre-trade state frame...")
        state = build_state_frame(args)

    panel = labels.join(state, on=["date", "year"], how="inner").sort("date")
    if panel.is_empty():
        raise ValueError("attack_ok panel is empty after joining labels with state features")
    feature_cols = STATE_FEATURES
    classification, economic, predictions, importance = train_attack_model(
        panel,
        feature_cols=feature_cols,
        thresholds=args.thresholds,
        args=args,
    )
    strategy_summary = summarize_economics(economic)
    importance_summary = summarize_importance(importance)

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.write_csv(output_dir / "attack_ok_panel.csv")
    classification.write_csv(output_dir / "fold_classification_metrics.csv")
    economic.write_csv(output_dir / "fold_strategy_metrics.csv")
    strategy_summary.write_csv(output_dir / "strategy_summary.csv")
    predictions.write_csv(output_dir / "model_predictions.csv")
    importance.write_csv(output_dir / "feature_importance_by_fold.csv")
    importance_summary.write_csv(output_dir / "feature_importance_summary.csv")

    summary: dict[str, Any] = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "constrained_dir": str(args.constrained_dir),
            "state_panel": str(args.state_panel),
            "target_metric": args.target_metric,
            "margin": args.margin,
            "allow_cash": args.allow_cash,
            "features": feature_cols,
            "thresholds": args.thresholds,
            "folds": default_folds(),
        },
        "panel": {
            "rows": panel.height,
            "dates": panel["date"].n_unique(),
            "date_min": str(panel["date"].min()),
            "date_max": str(panel["date"].max()),
            "attack_days": int(panel["attack_ok"].sum()),
            "base_days": int(panel["base_ok"].sum()),
            "cash_days": int(panel["cash_ok"].sum()),
        },
        "metrics": {
            "classification": classification.to_dicts(),
            "strategy_summary": strategy_summary.to_dicts(),
            "feature_importance_summary": importance_summary.to_dicts(),
        },
        "files": {
            "attack_ok_panel": "attack_ok_panel.csv",
            "fold_classification_metrics": "fold_classification_metrics.csv",
            "fold_strategy_metrics": "fold_strategy_metrics.csv",
            "strategy_summary": "strategy_summary.csv",
            "model_predictions": "model_predictions.csv",
            "feature_importance_by_fold": "feature_importance_by_fold.csv",
            "feature_importance_summary": "feature_importance_summary.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Saved: {output_dir / 'summary.json'}")
    print(
        f"Panel: dates={panel['date'].n_unique()} attack={int(panel['attack_ok'].sum())} "
        f"base={int(panel['base_ok'].sum())} cash={int(panel['cash_ok'].sum())}"
    )
    print("Strategy summary:")
    for row in strategy_summary.to_dicts():
        print(
            f"- {row['strategy_id']}: chosen_dailyized={row['avg_chosen_dailyized_ret'] * 100:+.2f}% "
            f"lift={row['avg_dailyized_lift_vs_base'] * 100:+.2f}pp "
            f"precision={row['avg_precision']:.3f} recall={row['avg_recall']:.3f} "
            f"pred_attack={row['pred_attack_days']}"
        )
    print("Top features:")
    for row in importance_summary.head(8).to_dicts():
        print(f"- {row['feature']}: gain={row['avg_importance_gain']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
