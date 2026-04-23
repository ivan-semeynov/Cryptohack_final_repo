from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

HORIZON_DAYS = 30
DELTA_C = 5.0
STEP_HOURS = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_STATE = 42

LEAKY_COLUMNS = {
    "datetime",
    "WABT",
    "limit_wabt",
    "gap_to_limit",
    "future_wabt_max",
    "future_wabt_delta",
    "target_rise_in_horizon",
    "is_shutdown",
    "WABT_lag_1d",
    "WABT_lag_7d",
    "WABT_rolling_mean_7d_prev",
}


@dataclass
class ModelPackage:
    pipeline: Pipeline
    feature_columns: list[str]
    threshold: float
    metrics: dict[str, Any]
    feature_importance: list[dict[str, float]]
    config: dict[str, Any]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]


def load_source(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    return df.sort_values(["dataset_id", "datetime"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("dataset_id", sort=True):
        g = group.copy().sort_values("datetime").reset_index(drop=True)
        t0 = g["datetime"].min()
        elapsed_hours = (g["datetime"] - t0).dt.total_seconds() / 3600.0
        elapsed_max = float(elapsed_hours.max()) if len(g) else 0.0
        g["elapsed_hours"] = elapsed_hours
        g["elapsed_share"] = elapsed_hours / elapsed_max if elapsed_max > 0 else np.zeros(len(g))
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def build_target(
    df: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
    delta_c: float = DELTA_C,
    step_hours: int = STEP_HOURS,
) -> pd.DataFrame:
    out = df.copy().sort_values(["dataset_id", "datetime"]).reset_index(drop=True)
    horizon_steps = int(horizon_days * 24 / step_hours)

    future_max_all: list[np.ndarray] = []
    future_delta_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []

    for _, group in out.groupby("dataset_id", sort=True):
        wabt = group["WABT"].to_numpy(dtype=float)
        future_max = np.empty(len(group), dtype=float)
        future_delta = np.empty(len(group), dtype=float)
        target = np.zeros(len(group), dtype=int)

        for idx in range(len(group)):
            end_idx = min(len(group), idx + horizon_steps + 1)
            future_window = wabt[idx:end_idx]
            max_future = float(np.max(future_window))
            delta = max_future - float(wabt[idx])
            future_max[idx] = max_future
            future_delta[idx] = delta
            target[idx] = int(delta >= delta_c)

        future_max_all.append(future_max)
        future_delta_all.append(future_delta)
        target_all.append(target)

    out["future_wabt_max"] = np.concatenate(future_max_all)
    out["future_wabt_delta"] = np.concatenate(future_delta_all)
    out["target_rise_in_horizon"] = np.concatenate(target_all)
    return out


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_candidates = [c for c in df.columns if c not in LEAKY_COLUMNS]
    missing_target = df["target_rise_in_horizon"].isna()
    missing_keys = df["dataset_id"].isna() | df["datetime"].isna()
    all_features_nan = df[feature_candidates].isna().all(axis=1)
    valid_mask = ~(missing_target | missing_keys | all_features_nan)
    cleaned = df.loc[valid_mask].copy().reset_index(drop=True)
    report = pd.DataFrame(
        [
            {"rule": "total_before", "rows": int(len(df))},
            {"rule": "drop_missing_target", "rows": int(missing_target.sum())},
            {"rule": "drop_missing_dataset_or_datetime", "rows": int(missing_keys.sum())},
            {"rule": "drop_all_features_nan", "rows": int(all_features_nan.sum())},
            {"rule": "total_after", "rows": int(len(cleaned))},
            {"rule": "removed_share", "rows": float((len(df) - len(cleaned)) / max(len(df), 1))},
        ]
    )
    return cleaned, report


def temporal_split_per_dataset(df: pd.DataFrame) -> SplitData:
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.sort_values(["dataset_id", "datetime"]).groupby("dataset_id", sort=True):
        n_rows = len(group)
        train_end = int(n_rows * TRAIN_RATIO)
        val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_df = pd.concat(train_parts).sort_values(["dataset_id", "datetime"]).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(["dataset_id", "datetime"]).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(["dataset_id", "datetime"]).reset_index(drop=True)
    feature_columns = [c for c in df.columns if c not in LEAKY_COLUMNS]
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df, feature_columns=feature_columns)


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def tune_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(y_true)),
        "positive_share": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "score_mean": float(np.mean(y_score)),
    }


def train_model_from_dataframe(df: pd.DataFrame) -> tuple[ModelPackage, dict[str, pd.DataFrame]]:
    df = add_time_features(df)
    df = build_target(df)
    clean_df, quality_report = prepare_dataset(df)
    split = temporal_split_per_dataset(clean_df)

    X_train = split.train_df[split.feature_columns]
    y_train = split.train_df["target_rise_in_horizon"].astype(int)
    X_val = split.val_df[split.feature_columns]
    y_val = split.val_df["target_rise_in_horizon"].astype(int)
    X_test = split.test_df[split.feature_columns]
    y_test = split.test_df["target_rise_in_horizon"].astype(int)

    pipeline = build_pipeline()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)

    train_score = pipeline.predict_proba(X_train)[:, 1]
    val_score = pipeline.predict_proba(X_val)[:, 1]
    test_score = pipeline.predict_proba(X_test)[:, 1]

    threshold, val_best_f1 = tune_threshold(y_val, val_score)

    metrics = {
        "train": compute_metrics(y_train, (train_score >= threshold).astype(int), train_score),
        "validation": compute_metrics(y_val, (val_score >= threshold).astype(int), val_score),
        "test": compute_metrics(y_test, (test_score >= threshold).astype(int), test_score),
        "threshold": threshold,
        "validation_best_f1": val_best_f1,
    }

    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_importance = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(
            zip(split.feature_columns, importances, strict=False), key=lambda x: x[1], reverse=True
        )
    ]

    package = ModelPackage(
        pipeline=pipeline,
        feature_columns=split.feature_columns,
        threshold=threshold,
        metrics=metrics,
        feature_importance=feature_importance,
        config={
            "horizon_days": HORIZON_DAYS,
            "delta_c": DELTA_C,
            "step_hours": STEP_HOURS,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "random_state": RANDOM_STATE,
            "algorithm": "RandomForestClassifier",
        },
    )

    artifacts = {
        "prepared_dataset": clean_df,
        "quality_report": quality_report,
        "train_df": split.train_df,
        "val_df": split.val_df,
        "test_df": split.test_df,
    }
    return package, artifacts


def save_model_package(package: ModelPackage, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.joblib"
    metadata_path = output_path / "metadata.json"

    joblib.dump(package.pipeline, model_path)
    metadata = {
        "feature_columns": package.feature_columns,
        "threshold": package.threshold,
        "metrics": package.metrics,
        "feature_importance": package.feature_importance,
        "config": package.config,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model": model_path, "metadata": metadata_path}


def load_model_package(model_dir: str | Path) -> ModelPackage:
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return ModelPackage(
        pipeline=pipeline,
        feature_columns=list(metadata["feature_columns"]),
        threshold=float(metadata["threshold"]),
        metrics=metadata["metrics"],
        feature_importance=metadata["feature_importance"],
        config=metadata["config"],
    )


def align_features(records: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for record in records:
        row = {feature: record.get(feature, np.nan) for feature in feature_columns}
        normalized.append(row)
    return pd.DataFrame(normalized, columns=feature_columns)


def predict_records(records: list[dict[str, Any]], package: ModelPackage) -> list[dict[str, Any]]:
    if not records:
        return []
    features = align_features(records, package.feature_columns)
    probabilities = package.pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= package.threshold).astype(int)
    result: list[dict[str, Any]] = []
    for idx, (record, probability, prediction) in enumerate(zip(records, probabilities, predictions, strict=False)):
        risk_level = "high" if probability >= max(package.threshold, 0.75) else "medium" if probability >= package.threshold else "low"
        result.append(
            {
                "row_id": idx,
                "prediction": int(prediction),
                "probability": float(probability),
                "threshold": float(package.threshold),
                "risk_level": risk_level,
                "missing_features": [feature for feature in package.feature_columns if feature not in record or pd.isna(record.get(feature))],
            }
        )
    return result
