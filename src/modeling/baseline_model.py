"""Lightweight baseline model for Phase 2 check-in."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.features import DEFAULT_OUTPUT_PATH, normalize_zip


@dataclass(frozen=True)
class BaselineModelConfig:
    input_path: Path = DEFAULT_OUTPUT_PATH
    output_path: Path = Path("outputs/tables/baseline_model_results.csv")
    test_size: float = 0.30
    random_state: int = 42


def build_zip_level_modeling_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate property-level features to ZIP-level units for a defensible baseline."""
    if "violation_zip" not in feature_df.columns:
        raise ValueError("Feature table must contain violation_zip for ZIP-level modeling.")

    working = feature_df.copy()
    working["violation_zip"] = working["violation_zip"].map(normalize_zip).astype("string")
    working = working.dropna(subset=["violation_zip"])
    working = working[working["violation_zip"].astype("string").str.len() > 0]

    if working.empty:
        raise ValueError("No usable ZIP values are available for baseline modeling.")

    zip_df = (
        working.groupby("violation_zip")
        .agg(
            total_violations=("total_violations", "sum"),
            open_violations=("open_violations", "sum"),
            closed_violations=("closed_violations", "sum"),
            property_count=("property_key", "nunique"),
            repeat_property_count=("total_violations", lambda s: int((s >= 2).sum())),
            avg_distinct_violation_types=("distinct_violation_types", "mean"),
            avg_violations_per_year=("violations_per_year", "mean"),
            median_days_since_last_violation=("days_since_last_violation", "median"),
        )
        .reset_index()
    )
    zip_df["open_violation_share"] = zip_df["open_violations"] / zip_df["total_violations"]
    threshold = float(zip_df["open_violation_share"].median())
    zip_df["higher_open_share_than_median"] = (
        zip_df["open_violation_share"] > threshold
    ).astype(int)
    zip_df["target_threshold_open_share"] = threshold
    return zip_df


def run_baseline_model(config: BaselineModelConfig) -> Path:
    """Train a simple ZIP-level logistic-regression baseline and save one-row results."""
    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Missing feature table at {config.input_path}. Run Phase 2 feature engineering first."
        )

    feature_df = pd.read_csv(config.input_path)
    modeling_df = build_zip_level_modeling_frame(feature_df)

    target = modeling_df["higher_open_share_than_median"]
    if target.nunique() < 2:
        raise ValueError("Baseline model requires at least two target classes.")

    feature_columns = [
        "total_violations",
        "closed_violations",
        "property_count",
        "repeat_property_count",
        "avg_distinct_violation_types",
        "avg_violations_per_year",
        "median_days_since_last_violation",
    ]
    X = modeling_df.loc[:, feature_columns].copy()
    y = target

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_columns,
            )
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "unit_of_analysis": "zip_code",
                "target_name": "higher_open_share_than_median",
                "target_threshold_open_share": round(
                    float(modeling_df["target_threshold_open_share"].iloc[0]),
                    4,
                ),
                "n_rows": len(modeling_df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "positive_class_rate": round(float(y.mean()), 4),
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
                "feature_count": len(feature_columns),
            }
        ]
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_path, index=False)

    row = results.iloc[0]
    print(
        "Baseline model summary: "
        f"unit={row['unit_of_analysis']}, "
        f"target={row['target_name']}, "
        f"accuracy={row['accuracy']:.4f}, "
        f"precision={row['precision']:.4f}, "
        f"recall={row['recall']:.4f}, "
        f"f1={row['f1']:.4f}"
    )
    print(f"Baseline model results saved to: {config.output_path}")
    return config.output_path


def main() -> None:
    run_baseline_model(BaselineModelConfig())


if __name__ == "__main__":
    main()
