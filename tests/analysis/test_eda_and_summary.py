from pathlib import Path

import pandas as pd

from src.analysis.checkin_summary import write_checkin_summary
from src.analysis.eda import generate_student_housing_outputs


def test_generate_student_housing_outputs_writes_relationship_artifacts(tmp_path: Path):
    context_path = tmp_path / "student_housing_summary_v1.csv"
    pd.DataFrame(
        {
            "zip": ["02115", "02134", "02118"],
            "total_violations": [120, 200, 80],
            "open_violations": [10, 20, 5],
            "property_count": [60, 100, 40],
            "all_students": [5000, 4500, 1500],
        }
    ).to_csv(context_path, index=False)

    table_paths, figure_paths = generate_student_housing_outputs(
        context_path,
        tmp_path / "tables",
        tmp_path / "figures",
    )

    assert any(path.name == "student_housing_relationship.csv" for path in table_paths)
    assert any(path.name == "student_housing_correlation_summary.csv" for path in table_paths)
    assert any(path.name == "student_housing_relationship.png" for path in figure_paths)


def test_write_checkin_summary_mentions_new_target_and_student_relationship(tmp_path: Path):
    model_results_path = tmp_path / "baseline_model_results.csv"
    coefficients_path = tmp_path / "baseline_model_feature_coefficients.csv"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "unit_of_analysis": "property_key",
                "target_name": "will_receive_high_risk_violation_next_period",
                "target_definition": "new high risk (proxy) violation within next 365 days",
                "training_cutoff_date": "2025-03-27",
                "reference_date": "2026-03-27",
                "prediction_window_days": 365,
                "n_rows": 100,
                "n_train": 70,
                "n_test": 30,
                "n_positive": 8,
                "n_positive_test": 2,
                "positive_class_rate": 0.08,
                "majority_class_accuracy": 0.92,
                "accuracy": 0.70,
                "balanced_accuracy": 0.65,
                "precision": 0.25,
                "recall": 0.50,
                "f1": 0.3333,
                "roc_auc": 0.72,
                "feature_count": 5,
            }
        ]
    ).to_csv(model_results_path, index=False)
    pd.DataFrame(
        {
            "feature_name": ["history_high_risk_violations", "days_since_last_violation"],
            "standardized_coefficient": [1.2, -0.8],
            "abs_standardized_coefficient": [1.2, 0.8],
            "odds_ratio_per_sd_increase": [3.32, 0.45],
        }
    ).to_csv(coefficients_path, index=False)
    pd.DataFrame(
        [{"owner_data_available_pct": 88.5, "rows_with_owner_data": 9265, "total_rows": 10471}]
    ).to_csv(tables_dir / "owner_data_availability_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "matched_zip_count": 20,
                "student_metric_column": "all_students",
                "pearson_corr_total_violations": -0.4526,
                "pearson_corr_violations_per_property": -0.1762,
                "median_violations_per_1000_students": 194.0764,
            }
        ]
    ).to_csv(tables_dir / "student_housing_correlation_summary.csv", index=False)

    output_path = write_checkin_summary(
        output_path=tmp_path / "checkin_summary.md",
        feature_path=tmp_path / "violations_feature_table_v1.csv",
        property_risk_path=tmp_path / "property_risk_table_v1.csv",
        student_context_path=tmp_path / "student_housing_summary_v1.csv",
        model_results_path=model_results_path,
        coefficients_path=coefficients_path,
        tables_dir=tables_dir,
        property_key_diagnostics={
            "address_based_key_pct": 90.0,
            "case_no_fallback_pct": 2.0,
        },
        property_risk_diagnostics={
            "sam_loaded": True,
            "property_assessment_loaded": True,
            "parcels_loaded": True,
            "service_requests_loaded": True,
            "permits_loaded": True,
            "acs_loaded": True,
        },
        student_housing_diagnostics={"student_housing_available": True},
    )

    body = output_path.read_text(encoding="utf-8")
    assert "property-level prediction problem" in body
    assert "will_receive_high_risk_violation_next_period" in body
    assert "student_housing_relationship.csv" in body
