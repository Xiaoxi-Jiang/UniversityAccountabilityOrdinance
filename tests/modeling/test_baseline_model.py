from pathlib import Path

import pandas as pd

from src.modeling.baseline_model import (
    BaselineModelConfig,
    build_property_level_modeling_frame,
    run_baseline_model,
)


def test_build_property_level_modeling_frame_creates_next_period_target():
    df = pd.DataFrame(
        {
            "case_no": ["a1", "a2", "b1", "b2", "c1"],
            "status": ["closed", "open", "closed", "closed", "open"],
            "description": [
                "Maintenance",
                "Unsafe and Dangerous",
                "Failure to Obtain Permit",
                "Failure to Obtain Permit",
                "Unsafe and Dangerous",
            ],
            "violation_st": [
                "1 Comm Ave",
                "1 Comm Ave",
                "2 Bay State Rd",
                "2 Bay State Rd",
                "3 Park Dr",
            ],
            "violation_zip": ["02215", "02215", "02215", "02215", "02115"],
            "violdttm": [
                "2023-01-01",
                "2025-01-15",
                "2023-02-01",
                "2025-01-20",
                "2025-02-01",
            ],
        }
    )

    out = build_property_level_modeling_frame(df, prediction_window_days=365)

    assert set(out["property_key"]) == {"1 comm ave|02215", "2 bay state rd|02215"}
    target_lookup = out.set_index("property_key")["will_receive_high_risk_violation_next_period"].to_dict()
    assert target_lookup["1 comm ave|02215"] == 1
    assert target_lookup["2 bay state rd|02215"] == 0
    assert "history_high_risk_violations" in out.columns
    assert "history_open_share" in out.columns


def test_run_baseline_model_writes_property_level_results_and_coefficients(tmp_path: Path):
    cleaned_path = tmp_path / "violations_clean.csv"
    rows: list[dict[str, object]] = []

    positive_properties = ["10 main st", "11 main st", "12 main st", "13 main st"]
    negative_properties = ["20 main st", "21 main st", "22 main st", "23 main st", "24 main st", "25 main st"]

    case_number = 1
    for address in positive_properties:
        zip_code = "02118"
        rows.append(
            {
                "case_no": f"C{case_number}",
                "status": "closed",
                "description": "Unsafe and Dangerous",
                "violation_st": address,
                "violation_zip": zip_code,
                "violdttm": "2024-01-01",
            }
        )
        case_number += 1
        rows.append(
            {
                "case_no": f"C{case_number}",
                "status": "open",
                "description": "Unsafe Structures",
                "violation_st": address,
                "violation_zip": zip_code,
                "violdttm": "2025-02-01",
            }
        )
        case_number += 1

    for address in negative_properties:
        zip_code = "02119"
        rows.append(
            {
                "case_no": f"C{case_number}",
                "status": "closed",
                "description": "Failure to Obtain Permit",
                "violation_st": address,
                "violation_zip": zip_code,
                "violdttm": "2024-01-05",
            }
        )
        case_number += 1
        rows.append(
            {
                "case_no": f"C{case_number}",
                "status": "closed",
                "description": "Certificate of Occupancy",
                "violation_st": address,
                "violation_zip": zip_code,
                "violdttm": "2025-02-05",
            }
        )
        case_number += 1

    pd.DataFrame(rows).to_csv(cleaned_path, index=False)

    config = BaselineModelConfig(
        input_path=cleaned_path,
        raw_path=tmp_path / "missing_raw.csv",
        output_path=tmp_path / "baseline_model_results.csv",
        coefficients_output_path=tmp_path / "baseline_model_feature_coefficients.csv",
        prediction_window_days=365,
        test_size=0.30,
        random_state=42,
    )

    output_path = run_baseline_model(config)

    results = pd.read_csv(output_path)
    coefficients = pd.read_csv(config.coefficients_output_path)

    assert results.iloc[0]["unit_of_analysis"] == "property_key"
    assert results.iloc[0]["target_name"] == "will_receive_high_risk_violation_next_period"
    assert results.iloc[0]["feature_count"] > 0
    assert "balanced_accuracy" in results.columns
    assert "roc_auc" in results.columns
    assert coefficients["feature_name"].notna().all()
    assert "standardized_coefficient" in coefficients.columns
