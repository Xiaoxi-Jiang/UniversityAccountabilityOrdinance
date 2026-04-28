from pathlib import Path

import pandas as pd
import pytest

from src.modeling.improved_model import (
    ImprovedModelConfig,
    add_owner_level_features,
    build_improved_modeling_frame,
    run_improved_model,
)


def _make_violations_df() -> pd.DataFrame:
    rows = []
    case = 1
    for address, zip_code, future_viol in [
        ("10 main st", "02118", True),
        ("11 main st", "02118", True),
        ("12 main st", "02118", True),
        ("13 main st", "02118", True),
        ("20 main st", "02119", False),
        ("21 main st", "02119", False),
        ("22 main st", "02119", False),
        ("23 main st", "02119", False),
        ("24 main st", "02119", False),
        ("25 main st", "02119", False),
    ]:
        rows.append({
            "case_no": f"C{case}",
            "status": "closed",
            "description": "Unsafe and Dangerous" if future_viol else "Failure to Obtain Permit",
            "violation_st": address,
            "violation_zip": zip_code,
            "violdttm": "2024-01-01",
        })
        case += 1
        rows.append({
            "case_no": f"C{case}",
            "status": "open" if future_viol else "closed",
            "description": "Unsafe Structures" if future_viol else "Certificate of Occupancy",
            "violation_st": address,
            "violation_zip": zip_code,
            "violdttm": "2025-02-01",
        })
        case += 1
    return pd.DataFrame(rows)


def _make_risk_df() -> pd.DataFrame:
    return pd.DataFrame({
        "property_key": [
            "10 main st|02118", "11 main st|02118", "12 main st|02118", "13 main st|02118",
            "20 main st|02119", "21 main st|02119", "22 main st|02119",
            "23 main st|02119", "24 main st|02119", "25 main st|02119",
        ],
        "total_violations": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "history_high_risk_violations": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "assessment_owner_clean": [
            "landlord a", "landlord a", "landlord b", "landlord b",
            "landlord c", "landlord c", "landlord c", "landlord d", "landlord d", "landlord d",
        ],
        "acs_median_household_income": [80000.0] * 10,
        "acs_renter_occupied_share": [0.7] * 10,
        "acs_vacancy_rate": [0.05] * 10,
        "acs_young_adult_share": [0.15] * 10,
        "assessment_yr_built": [1960.0] * 10,
        "assessment_gross_area": [2000.0] * 10,
        "assessment_total_value": [500000.0] * 10,
        "rentsmart_record_count": [1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "rentsmart_complaint_indicator": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "service_request_count": [3.0] * 10,
        "housing_related_service_request_count": [2.0] * 10,
        "service_requests_365d": [1.0] * 10,
        "permit_count": [1.0] * 10,
        "major_permit_count": [0.0] * 10,
        "permits_730d": [1.0] * 10,
    })


def test_add_owner_level_features_aggregates_correctly():
    risk_df = _make_risk_df()
    enriched = add_owner_level_features(risk_df)

    assert "owner_property_count" in enriched.columns
    assert "owner_total_violations" in enriched.columns
    assert "owner_avg_violations_per_property" in enriched.columns
    assert "owner_high_risk_share" in enriched.columns

    landlord_a = enriched.loc[enriched["assessment_owner_clean"] == "landlord a"].iloc[0]
    assert landlord_a["owner_property_count"] == 2
    assert landlord_a["owner_total_violations"] == 4
    assert landlord_a["owner_avg_violations_per_property"] == 2.0


def test_build_improved_modeling_frame_adds_broader_target_and_static_cols():
    source_df = _make_violations_df()
    risk_df = _make_risk_df()
    frame = build_improved_modeling_frame(source_df, risk_df, prediction_window_days=365)

    assert "will_receive_any_violation_next_period" in frame.columns
    assert "will_receive_high_risk_violation_next_period" in frame.columns
    assert "acs_median_household_income" in frame.columns
    assert "owner_property_count" in frame.columns
    # Broader target should have more positives than narrow target
    any_viol = frame["will_receive_any_violation_next_period"].sum()
    high_risk = frame["will_receive_high_risk_violation_next_period"].sum()
    assert any_viol >= high_risk


def test_build_improved_modeling_frame_recomputes_owner_features_from_cutoff_history():
    source_df = _make_violations_df()
    risk_df = _make_risk_df()
    risk_df["total_violations"] = 999
    risk_df["history_high_risk_violations"] = 999

    frame = build_improved_modeling_frame(source_df, risk_df, prediction_window_days=365)

    landlord_a = frame.loc[frame["assessment_owner_clean"] == "landlord a"].iloc[0]
    assert landlord_a["owner_property_count"] == 2
    assert landlord_a["owner_total_violations"] == 2
    assert landlord_a["owner_avg_violations_per_property"] == 1.0
    assert landlord_a["owner_high_risk_share"] == 1.0
    assert "service_request_count" not in frame.columns
    assert "permit_count" not in frame.columns
    assert "rentsmart_record_count" not in frame.columns


def test_build_improved_modeling_frame_adds_only_cutoff_safe_dynamic_features():
    source_df = _make_violations_df()
    risk_df = _make_risk_df()
    risk_df["assessment_map_par_id"] = [f"P{i}" for i in range(1, len(risk_df) + 1)]

    permits_df = pd.DataFrame(
        {
            "map_par_id": ["P1", "P1", "P1", "P2"],
            "permit_issue_date": ["2024-01-15", "2024-03-15", "2021-01-01", "2024-03-15"],
            "major_permit_flag": [1, 1, 0, 1],
            "permit_record_count": [1, 1, 1, 1],
        }
    )
    rentsmart_df = pd.DataFrame(
        {
            "map_par_id": ["P1", "P1", "P1", "P2"],
            "violation_date": ["2024-01-10", "2024-04-01", pd.NA, "2024-04-01"],
            "violation_type": ["inspection", "inspection", "inspection", "inspection"],
        }
    )

    frame = build_improved_modeling_frame(
        source_df,
        risk_df,
        prediction_window_days=365,
        permits_df=permits_df,
        rentsmart_df=rentsmart_df,
    )

    first_property = frame.loc[frame["property_key"] == "10 main st|02118"].iloc[0]
    assert first_property["permit_count"] == 2
    assert first_property["major_permit_count"] == 1
    assert first_property["permits_730d"] == 1
    assert first_property["rentsmart_record_count"] == 1
    assert first_property["rentsmart_complaint_indicator"] == 1

    second_property = frame.loc[frame["property_key"] == "11 main st|02118"].iloc[0]
    assert second_property["permit_count"] == 0
    assert second_property["rentsmart_record_count"] == 0
    assert "service_request_count" not in frame.columns


def test_run_improved_model_writes_results_with_multiple_models(tmp_path: Path):
    violations_path = tmp_path / "violations_clean.csv"
    risk_path = tmp_path / "property_risk_table_v1.csv"

    _make_violations_df().to_csv(violations_path, index=False)
    _make_risk_df().to_csv(risk_path, index=False)

    config = ImprovedModelConfig(
        input_path=violations_path,
        raw_path=tmp_path / "missing_raw.csv",
        property_risk_path=risk_path,
        permits_context_path=tmp_path / "missing_permits.csv",
        rentsmart_context_path=tmp_path / "missing_rentsmart.csv",
        output_path=tmp_path / "improved_model_results.csv",
        feature_importance_path=tmp_path / "improved_model_feature_importance.csv",
        prediction_window_days=365,
        cv_folds=2,
        random_state=42,
    )
    output_path = run_improved_model(config)
    results = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(results) > 0
    assert "model_name" in results.columns
    assert "feature_set" in results.columns
    assert "roc_auc" in results.columns
    assert "pr_auc" in results.columns
    assert results["model_name"].nunique() >= 2

    # Full feature set should produce a row
    assert "behavioral_plus_static" in results["feature_set"].values
