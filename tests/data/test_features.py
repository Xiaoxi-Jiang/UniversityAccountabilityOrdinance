import pandas as pd

from src.data.features import (
    build_feature_table,
    clean_violator_name,
    Phase2FeatureConfig,
    generate_property_key,
    get_property_key_diagnostics,
    load_phase2_source_data,
    normalize_zip,
    prepare_violations_frame,
)


def test_generate_property_key_prefers_address_and_zip():
    df = pd.DataFrame(
        {
            "violation_st": ["12 Main St", "12 MAIN ST", None],
            "violation_zip": ["02118", "02118", "02119"],
            "case_no": ["A", "B", "C"],
        }
    )

    out = generate_property_key(df)

    assert out.iloc[0] == "12 main st|02118"
    assert out.iloc[1] == "12 main st|02118"
    assert out.iloc[2] == "c"


def test_clean_violator_name_normalizes_spacing_and_case():
    series = pd.Series(["  Acme, LLC ", "ACME LLC", None])
    out = clean_violator_name(series)
    assert out.iloc[0] == "acme llc"
    assert out.iloc[1] == "acme llc"
    assert out.iloc[2] == ""


def test_normalize_zip_handles_float_artifacts():
    assert normalize_zip(2128.0) == "02128"
    assert normalize_zip("2128.0") == "02128"


def test_build_feature_table_handles_missing_status_and_date_columns():
    df = pd.DataFrame(
        {
            "violation_st": ["1 Comm Ave", "1 Comm Ave", "2 Bay State Rd"],
            "violation_zip": ["02215", "02215", "02215"],
            "violator_name": ["Owner A", "Owner A", "Owner B"],
            "violationtype": ["Trash", "Heat", "Trash"],
        }
    )

    out = build_feature_table(df)

    assert "property_key" in out.columns
    assert "total_violations" in out.columns
    assert "open_violations" in out.columns
    assert "closed_violations" in out.columns
    assert "distinct_violation_types" in out.columns
    assert "first_violation_date" not in out.columns

    row = out.loc[out["property_key"] == "1 comm ave|02215"].iloc[0]
    assert row["total_violations"] == 2
    assert row["open_violations"] == 0
    assert row["closed_violations"] == 0
    assert row["distinct_violation_types"] == 2


def test_build_feature_table_derives_temporal_features_when_available():
    df = pd.DataFrame(
        {
            "violation_st": ["1 Comm Ave", "1 Comm Ave", "1 Comm Ave"],
            "violation_zip": ["02215", "02215", "02215"],
            "status": ["open", "closed", "open"],
            "violdttm": ["2024-01-01", "2024-06-01", "2024-12-31"],
            "violationtype": ["Trash", "Trash", "Heat"],
        }
    )

    out = build_feature_table(df)
    row = out.iloc[0]

    assert row["total_violations"] == 3
    assert row["open_violations"] == 2
    assert row["closed_violations"] == 1
    assert str(row["first_violation_date"]).startswith("2024-01-01")
    assert str(row["last_violation_date"]).startswith("2024-12-31")
    assert row["active_span_days"] == 365
    assert row["recent_violation_count_365d"] == 3


def test_load_phase2_source_data_recovers_raw_date_and_address(tmp_path):
    cleaned_path = tmp_path / "violations_clean.csv"
    raw_path = tmp_path / "violations.csv"
    pd.DataFrame(
        {
            "case_no": ["A1"],
            "status": ["open"],
            "description": ["Unsafe and Dangerous"],
            "violation_zip": ["02118"],
        }
    ).to_csv(cleaned_path, index=False)
    pd.DataFrame(
        {
            "case_no": ["A1"],
            "status_dttm": ["2024-02-01 12:00:00"],
            "violation_stno": ["12"],
            "violation_street": ["Main"],
            "violation_suffix": ["St"],
            "violation_zip": ["2118"],
        }
    ).to_csv(raw_path, index=False)

    loaded = load_phase2_source_data(
        Phase2FeatureConfig(
            input_path=cleaned_path,
            output_path=tmp_path / "feature_table.csv",
            raw_path=raw_path,
        )
    )

    assert "status_dttm" in loaded.columns
    assert "violation_st" in loaded.columns
    assert loaded.iloc[0]["violation_st"] == "12 main st"
    assert loaded.iloc[0]["violation_zip"] == "02118"


def test_property_key_diagnostics_report_case_no_fallback():
    df = pd.DataFrame(
        {
            "case_no": ["A1", "A2", "A3"],
            "violation_st": ["", "10 Main St", "10 Main St"],
            "violation_zip": ["", "02118", "02118"],
        }
    )

    prepared, _ = prepare_violations_frame(df)
    diagnostics = get_property_key_diagnostics(prepared)

    assert diagnostics["unique_property_keys"] == 2
    assert diagnostics["case_no_fallback_pct"] > 0
