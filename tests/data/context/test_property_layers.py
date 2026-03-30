from pathlib import Path

import pandas as pd

from src.data.context.property import (
    PropertyDataConfig,
    build_property_risk_table,
    clean_property_assessment,
)
from src.data.context.student_housing import (
    StudentHousingConfig,
    load_student_housing_data,
)


def test_clean_property_assessment_builds_join_key():
    df = pd.DataFrame(
        {
            "MAP_PAR_ID": ["0001"],
            "LOC_ID": ["L1"],
            "ST_NUM": [12],
            "ST_NAME": ["Main St"],
            "ZIP_CODE": ["2118"],
            "OWNER": ["ACME LLC"],
            "LU_DESC": ["Residential"],
        }
    )

    out = clean_property_assessment(df)

    assert "address_zip_key" in out.columns
    assert out.iloc[0]["address_zip_key"] == "12 main st|02118"
    assert out.iloc[0]["owner_clean"] == "acme"


def test_clean_property_assessment_handles_float_street_number_and_zip():
    df = pd.DataFrame(
        {
            "MAP_PAR_ID": [100001000],
            "LOC_ID": ["L1"],
            "ST_NUM": [195.0],
            "ST_NAME": ["Lexington ST"],
            "ZIP_CODE": [2128.0],
        }
    )

    out = clean_property_assessment(df)

    assert out.iloc[0]["property_address"] == "195 Lexington ST"
    assert out.iloc[0]["address_zip_key"] == "195 lexington st|02128"


def test_build_property_risk_table_joins_assessment_by_address():
    feature_df = pd.DataFrame(
        {
            "property_key": ["12 main st|02118"],
            "violation_st": ["12 Main St"],
            "violation_zip": ["02118"],
            "total_violations": [3],
            "open_violations": [1],
        }
    )
    assessment_df = pd.DataFrame(
        {
            "address_zip_key": ["12 main st|02118"],
            "map_par_id": ["0001"],
            "loc_id": ["L1"],
            "lu_desc": ["Residential"],
            "owner_clean": ["acme llc"],
        }
    )

    out, diagnostics = build_property_risk_table(
        feature_df,
        property_assessment_df=assessment_df,
        parcels_df=None,
        rentsmart_df=None,
    )

    assert "assessment_map_par_id" in out.columns
    assert out.iloc[0]["assessment_owner_clean"] == "acme llc"
    assert diagnostics["property_assessment_loaded"] is True


def test_load_student_housing_data_handles_missing_file(tmp_path: Path):
    student_df, diagnostics = load_student_housing_data(
        StudentHousingConfig(
            raw_dir=tmp_path,
            default_summary_path=None,
            clean_output_path=tmp_path / "student_clean.csv",
            output_path=tmp_path / "student_context.csv",
            summary_output_path=tmp_path / "student_summary.csv",
        )
    )

    assert student_df is None
    assert diagnostics["student_housing_available"] is False


def test_load_student_housing_data_uses_default_summary_when_local_missing(tmp_path: Path):
    summary_path = tmp_path / "student_housing_zip_2023.csv"
    pd.DataFrame(
        {
            "neighborhood": ["Allston"],
            "zip": ["02134"],
            "undergraduates": [2036],
            "graduates": [2471],
            "all_students": [4507],
            "report_year": [2023],
        }
    ).to_csv(summary_path, index=False)

    student_df, diagnostics = load_student_housing_data(
        StudentHousingConfig(
            raw_dir=tmp_path,
            default_summary_path=summary_path,
            clean_output_path=tmp_path / "student_clean.csv",
            output_path=tmp_path / "student_context.csv",
            summary_output_path=tmp_path / "student_summary.csv",
        )
    )

    assert student_df is not None
    assert diagnostics["student_housing_available"] is True
    assert diagnostics["grain"] == "summary_level"
    assert diagnostics["student_housing_default_source"] is True
