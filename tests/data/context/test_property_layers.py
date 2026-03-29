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
        StudentHousingConfig(raw_dir=tmp_path)
    )

    assert student_df is None
    assert diagnostics["student_housing_available"] is False
