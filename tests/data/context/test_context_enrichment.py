import pandas as pd

from src.data.context.acs import clean_acs_context
from src.data.context.address import clean_sam_addresses
from src.data.context.permits import aggregate_permits, clean_permits
from src.data.context.property import build_property_risk_table
from src.data.context.service_requests import aggregate_service_requests, clean_service_requests


def test_clean_sam_addresses_builds_normalized_keys_and_ids():
    df = pd.DataFrame(
        {
            "FULL_ADDRESS": ["12 Main Street"],
            "ZIP_CODE": ["2118"],
            "PARCEL": ["0001"],
            "LOC_ID": ["L1"],
            "MAILING_NEIGHBORHOOD": ["South End"],
        }
    )

    out = clean_sam_addresses(df)

    assert out.iloc[0]["sam_address_zip_key"] == "12 main st|02118"
    assert out.iloc[0]["map_par_id"] == "0001"
    assert out.iloc[0]["neighborhood"] == "south end"


def test_aggregate_service_requests_derives_recent_and_housing_counts():
    df = pd.DataFrame(
        {
            "ADDRESS": ["12 Main St", "12 Main St"],
            "ZIP_CODE": ["02118", "02118"],
            "OPEN_DT": ["2024-02-01", "2023-01-01"],
            "REASON": ["Housing Complaint", "Parking"],
        }
    )

    cleaned = clean_service_requests(df)
    aggregated = aggregate_service_requests(
        cleaned,
        join_key="address_zip_key",
        reference_date=pd.Timestamp("2024-03-01"),
    )

    row = aggregated.iloc[0]
    assert row["service_request_count"] == 2
    assert row["housing_related_service_request_count"] == 1
    assert row["service_requests_365d"] == 1


def test_clean_service_requests_handles_new_system_schema():
    df = pd.DataFrame(
        {
            "CASE_ID": ["BCS-1"],
            "OPEN_DATE": ["2026-02-22 10:34:58.338+00"],
            "SERVICE_NAME": ["Street Light Other"],
            "ASSIGNED_DEPARTMENT": ["Public Works Department (PWD)"],
            "FULL_ADDRESS": ["30 B St, Boston, MA 02127"],
            "ZIP_CODE": ["02127"],
            "NEIGHBORHOOD": ["South Boston"],
        }
    )

    out = clean_service_requests(df)

    assert out.iloc[0]["service_request_open_date"].year == 2026
    assert out.iloc[0]["address_zip_key"] == "30 b st|02127"


def test_aggregate_permits_derives_major_and_recent_signals():
    df = pd.DataFrame(
        {
            "ADDRESS": ["12 Main St", "12 Main St"],
            "ZIP_CODE": ["02118", "02118"],
            "ISSUE_DATE": ["2024-01-10", "2021-01-10"],
            "PERMIT_TYPE": ["Structural Renovation", "Minor Repair"],
        }
    )

    cleaned = clean_permits(df)
    aggregated = aggregate_permits(
        cleaned,
        join_key="address_zip_key",
        reference_date=pd.Timestamp("2024-03-01"),
    )

    row = aggregated.iloc[0]
    assert row["permit_count"] == 2
    assert row["major_permit_count"] == 1
    assert row["permits_730d"] == 1


def test_clean_permits_parses_epoch_dates_and_maps_parcel_identifier():
    df = pd.DataFrame(
        {
            "ADDRESS": ["12 Main St"],
            "ZIP": [2118.0],
            "ISSUED_DATE": [1611851366000],
            "PARCEL_ID": [303807000.0],
        }
    )

    out = clean_permits(df)

    assert out.iloc[0]["permit_zip"] == "02118"
    assert out.iloc[0]["map_par_id"] == "303807000"
    assert str(out.iloc[0]["permit_issue_date"]).startswith("2021-01-28")


def test_clean_acs_context_derives_zip_level_features():
    df = pd.DataFrame(
        {
            "ZIP": ["02118"],
            "median_household_income": [100000],
            "total_tenure": [100],
            "renter_occupied": [60],
            "total_housing_units": [120],
            "vacant_housing_units": [12],
            "median_gross_rent": [2500],
            "total_population": [500],
            "B01001_007E": [10],
            "B01001_008E": [5],
            "B01001_009E": [5],
            "B01001_010E": [15],
            "B01001_031E": [8],
            "B01001_032E": [4],
            "B01001_033E": [6],
            "B01001_034E": [12],
        }
    )

    out = clean_acs_context(df)

    assert out.iloc[0]["acs_zip"] == "02118"
    assert out.iloc[0]["acs_renter_occupied_share"] == 0.6
    assert out.iloc[0]["acs_vacancy_rate"] == 0.1


def test_build_property_risk_table_prefers_sam_identifier_path():
    feature_df = pd.DataFrame(
        {
            "property_key": ["12 main st|02118"],
            "violation_st": ["12 Main St"],
            "violation_zip": ["02118"],
            "total_violations": [3],
            "open_violations": [1],
        }
    )
    sam_df = pd.DataFrame(
        {
            "sam_address_zip_key": ["12 main st|02118"],
            "sam_address_only_key": ["12 main st"],
            "map_par_id": ["0001"],
            "loc_id": ["l1"],
            "neighborhood": ["south end"],
        }
    )
    assessment_df = pd.DataFrame(
        {
            "map_par_id": ["0001"],
            "loc_id": ["l1"],
            "owner_clean": ["acme"],
            "lu_desc": ["Residential"],
        }
    )
    parcels_df = pd.DataFrame(
        {
            "loc_id": ["l1"],
            "map_par_id": ["0001"],
            "poly_type": ["parcel"],
        }
    )
    service_requests_df = clean_service_requests(
        pd.DataFrame(
            {
                "ADDRESS": ["12 Main St"],
                "ZIP_CODE": ["02118"],
                "OPEN_DT": ["2024-02-01"],
                "REASON": ["Housing Complaint"],
            }
        )
    )
    permits_df = clean_permits(
        pd.DataFrame(
            {
                "ADDRESS": ["12 Main St"],
                "ZIP_CODE": ["02118"],
                "ISSUE_DATE": ["2023-10-01"],
                "PERMIT_TYPE": ["Structural Renovation"],
            }
        )
    )
    acs_df = pd.DataFrame(
        {
            "acs_zip": ["02118"],
            "acs_median_household_income": [100000],
        }
    )

    out, diagnostics = build_property_risk_table(
        feature_df,
        property_assessment_df=assessment_df,
        parcels_df=parcels_df,
        rentsmart_df=None,
        sam_df=sam_df,
        service_requests_df=service_requests_df,
        permits_df=permits_df,
        acs_df=acs_df,
    )

    row = out.iloc[0]
    assert row["sam_map_par_id"] == "0001"
    assert row["assessment_owner_clean"] == "acme"
    assert row["parcel_poly_type"] == "parcel"
    assert row["service_request_count"] == 1
    assert row["permit_count"] == 1
    assert row["acs_median_household_income"] == 100000
    assert diagnostics["sam_match_rate_pct"] == 100.0
