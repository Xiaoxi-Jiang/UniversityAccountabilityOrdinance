import pandas as pd
import requests

from src.data.context.acs import clean_acs_context
from src.data.context.address import clean_sam_addresses
from src.data.context.common import download_arcgis_layer
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


def test_download_arcgis_layer_reports_progress_and_saves_csv(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "building_permits.csv"

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "features": [
                    {
                        "attributes": {
                            "permitnumber": "A1000569",
                            "address": "181-183 State ST",
                        }
                    }
                ]
            }

    def fake_get(url, params, timeout):
        del url, params, timeout
        return FakeResponse()

    monkeypatch.setattr("src.data.context.common.requests.get", fake_get)

    path = download_arcgis_layer(
        "https://example.com/arcgis/query",
        output_path,
        timeout=12,
    )

    captured = capsys.readouterr().out
    assert path == output_path
    assert output_path.exists()
    assert "Starting remote context-data download: building_permits.csv" in captured
    assert "Requesting page 1 (offset=0)" in captured
    assert "Response status for page 1: 200" in captured
    assert "Received 1 records on page 1; total downloaded: 1" in captured
    assert "Saved raw context-data snapshot:" in captured


def test_download_arcgis_layer_reports_timeout_context(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "building_permits.csv"

    def fake_get(url, params, timeout):
        del url, params, timeout
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("src.data.context.common.requests.get", fake_get)

    path = download_arcgis_layer(
        "https://example.com/arcgis/query",
        output_path,
        timeout=12,
    )

    captured = capsys.readouterr().out
    assert path is None
    assert "Starting remote context-data download: building_permits.csv" in captured
    assert "Requesting page 1 (offset=0)" in captured
    assert "Skipping remote context-data download for building_permits.csv" in captured
    assert "0 successful page(s) and 0 records" in captured
    assert "no HTTP response status was received" in captured


def test_download_arcgis_layer_by_object_ids_saves_complete_snapshot(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "building_permits.csv"

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        del url, timeout
        if params.get("returnIdsOnly") == "true":
            return FakeResponse(
                {
                    "objectIdFieldName": "objectid",
                    "objectIds": [1, 2, 3],
                }
            )
        raise AssertionError(f"Unexpected GET params: {params}")

    def fake_post(url, data, timeout):
        del url, timeout
        object_ids = data.get("objectIds")
        if object_ids == "1,2":
            return FakeResponse(
                {
                    "features": [
                        {"attributes": {"objectid": 1, "permitnumber": "A1"}},
                        {"attributes": {"objectid": 2, "permitnumber": "A2"}},
                    ]
                }
            )
        if object_ids == "3":
            return FakeResponse(
                {
                    "features": [
                        {"attributes": {"objectid": 3, "permitnumber": "A3"}},
                    ]
                }
            )
        raise AssertionError(f"Unexpected POST data: {data}")

    monkeypatch.setattr("src.data.context.common.requests.get", fake_get)
    monkeypatch.setattr("src.data.context.common.requests.post", fake_post)

    path = download_arcgis_layer(
        "https://example.com/arcgis/query",
        output_path,
        timeout=12,
        use_object_id_chunks=True,
        chunk_size=2,
        max_workers=2,
        max_attempts=1,
    )

    captured = capsys.readouterr().out
    out = pd.read_csv(output_path)
    assert path == output_path
    assert len(out) == 3
    assert set(out["objectid"]) == {1, 2, 3}
    assert "Download mode: object-ID chunks" in captured
    assert "Requesting object ID inventory" in captured
    assert "Received 3 object IDs. Downloading in 2 chunk(s) with 2 worker(s)" in captured
    assert "Saved raw context-data snapshot:" in captured


def test_download_arcgis_layer_by_object_ids_rejects_incomplete_snapshot(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "building_permits.csv"

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        del url, timeout
        if params.get("returnIdsOnly") == "true":
            return FakeResponse(
                {
                    "objectIdFieldName": "objectid",
                    "objectIds": [1, 2, 3],
                }
            )
        raise AssertionError(f"Unexpected GET params: {params}")

    def fake_post(url, data, timeout):
        del url, timeout
        object_ids = data.get("objectIds")
        if object_ids == "1,2":
            return FakeResponse(
                {
                    "features": [
                        {"attributes": {"objectid": 1, "permitnumber": "A1"}},
                        {"attributes": {"objectid": 2, "permitnumber": "A2"}},
                    ]
                }
            )
        if object_ids == "3":
            return FakeResponse({"features": []})
        raise AssertionError(f"Unexpected POST data: {data}")

    monkeypatch.setattr("src.data.context.common.requests.get", fake_get)
    monkeypatch.setattr("src.data.context.common.requests.post", fake_post)

    path = download_arcgis_layer(
        "https://example.com/arcgis/query",
        output_path,
        timeout=12,
        use_object_id_chunks=True,
        chunk_size=2,
        max_workers=1,
        max_attempts=1,
    )

    captured = capsys.readouterr().out
    assert path is None
    assert not output_path.exists()
    assert "during object-ID chunk download" in captured
    assert "could not recover object ID 3" in captured


def test_download_arcgis_layer_by_object_ids_recovers_by_splitting_incomplete_chunk(
    tmp_path,
    monkeypatch,
    capsys,
):
    output_path = tmp_path / "building_permits.csv"

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        del url, timeout
        if params.get("returnIdsOnly") == "true":
            return FakeResponse(
                {
                    "objectIdFieldName": "objectid",
                    "objectIds": [1, 2, 3, 4],
                }
            )
        raise AssertionError(f"Unexpected GET params: {params}")

    post_calls: dict[str, int] = {}

    def fake_post(url, data, timeout):
        del url, timeout
        object_ids = data.get("objectIds")
        post_calls[object_ids] = post_calls.get(object_ids, 0) + 1
        if object_ids == "1,2,3,4":
            return FakeResponse(
                {
                    "features": [
                        {"attributes": {"objectid": 1, "permitnumber": "A1"}},
                        {"attributes": {"objectid": 2, "permitnumber": "A2"}},
                        {"attributes": {"objectid": 3, "permitnumber": "A3"}},
                    ]
                }
            )
        if object_ids == "1,2":
            return FakeResponse(
                {
                    "features": [
                        {"attributes": {"objectid": 1, "permitnumber": "A1"}},
                        {"attributes": {"objectid": 2, "permitnumber": "A2"}},
                    ]
                }
            )
        if object_ids == "3,4":
            return FakeResponse(
                {
                    "features": [
                        {"attributes": {"objectid": 3, "permitnumber": "A3"}},
                        {"attributes": {"objectid": 4, "permitnumber": "A4"}},
                    ]
                }
            )
        raise AssertionError(f"Unexpected POST data: {data}")

    monkeypatch.setattr("src.data.context.common.requests.get", fake_get)
    monkeypatch.setattr("src.data.context.common.requests.post", fake_post)

    path = download_arcgis_layer(
        "https://example.com/arcgis/query",
        output_path,
        timeout=12,
        use_object_id_chunks=True,
        chunk_size=4,
        max_workers=1,
        max_attempts=1,
    )

    captured = capsys.readouterr().out
    out = pd.read_csv(output_path)
    assert path == output_path
    assert len(out) == 4
    assert set(out["objectid"]) == {1, 2, 3, 4}
    assert post_calls["1,2,3,4"] == 1
    assert post_calls["1,2"] == 1
    assert post_calls["3,4"] == 1
    assert "splitting into subchunks of 2 and 2 IDs" in captured


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
