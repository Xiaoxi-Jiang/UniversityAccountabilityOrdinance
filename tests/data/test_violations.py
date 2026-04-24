import pandas as pd

from src.data.violations import Phase1Config, clean_violations, run_phase1


def test_clean_violations_standardizes_and_derives_flag():
    df = pd.DataFrame(
        {
            "Case No": ["A1", "A1", "B2"],
            "Status": [" Open ", "Closed", "pending"],
            "ViolDttm": ["2024-01-10", "2024-01-10", "bad-date"],
            "Description": ["d1", "d1", "d2"],
        }
    )

    out = clean_violations(df)

    assert "case_no" in out.columns
    assert "status" in out.columns
    assert "violdttm" in out.columns
    assert "is_open_violation" in out.columns
    assert len(out) == 2

    open_row = out.loc[out["case_no"] == "A1"].iloc[0]
    pending_row = out.loc[out["case_no"] == "B2"].iloc[0]

    assert open_row["status"] == "open"
    assert open_row["is_open_violation"] == 1
    assert pending_row["is_open_violation"] == 1


def test_clean_violations_handles_missing_status():
    df = pd.DataFrame({"Case No": ["X1"], "Status": [None]})
    out = clean_violations(df)
    assert out.iloc[0]["status"] == "unknown"


def test_run_data_preparation_preloads_optional_phase1_sources(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    violations_df = pd.DataFrame(
        {
            "Case No": ["A1"],
            "Status": ["Open"],
            "ViolDttm": ["2024-01-10"],
            "Description": ["unsafe condition"],
        }
    )
    property_df = pd.DataFrame(
        {
            "MAP_PAR_ID": ["0001"],
            "LOC_ID": ["L1"],
            "ST_NUM": [12],
            "ST_NAME": ["Main St"],
            "ZIP_CODE": ["2118"],
            "OWNER": ["ACME LLC"],
        }
    )
    parcels_df = pd.DataFrame(
        {
            "MAP_PAR_ID": ["0001"],
            "LOC_ID": ["L1"],
            "POLY_TYPE": ["parcel"],
        }
    )
    sam_df = pd.DataFrame(
        {
            "FULL_ADDRESS": ["12 Main St"],
            "ZIP_CODE": ["02118"],
            "MAP_PAR_ID": ["0001"],
            "LOC_ID": ["L1"],
            "NEIGHBORHOOD": ["South End"],
        }
    )
    service_requests_df = pd.DataFrame(
        {
            "ADDRESS": ["12 Main St"],
            "ZIP_CODE": ["02118"],
            "OPEN_DT": ["2024-02-01"],
            "REASON": ["Unsanitary Conditions"],
        }
    )
    permits_df = pd.DataFrame(
        {
            "ADDRESS": ["12 Main St"],
            "ZIP_CODE": ["02118"],
            "ISSUE_DATE": ["2023-08-15"],
            "PERMIT_TYPE": ["Renovation"],
        }
    )
    rentsmart_df = pd.DataFrame(
        {
            "FULL_ADDRESS": ["12 Main St"],
            "ZIP_CODE": ["02118"],
            "PARCEL": ["0001"],
            "DATE": ["2024-02-01"],
            "TYPE": ["Housing Complaint"],
            "DESCRIPTION": ["Unsanitary conditions"],
        }
    )
    acs_df = pd.DataFrame(
        {
            "ZIP": ["02118"],
            "median_household_income": [95000],
            "total_tenure": [1000],
            "renter_occupied": [700],
            "total_housing_units": [1100],
            "vacant_housing_units": [100],
            "median_gross_rent": [2400],
            "total_population": [3000],
            "B01001_007E": [50],
            "B01001_008E": [40],
            "B01001_009E": [35],
            "B01001_010E": [90],
            "B01001_031E": [45],
            "B01001_032E": [40],
            "B01001_033E": [30],
            "B01001_034E": [85],
        }
    )
    student_df = pd.DataFrame(
        {
            "School": ["Boston University"],
            "Address": ["12 Main St"],
            "Zip": ["02118"],
            "Year": [2024],
        }
    )

    property_df.to_csv(raw_dir / "property_assessment_fy25.csv", index=False)
    parcels_df.to_csv(raw_dir / "parcels_current.csv", index=False)
    sam_df.to_csv(raw_dir / "sam_addresses.csv", index=False)
    service_requests_df.to_csv(raw_dir / "service_requests_311.csv", index=False)
    permits_df.to_csv(raw_dir / "building_permits.csv", index=False)
    acs_df.to_csv(raw_dir / "acs_context.csv", index=False)
    student_df.to_csv(raw_dir / "student_housing.csv", index=False)

    def fake_download(_url, output_path, timeout=30):
        del timeout
        violations_df.to_csv(output_path, index=False)

    monkeypatch.setattr(
        "src.data.violations.download_violations_csv",
        fake_download,
    )

    def fake_download_rentsmart(config):
        rentsmart_df.to_csv(config.output_path, index=False)
        return config.output_path

    monkeypatch.setattr(
        "src.data.context.rentsmart.download_rentsmart_csv",
        fake_download_rentsmart,
    )

    raw_path, cleaned_path = run_phase1(
        Phase1Config(raw_dir=raw_dir, processed_dir=processed_dir)
    )

    assert raw_path.exists()
    assert cleaned_path.exists()
    assert (processed_dir / "property_assessment_clean.csv").exists()
    assert (processed_dir / "sam_addresses_clean.csv").exists()
    assert (processed_dir / "parcels_clean.csv").exists()
    assert (processed_dir / "rentsmart_clean.csv").exists()
    assert (processed_dir / "service_requests_311_clean.csv").exists()
    assert (processed_dir / "building_permits_clean.csv").exists()
    assert (processed_dir / "acs_context_clean.csv").exists()
    assert (processed_dir / "student_housing_clean.csv").exists()
