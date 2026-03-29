# University Accountability Ordinance Data Project

## Team
- xiaoxij@bu.edu
- chez0212@bu.edu
- zywang1@bu.edu

## Project Description
This project studies off-campus student housing in Boston and builds a reproducible data pipeline to evaluate accountability outcomes across neighborhoods, landlords, and time. We integrate housing, violations, and city service data to understand where students live, housing quality conditions, and compliance patterns.

## Measurable Project Goals
1. Estimate student rental concentration by district and year.
2. Identify high-risk landlord/property patterns using violation severity and frequency features.
3. Build a baseline predictive model for whether a property will receive a new severe violation in the next period.
4. Produce clear visualizations for district-level trends and landlord risk distributions.

## Data Sources and Collection Plan
- Building and Property Violations: Boston Open Data CSV/API
- Property Assessment: Boston ArcGIS/Analyze Boston
- Parcels (current/FY25-compatible): Boston ArcGIS parcels service
- RentSmart: optional local export if available
- Student Housing: optional team-provided CSV/XLSX
- 311 Service Requests: Boston Open Data CSV/API
- SAM Addresses: Boston Open Data CSV/API
- Student Housing (2016-2024): course/client shared spreadsheet
- Neighborhood and council district boundaries: shapefiles from city/open sources

Collection method:
- Programmatic download of public datasets via URLs/API.
- Raw snapshots saved in `data/raw/`.
- Processed outputs saved in `data/processed/`.
- Versioned transformation logic under `src/`.

## Segmented Schedule
### Phase 0: Proposal Finalization (Feb 25 - Mar 3)
- Finalize research questions and measurable success metrics.
- Lock target data schema and source list.
- Deliverable: proposal-quality README sections complete.

### Phase 1: Data Collection + Cleaning (Mar 4 - Mar 24)
- Implement reproducible data fetch scripts.
- Implement first cleaning pass for core violation dataset (date parsing, missing values, de-duplication, status normalization).
- Generate initial cleaned table and data quality summary.
- Deliverable: runnable phase-1 pipeline and March check-in materials.

### Phase 2: Feature Extraction + EDA (Mar 25 - Apr 14)
- Engineer landlord/property and temporal features.
- Build preliminary visualizations (district trends, severity distribution, violation density).
- Deliverable: feature table v1 and exploratory figures.

### Phase 3: Modeling + Evaluation (Apr 15 - Apr 24)
- Train baseline models (e.g., logistic regression / tree-based models).
- Evaluate with held-out split and report metrics.
- Document limitations/failure cases.
- Deliverable: reproducible model training script and results table.

### Phase 4: Reproducibility Hardening + Final Report (Apr 25 - May 1)
- Finalize Makefile commands and README runbook.
- Add and stabilize tests + GitHub Actions workflow.
- Publish final visualizations, result interpretation, and presentation video link.
- Deliverable: submission-ready GitHub repository.

## Runbook
Main entrypoints:

```bash
make install
make prepare-data
make pipeline
make baseline-model
```

What they do:
- `make prepare-data`: downloads and cleans the source violations dataset
  and preloads optional property/student-housing source tables into cleaned Phase 1 outputs when local files or public endpoints are available
- `make pipeline`: builds violations features, attempts property-layer enrichment, optionally integrates student housing context, generates EDA tables/figures, and runs the baseline model
- `make baseline-model`: reruns only the baseline model from the feature table

Generated outputs:
- `data/processed/violations_clean.csv`
- `data/processed/property_assessment_clean.csv` when Property Assessment data is available
- `data/processed/parcels_clean.csv` when Parcels data is available
- `data/processed/rentsmart_clean.csv` when RentSmart data is available
- `data/processed/student_housing_clean.csv` when a local student housing file is available
- `data/processed/violations_feature_table_v1.csv`
- `data/processed/property_risk_table_v1.csv`
- summary tables in `outputs/tables/`
- exploratory figures in `outputs/figures/`
- baseline model results in `outputs/tables/baseline_model_results.csv`

Optional outputs:
- `data/processed/student_housing_context_v1.csv` if a local student housing file is available
- `data/processed/student_housing_summary_v1.csv` if only summary-level student housing integration is possible
- additional property-risk tables/figures when property assessment, parcels, or RentSmart context is available

## Build, Run, and Test
```bash
make install
make prepare-data
make pipeline
make baseline-model
make test
```

Environment:
- Python 3.10+
- Tested on Windows PowerShell and GitHub Actions Ubuntu runner

## Repository Layout
```text
UniversityAccountabilityOrdinance/
  data/
    raw/
    processed/
  outputs/
    tables/
    figures/
  notebooks/
  reports/
    figures/
  src/
    analysis/
    data/
      context/
    modeling/
    viz/
    pipeline.py
  tests/
    data/
      context/
    modeling/
    viz/
  .github/workflows/
  Makefile
  requirements.txt
  README.md
```

Structure notes:
- `src/`: importable project package root
- `src/data/`: core violations data flow
- `src/data/violations.py`: Phase 1 violations download and cleaning
- `src/data/features.py`: Phase 2 feature engineering over cleaned violations
- `src/data/context/`: optional enrichment layers used after the core violations flow
- `src/data/context/property.py`: optional property assessment / parcels / RentSmart integration
- `src/data/context/student_housing.py`: optional student housing loader and context joins
- `src/analysis/`: EDA and reporting entrypoints
- `src/modeling/`: baseline modeling code
- `src/viz/`: plotting utilities and figure generation
- `src/pipeline.py`: top-level reproducible project pipeline
- `tests/data/`, `tests/modeling/`, `tests/viz/`: tests organized to mirror the source tree
- `outputs/`: generated tables and figures from reproducible runs
- `reports/`: presentation-ready assets for final deliverables

Optional data notes:
- Property Assessment and Parcels are configured with official Boston ArcGIS service endpoints and are cached to `data/raw/` when available.
- RentSmart is treated as optional because a stable bulk machine-readable source may not always be available.
- Expected local raw file names include:
  `data/raw/student_housing.xlsx`, `data/raw/student_housing.csv`,
  `data/raw/uar_fall_2022.xlsx`, `data/raw/uar_fall_2023.xlsx`,
  and optionally `data/raw/rentsmart.csv` / `data/raw/rentsmart.xlsx` / `data/raw/rentsmart.geojson`.
- Student housing integration expects one of those local files. If missing, the pipeline logs the limitation and skips that layer.

## Contributing
1. Create a feature branch.
2. Keep raw/processed data contracts stable.
3. Add/update tests for data logic changes.
4. Open a PR with a short validation note (`make test`, sample output paths).

## Notes on Reproducibility
- Use the Make targets as the single execution interface.
- Keep source URLs and output paths centralized in pipeline code.
- Avoid manual spreadsheet edits in processed outputs.
