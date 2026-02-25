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
- 311 Service Requests: Boston Open Data CSV/API
- SAM Addresses: Boston Open Data CSV/API
- Property Assessment: Boston Open Data CSV/API
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

## Current Stage Implementation: Phase 1
This repository currently includes a phase-1 pipeline:
- Downloads Boston violations data (`data/raw/violations.csv`)
- Cleans and standardizes fields
- Exports cleaned dataset (`data/processed/violations_clean.csv`)

Run:
```bash
make install
make phase1
```

## Build, Run, and Test
```bash
make install
make phase1
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
  src/
    data/
    analysis/
    viz/
  tests/
  .github/workflows/
  Makefile
  requirements.txt
  README.md
```

## Contributing
1. Create a feature branch.
2. Keep raw/processed data contracts stable.
3. Add/update tests for data logic changes.
4. Open a PR with a short validation note (`make test`, sample output paths).

## Notes on Reproducibility
- Use the Make targets as the single execution interface.
- Keep source URLs and output paths centralized in pipeline code.
- Avoid manual spreadsheet edits in processed outputs.
