# University Accountability Ordinance Data Project

Presentation video link: https://youtu.be/dbY66ogjLk4

Final report date: May 1, 2026

## Team

- xiaoxij@bu.edu
- chez0212@bu.edu
- zywang1@bu.edu

## Reproducibility Quick Start

This repository is organized so the project can be rebuilt from the command line. The `Makefile` is the main entry point for installing dependencies, preparing data, running the pipeline, training models, building visualizations, and running tests.

### Environment

- Python 3.10 or newer
- GNU Make
- Tested with Windows PowerShell and the GitHub Actions Ubuntu runner

On systems where `python3` is not available, pass `PYTHON=python` to `make`, for example:

```bash
make PYTHON=python install
```

### Build, Run, and Test

Run these commands from the repository root:

```bash
make install
make prepare-data
make pipeline
make baseline-model
make improved-model
make interactive-viz
make test
```

What each command does:

| Command | Purpose | Main outputs |
|---|---|---|
| `make install` | Installs Python dependencies from `requirements.txt`. | Local Python environment |
| `make fetch-rentsmart` | Fetches the public RentSmart Boston export when a local file is not present. | `data/raw/rentsmart.csv` |
| `make prepare-data` | Downloads and cleans the core violations dataset and attempts to preload optional public context data. | `data/processed/violations_clean.csv` and context clean files |
| `make pipeline` | Builds property-level features, joins context layers, creates EDA tables and figures, and writes a check-in summary. | `data/processed/property_risk_table_v1.csv`, `outputs/tables/`, `outputs/figures/` |
| `make baseline-model` | Trains the baseline property-level model. | `outputs/tables/baseline_model_results.csv` |
| `make improved-model` | Trains and evaluates expanded risk-ranking models. | `outputs/tables/improved_model_results.csv`, feature importance, ablation, grouped-CV, calibration, and top-risk tables |
| `make interactive-viz` | Builds browser-based interactive visualizations. | `outputs/interactive/index.html` |
| `make test` | Runs the automated test suite. | Pytest results |

The GitHub Actions workflow in `.github/workflows/tests.yml` installs dependencies and runs `pytest -q` on every push and pull request.

## Project Goal

This project studies off-campus student housing and housing accountability in Boston. We build a reproducible data pipeline that combines housing violations, city service requests, permits, property assessment data, RentSmart records, ACS context, and student-housing context. The goal is to understand where violations concentrate, how they relate to student-housing geography, and whether property-level risk rankings can support inspection or accountability decisions.

Measurable goals:

1. Estimate student rental concentration by ZIP code and property context.
2. Identify repeated-risk property patterns using violation frequency, severity proxy, owner context, 311 service requests, permits, RentSmart records, and student-housing context.
3. Train property-level ranking models for whether a property receives a future violation or a future medium/high-risk violation.
4. Produce clear static and interactive visualizations for dataset trends, student ZIP context, property risk, and model performance.

## Repository Organization

```text
UniversityAccountabilityOrdinance/
  .github/workflows/
    tests.yml
  data/
    raw/
    processed/
    reference/
  outputs/
    tables/
    figures/
    interactive/
  reports/
  src/
    analysis/
    data/
      context/
    modeling/
    viz/
    pipeline.py
  tests/
    analysis/
    data/
      context/
    modeling/
    viz/
  Makefile
  requirements.txt
  README.md
```

Important source modules:

| Path | Role |
|---|---|
| `src/data/violations.py` | Downloads and cleans the core Boston building/property violations dataset. |
| `src/data/features.py` | Builds property-level keys and violation history features. |
| `src/data/context/` | Loads and cleans optional context layers, including SAM addresses, property assessment, parcels, 311 requests, permits, ACS, RentSmart, and student housing. |
| `src/pipeline.py` | Coordinates feature generation, context joins, EDA tables, figures, and reporting outputs. |
| `src/modeling/baseline_model.py` | Builds the baseline property-level modeling frame and logistic regression model. |
| `src/modeling/improved_model.py` | Adds static, owner, temporal, RentSmart, 311, permit, ACS, and student-housing features; evaluates multiple models. |
| `src/viz/phase2_visualizations.py` | Builds static figures and summary tables. |
| `src/viz/interactive_visualizations.py` | Builds the interactive HTML dashboard. |

## Data Collection

The project uses public city data whenever possible and uses local or reference student-housing data when public record-level student housing data is not available. Data collection is implemented in code, not by manual spreadsheet editing.

| Data source | Use in project | Justification | Collection implementation |
|---|---|---|---|
| Boston building/property violations | Core outcome dataset for property-level violations and risk history. | Directly measures housing compliance and enforcement activity. | `src/data/violations.py` downloads the CKAN CSV and saves `data/raw/violations.csv`. |
| SAM addresses | Address normalization and crosswalk support. | Improves joins between violation addresses and city property records. | `src/data/context/address.py`. |
| Property assessment | Property class, owner, building attributes, assessed values, owner occupancy, residential-unit signals. | Adds static property context and owner accountability features. | `src/data/context/property.py`. |
| Parcels | Parcel and geospatial/property identifier context. | Helps connect addresses, assessment records, and property-level context. | `src/data/context/property.py`. |
| 311 service requests | Housing complaint and service-request context before modeling cutoffs. | Adds resident complaint signals beyond formal violations. | `src/data/context/service_requests.py`. |
| Building permits | Recent construction, occupancy, and major-permit activity. | Permits can indicate building change, renovation, or code-relevant activity. | `src/data/context/permits.py`. |
| ACS ZIP context | ZIP-level demographic and housing context. | Provides neighborhood context such as renter share and young-adult share. | `src/data/context/acs.py`. |
| RentSmart Boston | Complaint, inspection, and violation context from the public RentSmart dashboard. | Adds public accountability signals linked to property condition. | `src/data/context/rentsmart.py` and `make fetch-rentsmart`. |
| Student housing data | ZIP-level student counts, student units, and student concentration features. | Needed to evaluate off-campus student-housing context. | `src/data/context/student_housing.py`; local course/client files are used when present, otherwise `data/reference/student_housing_zip_2023.csv` provides a fallback summary. |

Raw snapshots are stored under `data/raw/`. Cleaned and standardized outputs are stored under `data/processed/`.

Current processed outputs include:

- `data/processed/violations_clean.csv`
- `data/processed/violations_feature_table_v1.csv`
- `data/processed/property_risk_table_v1.csv`
- `data/processed/sam_addresses_clean.csv`
- `data/processed/property_assessment_clean.csv`
- `data/processed/parcels_clean.csv`
- `data/processed/service_requests_311_clean.csv`
- `data/processed/service_requests_311_historical.csv`
- `data/processed/building_permits_clean.csv`
- `data/processed/acs_context_clean.csv`
- `data/processed/rentsmart_clean.csv`
- `data/processed/student_housing_clean.csv`
- `data/processed/student_housing_summary_v1.csv`

## Data Cleaning

Cleaning is implemented in the `src/data/` and `src/data/context/` modules. The cleaning logic is designed to preserve source records while standardizing the fields needed for reproducible joins, aggregation, visualization, and modeling.

Core violations cleaning:

| Issue | Cleaning step | Code |
|---|---|---|
| Inconsistent column names | Standardize columns to lowercase snake-case names. | `src/data/violations.py` |
| Duplicate violation rows | Drop duplicate rows by `case_no` when available. | `src/data/violations.py` |
| Inconsistent status values | Normalize `status` to lowercase strings and derive `is_open_violation`. | `src/data/violations.py` |
| Date parsing problems | Parse violation/status date fields with coercion for invalid dates. | `src/data/violations.py`, `src/data/features.py` |
| ZIP formatting inconsistencies | Normalize ZIP codes to five-character strings. | `src/data/features.py` |
| Missing address fields | Recover address components from the raw violation file when the cleaned file lacks usable address fields. | `src/data/features.py` |
| Missing property identifiers | Build a best-available `property_key` from address + ZIP, address only, case number, or row fallback. | `src/data/features.py` |

Context cleaning:

- SAM, parcels, property assessment, permits, 311, RentSmart, ACS, and student-housing loaders each standardize expected columns before joining.
- Address-like fields are normalized before address-based joins.
- Numeric context fields are coerced to numeric values and filled where absence means no observed record, for example no recent 311 request or no recent permit.
- Temporal context used for modeling is filtered to records at or before the training cutoff to reduce data leakage.
- Student housing is handled as optional: if a local source file is not available, the pipeline falls back to a bundled ZIP-level summary and logs the limitation.

## Feature Extraction

Feature extraction is centered on property-level accountability. Each row in the main feature table represents one modeled property key.

Current feature table summary:

| Metric | Value |
|---|---:|
| Violation records represented | 16,987 |
| Property-level rows | 10,500 |
| Open violations | 861 |
| First violation date | 2009-12-01 |
| Latest violation date | 2026-04-28 |

Main feature groups:

| Feature group | Examples | Why it is appropriate |
|---|---|---|
| Property identity | `property_key`, `property_key_source`, address/ZIP keys | Allows violations and context sources to be aggregated to the property level. |
| Violation history | `total_violations`, `open_violations`, `closed_violations`, `distinct_violation_types`, `primary_violation_type` | Captures repeated-risk patterns and enforcement history. |
| Time features | `first_violation_date`, `last_violation_date`, `active_span_days`, `violations_per_year`, `days_since_last_violation`, `recent_violation_count_365d` | Separates old historical patterns from recent activity. |
| Severity proxy features | high, medium, and low-risk violation counts and shares | Converts violation descriptions/types into a risk-relevant modeling signal. |
| Property/owner features | owner availability, property class, residential units, building age, owner-level violation history | Supports accountability analysis beyond individual addresses. |
| 311 features | housing-related requests, recent windows, growth from prior year, heat/pest/sanitation/building-code categories | Adds complaint history that can precede formal violations. |
| Permit features | permit counts, recent permits, major permits, occupancy-related permits | Captures building activity that may relate to inspection or compliance risk. |
| RentSmart features | record counts, complaint counts, inspection counts, issue-type counts | Adds another public view of property condition and accountability. |
| ACS features | renter share, vacancy rate, young-adult share, median income | Adds ZIP-level housing and demographic context. |
| Student housing features | students, student units, students per modeled property, student ZIP flag | Connects property risk patterns to off-campus student-housing concentration. |

These features are appropriate because the prediction and reporting tasks are property-level risk and accountability tasks. The features combine direct violation history with contextual signals that a city or university accountability process could use for prioritization.

## Model Training and Evaluation

### Modeling Frame

The modeling target is property-level future violation risk. The improved modeling frame contains 9,828 property-level rows. We evaluate three targets:

| Target | Role | Positive cases | Positive class rate |
|---|---|---:|---:|
| Any future violation | Primary | 489 | 4.98% |
| Future medium/high-risk violation | Primary | 188 | 1.91% |
| Future high-risk violation | Secondary | 68 | 0.69% |

The high-risk-only target is reported as secondary because the positive class is extremely sparse.

### Training Procedure

The baseline model uses logistic regression on property-level violation history. The improved model expands the feature set and evaluates:

- Logistic regression
- Random forest
- XGBoost, when available

The improved model compares multiple feature sets:

- `behavioral_only`
- `behavioral_plus_static`
- `plus_static_property`
- `plus_311`
- `plus_permits`
- `plus_rentsmart`
- `plus_student_housing`

The model pipeline uses imputation for missing numeric values and appropriate scaling for logistic regression. Temporal context features are filtered relative to the training cutoff before joining to the modeling frame.

### Evaluation Strategy

Because the positive class is rare, plain accuracy is not a good primary metric. The evaluation focuses on ranking and rare-event performance:

| Metric | Why used |
|---|---|
| ROC-AUC | Measures ranking/discrimination across thresholds. |
| PR-AUC | More informative than accuracy for imbalanced targets. |
| Balanced accuracy | Avoids rewarding a model for predicting only the majority class. |
| Precision, recall, F1 | Summarize classification behavior at a chosen threshold. |
| Precision@10, Precision@25, Precision@50, Precision@100 | Measures whether the top-ranked properties are useful for inspection prioritization. |
| ZIP grouped cross-validation | Checks robustness when geographic leakage is reduced. |
| Calibration/risk decile summaries | Helps interpret whether predicted scores align with observed risk. |

### Results

Baseline model result:

| Model | Target | Rows | Positive rate | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | Future high-risk violation within 730 days | 9,828 | 0.69% | 0.582 | 0.713 | 0.038 |

Selected improved model results:

| Target | Best selected model/feature set | ROC-AUC | PR-AUC | Precision@10 | Precision@50 |
|---|---|---:|---:|---:|---:|
| Any violation | Random forest, `behavioral_plus_static` | 0.649 | 0.102 | 0.12 | 0.14 |
| Medium/high-risk violation | Random forest, `behavioral_plus_static` | 0.652 | 0.051 | 0.06 | 0.064 |
| High-risk violation | Random forest, `behavioral_plus_static` | 0.695 | 0.026 | 0.02 | 0.024 |

Ablation results show that adding property, permit, RentSmart, and student-housing context can improve ranking metrics over behavioral-only features for some targets. For example, the any-violation random forest improves from ROC-AUC 0.522 with behavioral-only features to ROC-AUC 0.648 with student-housing context included in the ablation run.

### Interpretation

The model should be interpreted as a prioritization and ranking tool, not a deterministic prediction system. The most useful output is the ranked property table:

```text
outputs/tables/top_predicted_risk_properties.csv
```

The top-risk table includes property key, ZIP, owner, violation history, context signals, and predicted risk scores. Example top context signals include RentSmart record counts, student counts by ZIP, building-code 311 requests, students per modeled property, and permit intensity.

### Limitations and Failure Cases

- Positive classes are rare, especially the high-risk target. This limits model stability and makes PR-AUC and top-K precision more important than accuracy.
- Student housing is available mainly at ZIP level, so the student-housing features should be interpreted as neighborhood context rather than proof that a specific unit houses students.
- Address-based joins can miss records when source systems format addresses differently.
- Public datasets can change over time, so exact row counts may shift when the pipeline is rerun against refreshed city data.
- Violations measure enforcement records, not every true housing condition issue.
- Some context sources are optional or depend on public endpoint availability; the pipeline is designed to run with fallback behavior when a layer is missing.

## Data Visualization and Results

Visualizations are generated as both static figures and interactive HTML.

Static figures:

```text
outputs/figures/
  severity_distribution.png
  status_distribution.png
  top_violation_types.png
  violations_over_time.png
  student_housing_relationship.png
  student_housing_violation_intensity_by_zip.png
  student_housing_zip_context.png
  top_repeated_properties_with_owner.png
  violations_by_property_class.png
```

Interactive dashboard:

```text
outputs/interactive/index.html
```

The dashboard includes five presentation tabs:

| Tab | What it shows | Result supported |
|---|---|---|
| Overview | Dataset counts, severity distribution, status distribution, top violation types, violations over time | Confirms the scale and time coverage of the violations dataset. |
| Student ZIP Context | Student concentration, violations per property, violations per 1,000 students, ZIP ranking, and map views | Shows that student concentration and violation intensity do not follow a simple one-variable pattern. |
| Property Risk | Repeated-risk properties, owner context, and property-class violation rates | Supports property-level accountability analysis. |
| Model Performance | Model comparison, feature importance, Precision@K, calibration, and grouped-CV views | Supports the conclusion that models are most useful for prioritization. |
| HTML Exports | Links to individual interactive charts | Makes the results easy to inspect and present. |

Current visualization-supported findings:

- The processed violations feature table represents 16,987 violation records across 10,500 property-level rows.
- The current status summary contains 16,126 closed records and 861 open records.
- The most common violation types include Failure to Obtain Permit, Unsafe and Dangerous, Maintenance, Testing & Certification, and Unsafe Structures.
- Owner data is available for 88.5% of property-risk rows, which makes owner-level accountability analysis feasible for most modeled properties.
- The student ZIP context matches 20 ZIP codes. In the current summary, the Pearson correlation between students per property and violations per property is -0.255, so student concentration alone does not explain violation intensity at the ZIP level.
- Property-class visualizations show that violation rates vary by property class, with several commercial, mixed-use, and apartment categories showing higher violations per property.
- Model visualizations show that risk scores are better used to rank properties for limited inspection resources than to make absolute yes/no predictions.

## Reproducibility Notes

The project is reproducible because:

- The `Makefile` documents the full run order.
- Data collection and cleaning are implemented in Python modules under `src/`.
- Generated data, tables, figures, and HTML outputs use stable paths.
- Tests are organized to mirror the source tree.
- GitHub Actions runs the test suite automatically.
- The README lists both the commands and the expected artifacts.

If rerunning from scratch, the exact row counts may change when public city datasets are refreshed. The pipeline structure and output paths should remain stable.

## Automated Tests

The test suite covers representative data, context, modeling, analysis, and visualization behavior.

Important tests:

| Test path | Coverage |
|---|---|
| `tests/data/test_violations.py` | Violation cleaning behavior. |
| `tests/data/test_features.py` | Property key and feature engineering behavior. |
| `tests/data/context/` | Context loading and enrichment logic. |
| `tests/modeling/test_baseline_model.py` | Baseline modeling behavior. |
| `tests/modeling/test_improved_model.py` | Improved model feature and evaluation behavior. |
| `tests/analysis/test_eda_and_summary.py` | EDA and summary generation. |
| `tests/viz/` | Static and interactive visualization generation. |

Run all tests with:

```bash
make test
```

