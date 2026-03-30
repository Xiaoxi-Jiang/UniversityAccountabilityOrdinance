# March Check-In Summary

## Project Framing
This pipeline now treats the baseline task as a property-level prediction problem: given each property's historical violations up to the cutoff date, predict whether it will receive a new high-risk violation during the next prediction window.

## Data Processing Progress
- Core cleaned violations table: `data/processed/violations_clean.csv`
- Property-level feature table: `data/processed/violations_feature_table_v1.csv`
- Enriched property-risk table: `data/processed/property_risk_table_v1.csv`
- Student housing context output: `data/processed/student_housing_summary_v1.csv`
- Property-key coverage relies primarily on normalized address and ZIP joins; 99.9% of rows use address-based keys and 0.1% fall back to case numbers.
- Context coverage from optional sources: SAM=yes, assessment=yes, parcels=yes, 311=yes, permits=yes, ACS=yes.
- Owner coverage in the property-risk table: 88.5% (9265 of 10471 properties).

## Modeling Method
- Target: `will_receive_high_risk_violation_next_period`, defined as at least one new `high risk (proxy)` violation within 365 days after the cutoff date 2025-03-27.
- Feature groups: historical volume (`total_violations`, `recent_violation_count_365d`), recency (`days_since_last_violation`), diversity (`distinct_violation_types`), and severity composition (`history_high_risk_violations`, `history_medium_risk_violations`, `history_low_risk_violations`, `history_high_risk_share`).
- Model: logistic regression with standardized numeric features and balanced class weights to offset the rare-event target.

Top positive coefficient directions:
- `history_high_risk_share` (0.964)
- `history_medium_risk_violations` (0.784)
- `recent_violation_count_365d` (0.417)

Top negative coefficient directions:
- `days_since_last_violation` (-0.937)
- `open_violations` (-0.613)
- `history_low_risk_violations` (-0.216)

## Student Housing Relationship
- Student housing is now analyzed directly with ZIP-level relationship outputs, including `student_housing_relationship.csv`, `student_housing_correlation_summary.csv`, and `student_housing_relationship.png`.
- Matched ZIP summary: 20 ZIP codes with student metric `all_students`; correlation with total violations = -0.4526, correlation with violations per property = -0.1762.

## Preliminary Results and Interpretation
- Modeling frame: 10176 properties, 37 positive examples (0.0036 positive rate).
- Holdout metrics: accuracy=0.8198, balanced_accuracy=0.7737, precision=0.0144, recall=0.7273, f1=0.0283, roc_auc=0.8425.
- Majority-class accuracy is 0.9964, so balanced accuracy and recall are more informative than raw accuracy because the target event is rare.
- Current limitations: severity is still proxy-based rather than an official city severity field; student housing is measured mostly at ZIP level; and the baseline model only uses historical violation behavior, not the full static property context yet.
- Student housing context is therefore informative for exploration, but not yet evidence of a causal relationship between student concentration and violations.
