# University-Accountability-Ordinance
This project aims to enhance the transparency and accountability of off-campus student housing data reported by higher education institutions. By examining and standardizing data from the past decade, the project seeks to understand the impact on housing affordability and inform land use decisions. The goal of this project is to better understand where students live, what percentage of renters are students in each district, and what are the housing conditions like where students live. The initiative involves collaboration with the Inspectional Services Department to restore and clarify housing violation data, develop tools to identify problematic landlords, and integrate data across city departments. This ordinance will establish clear criteria for data collection and reporting, ultimately creating a publicly accessible database to ensure compliance and promote responsible property management.
## Project Overview
This project aims to enhance the transparency and accountability of off-campus student housing data reported by higher education institutions. By examining and standardizing data from the past decade, the project seeks to understand the impact on housing affordability and inform land use decisions.
## Key Research Questions
### Base Questions
1. What are the trends regarding student housing across the city, by district, e.g. what % of the rental housing is taken up by students for each district and how has this changed over time?
2. What are the housing conditions for students living off-campus? 
   - e.g. how many students per unit 
   - Is the unit managed by a “bad landlord’ e.g. how many building violations have student housing
3. What is the spectrum of violations and severity in regards to worst landlords classifications?
4. What landlords are non-compliant? Overall volume, severe violations.
   - Are there clusters of properties where landlords are non-compliant?
5. How has the value of these off campus housing options changed over time?
4. Data Needed & Collection Plan
## Datasets to Collect
We will collect and integrate the following datasets:
Building and Property Violations
311 Service Requests
SAM Addresses (Street Address Management)
Property Assessment Data
Student Housing Data (2016–2024)
Neighborhood shapefiles
City Council District shapefiles
### How We Will Collect It
Boston Open Data portal: download via API/CSV exports for violations, 311, SAM, and assessment datasets (and available boundaries). 
Client/course shared Google Drive: obtain Student Housing (2016–2024) and any supporting documentation. 
Versioning & provenance: store raw snapshots in data/raw/ with fetch timestamps; store cleaned outputs in data/processed/ with data dictionaries and transformation logs.
## Project Structure
```
UniversityAccountabilityOrdinance/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── src/
│   ├── ingest/
│   ├── clean/
│   ├── integrate/
│   ├── analysis/
│   └── viz/
├── reports/
│   ├── figures/
│   ├── early_insights/
│   └── final_report/
├── requirements.txt
└── README.md
```
## Team
xiaoxij@bu.edu
chez0212@bu.edu
zywang1@bu.edu