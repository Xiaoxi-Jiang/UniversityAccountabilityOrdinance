# University Accountability Ordinance

本项目已经实现从数据采集到风险评估与空间分析的完整流程，目标是评估 Boston 校外学生住房在 district 维度的规模、违规风险、房东责任与时序变化。

## 1. 已完成能力
1. Boston Open Data 抓取脚本（violations / 311 / SAM / assessment）。
2. 学生住房数据标准化清洗（含经纬度字段）。
3. 多源主键统一（student + SAM + assessment -> `property_key`）。
4. 违规与 311 融合风险模型（严重度加权 + 时间衰减）。
5. bad landlord 识别（按聚合风险阈值）。
6. district 年度趋势表。
7. GIS 空间聚合（GeoJSON polygon 与点位匹配，支持属性回退）。
8. 可视化产物（年度趋势 SVG）。
9. 数据质量检查脚本与自动化测试（unittest）。

## 2. 项目结构
```
UniversityAccountabilityOrdinance/
├── data/
│   ├── raw/
│   │   ├── student_housing.csv
│   │   ├── violations.csv
│   │   ├── service_requests_311.csv
│   │   ├── sam_addresses.csv
│   │   ├── property_assessment.csv
│   │   └── city_council_districts.geojson
│   └── processed/
├── reports/
│   ├── student_housing_summary.md
│   ├── spatial_district_summary.md
│   └── figures/
├── src/
│   ├── common/
│   ├── data/
│   ├── analysis/
│   └── viz/
└── tests/
```

## 3. 一步一步运行

### Step 0: 进入项目目录
```bash
cd /Users/macofzywang/Downloads/UniversityAccountabilityOrdinance-main
```

### Step 1: （可选）创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: 安装依赖
```bash
pip install -r requirements.txt
```

### Step 3: 下载 Boston Open Data（可选）
```bash
python3 src/data/fetch_boston_data.py --datasets all
```

### Step 4: 准备输入数据
将真实文件放入以下路径：
- `data/raw/student_housing.csv`
- `data/raw/violations.csv`
- `data/raw/service_requests_311.csv`
- `data/raw/sam_addresses.csv`
- `data/raw/property_assessment.csv`
- `data/raw/city_council_districts.geojson`

无真实数据时，可先用样例：
```bash
cp data/raw/student_housing_template.csv data/raw/student_housing.csv
cp data/raw/violations_sample.csv data/raw/violations.csv
cp data/raw/service_requests_311_sample.csv data/raw/service_requests_311.csv
cp data/raw/sam_addresses_sample.csv data/raw/sam_addresses.csv
cp data/raw/property_assessment_sample.csv data/raw/property_assessment.csv
cp data/raw/city_council_districts_sample.geojson data/raw/city_council_districts.geojson
```

### Step 5: 清洗学生住房数据
```bash
python3 src/data/prepare_student_housing.py \
  --input data/raw/student_housing.csv \
  --output data/processed/student_housing_clean.csv
```

### Step 6: 构建统一主键注册表
```bash
python3 src/data/build_property_registry.py \
  --student-housing data/processed/student_housing_clean.csv \
  --sam data/raw/sam_addresses.csv \
  --assessment data/raw/property_assessment.csv \
  --output data/processed/property_registry.csv
```

### Step 7: 生成基础 district 汇总报告
```bash
python3 src/analysis/generate_report.py \
  --student-housing data/processed/student_housing_clean.csv \
  --output-csv data/processed/district_summary.csv \
  --output-md reports/student_housing_summary.md
```

### Step 8: 风险模型（violations + 311）
```bash
python3 src/analysis/integrated_risk_model.py \
  --registry data/processed/property_registry.csv \
  --violations data/raw/violations.csv \
  --service-311 data/raw/service_requests_311.csv \
  --output-property data/processed/property_risk_model.csv \
  --output-landlord data/processed/landlord_risk_model.csv
```

### Step 9: 兼容版违规关联（仅 student + violations）
```bash
python3 src/analysis/link_violations.py \
  --student-housing data/processed/student_housing_clean.csv \
  --violations data/raw/violations.csv \
  --output-linked data/processed/student_housing_with_violations.csv \
  --output-landlord data/processed/landlord_risk_summary.csv
```

### Step 10: 年度趋势
```bash
python3 src/analysis/yearly_trend.py \
  --student-housing data/processed/student_housing_clean.csv \
  --output data/processed/district_yearly_trend.csv
```

### Step 11: GIS 空间风险聚合
```bash
python3 src/analysis/spatial_district_analysis.py \
  --property-risk data/processed/property_risk_model.csv \
  --district-geojson data/raw/city_council_districts.geojson \
  --output-csv data/processed/spatial_district_risk.csv \
  --output-md reports/spatial_district_summary.md
```

### Step 12: 生成趋势图
```bash
python3 src/viz/plot_yearly_trend_svg.py \
  --input data/processed/district_yearly_trend.csv \
  --output reports/figures/district_yearly_trend.svg
```

### Step 13: 数据质量检查
```bash
python3 src/data/validate_data_quality.py \
  --student-clean data/processed/student_housing_clean.csv \
  --property-risk data/processed/property_risk_model.csv
```

### Step 14: 运行测试
```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## 4. 主要输出文件
- `data/processed/student_housing_clean.csv`
- `data/processed/property_registry.csv`
- `data/processed/district_summary.csv`
- `data/processed/property_risk_model.csv`
- `data/processed/landlord_risk_model.csv`
- `data/processed/student_housing_with_violations.csv`
- `data/processed/landlord_risk_summary.csv`
- `data/processed/district_yearly_trend.csv`
- `data/processed/spatial_district_risk.csv`
- `reports/student_housing_summary.md`
- `reports/spatial_district_summary.md`
- `reports/figures/district_yearly_trend.svg`

## Team
- xiaoxij@bu.edu
- chez0212@bu.edu
- zywang1@bu.edu
