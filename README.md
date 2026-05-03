# EV Charging Infrastructure & Range Anxiety — Data Engineering Pipeline

**Course:** CS-GY 6513 – Big Data | NYU Tandon School of Engineering  
**Semester:** Spring 2026  
**Project:** EV Charging Infrastructure & Range Anxiety: A Big Data Gap Analysis for Equitable Electric Mobility

**Full Team:**
| Name | NetID | Role |
|------|-------|------|
| Krish Jani | kj2743 | Data Ingestion, Cleaning & Engineering |
| Vandana Rawat | vr2645 | Exploratory Analysis & Visualizations |
| Riddhi Raina Prasad | rrp4822 | Machine Learning (XGBoost Classification) |

> This README covers the **data engineering layer** of the project — everything from raw data acquisition to the final cleaned Parquet files consumed by the analysis and ML notebooks. If you are Vandana or Riddhi, start here to understand what data is available to you and how to load it.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Environment Setup](#environment-setup)
3. [Raw Data Sources](#raw-data-sources)
4. [Notebook Pipeline](#notebook-pipeline)
5. [Output Files](#output-files)
6. [What Was Filtered and Why](#what-was-filtered-and-why)
7. [What Was Transformed and Why](#what-was-transformed-and-why)
8. [Schema Reference](#schema-reference)
9. [Key Statistics](#key-statistics)
10. [Known Limitations](#known-limitations)
11. [For Vandana — How to Load Your Data](#for-vandana)
12. [For Riddhi — How to Load Your Data](#for-riddhi)

---

## Repository Structure

```
ev-infrastructure-equity-analyzer/
│
├── data/
│   ├── raw/                                    # Original unmodified source files
│   │   ├── alt_fuel_stations.csv               # AFDC EV charging station data
│   │   ├── acs_zcta_combined.csv               # Census ACS demographic data
│   │   ├── 2023_Gaz_zcta_national.txt          # Census ZCTA centroid coordinates
│   │   ├── 2023_Gaz_zcta_national.zip          # Zipped version of above
│   │   └── tab20_zcta520_state20_natl.txt      # ZCTA-to-state relationship reference
│   │
│   └── processed/                              # Spark output files (Parquet format)
│       ├── cleaned_stations.parquet/           # Cleaned station-level data
│       ├── stations_with_census.parquet/       # Stations + Census demographics
│       └── county_level_features.parquet/      # ZCTA-level ML feature table
│
├── notebooks/
│   ├── 01_data_loading_and_profiling.ipynb     # Data exploration and quality checks
│   ├── 02_data_cleaning.ipynb                  # Cleaning and feature engineering
│   ├── 03_census_join.ipynb                    # Join with Census demographics
│   └── 04_county_aggregation.ipynb             # ZCTA aggregation + desert detection
│
├── .gitignore
└── README.md
```

---

## Environment Setup

### Prerequisites

- Python 3.11 or higher (tested on 3.14.4)
- Java 17 — required for PySpark 4.x. Install via Homebrew on Mac:
  ```bash
  brew install openjdk@17
  ```
- macOS or Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ev-infrastructure-equity-analyzer.git
cd ev-infrastructure-equity-analyzer

# Create and activate virtual environment
python3 -m venv ev_env
source ev_env/bin/activate

# Install all dependencies
pip install pyspark jupyter pandas matplotlib plotly
```

### Launching Jupyter

Always activate the virtual environment before launching Jupyter.
If you skip this step, none of the installed packages will be found.

```bash
source ev_env/bin/activate
jupyter notebook
```

### Java Configuration (Required — Add to Top of Every Notebook)

Jupyter does not always pick up the system Java path. Add this as the
very first cell in every notebook before any Spark code:

```python
import os
os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']
```

### Run Order

Notebooks must be run in order — each one depends on the output of the previous:

```
01 → 02 → 03 → 04
```

Do not skip notebooks or run them out of order.

---

## Raw Data Sources

### Source 1: AFDC Alternative Fueling Station Locator

| Property | Value |
|----------|-------|
| Provider | U.S. Department of Energy — Alternative Fuels Data Center |
| Download URL | https://afdc.energy.gov/data_download |
| File | `data/raw/alt_fuel_stations.csv` |
| Parameters used | Fuel Type = Electric, Country = US, Format = CSV |
| Raw row count | 85,659 stations |
| Column count | 75 columns |
| Coverage | All publicly reported EV charging stations in the US, 2014–present |
| Update frequency | Regularly updated by AFDC; this is a point-in-time snapshot |

The AFDC aggregates station data from charging networks (ChargePoint, Tesla, Blink, etc.),
utilities, and station operators. It is the most comprehensive public database of US EV
charging infrastructure and is used directly in NEVI program planning.

**Key raw columns used in this project:**

| Column Name | Raw Type | Description |
|------------|----------|-------------|
| ID | integer | Unique station identifier |
| Station Name | string | Name of the charging location |
| City | string | City where station is located |
| State | string | State abbreviation |
| ZIP | string | ZIP code (may be fewer than 5 digits due to CSV parsing) |
| Latitude | double | Station latitude coordinate |
| Longitude | double | Station longitude coordinate |
| Access Code | string | "public" or "private" |
| Status Code | string | E=Open, T=Temporarily unavailable, P=Planned |
| Open Date | string | Date station opened, format: YYYY-MM-DD |
| EV Level1 EVSE Num | integer | Number of Level 1 ports (standard 120V outlet) |
| EV Level2 EVSE Num | integer | Number of Level 2 ports (240V, most common public type) |
| EV DC Fast Count | integer | Number of DCFC ports (fast charging) |
| EV Network | string | Charging network operator |
| EV Connector Types | string | Available connector standards |
| Country | string | Country code — all rows are "US" |

---

### Source 2: U.S. Census Bureau — ACS 5-Year Estimates

| Property | Value |
|----------|-------|
| Provider | U.S. Census Bureau — American Community Survey |
| URL | https://www.census.gov/data/developers/data-sets/acs-5year.html |
| File | `data/raw/acs_zcta_combined.csv` |
| Tables | DP03 (Economic Characteristics), DP05 (Demographic Estimates) |
| Geography level | ZCTA (ZIP Code Tabulation Area) |
| Raw row count | 33,772 ZCTAs |
| Column count | 3 columns |

**Important — ZCTA vs ZIP Code:**
Census data uses ZCTAs, not USPS ZIP codes. ZCTAs are geographic polygons
drawn by the Census Bureau for statistical reporting. They closely but not
perfectly align with postal ZIP codes. This causes a small join mismatch
(see Census Join section below).

**Raw columns:**

| Column | Raw Type | Description |
|--------|----------|-------------|
| zcta | integer | ZCTA identifier — loads as integer, losing leading zeros |
| total_population | integer | Total population of the ZCTA |
| median_household_income | double | Median household income in dollars |

**Known issue in raw data:** 3,154 ZCTAs have null `median_household_income`.
This is intentional Census suppression — for ZCTAs with very small populations,
the Census withholds income estimates for statistical privacy. These nulls are
carried through to processed outputs and documented as a known limitation.

---

### Source 3: Census ZCTA Gazetteer (2023)

| Property | Value |
|----------|-------|
| Provider | U.S. Census Bureau |
| URL | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/ |
| File | `data/raw/2023_Gaz_zcta_national.txt` |
| Format | Tab-separated text |
| Row count | 33,791 ZCTAs |

This file provides official geographic centroid coordinates for every ZCTA in the US.
It was needed because the charging desert analysis requires a lat/lon coordinate for
every ZCTA — including those with zero charging stations. Since station data only
provides coordinates for ZCTAs that already have chargers, the Gazetteer fills in
coordinates for all 22,104 ZCTAs with no stations.

**Key columns used:**

| Column | Description |
|--------|-------------|
| GEOID | ZCTA identifier |
| INTPTLAT | Official centroid latitude |
| INTPTLONG | Official centroid longitude (has trailing whitespace in raw file — handled in code) |

---

### Source 4: ZCTA-to-State Relationship File

| Property | Value |
|----------|-------|
| Provider | U.S. Census Bureau |
| URL | https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/ |
| File | `data/raw/tab20_zcta520_state20_natl.txt` |

Downloaded as a reference but the URL returned an HTML page at time of download.
State assignment was handled instead using a hardcoded ZIP prefix-to-state lookup
(see notebook 04 for implementation). This file is retained for reference only.

---

## Notebook Pipeline

### Notebook 01 — Data Loading and Profiling
**File:** `notebooks/01_data_loading_and_profiling.ipynb`
**Input:** Raw CSV files
**Output:** None (exploration only)
**Purpose:** Understand the data before touching it

This notebook loads both raw datasets into Spark DataFrames and systematically
profiles every key column. No data is modified or saved here. The profiling
results directly informed every cleaning decision made in notebook 02.

**What was profiled:**
- Null counts and percentages for every key column
- Unique values and distributions for categorical columns (Access Code, Status Code, EV Network)
- Latitude/longitude range validation against US geographic bounds
- Open Date format inspection
- ZIP code length distribution (revealed leading zero truncation)
- Rows where all three charger count columns are simultaneously null

**Key profiling findings that drove cleaning decisions:**

| Finding | Value | Action Taken |
|---------|-------|-------------|
| EV Level1 EVSE Num nulls | 99.2% | Fill with 0 — Level 1 is residential only |
| EV DC Fast Count nulls | 82.2% | Fill with 0 — most stations are Level 2 only |
| EV Level2 EVSE Num nulls | 16.5% | Fill with 0 |
| Open Date nulls | 0.6% (531 rows) | Keep rows, exclude from time analysis only |
| City/ZIP nulls | 6 rows | Drop — unusable for geographic analysis |
| Status Code = T or P | ~2,009 rows | Drop — not real open infrastructure |
| Access Code = private | 5,695 rows | Drop — not public infrastructure |
| ZIP codes under 5 digits | 150 rows | Zero-pad to fix leading zero truncation |
| All charger counts null | 17 rows | Drop — station with zero chargers is invalid |
| 1 row outside US lat/lon range | Puerto Rico, Status=T | Handled by status filter |

---

### Notebook 02 — Data Cleaning
**File:** `notebooks/02_data_cleaning.ipynb`
**Input:** `data/raw/alt_fuel_stations.csv`
**Output:** `data/processed/cleaned_stations.parquet`

Applies all cleaning filters and feature engineering transformations to produce
the foundational clean dataset used by all downstream work.

**Row count:** 85,659 → 78,085 (7,574 rows removed total)

---

### Notebook 03 — Census Join
**File:** `notebooks/03_census_join.ipynb`
**Input:** `data/processed/cleaned_stations.parquet` + `data/raw/acs_zcta_combined.csv`
**Output:** `data/processed/stations_with_census.parquet`

Joins station-level data with Census demographic data at the ZIP/ZCTA level.

**Join type:** LEFT join — all 78,085 stations are preserved even if no Census match exists.

**The join key problem:**
Census zcta loads as an integer, so "00601" becomes 601. AFDC ZIP is a string "00601".
These don't match directly. Fix: cast Census zcta to string and zero-pad to 5 characters
before joining, so both sides use "00601" as the key.

**Join results:**
- Matched: 76,695 stations (98.2%)
- Unmatched: 1,390 stations (1.8%) — typically commercial/industrial ZIPs
  that the Census Bureau doesn't assign a ZCTA to (airports, warehouses, etc.)

**Row count:** 78,085 → 78,085 (left join, no rows lost)

---

### Notebook 04 — County Aggregation and Desert Detection
**File:** `notebooks/04_county_aggregation.ipynb`
**Input:** `data/processed/stations_with_census.parquet` + Gazetteer + Raw Census
**Output:** `data/processed/county_level_features.parquet`

Builds the ZCTA-level feature table for Riddhi's ML model, including the
charging desert binary target variable.

**Critical design decision — base table:**
The table is built starting from ALL 33,772 Census ZCTAs, not from stations.
If we had started from stations, zero-station ZCTAs (the actual deserts) would
never appear in the output. The Census base ensures every ZCTA gets a row,
with zeros for all station counts when no stations exist.

**Charging desert detection methodology:**
For every ZCTA, we compute the straight-line distance to the nearest DCFC
(fast charger) station using the Haversine formula. A ZCTA is classified as
a charging desert if that distance exceeds 50 miles.

The 50-mile threshold is the standard used in EV infrastructure research
and NEVI program planning documents.

**Haversine formula:**
Calculates great-circle distance between two lat/lon points on a sphere.
Implemented as a PySpark UDF with all 14,461 DCFC coordinates broadcasted
to Spark worker nodes for efficient parallel computation.

**Coordinate coverage:**
ZCTAs with stations use the average lat/lon of their stations as the centroid.
ZCTAs with zero stations use the official Census Gazetteer centroid as fallback.
This achieves 0 null coordinates across all 33,666 ZCTAs after filtering.

**State assignment for zero-station ZCTAs:**
ZCTAs with no stations had no State value from the station join.
State was assigned using a hardcoded ZIP prefix-to-state mapping
(first 3 digits of ZIP code reliably identify the state per USPS assignment).
132 ZCTAs for US territories (Guam, USVI, etc.) remain with null State.

**Puerto Rico removal:**
Puerto Rico ZCTAs (State = 'PR') are excluded. PR is outside the scope
of the continental US NEVI program analysis.

**Row count:** 33,772 Census ZCTAs → 33,666 (106 Puerto Rico ZCTAs removed)

---

## Output Files

### cleaned_stations.parquet
**Path:** `data/processed/cleaned_stations.parquet/`
**Rows:** 78,085 | **Columns:** 23
**Who uses it:** Vandana (all visualizations and geographic analysis)

```python
df = spark.read.parquet("data/processed/cleaned_stations.parquet")
```

---

### stations_with_census.parquet
**Path:** `data/processed/stations_with_census.parquet/`
**Rows:** 78,085 | **Columns:** 25
**Who uses it:** Vandana (income-based analysis) + Riddhi (station-level features)

```python
df = spark.read.parquet("data/processed/stations_with_census.parquet")
```

---

### county_level_features.parquet
**Path:** `data/processed/county_level_features.parquet/`
**Rows:** 33,666 | **Columns:** 17
**Who uses it:** Riddhi (XGBoost ML model — this is her primary input)

```python
df = spark.read.parquet("data/processed/county_level_features.parquet")
```

---

## What Was Filtered and Why

| Filter | Rows Removed | Reason |
|--------|-------------|--------|
| Status Code != 'E' | ~2,009 | T=Temporarily unavailable, P=Planned. Neither represents functional public infrastructure available to EV drivers today. |
| Access Code != 'public' | ~5,695 | Private chargers at corporate campuses, gated communities, or restricted facilities do not serve the public and should not count toward infrastructure coverage. |
| Null City or ZIP | 6 | Without a city or ZIP code, these records cannot be used in any geographic aggregation or Census join. |
| All charger counts = 0 after null fill | 17 | A charging station with zero Level 1, zero Level 2, and zero DCFC ports is not a functioning charging station. Most had Status T or P anyway. |
| State = 'PR' (Puerto Rico) | 35 | Puerto Rico is a US territory but outside the scope of the NEVI continental US state-level analysis. PR also lacks DCFC coverage which would artificially inflate desert counts. |

---

## What Was Transformed and Why

| Transformation | Before | After | Reason |
|---------------|--------|-------|--------|
| ZIP zero-padding | "601" | "00601" | CSV parsers interpret ZIP codes as integers and silently drop leading zeros. Puerto Rico (006xx, 007xx) and Northeast ZIPs (e.g., 01234) are affected. Without this fix, the Census join fails for ~150 stations. |
| Null charger counts → 0 | NULL | 0 | In the AFDC schema, null in a charger count column means that station has no ports of that type — not that data is missing. Null and zero carry the same meaning here. |
| Open Date string → DateType | "2022-03-15" | 2022-03-15 | Enables Spark date functions like year() and month() extraction. |
| install_year extracted | (not present) | 2022 | Vandana needs this as a direct integer column for groupBy in time-series charts. |
| install_month extracted | (not present) | 3 | Same reason — direct integer for groupBy. |
| State → uppercase | "Ca" | "CA" | Inconsistent casing causes groupBy to treat "CA" and "Ca" as different states. |
| City → initcap | "los angeles" | "Los Angeles" | Consistent display formatting for charts. |
| EV Network → trimmed | "ChargePoint " | "ChargePoint" | Trailing whitespace causes groupBy to split one network into multiple groups. |
| total_ports added | (computed) | integer | Sum of Level1 + Level2 + DCFC counts per station. Used for density calculations. |
| charger_level added | (not present) | "DCFC"/"Level2"/"Level1" | Classifies each station by its primary charger type. A station with any DCFC is classified DCFC regardless of what else it has, since DCFC is what matters most for range anxiety. |
| is_dcfc added | (not present) | 1 or 0 | Binary flag Riddhi needs for station-level features in her ML model. |
| region added | (not present) | "West"/"South"/etc. | Maps State to US Census Bureau region. Required for Vandana's regional comparison charts. |
| Census zcta integer → string | 601 | "00601" | Census loads zcta as integer, dropping leading zeros. Must be converted back to zero-padded string to match AFDC ZIP format for the join key to work correctly. |
| chargers_per_10k added | (computed) | double | (total_stations / total_population) × 10,000. Raw count is misleading — 100 chargers in a city of 1 million is worse than 10 chargers in a town of 5,000. |
| nearest_dcfc_miles added | (computed) | double | Haversine distance from ZCTA centroid to nearest DCFC station. The core metric for charging desert classification. |
| is_charging_desert added | (computed) | 1 or 0 | TARGET VARIABLE for Riddhi's XGBoost model. 1 if nearest_dcfc_miles > 50, else 0. |

---

## Schema Reference

### cleaned_stations.parquet — Full Column List

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| ID | string | No | Unique station identifier |
| Station Name | string | No | Name of the charging location |
| City | string | No | City (initcap, trimmed) |
| State | string | No | 2-letter state code (uppercase) |
| ZIP | string | No | 5-digit zero-padded ZIP code |
| Latitude | double | No | Station latitude |
| Longitude | double | No | Station longitude |
| Access Code | string | No | Always "public" after filtering |
| Status Code | string | No | Always "E" (open) after filtering |
| Open Date | string | Yes | Original date string from AFDC |
| EV Level1 EVSE Num | integer | No | Level 1 port count (0 if none) |
| EV Level2 EVSE Num | integer | No | Level 2 port count (0 if none) |
| EV DC Fast Count | integer | No | DCFC port count (0 if none) |
| EV Network | string | No | Network operator name (trimmed) |
| EV Connector Types | string | Yes | Available connector types |
| Country | string | No | Always "US" after filtering |
| open_date_parsed | date | Yes | Parsed installation date (null for 234 stations with unparseable dates) |
| install_year | integer | Yes | Year of installation (null if open_date_parsed is null) |
| install_month | integer | Yes | Month of installation (null if open_date_parsed is null) |
| total_ports | integer | No | Sum of Level1 + Level2 + DCFC ports |
| charger_level | string | No | Primary type: "DCFC" / "Level2" / "Level1" |
| is_dcfc | integer | No | 1 if station has any DCFC ports, else 0 |
| region | string | No | Census region: Northeast / South / Midwest / West |

---

### stations_with_census.parquet — Additional Columns

All 23 columns from cleaned_stations.parquet, plus:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| total_population | integer | Yes | Total population of the station's ZCTA. Null for the 1,390 stations whose ZIP didn't match any ZCTA. |
| median_household_income | double | Yes | Median household income of the station's ZCTA. Null for unmatched stations AND for 3,154 ZCTAs where Census suppressed the value. |

---

### county_level_features.parquet — Full Column List

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| ZIP_ZCTA | string | No | 5-digit ZCTA identifier (primary key) |
| total_population | integer | Yes | Total ZCTA population from Census |
| median_household_income | double | Yes | Median household income (null if Census-suppressed) |
| State | string | Yes | 2-letter state code. Null for 132 territory ZCTAs |
| region | string | Yes | Census region. Null for 132 territory ZCTAs |
| total_stations | integer | No | Total public open EV stations in ZCTA (0 if none) |
| dcfc_stations | integer | No | Number of stations with DCFC in ZCTA (0 if none) |
| total_ports | integer | No | Total charging ports in ZCTA (0 if none) |
| total_level2_ports | integer | No | Total Level 2 ports in ZCTA (0 if none) |
| total_dcfc_ports | integer | No | Total DCFC ports in ZCTA (0 if none) |
| zip_lat | double | Yes | Average latitude of stations in ZCTA. Null for zero-station ZCTAs. |
| zip_lon | double | Yes | Average longitude of stations in ZCTA. Null for zero-station ZCTAs. |
| chargers_per_10k | double | Yes | EV stations per 10,000 population. Null if population is null or 0. |
| final_lat | double | No | ZCTA centroid latitude. Station average if available, Gazetteer centroid otherwise. Never null. |
| final_lon | double | No | ZCTA centroid longitude. Station average if available, Gazetteer centroid otherwise. Never null. |
| nearest_dcfc_miles | double | No | Straight-line miles from ZCTA centroid to the nearest DCFC station. |
| is_charging_desert | integer | No | **TARGET VARIABLE.** 1 if nearest_dcfc_miles > 50, else 0. |

---

## Key Statistics

### Raw Data
| Metric | Value |
|--------|-------|
| Raw AFDC station records | 85,659 |
| Raw Census ZCTAs | 33,772 |
| AFDC columns | 75 |

### After Cleaning (cleaned_stations.parquet)
| Metric | Value |
|--------|-------|
| Clean public open stations | 78,085 |
| Rows removed | 7,574 |
| Stations with DCFC | 14,488 (18.5%) |
| Stations Level 2 only | 63,543 (81.4%) |
| Stations Level 1 only | 89 (0.1%) |
| Stations with null open date | 234 (0.3%) |
| States covered | 50 + DC |

### After Census Join (stations_with_census.parquet)
| Metric | Value |
|--------|-------|
| Stations matched to Census | 76,695 (98.2%) |
| Stations unmatched | 1,390 (1.8%) |
| ZCTAs with suppressed income | 3,154 |

### Desert Analysis (county_level_features.parquet)
| Metric | Value |
|--------|-------|
| Total US ZCTAs analyzed | 33,666 |
| ZCTAs with zero stations | 22,104 (65.7%) |
| ZCTAs with at least one station | 11,562 (34.3%) |
| Charging deserts (>50 mi from DCFC) | 620 (1.8%) |
| Adequate coverage | 33,046 (98.2%) |
| Median distance to nearest DCFC | 8.15 miles |
| 90th percentile distance | 25.49 miles |
| Maximum distance (remote Alaska) | 676.27 miles |

### Desert by Region
| Region | Desert ZCTAs | Total ZCTAs | Desert % |
|--------|-------------|-------------|---------|
| West | 376 | 5,807 | 6.5% |
| Midwest | 181 | 10,153 | 1.8% |
| South | 57 | 11,600 | 0.5% |
| Northeast | 6 | 6,106 | 0.1% |

---

## Known Limitations

**1. Straight-line vs road distance**
The Haversine formula computes great-circle (straight-line) distance. It does not
account for roads, terrain, or actual driving routes. A ZCTA centroid 40 miles
straight-line from a DCFC station may be 80+ miles by road in mountainous terrain.
This means the 1.8% desert rate likely undercounts true road-based charging deserts.

**2. 1,390 unmatched ZIP codes**
These stations (1.8% of the clean dataset) have no Census demographic data.
Most are in commercial, industrial, or airport ZIP codes that the Census Bureau
does not assign a ZCTA to. These stations are included in geographic analysis
but their ZCTA-level income and population are null.

**3. 3,154 suppressed income values**
Census statistically suppresses median_household_income for very low population ZCTAs.
These nulls are intentional and cannot be recovered from the source.
Riddhi's ML model should handle these via imputation (recommended: state-level median).

**4. 132 US territory ZCTAs**
Guam, US Virgin Islands, American Samoa, and Northern Mariana Islands ZCTAs are present
with null State and region. Excluded from regional analysis but counted in total rows.

**5. Static snapshot**
The AFDC dataset is a point-in-time download from Spring 2026. Station counts change daily.

**6. ZCTA centroid approximation**
For ZCTAs with stations, the centroid is the average lat/lon of stations — not the true
geographic center of the ZCTA polygon. Difference is negligible for most ZCTAs.

---

## For Vandana

Your primary input is `cleaned_stations.parquet`. Load it like this:

```python
import os
from pyspark.sql import functions as F

BASE_DIR = os.getcwd()
df = spark.read.parquet(os.path.join(BASE_DIR, "data/processed/cleaned_stations.parquet"))
```

**Columns you will use most:**

| Your Analysis | Columns to Use |
|--------------|----------------|
| Chargers by state | `State`, `total_ports`, `total_stations` |
| Chargers by city | `City`, `State` |
| Installations over time (yearly) | `install_year` — already extracted, just groupBy |
| Installations by month | `install_month` — already extracted, just groupBy |
| Charger type distribution | `charger_level`, `EV DC Fast Count`, `EV Level2 EVSE Num` |
| Network distribution | `EV Network` |
| Regional analysis | `region` |
| Charging desert by state/region | Use `county_level_features.parquet` instead |
| Income-based analysis | Use `stations_with_census.parquet` — has `median_household_income` |

**Important tip on time analysis:**
234 stations have null `open_date_parsed`. Always filter these out before
time-based groupBy operations:

```python
df.filter(F.col("install_year").isNotNull()).groupBy("install_year").count()
```

For geographic charts (state counts, city counts) you do NOT need to filter them out.

---

## For Riddhi

Your primary input is `county_level_features.parquet`. Load it like this:

```python
import os
from pyspark.sql import functions as F

BASE_DIR = os.getcwd()
df = spark.read.parquet(os.path.join(BASE_DIR, "data/processed/county_level_features.parquet"))
```

**Target variable:** `is_charging_desert` (1 = desert, 0 = adequate coverage)

**Feature columns for your XGBoost model:**

| Feature | Type | Notes |
|---------|------|-------|
| total_population | integer | No nulls |
| median_household_income | double | 3,154 nulls — impute with state median |
| total_stations | integer | 0 = no chargers in this ZCTA |
| dcfc_stations | integer | 0 = no DCFC in this ZCTA |
| total_ports | integer | 0 = no ports |
| chargers_per_10k | double | Some nulls where population = 0 — fill with 0 |
| nearest_dcfc_miles | double | No nulls — complete coverage |
| State | string | Encode as categorical — 132 nulls, drop or encode as "Other" |
| region | string | Encode as categorical — 132 nulls, same as State |

**Class imbalance warning:**
Only 1.8% of ZCTAs are deserts (620 of 33,666). Set XGBoost's
`scale_pos_weight` to approximately `33,046 / 620 ≈ 53`, or use SMOTE.

**Suggested preprocessing before model training:**
1. Drop the 132 rows where `State` is null (US territory ZCTAs)
2. Impute null `median_household_income` with state-level median income
3. Fill null `chargers_per_10k` with 0
4. One-hot encode or ordinal encode `State` and `region`
5. Set `scale_pos_weight` in XGBoost to handle class imbalance

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14.4 | Primary language |
| Apache Spark | 4.1.1 | Distributed data processing |
| PySpark | 4.1.1 | Python API for Spark |
| SparkSQL | 4.1.1 | SQL-style aggregations and joins |
| Parquet + Snappy | — | Compressed columnar output format |
| Jupyter Notebook | — | Interactive development environment |
| Java | OpenJDK 17.0.19 | Required runtime for Spark |
| XGBoost | TBD | ML classification (Riddhi) |
| Plotly / Matplotlib | TBD | Visualizations (Vandana) |

---

## References

1. AFDC Data Download: https://afdc.energy.gov/data_download
2. AFDC Station Map: https://afdc.energy.gov/stations/
3. NEVI Program Overview: https://www.fhwa.dot.gov/environment/nevi/
4. Census ACS Data: https://www.census.gov/data/developers/data-sets/acs-5year.html
5. Census Gazetteer Files: https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
6. PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
