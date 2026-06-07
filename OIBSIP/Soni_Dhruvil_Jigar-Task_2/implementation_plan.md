# Production-Grade Unemployment Analysis Data Science Project

This project focuses on performing an Exploratory Data Analysis (EDA) and Time-Series Analysis of unemployment rate trends in India, specifically investigating the impact of the Covid-19 pandemic. Using historical regional data, we aim to uncover key insights regarding how unemployment changed across regions, areas (Rural vs. Urban), and periods (Pre-Covid vs. During/Post-Covid).

## User Review Required

> [!IMPORTANT]
> The dataset CSV `Unemployment in India.csv` currently resides in the root workspace folder `C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\unemployment-analysis`. In the implementation, we will automatically create the `data/` directory and move/copy the CSV into `data/Unemployment in India.csv` to match the required project folder structure.

> [!NOTE]
> The CSV contains empty rows in the middle and at the end of the file. The pipeline will drop rows where vital fields (like `Date`, `Region`, or `Estimated Unemployment Rate (%)`) are missing or empty.

## Open Questions

* No open questions. The project requirements and directory structure are fully specified.

## Proposed Changes

### Project Structure (Directory Tree)
```
unemployment-analysis/
├── data/                              ← dataset CSV moved here
│   └── Unemployment in India.csv
├── notebooks/
│   └── 01_Unemployment_Analysis.ipynb  ← Interactive analysis notebook
├── src/
│   ├── __init__.py
│   ├── config.py                      ← Project paths & configuration dataclass
│   ├── data_loader.py                 ← Load, trim, parse, & sort data
│   ├── preprocessor.py                ← Impute, standardise, and engineer features
│   ├── eda.py                         ← Visualisations for distribution & comparisons
│   ├── time_series_analysis.py        ← Analysis of national trends, rolling averages
│   ├── regional_analysis.py           ← Region and Area comparisons, Covid impact calculation
│   └── report_generator.py            ← Generates the final markdown report
├── reports/
│   ├── figures/                       ← Visualisations (01_ to 14_) saved here
│   └── unemployment_insights.md       ← Generated markdown report
├── logs/
│   └── analysis.log                   ← Analysis runtime log
├── tests/
│   └── test_pipeline.py               ← Pipeline validation tests (pytest)
├── main.py                            ← Command-line interface entrypoint
├── requirements.txt                   ← Python dependencies
├── README.md                          ← Project overview & installation guide
└── RUNNING_GUIDE.md                   ← Detailed running instructions
```

---

### Configuration & Data Pipeline

#### [NEW] [config.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/config.py)
- Defines config dataclass.
- Uses `pathlib.Path` for cross-platform support.
- Automatically creates `data/`, `reports/`, `reports/figures/`, `logs/`, `notebooks/` directories if they do not exist.
- Defines constants:
  - `DATA_PATH`: `C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\unemployment-analysis\data\`
  - `COVID_START`: `"2020-03-01"`
  - Column mappings (Date, Rate, Region, Area).

#### [NEW] [data_loader.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/data_loader.py)
- Loads dataset from CSV.
- Automatically strips leading/trailing whitespaces from column names and string contents.
- Parses dates using multiple format support (e.g. `%d-%m-%Y`).
- Drops completely empty/blank rows.
- Sorts DataFrame chronologically by Date.
- Logs metadata: shape, dtypes, date range, null counts, unique regions.

#### [NEW] [preprocessor.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/preprocessor.py)
- Imputes missing values: forward-fill for time-series columns (`Estimated Unemployment Rate (%)`, etc.) and mode for categorical columns.
- Standardises case structure of `Region` and `Area` columns to Title Case.
- Extracts derived features: `month`, `year`, `quarter` (Q1-Q4), and `covid_period` ("Pre-Covid" vs. "During/Post-Covid").
- Detects outliers using standard IQR method (flagged, not dropped).

---

### Analysis & Reporting Components

#### [NEW] [eda.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/eda.py)
Generates and saves the following plots to `reports/figures/`:
1. `01_unemployment_rate_distribution.png`: Histogram + KDE.
2. `02_unemployment_by_area.png`: Boxplot comparing Rural vs. Urban rates.
3. `03_unemployment_by_region.png`: Horizontal bar chart of average rates by region.
4. `04_labour_participation_vs_unemployment.png`: Scatter plot with regression line.
5. `05_correlation_heatmap.png`: Numeric correlation heatmap.
6. `06_covid_period_comparison.png`: Boxplot comparing Pre-Covid and During/Post-Covid rates.
- Returns statistics dictionary for report generation.

#### [NEW] [time_series_analysis.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/time_series_analysis.py)
Generates and saves the following plots to `reports/figures/`:
7. `07_unemployment_rate_over_time.png`: National average line plot with vertical dashed line for Covid start.
8. `08_monthly_trend_by_region.png`: Multi-line plot for top 5 most affected regions.
9. `09_rolling_average.png`: 3-month rolling average line plot.
10. `10_pre_vs_post_covid_trend.png`: Side-by-side line plot comparing trends before and after Covid start.
- Calculates and returns key time-series metrics.

#### [NEW] [regional_analysis.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/regional_analysis.py)
Generates and saves the following plots to `reports/figures/`:
11. `11_top10_highest_unemployment_regions.png`: Bar chart of top 10 regions by average rate.
12. `12_top10_lowest_unemployment_regions.png`: Bar chart of bottom 10 regions by average rate.
13. `13_rural_vs_urban_by_region.png`: Grouped bar chart comparing Rural/Urban rates per region.
14. `14_covid_impact_by_region.png`: Bar chart showing % increase in rate pre vs during Covid per region, sorted.
- Returns ranked regional data dictionary.

#### [NEW] [report_generator.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/src/report_generator.py)
- Collects outputs from EDA, time-series, and regional analysis.
- Automatically generates the markdown file `reports/unemployment_insights.md` with:
  - Executive Summary.
  - Key Statistics table.
  - Covid-19 Impact section with % increase figures.
  - Top 5 Most and Least Affected regions.
  - Rural vs. Urban comparison.
  - Table of all generated visualisations.

---

### CLI, Notebook, & Tests

#### [NEW] [main.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/main.py)
- Command-line entrypoint using `argparse` flags:
  - `--analyse`: Runs the full pipeline.
  - `--eda-only`: Generates only EDA plots.
  - `--report`: Regenerates the insights report.
- Configures structured logging to console and `logs/analysis.log`.
- Outputs summary stats on pipeline completion.

#### [NEW] [01_Unemployment_Analysis.ipynb](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/notebooks/01_Unemployment_Analysis.ipynb)
- Multi-section notebook with clear markdown explanations and interactive cell execution.

#### [NEW] [test_pipeline.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/tests/test_pipeline.py)
- Integrates `pytest` tests validating data loading, preprocessor outputs, output directory creation, and report generation.

#### [NEW] [requirements.txt](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/requirements.txt)
- Defines pinned project dependencies.

#### [NEW] [README.md](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/README.md)
- Professional project description with installation instructions, usage, and structure.

#### [NEW] [RUNNING_GUIDE.md](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis/RUNNING_GUIDE.md)
- Complete step-by-step pipeline execution manual.

---

## Verification Plan

### Automated Tests
Run validation tests using `pytest`:
```bash
pytest tests/
```

### Manual Verification
1. Run pipeline:
   ```bash
   python main.py --analyse
   ```
2. Check that all 14 plots exist in `reports/figures/`.
3. Check that `reports/unemployment_insights.md` is populated with correct statistics.
4. Run Jupyter notebook to ensure it executes without errors.
