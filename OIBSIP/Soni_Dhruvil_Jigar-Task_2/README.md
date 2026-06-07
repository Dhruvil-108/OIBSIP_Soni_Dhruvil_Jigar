# Unemployment Analysis in India

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

A production-grade Data Science project analyzing monthly unemployment rates in India across various states and regions. This project focuses on performing rigorous Exploratory Data Analysis (EDA) and Time-Series Analysis, with a specific focus on evaluating the economic shock and recovery associated with the Covid-19 pandemic.

## Project Overview

Unemployment is a key indicator used to assess the health of an economy. The onset of the Covid-19 pandemic in early 2020 and the subsequent lockdowns significantly affected the labor market. This project builds a modular, maintainable, and testable analysis pipeline to load, preprocess, analyze, and visualize unemployment trends, generating a final insights report to aid understanding of:
- General distribution of unemployment rates across regions and areas (Rural vs. Urban).
- Monthly national trends, including the impact of the Covid-19 lockdowns.
- Relative and absolute surges in unemployment within individual states.

## Tech Stack

| Library | Pinned Version | Purpose |
| --- | --- | --- |
| **pandas** | `>=2.0` | Data manipulation, loading, and time-series grouping |
| **numpy** | `>=1.24` | Mathematical calculations and array operations |
| **matplotlib** | `>=3.7` | Base visualisations and chart generation |
| **seaborn** | `>=0.12` | Statistical visualisations with premium aesthetics |
| **plotly** | `>=5.15` | Interactive graphing and dashboards |
| **pytest** | `>=7.4` | Automated test suite execution |
| **notebook** | `>=7.0` | Interactive Jupyter Notebook exploration |

## Project Structure

```
unemployment-analysis/
├── data/                              ← Contains dataset CSV
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

## Dataset

The analysis utilizes the existing dataset file `Unemployment in India.csv` containing monthly observations:
- **Region**: The state in India.
- **Date**: The date of observation (monthly frequency).
- **Frequency**: Observation frequency (e.g. Monthly).
- **Estimated Unemployment Rate (%)**: Percentage of the active labour force that is unemployed.
- **Estimated Employed**: Number of employed individuals.
- **Estimated Labour Participation Rate (%)**: Percentage of the working-age population active in the labour market.
- **Area**: Geographic designation (`Rural` or `Urban`).

## Setup & Installation

1. Navigate to the project root:
   ```bash
   cd unemployment-analysis
   ```

2. Activate the existing virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
   *(Note: If the virtual environment is in the parent directory, activate it from there, e.g. `..\.venv\Scripts\activate`)*

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

The pipeline coordinates steps through `main.py`:

- **Run Full Analysis Pipeline**:
  ```bash
  python main.py --analyse
  ```
  *This loads/preprocesses the dataset, executes EDA, time-series, and regional modules, saves 14 plots to `reports/figures/`, caches stats, and outputs the markdown insights report.*

- **Generate EDA Plots Only**:
  ```bash
  python main.py --eda-only
  ```
  *Generates and saves the first 6 EDA plots (01–06).*

- **Regenerate Insights Report**:
  ```bash
  python main.py --report
  ```
  *Quickly rewrites `reports/unemployment_insights.md` using the cached stats.*

## Key Findings

- **National Covid Impact:** The national average unemployment rate increased by over **69%** during the pandemic, rising from a pre-lockdown baseline.
- **Lockdown Peak:** National average unemployment peaked in **May 2020**, corresponding to the most restrictive phase of nationwide Covid-19 lockdowns.
- **Urban vs Rural Disparity:** Urban unemployment consistently exceeded rural unemployment (by ~2.3%), showing that city services and industrial jobs were harder hit by physical closures.
- **Highest Impact Region:** Puducherry experienced the highest relative percentage surge, while Tripura and Haryana recorded the highest overall average unemployment rates.

## Sample Outputs

The pipeline saves 14 distinct visualisations in `reports/figures/`:
- `01_unemployment_rate_distribution.png` (Histogram & KDE)
- `02_unemployment_by_area.png` (Rural vs. Urban boxplot)
- `03_unemployment_by_region.png` (Average unemployment per state)
- `04_labour_participation_vs_unemployment.png` (Scatter plot with regression line)
- `05_correlation_heatmap.png` (Heatmap of numeric features)
- `06_covid_period_comparison.png` (Pre vs. Post Covid boxplot)
- `07_unemployment_rate_over_time.png` (Line plot with lockdown marker line)
- `08_monthly_trend_by_region.png` (Line plot of top 5 affected states)
- `09_rolling_average.png` (Monthly vs. 3-month rolling average)
- `10_pre_vs_post_covid_trend.png` (Side-by-side pre/post trend lines)
- `11_top10_highest_unemployment_regions.png` (Bar chart of top 10 states)
- `12_top10_lowest_unemployment_regions.png` (Bar chart of bottom 10 states)
- `13_rural_vs_urban_by_region.png` (Grouped bar chart for Rural vs. Urban)
- `14_covid_impact_by_region.png` (Percentage change in rate pre vs post Covid per state)

## Contributing & License

This project is open-source and licensed under the [MIT License](LICENSE).
