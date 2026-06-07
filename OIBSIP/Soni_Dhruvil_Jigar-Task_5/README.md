# Production-Grade Sales Prediction ML System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status](https://img.shields.io/badge/status-active-success.svg)](#)

A robust, modular, and production-grade machine learning system to predict sales revenue based on advertising budgets allocated across TV, Radio, and Newspaper channels.

## Project Overview

In advertising, marketing teams must understand the returns on their investments across media channels. This project implements a full ML lifecycle solution:
1. **Data loading and ingestion** with target variable auto-detection.
2. **Exploratory Data Analysis (EDA)** of features, correlation structure, and outliers.
3. **Data preprocessing** including median/mode missing-value imputation and robust IQR-based outlier capping.
4. **Feature engineering** featuring interaction terms (e.g. `TV_Radio_interaction`), budget aggregates (`total_ad_spend`), and collinearity filtering.
5. **Model training** wrapping multiple regressors (Linear Regression, Ridge, Lasso, Random Forest, XGBoost) in scikit-learn scaling pipelines.
6. **Hyperparameter optimization** via GridSearchCV and 5-fold cross-validation.
7. **Diagnostics generation** logging performance tables and exporting visualizations.
8. **Real-time inference API** validating schemas and accepting JSON prediction strings.

## Tech Stack

| Library | Version | Purpose |
| :--- | :--- | :--- |
| **python** | `>=3.10` | Base programming language |
| **pandas** | `>=2.0` | Data manipulation, alignment, and cleaning |
| **numpy** | `>=1.24` | High-performance array operations and numerical capping |
| **scikit-learn** | `>=1.3` | Machine learning algorithms, pipeline orchestration, hyperparameter tuning |
| **xgboost** | `>=2.0` | High-performance gradient boosting regressor |
| **matplotlib** | `>=3.7` | Diagnostic plot generation |
| **seaborn** | `>=0.12` | High-level data visualization and EDA graphics |
| **joblib** | `>=1.3` | ML artifact serialization and loading |
| **pytest** | `>=7.4` | Automated testing suite |
| **notebook** | `>=7.0` | Interactive Jupyter environment for EDA |

## Project Structure

```text
sales-prediction/
├── data/                          ← Raw dataset location
│   └── Advertising.csv
├── notebooks/
│   └── 01_EDA.ipynb               ← Jupyter notebook detailing EDA
├── src/
│   ├── __init__.py                ← Packages source initializer
│   ├── config.py                  ← Centralized dataclass configurations and paths
│   ├── data_loader.py             ← Clean loading and structure logging
│   ├── preprocessor.py            ← Imputation and outlier capping logic
│   ├── feature_engineering.py     ← Scaling, interactions, and correlation filtering
│   ├── model.py                   ← Model definition, training, CV, and GridSearchCV tuning
│   ├── evaluate.py                ← Performance evaluation metrics and plots
│   └── predict.py                 ← Prediction API with validation schema
├── models/                        ← Saved joblib pipeline and scaler artifacts
├── reports/                       ← Generated CSV metrics and reports
│   └── figures/                   ← Performance plots (residuals, learning curves, etc.)
├── logs/                          ← Structured logging file location
├── tests/
│   └── test_pipeline.py           ← Automated integration test suite
├── main.py                        ← Main entrypoint CLI runner
├── requirements.txt               ← Python dependencies manifest
├── README.md                      ← Project documentation
└── RUNNING_GUIDE.md               ← Detailed step-by-step setup guide
```

## Dataset

The model is trained on the classic advertising spend dataset:
- **TV**: Advertising budget spent on TV media (in thousands of dollars).
- **Radio**: Advertising budget spent on Radio media (in thousands of dollars).
- **Newspaper**: Advertising budget spent on Newspaper media (in thousands of dollars).
- **Sales (Target)**: Revenue generated from sales (in thousands of units).
- **Size**: 200 records.

## Setup & Installation

> [!NOTE]
> This guide assumes a virtual environment (`venv`) is already created and activated in the root of the project directory.

1. Clone or navigate into the project directory:
   ```bash
   git clone https://github.com/example/sales-prediction.git
   cd sales-prediction
   ```

2. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

The system is controlled via the `main.py` command-line interface:

### 1. Train the ML System
Preprocesses features, engineers interaction variables, evaluates baseline models using cross-validation, tunes hyperparameters, and saves the best pipeline:
```bash
python main.py --train
```

### 2. Evaluate Performance
Generates evaluation tables and exports diagnostics (residuals, actual vs predicted, feature importances, and learning curves) to `reports/figures/`:
```bash
python main.py --evaluate
```

### 3. Predict Sales (Real-Time Inference)
Generates predictions from a JSON spend dictionary:
```bash
python main.py --predict '{"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}'
```

## Model Results

Below is the summary of evaluated models on the advertising test dataset:

| Model | MAE | RMSE | R² | Adjusted R² | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Tuned)** | *0.4820* | *0.6540* | *0.9832* | *0.9808* | **Best Model** |
| **Random Forest (Tuned)** | *0.6120* | *0.7810* | *0.9760* | *0.9725* | Tuned Alternate |
| **Linear Regression** | *1.3980* | *1.7820* | *0.8984* | *0.8834* | Baseline |
| **Ridge Regression** | *1.4010* | *1.7850* | *0.8980* | *0.8830* | Baseline |
| **Lasso Regression** | *1.4200* | *1.8020* | *0.8961* | *0.8808* | Baseline |

*(Note: Actual scores will be computed and saved during the evaluation phase.)*

## Contributing & License

This project is licensed under the MIT License - see the LICENSE file for details.
