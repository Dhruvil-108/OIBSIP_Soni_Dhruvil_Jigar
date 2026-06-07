# Production-Grade Used Car Price Prediction ML System

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## 1. Project Overview
In the used car market, estimating the fair value of a vehicle is a major business problem for buyers, sellers, and dealership platforms. Pricing a car too high leads to long listings and lost storage space, while pricing it too low cuts into profitability.

This project delivers a production-grade machine learning system to predict the market resale price of a used car (in Lakhs) based on key vehicle attributes. It implements a complete, structured pipeline: **Data Loading & Validation** $\rightarrow$ **Preprocessing & Outlier Capping** $\rightarrow$ **Feature Engineering** $\rightarrow$ **Multi-Model Training & Hyperparameter Tuning** $\rightarrow$ **Evaluation Reports** $\rightarrow$ **CLI Inference Engine**.

---

## 2. Tech Stack

| Library | Pinned Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `>=3.10` | Core programming language |
| **pandas** | `>=2.0` | Data manipulation, alignment, and schema standardisation |
| **numpy** | `>=1.24` | Mathematical utilities and vectorized transformations |
| **scikit-learn** | `>=1.3` | Regression algorithms, scaling, pipelines, KFold CV, and GridSearchCV |
| **matplotlib** | `>=3.7` | Visual plots (scatter charts, curves, comparisons) |
| **seaborn** | `>=0.12` | High-level data visualization overlays |
| **joblib** | `>=1.3` | Model serialization and preprocessing state storage |
| **xgboost** | `>=2.0` | Advanced gradient boosting trees algorithm |
| **pytest** | `>=7.4` | Automated test assertions |
| **notebook** | `>=7.0` | Jupyter environment for EDA |

---

## 3. Project Structure

```
car-price-prediction/
├── data/
│   └── car_data.csv               # Standardised dataset CSV
├── notebooks/
│   └── 01_EDA.ipynb               # Jupyter notebook for exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── config.py                  # Dataclass holding project constants and paths
│   ├── data_loader.py             # CSV reader and statistic logger
│   ├── preprocessor.py            # Age calculation, imputation, IQR capping, and encoding
│   ├── feature_engineering.py     # Numeric scaling, collinearity analysis, interaction terms
│   ├── model.py                   # Multi-model training, grid search, and best-model selection
│   ├── evaluate.py                # Pipeline metrics calculator and plotter
│   └── predict.py                 # Schema validator and single-row inference engine
├── models/                        # Serialized fitted scaler, preprocessor state, and model pipelines
├── reports/
│   ├── figures/                   # Evaluation charts (scatter, residuals, feature importance, etc.)
│   └── model_metrics.csv          # Comparative metrics log for all models
├── logs/
│   └── pipeline.log               # Persistent central execution logs
├── tests/
│   └── test_pipeline.py           # Automated unit test suite
├── main.py                        # CLI entrypoint orchestrating all steps
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Project documentation
└── RUNNING_GUIDE.md               # Step-by-step developer guide
```

---

## 4. Dataset Details
The dataset used is `car_data.csv`, containing details about resale transactions:

- **Car_Name**: Name/make of the car (high-cardinality, dropped during preprocessing).
- **Year**: Year of manufacture (numerical, converted to `car_age = 2026 - Year`).
- **Selling_Price** (Target): Resale price in Lakhs (numerical, target to predict).
- **Present_Price**: Ex-showroom showroom price in Lakhs (numerical predictor).
- **Kms_Driven**: Total mileage driven (numerical predictor, converted from raw `Driven_kms`).
- **Fuel_Type**: CNG, Diesel, or Petrol (one-hot encoded).
- **Seller_Type**: Dealer or Individual (one-hot encoded, converted from raw `Selling_type`).
- **Transmission**: Automatic or Manual (one-hot encoded).
- **Owner**: Number of previous owners (0, 1, 3) (numerical predictor).

*Note: The dataset size consists of 301 records.*

---

## 5. Setup & Installation

1. Open a terminal and navigate to the project directory:
   ```bash
   cd C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\car-price-prediction
   ```

2. Activate the existing virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
   *(Note: Do not create a new virtual environment. Use the one already present in the workspace).*

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. How to Run

### Train the Models
Train 5 regression models, perform GridSearchCV tuning, and save the best model:
```bash
python main.py --train
```

### Run Evaluation
Evaluate all models on the test set, output comparison metrics, and save figures:
```bash
python main.py --evaluate
```

### Make a Prediction
Input a JSON string representing a vehicle's attributes to predict its resale price:
```bash
python main.py --predict '{"Year": 2015, "Present_Price": 8.5, "Kms_Driven": 35000, "Fuel_Type": "Petrol", "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0}'
```

### Run Tests
Execute the pytest suite to verify pipeline functionality:
```bash
pytest tests/
```

---

## 7. Model Results
Comparative performance across all regressors on the test partition:

| Model | MAE | RMSE | R² | Adjusted R² |
| :--- | :--- | :--- | :--- | :--- |
| *LinearRegression* | *TBD* | *TBD* | *TBD* | *TBD* |
| *Ridge* | *TBD* | *TBD* | *TBD* | *TBD* |
| *Lasso* | *TBD* | *TBD* | *TBD* | *TBD* |
| *RandomForest_Tuned* | *TBD* | *TBD* | *TBD* | *TBD* |
| *XGBoost_Tuned* | *TBD* | *TBD* | *TBD* | *TBD* |

*(Note: Run `python main.py --evaluate` to populate this metrics summary).*

---

## 8. License
This project is licensed under the MIT License.
