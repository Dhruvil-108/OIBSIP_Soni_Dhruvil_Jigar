# Production-Grade Car Price Prediction ML System

This plan outlines the design and implementation of a production-grade machine learning system to predict used car resale/market prices (in Lakhs) based on vehicle features (e.g., year, mileage, fuel type, transmission, seller type, owners).

## User Review Required

> [!IMPORTANT]
> - **CSV Schema Mapping**: The existing file is named `car data.csv` and is in the root directory. It contains headers `Driven_kms` and `Selling_type`. Our data loader will standardise these to `Kms_Driven` and `Seller_Type` respectively to align with the downstream requirement of predicting with clean and standardised fields.
> - **Car Age Calculation**: We will use the current year (2026, extracted dynamically using `datetime.now().year`) to calculate `car_age = current_year - Year`.
> - **Outlier Handling**: Outliers in numerical fields (`Present_Price`, `Kms_Driven`) will be capped using the IQR method during preprocessing to stabilize regression models.

## Open Questions

None at this moment. The dataset is already checked, and the schema aligns perfectly with the target requirements.

---

## Proposed Changes

### Folder Structure

```
C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\car-price-prediction/
├── data/
│   └── car_data.csv               ← Dataset CSV (moved/renamed from root "car data.csv")
├── notebooks/
│   └── 01_EDA.ipynb               ← Jupyter notebook for exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── config.py                  ← Constants, paths, target and feature names
│   ├── data_loader.py             ← Data loading and logging of basic stats
│   ├── preprocessor.py            ← Dropping name, calculating age, capping outliers, imputing nulls
│   ├── feature_engineering.py     ← Scaling features, generating interactions, correlation check
│   ├── model.py                   ← Multi-model training, GridSearchCV tuning, cross-validation
│   ├── evaluate.py                ← Evaluation metrics (MAE, RMSE, R2) and report plotting
│   └── predict.py                 ← Prediction API for incoming dictionary/JSON inputs
├── models/                        ← Fitted scaler.pkl and best_model.pkl
├── reports/
│   ├── figures/                   ← Evaluation charts (Scatter, Residuals, Importance, etc.)
│   └── model_metrics.csv          ← Aggregated evaluation metrics
├── logs/
│   └── pipeline.log               ← Centralized runtime logs
├── tests/
│   └── test_pipeline.py           ← pytest suite for standard pipeline assertions
├── main.py                        ← CLI entrypoint (--train, --evaluate, --predict)
├── requirements.txt               ← Pinned dependencies
├── README.md                      ← Overview and instructions
└── RUNNING_GUIDE.md               ← Step-by-step instructions
```

---

### Module Dependency Map

```
                +--------------+
                |  config.py   |
                +-------+------+
                        |
      +-----------------+-----------------+
      |                 |                 |
      v                 v                 v
+-----+-----+     +-----+-----+     +-----+-----+
|data_loader|     |preprocessor|     | predict.py|
+-----+-----+     +-----+-----+     +-----+-----+
      |                 |                 |
      |                 |                 |
      |                 v                 v
      |          +------+------+   +------+------+
      |          |feature_eng. |   |feature_eng. |
      |          +------+------+   +------+------+
      |                 |                 |
      +--------+--------+                 |
               |                          |
               v                          |
          +----+----+                     |
          |model.py |                     |
          +----+----+                     |
               |                          |
               v                          |
          +----+----+                     |
          |evaluate |                     |
          +----+----+                     |
               |                          |
               +------------+-------------+
                            |
                            v
                      +-----+-----+
                      |  main.py  |
                      +-----------+
```

---

### Dataset Schema & Assumptions

Based on inspecting the `car data.csv` file, the columns are:
- `Car_Name` (Categorical, high-cardinality -> dropped during preprocessing)
- `Year` (Numerical -> transformed to `car_age` using `2026 - Year`)
- `Selling_Price` (Numerical, continuous -> **Target column**)
- `Present_Price` (Numerical, continuous -> Predictor)
- `Driven_kms` (Numerical, continuous -> Predictor, will be standardised to `Kms_Driven`)
- `Fuel_Type` (Categorical: Petrol, Diesel, CNG -> One-Hot Encoded)
- `Selling_type` (Categorical: Dealer, Individual -> One-Hot Encoded, standardised to `Seller_Type`)
- `Transmission` (Categorical: Manual, Automatic -> One-Hot Encoded)
- `Owner` (Numerical, discrete/ordinal: 0, 1, 3 -> Treated as numeric feature)

---

### ML Pipeline Stages

1. **Data Loading**: Load CSV, standardise headers to clean names, perform dataset validation (shape, data types, counts).
2. **Preprocessing**: Compute `car_age`, drop `Car_Name` and `Year`, handle any missing values, compute IQR bounds and cap outliers, perform one-hot encoding on categorical columns.
3. **Feature Engineering**: Compute engineered features (`price_per_km`, `depreciation`, `km_per_year`). Scale all numerical columns with `StandardScaler` (save fit state). Drop any features with collinearity > 0.95.
4. **Model Training**: Train 5 models:
   - Linear Regression
   - Ridge & Lasso Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
   Using 5-fold cross-validation. Perform GridSearchCV hyperparameter tuning on Random Forest and XGBoost. Choose best model by R² score and save.
5. **Evaluation**: Compute evaluation metrics (MAE, MSE, RMSE, R², Adjusted R²). Output plots to `reports/figures/` (Residuals, Feature Importances, Learning Curve, etc.) and save metrics to `reports/model_metrics.csv`.
6. **Prediction**: Load the best model and scaler. Preprocess a new input dictionary, validate features, and output prediction in Lakhs.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_pipeline.py` to assert:
  - Dataset loads into a valid non-empty DataFrame.
  - Preprocessed dataset has no missing values.
  - `car_age` column is successfully derived.
  - Saved model artifact is produced and predictions are valid dicts with `predicted_price_lakhs`.

### Manual Verification
- Move the CSV into `data/`.
- Run `python main.py --train` and confirm it logs all training metrics.
- Run `python main.py --evaluate` and inspect `reports/figures/` plots and metrics file.
- Run `python main.py --predict '<JSON>'` and verify predicted price is reasonable and does not fail.
