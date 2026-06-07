# Walkthrough - Used Car Price Prediction ML System

The Used Car Price Prediction ML System has been successfully built, verified, and validated. This document details the changes, model evaluation results, sample prediction verification, and pytest suite run outcomes.

---

## 1. Summary of Built Modules

1. **[config.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/config.py)**: Stores directory structures, random seeds, hyperparameters, target name (`Selling_Price`), and column renaming maps. Automatically creates directories on load.
2. **[data_loader.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/data_loader.py)**: Loads CSV datasets, standardizes raw columns (`Driven_kms` $\rightarrow$ `Kms_Driven`; `Selling_type` $\rightarrow$ `Seller_Type`), handles text encoding fallbacks, and logs statistical descriptors.
3. **[preprocessor.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/preprocessor.py)**: Drops high-cardinality metadata (`Car_Name`), derives `car_age`, imputes missing values using training medians/modes, caps numerical outliers using the IQR method, and encodes categories. Saves preprocessing state to disk to prevent data leakage during predictions.
4. **[feature_engineering.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/feature_engineering.py)**: Engineers interaction features (`price_per_km`, `km_per_year`, and target-based `depreciation`), scales columns using `StandardScaler` (saved to disk), and drops collinear features with correlations $> 0.95$.
5. **[model.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/model.py)**: Implements 5 estimators (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost), conducts 5-fold cross-validation, tunes hyperparameters via `GridSearchCV` on RF and XGBoost, and exports individual pipelines and the best model.
6. **[evaluate.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/evaluate.py)**: Computes test metrics (MAE, MSE, RMSE, R², Adjusted R²), saves them to a CSV, and exports plots (scatter fit, residuals, importances, learning curve, and model comparison) to `reports/figures/`.
7. **[predict.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/src/predict.py)**: Evaluates input parameters (casing, range, categories, type validation), preprocesses/scales inputs using saved training states, and outputs predicted values in Lakhs.
8. **[main.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/main.py)**: Handles CLI command routing (`--train`, `--evaluate`, and `--predict`) with structured log files.

---

## 2. Model Evaluation Results

After running `python main.py --train` and `python main.py --evaluate`, the following performance metrics were logged on the holdout test set (stored in [model_metrics.csv](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/reports/model_metrics.csv)):

| Model | MAE | RMSE | R² | Adjusted R² |
| :--- | :---: | :---: | :---: | :---: |
| **GradientBoosting** | **0.6959** | **1.2938** | **0.9273** | **0.9128** |
| XGBoost_Tuned | 0.9836 | 1.4241 | 0.9120 | 0.8943 |
| RandomForest_Tuned | 0.8119 | 1.4991 | 0.9024 | 0.8829 |
| XGBoost | 0.7388 | 1.5223 | 0.8994 | 0.8793 |
| RandomForest | 0.8010 | 1.4320 | 0.9110 | 0.8932 |
| LinearRegression | 1.3678 | 2.2717 | 0.7760 | 0.7312 |
| Ridge | 1.3910 | 2.2746 | 0.7754 | 0.7305 |
| Lasso | 1.7821 | 2.7906 | 0.6619 | 0.5943 |

*Note: The Gradient Boosting Regressor achieved the highest R² score of **0.9273** on the test set and was saved as [best_model.pkl](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/models/best_model.pkl).*

### Generated Reports & Charts
You can view the saved plots here:
- **Model Comparison R²**: [model_comparison.png](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/reports/figures/model_comparison.png)
- **Actual vs Predicted Fit**: [actual_vs_predicted.png](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/reports/figures/actual_vs_predicted.png)
- **Residuals Distribution**: [residuals_distribution.png](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/reports/figures/residuals_distribution.png)
- **Feature Importance**: [feature_importance.png](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/reports/figures/feature_importance.png)
- **Learning Curve**: [learning_curve.png](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/car-price-prediction/reports/figures/learning_curve.png)

---

## 3. Sample Prediction Verification

Running model inference with a sample used car payload:
```bash
python main.py --predict "{\"Year\": 2015, \"Present_Price\": 8.5, \"Kms_Driven\": 35000, \"Fuel_Type\": \"Petrol\", \"Seller_Type\": \"Dealer\", \"Transmission\": \"Manual\", \"Owner\": 0}"
```
Returns:
```json
{"predicted_price_lakhs": 6.14, "model_used": "GradientBoostingRegressor (Pipeline)"}
```
*This indicates a vehicle bought new for 8.5 Lakhs in 2015 with 35,000 km mileage has an estimated resale price of 6.14 Lakhs in 2026, which aligns with expected depreciation curves.*

---

## 4. Test Suite Validation

Running the test suite via the virtual environment:
```bash
python -m pytest tests/
```
Returned:
```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\car-price-prediction
plugins: anyio-4.13.0
collected 5 items

tests\test_pipeline.py .....                                             [100%]

============================== 5 passed in 4.70s ==============================
```
All unit tests representing correct data shapes, non-null preprocessed results, car age derivation, model file existence, and output types passed without issues.
