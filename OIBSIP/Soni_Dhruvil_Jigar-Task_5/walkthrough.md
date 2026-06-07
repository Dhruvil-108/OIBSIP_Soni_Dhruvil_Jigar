# Sales Prediction ML System - Implementation Walkthrough

This document summarizes the final state of the Sales Prediction system, testing outcomes, and verification metrics.

## Accomplishments

The system was successfully built and tested in `C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\sales-prediction`. Every component follows high-quality, production-grade coding standards (strict typing, Google-style docstrings, structured logging, and modular architecture).

Here is a summary of the completed files:
1. **[config.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/config.py)**: Centralizes all paths, constants, random seeds, and pipeline parameters. Automatically creates required workspace folders on load.
2. **[data_loader.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/data_loader.py)**: Loads CSV data, auto-detects target columns, drops indices, and logs metadata.
3. **[preprocessor.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/preprocessor.py)**: Handles missing value imputation (median for numeric, mode for categorical) and caps outliers using the IQR method. Preserves training-set statistics for inference.
4. **[feature_engineering.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/feature_engineering.py)**: Creates interaction (`TV_Radio_interaction`) and total expenditure (`total_ad_spend`) features. Filters collinear columns and fits standard scaling.
5. **[model.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/model.py)**: Evaluates 5 models under 5-fold cross-validation. Tunes hyperparameters for the top two models (Random Forest and XGBoost) using `GridSearchCV`, selecting the best pipeline.
6. **[evaluate.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/evaluate.py)**: Calculates MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$. Generates evaluation plots and logs metrics.
7. **[predict.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/src/predict.py)**: Implements validation checks and inferences.
8. **[main.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/main.py)**: Command-line interface exposing `--train`, `--evaluate`, and `--predict` flags. Sets up logging to both console and `logs/pipeline.log`.
9. **[test_pipeline.py](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/tests/test_pipeline.py)**: Pytest suite containing 4 test cases.
10. **[01_EDA.ipynb](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/notebooks/01_EDA.ipynb)**: Detailed professional exploratory data analysis notebook.
11. **[README.md](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/README.md)** & **[RUNNING_GUIDE.md](file:///C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/RUNNING_GUIDE.md)**: Standard project documentation and execution manual.

---

## Verification & Validation Results

### 1. Model Selection & Cross-Validation
During cross-validation (5-fold) on the training set, the following mean $R^2$ scores were obtained:
- **Linear Regression**: `0.9571`
- **Ridge Regression**: `0.9569`
- **Lasso Regression**: `0.9101`
- **Random Forest**: `0.9835`
- **XGBoost**: `0.9822`

The top two models (**Random Forest** and **XGBoost**) were selected for hyperparameter tuning. The tuned **XGBoost Regressor** yielded the highest cross-validated score (`0.9838`) and was saved as the final production model.
- **XGBoost Best Params:** `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}`

### 2. Test Set Evaluation Metrics
The final evaluation on the holdout test set (`20%` of data) returned these performance scores:
- **Mean Absolute Error (MAE)**: `0.5822`
- **Mean Squared Error (MSE)**: `0.5070`
- **Root Mean Squared Error (RMSE)**: `0.7120`
- **R² Coefficient of Determination**: `0.9839`
- **Adjusted R²**: `0.9816`

### 3. Diagnostic Plots Created
The following plots were generated and saved to `reports/figures/`:
1. **Actual vs Predicted Scatter Plot (`actual_vs_predicted.png`)**: Confirms linear alignment of predictions near the diagonal identity line.
2. **Residual Plot (`residual_plot.png`)**: Confirms random, homoscedastic scattering of residuals centered around zero.
3. **Feature Importance Bar Chart (`feature_importance.png`)**: Ranks predictor influence. As expected, `TV_Radio_interaction` is the most dominant predictor, followed by `TV` and `total_ad_spend`.
4. **Learning Curve (`learning_curve.png`)**: Shows convergence of train and validation $R^2$ scores, indicating low bias and variance (no overfitting).

### 4. Real-Time Inference (Prediction Output)
Running predictions with raw input spends using:
```bash
python main.py --predict '{""TV"": 230.1, ""Radio"": 37.8, ""Newspaper"": 69.2}'
```
returned:
```json
{
  "predicted_sales": 22.0774,
  "model_used": "XGBRegressor"
}
```

### 5. Automated Tests Suite
Executing `python -m pytest tests/` completed successfully with **all 4 tests passing**:
- `test_csv_loading`: Passed (Verify columns, drop indexes).
- `test_preprocessor_shape`: Passed (Verify imputation, outlier capping, shape preservation).
- `test_model_artifact_exists_after_training`: Passed (Verify joblib serialization files exist).
- `test_prediction_output_type`: Passed (Verify JSON key matching and float types).
