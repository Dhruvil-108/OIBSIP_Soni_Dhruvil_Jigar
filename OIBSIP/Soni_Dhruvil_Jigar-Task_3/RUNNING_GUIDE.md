# Step-by-Step Running Guide

This document describes how to set up, train, evaluate, and test the Used Car Price Prediction ML System.

## Prerequisites Checklist
Ensure you have the following installed on your local system:
- [ ] Python 3.10 or higher
- [ ] pip (Python package installer)
- [ ] git (optional, for cloning)

---

## Step 1: Open the Project Workspace
Open your terminal or command prompt and navigate to the project directory:
```bash
cd C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\car-price-prediction
```

---

## Step 2: Activate the Virtual Environment
Activate the **existing** virtual environment inside the workspace. **Do not create a new one.**

- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\activate
  ```
- **Windows (Command Prompt - CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

Once activated, your command prompt should show `(venv)` at the beginning of the line.

---

## Step 3: Install Project Dependencies
Install all required libraries specified in the requirements config:
```bash
pip install -r requirements.txt
```

---

## Step 4: Verify Dataset Placement
Ensure that the dataset file `car_data.csv` is correctly placed inside the `data/` folder. The system expects:
`C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\car-price-prediction\data\car_data.csv`

---

## Step 5: Execute Model Training
Run the training command. This will read the CSV, perform preprocessing, feature engineering, train multiple estimators (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost), run GridSearchCV hyperparameter tuning on Random Forest and XGBoost, and save the best model pipeline to `models/best_model.pkl`.
```bash
python main.py --train
```

---

## Step 6: View Comparative Evaluation Reports
Generate prediction metrics (MAE, MSE, RMSE, R², Adjusted R²) on the test set, save details to `reports/model_metrics.csv`, and output comparison plots to `reports/figures/`:
```bash
python main.py --evaluate
```
Plots generated:
- `reports/figures/model_comparison.png` (R² bar chart)
- `reports/figures/actual_vs_predicted.png` (best model scatter fit)
- `reports/figures/residuals_distribution.png` (residuals histogram)
- `reports/figures/feature_importance.png` (top 15 feature importances)
- `reports/figures/learning_curve.png` (learning rate vs. score)

---

## Step 7: Run Sample Predictions (Inference)
Use the CLI prediction interface by passing a JSON payload of car attributes.
```bash
python main.py --predict '{"Year": 2015, "Present_Price": 8.5, "Kms_Driven": 35000, "Fuel_Type": "Petrol", "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0}'
```
Expected Output Structure (JSON stdout):
```json
{"predicted_price_lakhs": 5.42, "model_used": "XGBRegressor (Pipeline)"}
```

---

## Step 8: Run the Automated Test Suite
Run tests using pytest to confirm that all pipeline transformations, loaders, prediction engines, and artifacts operate correctly:
```bash
pytest tests/
```

---

## Common Errors & Troubleshooting

### 1. `FileNotFoundError: Dataset CSV not found at ...`
- **Cause**: The dataset file `car_data.csv` is missing from the `data/` subdirectory.
- **Fix**: Move `car data.csv` from the root workspace folder to the `data/` folder and make sure it is named `car_data.csv`.

### 2. `ModuleNotFoundError: No module named 'src'`
- **Cause**: Python is unable to resolve relative package paths because the workspace root is not in your python path.
- **Fix**: Ensure you run commands from the workspace root directory: `C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\car-price-prediction`. When running individual src files directly, use Python's module syntax: `python -m src.data_loader` rather than running the script directly.

### 3. `ValueError: Missing required fields` or `Invalid Fuel_Type`
- **Cause**: The JSON payload provided to `--predict` has missing keys or contains spelling/type errors.
- **Fix**: Check that the JSON keys match the exact casings and names: `"Year"`, `"Present_Price"`, `"Kms_Driven"`, `"Fuel_Type"`, `"Seller_Type"`, `"Transmission"`, `"Owner"`. Check that categorical values are valid (e.g. Fuel_Type must be `Petrol`, `Diesel`, or `CNG`).
