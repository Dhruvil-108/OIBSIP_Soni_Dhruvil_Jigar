# Pipeline Execution and Running Guide

This guide describes the complete step-by-step instructions for running, testing, and verifying the Sales Prediction Machine Learning System.

## Prerequisites Checklist

Before executing any commands, ensure you have the following installed:
- [x] Python 3.10 or higher
- [x] Python package manager (`pip`)
- [x] Git version control tool

---

## Step 1: Open Project Workspace
Open your terminal and navigate to the project root directory:
```powershell
cd C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\sales-prediction
```

---

## Step 2: Activate the Virtual Environment
An existing virtual environment is already set up in the workspace. Do not create a new environment; activate the existing one:

- **Windows Powershell/Command Prompt:**
  ```powershell
  .\venv\Scripts\activate
  ```
- **Bash / macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

---

## Step 3: Install Dependencies
Install the required packages from the requirements list:
```powershell
pip install -r requirements.txt
```

---

## Step 4: Verify Dataset Placement
Confirm that the dataset file `Advertising.csv` is present in the `data/` directory:
- Path: `C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\sales-prediction\data\Advertising.csv`

---

## Step 5: Run the Training Pipeline
Run the model selection and training workflow. This step fits data preprocessors, constructs interaction variables, compares models via cross-validation, tunes hyperparameters, and saves the best model:
```powershell
python main.py --train
```

*Outputs produced in this step:*
- Fitted preprocessor: `models/preprocessor.joblib`
- Fitted feature engineer: `models/feature_engineer.joblib`
- Fitted scaler: `models/scaler.joblib`
- Trained model pipeline: `models/best_model.joblib`
- Runtime logs recorded in `logs/pipeline.log`

---

## Step 6: Evaluate the Model & Export Metrics
Evaluate the best model pipeline against the holdout test set to generate performance tables and diagnostic graphs:
```powershell
python main.py --evaluate
```

*Outputs produced in this step:*
- Metrics CSV summary: `reports/model_metrics.csv`
- Actual vs Predicted Scatter Plot: `reports/figures/actual_vs_predicted.png`
- Residual distribution plot: `reports/figures/residual_plot.png`
- Relative Feature Importance / coefficient chart: `reports/figures/feature_importance.png`
- Validation Learning curve: `reports/figures/learning_curve.png`

---

## Step 7: Run Predictions (Inference CLI)
Make a prediction for new advertising budgets using the CLI. Pass features as a JSON-encoded string:
```powershell
python main.py --predict "{\"TV\": 230.1, \"Radio\": 37.8, \"Newspaper\": 69.2}"
```

*(Note: On Windows PowerShell, double quotes in the JSON string must be escaped with a backslash as shown above.)*

---

## Step 8: Run Automated Integration Tests
Execute the pytest suite to verify component integrity:
```powershell
pytest tests/
```

---

## Common Errors & Troubleshooting

### 1. `ModuleNotFoundError: No module named 'src'`
**Cause:** Python cannot locate the project directory in its package search paths.
**Fix:** Run commands from the project root directory (`sales-prediction/`), or add the project root to `PYTHONPATH`:
```powershell
$env:PYTHONPATH="C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\sales-prediction"
```

### 2. JSON Decoding Errors in `--predict`
**Cause:** The command shell (especially Windows Powershell) parses quotes inside the JSON string incorrectly.
**Fix:** Make sure quotes are properly escaped.
- In PowerShell:
  `python main.py --predict "{\"TV\": 230.1, \"Radio\": 37.8, \"Newspaper\": 69.2}"`
- In Cmd:
  `python main.py --predict "{\"TV\": 230.1, \"Radio\": 37.8, \"Newspaper\": 69.2}"`
- In Bash:
  `python main.py --predict '{"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}'`

### 3. Missing Artifact Errors
**Cause:** Running `--evaluate` or `--predict` without running `--train` first.
**Fix:** Execute training to generate the required serialization files:
```powershell
python main.py --train
```
