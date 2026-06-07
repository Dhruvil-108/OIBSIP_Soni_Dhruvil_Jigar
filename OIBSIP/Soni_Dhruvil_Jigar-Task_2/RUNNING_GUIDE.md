# Running Guide: Unemployment Analysis

This document provides a detailed step-by-step walk-through for setting up, running, testing, and exploring the Unemployment Analysis project.

---

## Prerequisites Checklist
Before executing the pipeline, make sure you have:
- **Python 3.10+** installed on your system.
- **pip** (Python package manager) updated.
- **git** (optional, for project source version control).

---

## Step-by-Step Operations Guide

### Step 1: Navigate to the Project Folder
Open your terminal (PowerShell, Command Prompt, or bash) and change to the project directory:
```bash
cd C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\unemployment-analysis
```

### Step 2: Activate the Existing Virtual Environment
The virtual environment for this project is shared in the parent directory. Activate it using:
- **Windows (PowerShell)**:
  ```powershell
  ..\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  ..\.venv\Scripts\activate.bat
  ```
- **Mac / Linux**:
  ```bash
  source ../.venv/bin/activate
  ```
*Note: Do NOT create a new virtual environment — use the pre-existing environment.*

### Step 3: Install Dependencies
Ensure all package requirements are installed and up to date:
```bash
pip install -r requirements.txt
```

### Step 4: Confirm Dataset Location
Verify that the dataset file `Unemployment in India.csv` is correctly positioned under the `data/` subdirectory:
```bash
# Path verification
data/Unemployment in India.csv
```

### Step 5: Run the Analysis Pipeline
Execute the main script to process the dataset and generate outputs:
- **Full Pipeline Run (Recommended)**:
  ```bash
  python main.py --analyse
  ```
  This command loads the CSV, preprocesses the data, runs EDA and time-series analyses, produces 14 visual charts, caches the intermediate statistics, and outputs a formatted insights report.

- **EDA Plots Only**:
  ```bash
  python main.py --eda-only
  ```
  Generates and saves only the first 6 distribution and correlation plots.

- **Report Regeneration (From Cache)**:
  ```bash
  python main.py --report
  ```
  Reads the cached metrics from `reports/cached_stats.json` and updates the markdown insights report.

### Step 6: View Visual Charts
After running, inspect the 14 visualisations generated at:
`reports/figures/` (from `01_unemployment_rate_distribution.png` to `14_covid_impact_by_region.png`).

### Step 7: Read the Insights Report
Read the final report written by the pipeline to:
`reports/unemployment_insights.md`.

### Step 8: Open the Interactive Notebook
To run the notebook cells interactively:
```bash
jupyter notebook notebooks/01_Unemployment_Analysis.ipynb
```
Select the standard Python 3 kernel and run the cells.

### Step 9: Run Automated Tests
Execute the unit testing suite to confirm everything compiles and runs perfectly:
```bash
pytest tests/
```

---

## Troubleshooting Common Errors

### 1. `FileNotFoundError` for CSV
- **Symptoms**: Output logs show that the CSV file was not found.
- **Solution**: Confirm that `Unemployment in India.csv` is inside the `data/` directory. If you ran the initial directories setup, it should have been moved there automatically.

### 2. Date Parse Error or Out-of-Bounds
- **Symptoms**: Date parsing logs drop large numbers of rows or dates appear as `NaT`.
- **Solution**: The pipeline supports multiple string formats. If your local CSV uses a different delimiter or formatting than standard `DD-MM-YYYY`, update the date parser format settings in `src/data_loader.py`.

### 3. `ModuleNotFoundError`
- **Symptoms**: Python fails to run saying `ModuleNotFoundError: No module named 'src'`.
- **Solution**: Make sure you are running commands from the root directory (`C:\Users\DHRUVIL\OneDrive\Documents\Oasis_InfoByte\OIBSIP\unemployment-analysis`). If you run code within subfolders, Python might not resolve the relative imports.
