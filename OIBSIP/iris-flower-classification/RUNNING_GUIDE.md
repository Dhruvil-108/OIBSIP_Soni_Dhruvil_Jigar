> ⚠️ Always activate your virtual environment before running any command

# 🔧 Prerequisites

Install the following software before you begin.

Python 3.10+ installation link:

```text
https://www.python.org/downloads/
```

Pip or conda setup:

```text
https://pip.pypa.io/en/stable/installation/
```

```text
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
```

Git installation:

```text
https://git-scm.com/downloads
```

VS Code + Python extension:

```text
https://code.visualstudio.com/
```

```text
https://marketplace.visualstudio.com/items?itemName=ms-python.python
```

MLflow UI setup (optional but recommended):

```text
https://mlflow.org/
```

# 🚀 Project Setup (Step-by-Step)

1. Clone or open the project.

```bash
git clone <repo-url>
cd iris-flower-classification
```

If the project is already on your machine:

```bash
cd path/to/iris-flower-classification
```

2. Create a virtual environment.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install all dependencies.

```bash
pip install -r requirements.txt
```

4. Place the dataset.

```text
Copy iris.csv into: data/raw/iris.csv
Confirm columns: sepal_length, sepal_width, petal_length, petal_width, species
```

# 📊 Running the Project

5. Train all models.

```bash
python main.py --mode train
```

Expected output:

- Models saved to `models/`
- Scaler saved to `models/scaler.pkl`
- Logs written to `logs/training.log`
- MLflow experiment logged

6. Evaluate the best model.

```bash
python main.py --mode evaluate
```

Expected output:

- Confusion matrix saved to `data/reports/`
- ROC-AUC curve saved to `data/reports/`
- Feature importance plot saved to `data/reports/`
- Classification report printed in terminal

7. Predict on new flower input.

```bash
python main.py --mode predict --input "5.1,3.5,1.4,0.2"
```

Input format:

```text
sepal_length, sepal_width, petal_length, petal_width
```

Expected output:

- Predicted species name
- Confidence score (%)

# 📈 View MLflow Experiment Dashboard

8. Launch MLflow UI.

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

View:

- All model runs and metrics
- Accuracy, F1, Precision, Recall per model
- Best model highlighted

# 🧪 Run Unit Tests

9. Run all tests.

```bash
pytest tests/ -v
```

Expected output:

- `test_data_loader.py` PASSED
- `test_preprocessor.py` PASSED
- `test_predict.py` PASSED

# 🪵 View Logs

10. Check training logs.

Windows:

```bash
type logs\training.log
```

Mac/Linux:

```bash
cat logs/training.log
```

Or open:

```text
logs/training.log
```

# 🛠 Common Errors & Fixes

| Error | Cause | Fix |
| --- | --- | --- |
| ModuleNotFoundError | venv not activated | Run activate command again |
| FileNotFoundError: iris.csv | CSV not in data/raw/ | Move CSV to data/raw/ |
| MLflow connection error | mlflow ui not running | Run mlflow ui in separate terminal |
| Scaler not found | Train not run yet | Run --mode train first |
| Low accuracy warning | Data not scaled properly | Check preprocessor.py logs |

# 📦 Project Output Summary

After a full run, you should have:

```text
models/
  ├── best_model.pkl
  └── scaler.pkl
data/reports/
  ├── confusion_matrix.png
  ├── roc_auc_curve.png
  └── feature_importance.png
logs/
  └── training.log
mlruns/
  └── (MLflow experiment data)
```

> ✅ If all steps completed without error, your Iris Classification pipeline is fully working!
