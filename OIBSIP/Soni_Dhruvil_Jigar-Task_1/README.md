# Iris Flower Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![pandas](https://img.shields.io/badge/pandas-data%20handling-150458)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20pipeline-F7931E)
![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2)

Production-grade machine learning project for classifying Iris flowers into setosa, versicolor, and virginica using a reproducible training, evaluation, and inference pipeline.

## Problem Statement

Given four flower measurements, predict the species of an Iris flower with a robust, well-logged, and testable ML workflow.

## Project Structure

```text
iris-flower-classification/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── logs/
├── tests/
├── config.yaml
├── requirements.txt
├── README.md
├── RUNNING_GUIDE.md
└── main.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Train all models:

```bash
python main.py --mode train
```

Evaluate the best model:

```bash
python main.py --mode evaluate
```

Predict a new flower:

```bash
python main.py --mode predict --input "5.1,3.5,1.4,0.2"
```

## Model Comparison

| Model | CV Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| Logistic Regression | Placeholder | Placeholder | Placeholder | Placeholder |
| Decision Tree | Placeholder | Placeholder | Placeholder | Placeholder |
| Random Forest | Placeholder | Placeholder | Placeholder | Placeholder |
| SVM | Placeholder | Placeholder | Placeholder | Placeholder |
| KNN | Placeholder | Placeholder | Placeholder | Placeholder |
| Gradient Boosting | Placeholder | Placeholder | Placeholder | Placeholder |

## Notes

- Logs are written to `logs/training.log`.
- MLflow artifacts are stored in `mlruns/` by default.
- All runtime paths are controlled through `config.yaml`.
