"""Tests for the inference module."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.predict import predict_species


def test_predict_flower_returns_valid_species_and_confidence(tmp_path: Path) -> None:
    """Prediction should return a known species label and a confidence score."""

    raw_path = tmp_path / "iris.csv"
    raw_frame = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "SepalLengthCm": [5.1, 6.0, 6.5],
            "SepalWidthCm": [3.5, 3.0, 3.2],
            "PetalLengthCm": [1.4, 4.5, 5.5],
            "PetalWidthCm": [0.2, 1.5, 2.0],
            "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        }
    )
    raw_frame.to_csv(raw_path, index=False)

    scaler_path = tmp_path / "scaler.pkl"
    model_path = tmp_path / "best_model.pkl"
    config_path = tmp_path / "config.yaml"

    training_features = pd.DataFrame(
        {
            "sepal_length": [5.1, 6.0, 6.5, 5.8, 4.9, 6.1],
            "sepal_width": [3.5, 3.0, 3.2, 2.7, 3.1, 2.9],
            "petal_length": [1.4, 4.5, 5.5, 4.2, 1.5, 4.9],
            "petal_width": [0.2, 1.5, 2.0, 1.3, 0.2, 1.8],
        }
    )
    labels = [0, 1, 2, 1, 0, 2]
    scaler = StandardScaler().fit(training_features)
    model = LogisticRegression(max_iter=1000, multi_class="auto").fit(scaler.transform(training_features), labels)

    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)

    config = {
        "paths": {
            "raw_data": str(raw_path),
            "scaler_path": str(scaler_path),
            "best_model_path": str(model_path),
        },
        "data": {
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "label_to_species": {0: "setosa", 1: "versicolor", 2: "virginica"},
            "feature_bounds": {
                "sepal_length": [4.0, 8.5],
                "sepal_width": [2.0, 4.5],
                "petal_length": [1.0, 7.5],
                "petal_width": [0.1, 3.0],
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = predict_species("5.1,3.5,1.4,0.2", config)

    assert result["species"] in {"setosa", "versicolor", "virginica"}
    assert 0.0 <= result["confidence"] <= 1.0
