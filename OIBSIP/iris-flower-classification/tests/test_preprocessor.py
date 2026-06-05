"""Tests for the preprocessing module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessor import preprocess_data


def test_preprocess_data_scales_training_features() -> None:
    """The training split should be standardized after fitting the scaler."""

    frame = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 4.7, 6.0, 5.5, 6.3, 5.8, 5.7, 6.4, 5.0, 4.6, 5.4],
            "sepal_width": [3.5, 3.0, 3.2, 2.9, 2.8, 3.3, 2.7, 2.8, 3.1, 3.6, 3.4, 3.9],
            "petal_length": [1.4, 1.4, 1.3, 4.5, 4.0, 6.0, 5.1, 4.2, 5.6, 1.5, 1.4, 1.7],
            "petal_width": [0.2, 0.2, 0.2, 1.5, 1.3, 2.5, 1.8, 1.3, 2.1, 0.2, 0.2, 0.4],
            "species": ["setosa", "setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica", "versicolor", "virginica", "setosa", "setosa", "setosa"],
        }
    )
    config = {
        "paths": {
            "processed_data": "data/processed/cleaned_and_encoded_iris.csv",
            "scaler_path": "models/scaler.pkl",
        },
        "data": {
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target_column": "species",
            "species_to_label": {"setosa": 0, "versicolor": 1, "virginica": 2},
        },
        "training": {"test_size": 0.25, "random_state": 42},
    }

    result = preprocess_data(frame, config)

    assert result.X_test.shape[0] == 3
    assert result.X_train.shape[0] == 9
    assert result.y_train.shape[0] == 9
    assert result.y_test.shape[0] == 3
    assert np.allclose(result.X_train.mean(axis=0), np.zeros(result.X_train.shape[1]), atol=1e-7)
    assert hasattr(result.scaler, "mean_")
