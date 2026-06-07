"""Tests for the data loader module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_raw_data


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame to a CSV path for testing."""

    frame.to_csv(path, index=False)


def test_load_raw_data_normalizes_schema(tmp_path: Path) -> None:
    """The loader should normalize the raw Kaggle schema."""

    csv_path = tmp_path / "iris.csv"
    frame = pd.DataFrame(
        {
            "Id": [1, 2],
            "SepalLengthCm": [5.1, 4.9],
            "SepalWidthCm": [3.5, 3.0],
            "PetalLengthCm": [1.4, 1.4],
            "PetalWidthCm": [0.2, 0.2],
            "Species": ["Iris-setosa", "Iris-setosa"],
        }
    )
    _write_csv(csv_path, frame)

    loaded = load_raw_data({
        "paths": {"raw_data": str(csv_path)},
        "data": {
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target_column": "species",
            "raw_column_map": {
                "Id": "id",
                "SepalLengthCm": "sepal_length",
                "SepalWidthCm": "sepal_width",
                "PetalLengthCm": "petal_length",
                "PetalWidthCm": "petal_width",
                "Species": "species",
            },
            "species_to_label": {"setosa": 0, "versicolor": 1, "virginica": 2},
        },
    })

    assert list(loaded.columns) == ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    assert loaded["species"].tolist() == ["setosa", "setosa"]


def test_load_raw_data_raises_for_null_values(tmp_path: Path) -> None:
    """The loader should reject null values in required columns."""

    csv_path = tmp_path / "iris_null.csv"
    frame = pd.DataFrame(
        {
            "Id": [1],
            "SepalLengthCm": [None],
            "SepalWidthCm": [3.5],
            "PetalLengthCm": [1.4],
            "PetalWidthCm": [0.2],
            "Species": ["Iris-setosa"],
        }
    )
    _write_csv(csv_path, frame)

    with pytest.raises(DataValidationError):
        load_raw_data({
            "paths": {"raw_data": str(csv_path)},
            "data": {
                "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                "target_column": "species",
                "raw_column_map": {
                    "Id": "id",
                    "SepalLengthCm": "sepal_length",
                    "SepalWidthCm": "sepal_width",
                    "PetalLengthCm": "petal_length",
                    "PetalWidthCm": "petal_width",
                    "Species": "species",
                },
                "species_to_label": {"setosa": 0, "versicolor": 1, "virginica": 2},
            },
        })
