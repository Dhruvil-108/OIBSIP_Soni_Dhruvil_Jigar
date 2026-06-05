"""Inference helpers for Iris flower classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from loguru import logger


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


def _parse_input(raw_input: str | Sequence[float]) -> list[float]:
    """Parse a prediction input into four floating-point measurements."""

    if isinstance(raw_input, str):
        values = [value.strip() for value in raw_input.split(",")]
    else:
        values = list(raw_input)

    if len(values) != 4:
        raise ValueError("Prediction input must contain exactly four measurements.")

    parsed_values: list[float] = []
    for value in values:
        if value in (None, ""):
            raise ValueError("Prediction input contains missing values.")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError("Prediction input contains non-finite values.")
        parsed_values.append(numeric_value)
    return parsed_values


def _validate_ranges(values: Sequence[float], config: Mapping[str, Any]) -> None:
    """Validate prediction inputs against configured feature bounds."""

    bounds = config["data"]["feature_bounds"]
    feature_columns = config["data"]["feature_columns"]
    for feature_name, value in zip(feature_columns, values, strict=True):
        minimum, maximum = bounds[feature_name]
        if value < minimum or value > maximum:
            raise ValueError(
                f"Value for {feature_name}={value} is outside the configured range [{minimum}, {maximum}]."
            )


def predict_species(raw_input: str | Sequence[float], config: Mapping[str, Any]) -> dict[str, Any]:
    """Predict an Iris species and confidence for new measurements."""

    values = _parse_input(raw_input)
    _validate_ranges(values, config)

    scaler = joblib.load(_project_root() / Path(config["paths"]["scaler_path"]))
    model = joblib.load(_project_root() / Path(config["paths"]["best_model_path"]))

    sample = np.asarray(values, dtype=float).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    probabilities = model.predict_proba(sample_scaled)[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    species = config["data"]["label_to_species"][predicted_label]

    logger.info(
        "Prediction completed: species={} confidence={:.2f}%",
        species,
        confidence * 100,
    )

    return {
        "predicted_label": predicted_label,
        "species": species,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
    }
