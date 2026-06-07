"""Raw Iris dataset loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from loguru import logger


class DataValidationError(ValueError):
    """Raised when the raw iris dataset does not satisfy validation rules."""


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


def _expected_raw_columns(config: Mapping[str, Any]) -> list[str]:
    """Return the expected raw CSV columns from configuration."""

    return list(config["data"]["raw_column_map"].keys())


def _normalized_species_values(config: Mapping[str, Any]) -> set[str]:
    """Return the set of valid normalized species names."""

    return set(config["data"]["species_to_label"].keys())


def _normalize_species(value: Any) -> str:
    """Normalize raw Iris species labels to compact lowercase names."""

    species = str(value).strip()
    species = species.replace("Iris-", "")
    return species.lower()


def validate_raw_data(df: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate raw Iris data against the configured schema."""

    expected_columns = _expected_raw_columns(config)
    if list(df.columns) != expected_columns:
        raise DataValidationError(
            f"Unexpected raw columns: {list(df.columns)}. Expected: {expected_columns}"
        )

    if df.isnull().any().any():
        raise DataValidationError("Raw dataset contains null values.")

    feature_columns = config["data"]["feature_columns"]
    raw_column_map = config["data"]["raw_column_map"]
    for raw_column, normalized_column in raw_column_map.items():
        if normalized_column in feature_columns:
            if not pd.api.types.is_numeric_dtype(df[raw_column]):
                raise DataValidationError(f"Column {raw_column} must be numeric.")


def load_raw_data(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load, validate, and clean the raw Iris dataset."""

    raw_path = _project_root() / Path(config["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    df_raw = pd.read_csv(raw_path)
    validate_raw_data(df_raw, config)

    df_clean = df_raw.rename(columns=config["data"]["raw_column_map"]).copy()
    if "id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id"])

    feature_columns = config["data"]["feature_columns"]
    target_column = config["data"]["target_column"]

    for column in feature_columns:
        df_clean[column] = pd.to_numeric(df_clean[column], errors="raise")

    df_clean[target_column] = df_clean[target_column].map(_normalize_species)
    valid_species = _normalized_species_values(config)
    if not set(df_clean[target_column].unique()).issubset(valid_species):
        raise DataValidationError("Dataset contains unexpected species labels.")

    if df_clean.isnull().any().any():
        raise DataValidationError("Clean dataset contains null values.")

    logger.info("Loaded dataset shape: {}", df_clean.shape)
    logger.info(
        "Class distribution:\n{}",
        df_clean[target_column].value_counts().to_string(),
    )

    return df_clean[[*feature_columns, target_column]]
