"""Data preprocessing utilities for Iris flower classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessResult:
    """Container for train-test splits and preprocessing artifacts."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_columns: list[str]


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


def _save_processed_data(df: pd.DataFrame, config: Mapping[str, Any]) -> Path:
    """Persist the cleaned and encoded dataset for downstream inspection."""

    processed_path = _project_root() / Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return processed_path


def preprocess_data(df: pd.DataFrame, config: Mapping[str, Any]) -> PreprocessResult:
    """Encode the target, split the data, and scale features without leakage."""

    feature_columns = list(config["data"]["feature_columns"])
    target_column = config["data"]["target_column"]
    training_config = config["training"]

    encoded_df = df.copy()
    label_encoder = LabelEncoder()
    encoded_df[target_column] = label_encoder.fit_transform(encoded_df[target_column])
    _save_processed_data(encoded_df, config)

    X = encoded_df[feature_columns]
    y = encoded_df[target_column]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=feature_columns,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_columns,
        index=X_test_raw.index,
    )

    scaler_path = _project_root() / Path(config["paths"]["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    logger.info("Encoded target labels with classes: {}", list(label_encoder.classes_))
    logger.info(
        "Completed stratified split with train size {} and test size {}",
        len(X_train_raw),
        len(X_test_raw),
    )
    logger.info("Saved scaler to {}", scaler_path)

    return PreprocessResult(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        scaler=scaler,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
    )
