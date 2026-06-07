"""Data loader module for the Car Price Prediction ML system.

This module provides functionality to load the used car dataset from disk,
detect the target column, standardize headers, handle potential encoding issues,
and log dataset characteristics.
"""

import logging
from pathlib import Path
import pandas as pd
from src.config import config

# Setup module logger
logger = logging.getLogger(__name__)


def load_data(data_dir: Path = config.DATA_PATH, filename: str = config.DATA_FILE) -> pd.DataFrame:
    """Loads the car data CSV and standardises its column names.

    Logs statistics such as shape, data types, missing values, target variable
    distribution, and categorical value counts.

    Args:
        data_dir: Path to the directory containing the dataset CSV.
        filename: Name of the dataset CSV file.

    Returns:
        A cleaned and standardised pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or cannot be parsed.
    """
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {file_path}")

    logger.info("Loading dataset from: %s", file_path)

    # Try loading with UTF-8 first, fallback to ISO-8859-1 if needed
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Retrying with ISO-8859-1 encoding.")
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    logger.info("Dataset loaded successfully. Original shape: %s", df.shape)

    # Standardise column names using rename map from config
    df = df.rename(columns=config.COLUMN_RENAME_MAP)

    # Auto-detect target column
    detected_target = None
    target_candidates = ["selling_price", "price", "target"]
    for col in df.columns:
        col_clean = col.lower().replace(" ", "_").replace("-", "_")
        if col_clean in target_candidates:
            detected_target = col
            break

    if detected_target:
        logger.info("Auto-detected target column: '%s'", detected_target)
        if detected_target != config.TARGET_COLUMN:
            df = df.rename(columns={detected_target: config.TARGET_COLUMN})
            logger.info("Renamed target column '%s' to standard '%s'", 
                        detected_target, config.TARGET_COLUMN)
    else:
        logger.warning("Could not auto-detect target column from candidates %s. "
                       "Using configured default: '%s'", 
                       target_candidates, config.TARGET_COLUMN)

    # Log basic dataset information
    _log_dataframe_stats(df)

    return df


def _log_dataframe_stats(df: pd.DataFrame) -> None:
    """Helper function to log details about the loaded DataFrame."""
    logger.info("----- Dataset Info -----")
    logger.info("Data types:\n%s", df.dtypes.to_string())
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())

    # Log target variable distribution if present
    if config.TARGET_COLUMN in df.columns:
        target_series = df[config.TARGET_COLUMN]
        if pd.api.types.is_numeric_dtype(target_series):
            logger.info("Target column (%s) distribution:\n%s", 
                        config.TARGET_COLUMN, target_series.describe().to_string())
        else:
            logger.warning("Target column (%s) is not numeric!", config.TARGET_COLUMN)

    # Log categorical value counts
    logger.info("Categorical columns unique values count:")
    for col in df.columns:
        if df[col].dtype == "object":
            logger.info("Column '%s': %d unique values", col, df[col].nunique())


if __name__ == "__main__":
    # Setup standalone logger to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    try:
        data = load_data()
        logger.info("Data loaded sample head:\n%s", data.head().to_string())
    except Exception as e:
        logger.exception("Failed to run data loader:")
