"""Data loader module for the Sales Prediction ML pipeline.

This module reads the raw CSV dataset from the configured directory,
normalizes and cleans column structures, automatically detects the target
variable, and logs data characteristics.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import config

# Configure logger
logger = logging.getLogger(__name__)


def load_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the CSV dataset and standardizes its columns.

    Args:
        file_path: Optional path to the CSV file. If None, it searches
            config.DATA_PATH for CSV files.

    Returns:
        pd.DataFrame: A cleaned DataFrame with index columns removed and
            standardized column names.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If no target column is detected or if the CSV is empty.
    """
    if file_path is None:
        csv_files = list(config.DATA_PATH.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in the data directory: {config.DATA_PATH}"
            )
        file_path = csv_files[0]
        logger.info("Auto-detected CSV file: %s", file_path.name)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Load with pandas, handling potential encoding issues and mixed data types
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decoding failed. Retrying with Latin-1 encoding."
        )
        df = pd.read_csv(file_path, encoding="latin-1")

    if df.empty:
        raise ValueError(f"The dataset at {file_path} is empty.")

    # Drop index columns (unnamed columns or columns containing incrementing numbers)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed:") or col == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info("Dropped index columns: %s", unnamed_cols)

    # Clean whitespace from column names
    df.columns = df.columns.str.strip()

    # Log shape, dtypes, null counts
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Dataset column types:\n%s", df.dtypes.to_string())
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())

    # Detect the target/sales column
    target_col: Optional[str] = None
    for col in df.columns:
        if col.lower() in config.TARGET_COL_PATTERNS:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Could not auto-detect target column. Checked patterns: {config.TARGET_COL_PATTERNS}. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("Auto-detected target variable column: '%s'", target_col)

    # Standardize column naming (e.g., target column to 'sales')
    # If the target is not 'Sales', rename it to standard 'Sales'
    if target_col != "Sales":
        df = df.rename(columns={target_col: "Sales"})
        logger.info("Renamed target column '%s' to 'Sales'", target_col)

    # Log target variable distribution summary
    logger.info("Target (Sales) variable distribution summary:\n%s", df["Sales"].describe().to_string())

    return df


if __name__ == "__main__":
    # Configure standard logging to console for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        data = load_dataset()
        print("\nFirst 5 rows of standardized dataset:")
        print(data.head())
    except Exception as e:
        logger.error("Error executing data_loader stand-alone: %s", e, exc_info=True)
