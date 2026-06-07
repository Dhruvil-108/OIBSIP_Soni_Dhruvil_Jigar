"""Data loader module for the Email Spam Detection System.

This module provides functions to load the raw spam CSV dataset,
standardize the schema, handle encoding issues, and log details of the dataset.
"""

import logging
from pathlib import Path
import pandas as pd
from src.config import Config

logger = logging.getLogger(__name__)


def load_data(config: Config) -> pd.DataFrame:
    """Loads the email spam CSV dataset and standardizes the columns.

    Args:
        config (Config): Configuration containing project variables, including DATA_PATH.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns 'text' and 'label'.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is not parsed correctly.
    """
    csv_path: Path = config.DATA_PATH
    if not csv_path.exists():
        logger.error(f"Dataset not found at: {csv_path}")
        raise FileNotFoundError(f"Dataset file does not exist at {csv_path}")

    # Encoding fallbacks
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    df = None
    for encoding in encodings:
        try:
            logger.info(f"Attempting to load CSV with encoding: {encoding}")
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info(f"Successfully loaded CSV with encoding: {encoding}")
            break
        except (UnicodeDecodeError, Exception) as e:
            logger.warning(f"Failed to load with encoding {encoding}: {e}")

    if df is None:
        raise ValueError("Could not read CSV file with any of the attempted encodings.")

    logger.info(f"Raw dataset shape: {df.shape}")

    # Auto-detect text and label columns
    if df.shape[1] < 2:
        raise ValueError("Dataset has less than 2 columns. Cannot identify text and label.")

    # Drop any unnamed extra columns
    unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
    if unnamed_cols:
        logger.info(f"Dropping unnamed extra columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # Rename first two columns to 'label' and 'text'
    col_mapping = {df.columns[0]: "label", df.columns[1]: "text"}
    df = df.rename(columns=col_mapping)
    
    # Standardise column order to ['text', 'label']
    df = df[["text", "label"]]

    # Drop null values in required columns
    null_counts = df.isnull().sum()
    logger.info(f"Null values in each column:\n{null_counts}")
    
    df = df.dropna(subset=["text", "label"])
    logger.info(f"Cleaned dataset shape (after dropping nulls): {df.shape}")

    # Log class distribution
    class_dist = df["label"].value_counts()
    class_pct = df["label"].value_counts(normalize=True) * 100
    logger.info("Class distribution:")
    for cls in class_dist.index:
        logger.info(f"  {cls}: {class_dist[cls]} ({class_pct[cls]:.2f}%)")

    return df


if __name__ == "__main__":
    # Test loading standalone
    logging.basicConfig(level=logging.INFO)
    cfg = Config()
    try:
        data = load_data(cfg)
        print("First 5 rows:\n", data.head())
    except Exception as ex:
        print("Error during test run:", ex)
