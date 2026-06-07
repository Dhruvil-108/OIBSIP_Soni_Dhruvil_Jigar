"""Module for loading and standardising the unemployment dataset.

Handles CSV loading, column header cleaning, whitespace stripping from text,
date parsing, sorting, and logging of basic dataset information.
"""

import logging
import pandas as pd
from src.config import AnalysisConfig

# Set up logging for this module
logger = logging.getLogger(__name__)


def load_data(config: AnalysisConfig) -> pd.DataFrame:
    """Loads the unemployment dataset from the path configured in config.

    Cleans columns, strips spaces, parses dates, drops completely empty rows,
    sorts chronologically, and logs detailed statistics.

    Args:
        config (AnalysisConfig): Configuration settings including paths and columns.

    Returns:
        pd.DataFrame: Cleaned and sorted DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = config.DATA_PATH / config.CSV_FILENAME
    logger.info("Attempting to load dataset from: %s", csv_path)

    if not csv_path.exists():
        error_msg = f"Dataset file not found at path: {csv_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Load raw CSV file
    df = pd.read_csv(csv_path)
    logger.info("Initial loaded shape: %s", df.shape)

    # Clean column headers
    df.columns = df.columns.str.strip()
    logger.info("Cleaned columns: %s", list(df.columns))

    # Drop completely empty/blank rows (like rows in the middle or end of file)
    df = df.dropna(how="all")

    # Clean string data
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Parse date column
    logger.info("Parsing Date column: %s", config.DATE_COLUMN)
    df[config.DATE_COLUMN] = pd.to_datetime(
        df[config.DATE_COLUMN],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # Drop rows where Date is null after parsing
    initial_count = len(df)
    df = df.dropna(subset=[config.DATE_COLUMN])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        logger.warning("Dropped %d rows with invalid dates", dropped_count)

    # Sort dataset chronologically
    df = df.sort_values(by=config.DATE_COLUMN).reset_index(drop=True)

    # Log data attributes
    logger.info("Cleaned dataset shape: %s", df.shape)
    logger.info("Data types:\n%s", df.dtypes)
    logger.info("Missing value counts:\n%s", df.isnull().sum())
    logger.info("Date range: %s to %s", df[config.DATE_COLUMN].min(), df[config.DATE_COLUMN].max())
    
    if config.REGION_COLUMN in df.columns:
        logger.info("Unique regions count: %d", df[config.REGION_COLUMN].nunique())

    return df


if __name__ == "__main__":
    # Standard logger configuration for standalone run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = AnalysisConfig()
    try:
        loaded_df = load_data(cfg)
        print("\n--- STANDALONE LOAD SUCCESS ---")
        print(loaded_df.head())
    except Exception as exc:
        logger.error("Standalone run failed: %s", exc, exc_info=True)
