"""Module for preprocessing the unemployment dataset.

Handles imputation of missing values, case standardisation, derived column
engineering (month, year, quarter, covid period), and outlier detection.
"""

import logging
import pandas as pd
from src.config import AnalysisConfig

# Set up logging for this module
logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Preprocesses the raw dataset.

    Imputes missing values (forward fill for time-series columns grouped by
    region/area, mode for categorical variables), title-cases strings, extracts
    date features, and flags outliers in the unemployment rate column.

    Args:
        df (pd.DataFrame): Raw DataFrame loaded by the data loader.
        config (AnalysisConfig): Configuration settings.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with engineered columns.
    """
    df = df.copy()

    # Standardise text case for Region and Area to Title Case
    logger.info("Standardising text case for Region and Area columns.")
    if config.REGION_COLUMN in df.columns:
        df[config.REGION_COLUMN] = df[config.REGION_COLUMN].astype(str).str.title()
    if config.AREA_COLUMN in df.columns:
        df[config.AREA_COLUMN] = df[config.AREA_COLUMN].astype(str).str.title()

    # Handle missing values
    logger.info("Imputing missing values...")

    # Impute categorical columns with mode
    cat_cols = [config.REGION_COLUMN, config.AREA_COLUMN, config.FREQUENCY_COLUMN]
    for col in cat_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                mode_vals = df[col].mode()
                fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                logger.info("Imputed %d missing values in %s with mode '%s'", null_count, col, fill_val)

    # Impute numerical (time-series) columns
    # We group by Region and Area to forward-fill since each region/area combination
    # constitutes an independent time series.
    num_cols = [config.RATE_COLUMN, config.EMPLOYED_COLUMN, config.LABOUR_PART_COLUMN]
    for col in num_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                if config.REGION_COLUMN in df.columns and config.AREA_COLUMN in df.columns:
                    df[col] = df.groupby([config.REGION_COLUMN, config.AREA_COLUMN])[col].ffill()
                else:
                    df[col] = df[col].ffill()

                # If there are still missing values at the beginning of the series, fill with overall median
                remaining_nulls = df[col].isnull().sum()
                if remaining_nulls > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info("Filled %d remaining NaNs in %s with median %f", remaining_nulls, col, median_val)

                logger.info("Handled %d missing values in numerical column: %s", null_count, col)

    # Add derived columns
    logger.info("Adding date-derived columns (month, year, quarter, covid_period)...")
    df[config.MONTH_COLUMN] = df[config.DATE_COLUMN].dt.month
    df[config.YEAR_COLUMN] = df[config.DATE_COLUMN].dt.year

    # covid_period: "Pre-Covid" if Date < COVID_START else "During/Post-Covid"
    covid_start_dt = pd.to_datetime(config.COVID_START)
    df[config.COVID_PERIOD_COLUMN] = df[config.DATE_COLUMN].apply(
        lambda x: "Pre-Covid" if x < covid_start_dt else "During/Post-Covid"
    )

    # quarter: Q1/Q2/Q3/Q4 based on month
    df[config.QUARTER_COLUMN] = df[config.MONTH_COLUMN].map({
        1: "Q1", 2: "Q1", 3: "Q1",
        4: "Q2", 5: "Q2", 6: "Q2",
        7: "Q3", 8: "Q3", 9: "Q3",
        10: "Q4", 11: "Q4", 12: "Q4"
    })

    # Outlier detection (IQR method) on estimated unemployment rate
    logger.info("Detecting outliers in the unemployment rate column...")
    if config.RATE_COLUMN in df.columns:
        rates = df[config.RATE_COLUMN]
        q1 = rates.quantile(0.25)
        q3 = rates.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[config.IS_OUTLIER_COLUMN] = (rates < lower_bound) | (rates > upper_bound)
        outlier_count = df[config.IS_OUTLIER_COLUMN].sum()
        logger.info(
            "IQR Outlier detection bounds: [%.2f, %.2f]. Detected %d outliers (%.2f%% of data).",
            lower_bound,
            upper_bound,
            outlier_count,
            (outlier_count / len(df)) * 100.0
        )

    return df


if __name__ == "__main__":
    # Standalone execution configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.data_loader import load_data
    cfg = AnalysisConfig()
    try:
        raw_df = load_data(cfg)
        processed_df = preprocess(raw_df, cfg)
        print("\n--- STANDALONE PREPROCESS SUCCESS ---")
        print(processed_df[[cfg.DATE_COLUMN, cfg.COVID_PERIOD_COLUMN, cfg.IS_OUTLIER_COLUMN]].head(10))
    except Exception as exc:
        logger.error("Preprocessor standalone execution failed: %s", exc, exc_info=True)
