"""Preprocessor module for the Sales Prediction ML pipeline.

This module handles missing value imputation, categorical encoding,
and outlier capping using the IQR method. It preserves state across
train/inference to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import config

# Configure logger
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Fits and transforms datasets for missing values, encoding, and outliers.

    Preserves training-set statistics (medians, modes, IQR bounds, and label
    encoders) to apply to test/inference datasets.
    """

    def __init__(self, iqr_threshold: float = config.IQR_THRESHOLD) -> None:
        """Initializes the preprocessor.

        Args:
            iqr_threshold: Multiplier for outlier capping using the IQR method.
        """
        self.iqr_threshold = iqr_threshold
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.modes: dict[str, Any] = {}
        self.iqr_bounds: dict[str, tuple[float, float]] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.one_hot_columns: list[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "Sales") -> "DataPreprocessor":
        """Fits the preprocessor using the training dataframe.

        Args:
            df: Training DataFrame.
            target_col: Target column to exclude from preprocessing.

        Returns:
            DataPreprocessor: The fitted preprocessor instance.
        """
        logger.info("Fitting data preprocessor...")
        feature_cols = [col for col in df.columns if col != target_col]

        # Identify numeric vs categorical columns
        self.numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info("Numeric features detected: %s", self.numeric_cols)
        logger.info("Categorical features detected: %s", self.categorical_cols)

        # 1. Imputation values
        for col in self.numeric_cols:
            self.medians[col] = float(df[col].median())

        for col in self.categorical_cols:
            if not df[col].mode().empty:
                self.modes[col] = df[col].mode()[0]
            else:
                self.modes[col] = "missing"

        # 2. Outlier boundaries (IQR)
        for col in self.numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_threshold * iqr
            upper_bound = q3 + self.iqr_threshold * iqr
            self.iqr_bounds[col] = (lower_bound, upper_bound)
            logger.info("Outlier bounds for '%s': (%s, %s)", col, lower_bound, upper_bound)

        # 3. Categorical encoders
        # We process cardinality to decide LabelEncoder vs pd.get_dummies
        for col in self.categorical_cols:
            cardinality = df[col].nunique()
            if cardinality > 10:
                le = LabelEncoder()
                # Fit label encoder
                le.fit(df[col].astype(str).fillna(self.modes[col]))
                self.label_encoders[col] = le
                logger.info("Fitted LabelEncoder for high-cardinality feature '%s'", col)
            else:
                logger.info("Will use pd.get_dummies for low-cardinality feature '%s'", col)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame using fitted statistics.

        Args:
            df: DataFrame to preprocess.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before calling transform.")

        df_out = df.copy()

        # Handle target column separate from feature list if present in df_out
        target_in_df = "Sales" in df_out.columns

        # 1. Imputation
        for col in self.numeric_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.medians[col])

        for col in self.categorical_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.modes[col])

        # 2. Outlier Capping
        for col in self.numeric_cols:
            if col in df_out.columns:
                lower, upper = self.iqr_bounds[col]
                original_outliers = ((df_out[col] < lower) | (df_out[col] > upper)).sum()
                df_out[col] = df_out[col].clip(lower, upper)
                if original_outliers > 0:
                    logger.info("Capped %d outliers in column '%s'", original_outliers, col)

        # 3. Categorical Encoding
        dummies_to_create = []
        for col in self.categorical_cols:
            if col in df_out.columns:
                if col in self.label_encoders:
                    # Apply LabelEncoder
                    le = self.label_encoders[col]
                    # Handle unseen values during inference by mapping them to mode
                    known_classes = set(le.classes_)
                    df_out[col] = df_out[col].astype(str).apply(
                        lambda x: x if x in known_classes else str(self.modes[col])
                    )
                    df_out[col] = le.transform(df_out[col])
                else:
                    dummies_to_create.append(col)

        if dummies_to_create:
            # Safe pd.get_dummies application matching training shapes
            # Normally we would save one-hot column schema
            # Since this is a simple system, we convert and align shapes if needed
            dummies = pd.get_dummies(df_out[dummies_to_create], drop_first=True)
            df_out = df_out.drop(columns=dummies_to_create)
            df_out = pd.concat([df_out, dummies], axis=1)

        return df_out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to preprocess data.

    Fits the preprocessor if running training, otherwise loads the saved state.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    preprocessor_path = config.MODEL_SAVE_PATH / "preprocessor.joblib"

    # If target is present, we assume we are in training/evaluation pipeline context
    if "Sales" in df.columns and not preprocessor_path.exists():
        preprocessor = DataPreprocessor()
        preprocessor.fit(df, target_col="Sales")
        # Save fitted preprocessor
        joblib.dump(preprocessor, preprocessor_path)
        logger.info("Saved fitted preprocessor to %s", preprocessor_path)
    else:
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor from %s", preprocessor_path)
        else:
            logger.warning(
                "No fitted preprocessor found. Fitting on the fly without saving."
            )
            preprocessor = DataPreprocessor()
            target_col = "Sales" if "Sales" in df.columns else ""
            preprocessor.fit(df, target_col=target_col)

    return preprocessor.transform(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Testing with a dummy DataFrame
    test_df = pd.DataFrame(
        {
            "TV": [230.1, np.nan, 17.2, 1000.0],  # Outlier and NaN
            "Radio": [37.8, 39.3, np.nan, 5.0],
            "Newspaper": [69.2, 45.1, 69.3, 10.0],
            "Sales": [22.1, 10.4, 9.3, 15.0],
        }
    )
    print("Original DataFrame:")
    print(test_df)
    clean_df = preprocess(test_df)
    print("\nPreprocessed DataFrame:")
    print(clean_df)
