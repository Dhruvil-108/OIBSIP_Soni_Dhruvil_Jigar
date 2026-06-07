"""Configuration module for the Email Spam Detection System.

This module contains system configuration details including file paths
and model training hyper-parameters using a frozen dataclass.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """System configuration parameters and paths.

    Attributes:
        BASE_DIR (Path): The root directory of the project.
        DATA_PATH (Path): Path to the input dataset CSV.
        MODEL_SAVE_PATH (Path): Path to save the trained model.
        VECTORIZER_SAVE_PATH (Path): Path to save the TF-IDF vectorizer.
        REPORTS_DIR (Path): Path to save evaluation reports and plots.
        LOGS_DIR (Path): Path to save running logs.
        TEST_SIZE (float): Proportion of dataset to include in the test split.
        RANDOM_STATE (int): Random seed for reproducibility.
        MAX_FEATURES (int): Max number of features for TF-IDF Vectorizer.
    """
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / "data" / "spam.csv"
    MODEL_SAVE_PATH: Path = BASE_DIR / "models" / "best_model.joblib"
    VECTORIZER_SAVE_PATH: Path = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Model parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    MAX_FEATURES: int = 5000
