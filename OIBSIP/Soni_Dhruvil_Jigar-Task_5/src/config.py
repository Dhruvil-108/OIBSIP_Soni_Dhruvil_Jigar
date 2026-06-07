"""Configuration module for the Sales Prediction ML pipeline.

This module defines the Config dataclass to centralize all constants,
paths, and hyperparameters used across the project modules.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the pipeline constants and paths.

    Attributes:
        BASE_DIR: Root directory of the sales-prediction project.
        DATA_PATH: Absolute path to the dataset folder.
        MODEL_SAVE_PATH: Directory path to save trained models and scalers.
        REPORTS_DIR: Directory path to save metrics CSV files.
        FIGURES_DIR: Directory path to save performance plots.
        LOGS_DIR: Directory path to save runtime logs.
        TEST_SIZE: Train-test split ratio.
        RANDOM_STATE: Random state seed for reproducibility.
        IQR_THRESHOLD: Multiplier for identifying outliers using the IQR method.
        CV_FOLDS: Number of cross-validation folds.
        TARGET_COL_PATTERNS: List of potential target column name strings.
    """

    BASE_DIR: Path = Path(
        "C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction"
    ).resolve()
    DATA_PATH: Path = Path(
        "C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/data"
    ).resolve()
    MODEL_SAVE_PATH: Path = Path(
        "C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/models"
    ).resolve()
    REPORTS_DIR: Path = Path(
        "C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/reports"
    ).resolve()
    FIGURES_DIR: Path = Path(
        "C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/reports/figures"
    ).resolve()
    LOGS_DIR: Path = Path(
        "C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/sales-prediction/logs"
    ).resolve()

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    IQR_THRESHOLD: float = 1.5
    CV_FOLDS: int = 5
    TARGET_COL_PATTERNS: tuple[str, ...] = (
        "sales",
        "sales_amount",
        "revenue",
        "target",
    )

    def __post_init__(self) -> None:
        """Auto-creates directories if they do not exist."""
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Global instance of the configuration
config = Config()

if __name__ == "__main__":
    print(f"Base Directory: {config.BASE_DIR}")
    print(f"Data Path: {config.DATA_PATH}")
    print(f"Model Save Path: {config.MODEL_SAVE_PATH}")
    print(f"Reports Path: {config.REPORTS_DIR}")
    print(f"Figures Path: {config.FIGURES_DIR}")
    print(f"Logs Path: {config.LOGS_DIR}")
