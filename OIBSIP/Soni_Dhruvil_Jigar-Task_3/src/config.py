"""Configuration module for the Car Price Prediction ML system.

This module defines directory paths, model constants, and feature column names 
used throughout the machine learning pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class AppConfig:
    """Application configuration container."""

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / "data"
    DATA_FILE: str = "car_data.csv"
    
    MODEL_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Model tuning and split parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_SPLITS: int = 5
    
    # Columns
    TARGET_COLUMN: str = "Selling_Price"
    
    # Raw dataset column lists
    CATEGORICAL_COLUMNS: List[str] = field(
        default_factory=lambda: ["Fuel_Type", "Seller_Type", "Transmission"]
    )
    NUMERICAL_COLUMNS: List[str] = field(
        default_factory=lambda: ["Present_Price", "Kms_Driven", "Owner", "car_age"]
    )
    
    # Column mapping for raw-to-standardised column names
    COLUMN_RENAME_MAP: dict = field(
        default_factory=lambda: {
            "Driven_kms": "Kms_Driven",
            "Selling_type": "Seller_Type"
        }
    )
    
    # High cardinality columns to drop
    COLUMNS_TO_DROP: List[str] = field(
        default_factory=lambda: ["Car_Name"]
    )

    def __post_init__(self) -> None:
        """Create necessary directories if they do not exist."""
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (self.REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Instantiated configuration object for global use
config = AppConfig()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Loaded config successfully.")
    logger.info("Base Directory: %s", config.BASE_DIR)
    logger.info("Data Path: %s", config.DATA_PATH / config.DATA_FILE)
