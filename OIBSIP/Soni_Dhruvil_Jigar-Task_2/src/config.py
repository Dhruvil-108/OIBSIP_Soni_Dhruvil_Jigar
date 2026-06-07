"""Configuration module for the Unemployment Analysis project.

Contains dataclass configuration for paths, file patterns, and constants.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Dataclass holding all configuration settings and directory paths."""

    # Base directory paths
    BASE_DIR: Path = Path("C:/Users/DHRUVIL/OneDrive/Documents/Oasis_InfoByte/OIBSIP/unemployment-analysis")
    DATA_PATH: Path = BASE_DIR / "data"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Dataset file name
    CSV_FILENAME: str = "Unemployment in India.csv"

    # Column configuration
    DATE_COLUMN: str = "Date"
    RATE_COLUMN: str = "Estimated Unemployment Rate (%)"
    REGION_COLUMN: str = "Region"
    AREA_COLUMN: str = "Area"
    EMPLOYED_COLUMN: str = "Estimated Employed"
    LABOUR_PART_COLUMN: str = "Estimated Labour Participation Rate (%)"
    FREQUENCY_COLUMN: str = "Frequency"

    # Derived Column names
    MONTH_COLUMN: str = "month"
    YEAR_COLUMN: str = "year"
    QUARTER_COLUMN: str = "quarter"
    COVID_PERIOD_COLUMN: str = "covid_period"
    IS_OUTLIER_COLUMN: str = "is_outlier"

    # Pandemic time marker
    COVID_START: str = "2020-03-01"

    def __post_init__(self) -> None:
        """Create all required project directories on initialization."""
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Self-test configuration output
    config = AnalysisConfig()
    print(f"Base Directory: {config.BASE_DIR}")
    print(f"Data Path: {config.DATA_PATH}")
    print(f"Reports Directory: {config.REPORTS_DIR}")
    print(f"Figures Directory: {config.FIGURES_DIR}")
    print(f"Logs Directory: {config.LOGS_DIR}")
    print("All directories initialized and verified successfully.")
