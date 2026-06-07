"""Unit tests for the Unemployment Analysis data science pipeline.

Validates CSV loading, date parsing, column preprocessing, feature engineering,
plot generation, and insights report generation.
"""

import sys
import pytest
import pandas as pd
from src.config import AnalysisConfig
from src.data_loader import load_data
from src.preprocessor import preprocess
from src.eda import run_eda
from src.time_series_analysis import run_time_series_analysis
from src.regional_analysis import run_regional_analysis
from src.report_generator import generate_report


@pytest.fixture
def config() -> AnalysisConfig:
    """Fixture providing the pipeline configuration."""
    return AnalysisConfig()


def test_data_loading(config: AnalysisConfig) -> None:
    """Tests that dataset loads successfully and date column is parsed correctly.

    Args:
        config (AnalysisConfig): Config fixture.
    """
    df = load_data(config)
    assert isinstance(df, pd.DataFrame), "Loaded object should be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert pd.api.types.is_datetime64_any_dtype(df[config.DATE_COLUMN]), "Date column should be parsed as datetime"


def test_preprocessing(config: AnalysisConfig) -> None:
    """Tests that preprocessor imputes missing values and engineers correct features.

    Args:
        config (AnalysisConfig): Config fixture.
    """
    df_raw = load_data(config)
    df_clean = preprocess(df_raw, config)

    # Verify column existence
    assert config.MONTH_COLUMN in df_clean.columns, "month column is missing"
    assert config.YEAR_COLUMN in df_clean.columns, "year column is missing"
    assert config.QUARTER_COLUMN in df_clean.columns, "quarter column is missing"
    assert config.COVID_PERIOD_COLUMN in df_clean.columns, "covid_period column is missing"
    assert config.IS_OUTLIER_COLUMN in df_clean.columns, "is_outlier column is missing"

    # Verify covid period values
    unique_periods = df_clean[config.COVID_PERIOD_COLUMN].unique()
    assert set(unique_periods).issubset({"Pre-Covid", "During/Post-Covid"}), "Invalid values in covid_period"

    # Verify Area casing
    assert set(df_clean[config.AREA_COLUMN].unique()).issubset({"Rural", "Urban"}), "Area names not correctly Title Cased"


def test_eda_plots_and_report_generation(config: AnalysisConfig) -> None:
    """Tests that visualisations are generated and insights report is successfully written.

    Args:
        config (AnalysisConfig): Config fixture.
    """
    df_raw = load_data(config)
    df_clean = preprocess(df_raw, config)

    # Run analysis steps
    eda_stats = run_eda(df_clean, config)
    ts_stats = run_time_series_analysis(df_clean, config)
    reg_stats = run_regional_analysis(df_clean, config)

    # Verify figures directory and critical plots exist
    assert config.FIGURES_DIR.exists(), "Figures directory was not created"
    assert (config.FIGURES_DIR / "01_unemployment_rate_distribution.png").exists(), "Plot 01 missing"
    assert (config.FIGURES_DIR / "07_unemployment_rate_over_time.png").exists(), "Plot 07 missing"
    assert (config.FIGURES_DIR / "14_covid_impact_by_region.png").exists(), "Plot 14 missing"

    # Compile and generate report
    combined_stats = {**eda_stats, **ts_stats, **reg_stats}
    generate_report(combined_stats, config)

    report_file = config.REPORTS_DIR / "unemployment_insights.md"
    assert report_file.exists(), "Insights report was not generated"
    assert report_file.stat().st_size > 0, "Generated report is empty"


if __name__ == "__main__":
    # Allow running tests directly with Python
    sys.exit(pytest.main([__file__]))
