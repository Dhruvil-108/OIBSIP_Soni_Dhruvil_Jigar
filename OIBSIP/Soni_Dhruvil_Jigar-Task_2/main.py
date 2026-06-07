"""Main entrypoint for the Unemployment Analysis data science pipeline.

Provides a Command Line Interface (CLI) using argparse to run the full analysis,
run EDA plots only, or regenerate the insights report from cached statistics.
"""

import argparse
import json
import logging
from pathlib import Path
from src.config import AnalysisConfig
from src.data_loader import load_data
from src.preprocessor import preprocess
from src.eda import run_eda
from src.time_series_analysis import run_time_series_analysis
from src.regional_analysis import run_regional_analysis
from src.report_generator import generate_report


def setup_logging(config: AnalysisConfig) -> None:
    """Sets up console and file logging.

    Logs are directed to both stdout and to logs/analysis.log.

    Args:
        config (AnalysisConfig): Configuration settings containing the logs directory.
    """
    log_file = config.LOGS_DIR / "analysis.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8")
        ]
    )


def main() -> None:
    """Parses command-line arguments and routes flow to specific pipeline phases."""
    parser = argparse.ArgumentParser(
        description="Production-Grade Unemployment Analysis Pipeline CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--analyse",
        action="store_true",
        help="Run the complete analysis pipeline (data load, preprocess, EDA, time-series, regional, report)"
    )
    group.add_argument(
        "--eda-only",
        action="store_true",
        help="Run only the Exploratory Data Analysis (EDA) plotting stage"
    )
    group.add_argument(
        "--report",
        action="store_true",
        help="Regenerate the final report from previously cached pipeline statistics"
    )

    args = parser.parse_args()

    # Initialise configuration
    config = AnalysisConfig()
    setup_logging(config)

    logger = logging.getLogger("main_pipeline")
    logger.info("Started pipeline run.")

    cache_path = config.REPORTS_DIR / "cached_stats.json"

    if args.analyse:
        logger.info("Executing FULL analysis pipeline.")

        # Stage 1: Load
        df_raw = load_data(config)

        # Stage 2: Preprocess
        df_clean = preprocess(df_raw, config)

        # Stage 3: EDA
        logger.info("Running EDA stage.")
        eda_stats = run_eda(df_clean, config)

        # Stage 4: Time-Series Analysis
        logger.info("Running Time-Series Analysis stage.")
        ts_stats = run_time_series_analysis(df_clean, config)

        # Stage 5: Regional Analysis
        logger.info("Running Regional Analysis stage.")
        reg_stats = run_regional_analysis(df_clean, config)

        # Merge statistical dictionaries
        combined_stats = {**eda_stats, **ts_stats, **reg_stats}

        # Cache statistics to JSON file
        logger.info("Caching statistics to: %s", cache_path)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(combined_stats, f, indent=4)

        # Stage 6: Report Generation
        logger.info("Running Report Generation stage.")
        generate_report(combined_stats, config)

        logger.info("FULL analysis pipeline finished successfully.")
        print("\n=== PIPELINE SUCCESSFUL ===")
        print("Generated 14 plots saved in reports/figures/")
        print(f"Generated report at: {config.REPORTS_DIR / 'unemployment_insights.md'}")
        print(f"National Unemployment Average: {combined_stats.get('overall_mean', 0.0):.2f}%")
        print(f"Pre-Covid National Average: {combined_stats.get('pre_covid_national_mean', 0.0):.2f}%")
        print(f"Post-Covid National Average: {combined_stats.get('post_covid_national_mean', 0.0):.2f}%")
        print(f"National Percentage Increase: {combined_stats.get('national_percentage_increase', 0.0):.2f}%")
        print(f"Peak Month: {combined_stats.get('peak_unemployment_month', 'N/A')} ({combined_stats.get('peak_unemployment_rate', 0.0):.2f}%)")
        print(f"Highest Spike Region: {combined_stats.get('highest_spike_region', 'N/A')} (+{combined_stats.get('highest_spike_value', 0.0):.2f}%)")
        print("===========================")

    elif args.eda_only:
        logger.info("Executing EDA-only plotting.")

        # Stage 1: Load
        df_raw = load_data(config)

        # Stage 2: Preprocess
        df_clean = preprocess(df_raw, config)

        # Stage 3: EDA
        run_eda(df_clean, config)

        logger.info("EDA plotting finished successfully.")
        print("\n=== EDA ONLY SUCCESSFUL ===")
        print("Generated 6 plots (01–06) saved in reports/figures/")
        print("===========================")

    elif args.report:
        logger.info("Executing Report Regeneration from cached file.")

        if not cache_path.exists():
            error_msg = f"Cannot regenerate report. No cache file found at: {cache_path}. Run with --analyse first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Reading cached statistics from: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_stats = json.load(f)

        # Regenerate report
        generate_report(cached_stats, config)

        logger.info("Report regeneration finished successfully.")
        print("\n=== REPORT REGENERATION SUCCESSFUL ===")
        print(f"Report path: {config.REPORTS_DIR / 'unemployment_insights.md'}")
        print("===========================")


if __name__ == "__main__":
    main()
