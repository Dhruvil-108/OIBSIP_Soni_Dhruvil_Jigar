"""Module for Regional Analysis of unemployment rates.

Analyzes unemployment patterns across different states (regions) and areas
(Rural vs. Urban), visualising top/bottom regions and Covid-19 impact per region.
"""

import logging
from pathlib import Path
import matplotlib
# Use non-interactive Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.config import AnalysisConfig

# Set up logging for this module
logger = logging.getLogger(__name__)


def run_regional_analysis(df: pd.DataFrame, config: AnalysisConfig) -> dict:
    """Performs regional and area-based analysis on the unemployment rate.

    Generates visualisations 11_ to 14_ and saves them to reports/figures/.
    Computes and returns a dictionary of regional metrics for reporting.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        config (AnalysisConfig): Configuration settings.

    Returns:
        dict: Regional analysis metrics.
    """
    logger.info("Starting Regional Analysis...")
    metrics = {}

    sns.set_theme(style="whitegrid")

    # Compute overall regional averages
    regional_avg = (
        df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN]
        .mean()
        .reset_index()
        .sort_values(by=config.RATE_COLUMN, ascending=False)
    )

    # Top 10 highest and lowest regions
    top_10_highest = regional_avg.head(10)
    top_10_lowest = regional_avg.tail(10).sort_values(by=config.RATE_COLUMN, ascending=True)

    metrics["top_5_highest_unemployment"] = regional_avg.head(5)[config.REGION_COLUMN].tolist()
    metrics["top_5_highest_rates"] = regional_avg.head(5)[config.RATE_COLUMN].tolist()
    metrics["top_5_lowest_unemployment"] = regional_avg.tail(5).iloc[::-1][config.REGION_COLUMN].tolist()
    metrics["top_5_lowest_rates"] = regional_avg.tail(5).iloc[::-1][config.RATE_COLUMN].tolist()

    # Plot 11: Top 10 Highest Unemployment Regions
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=top_10_highest,
        x=config.RATE_COLUMN,
        y=config.REGION_COLUMN,
        hue=config.REGION_COLUMN,
        palette="Reds_r",
        legend=False,
        ax=ax
    )
    ax.set_title("Top 10 Regions with Highest Average Unemployment Rate")
    ax.set_xlabel("Average Unemployment Rate (%)")
    ax.set_ylabel("Region")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "11_top10_highest_unemployment_regions.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 11: Top 10 highest regions.")

    # Plot 12: Top 10 Lowest Unemployment Regions
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=top_10_lowest,
        x=config.RATE_COLUMN,
        y=config.REGION_COLUMN,
        hue=config.REGION_COLUMN,
        palette="Greens_d",
        legend=False,
        ax=ax
    )
    ax.set_title("Top 10 Regions with Lowest Average Unemployment Rate")
    ax.set_xlabel("Average Unemployment Rate (%)")
    ax.set_ylabel("Region")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "12_top10_lowest_unemployment_regions.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 12: Top 10 lowest regions.")

    # Plot 13: Rural vs Urban by Region (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    area_region_avg = (
        df.groupby([config.REGION_COLUMN, config.AREA_COLUMN])[config.RATE_COLUMN]
        .mean()
        .reset_index()
    )
    # Sort regions by their overall mean for readability in grouped bar chart
    region_order = regional_avg[config.REGION_COLUMN].tolist()
    sns.barplot(
        data=area_region_avg,
        x=config.REGION_COLUMN,
        y=config.RATE_COLUMN,
        hue=config.AREA_COLUMN,
        order=region_order,
        palette={"Rural": "#27ae60", "Urban": "#2980b9"},
        ax=ax
    )
    ax.set_title("Unemployment Rate: Rural vs Urban Comparison per Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Unemployment Rate (%)")
    plt.xticks(rotation=90)
    ax.legend(title="Area")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "13_rural_vs_urban_by_region.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 13: Grouped Rural/Urban by region.")

    # Plot 14: Covid Impact by Region (Percentage Change)
    pre_covid_df = df[df[config.COVID_PERIOD_COLUMN] == "Pre-Covid"]
    post_covid_df = df[df[config.COVID_PERIOD_COLUMN] == "During/Post-Covid"]

    pre_avg_region = pre_covid_df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN].mean()
    post_avg_region = post_covid_df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN].mean()

    # Calculate % Change
    # % Change = ((Post - Pre) / Pre) * 100
    impact_pct = ((post_avg_region - pre_avg_region) / pre_avg_region) * 100.0
    impact_pct = impact_pct.dropna().reset_index()
    impact_pct.columns = [config.REGION_COLUMN, "percentage_change"]
    impact_pct = impact_pct.sort_values(by="percentage_change", ascending=False)

    # Metrics for report
    metrics["highest_pct_increase_region"] = impact_pct.iloc[0][config.REGION_COLUMN]
    metrics["highest_pct_increase_val"] = float(impact_pct.iloc[0]["percentage_change"])

    # Rural vs Urban comparison summary
    rural_mask = df[config.AREA_COLUMN] == "Rural"
    urban_mask = df[config.AREA_COLUMN] == "Urban"
    metrics["rural_mean"] = float(df[rural_mask][config.RATE_COLUMN].mean()) if rural_mask.any() else 0.0
    metrics["urban_mean"] = float(df[urban_mask][config.RATE_COLUMN].mean()) if urban_mask.any() else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    # Color bars dynamically: red for increase, green for decrease
    colors = ["#e74c3c" if val >= 0 else "#2ecc71" for val in impact_pct["percentage_change"]]
    sns.barplot(
        data=impact_pct,
        x="percentage_change",
        y=config.REGION_COLUMN,
        hue=config.REGION_COLUMN,
        palette=colors,
        legend=False,
        ax=ax
    )
    ax.set_title("Covid-19 Impact: Percentage Change in Unemployment Rate per Region")
    ax.set_xlabel("Percentage Change (%)")
    ax.set_ylabel("Region")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "14_covid_impact_by_region.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 14: Covid Impact by region.")

    logger.info("All regional analysis plots generated successfully.")
    return metrics


if __name__ == "__main__":
    # Standalone execution configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.data_loader import load_data
    from src.preprocessor import preprocess
    cfg = AnalysisConfig()
    try:
        raw_df = load_data(cfg)
        processed_df = preprocess(raw_df, cfg)
        reg_metrics = run_regional_analysis(processed_df, cfg)
        print("\n--- STANDALONE REGIONAL SUCCESS ---")
        print("Regional Metrics:")
        for k, v in reg_metrics.items():
            print(f"{k}: {v}")
    except Exception as exc:
        logger.error("Regional standalone execution failed: %s", exc, exc_info=True)
