"""Module for Exploratory Data Analysis (EDA).

Generates distribution, area, regional, regression, correlation, and
pandemic-comparison plots, and saves them to reports/figures/.
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


def run_eda(df: pd.DataFrame, config: AnalysisConfig) -> dict:
    """Executes Exploratory Data Analysis.

    Generates visualisations and saves them to reports/figures/. Also computes
    and returns a dictionary of key statistics.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        config (AnalysisConfig): Configuration settings.

    Returns:
        dict: Key statistical indicators computed from the dataset.
    """
    logger.info("Starting Exploratory Data Analysis...")

    # Set styling parameters for premium look
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16
    })

    # Color palette
    colors_area = {"Rural": "#27ae60", "Urban": "#2980b9"}
    colors_covid = {"Pre-Covid": "#34495e", "During/Post-Covid": "#e74c3c"}

    stats = {}

    # Calculate statistics
    rates = df[config.RATE_COLUMN]
    stats["overall_mean"] = float(rates.mean())
    stats["overall_median"] = float(rates.median())
    stats["overall_std"] = float(rates.std())

    # Pre vs Post-covid metrics
    pre_mask = df[config.COVID_PERIOD_COLUMN] == "Pre-Covid"
    post_mask = df[config.COVID_PERIOD_COLUMN] == "During/Post-Covid"

    stats["pre_covid_mean"] = float(rates[pre_mask].mean()) if pre_mask.any() else 0.0
    stats["pre_covid_median"] = float(rates[pre_mask].median()) if pre_mask.any() else 0.0
    stats["post_covid_mean"] = float(rates[post_mask].mean()) if post_mask.any() else 0.0
    stats["post_covid_median"] = float(rates[post_mask].median()) if post_mask.any() else 0.0

    logger.info("Computed basic summary statistics.")

    # 1. Unemployment Rate Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=config.RATE_COLUMN, kde=True, color="#34495e", bins=30, ax=ax)
    ax.set_title("Distribution of Estimated Unemployment Rate (%)")
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path1 = config.FIGURES_DIR / "01_unemployment_rate_distribution.png"
    plt.savefig(plot_path1, dpi=150)
    plt.close()
    logger.info("Saved Plot 01: Distribution.")

    # 2. Unemployment by Area (Rural vs Urban)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x=config.AREA_COLUMN,
        y=config.RATE_COLUMN,
        hue=config.AREA_COLUMN,
        palette=colors_area,
        legend=False,
        ax=ax
    )
    ax.set_title("Unemployment Rate Comparison: Rural vs Urban")
    ax.set_xlabel("Area")
    ax.set_ylabel("Unemployment Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path2 = config.FIGURES_DIR / "02_unemployment_by_area.png"
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    logger.info("Saved Plot 02: Rural vs Urban Comparison.")

    # 3. Unemployment by Region
    fig, ax = plt.subplots(figsize=(10, 8))
    avg_by_region = (
        df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN]
        .mean()
        .reset_index()
        .sort_values(by=config.RATE_COLUMN, ascending=False)
    )
    sns.barplot(
        data=avg_by_region,
        x=config.RATE_COLUMN,
        y=config.REGION_COLUMN,
        hue=config.REGION_COLUMN,
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.set_title("Average Unemployment Rate by Region")
    ax.set_xlabel("Average Unemployment Rate (%)")
    ax.set_ylabel("Region")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_path3 = config.FIGURES_DIR / "03_unemployment_by_region.png"
    plt.savefig(plot_path3, dpi=150)
    plt.close()
    logger.info("Saved Plot 03: Avg by Region.")

    # 4. Labour Participation vs Unemployment
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=config.LABOUR_PART_COLUMN,
        y=config.RATE_COLUMN,
        scatter_kws={"alpha": 0.5, "color": "#16a085"},
        line_kws={"color": "#e74c3c", "linewidth": 2},
        ax=ax
    )
    ax.set_title("Labour Participation Rate vs Unemployment Rate")
    ax.set_xlabel("Labour Participation Rate (%)")
    ax.set_ylabel("Unemployment Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path4 = config.FIGURES_DIR / "04_labour_participation_vs_unemployment.png"
    plt.savefig(plot_path4, dpi=150)
    plt.close()
    logger.info("Saved Plot 04: Scatter & Regression.")

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = [
        config.RATE_COLUMN,
        config.EMPLOYED_COLUMN,
        config.LABOUR_PART_COLUMN,
        config.MONTH_COLUMN,
        config.YEAR_COLUMN
    ]
    # Filter columns that exist
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plot_path5 = config.FIGURES_DIR / "05_correlation_heatmap.png"
    plt.savefig(plot_path5, dpi=150)
    plt.close()
    logger.info("Saved Plot 05: Correlation Heatmap.")

    # 6. Covid Period Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x=config.COVID_PERIOD_COLUMN,
        y=config.RATE_COLUMN,
        hue=config.COVID_PERIOD_COLUMN,
        palette=colors_covid,
        legend=False,
        ax=ax
    )
    ax.set_title("Unemployment Rate: Pre-Covid vs During/Post-Covid")
    ax.set_xlabel("Pandemic Period")
    ax.set_ylabel("Unemployment Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path6 = config.FIGURES_DIR / "06_covid_period_comparison.png"
    plt.savefig(plot_path6, dpi=150)
    plt.close()
    logger.info("Saved Plot 06: Covid Period Comparison.")

    logger.info("All EDA plots generated successfully.")
    return stats


if __name__ == "__main__":
    # Standalone execution configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.data_loader import load_data
    from src.preprocessor import preprocess
    cfg = AnalysisConfig()
    try:
        raw_df = load_data(cfg)
        processed_df = preprocess(raw_df, cfg)
        eda_stats = run_eda(processed_df, cfg)
        print("\n--- STANDALONE EDA SUCCESS ---")
        print("EDA Stats:")
        for k, v in eda_stats.items():
            print(f"{k}: {v:.4f}")
    except Exception as exc:
        logger.error("EDA standalone execution failed: %s", exc, exc_info=True)
