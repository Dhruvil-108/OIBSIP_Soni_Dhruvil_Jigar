"""Module for Time-Series Analysis of unemployment rates.

Plots national average trends, monthly trends for the top 5 most affected regions,
rolling averages, and pre- vs. post-covid trends.
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


def run_time_series_analysis(df: pd.DataFrame, config: AnalysisConfig) -> dict:
    """Performs time-series analysis on the unemployment rate.

    Generates visualisations 07_ to 10_ and saves them to reports/figures/.
    Computes key time-series metrics: peak month, highest spike region,
    and percentage change from pre to post Covid.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        config (AnalysisConfig): Configuration settings.

    Returns:
        dict: Time-series analysis metrics.
    """
    logger.info("Starting Time-Series Analysis...")
    metrics = {}

    # Set styling parameters for consistent premium look
    sns.set_theme(style="whitegrid")

    # 1. National Average over time
    national_avg = (
        df.groupby(config.DATE_COLUMN)[config.RATE_COLUMN]
        .mean()
        .reset_index()
        .sort_values(by=config.DATE_COLUMN)
    )

    # Compute peak unemployment month (national)
    peak_row = national_avg.loc[national_avg[config.RATE_COLUMN].idxmax()]
    peak_date = peak_row[config.DATE_COLUMN]
    peak_rate = peak_row[config.RATE_COLUMN]
    metrics["peak_unemployment_month"] = peak_date.strftime("%B %Y")
    metrics["peak_unemployment_date_str"] = peak_date.strftime("%Y-%m-%d")
    metrics["peak_unemployment_rate"] = float(peak_rate)

    logger.info("Peak national unemployment of %.2f%% occurred in %s", peak_rate, metrics["peak_unemployment_month"])

    # Plot 07: Unemployment Rate Over Time with Covid marker
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=national_avg, x=config.DATE_COLUMN, y=config.RATE_COLUMN, marker="o", color="#2c3e50", linewidth=2.5, ax=ax)
    covid_start_dt = pd.to_datetime(config.COVID_START)
    ax.axvline(covid_start_dt, color="#e74c3c", linestyle="--", linewidth=2, label=f"Covid-19 Lockdowns ({config.COVID_START})")
    ax.set_title("National Average Unemployment Rate Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "07_unemployment_rate_over_time.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 07: Unemployment over time.")

    # 2. Monthly Trend by Region (Top 5 most affected regions)
    # Define "most affected" as having the highest average rate during Covid period
    covid_df = df[df[config.COVID_PERIOD_COLUMN] == "During/Post-Covid"]
    top_5_regions = (
        covid_df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN]
        .mean()
        .nlargest(5)
        .index
        .tolist()
    )
    metrics["top_5_affected_covid_regions"] = top_5_regions
    logger.info("Top 5 affected regions during Covid: %s", top_5_regions)

    # Plot 08: Monthly trend by region
    fig, ax = plt.subplots(figsize=(11, 6))
    top_5_df = df[df[config.REGION_COLUMN].isin(top_5_regions)]
    # Group by region and date to average out rural/urban differences
    top_5_monthly = (
        top_5_df.groupby([config.REGION_COLUMN, config.DATE_COLUMN])[config.RATE_COLUMN]
        .mean()
        .reset_index()
    )
    sns.lineplot(
        data=top_5_monthly,
        x=config.DATE_COLUMN,
        y=config.RATE_COLUMN,
        hue=config.REGION_COLUMN,
        marker="o",
        linewidth=2,
        ax=ax
    )
    ax.axvline(covid_start_dt, color="#e74c3c", linestyle="--", linewidth=1.5, label="Covid Start")
    ax.set_title("Unemployment Trend for Top 5 Covid-Affected Regions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "08_monthly_trend_by_region.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 08: Monthly trend by region.")

    # 3. Rolling Average (3-month window)
    national_avg["rolling_3m"] = (
        national_avg[config.RATE_COLUMN]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # Plot 09: 3-month rolling average
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=national_avg,
        x=config.DATE_COLUMN,
        y=config.RATE_COLUMN,
        label="Monthly Avg",
        color="#bdc3c7",
        alpha=0.8,
        linestyle=":",
        ax=ax
    )
    sns.lineplot(
        data=national_avg,
        x=config.DATE_COLUMN,
        y="rolling_3m",
        label="3-Month Rolling Avg",
        color="#2980b9",
        linewidth=2.5,
        ax=ax
    )
    ax.axvline(covid_start_dt, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.set_title("National Unemployment Rate: Monthly vs 3-Month Rolling Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "09_rolling_average.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 09: Rolling average.")

    # 4. Side-by-side line plot: Pre-Covid vs Covid trend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    pre_covid_avg = national_avg[national_avg[config.DATE_COLUMN] < covid_start_dt]
    post_covid_avg = national_avg[national_avg[config.DATE_COLUMN] >= covid_start_dt]

    sns.lineplot(data=pre_covid_avg, x=config.DATE_COLUMN, y=config.RATE_COLUMN, marker="o", color="#34495e", linewidth=2, ax=ax1)
    ax1.set_title("Pre-Covid National Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Unemployment Rate (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    sns.lineplot(data=post_covid_avg, x=config.DATE_COLUMN, y=config.RATE_COLUMN, marker="o", color="#e74c3c", linewidth=2, ax=ax2)
    ax2.set_title("During/Post-Covid National Trend")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Unemployment Trend Comparison: Pre vs During/Post Covid", y=0.98)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "10_pre_vs_post_covid_trend.png", dpi=150)
    plt.close()
    logger.info("Saved Plot 10: Side-by-side trends.")

    # Region with highest spike during Covid
    # Spike = Average Rate During/Post Covid - Average Rate Pre Covid
    pre_covid_df = df[df[config.COVID_PERIOD_COLUMN] == "Pre-Covid"]
    pre_avg_per_region = pre_covid_df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN].mean()
    post_avg_per_region = covid_df.groupby(config.REGION_COLUMN)[config.RATE_COLUMN].mean()

    # Align regions
    spike_diff = post_avg_per_region - pre_avg_per_region
    spike_diff = spike_diff.dropna().sort_values(ascending=False)

    highest_spike_region = spike_diff.index[0] if not spike_diff.empty else "N/A"
    highest_spike_val = float(spike_diff.iloc[0]) if not spike_diff.empty else 0.0
    metrics["highest_spike_region"] = highest_spike_region
    metrics["highest_spike_value"] = highest_spike_val

    # National average pre vs post metrics
    pre_national_mean = float(pre_covid_df[config.RATE_COLUMN].mean())
    post_national_mean = float(covid_df[config.RATE_COLUMN].mean())

    pct_increase = ((post_national_mean - pre_national_mean) / pre_national_mean) * 100.0 if pre_national_mean > 0 else 0.0
    metrics["pre_covid_national_mean"] = pre_national_mean
    metrics["post_covid_national_mean"] = post_national_mean
    metrics["national_percentage_increase"] = pct_increase

    logger.info("Highest Covid spike region: %s with spike of +%.2f%%", highest_spike_region, highest_spike_val)
    logger.info("National pre-Covid mean: %.2f%%, post-Covid mean: %.2f%%. Increase of %.2f%%",
                pre_national_mean, post_national_mean, pct_increase)

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
        ts_metrics = run_time_series_analysis(processed_df, cfg)
        print("\n--- STANDALONE TIME-SERIES SUCCESS ---")
        print("Time Series Metrics:")
        for k, v in ts_metrics.items():
            print(f"{k}: {v}")
    except Exception as exc:
        logger.error("Time-Series standalone execution failed: %s", exc, exc_info=True)
