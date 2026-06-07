"""Module for generating the final insights report.

Compiles numerical statistics and plots lists into a structured markdown report
saved at reports/unemployment_insights.md.
"""

import logging
from src.config import AnalysisConfig

# Set up logging for this module
logger = logging.getLogger(__name__)


def generate_report(stats: dict, config: AnalysisConfig) -> None:
    """Generates a comprehensive Markdown report of unemployment insights.

    Extracts statistics from the provided dictionary and writes the results to
    reports/unemployment_insights.md.

    Args:
        stats (dict): Compiled metrics from loader, EDA, time-series, and regional modules.
        config (AnalysisConfig): Configuration settings containing file paths.
    """
    logger.info("Generating final insights report...")

    report_path = config.REPORTS_DIR / "unemployment_insights.md"

    # Default fallbacks if metrics are missing
    overall_mean = stats.get("overall_mean", 0.0)
    overall_median = stats.get("overall_median", 0.0)
    overall_std = stats.get("overall_std", 0.0)
    pre_covid_national_mean = stats.get("pre_covid_national_mean", 0.0)
    post_covid_national_mean = stats.get("post_covid_national_mean", 0.0)
    national_percentage_increase = stats.get("national_percentage_increase", 0.0)
    peak_unemployment_month = stats.get("peak_unemployment_month", "N/A")
    peak_unemployment_rate = stats.get("peak_unemployment_rate", 0.0)
    highest_spike_region = stats.get("highest_spike_region", "N/A")
    highest_spike_value = stats.get("highest_spike_value", 0.0)
    highest_pct_increase_region = stats.get("highest_pct_increase_region", "N/A")
    highest_pct_increase_val = stats.get("highest_pct_increase_val", 0.0)
    rural_mean = stats.get("rural_mean", 0.0)
    urban_mean = stats.get("urban_mean", 0.0)

    top_5_high_reg = stats.get("top_5_highest_unemployment", ["N/A"] * 5)
    top_5_high_rates = stats.get("top_5_highest_rates", [0.0] * 5)
    top_5_low_reg = stats.get("top_5_lowest_unemployment", ["N/A"] * 5)
    top_5_low_rates = stats.get("top_5_lowest_rates", [0.0] * 5)

    markdown_content = f"""# Unemployment Rate Analysis Insights Report

## 1. Executive Summary
During the analyzed period, the national average unemployment rate in India stood at **{overall_mean:.2f}%**, with a median of **{overall_median:.2f}%** and standard deviation of **{overall_std:.2f}%**. The onset of the Covid-19 pandemic and the subsequent lockdown measures triggered a massive surge in unemployment across the country. The national average unemployment rate rose from **{pre_covid_national_mean:.2f}%** in the Pre-Covid period to **{post_covid_national_mean:.2f}%** During/Post-Covid, representing a sharp relative increase of **{national_percentage_increase:.2f}%** in national unemployment.

## 2. Key Statistics
| Indicator | Value |
| --- | --- |
| **National Average Unemployment Rate** | {overall_mean:.2f}% |
| **Median Unemployment Rate** | {overall_median:.2f}% |
| **Standard Deviation of Unemployment Rate** | {overall_std:.2f}% |
| **Peak National Unemployment Month** | {peak_unemployment_month} |
| **Peak National Unemployment Rate** | {peak_unemployment_rate:.2f}% |
| **Pre-Covid National Average Rate** | {pre_covid_national_mean:.2f}% |
| **Covid/Post-Covid National Average Rate** | {post_covid_national_mean:.2f}% |
| **National Percentage Increase (Pre vs Post-Covid)** | {national_percentage_increase:.2f}% |

## 3. Covid-19 Pandemic Impact
The pandemic caused severe and immediate disruption to the Indian labor market:
- **National Surge:** The national unemployment rate increased by **{national_percentage_increase:.2f}%** on average, jumping from a baseline of **{pre_covid_national_mean:.2f}%** to **{post_covid_national_mean:.2f}%**.
- **Most Severe Absolute Spike:** **{highest_spike_region}** was the hardest-hit region in absolute terms, experiencing an average rate increase of **+{highest_spike_value:.2f}%** (comparing average unemployment rate post-Covid vs pre-Covid).
- **Highest Percentage Relative Spike:** **{highest_pct_increase_region}** recorded the highest relative growth in unemployment, experiencing a **{highest_pct_increase_val:.2f}%** relative increase from its pre-pandemic baseline.

## 4. Regional Analysis

### Top 5 Most Affected Regions (Overall Average)
1. **{top_5_high_reg[0]}** ({top_5_high_rates[0]:.2f}%)
2. **{top_5_high_reg[1]}** ({top_5_high_rates[1]:.2f}%)
3. **{top_5_high_reg[2]}** ({top_5_high_rates[2]:.2f}%)
4. **{top_5_high_reg[3]}** ({top_5_high_rates[3]:.2f}%)
5. **{top_5_high_reg[4]}** ({top_5_high_rates[4]:.2f}%)

### Top 5 Least Affected Regions (Overall Average)
1. **{top_5_low_reg[0]}** ({top_5_low_rates[0]:.2f}%)
2. **{top_5_low_reg[1]}** ({top_5_low_rates[1]:.2f}%)
3. **{top_5_low_reg[2]}** ({top_5_low_rates[2]:.2f}%)
4. **{top_5_low_reg[3]}** ({top_5_low_rates[3]:.2f}%)
5. **{top_5_low_reg[4]}** ({top_5_low_rates[4]:.2f}%)

## 5. Rural vs Urban Comparison
Urban areas consistently experienced higher average unemployment compared to rural areas during this timeframe:
- **Rural Average Unemployment Rate:** **{rural_mean:.2f}%**
- **Urban Average Unemployment Rate:** **{urban_mean:.2f}%**
- **Area Gap:** Urban unemployment was higher than rural unemployment by **{urban_mean - rural_mean:.2f}%** percentage points on average.

## 6. Generated Visualisations
The following plots have been generated and saved under the `reports/figures/` directory:
1. `01_unemployment_rate_distribution.png` - Distribution of the unemployment rate (Histogram & KDE)
2. `02_unemployment_by_area.png` - Rural vs. Urban unemployment rate comparison (Boxplot)
3. `03_unemployment_by_region.png` - Average unemployment rate per region (Bar chart)
4. `04_labour_participation_vs_unemployment.png` - Labour participation vs. unemployment rates (Regression plot)
5. `05_correlation_heatmap.png` - Heatmap of all numerical features
6. `06_covid_period_comparison.png` - Boxplot comparing Pre-Covid vs. During/Post-Covid rates
7. `07_unemployment_rate_over_time.png` - National average over time with Covid-19 marker line
8. `08_monthly_trend_by_region.png` - Multi-line plot of top 5 most affected regions over time
9. `09_rolling_average.png` - 3-month rolling average trend of national unemployment rate
10. `10_pre_vs_post_covid_trend.png` - Side-by-side line plot comparing pre and post Covid trends
11. `11_top10_highest_unemployment_regions.png` - Top 10 regions with highest unemployment rates
12. `12_top10_lowest_unemployment_regions.png` - Top 10 regions with lowest unemployment rates
13. `13_rural_vs_urban_by_region.png` - Grouped bar chart comparing Rural and Urban rates per region
14. `14_covid_impact_by_region.png` - Percentage change in unemployment rate pre vs during Covid per region
"""

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    logger.info("Insights report successfully written to: %s", report_path)


if __name__ == "__main__":
    # Standalone execution configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = AnalysisConfig()
    dummy_stats = {
        "overall_mean": 11.7,
        "overall_median": 9.5,
        "overall_std": 10.1,
        "pre_covid_national_mean": 9.2,
        "post_covid_national_mean": 15.6,
        "national_percentage_increase": 69.5,
        "peak_unemployment_month": "May 2020",
        "peak_unemployment_rate": 23.5,
        "highest_spike_region": "Tripura",
        "highest_spike_value": 8.5,
        "highest_pct_increase_region": "Puducherry",
        "highest_pct_increase_val": 150.0,
        "rural_mean": 10.3,
        "urban_mean": 13.1,
        "top_5_highest_unemployment": ["Tripura", "Haryana", "Jharkhand", "Bihar", "Delhi"],
        "top_5_highest_rates": [28.3, 27.5, 20.2, 19.8, 16.4],
        "top_5_lowest_unemployment": ["Meghalaya", "Odisha", "Assam", "Gujarat", "Karnataka"],
        "top_5_lowest_rates": [4.2, 5.6, 6.1, 6.3, 6.8]
    }
    try:
        generate_report(dummy_stats, cfg)
        print("\n--- STANDALONE REPORT GENERATION SUCCESS ---")
    except Exception as exc:
        logger.error("Report standalone execution failed: %s", exc, exc_info=True)
