# Unemployment Rate Analysis Insights Report

## 1. Executive Summary
During the analyzed period, the national average unemployment rate in India stood at **11.79%**, with a median of **8.35%** and standard deviation of **10.72%**. The onset of the Covid-19 pandemic and the subsequent lockdown measures triggered a massive surge in unemployment across the country. The national average unemployment rate rose from **9.51%** in the Pre-Covid period to **17.77%** During/Post-Covid, representing a sharp relative increase of **86.91%** in national unemployment.

## 2. Key Statistics
| Indicator | Value |
| --- | --- |
| **National Average Unemployment Rate** | 11.79% |
| **Median Unemployment Rate** | 8.35% |
| **Standard Deviation of Unemployment Rate** | 10.72% |
| **Peak National Unemployment Month** | May 2020 |
| **Peak National Unemployment Rate** | 24.88% |
| **Pre-Covid National Average Rate** | 9.51% |
| **Covid/Post-Covid National Average Rate** | 17.77% |
| **National Percentage Increase (Pre vs Post-Covid)** | 86.91% |

## 3. Covid-19 Pandemic Impact
The pandemic caused severe and immediate disruption to the Indian labor market:
- **National Surge:** The national unemployment rate increased by **86.91%** on average, jumping from a baseline of **9.51%** to **17.77%**.
- **Most Severe Absolute Spike:** **Puducherry** was the hardest-hit region in absolute terms, experiencing an average rate increase of **+37.36%** (comparing average unemployment rate post-Covid vs pre-Covid).
- **Highest Percentage Relative Spike:** **Puducherry** recorded the highest relative growth in unemployment, experiencing a **2345.39%** relative increase from its pre-pandemic baseline.

## 4. Regional Analysis

### Top 5 Most Affected Regions (Overall Average)
1. **Tripura** (28.35%)
2. **Haryana** (26.28%)
3. **Jharkhand** (20.59%)
4. **Bihar** (18.92%)
5. **Himachal Pradesh** (18.54%)

### Top 5 Least Affected Regions (Overall Average)
1. **Meghalaya** (4.80%)
2. **Odisha** (5.66%)
3. **Assam** (6.43%)
4. **Uttarakhand** (6.58%)
5. **Gujarat** (6.66%)

## 5. Rural vs Urban Comparison
Urban areas consistently experienced higher average unemployment compared to rural areas during this timeframe:
- **Rural Average Unemployment Rate:** **10.32%**
- **Urban Average Unemployment Rate:** **13.17%**
- **Area Gap:** Urban unemployment was higher than rural unemployment by **2.84%** percentage points on average.

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
