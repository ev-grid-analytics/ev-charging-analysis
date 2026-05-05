# EV Charging Infrastructure Visualization

This contains the final visualization outputs for the EV Charging Infrastructure and Range Anxiety Big Data project. The visualizations were created from cleaned and processed project datasets to analyze EV charger deployment patterns, geographic infrastructure gaps, charger-type availability, network distribution, and charging desert coverage.

## Visualization Notebook

The main visualization workflow is implemented in:

`notebooks/05_visual_eda_dashboard.ipynb`

This notebook generates summary tables, static figures, interactive charts, maps, and the final dashboard-style HTML page.

## Dashboard Output

The final dashboard-style HTML summary page is available at:

`outputs/dashboard_exports/ev_charging_visual_dashboard.html`

This dashboard combines the main project visualizations into one presentation-ready artifact for demo and report usage.

## Included Visualizations

The visualization module includes:

- State-level EV charging station deployment ranking
- City-level EV charging station deployment ranking
- Yearly EV charger installation trend
- Monthly EV charger installation trend
- Charger type distribution over time
- Overall charger type distribution
- EV charging network distribution
- State-level EV charging choropleth map
- Station-level interactive EV charging map
- State-level charging desert share analysis
- Region-level charging desert versus adequate coverage analysis
- Region-level charging desert share analysis

## Generated Output Folders

The visualization workflow produces three main categories of outputs:

- `outputs/tables/` contains reusable CSV summary tables used for charts, maps, and report analysis.
- `outputs/figures/` contains static PNG figures for the final report and interactive HTML charts for project demo usage.
- `outputs/dashboard_exports/` contains the final dashboard HTML page and this visualization README file.

## Key Output Files

Important generated files include:

- `outputs/figures/top_15_states_ev_charging_stations.png`
- `outputs/figures/top_15_cities_ev_charging_stations.png`
- `outputs/figures/yearly_ev_charger_installation_trend.png`
- `outputs/figures/monthly_ev_charger_installation_trend.png`
- `outputs/figures/charger_type_distribution_by_year.png`
- `outputs/figures/top_15_ev_charging_networks.png`
- `outputs/figures/charger_type_distribution_bar.png`
- `outputs/figures/state_level_ev_charging_choropleth.png`
- `outputs/figures/station_level_ev_charging_map.png`
- `outputs/figures/state_charging_desert_share.png`
- `outputs/figures/region_charging_desert_vs_adequate_coverage.png`
- `outputs/figures/region_charging_desert_share.png`

Interactive versions of the charts are also saved as `.html` files in the same folder.

## Visualization Purpose

The purpose of this visualization work is to convert processed EV charging datasets into clear, interpretable, and presentation-ready insights. The visuals help explain where EV charging infrastructure is concentrated, how deployment has changed over time, which charger types and networks dominate the market, and which areas may still face charging access limitations.

## Project Relevance

These visualizations support the broader project goal of analyzing EV charging infrastructure gaps and range anxiety across the United States. By combining deployment trends, geographic mapping, charger-type analysis, and charging desert classification, the dashboard provides evidence for infrastructure planning, policy prioritization, and equitable EV adoption support.

## Report and Demo Usage

For the final report, use the PNG files from:

`outputs/figures/`

For the final demo or presentation, use the interactive dashboard:

`outputs/dashboard_exports/ev_charging_visual_dashboard.html`

The dashboard can be opened directly in a browser and includes links to interactive chart versions.