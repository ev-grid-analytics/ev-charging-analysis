"""Generate core EDA figures from processed tables."""

from pathlib import Path

import pandas as pd
import plotly.express as px


def main():
    base_dir = Path.cwd()
    processed = base_dir / "data" / "processed"
    reports = base_dir / "reports"
    reports.mkdir(exist_ok=True)

    clean = pd.read_parquet(processed / "cleaned_stations.parquet")
    zcta = pd.read_parquet(processed / "zcta_level_features.parquet")

    by_month = clean.dropna(subset=["install_month"]).groupby("install_month", as_index=False)["ID"].count()
    by_month = by_month.rename(columns={"ID": "station_count"})
    fig_month = px.line(by_month, x="install_month", y="station_count", title="Installations by Month")
    fig_month.write_html(reports / "fig_installations_by_month.html")

    by_region = zcta.groupby("region", as_index=False)["is_charging_desert"].mean()
    by_region["desert_pct"] = by_region["is_charging_desert"] * 100
    fig_region = px.bar(by_region, x="region", y="desert_pct", title="Charging Desert Rate by Region (%)")
    fig_region.write_html(reports / "fig_desert_rate_by_region.html")

    zplot = zcta.dropna(subset=["final_lat", "final_lon"]).copy()
    fig_map = px.scatter_geo(
        zplot,
        lat="final_lat",
        lon="final_lon",
        color="nearest_dcfc_miles",
        hover_name="ZIP_ZCTA",
        color_continuous_scale="Viridis",
        title="Nearest DCFC Distance by ZCTA Centroid",
    )
    fig_map.write_html(reports / "fig_nearest_dcfc_map.html")

    print("Wrote EDA figures to reports/*.html")


if __name__ == "__main__":
    main()
