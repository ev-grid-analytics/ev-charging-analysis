"""Minimal Streamlit dashboard for desert risk and recommendations."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def style_us_map(fig):
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(243, 243, 243)",
    )
    return fig


def main():
    base_dir = Path.cwd()
    processed = base_dir / "data" / "processed"

    st.title("EV Charging Desert Analysis and Site Recommendations")

    zcta = pd.read_parquet(processed / "zcta_modeling_features.parquet")
    preds = pd.read_parquet(processed / "predictions_zcta.parquet")
    sites = pd.read_parquet(processed / "recommended_sites_topN.parquet")

    merged = zcta.merge(
        preds[
            [
                "ZIP_ZCTA",
                "predicted_desert_prob",
                "distance_residual_miles",
                "nearest_dcfc_miles",
            ]
        ],
        on="ZIP_ZCTA",
        how="left",
    )

    st.subheader("Current Deserts (nearest DCFC miles)")
    fig1 = px.scatter_geo(
        merged.dropna(subset=["final_lat", "final_lon"]),
        lat="final_lat",
        lon="final_lon",
        color="nearest_dcfc_miles",
        hover_name="ZIP_ZCTA",
        color_continuous_scale="Viridis",
        title="ZCTA centroids colored by nearest_dcfc_miles",
    )
    fig1 = style_us_map(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Predicted High-Risk ZCTAs")
    fig2 = px.scatter_geo(
        merged.dropna(subset=["final_lat", "final_lon"]),
        lat="final_lat",
        lon="final_lon",
        color="predicted_desert_prob",
        hover_name="ZIP_ZCTA",
        color_continuous_scale="Reds",
        title="Predicted desert probability",
    )
    fig2 = style_us_map(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recommended Sites")
    n = st.slider("Top N scenario", min_value=100, max_value=1000, step=100, value=100)
    chosen = sites[sites["scenario_top_n"] == n].copy()
    fig3 = px.scatter_geo(
        chosen,
        lat="lat",
        lon="lon",
        color="composite_score",
        hover_name="candidate_id",
        color_continuous_scale="Plasma",
        title=f"Top {n} recommended sites",
    )
    fig3 = style_us_map(fig3)
    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
