"""Generate candidate EV charging sites from high-priority ZCTAs."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    base_dir = Path.cwd()
    processed = base_dir / "data" / "processed"

    preds = pd.read_parquet(processed / "predictions_zcta.parquet")
    features = pd.read_parquet(processed / "zcta_modeling_features.parquet")

    df = preds.merge(
        features[["ZIP_ZCTA", "final_lat", "final_lon"]],
        on="ZIP_ZCTA",
        how="left",
        suffixes=("", "_feat"),
    )
    df["lat"] = df["final_lat"].fillna(df["final_lat_feat"])
    df["lon"] = df["final_lon"].fillna(df["final_lon_feat"])

    # Prioritize actual deserts, high predicted probability, and large positive residuals.
    residual_cut = df["distance_residual_miles"].quantile(0.9)
    mask = (
        (df["is_charging_desert"] == 1)
        | (df["predicted_desert_prob"] >= 0.5)
        | (df["distance_residual_miles"] >= residual_cut)
    )
    target = df[mask & df["lat"].notna() & df["lon"].notna()].copy()

    # Candidate Type 1: ZCTA centroid
    centroid = target.copy()
    centroid["candidate_type"] = "zcta_centroid"
    centroid["candidate_id"] = "CENTROID_" + centroid["ZIP_ZCTA"].astype(str)
    centroid["candidate_rank_hint"] = 1

    # Candidate Type 2: small north/south offsets for optional micro-siting exploration.
    offset = target.copy()
    offset["candidate_type"] = "zcta_offset_north"
    offset["candidate_id"] = "OFFSETN_" + offset["ZIP_ZCTA"].astype(str)
    offset["lat"] = offset["lat"] + 0.03
    offset["candidate_rank_hint"] = 2

    offset2 = target.copy()
    offset2["candidate_type"] = "zcta_offset_south"
    offset2["candidate_id"] = "OFFSETS_" + offset2["ZIP_ZCTA"].astype(str)
    offset2["lat"] = offset2["lat"] - 0.03
    offset2["candidate_rank_hint"] = 3

    candidates = pd.concat([centroid, offset, offset2], ignore_index=True)
    candidates = candidates[
        [
            "candidate_id",
            "candidate_type",
            "candidate_rank_hint",
            "ZIP_ZCTA",
            "State",
            "region",
            "lat",
            "lon",
            "is_charging_desert",
            "predicted_desert_prob",
            "distance_residual_miles",
            "total_population",
            "median_household_income",
        ]
    ].rename(columns={"ZIP_ZCTA": "parent_zcta"})

    # Keep finite coordinates only.
    candidates = candidates[np.isfinite(candidates["lat"]) & np.isfinite(candidates["lon"])]
    candidates.to_parquet(processed / "candidate_sites.parquet", index=False)

    print("Wrote candidate sites:", len(candidates))


if __name__ == "__main__":
    main()
