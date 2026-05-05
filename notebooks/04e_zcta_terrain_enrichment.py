"""Build ZCTA-level terrain features from Open-Meteo elevation API.

Outputs:
  - data/processed/zcta_terrain_enrichment.csv
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors


def _fetch_elevations(lat_vals, lon_vals):
    lat_s = ",".join([f"{x:.6f}" for x in lat_vals])
    lon_s = ",".join([f"{x:.6f}" for x in lon_vals])
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_s}&longitude={lon_s}"

    resp = None
    for attempt in range(7):
        resp = requests.get(url, timeout=90)
        if resp.status_code != 429:
            break
        time.sleep(min(45, 2 ** (attempt + 1)))
    if resp is None:
        return None
    if resp.status_code == 429:
        return None
    resp.raise_for_status()
    data = resp.json()
    return data.get("elevation", [])


def main():
    base_dir = Path.cwd()
    processed = base_dir / "data" / "processed"
    out_path = processed / "zcta_terrain_enrichment.csv"

    src = processed / "zcta_level_features.parquet"
    if not src.exists():
        src = processed / "county_level_features.parquet"
    df = pd.read_parquet(src)[["ZIP_ZCTA", "final_lat", "final_lon"]].dropna().copy()
    df["ZIP_ZCTA"] = pd.to_numeric(df["ZIP_ZCTA"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    df = df.groupby("ZIP_ZCTA", as_index=False).agg({"final_lat": "mean", "final_lon": "mean"})

    # Batch elevation API calls with resume support.
    if out_path.exists():
        prior = pd.read_csv(out_path)
        prior["ZIP_ZCTA"] = pd.to_numeric(prior["ZIP_ZCTA"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
        df = df.merge(prior[["ZIP_ZCTA", "terrain_mean_elevation_m"]], on="ZIP_ZCTA", how="left")
    else:
        df["terrain_mean_elevation_m"] = np.nan

    batch_size = 30
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        chunk = df.iloc[start:end]
        if chunk["terrain_mean_elevation_m"].notna().all():
            continue
        elev = _fetch_elevations(chunk["final_lat"].tolist(), chunk["final_lon"].tolist())
        if elev is None or len(elev) != len(chunk):
            continue
        df.loc[chunk.index, "terrain_mean_elevation_m"] = pd.to_numeric(
            pd.Series(elev), errors="coerce"
        ).to_numpy()
        if start % (batch_size * 20) == 0:
            df[["ZIP_ZCTA", "terrain_mean_elevation_m"]].to_csv(out_path, index=False)
        time.sleep(0.4)

    # Local ruggedness = std elevation among k nearest centroid neighbors.
    valid = df["terrain_mean_elevation_m"].notna()
    valid_df = df[valid].copy()
    coords = valid_df[["final_lat", "final_lon"]].to_numpy()
    neigh = NearestNeighbors(n_neighbors=min(9, len(valid_df)), algorithm="ball_tree").fit(coords)
    _, idx = neigh.kneighbors(coords)
    elev_arr = valid_df["terrain_mean_elevation_m"].to_numpy()
    rugged = np.array([np.nanstd(elev_arr[row]) for row in idx])
    valid_df["terrain_ruggedness"] = rugged

    out = df[["ZIP_ZCTA"]].merge(
        valid_df[["ZIP_ZCTA", "terrain_mean_elevation_m", "terrain_ruggedness"]],
        on="ZIP_ZCTA",
        how="left",
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(out):,} rows)")
    print(f"Elevation coverage: {(out['terrain_mean_elevation_m'].notna().mean()*100):.2f}%")


if __name__ == "__main__":
    main()
