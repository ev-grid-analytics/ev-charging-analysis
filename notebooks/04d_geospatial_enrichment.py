"""Compute geospatial enrichment features for modeling.

Outputs:
  - data/processed/zcta_interstate_distance.csv
"""

from pathlib import Path
import zipfile

import geopandas as gpd
import pandas as pd


def _extract_primary_roads_shapefile(raw_dir: Path, tmp_dir: Path) -> Path:
    zip_path = raw_dir / "tl_2023_us_primaryroads.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing roads zip: {zip_path}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    shp_files = list(tmp_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No shapefile found after extracting primary roads zip.")
    return shp_files[0]


def main():
    base_dir = Path.cwd()
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    tmp_dir = base_dir / "data" / "raw" / "_tmp_primary_roads"
    out_path = processed_dir / "zcta_interstate_distance.csv"

    zcta_path = processed_dir / "zcta_level_features.parquet"
    if not zcta_path.exists():
        zcta_path = processed_dir / "county_level_features.parquet"
    zcta_df = pd.read_parquet(zcta_path)[["ZIP_ZCTA", "final_lat", "final_lon"]].dropna()

    roads_shp = _extract_primary_roads_shapefile(raw_dir, tmp_dir)
    roads = gpd.read_file(roads_shp)
    if roads.empty:
        raise ValueError("Primary roads shapefile loaded but has no rows.")

    pts = gpd.GeoDataFrame(
        zcta_df,
        geometry=gpd.points_from_xy(zcta_df["final_lon"], zcta_df["final_lat"]),
        crs="EPSG:4326",
    )

    # Use CONUS Albers Equal Area for distance in meters.
    roads_proj = roads.to_crs(epsg=5070)
    pts_proj = pts.to_crs(epsg=5070)

    nearest = gpd.sjoin_nearest(
        pts_proj[["ZIP_ZCTA", "geometry"]],
        roads_proj[["geometry"]],
        how="left",
        distance_col="distance_to_nearest_interstate_meters",
    )
    nearest["distance_to_nearest_interstate_miles"] = (
        nearest["distance_to_nearest_interstate_meters"] / 1609.344
    )

    out_df = nearest[["ZIP_ZCTA", "distance_to_nearest_interstate_miles"]].copy()
    out_df["ZIP_ZCTA"] = (
        pd.to_numeric(out_df["ZIP_ZCTA"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    )
    out_df = (
        out_df.groupby("ZIP_ZCTA", as_index=False)["distance_to_nearest_interstate_miles"]
        .min()
    )
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
