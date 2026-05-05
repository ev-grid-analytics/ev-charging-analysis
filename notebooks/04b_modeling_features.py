"""Build leakage-free modeling features for desert-risk ML."""

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_area_km2(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)
    if "ALAND_SQMI" in cols:
        return pd.to_numeric(df["ALAND_SQMI"], errors="coerce") * 2.58999
    if "ALAND" in cols:
        aland = pd.to_numeric(df["ALAND"], errors="coerce")
        return np.where(aland > 1_000_000, aland / 1_000_000.0, aland * 2.58999)
    if "AREALAND" in cols:
        return pd.to_numeric(df["AREALAND"], errors="coerce") / 1_000_000.0
    raise ValueError("Expected one of ALAND_SQMI, ALAND, AREALAND in Gazetteer.")


def build_modeling_features(base_dir: Path) -> None:
    processed_dir = base_dir / "data" / "processed"
    raw_dir = base_dir / "data" / "raw"

    src_legacy = processed_dir / "county_level_features.parquet"
    src_renamed = processed_dir / "zcta_level_features.parquet"
    gazetteer_path = raw_dir / "2023_Gaz_zcta_national.txt"
    weather_path = raw_dir / "state_weather_2020_2024.csv"
    terrain_path = raw_dir / "state_terrain_2020_2024.csv"
    interstate_dist_path = processed_dir / "zcta_interstate_distance.csv"
    zcta_terrain_path = processed_dir / "zcta_terrain_enrichment.csv"
    output_path = processed_dir / "zcta_modeling_features.parquet"

    zcta_path = src_renamed if src_renamed.exists() else src_legacy
    zcta_df = pd.read_parquet(zcta_path)
    # Ensure one record per ZCTA key (some source joins can create rare duplicates).
    zcta_df["__state_notna"] = zcta_df["State"].notna().astype(int)
    zcta_df = (
        zcta_df.sort_values(
            by=["ZIP_ZCTA", "__state_notna", "total_population"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["ZIP_ZCTA"], keep="first")
        .drop(columns=["__state_notna"])
    )

    # Write ZCTA alias so naming matches content in downstream notebooks.
    if not src_renamed.exists():
        zcta_df.to_parquet(src_renamed, index=False)

    gaz_df = pd.read_csv(gazetteer_path, sep="\t")
    gaz_df["ZIP_ZCTA"] = pd.to_numeric(gaz_df["GEOID"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    gaz_df["area_km2"] = _resolve_area_km2(gaz_df)

    model_df = zcta_df.merge(gaz_df[["ZIP_ZCTA", "area_km2"]], on="ZIP_ZCTA", how="left")
    model_df["ZIP_ZCTA"] = (
        pd.to_numeric(model_df["ZIP_ZCTA"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    )

    model_df["total_population"] = pd.to_numeric(model_df["total_population"], errors="coerce")
    model_df["population_density"] = np.where(
        (model_df["total_population"].notna()) & (model_df["area_km2"] > 0),
        model_df["total_population"] / model_df["area_km2"],
        np.nan,
    )
    model_df["rurality_flag"] = np.where(
        model_df["population_density"].isna(),
        np.nan,
        np.where(model_df["population_density"] < 100.0, 1, 0),
    )

    # NEVI-style scope: flag continental US (exclude AK/HI/territories). Rows stay in the
    # table; training/evaluation in notebook 06 filters on this flag.
    non_continental = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}
    model_df["is_continental_us"] = (
        model_df["State"].notna() & ~model_df["State"].isin(non_continental)
    ).astype(np.int8)

    # Geospatial context features.
    if interstate_dist_path.exists():
        dist_df = pd.read_csv(interstate_dist_path)
        dist_df["ZIP_ZCTA"] = (
            pd.to_numeric(dist_df["ZIP_ZCTA"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
        )
        dist_df = dist_df.groupby("ZIP_ZCTA", as_index=False)["distance_to_nearest_interstate_miles"].min()
        model_df = model_df.merge(
            dist_df[["ZIP_ZCTA", "distance_to_nearest_interstate_miles"]],
            on="ZIP_ZCTA",
            how="left",
        )
    else:
        model_df["distance_to_nearest_interstate_miles"] = np.nan

    model_df["terrain_ruggedness"] = np.nan
    model_df["terrain_mean_elevation_m"] = np.nan
    if zcta_terrain_path.exists():
        zterrain = pd.read_csv(zcta_terrain_path)
        zterrain["ZIP_ZCTA"] = (
            pd.to_numeric(zterrain["ZIP_ZCTA"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
        )
        zterrain = zterrain.groupby("ZIP_ZCTA", as_index=False).agg(
            {"terrain_ruggedness": "mean", "terrain_mean_elevation_m": "mean"}
        )
        model_df = model_df.merge(zterrain, on="ZIP_ZCTA", how="left", suffixes=("", "_zcta"))
        model_df["terrain_ruggedness"] = model_df["terrain_ruggedness_zcta"].combine_first(
            model_df["terrain_ruggedness"]
        )
        model_df["terrain_mean_elevation_m"] = model_df["terrain_mean_elevation_m_zcta"].combine_first(
            model_df["terrain_mean_elevation_m"]
        )
        model_df = model_df.drop(columns=["terrain_ruggedness_zcta", "terrain_mean_elevation_m_zcta"], errors="ignore")

    if terrain_path.exists():
        terrain_df = pd.read_csv(terrain_path)
        terrain_cols = [c for c in ["State", "terrain_ruggedness", "terrain_mean_elevation_m"] if c in terrain_df.columns]
        model_df = model_df.merge(terrain_df[terrain_cols], on="State", how="left", suffixes=("", "_state"))
        if "terrain_ruggedness_state" in model_df.columns:
            model_df["terrain_ruggedness"] = model_df["terrain_ruggedness"].combine_first(
                model_df["terrain_ruggedness_state"]
            )
            model_df = model_df.drop(columns=["terrain_ruggedness_state"], errors="ignore")
        if "terrain_mean_elevation_m_state" in model_df.columns:
            model_df["terrain_mean_elevation_m"] = model_df["terrain_mean_elevation_m"].combine_first(
                model_df["terrain_mean_elevation_m_state"]
            )
            model_df = model_df.drop(columns=["terrain_mean_elevation_m_state"], errors="ignore")

    # Weather context by state if available.
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path)
        weather_cols = [
            "State",
            "weather_avg_temp_c",
            "weather_avg_precip_mm_day",
            "weather_extreme_heat_days_year",
            "weather_heavy_precip_days_year",
        ]
        weather_cols = [c for c in weather_cols if c in weather_df.columns]
        model_df = model_df.merge(weather_df[weather_cols], on="State", how="left")
    else:
        model_df["weather_avg_temp_c"] = np.nan
        model_df["weather_avg_precip_mm_day"] = np.nan
        model_df["weather_extreme_heat_days_year"] = np.nan
        model_df["weather_heavy_precip_days_year"] = np.nan

    keep_cols = [
        "ZIP_ZCTA",
        "State",
        "region",
        "is_continental_us",
        "total_population",
        "median_household_income",
        "area_km2",
        "population_density",
        "rurality_flag",
        "distance_to_nearest_interstate_miles",
        "terrain_ruggedness",
        "terrain_mean_elevation_m",
        "weather_avg_temp_c",
        "weather_avg_precip_mm_day",
        "weather_extreme_heat_days_year",
        "weather_heavy_precip_days_year",
        "nearest_dcfc_miles",  # for Reframing B target only
        "is_charging_desert",  # for Reframing A target
        "final_lat",
        "final_lon",
    ]

    missing = [col for col in keep_cols if col not in model_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    final_df = model_df[keep_cols].copy()
    final_df.to_parquet(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(final_df):,}")
    print(f"Columns: {len(final_df.columns)}")
    print("Legacy alias written:", src_renamed)


if __name__ == "__main__":
    build_modeling_features(Path.cwd())
