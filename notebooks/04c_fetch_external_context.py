"""Fetch external raw context datasets (roads + weather).

Outputs in data/raw:
  - tl_2023_us_primaryroads.zip
  - state_weather_2020_2024.csv
"""

from pathlib import Path
import io
import time

import pandas as pd
import requests


PRIMARY_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/"
    "tl_2023_us_primaryroads.zip"
)

# Approximate state centroid coordinates for weather aggregation.
STATE_COORDS = {
    "AL": (32.8, -86.8),
    "AK": (64.0, -152.0),
    "AZ": (34.2, -111.7),
    "AR": (35.1, -92.4),
    "CA": (37.1, -119.7),
    "CO": (39.0, -105.5),
    "CT": (41.6, -72.7),
    "DE": (39.0, -75.5),
    "DC": (38.9, -77.0),
    "FL": (27.8, -81.7),
    "GA": (32.6, -83.4),
    "HI": (20.8, -156.3),
    "ID": (44.2, -114.5),
    "IL": (40.0, -89.2),
    "IN": (39.9, -86.3),
    "IA": (42.0, -93.5),
    "KS": (38.5, -98.0),
    "KY": (37.8, -85.8),
    "LA": (31.0, -91.9),
    "ME": (45.2, -69.0),
    "MD": (39.0, -76.7),
    "MA": (42.3, -71.8),
    "MI": (44.3, -85.4),
    "MN": (46.7, -94.6),
    "MS": (32.7, -89.7),
    "MO": (38.5, -92.5),
    "MT": (46.9, -110.4),
    "NE": (41.5, -99.7),
    "NV": (39.3, -116.6),
    "NH": (43.7, -71.6),
    "NJ": (40.1, -74.5),
    "NM": (34.4, -106.1),
    "NY": (42.9, -75.5),
    "NC": (35.5, -79.4),
    "ND": (47.5, -100.5),
    "OH": (40.3, -82.8),
    "OK": (35.6, -97.5),
    "OR": (44.0, -120.5),
    "PA": (41.0, -77.8),
    "RI": (41.7, -71.6),
    "SC": (33.8, -80.9),
    "SD": (44.4, -100.2),
    "TN": (35.8, -86.4),
    "TX": (31.5, -99.3),
    "UT": (39.3, -111.7),
    "VT": (44.0, -72.7),
    "VA": (37.5, -78.7),
    "WA": (47.4, -120.7),
    "WV": (38.6, -80.6),
    "WI": (44.5, -89.5),
    "WY": (43.0, -107.6),
}


def fetch_primary_roads(raw_dir: Path) -> None:
    out_path = raw_dir / "tl_2023_us_primaryroads.zip"
    if out_path.exists():
        print(f"Exists, skipping: {out_path.name}")
        return
    resp = requests.get(PRIMARY_ROADS_URL, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Downloaded: {out_path.name}")


def fetch_state_weather(raw_dir: Path) -> None:
    out_path = raw_dir / "state_weather_2020_2024.csv"
    if out_path.exists():
        print(f"Exists, skipping: {out_path.name}")
        return
    rows = []
    for state, (lat, lon) in STATE_COORDS.items():
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            "&start_date=2020-01-01&end_date=2024-12-31"
            "&daily=temperature_2m_mean,precipitation_sum,"
            "temperature_2m_max,temperature_2m_min"
            "&timezone=UTC"
        )
        resp = None
        for attempt in range(6):
            resp = requests.get(url, timeout=60)
            if resp.status_code != 429:
                break
            time.sleep(min(30, 2**attempt))
        if resp is None:
            continue
        resp.raise_for_status()
        payload = resp.json()
        daily = payload.get("daily", {})
        if not daily:
            continue
        tmean = pd.to_numeric(pd.Series(daily.get("temperature_2m_mean", [])), errors="coerce")
        tmax = pd.to_numeric(pd.Series(daily.get("temperature_2m_max", [])), errors="coerce")
        precip = pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce")
        rows.append(
            {
                "State": state,
                "weather_avg_temp_c": float(tmean.mean()),
                "weather_avg_precip_mm_day": float(precip.mean()),
                "weather_extreme_heat_days_year": float((tmax >= 35.0).mean() * 365.25),
                "weather_heavy_precip_days_year": float((precip >= 20.0).mean() * 365.25),
            }
        )
        time.sleep(0.35)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote: {out_path.name} ({len(rows)} states)")


def fetch_state_terrain(raw_dir: Path) -> None:
    out_path = raw_dir / "state_terrain_2020_2024.csv"
    if out_path.exists():
        print(f"Exists, skipping: {out_path.name}")
        return
    rows = []
    # Use small coordinate neighborhood around each state centroid.
    offsets = [(-0.6, -0.6), (-0.6, 0.0), (-0.6, 0.6), (0.0, -0.6), (0.0, 0.0), (0.0, 0.6), (0.6, -0.6), (0.6, 0.0), (0.6, 0.6)]
    for state, (lat, lon) in STATE_COORDS.items():
        lats = [lat + dlat for dlat, _ in offsets]
        lons = [lon + dlon for _, dlon in offsets]
        lat_s = ",".join([f"{x:.4f}" for x in lats])
        lon_s = ",".join([f"{x:.4f}" for x in lons])
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_s}&longitude={lon_s}"
        resp = None
        for attempt in range(6):
            resp = requests.get(url, timeout=60)
            if resp.status_code != 429:
                break
            time.sleep(min(30, 2**attempt))
        if resp is None:
            continue
        resp.raise_for_status()
        payload = resp.json()
        elev = pd.to_numeric(pd.Series(payload.get("elevation", [])), errors="coerce")
        rows.append(
            {
                "State": state,
                "terrain_mean_elevation_m": float(elev.mean()),
                "terrain_ruggedness": float(elev.std()),
            }
        )
        time.sleep(0.25)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote: {out_path.name} ({len(rows)} states)")


def main():
    raw_dir = Path.cwd() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetch_primary_roads(raw_dir)
    fetch_state_weather(raw_dir)
    fetch_state_terrain(raw_dir)


if __name__ == "__main__":
    main()
