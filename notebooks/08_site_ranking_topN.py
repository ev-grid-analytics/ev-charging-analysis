"""Rank candidate charging sites using scalable one-pass scoring.

Composite score uses a minimum distance-reduction gate (serves real gaps), then a
weighted sum of normalized components so dense metros do not overwhelm rural
gap closure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import json


EARTH_RADIUS_MILES = 3958.8

# Tunable weights (document in README): gap closure + need remain primary.
# Market-gap is intentionally modest (tie-breaker / additionality cue).
W_GAP = 0.28
W_NEED = 0.33
W_POP = 0.12
W_EQUITY = 0.15
W_MARKET_GAP = 0.12
MIN_DISTANCE_REDUCTION = 25.0
COVERAGE_RADIUS_MILES = 25.0
TOP_N_SCENARIOS = [100, 500, 1000]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in miles."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MILES * c


def _normalize_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """In-place min-max normalize to [0,1] per column."""
    for c in cols:
        s = df[c].astype(float)
        lo = s.min()
        hi = s.max()
        if hi - lo < 1e-9:
            df[c + "_norm"] = 0.0
        else:
            df[c + "_norm"] = (s - lo) / (hi - lo)


def _load_zcta_scoring_frame(processed: Path) -> pd.DataFrame:
    """Load ZCTA frame used for scoring, honoring continental scope when present."""
    cols = [
        "ZIP_ZCTA",
        "final_lat",
        "final_lon",
        "total_population",
        "median_household_income",
        "nearest_dcfc_miles",
        "weather_extreme_heat_days_year",
        "weather_heavy_precip_days_year",
    ]
    zcta_all = pd.read_parquet(processed / "zcta_modeling_features.parquet")
    zcta = zcta_all[cols].copy()
    if "is_continental_us" in zcta_all.columns:
        zcta = zcta_all[zcta_all["is_continental_us"] == 1][cols].copy()
    return zcta[zcta["final_lat"].notna() & zcta["final_lon"].notna()]


def _load_forecast_map(processed: Path) -> pd.DataFrame:
    """Latest per-ZCTA installation forecast for market-gap scoring."""
    latest_path = processed / "zcta_installation_forecast_latest.parquet"
    full_path = processed / "zcta_installation_forecast.parquet"
    if latest_path.exists():
        fc = pd.read_parquet(latest_path)
    elif full_path.exists():
        fc = pd.read_parquet(full_path)
        fc = (
            fc.sort_values(["ZIP_ZCTA", "month_start"])
            .groupby("ZIP_ZCTA", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
    else:
        return pd.DataFrame(columns=["ZIP_ZCTA", "forecast_new_dcfc_ports_next_12m"])
    return fc[["ZIP_ZCTA", "forecast_new_dcfc_ports_next_12m"]].copy()


def score_candidates(candidates: pd.DataFrame, zcta: pd.DataFrame, forecast_map: pd.DataFrame) -> pd.DataFrame:
    """Score candidate sites by gap closure, need, population, equity, and market gap."""
    income_median = zcta["median_household_income"].median(skipna=True)
    z_lat = zcta["final_lat"].to_numpy()
    z_lon = zcta["final_lon"].to_numpy()
    z_pop = zcta["total_population"].fillna(0).to_numpy()
    z_nearest = zcta["nearest_dcfc_miles"].fillna(0).to_numpy()
    z_income = zcta["median_household_income"].to_numpy()
    z_zip = zcta["ZIP_ZCTA"].astype(str).to_numpy()
    fc = forecast_map.copy()
    if not fc.empty:
        fc["ZIP_ZCTA"] = fc["ZIP_ZCTA"].astype(str).str.zfill(5)
        fc = fc.drop_duplicates("ZIP_ZCTA", keep="last")
        zip_to_forecast = dict(
            zip(fc["ZIP_ZCTA"], pd.to_numeric(fc["forecast_new_dcfc_ports_next_12m"], errors="coerce").fillna(0.0))
        )
    else:
        zip_to_forecast = {}
    weather_risk = (
        0.5 * zcta.get("weather_extreme_heat_days_year", pd.Series(np.zeros(len(zcta)))).fillna(0).to_numpy()
        + 0.5 * zcta.get("weather_heavy_precip_days_year", pd.Series(np.zeros(len(zcta)))).fillna(0).to_numpy()
    )
    if weather_risk.max() > weather_risk.min():
        weather_risk = (weather_risk - weather_risk.min()) / (weather_risk.max() - weather_risk.min())
    else:
        weather_risk = np.zeros_like(weather_risk)

    rows = []
    for row in candidates.itertuples(index=False):
        d = haversine_miles(row.lat, row.lon, z_lat, z_lon)
        within = d <= COVERAGE_RADIUS_MILES
        if not within.any():
            continue

        idx = np.where(within)[0]
        pop_cov = float(z_pop[idx].sum())
        distance_reduction = float(np.maximum(z_nearest[idx] - d[idx], 0.0).sum())
        mean_neighbor_dcfc_mi = float(np.mean(z_nearest[idx]))
        low_income_pop = float(z_pop[idx][np.nan_to_num(z_income[idx], nan=income_median) <= income_median].sum())
        equity_weight = (low_income_pop / pop_cov) if pop_cov > 0 else 0.0
        covered_zips = z_zip[idx]
        forecast_vals = np.array([zip_to_forecast.get(z, 0.0) for z in covered_zips], dtype=float)
        avg_forecast = float(np.mean(forecast_vals)) if len(forecast_vals) else 0.0
        market_gap = float(-avg_forecast)  # larger gap when market forecast is lower
        weather_weight = float(weather_risk[idx].mean()) if len(idx) else 0.0

        rows.append(
            {
                "candidate_id": row.candidate_id,
                "parent_zcta": row.parent_zcta,
                "lat": row.lat,
                "lon": row.lon,
                "population_covered_25mi": pop_cov,
                "distance_reduction": distance_reduction,
                "mean_neighbor_dcfc_mi": mean_neighbor_dcfc_mi,
                "equity_weight": equity_weight,
                "avg_forecast_installations_12m": avg_forecast,
                "market_gap_raw": market_gap,
                "weather_weight": weather_weight,
                "log_population_covered": float(np.log1p(pop_cov)),
                "covered_zctas": int(len(idx)),
            }
        )

    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored

    scored = scored[scored["distance_reduction"] >= MIN_DISTANCE_REDUCTION].copy()
    if scored.empty:
        return scored
    scored["distance_reduction_per_covered_zcta"] = (
        scored["distance_reduction"] / scored["covered_zctas"].clip(lower=1)
    )

    _normalize_cols(
        scored,
        [
            "log_population_covered",
            "distance_reduction",
            "mean_neighbor_dcfc_mi",
            "equity_weight",
        ],
    )
    # Region-relative percentile rank for market additionality.
    # Higher percentile => lower expected market installations for peer region.
    region_map = candidates[["candidate_id", "region"]].drop_duplicates()
    scored = scored.merge(region_map, on="candidate_id", how="left")
    scored["market_gap_pct_region"] = scored.groupby("region", dropna=False)["market_gap_raw"].rank(
        method="average", pct=True
    )
    scored["market_gap_pct_region"] = scored["market_gap_pct_region"].fillna(0.5)
    scored["composite_without_market"] = (
        W_GAP * scored["distance_reduction_norm"]
        + W_NEED * scored["mean_neighbor_dcfc_mi_norm"]
        + W_POP * scored["log_population_covered_norm"]
        + W_EQUITY * scored["equity_weight_norm"]
    )
    scored["composite_score"] = (
        W_GAP * scored["distance_reduction_norm"]
        + W_NEED * scored["mean_neighbor_dcfc_mi_norm"]
        + W_POP * scored["log_population_covered_norm"]
        + W_EQUITY * scored["equity_weight_norm"]
        + W_MARKET_GAP * scored["market_gap_pct_region"]
    )
    return scored


def main():
    base_dir = Path.cwd()
    processed = base_dir / "data" / "processed"
    models = base_dir / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(processed / "candidate_sites.parquet")
    zcta = _load_zcta_scoring_frame(processed)
    forecast_map = _load_forecast_map(processed)

    scored = score_candidates(candidates, zcta, forecast_map)
    if scored.empty:
        raise ValueError(
            "No scored candidates after distance-reduction gate. "
            "Try lowering MIN_DISTANCE_REDUCTION or regenerate candidates."
        )

    scored = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
    scored.to_parquet(processed / "candidate_sites_scored.parquet", index=False)
    outputs = []
    for n in TOP_N_SCENARIOS:
        ranked = scored.head(n).copy()
        if ranked.empty:
            continue
        ranked["scenario_top_n"] = np.int32(n)
        ranked["selection_order"] = np.arange(1, len(ranked) + 1, dtype=np.int32)
        outputs.append(ranked)

    if not outputs:
        raise ValueError("No ranked sites generated. Check candidate and ZCTA inputs.")

    final = pd.concat(outputs, ignore_index=True)
    final.to_parquet(processed / "recommended_sites_topN.parquet", index=False)

    # Sensitivity: region mix under alternative policy weighting choices.
    c_region = candidates[["candidate_id", "region"]].drop_duplicates()

    def _region_counts_for(sort_col: str) -> dict:
        return (
            scored.sort_values(sort_col, ascending=False)
            .head(100)[["candidate_id"]]
            .merge(c_region, on="candidate_id", how="left")
            .groupby("region", dropna=False)
            .size()
            .to_dict()
        )

    top100_current = _region_counts_for("composite_score")
    top100_per_zcta_gap = (
        scored.assign(
            _tmp=(
                0.60 * scored["distance_reduction_per_covered_zcta"].rank(pct=True)
                + 0.10 * scored["log_population_covered_norm"]
                + 0.20 * scored["equity_weight_norm"]
                + 0.10 * scored["market_gap_pct_region"]
            )
        )
        .sort_values("_tmp", ascending=False)
        .head(100)[["candidate_id"]]
        .merge(c_region, on="candidate_id", how="left")
        .groupby("region", dropna=False)
        .size()
        .to_dict()
    )
    top100_gap_heavy = (
        scored.assign(
            _tmp=(
                0.60 * scored["distance_reduction_norm"]
                + 0.10 * scored["log_population_covered_norm"]
                + 0.20 * scored["equity_weight_norm"]
                + 0.10 * scored["market_gap_pct_region"]
            )
        )
        .sort_values("_tmp", ascending=False)
        .head(100)[["candidate_id"]]
        .merge(c_region, on="candidate_id", how="left")
        .groupby("region", dropna=False)
        .size()
        .to_dict()
    )
    top100_pop_heavy = (
        scored.assign(
            _tmp=(
                0.20 * scored["distance_reduction_norm"]
                + 0.50 * scored["log_population_covered_norm"]
                + 0.20 * scored["equity_weight_norm"]
                + 0.10 * scored["market_gap_pct_region"]
            )
        )
        .sort_values("_tmp", ascending=False)
        .head(100)[["candidate_id"]]
        .merge(c_region, on="candidate_id", how="left")
        .groupby("region", dropna=False)
        .size()
        .to_dict()
    )
    sensitivity = {
        "weights": {
            "W_GAP": W_GAP,
            "W_NEED": W_NEED,
            "W_POP": W_POP,
            "W_EQUITY": W_EQUITY,
            "W_MARKET_GAP": W_MARKET_GAP,
        },
        "top100_region_current": top100_current,
        "aggregate_gap_top100_region": top100_current,
        "per_zcta_gap_top100_region": top100_per_zcta_gap,
        "top100_region_gap_heavy": top100_gap_heavy,
        "top100_region_population_heavy": top100_pop_heavy,
    }
    (models / "site_ranking_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    print("Wrote ranked sites:", len(final), "(scenarios:", final["scenario_top_n"].unique().tolist(), ")")


if __name__ == "__main__":
    main()
