"""Forecast future DCFC additions at ZCTA level.

This stage builds a ZCTA-month panel from station open dates and predicts
`new_dcfc_ports_next_12m` using historical lag features + static demographics.

Outputs:
  - data/processed/zcta_forecast_panel.parquet
  - data/processed/zcta_installation_forecast.parquet
  - data/models/installation_forecast_metrics.json
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_XGBOOST = False


def _to_zip5(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str).str.zfill(5)


def _build_monthly_events(stations: pd.DataFrame) -> pd.DataFrame:
    df = stations.copy()
    df["ZIP_ZCTA"] = _to_zip5(df["ZIP"])
    df = df[df["ZIP_ZCTA"].str.match(r"^\d{5}$", na=False)].copy()
    df["install_year"] = pd.to_numeric(df["install_year"], errors="coerce")
    df["install_month"] = pd.to_numeric(df["install_month"], errors="coerce")
    df = df[df["install_year"].notna() & df["install_month"].notna()].copy()
    df["install_year"] = df["install_year"].astype(int)
    df["install_month"] = df["install_month"].astype(int)
    df = df[(df["install_month"] >= 1) & (df["install_month"] <= 12)].copy()
    df["month_start"] = pd.to_datetime(
        {"year": df["install_year"], "month": df["install_month"], "day": 1}, errors="coerce"
    )
    df = df[df["month_start"].notna()].copy()
    df["new_dcfc_ports"] = pd.to_numeric(df["EV DC Fast Count"], errors="coerce").fillna(0.0)
    df["new_dcfc_stations"] = pd.to_numeric(df.get("is_dcfc", 0), errors="coerce").fillna(0.0)

    monthly = (
        df.groupby(["ZIP_ZCTA", "month_start"], as_index=False)
        .agg(
            new_dcfc_ports=("new_dcfc_ports", "sum"),
            new_dcfc_stations=("new_dcfc_stations", "sum"),
        )
        .sort_values(["ZIP_ZCTA", "month_start"])
    )
    return monthly


def _expand_panel(monthly: pd.DataFrame, zcta_features: pd.DataFrame) -> pd.DataFrame:
    zips = pd.Series(sorted(zcta_features["ZIP_ZCTA"].dropna().astype(str).unique()))
    m0 = monthly["month_start"].min()
    m1 = monthly["month_start"].max()
    months = pd.date_range(m0, m1, freq="MS")
    idx = pd.MultiIndex.from_product([zips, months], names=["ZIP_ZCTA", "month_start"]).to_frame(index=False)
    panel = idx.merge(monthly, on=["ZIP_ZCTA", "month_start"], how="left")
    panel["new_dcfc_ports"] = panel["new_dcfc_ports"].fillna(0.0)
    panel["new_dcfc_stations"] = panel["new_dcfc_stations"].fillna(0.0)
    return panel


def _add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ZIP_ZCTA", "month_start"]).copy()
    g = panel.groupby("ZIP_ZCTA", sort=False)
    panel["ports_lag_1m"] = g["new_dcfc_ports"].shift(1)
    panel["ports_lag_3m"] = g["new_dcfc_ports"].rolling(3).sum().shift(1).reset_index(level=0, drop=True)
    panel["ports_lag_6m"] = g["new_dcfc_ports"].rolling(6).sum().shift(1).reset_index(level=0, drop=True)
    panel["ports_lag_12m"] = g["new_dcfc_ports"].rolling(12).sum().shift(1).reset_index(level=0, drop=True)
    panel["ports_cum_to_date"] = g["new_dcfc_ports"].cumsum() - panel["new_dcfc_ports"]
    panel["month_num"] = panel["month_start"].dt.month.astype(int)
    panel["year_num"] = panel["month_start"].dt.year.astype(int)

    # 12-month forward target (sum of next 12 monthly additions).
    panel["new_dcfc_ports_next_12m"] = (
        g["new_dcfc_ports"].shift(-1).rolling(12, min_periods=12).sum().reset_index(level=0, drop=True)
    )
    # Friendly aliases for diagnostics/reporting.
    panel["year_month"] = panel["month_start"].dt.to_period("M").astype(str)
    panel["new_dcfc_ports_this_month"] = panel["new_dcfc_ports"]
    panel["new_dcfc_ports_last_12m"] = panel["ports_lag_12m"]
    return panel


def _fit_poisson(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict]:
    prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), features)])
    model = Pipeline([("prep", prep), ("model", PoissonRegressor(alpha=1e-4, max_iter=2000))])
    model.fit(train[features], train["new_dcfc_ports_next_12m"].clip(lower=0.0))
    pred = model.predict(test[features]).clip(min=0.0)
    return pred, {"model": model}


def _fit_boosted(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict]:
    prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), features)])
    if HAS_XGBOOST:
        reg = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    else:
        reg = HistGradientBoostingRegressor(random_state=42)
    model = Pipeline([("prep", prep), ("model", reg)])
    model.fit(train[features], train["new_dcfc_ports_next_12m"].clip(lower=0.0))
    pred = model.predict(test[features]).clip(min=0.0)
    return pred, {"model": model}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    y_true_bin = (y_true >= 1.0).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "threshold": float(threshold),
    }


def main() -> None:
    base = Path.cwd()
    processed = base / "data" / "processed"
    models = base / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)

    stations = pd.read_parquet(processed / "cleaned_stations.parquet")
    zcta = pd.read_parquet(processed / "zcta_modeling_features.parquet")
    zcta["ZIP_ZCTA"] = _to_zip5(zcta["ZIP_ZCTA"])
    if "is_continental_us" in zcta.columns:
        zcta = zcta[zcta["is_continental_us"] == 1].copy()

    monthly = _build_monthly_events(stations)
    panel = _expand_panel(monthly, zcta[["ZIP_ZCTA"]])
    panel = _add_temporal_features(panel)
    panel = panel.merge(
        zcta[
            [
                "ZIP_ZCTA",
                "State",
                "region",
                "total_population",
                "median_household_income",
                "population_density",
                "rurality_flag",
                "distance_to_nearest_interstate_miles",
                "terrain_ruggedness",
                "terrain_mean_elevation_m",
                "weather_avg_temp_c",
                "weather_avg_precip_mm_day",
            ]
        ],
        on="ZIP_ZCTA",
        how="left",
    )
    panel = panel[panel["new_dcfc_ports_next_12m"].notna()].copy()

    features = [
        "ports_lag_1m",
        "ports_lag_3m",
        "ports_lag_6m",
        "ports_lag_12m",
        "ports_cum_to_date",
        "month_num",
        "year_num",
        "total_population",
        "median_household_income",
        "population_density",
        "rurality_flag",
        "distance_to_nearest_interstate_miles",
        "terrain_ruggedness",
        "terrain_mean_elevation_m",
        "weather_avg_temp_c",
        "weather_avg_precip_mm_day",
    ]

    # Temporal split: last 12 months as test.
    cutoff = panel["month_start"].max() - pd.DateOffset(months=12)
    train = panel[panel["month_start"] <= cutoff].copy()
    test = panel[panel["month_start"] > cutoff].copy()
    panel["is_test_period"] = panel["month_start"] > cutoff
    panel.to_parquet(processed / "zcta_forecast_panel.parquet", index=False)

    y_test = test["new_dcfc_ports_next_12m"].to_numpy(dtype=float)
    pred_pois, _ = _fit_poisson(train, test, features)
    pred_boost, _ = _fit_boosted(train, test, features)

    m_pois = _metrics(y_test, pred_pois)
    m_boost = _metrics(y_test, pred_boost)
    m_pois.update(_binary_metrics(y_test, pred_pois, threshold=0.5))
    m_boost.update(_binary_metrics(y_test, pred_boost, threshold=0.5))
    m_pois["spearman_rho"] = float(pd.Series(y_test).corr(pd.Series(pred_pois), method="spearman"))
    m_boost["spearman_rho"] = float(pd.Series(y_test).corr(pd.Series(pred_boost), method="spearman"))

    # Additional baseline diagnostics.
    pred_zero = np.zeros_like(y_test, dtype=float)
    pred_mean = np.full_like(y_test, float(np.mean(y_test)), dtype=float)
    pred_persist = test["new_dcfc_ports_last_12m"].fillna(0.0).to_numpy(dtype=float)
    baseline_metrics = {
        "constant_zero": _metrics(y_test, pred_zero),
        "constant_mean": _metrics(y_test, pred_mean),
        "persistence_last12m": _metrics(y_test, pred_persist),
        "target_distribution": {
            "prevalence_ge_1": float((y_test >= 1.0).mean()),
            "mean": float(np.mean(y_test)),
            "median": float(np.median(y_test)),
            "p95": float(np.quantile(y_test, 0.95)),
        },
    }
    better = "boosted" if m_boost["rmse"] <= m_pois["rmse"] else "poisson"
    pred_final = pred_boost if better == "boosted" else pred_pois

    out = test[["ZIP_ZCTA", "month_start", "State", "region", "new_dcfc_ports_next_12m"]].copy()
    out["forecast_new_dcfc_ports_next_12m"] = pred_final
    out["model_choice"] = better
    out.to_parquet(processed / "zcta_installation_forecast.parquet", index=False)
    latest = (
        out.sort_values(["ZIP_ZCTA", "month_start"])
        .groupby("ZIP_ZCTA", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest.to_parquet(processed / "zcta_installation_forecast_latest.parquet", index=False)

    metrics = {
        "target": "new_dcfc_ports_next_12m",
        "model_family": {"poisson": "PoissonRegressor", "boosted": "XGBRegressor" if HAS_XGBOOST else "HistGBR"},
        "time_split": {"cutoff": str(cutoff.date()), "train_rows": int(len(train)), "test_rows": int(len(test))},
        "baselines": baseline_metrics,
        "poisson": m_pois,
        "boosted": m_boost,
        "selected_model": better,
    }
    (models / "installation_forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote forecast panel, predictions, latest forecast snapshot, and installation_forecast_metrics.json")
    print("Selected model:", better, "| poisson rmse:", round(m_pois["rmse"], 4), "| boosted rmse:", round(m_boost["rmse"], 4))


if __name__ == "__main__":
    main()

