"""Train Reframing A (classification) and Reframing B (regression).

Continental-US scope for training/evaluation (NEVI-aligned). Spatial holdout is
stratified: two states per Census region. State/region one-hot and raw
coordinates are excluded; state/region demographic aggregates are included.

Outputs:
  - data/processed/predictions_zcta.parquet
  - data/models/desert_classifier.pkl
  - data/models/desert_regressor.pkl
  - data/models/model_metrics.json
  - data/models/spatial_holdout_config.json
"""

from __future__ import annotations

from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
    XGBOOST_IMPORT_ERROR = None
except Exception as ex:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = str(ex)


SEEDS_FOR_SPATIAL_METRICS = [0, 7, 42, 100, 2024]
PRIMARY_SEED = 42
REGIONS = ["West", "Midwest", "South", "Northeast"]
HOLDOUT_PER_REGION = 2
CLASSIFIER_THRESHOLD = 0.5
HOLDOUT_CONFIGS = [2, 3]
PRIMARY_HOLDOUT_PER_REGION = 3
N_SEEDS_PER_CONFIG = 5
MIN_TEST_DESERTS = 100
MAX_ATTEMPTS_PER_CONFIG = 200


def stratified_spatial_split(
    df: pd.DataFrame, seed: int = 42, holdout_per_region: int = 2
) -> set[str]:
    """Hold out up to `holdout_per_region` states from each Census region."""
    rng = np.random.default_rng(seed)
    test_states: set[str] = set()
    for region in REGIONS:
        region_states = sorted(df[df["region"] == region]["State"].dropna().unique())
        n_hold = min(holdout_per_region, len(region_states))
        if n_hold < 1:
            continue
        chosen = rng.choice(region_states, size=n_hold, replace=False)
        test_states.update(chosen.tolist())
    return test_states


def _precision_at_recall(y_true: pd.Series, y_score: np.ndarray, target_recall: float = 0.8) -> float:
    """Best precision among operating points with recall >= target_recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = np.where(recall >= target_recall)[0]
    if len(valid) == 0:
        return float("nan")
    return float(np.max(precision[valid]))


def add_state_region_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Demographic aggregates by state/region (continuous proxies, not identity one-hot)."""
    state_stats = (
        df.groupby("State", dropna=False)
        .agg(
            state_median_density=("population_density", "median"),
            state_median_income=("median_household_income", "median"),
            state_pct_rural=("rurality_flag", "mean"),
            state_total_pop=("total_population", "sum"),
        )
        .reset_index()
    )
    region_stats = (
        df.groupby("region", dropna=False)
        .agg(
            region_median_density=("population_density", "median"),
            region_median_income=("median_household_income", "median"),
            region_pct_rural=("rurality_flag", "mean"),
            region_total_pop=("total_population", "sum"),
        )
        .reset_index()
    )
    out = df.merge(state_stats, on="State", how="left").merge(region_stats, on="region", how="left")
    return out


def _optional_numeric_columns(df: pd.DataFrame) -> list[str]:
    optional = []
    for c in [
        "distance_to_nearest_interstate_miles",
        "terrain_ruggedness",
        "terrain_mean_elevation_m",
        "weather_avg_temp_c",
        "weather_avg_precip_mm_day",
        "weather_extreme_heat_days_year",
        "weather_heavy_precip_days_year",
    ]:
        if c in df.columns and df[c].notna().any():
            optional.append(c)
    return optional


def _numeric_feature_names(optional_numeric: list[str]) -> list[str]:
    base = [
        "total_population",
        "median_household_income",
        "area_km2",
        "population_density",
        "rurality_flag",
    ]
    aggregates = [
        "state_median_density",
        "state_median_income",
        "state_pct_rural",
        "state_total_pop",
        "region_median_density",
        "region_median_income",
        "region_pct_rural",
        "region_total_pop",
    ]
    return base + optional_numeric + aggregates


def _build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ]
    )


def _build_models(scale_pos_weight: float):
    """Build classifier/regressor pair using XGBoost or sklearn fallback."""
    if HAS_XGBOOST:
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=1.2,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=PRIMARY_SEED,
            n_jobs=4,
        )
        reg = XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=1.2,
            objective="reg:squarederror",
            random_state=PRIMARY_SEED,
            n_jobs=4,
        )
    else:
        clf = HistGradientBoostingClassifier(random_state=PRIMARY_SEED)
        reg = HistGradientBoostingRegressor(random_state=PRIMARY_SEED)
    return clf, reg


def _train_and_eval_spatial_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_states: set[str],
    *,
    models_random_state: int | None = None,
) -> tuple[Pipeline, Pipeline, dict, set[str]]:
    """Train on train-side states, evaluate on val + held-out spatial states."""
    spatial_test = df["State"].isin(test_states)
    train_df = df[~spatial_test].copy()
    test_df = df[spatial_test].copy()

    inner_train, inner_val = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["is_charging_desert"],
        random_state=PRIMARY_SEED,
    )

    y_train_cls = inner_train["is_charging_desert"].astype(int)
    y_val_cls = inner_val["is_charging_desert"].astype(int)
    y_test_cls = test_df["is_charging_desert"].astype(int)

    y_train_reg = inner_train["nearest_dcfc_miles"].astype(float)
    y_val_reg = inner_val["nearest_dcfc_miles"].astype(float)
    y_test_reg = test_df["nearest_dcfc_miles"].astype(float)

    neg = int((y_train_cls == 0).sum())
    pos = int((y_train_cls == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    preprocessor = _build_preprocessor(feature_cols)
    clf_model, reg_model = _build_models(scale_pos_weight)
    if models_random_state is not None:
        clf_model.set_params(random_state=models_random_state)
        reg_model.set_params(random_state=models_random_state)

    clf = Pipeline([("prep", preprocessor), ("model", clf_model)])
    reg = Pipeline([("prep", _build_preprocessor(feature_cols)), ("model", reg_model)])

    clf.fit(inner_train[feature_cols], y_train_cls)
    reg.fit(inner_train[feature_cols], y_train_reg)

    val_prob = clf.predict_proba(inner_val[feature_cols])[:, 1]
    test_prob = clf.predict_proba(test_df[feature_cols])[:, 1]
    val_reg_pred = reg.predict(inner_val[feature_cols])
    test_reg_pred = reg.predict(test_df[feature_cols])

    fold_metrics = {
        "validation_pr_auc": float(average_precision_score(y_val_cls, val_prob)),
        "validation_roc_auc": float(roc_auc_score(y_val_cls, val_prob)),
        "validation_precision_at_recall_0_8": float(_precision_at_recall(y_val_cls, val_prob, 0.8)),
        "spatial_test_pr_auc": float(average_precision_score(y_test_cls, test_prob)),
        "validation_mae": float(mean_absolute_error(y_val_reg, val_reg_pred)),
        "validation_rmse": float(mean_squared_error(y_val_reg, val_reg_pred) ** 0.5),
        "spatial_test_mae": float(mean_absolute_error(y_test_reg, test_reg_pred)),
    }
    test_pred = (test_prob >= CLASSIFIER_THRESHOLD).astype(int)
    tp = int(((y_test_cls == 1) & (test_pred == 1)).sum())
    fp = int(((y_test_cls == 0) & (test_pred == 1)).sum())
    fn = int(((y_test_cls == 1) & (test_pred == 0)).sum())
    precision_at_p50 = tp / (tp + fp) if (tp + fp) else 0.0
    recall_at_p50 = tp / (tp + fn) if (tp + fn) else 0.0
    fold_metrics["precision_at_p50"] = float(precision_at_p50)
    fold_metrics["recall_at_p50"] = float(recall_at_p50)
    return clf, reg, fold_metrics, test_states


def _print_top_gain_importance(clf: Pipeline, feature_labels: list[str], top_n: int = 20) -> None:
    """Print top gain importances for XGBoost classifier with readable names."""
    if not HAS_XGBOOST:
        return
    try:
        booster = clf.named_steps["model"].get_booster()
        scores = booster.get_score(importance_type="gain")
        labeled = []
        for k, v in scores.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                name = feature_labels[idx] if idx < len(feature_labels) else k
            else:
                name = k
            labeled.append((name, float(v)))
        labeled.sort(key=lambda x: x[1], reverse=True)
        print("Top feature gains (classification):")
        for name, gain in labeled[:top_n]:
            print(f"  {name}: {gain:.4f}")
    except Exception as ex:
        print("(Could not read feature importances:", ex, ")")


def _summarize_seed_results(seed_results: list[dict]) -> dict:
    """Aggregate per-seed metrics as mean ± std."""
    pr_aucs = np.array([r["pr_auc"] for r in seed_results], dtype=float)
    p50 = np.array([r["precision_at_p50"] for r in seed_results], dtype=float)
    r50 = np.array([r["recall_at_p50"] for r in seed_results], dtype=float)
    return {
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "precision_at_p50_mean": float(np.mean(p50)),
        "precision_at_p50_std": float(np.std(p50)),
        "recall_at_p50_mean": float(np.mean(r50)),
        "recall_at_p50_std": float(np.std(r50)),
    }


def _evaluate_split_config(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    holdout_per_region: int,
    n_seeds: int,
    min_test_deserts: int,
    max_attempts: int,
) -> dict:
    """Evaluate one holdout config over multiple valid seeds.

    For holdout_per_region==2 we keep all seeds for comparability, but add warning
    if positive counts are below the desert floor. For holdout_per_region>=3 we
    enforce the minimum test-desert floor.
    """
    seed_results: list[dict] = []
    attempts = 0
    seed = 0
    while len(seed_results) < n_seeds and attempts < max_attempts:
        test_states = stratified_spatial_split(df, seed=seed, holdout_per_region=holdout_per_region)
        test_mask = df["State"].isin(test_states)
        test_deserts = int(df.loc[test_mask, "is_charging_desert"].sum())
        keep_seed = (holdout_per_region == 2) or (test_deserts >= min_test_deserts)

        if keep_seed:
            _, _, fold_m, _ = _train_and_eval_spatial_fold(df, feature_cols, test_states, models_random_state=seed)
            seed_results.append(
                {
                    "seed": int(seed),
                    "test_states": sorted(test_states),
                    "test_deserts": int(test_deserts),
                    "test_rows": int(test_mask.sum()),
                    "pr_auc": float(fold_m["spatial_test_pr_auc"]),
                    "precision_at_p50": float(fold_m["precision_at_p50"]),
                    "recall_at_p50": float(fold_m["recall_at_p50"]),
                }
            )
            print(
                f"hpr={holdout_per_region} seed={seed}: deserts={test_deserts}, "
                f"pr_auc={fold_m['spatial_test_pr_auc']:.4f}, "
                f"p@0.5={fold_m['precision_at_p50']:.4f}, r@0.5={fold_m['recall_at_p50']:.4f}"
            )

        seed += 1
        attempts += 1

    if len(seed_results) < n_seeds:
        raise RuntimeError(
            f"Could not collect {n_seeds} seeds for holdout_per_region={holdout_per_region} "
            f"within {max_attempts} attempts."
        )

    summary = _summarize_seed_results(seed_results)
    config_result = {
        "holdout_per_region": int(holdout_per_region),
        "n_seeds": int(n_seeds),
        "min_test_deserts": int(min_test_deserts),
        "attempts_used": int(attempts),
        "seeds": seed_results,
        **summary,
    }
    low_desert = any(r["test_deserts"] < min_test_deserts for r in seed_results)
    if low_desert:
        config_result["warning"] = (
            f"Some seeds have <{min_test_deserts} test deserts; metrics are noisy for single-seed interpretation."
        )
    return config_result


def main():
    base_dir = Path.cwd()
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "zcta_modeling_features.parquet")
    df = df[df["State"].notna()].copy()
    if "is_continental_us" in df.columns:
        df = df[df["is_continental_us"] == 1].copy()
    else:
        raise ValueError("Run notebooks/04b_modeling_features.py to add is_continental_us.")

    df = add_state_region_aggregates(df)
    optional_numeric = _optional_numeric_columns(df)
    feature_cols = _numeric_feature_names(optional_numeric)

    config_results = []
    for hpr in HOLDOUT_CONFIGS:
        cfg_result = _evaluate_split_config(
            df,
            feature_cols,
            holdout_per_region=hpr,
            n_seeds=N_SEEDS_PER_CONFIG,
            min_test_deserts=MIN_TEST_DESERTS,
            max_attempts=MAX_ATTEMPTS_PER_CONFIG,
        )
        config_results.append(cfg_result)
        print(
            f"hpr={hpr}: spatial PR-AUC {cfg_result['pr_auc_mean']:.4f} ± {cfg_result['pr_auc_std']:.4f}, "
            f"precision@0.5 {cfg_result['precision_at_p50_mean']:.4f} ± {cfg_result['precision_at_p50_std']:.4f}, "
            f"recall@0.5 {cfg_result['recall_at_p50_mean']:.4f} ± {cfg_result['recall_at_p50_std']:.4f}"
        )

    primary_cfg = next(c for c in config_results if c["holdout_per_region"] == PRIMARY_HOLDOUT_PER_REGION)
    primary_seed_row = next((r for r in primary_cfg["seeds"] if r["seed"] == PRIMARY_SEED), primary_cfg["seeds"][0])
    primary_seed_used = int(primary_seed_row["seed"])
    primary_test_states = set(primary_seed_row["test_states"])
    if primary_seed_used != PRIMARY_SEED:
        print(
            f"WARNING: seed {PRIMARY_SEED} was not part of accepted primary seeds for "
            f"holdout_per_region={PRIMARY_HOLDOUT_PER_REGION}. Using seed {primary_seed_used}."
        )

    clf, reg, fold_metrics, _ = _train_and_eval_spatial_fold(
        df, feature_cols, primary_test_states, models_random_state=primary_seed_used
    )

    model_family = "XGBoost" if HAS_XGBOOST else "Sklearn fallback"
    print("Model family:", model_family)
    if not HAS_XGBOOST:
        print("WARNING: XGBoost is unavailable. Falling back to sklearn models.")
        if XGBOOST_IMPORT_ERROR is not None:
            print("WARNING detail:", XGBOOST_IMPORT_ERROR)
    print("(Primary seed", primary_seed_used, f", holdout_per_region={PRIMARY_HOLDOUT_PER_REGION}) Reframing A:")
    print("  Validation PR-AUC:", round(fold_metrics["validation_pr_auc"], 4))
    print("  Validation ROC-AUC:", round(fold_metrics["validation_roc_auc"], 4))
    print("  Validation Precision@Recall>=0.8:", round(fold_metrics["validation_precision_at_recall_0_8"], 4))
    print("  Spatial Test PR-AUC:", round(fold_metrics["spatial_test_pr_auc"], 4))
    print("Reframing B (regression):")
    print("  Validation MAE:", round(fold_metrics["validation_mae"], 4))
    print("  Validation RMSE:", round(fold_metrics["validation_rmse"], 4))
    print("  Spatial Test MAE:", round(fold_metrics["spatial_test_mae"], 4))

    _print_top_gain_importance(clf, feature_cols)

    df["predicted_desert_prob"] = clf.predict_proba(df[feature_cols])[:, 1]
    df["predicted_nearest_dcfc_miles"] = reg.predict(df[feature_cols])
    df["distance_residual_miles"] = df["nearest_dcfc_miles"] - df["predicted_nearest_dcfc_miles"]

    clf_flagged = df["predicted_desert_prob"] >= CLASSIFIER_THRESHOLD
    reg_flagged = df["predicted_nearest_dcfc_miles"] > 50
    agreement_rate_legacy = float((clf_flagged & reg_flagged).mean())
    n_clf = int(clf_flagged.sum())
    n_reg = int(reg_flagged.sum())
    agreement_reg_among_clf = (
        float((df.loc[clf_flagged, "predicted_nearest_dcfc_miles"] > 50).mean()) if n_clf else float("nan")
    )
    agreement_clf_among_reg = (
        float((df.loc[reg_flagged, "predicted_desert_prob"] >= CLASSIFIER_THRESHOLD).mean())
        if n_reg
        else float("nan")
    )
    print("A/B agreement (both flag same ZCTA, legacy):", round(agreement_rate_legacy, 4))
    print(
        f"Among classifier-flagged ZCTAs (n={n_clf}), regressor distance>50mi:",
        f"{agreement_reg_among_clf:.4f}" if n_clf else "n/a",
    )
    print(
        f"Among regressor-flagged ZCTAs (n={n_reg}), classifier p>=0.5:",
        f"{agreement_clf_among_reg:.4f}" if n_reg else "n/a",
    )

    preds_cols = [
        "ZIP_ZCTA",
        "State",
        "region",
        "is_charging_desert",
        "nearest_dcfc_miles",
        "predicted_desert_prob",
        "predicted_nearest_dcfc_miles",
        "distance_residual_miles",
        "total_population",
        "median_household_income",
        "final_lat",
        "final_lon",
    ]
    df[preds_cols].to_parquet(processed_dir / "predictions_zcta.parquet", index=False)

    with open(models_dir / "desert_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / "desert_regressor.pkl", "wb") as f:
        pickle.dump(reg, f)

    holdout_config = {
        "strategy": "stratified_per_region",
        "holdout_per_region": PRIMARY_HOLDOUT_PER_REGION,
        "primary_seed": primary_seed_used,
        "test_states": sorted(primary_test_states),
        "metric_seeds": [r["seed"] for r in primary_cfg["seeds"]],
        "spatial_test_pr_auc_per_seed": {str(r["seed"]): r["pr_auc"] for r in primary_cfg["seeds"]},
    }
    (models_dir / "spatial_holdout_config.json").write_text(json.dumps(holdout_config, indent=2))

    spatial_eval = {
        "primary": {
            "config": (
                f"holdout_per_region={PRIMARY_HOLDOUT_PER_REGION}, "
                f"n_seeds={N_SEEDS_PER_CONFIG}, min_test_deserts={MIN_TEST_DESERTS}"
            ),
            **_summarize_seed_results(primary_cfg["seeds"]),
        },
        "configurations": config_results,
    }
    (models_dir / "spatial_evaluation.json").write_text(json.dumps(spatial_eval, indent=2))

    metrics = {
        "model_family": model_family,
        "evaluation_scope": "continental_us_only",
        "classification": {
            "validation_pr_auc": fold_metrics["validation_pr_auc"],
            "validation_roc_auc": fold_metrics["validation_roc_auc"],
            "validation_precision_at_recall_0_8": fold_metrics["validation_precision_at_recall_0_8"],
            "spatial_test_pr_auc": fold_metrics["spatial_test_pr_auc"],
            "spatial_test_pr_auc_mean": primary_cfg["pr_auc_mean"],
            "spatial_test_pr_auc_std": primary_cfg["pr_auc_std"],
            "spatial_test_pr_auc_seeds": [r["seed"] for r in primary_cfg["seeds"]],
            "precision_at_p50_mean": primary_cfg["precision_at_p50_mean"],
            "precision_at_p50_std": primary_cfg["precision_at_p50_std"],
            "recall_at_p50_mean": primary_cfg["recall_at_p50_mean"],
            "recall_at_p50_std": primary_cfg["recall_at_p50_std"],
            "holdout_per_region_primary": PRIMARY_HOLDOUT_PER_REGION,
        },
        "regression": {
            "validation_mae": fold_metrics["validation_mae"],
            "validation_rmse": fold_metrics["validation_rmse"],
            "spatial_test_mae": fold_metrics["spatial_test_mae"],
        },
        "agreement_both_flag": agreement_rate_legacy,
        "agreement_regressor_conditioned_on_classifier": agreement_reg_among_clf if n_clf else None,
        "agreement_classifier_conditioned_on_regressor": agreement_clf_among_reg if n_reg else None,
        "agreement_rate": agreement_rate_legacy,
        "rows_modeled": int(len(df)),
    }
    (models_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Wrote predictions, spatial_holdout_config.json, spatial_evaluation.json, and model artifacts.")


if __name__ == "__main__":
    main()
