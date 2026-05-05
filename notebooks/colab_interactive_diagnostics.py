"""Colab-friendly interactive diagnostics for EV desert modeling.

Run this file in Colab cells, or import its functions to inspect intermediate
tables and model diagnostics quickly.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd


def load_tables(base_dir: str = "/content/ev-charging-analysis") -> dict[str, pd.DataFrame]:
    base = Path(base_dir)
    processed = base / "data" / "processed"
    tables = {
        "zcta_modeling_features": pd.read_parquet(processed / "zcta_modeling_features.parquet"),
        "predictions_zcta": pd.read_parquet(processed / "predictions_zcta.parquet"),
        "candidate_sites": pd.read_parquet(processed / "candidate_sites.parquet"),
        "recommended_sites_topN": pd.read_parquet(processed / "recommended_sites_topN.parquet"),
    }
    return tables


def show_basic_checks(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        print(f"\n{name}: rows={len(df):,}, cols={len(df.columns)}")
        print(df.head(3))

    ranked = tables["recommended_sites_topN"]
    top100 = ranked[ranked["scenario_top_n"].astype(int) == 100]
    print("\nTop-100 checks")
    print("rows:", len(top100))
    print("unique candidate_id:", top100["candidate_id"].nunique())
    print("unique lat/lon pairs:", top100[["lat", "lon"]].drop_duplicates().shape[0])


def load_metrics(base_dir: str = "/content/ev-charging-analysis") -> dict:
    models = Path(base_dir) / "data" / "models"
    metrics = json.loads((models / "model_metrics.json").read_text())
    holdout = json.loads((models / "spatial_holdout_config.json").read_text())
    return {"metrics": metrics, "holdout": holdout}


def plot_spatial_seed_pr_auc(base_dir: str = "/content/ev-charging-analysis") -> None:
    holdout = load_metrics(base_dir)["holdout"]
    s = pd.Series(holdout["spatial_test_pr_auc_per_seed"]).astype(float).sort_index()
    ax = s.plot(kind="bar", title="Spatial test PR-AUC by seed (stratified holdout)")
    ax.set_xlabel("seed")
    ax.set_ylabel("PR-AUC")
    plt.tight_layout()
    plt.show()


def plot_xgboost_gain_importance(base_dir: str = "/content/ev-charging-analysis", top_n: int = 20) -> pd.DataFrame:
    models = Path(base_dir) / "data" / "models"
    clf = pickle.loads((models / "desert_classifier.pkl").read_bytes())
    booster = clf.named_steps["model"].get_booster()
    scores = booster.get_score(importance_type="gain")

    rows = []
    for k, v in scores.items():
        rows.append({"feature_key": k, "gain": float(v)})
    out = pd.DataFrame(rows).sort_values("gain", ascending=False).head(top_n)
    out = out.iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(out["feature_key"], out["gain"])
    plt.title("XGBoost feature gain importance (top)")
    plt.xlabel("gain")
    plt.tight_layout()
    plt.show()
    return out.sort_values("gain", ascending=False)


def compare_scope_metrics(base_dir: str = "/content/ev-charging-analysis") -> None:
    m = load_metrics(base_dir)["metrics"]["classification"]
    print("Validation PR-AUC:", round(float(m["validation_pr_auc"]), 4))
    print("Spatial test PR-AUC (primary split):", round(float(m["spatial_test_pr_auc"]), 4))
    print(
        "Spatial test PR-AUC mean±std (5 seeds):",
        f"{float(m['spatial_test_pr_auc_mean']):.4f} ± {float(m['spatial_test_pr_auc_std']):.4f}",
    )

