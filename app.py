from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="EV Charging Desert Dashboard", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS = BASE_DIR / "data" / "models"

REGIONS = ["West", "Midwest", "South", "Northeast"]
DEFAULT_HOLDOUT_SEED = 42
HOLDOUT_PER_REGION = 2
THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def _stratified_spatial_split_states(df: pd.DataFrame, seed: int = 42, holdout_per_region: int = 2) -> set[str]:
    """Create a region-stratified state holdout set."""
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


def _data_version() -> tuple[int, int, int, int]:
    """Cache key based on processed parquet mtimes."""
    files = [
        PROCESSED / "zcta_modeling_features.parquet",
        PROCESSED / "predictions_zcta.parquet",
        PROCESSED / "recommended_sites_topN.parquet",
        PROCESSED / "zcta_installation_forecast_latest.parquet",
    ]
    return tuple(int(f.stat().st_mtime_ns) if f.exists() else 0 for f in files)


@st.cache_data
def load_data(_version: tuple[int, int, int, int]):
    zcta = pd.read_parquet(PROCESSED / "zcta_modeling_features.parquet")
    preds = pd.read_parquet(PROCESSED / "predictions_zcta.parquet")
    ranked = pd.read_parquet(PROCESSED / "recommended_sites_topN.parquet")
    forecast_path = PROCESSED / "zcta_installation_forecast_latest.parquet"
    if forecast_path.exists():
        forecast = pd.read_parquet(forecast_path)
    else:
        forecast = pd.DataFrame(columns=["ZIP_ZCTA", "forecast_new_dcfc_ports_next_12m"])
    return zcta, preds, ranked, forecast


def kpi_card(col, title, value):
    col.metric(title, value)


def style_us_map(fig):
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(243, 243, 243)",
    )
    return fig


def _spatial_test_states(df: pd.DataFrame) -> tuple[set[str], dict | None, bool]:
    """Held-out states from training run (spatial_holdout_config.json), with safe fallback."""
    config_path = MODELS / "spatial_holdout_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        return set(cfg["test_states"]), cfg, False
    return _stratified_spatial_split_states(df, seed=DEFAULT_HOLDOUT_SEED, holdout_per_region=HOLDOUT_PER_REGION), None, True


def _continental_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer continental rows when is_continental_us is available."""
    if "is_continental_us" in df.columns:
        return df[df["is_continental_us"] == 1].copy()
    return df


def _top_sites_for_scenario(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Scenario filter resilient to int/string parquet typing."""
    return ranked[ranked["scenario_top_n"].astype(int) == int(top_n)].copy()


def _threshold_sensitivity_table(view: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in THRESHOLD_GRID:
        s = _compute_eval(view["is_charging_desert"], view["predicted_desert_prob"], t)
        rows.append({"threshold": t, "precision": s["precision"], "recall": s["recall"], "f1": s["f1"]})
    return pd.DataFrame(rows)


def _load_spatial_evaluation() -> dict:
    """Load multi-config spatial evaluation summary when available."""
    path = MODELS / "spatial_evaluation.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _seed_rows_to_frame(spatial_eval: dict) -> pd.DataFrame:
    rows = []
    for cfg in spatial_eval.get("configurations", []):
        hpr = cfg.get("holdout_per_region")
        for s in cfg.get("seeds", []):
            rows.append(
                {
                    "Config (states/region)": hpr,
                    "Seed": s.get("seed"),
                    "Test states": ", ".join(s.get("test_states", [])),
                    "Test deserts": s.get("test_deserts"),
                    "PR-AUC": s.get("pr_auc"),
                    "Precision@0.5": s.get("precision_at_p50"),
                    "Recall@0.5": s.get("recall_at_p50"),
                }
            )
    return pd.DataFrame(rows)


def _compute_eval(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob))
    n = int(len(y_true))
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
    }


def _split_summary(df: pd.DataFrame) -> dict:
    base = df[df["State"].notna()].copy()
    test_states, _cfg, _used_fallback = _spatial_test_states(base)
    spatial_test = base[base["State"].isin(test_states)].copy()
    train_side = base[~base["State"].isin(test_states)].copy()
    inner_train, inner_val = train_test_split(
        train_side,
        test_size=0.2,
        stratify=train_side["is_charging_desert"],
        random_state=42,
    )
    return {
        "test_states": sorted(test_states),
        "rows_full": len(base),
        "rows_train": len(inner_train),
        "rows_val": len(inner_val),
        "rows_test": len(spatial_test),
        "rows_train_states": int(train_side["State"].nunique()),
        "rows_test_states": len(test_states),
    }


def _apply_risk_threshold_slider(t: float) -> None:
    t = float(round(t / 0.05) * 0.05)
    st.session_state["risk_threshold_slider"] = float(min(0.90, max(0.10, t)))


def _recommend_threshold(
    y_true: pd.Series,
    y_prob: pd.Series,
    target_recall: float,
    min_threshold: float = 0.10,
    max_threshold: float = 0.90,
    step: float = 0.01,
) -> dict:
    candidates = []
    t = min_threshold
    while t <= max_threshold + 1e-9:
        metrics = _compute_eval(y_true, y_prob, float(round(t, 4)))
        candidates.append({"threshold": float(round(t, 4)), **metrics})
        t += step
    df = pd.DataFrame(candidates)
    feasible = df[df["recall"] >= target_recall].copy()
    if feasible.empty:
        # Fall back to the highest-recall point in range if target is unattainable.
        best = df.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0]
        return {"feasible": False, **best.to_dict()}
    best = feasible.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
    return {"feasible": True, **best.to_dict()}


def main():
    st.title("EV Charging Desert Policy Dashboard")
    st.caption("Classification + regression risk signals with Top-N siting recommendations")

    zcta, preds, ranked, forecast = load_data(_data_version())
    pred_cols = [
        "ZIP_ZCTA",
        "predicted_desert_prob",
        "predicted_nearest_dcfc_miles",
        "distance_residual_miles",
    ]
    merged = zcta.merge(preds[pred_cols], on="ZIP_ZCTA", how="left")
    merged_maps = _continental_subset(merged)

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "ZCTAs (continental scope)", f"{len(merged_maps):,}")
    kpi_card(c2, "Actual Deserts", f"{int(merged_maps['is_charging_desert'].sum()):,}")
    kpi_card(c3, "High Risk (p>=0.5)", f"{int((merged_maps['predicted_desert_prob'] >= 0.5).sum()):,}")
    kpi_card(c4, "Top Sites Scored", f"{ranked['candidate_id'].nunique():,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Current Desert Severity")
        fig1 = px.scatter_geo(
            merged_maps.dropna(subset=["final_lat", "final_lon"]),
            lat="final_lat",
            lon="final_lon",
            color="nearest_dcfc_miles",
            hover_name="ZIP_ZCTA",
            color_continuous_scale="Viridis",
        )
        fig1 = style_us_map(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.subheader("Predicted Desert Risk")
        fig2 = px.scatter_geo(
            merged_maps.dropna(subset=["final_lat", "final_lon"]),
            lat="final_lat",
            lon="final_lon",
            color="predicted_desert_prob",
            hover_name="ZIP_ZCTA",
            color_continuous_scale="Reds",
        )
        fig2 = style_us_map(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    if not forecast.empty:
        fcols = ["ZIP_ZCTA", "forecast_new_dcfc_ports_next_12m"]
        merged_maps = merged_maps.merge(forecast[fcols], on="ZIP_ZCTA", how="left")
    else:
        merged_maps["forecast_new_dcfc_ports_next_12m"] = np.nan

    map_left, map_right = st.columns(2)
    with map_left:
        st.subheader("Predicted Installations (Next 12 Months)")
        fdf = merged_maps.dropna(subset=["final_lat", "final_lon"]).copy()
        if fdf["forecast_new_dcfc_ports_next_12m"].notna().any():
            # Clip at p95 to avoid a tail outlier flattening the color scale.
            p95 = float(fdf["forecast_new_dcfc_ports_next_12m"].quantile(0.95))
            fdf["forecast_clipped"] = fdf["forecast_new_dcfc_ports_next_12m"].clip(upper=p95)
            figf = px.scatter_geo(
                fdf,
                lat="final_lat",
                lon="final_lon",
                color="forecast_clipped",
                hover_name="ZIP_ZCTA",
                color_continuous_scale="Blues",
            )
            figf = style_us_map(figf)
            st.plotly_chart(figf, use_container_width=True)
            st.caption(
                f"Color scale clipped at 95th percentile ({p95:.2f} ports) for readability. "
                "Scope is continental US only (AK/HI excluded from modeling/evaluation)."
            )
        else:
            st.info("Run `notebooks/10_installation_forecasting.py` to render installation forecast maps.")

    with map_right:
        st.subheader("Additionality Intersection (High Risk × Low Forecast)")
        adf = merged_maps.dropna(subset=["final_lat", "final_lon"]).copy()
        if adf["forecast_new_dcfc_ports_next_12m"].notna().any():
            low_cut = adf["forecast_new_dcfc_ports_next_12m"].quantile(0.25)
            adf["additionality_flag"] = (
                (adf["predicted_desert_prob"] >= 0.5)
                & (adf["forecast_new_dcfc_ports_next_12m"] <= low_cut)
            ).astype(int)
            add_view = adf[adf["additionality_flag"] == 1].copy()
            if add_view.empty:
                st.info("No ZCTAs satisfy high-risk and low-forecast filters at current thresholds.")
            else:
                figa = px.scatter_geo(
                    add_view,
                    lat="final_lat",
                    lon="final_lon",
                    color="predicted_desert_prob",
                    hover_name="ZIP_ZCTA",
                    hover_data=["forecast_new_dcfc_ports_next_12m"],
                    color_continuous_scale="OrRd",
                )
                figa = style_us_map(figa)
                st.plotly_chart(figa, use_container_width=True)
                st.caption(
                    f"Low-forecast threshold uses 25th percentile "
                    f"({float(low_cut):.2f} forecasted ports in next 12 months)."
                )
        else:
            st.info("Forecast data unavailable.")

    st.subheader("Top-N Recommended Sites")
    top_n = st.select_slider("Scenario", options=[100, 500, 1000], value=100)
    top_sites = _top_sites_for_scenario(ranked, top_n)
    fig3 = px.scatter_geo(
        top_sites,
        lat="lat",
        lon="lon",
        color="composite_score",
        hover_name="candidate_id",
        hover_data=["population_covered_25mi", "distance_reduction", "equity_weight", "weather_weight"],
        color_continuous_scale="Plasma",
    )
    fig3 = style_us_map(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Model Metrics")
    metrics_path = MODELS / "model_metrics.json"
    metrics_obj = {}
    if metrics_path.exists():
        metrics_obj = json.loads(metrics_path.read_text())
        st.json(metrics_obj)
    else:
        st.info("Run notebooks/06_xgboost_classifier.py to generate model_metrics.json")

    spatial_eval = _load_spatial_evaluation()

    st.subheader("Inference Panel")
    cls_block = metrics_obj.get("classification", {})
    val_pr = cls_block.get("validation_pr_auc")
    spat_pr = cls_block.get("spatial_test_pr_auc")
    spat_mean = cls_block.get("spatial_test_pr_auc_mean")
    spat_std = cls_block.get("spatial_test_pr_auc_std")
    if val_pr is not None and spat_pr is not None:
        st.info(
            f"This model achieves **{val_pr:.3f} validation PR-AUC** on held-in states but "
            f"**{spat_pr:.3f} spatial-test PR-AUC** on held-out states (continental US scope). "
            "The gap highlights how much desert patterns depend on geography-specific context beyond "
            "demographics. Treat full-dataset metrics as optimistic when judging new states."
        )
    if spat_mean is not None and spat_std is not None:
        st.caption(
            f"Spatial test PR-AUC across {len(cls_block.get('spatial_test_pr_auc_seeds', [])) or 5} "
            f"stratified resplits: **{float(spat_mean):.3f} ± {float(spat_std):.3f}**"
        )
    if spatial_eval:
        primary = spatial_eval.get("primary", {})
        pr_mean = primary.get("pr_auc_mean")
        pr_std = primary.get("pr_auc_std")
        seed_frame = _seed_rows_to_frame(spatial_eval)
        if pr_mean is not None and pr_std is not None and not seed_frame.empty:
            pr_min = float(seed_frame["PR-AUC"].min())
            pr_max = float(seed_frame["PR-AUC"].max())
            pr_median = float(seed_frame["PR-AUC"].median())
            st.success(
                f"**Spatial PR-AUC:** {float(pr_mean):.3f} (mean) / {pr_median:.3f} (median)\n\n"
                f"Std: {float(pr_std):.3f}  ·  Range: {pr_min:.3f} to {pr_max:.3f}\n\n"
                "5-seed evaluation, 3 states held out per region"
            )

        comp_rows = []
        for cfg in spatial_eval.get("configurations", []):
            deserts = [s.get("test_deserts", 0) for s in cfg.get("seeds", [])]
            avg_deserts = float(np.mean(deserts)) if deserts else float("nan")
            comp_rows.append(
                {
                    "Holdout config": f"{cfg.get('holdout_per_region')} per region",
                    "PR-AUC": f"{cfg.get('pr_auc_mean', float('nan')):.3f} ± {cfg.get('pr_auc_std', float('nan')):.3f}",
                    "Precision@0.5": (
                        f"{cfg.get('precision_at_p50_mean', float('nan')):.3f} ± "
                        f"{cfg.get('precision_at_p50_std', float('nan')):.3f}"
                    ),
                    "Recall@0.5": (
                        f"{cfg.get('recall_at_p50_mean', float('nan')):.3f} ± "
                        f"{cfg.get('recall_at_p50_std', float('nan')):.3f}"
                    ),
                    "Avg test deserts": f"{avg_deserts:.1f}",
                    "Note": cfg.get("warning", ""),
                }
            )
        if comp_rows:
            st.markdown("**Spatial Holdout Comparison (2 vs 3 states per region)**")
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        with st.expander("Per-seed details (audit table)"):
            if seed_frame.empty:
                st.info("No per-seed rows found in spatial_evaluation.json")
            else:
                st.dataframe(seed_frame, use_container_width=True)
                st.caption(
                    "Per-seed PR-AUC variability reflects regional heterogeneity in desert formation: "
                    "states differ in EV adoption history, utility regulation, and prior infrastructure "
                    "investment, factors not directly captured in demographics. Use the 5-seed mean ± std "
                    "as the policy headline metric."
                )

    eval_df = merged_maps[
        [
            "ZIP_ZCTA",
            "State",
            "region",
            "median_household_income",
            "is_charging_desert",
            "predicted_desert_prob",
        ]
    ].copy()
    eval_df = eval_df[eval_df["State"].notna()].copy()
    test_states, holdout_cfg, used_split_fallback = _spatial_test_states(eval_df)
    n_hold_states = len(test_states)
    scope_spatial = f"Spatial test ({n_hold_states} held-out states)"
    eval_df["eval_scope"] = np.where(eval_df["State"].isin(test_states), scope_spatial, "Train-side states")
    if used_split_fallback:
        st.warning(
            "WARNING: `data/models/spatial_holdout_config.json` was not found. "
            "Using an in-app fallback stratified split for evaluation scope. "
            "Rerun `notebooks/06_xgboost_classifier.py` to regenerate the canonical holdout config."
        )

    debug_full_scope = st.toggle("Show full-dataset scope (debug only)", value=False)
    scope_options = [scope_spatial]
    if debug_full_scope:
        scope_options.append("Full dataset (training-contaminated; do not use for decisions)")
    scope_choice = st.radio("Evaluation scope", options=scope_options, horizontal=True)
    if scope_choice == scope_spatial:
        view = eval_df[eval_df["eval_scope"] == scope_spatial].copy()
        st.info(f"Metrics on held-out states only: {', '.join(sorted(test_states))}.")
        if holdout_cfg:
            st.caption(
                f"Split from `data/models/spatial_holdout_config.json` "
                f"({holdout_cfg.get('strategy', 'unknown')}, seed={holdout_cfg.get('primary_seed', '?')})."
            )
    else:
        view = eval_df.copy()
        st.warning(
            "Full-dataset metrics mix held-in and held-out states and are **optimistic** about "
            "performance on new geographies — prefer spatial test scope for generalization."
        )
    st.warning(
        "Single-seed metrics shown below. For policy interpretation, use the 5-seed headline "
        "from spatial evaluation."
    )

    # Baseline context for interpretability.
    base_rows = []
    baseline_scope = eval_df[eval_df["eval_scope"] == scope_spatial].copy()
    if len(baseline_scope):
        yb = baseline_scope["is_charging_desert"].astype(int)
        prevalence = float(yb.mean())
        const_score = np.full(len(baseline_scope), prevalence, dtype=float)
        pr_const = float(average_precision_score(yb, const_score))
        west_score = (baseline_scope["region"] == "West").astype(int)
        pr_west = float(average_precision_score(yb, west_score))

        # Density-only baseline (train-side -> spatial test), mirrors single-feature benchmark.
        train_side_df = eval_df[eval_df["eval_scope"] != scope_spatial].copy()
        merged_for_baseline = merged_maps[["ZIP_ZCTA", "population_density"]].copy()
        train_side_df = train_side_df.merge(merged_for_baseline, on="ZIP_ZCTA", how="left")
        baseline_scope = baseline_scope.merge(merged_for_baseline, on="ZIP_ZCTA", how="left")
        tr = train_side_df[train_side_df["population_density"].notna()].copy()
        te = baseline_scope[baseline_scope["population_density"].notna()].copy()
        pr_density = float("nan")
        if len(tr) and len(te):
            imp = SimpleImputer(strategy="median")
            xtr = imp.fit_transform(tr[["population_density"]])
            xte = imp.transform(te[["population_density"]])
            ytr = tr["is_charging_desert"].astype(int)
            yte = te["is_charging_desert"].astype(int)
            lr = LogisticRegression(max_iter=2000)
            lr.fit(xtr, ytr)
            pr_density = float(average_precision_score(yte, lr.predict_proba(xte)[:, 1]))

        full_model_pr = float("nan")
        if spatial_eval.get("primary", {}).get("pr_auc_mean") is not None:
            full_model_pr = float(spatial_eval["primary"]["pr_auc_mean"])

        def _fmt(x: float) -> str:
            return f"{x:.3f}" if pd.notna(x) else "NA"

        base_rows = [
            {"Model": "Constant prevalence", "PR-AUC": _fmt(pr_const), "vs random": f"{(pr_const / prevalence):.1f}x"},
            {"Model": "West-only rule", "PR-AUC": _fmt(pr_west), "vs random": f"{(pr_west / prevalence):.1f}x"},
            {"Model": "Population density only", "PR-AUC": _fmt(pr_density), "vs random": f"{(pr_density / prevalence):.1f}x" if pd.notna(pr_density) else "NA"},
            {"Model": "Full XGBoost (ours, 5-seed mean)", "PR-AUC": _fmt(full_model_pr), "vs random": f"{(full_model_pr / prevalence):.1f}x" if pd.notna(full_model_pr) else "NA"},
        ]
        st.markdown("**Performance vs Baselines (Spatial Test)**")
        st.dataframe(pd.DataFrame(base_rows), use_container_width=True)

    if "risk_threshold_slider" not in st.session_state:
        st.session_state["risk_threshold_slider"] = 0.50
    threshold = st.slider(
        "Risk threshold",
        min_value=0.10,
        max_value=0.90,
        step=0.05,
        key="risk_threshold_slider",
    )

    st.markdown("**Automatic Threshold Recommendation (Spatial test)**")
    target_recall = st.slider(
        "Target recall for recommendation",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )
    spatial_view = eval_df[eval_df["eval_scope"] == scope_spatial].copy()
    reco = _recommend_threshold(
        spatial_view["is_charging_desert"],
        spatial_view["predicted_desert_prob"],
        target_recall=target_recall,
    )
    if reco["feasible"]:
        st.success(
            f"Recommended threshold: {reco['threshold']:.2f} "
            f"(spatial precision={reco['precision']:.3f}, recall={reco['recall']:.3f}, f1={reco['f1']:.3f})"
        )
    else:
        st.warning(
            "Target recall is not achievable in the 0.10-0.90 search range. "
            f"Best available threshold is {reco['threshold']:.2f} "
            f"(precision={reco['precision']:.3f}, recall={reco['recall']:.3f}, f1={reco['f1']:.3f})."
        )
    reco_thr = float(reco["threshold"])
    st.button(
        "Use recommended threshold",
        on_click=_apply_risk_threshold_slider,
        args=(reco_thr,),
        help="Snaps to the slider step (0.05) and updates the risk threshold.",
    )

    stats = _compute_eval(view["is_charging_desert"], view["predicted_desert_prob"], threshold)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_card(k1, "Precision", f"{stats['precision']:.3f}")
    kpi_card(k2, "Recall", f"{stats['recall']:.3f}")
    kpi_card(k3, "F1", f"{stats['f1']:.3f}")
    kpi_card(k4, "PR-AUC", f"{stats['pr_auc']:.3f}")
    kpi_card(k5, "Accuracy", f"{stats['accuracy']:.3f}")
    st.caption(
        "Accuracy is shown for completeness but is often **misleading** when deserts are rare (~2% prevalence); "
        "prioritize precision, recall, F1, and PR-AUC."
    )
    st.markdown(
        """
**Why these numbers can look low**

Charging deserts are rare, and this panel defaults to a geographically held-out spatial test (the hardest evaluation regime).
Precision/recall/F1 at threshold 0.50 reflect that difficulty. The headline metric is PR-AUC, which remains far above
the random baseline for rare-event classification. Thresholds are policy-tunable:
- lower threshold -> higher recall (catch more deserts) with more false positives
- higher threshold -> higher precision (fewer false alarms) with more misses
"""
    )

    cm_df = pd.DataFrame(
        {
            "Predicted Desert": [stats["tp"], stats["fp"]],
            "Predicted Adequate": [stats["fn"], stats["tn"]],
        },
        index=["Actual Desert", "Actual Adequate"],
    )
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Threshold Sensitivity**")
    st.dataframe(_threshold_sensitivity_table(view), use_container_width=True)

    region_df = view.copy()
    region_df["is_flagged"] = (region_df["predicted_desert_prob"] >= threshold).astype(int)
    region_rates = region_df.groupby("region", dropna=False).agg(
        zctas=("ZIP_ZCTA", "count"),
        actual_deserts=("is_charging_desert", "sum"),
        flagged=("is_flagged", "sum"),
    )
    region_rates["Actual desert rate"] = region_rates["actual_deserts"] / region_rates["zctas"]
    region_rates["Model flagged rate"] = region_rates["flagged"] / region_rates["zctas"]
    region_rates = (
        region_rates[["Actual desert rate", "Model flagged rate"]]
        .reset_index()
        .melt(id_vars="region", var_name="metric", value_name="rate")
    )
    fig_region = px.bar(
        region_rates,
        x="region",
        y="rate",
        color="metric",
        barmode="group",
        title="Regional comparison: actual deserts vs model-flagged",
    )
    st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("**Missed Deserts (False Negatives) Profile**")
    fn_df = view[(view["is_charging_desert"] == 1) & (view["predicted_desert_prob"] < threshold)].copy()
    actual_deserts_n = int((view["is_charging_desert"] == 1).sum())
    miss_rate = (len(fn_df) / actual_deserts_n) if actual_deserts_n else 0.0
    m1, m2 = st.columns(2)
    kpi_card(m1, "Missed deserts (FN)", f"{len(fn_df):,}")
    kpi_card(m2, "Miss rate among actual deserts", f"{miss_rate:.1%}")

    if len(fn_df) == 0:
        st.success("No missed deserts at this threshold in the selected scope.")
    else:
        fn_region = (
            fn_df.groupby("region", dropna=False)
            .size()
            .reset_index(name="missed_deserts")
            .sort_values("missed_deserts", ascending=False)
        )
        fig_fn_region = px.bar(
            fn_region,
            x="region",
            y="missed_deserts",
            title="Missed deserts by region",
        )
        st.plotly_chart(fig_fn_region, use_container_width=True)

        income_bins = [0, 40_000, 60_000, 80_000, 100_000, float("inf")]
        income_labels = ["<40k", "40-60k", "60-80k", "80-100k", "100k+"]
        fn_df["income_band"] = pd.cut(
            fn_df["median_household_income"],
            bins=income_bins,
            labels=income_labels,
            include_lowest=True,
        ).astype("object")
        fn_df["income_band"] = fn_df["income_band"].fillna("Unknown")
        income_order = income_labels + ["Unknown"]
        fn_income = (
            fn_df.groupby("income_band", dropna=False)
            .size()
            .reindex(income_order, fill_value=0)
            .reset_index(name="missed_deserts")
        )
        fig_fn_income = px.bar(
            fn_income,
            x="income_band",
            y="missed_deserts",
            title="Missed deserts by median income band",
        )
        st.plotly_chart(fig_fn_income, use_container_width=True)

    split_info = _split_summary(merged_maps)
    st.markdown("**Train / validation / spatial-test summary** (continental ZCTAs with known state)")
    split_df = pd.DataFrame(
        [
            {"split": f"Train ({split_info['rows_train_states']} states, stratified val 20%)", "rows": split_info["rows_train"]},
            {"split": "Validation (held-in states)", "rows": split_info["rows_val"]},
            {
                "split": f"Spatial test ({split_info['rows_test_states']} held-out states)",
                "rows": split_info["rows_test"],
            },
            {"split": "Total continental modeled rows", "rows": split_info["rows_full"]},
        ]
    )
    st.dataframe(split_df, use_container_width=True)
    st.caption("Held-out unseen test states: " + ", ".join(split_info["test_states"]))

    st.markdown("**US Regional Prioritization (Deserts + Installation Need)**")
    install_df = merged_maps.copy()
    install_df["is_flagged"] = (install_df["predicted_desert_prob"] >= threshold).astype(int)
    install_df["high_residual_need"] = (
        install_df["distance_residual_miles"] >= install_df["distance_residual_miles"].quantile(0.9)
    ).astype(int)
    install_df["priority_zcta"] = ((install_df["is_flagged"] == 1) & (install_df["high_residual_need"] == 1)).astype(int)

    # Map recommended sites back to parent ZCTA region so siting can be viewed by region.
    region_map = merged_maps[["ZIP_ZCTA", "region"]].drop_duplicates().copy()
    top_sites_region = top_sites.merge(
        region_map,
        left_on="parent_zcta",
        right_on="ZIP_ZCTA",
        how="left",
    )
    sites_by_region = top_sites_region.groupby("region", dropna=False).size().rename("recommended_sites").reset_index()

    regional_priority = install_df.groupby("region", dropna=False).agg(
        zctas=("ZIP_ZCTA", "count"),
        actual_deserts=("is_charging_desert", "sum"),
        model_flagged=("is_flagged", "sum"),
        priority_zctas=("priority_zcta", "sum"),
        avg_predicted_risk=("predicted_desert_prob", "mean"),
    ).reset_index()
    regional_priority["actual_desert_rate"] = regional_priority["actual_deserts"] / regional_priority["zctas"]
    regional_priority["flagged_rate"] = regional_priority["model_flagged"] / regional_priority["zctas"]
    regional_priority["priority_rate"] = regional_priority["priority_zctas"] / regional_priority["zctas"]
    regional_priority = regional_priority.merge(sites_by_region, on="region", how="left")
    regional_priority["recommended_sites"] = regional_priority["recommended_sites"].fillna(0).astype(int)
    regional_priority = regional_priority.sort_values(
        ["priority_zctas", "actual_deserts", "avg_predicted_risk"],
        ascending=False,
    )

    st.dataframe(
        regional_priority[
            [
                "region",
                "zctas",
                "actual_deserts",
                "actual_desert_rate",
                "model_flagged",
                "flagged_rate",
                "priority_zctas",
                "priority_rate",
                "recommended_sites",
                "avg_predicted_risk",
            ]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
