# app.py
# ------------------------------------------------------------
# Travel Customer Churn Portal (Streamlit)
# - Travel-themed UI + interactive dashboard
# - Model comparison + threshold tuning + prediction form
# - Optional map-style visual + actions + download history
# - Resilient loading (handles missing/corrupted artifacts)
# ------------------------------------------------------------

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Travel Churn Portal",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    return joblib.load("gb_churn_pipeline.pkl")

@st.cache_resource
def load_meta():
    return joblib.load("dashboard_meta.pkl")

try:
    model = load_model()
    meta = load_meta()
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()
    
# ------------------------------------------------------------
# Travel-themed CSS (clean + modern)
# ------------------------------------------------------------
st.markdown(
    """
<style>
/* App background */
.stApp {
  background: linear-gradient(180deg, rgba(240,249,255,1) 0%, rgba(255,255,255,1) 55%, rgba(240,253,250,1) 100%);
}

/* Hero */
.hero {
  padding: 22px 26px;
  border-radius: 18px;
  background: linear-gradient(90deg, rgba(2,132,199,0.96) 0%, rgba(14,116,144,0.96) 45%, rgba(5,150,105,0.96) 100%);
  color: white;
  box-shadow: 0 12px 35px rgba(2, 6, 23, 0.18);
  margin: 8px 0 14px 0;
}
.hero h1 { margin: 0; font-size: 34px; line-height: 1.1; }
.hero p  { margin: 6px 0 0; opacity: 0.95; font-size: 14px; }

/* Cards */
.card {
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(2, 6, 23, 0.08);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 8px 24px rgba(2, 6, 23, 0.06);
}
.card-title { font-size: 12px; color: rgba(2, 6, 23, 0.55); margin-bottom: 8px; }
.card-value { font-size: 26px; font-weight: 800; color: rgba(2, 6, 23, 0.90); }

/* Section headers */
.section-title {
  font-size: 18px;
  font-weight: 800;
  color: rgba(2, 6, 23, 0.90);
  margin: 6px 0 4px 0;
}

/* Subtle caption */
.subtle { color: rgba(2, 6, 23, 0.60); font-size: 13px; }

/* Buttons */
.stButton button {
  border-radius: 12px !important;
  padding: 10px 14px !important;
  font-weight: 700 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV robustly:
    - tries utf-8 first (common if exported from pandas)
    - falls back to latin1/cp1252 (common Windows/Excel)
    - auto-detects delimiter
    - skips malformed lines
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            last_err = e
            continue
    raise last_err


def kpi_card(title: str, value: str) -> None:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def risk_label(prob: float) -> tuple[str, str]:
    """
    Returns (label, streamlit_status_type)
    """
    if prob < 0.50:
        return "Low churn risk ✅", "success"
    if prob < 0.80:
        return "Medium churn risk ⚠️", "warning"
    return "High churn risk ❗", "error"


def ensure_session_state():
    if "pred_history" not in st.session_state:
        st.session_state.pred_history = pd.DataFrame(
            columns=[
                "timestamp",
                "Age",
                "ServicesOpted",
                "AnnualIncomeClass",
                "AccountSyncedToSocialMedia",
                "BookedHotelOrNot",
                "FrequentFlyer",
                "threshold",
                "churn_probability",
                "prediction",
                "risk_band",
            ]
        )


ensure_session_state()

# ------------------------------------------------------------
# Load artifacts (robust + friendly messages)
# ------------------------------------------------------------
meta_defaults = {
    "suggested_model_from_cv": "Gradient Boosting",
    "roc_auc": 0.973,
    "best_threshold_by_f1": 0.30,
    "best_threshold_precision": 0.7636,
    "best_threshold_recall": 0.9333,
    "best_threshold_f1": 0.84,
}

# CSV artifacts
try:
    results_df = safe_read_csv("model_comparison.csv")
except Exception as e:
    results_df = pd.DataFrame(columns=["Model", "Mean Accuracy", "Std Accuracy"])
    st.warning(f"Could not read model_comparison.csv. The dashboard will still run. Error: {e}")

try:
    threshold_df = safe_read_csv("threshold_metrics.csv")
except Exception as e:
    threshold_df = pd.DataFrame(columns=["Threshold", "Precision", "Recall", "F1"])
    st.warning(f"Could not read threshold_metrics.csv. The dashboard will still run. Error: {e}")

# meta
try:
    meta = joblib.load("dashboard_meta.pkl")
except Exception as e:
    meta = meta_defaults
    st.warning(f"dashboard_meta.pkl could not be loaded (using defaults). Error: {e}")

# curves
try:
    curves = np.load("curves_data.npz")
except Exception as e:
    curves = None
    st.warning(f"curves_data.npz could not be loaded (curves disabled). Error: {e}")

# model
try:
    model = joblib.load("gb_churn_pipeline.pkl")
except Exception as e:
    model = None
    st.error(
        "❌ Could not load gb_churn_pipeline.pkl. "
        "Re-export the model from the SAME environment used to run Streamlit."
    )
    st.stop()

# ------------------------------------------------------------
# Hero header
# ------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>✈️ Travel Customer Churn Portal</h1>
  <p>Compare models • Review tuning • Predict churn probability for a traveller • Export results</p>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Top KPIs
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Suggested model (CV)", str(meta.get("suggested_model_from_cv", "Gradient Boosting")))
with c2:
    kpi_card("ROC-AUC", f"{float(meta.get('roc_auc', 0.0)):.3f}")
with c3:
    kpi_card("Best threshold (F1)", f"{float(meta.get('best_threshold_by_f1', 0.30)):.2f}")
with c4:
    kpi_card("F1 @ best threshold", f"{float(meta.get('best_threshold_f1', 0.0)):.2f}")

st.markdown(
    f"<p class='subtle'>Deployed model for prediction: <b>Gradient Boosting</b> • "
    f"Recommended threshold (best F1): <b>{float(meta.get('best_threshold_by_f1',0.30)):.2f}</b></p>",
    unsafe_allow_html=True,
)

st.divider()

# ------------------------------------------------------------
# Sidebar: traveller input (interactive)
# ------------------------------------------------------------
st.sidebar.title("🧳 Traveller Input")
st.sidebar.caption("Adjust values → click **Predict churn risk**.")

age = st.sidebar.slider("Age", 18, 80, 32)
services_opted = st.sidebar.slider("Services Opted", 1, 6, 2)

annual_income = st.sidebar.selectbox("Annual Income Class", ["Low", "Mid", "High"])
frequent_flyer = st.sidebar.selectbox("Frequent Flyer", ["Yes", "No", "No Record"])
synced = st.sidebar.selectbox("Account Synced To Social Media", ["Yes", "No"])
hotel = st.sidebar.selectbox("Booked Hotel Or Not", ["Yes", "No"])

st.sidebar.divider()
threshold_default = float(meta.get("best_threshold_by_f1", 0.30))
decision_threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, threshold_default, 0.01)

st.sidebar.divider()
st.sidebar.subheader("🎯 Business control")
risk_policy = st.sidebar.radio(
    "Optimize for…",
    ["Balanced (F1)", "Catch churners (Higher recall)", "Avoid false alarms (Higher precision)"],
    index=0,
)

if risk_policy == "Catch churners (Higher recall)":
    decision_threshold = min(decision_threshold, 0.30)
elif risk_policy == "Avoid false alarms (Higher precision)":
    decision_threshold = max(decision_threshold, 0.55)

st.sidebar.caption("Tip: Lower threshold → more churners caught (higher recall) but more false alarms.")

# ------------------------------------------------------------
# Encode input row (must match model training schema)
# ------------------------------------------------------------
income_map = {"Low": 0, "Mid": 1, "High": 2}
yn_map = {"No": 0, "Yes": 1}

ff_yes = 1 if frequent_flyer == "Yes" else 0
ff_no_record = 1 if frequent_flyer == "No Record" else 0

input_df = pd.DataFrame(
    [
        {
            "Age": int(age),
            "ServicesOpted": int(services_opted),
            "AnnualIncomeClass": income_map[annual_income],
            "AccountSyncedToSocialMedia": yn_map[synced],
            "BookedHotelOrNot": yn_map[hotel],
            "FrequentFlyer_No Record": int(ff_no_record),
            "FrequentFlyer_Yes": int(ff_yes),
        }
    ]
)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_pred, tab_models, tab_tuning, tab_explain = st.tabs(
    ["🔮 Predict", "📊 Model comparison", "🎚️ Threshold tuning", "🧭 Actions & Export"]
)

# ------------------------------------------------------------
# Tab: Predict
# ------------------------------------------------------------
with tab_pred:
    st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85])

    with left:
        st.write("**Traveller profile (input)**")
        st.dataframe(input_df, use_container_width=True)

        predict_btn = st.button("🚀 Predict churn risk", use_container_width=True)

        if predict_btn:
            prob = float(model.predict_proba(input_df)[:, 1][0])
            pred = int(prob >= decision_threshold)
            label, status = risk_label(prob)

            st.subheader("Result")
            st.write(f"**Churn probability:** `{prob:.2%}`")
            st.progress(min(max(prob, 0.0), 1.0))

            if status == "success":
                st.success(label)
            elif status == "warning":
                st.warning(label)
            else:
                st.error(label)

            st.caption(f"Decision rule: churn if probability ≥ {decision_threshold:.2f} → Predicted = {pred}")

            # Save to history
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "timestamp": ts,
                "Age": int(age),
                "ServicesOpted": int(services_opted),
                "AnnualIncomeClass": annual_income,
                "AccountSyncedToSocialMedia": synced,
                "BookedHotelOrNot": hotel,
                "FrequentFlyer": frequent_flyer,
                "threshold": float(decision_threshold),
                "churn_probability": prob,
                "prediction": "Churn" if pred == 1 else "No churn",
                "risk_band": label.replace(" ✅", "").replace(" ⚠️", "").replace(" ❗", ""),
            }
            st.session_state.pred_history = pd.concat(
                [st.session_state.pred_history, pd.DataFrame([row])], ignore_index=True
            )

    with right:
        st.markdown("<div class='section-title'>Route-style visual</div>", unsafe_allow_html=True)
        st.caption("A light, travel-themed visual indicator (not geographic data).")

        # Map-style placeholder using Streamlit's built-in map widget (no external libs).
        # This is purely cosmetic: a "route" line with two points.
        route_points = pd.DataFrame(
            {
                "lat": [25.2048, 48.8566],   # Dubai -> Paris (example)
                "lon": [55.2708, 2.3522],
            }
        )
        st.map(route_points, zoom=1)

        st.markdown("<div class='section-title'>Recommended actions</div>", unsafe_allow_html=True)
        st.markdown(
            """
- **High risk:** proactive retention offer, priority support, flexible change policy, loyalty incentive  
- **Medium risk:** targeted messaging + value bundle (upgrade / add-on)  
- **Low risk:** nurture + cross-sell (insurance, add-ons, premium support)
            """
        )
        st.info("If churners are a minority class, accuracy can look high even when recall is weak. Track PR/Recall too.")

# ------------------------------------------------------------
# Tab: Model comparison
# ------------------------------------------------------------
with tab_models:
    st.markdown("<div class='section-title'>Model comparison (Cross-validation)</div>", unsafe_allow_html=True)
    st.caption("Mean accuracy is helpful, but also consider recall/ROC-AUC for churn detection.")

    if len(results_df) == 0:
        st.warning("model_comparison.csv is missing or empty.")
    else:
        st.dataframe(results_df, use_container_width=True)

        # Safe bar chart (handles slight column name variations)
        acc_col = None
        for c in ["Mean Accuracy", "Mean_Accuracy", "mean_accuracy"]:
            if c in results_df.columns:
                acc_col = c
                break

        if acc_col:
            st.bar_chart(results_df.set_index("Model")[[acc_col]])
        else:
            st.info("Could not find a 'Mean Accuracy' column to plot.")

# ------------------------------------------------------------
# Tab: Threshold tuning
# ------------------------------------------------------------
with tab_tuning:
    st.markdown("<div class='section-title'>Threshold tuning</div>", unsafe_allow_html=True)
    st.caption("Use ROC/PR curves to choose thresholds aligned to business costs (false alarms vs missed churners).")

    if curves is None:
        st.warning("curves_data.npz not available. Curves are disabled.")
    else:
        cA, cB = st.columns(2)

        with cA:
            st.subheader("ROC curve")
            roc_plot_df = pd.DataFrame({"FPR": curves["fpr"], "TPR": curves["tpr"]})
            st.line_chart(roc_plot_df.set_index("FPR"))

        with cB:
            st.subheader("Precision–Recall curve")
            pr_plot_df = pd.DataFrame({"Recall": curves["pr_recall"], "Precision": curves["pr_precision"]})
            st.line_chart(pr_plot_df.set_index("Recall"))

    st.subheader("Threshold metrics")
    if len(threshold_df) == 0:
        st.warning("threshold_metrics.csv is missing or empty.")
    else:
        st.dataframe(threshold_df.sort_values("F1", ascending=False), use_container_width=True)

    st.info(
        f"Best threshold by F1 = **{float(meta.get('best_threshold_by_f1',0.30)):.2f}** "
        f"(Precision={float(meta.get('best_threshold_precision',0.0)):.2f}, "
        f"Recall={float(meta.get('best_threshold_recall',0.0)):.2f}, "
        f"F1={float(meta.get('best_threshold_f1',0.0)):.2f})"
    )

# ------------------------------------------------------------
# Tab: Actions & Export
# ------------------------------------------------------------
with tab_explain:
    st.markdown("<div class='section-title'>Prediction log & export</div>", unsafe_allow_html=True)
    st.caption("Your portal keeps a session history of predictions you run.")

    if st.session_state.pred_history.empty:
        st.info("No predictions yet. Go to **Predict** tab and click **Predict churn risk**.")
    else:
        st.dataframe(st.session_state.pred_history, use_container_width=True)

        csv_bytes = st.session_state.pred_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download prediction log (CSV)",
            data=csv_bytes,
            file_name=f"prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.markdown("<div class='section-title'>Explainability (lightweight)</div>", unsafe_allow_html=True)
    st.caption("A quick heuristic view (not SHAP). For full explainability, add SHAP later.")

    # Lightweight heuristics based on correlations you observed:
    st.markdown(
        """
**Observed directional signals (from your exploratory analysis):**
- Higher *AnnualIncomeClass* and being a *FrequentFlyer_Yes* were associated with **higher churn** in your correlation heatmap.
- *BookedHotelOrNot* showed a **negative** association with churn (booked hotel → slightly less churn).
- *Age* had a small negative association with churn (older → slightly less churn).
        
These are not causal; they are patterns in this dataset and are best validated with out-of-sample performance.
        """
    )

    st.divider()
    st.markdown("<div class='section-title'>Next enhancement (optional)</div>", unsafe_allow_html=True)
    st.markdown(
        """
- Add SHAP explanations (local + global)  
- Add a “batch scoring” uploader (CSV upload → churn probabilities)  
- Add authentication (if sharing with others)  
- Deploy on Streamlit Community Cloud (recommended) or a small VM  
        """
    )
