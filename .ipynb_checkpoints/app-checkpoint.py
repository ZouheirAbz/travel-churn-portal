import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# -----------------------------
# Load artifacts
# -----------------------------
results_df = pd.read_csv(
    "model_comparison.csv",
    encoding="latin1",
    sep=None,            # auto-detect delimiter
    engine="python",     # required for sep=None
    on_bad_lines="skip"  # skip malformed lines if any
)
threshold_df = pd.read_csv(
    "threshold_metrics.csv",
    encoding="latin1",
    sep=None,
    engine="python",
    on_bad_lines="skip"
)
meta = joblib.load("dashboard_meta.pkl")
model = joblib.load("gb_churn_pipeline.pkl")
curves = np.load("curves_data.npz")

# -----------------------------
# Header
# -----------------------------
st.title("Travel Customer Churn – Model Dashboard")
st.write(
    f"**Suggested model (by CV mean accuracy):** {meta['suggested_model_from_cv']}  \n"
    f"**Deployed model for prediction:** Gradient Boosting  \n"
    f"**ROC-AUC:** {meta['roc_auc']:.3f}  \n"
    f"**Recommended threshold (best F1):** {meta['best_threshold_by_f1']:.2f}"
)

tab1, tab2, tab3 = st.tabs(["Model Comparison", "Threshold Tuning", "Predict Churn"])

# -----------------------------
# Tab 1: Model comparison
# -----------------------------
with tab1:
    st.subheader("5-Fold Cross-Validation Accuracy")
    st.dataframe(results_df, use_container_width=True)

    st.bar_chart(results_df.set_index("Model")[["Mean Accuracy"]])

# -----------------------------
# Tab 2: Threshold tuning
# -----------------------------
with tab2:
    st.subheader("ROC Curve")
    roc_plot_df = pd.DataFrame({"FPR": curves["fpr"], "TPR": curves["tpr"]})
    st.line_chart(roc_plot_df.set_index("FPR"))

    st.subheader("Precision–Recall Curve")
    pr_plot_df = pd.DataFrame({"Recall": curves["pr_recall"], "Precision": curves["pr_precision"]})
    st.line_chart(pr_plot_df.set_index("Recall"))

    st.subheader("Threshold Metrics")
    st.dataframe(threshold_df.sort_values("F1", ascending=False), use_container_width=True)

    st.info(
        f"Best threshold by F1 = **{meta['best_threshold_by_f1']:.2f}** "
        f"(Precision={meta['best_threshold_precision']:.2f}, "
        f"Recall={meta['best_threshold_recall']:.2f}, "
        f"F1={meta['best_threshold_f1']:.2f})"
    )

# -----------------------------
# Tab 3: Prediction form
# -----------------------------
with tab3:
    st.subheader("Enter Traveller Data to Predict Churn Probability")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
        services_opted = st.number_input("Services Opted", min_value=1, max_value=20, value=2, step=1)

    with col2:
        income = st.selectbox("Annual Income Class", ["Low", "Mid", "High"])
        synced = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
        hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

    with col3:
        ff = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
        threshold = st.slider("Decision Threshold", 0.10, 0.90, float(meta["best_threshold_by_f1"]), 0.05)

    # Encoding helpers
    income_map = {"Low": 0, "Mid": 1, "High": 2}
    yn_map = {"No": 0, "Yes": 1}

    # one-hot rules for FrequentFlyer (baseline is "No")
    ff_yes = 1 if ff == "Yes" else 0
    ff_no_record = 1 if ff == "No Record" else 0

    row = pd.DataFrame([{
        "Age": int(age),
        "ServicesOpted": int(services_opted),
        "AnnualIncomeClass": income_map[income],
        "AccountSyncedToSocialMedia": yn_map[synced],
        "BookedHotelOrNot": yn_map[hotel],
        "FrequentFlyer_No Record": ff_no_record,
        "FrequentFlyer_Yes": ff_yes
    }])

    if st.button("Predict"):
        prob = model.predict_proba(row)[:, 1][0]
        pred = int(prob >= threshold)

        st.metric("Churn Probability", f"{prob:.3f}")
        st.write("**Prediction:**", "Churn" if pred == 1 else "No Churn")
        st.write("**Rule:**", f"Churn if probability ≥ {threshold:.2f}")

        if pred == 1:
            st.warning("High risk: consider retention action (offer, outreach, loyalty benefit).")
        else:
            st.success("Lower risk: standard engagement recommended.")

