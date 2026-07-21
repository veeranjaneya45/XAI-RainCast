"""
app/streamlit_app.py — live dashboard on top of the trained artifacts in
models/ and outputs/. Run from the project root with:

    streamlit run app/streamlit_app.py
"""
import json
import os
import sys

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# Make `src` importable when launched as `streamlit run app/streamlit_app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import XGB_MODEL_PATH, ARTIFACTS_JSON, MC_WEIGHTS_PATH, METRICS_JSON, UNCERTAINTY_JSON, OUTPUTS_DIR
from src.train_model import load_model
from src.uncertainty import MCDropoutNet

st.set_page_config(page_title="XAI-RainCast Dashboard", page_icon="\U0001F327\uFE0F", layout="wide")


@st.cache_resource
def load_all():
    with open(ARTIFACTS_JSON) as f:
        art = json.load(f)
    model = load_model(XGB_MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    net = MCDropoutNet.load(MC_WEIGHTS_PATH)
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    with open(UNCERTAINTY_JSON) as f:
        unc_metrics = json.load(f)
    return art, model, explainer, net, metrics, unc_metrics


ART, MODEL, EXPLAINER, NET, METRICS, UNC_METRICS = load_all()
FEATURE_ORDER = ART["feature_order"]
CAT_COLS = ART["cat_cols"]
ENCODERS = ART["encoders"]
DEFAULTS = ART["defaults"]

# ------------------------------------------------------------------
# Sidebar form
# ------------------------------------------------------------------
st.sidebar.title("\U0001F327\uFE0F Station Readings")
st.sidebar.caption("Edit values, then click Predict below.")

user_input = {}
with st.sidebar.form("readings_form"):
    for feat in FEATURE_ORDER:
        if feat in CAT_COLS:
            options = sorted(ENCODERS[feat].keys())
            default_idx = options.index(DEFAULTS[feat]) if DEFAULTS[feat] in options else 0
            user_input[feat] = st.selectbox(feat, options, index=default_idx)
        else:
            user_input[feat] = st.number_input(feat, value=float(DEFAULTS[feat]))
    submitted = st.form_submit_button("\U0001F52E  Run Prediction", width="stretch")

st.sidebar.divider()
st.sidebar.caption(
    "Model: XGBoost, trained on real weatherAUS.csv data.\n\n"
    f"Test accuracy **{METRICS['accuracy']*100:.1f}%** "
    f"(baseline {METRICS['baseline_majority_accuracy']*100:.1f}%) \u00b7 "
    f"ROC-AUC **{METRICS['roc_auc']:.3f}**"
)


def encode_row(user_input):
    row = []
    for feat in FEATURE_ORDER:
        val = user_input[feat]
        row.append(float(ENCODERS[feat][val]) if feat in CAT_COLS else float(val))
    return row


# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.title("XAI-RainCast \u2014 Live Dashboard")
st.caption(
    "Real trained model \u00b7 real SHAP attribution \u00b7 real MC Dropout calibrated uncertainty. "
    "MVP built on public weatherAUS.csv data as a stand-in for the full IMD/ERA5/NEXRAD pipeline."
)

tab_predict, tab_performance = st.tabs(["\U0001F52E Live Prediction", "\U0001F4CA Model Performance"])

with tab_predict:
    if not submitted:
        st.info("Set station readings in the sidebar and click **Run Prediction** to see a live result.")
    else:
        row = encode_row(user_input)
        X = np.array([row], dtype=np.float64)

        proba = float(MODEL.predict_proba(X)[0, 1])
        label = "Rain" if proba >= 0.5 else "No Rain"
        mc_mean, mc_std, mc_samples = NET.predict_mc_single(row, T=50, seed=42)

        col1, col2, col3 = st.columns(3)
        col1.metric("XGBoost \u2014 P(Rain Tomorrow)", f"{proba*100:.1f}%", label)
        col2.metric("MC Dropout \u2014 Mean Estimate", f"{mc_mean*100:.1f}%",
                    f"\u00b1{mc_std*100:.1f}% (1 std, T=50)")
        col3.metric("Model Test ROC-AUC", f"{METRICS['roc_auc']:.3f}",
                    f"ECE {UNC_METRICS['mc_ece']:.3f}")

        st.divider()
        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("Why the model predicted this \u2014 live SHAP attribution")
            sv = EXPLAINER(X)
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            shap.plots.waterfall(sv[0], show=False, max_display=10)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with right:
            st.subheader("MC Dropout predictive distribution (T=50 stochastic passes)")
            fig2, ax2 = plt.subplots(figsize=(6, 4.3))
            ax2.hist(mc_samples, bins=20, color="#1f6f6f")
            ax2.axvline(mc_mean, color="#e8935c", linestyle="--", linewidth=2,
                        label=f"mean = {mc_mean:.3f}")
            ax2.set_xlabel("Predicted P(Rain)")
            ax2.set_ylabel("Count across 50 passes")
            ax2.legend()
            ax2.set_title("Live predictive uncertainty for this input")
            st.pyplot(fig2, width="stretch")
            plt.close(fig2)
            st.caption(
                "Wider spread = the network is less confident for readings like these."
            )

with tab_performance:
    st.subheader("Real held-out test-set performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{METRICS['accuracy']*100:.1f}%", f"baseline {METRICS['baseline_majority_accuracy']*100:.1f}%")
    m2.metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
    m3.metric("Recall (rain days)", f"{METRICS['recall']*100:.1f}%")
    m4.metric("ECE (calibration)", f"{UNC_METRICS['mc_ece']:.3f}", "target < 0.05")

    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(OUTPUTS_DIR, "eval_plots.png"), caption="Confusion matrix + ROC curve", width="stretch")
    with c2:
        st.image(os.path.join(OUTPUTS_DIR, "shap_summary.png"), caption="Global SHAP feature importance", width="stretch")

    st.image(os.path.join(OUTPUTS_DIR, "mc_dropout_plots.png"), caption="MC Dropout calibration", width="stretch")