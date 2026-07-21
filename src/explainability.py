"""
explainability.py — real SHAP attribution on the trained model: global
importance ranking, a beeswarm summary plot, and a single local (per
-prediction) waterfall explanation.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import OUTPUTS_DIR, SHAP_RANKING_CSV, RANDOM_SEED


def run_shap_analysis(model, X_test, sample_size=2000):
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    # Global summary (beeswarm)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, show=False, max_display=12)
    plt.title("SHAP Global Feature Importance \u2014 Rain Tomorrow Prediction", fontsize=11, weight="bold")
    plt.tight_layout()
    summary_path = os.path.join(OUTPUTS_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[explainability] Saved {summary_path}")

    # Ranked mean |SHAP| table
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    rank = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False)
    rank.to_csv(SHAP_RANKING_CSV, header=["mean_abs_shap"])
    print(f"[explainability] Top features:\n{rank.head(6)}")

    # One local explanation
    plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(shap_values[0], show=False, max_display=10)
    plt.title("SHAP Local Explanation \u2014 Single Test-Set Prediction", fontsize=10, weight="bold")
    plt.tight_layout()
    waterfall_path = os.path.join(OUTPUTS_DIR, "shap_waterfall.png")
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[explainability] Saved {waterfall_path}")

    return explainer, rank