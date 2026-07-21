"""
evaluate.py — honest evaluation on the held-out test set: real metrics,
confusion matrix, and ROC curve. No numbers here are estimated or projected.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve, brier_score_loss)

from src.config import METRICS_JSON, OUTPUTS_DIR
import os


def evaluate_classifier(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    majority_pred = np.zeros(len(y_test))
    baseline_acc = accuracy_score(y_test, majority_pred)

    metrics = {
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "baseline_majority_accuracy": float(baseline_acc),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
    }
    cm = confusion_matrix(y_test, pred)
    metrics["confusion_matrix"] = cm.tolist()

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Saved metrics to {METRICS_JSON}")
    print(json.dumps(metrics, indent=2))

    _plot_eval(y_test, proba, pred, cm, metrics)
    return metrics, proba


def _plot_eval(y_test, proba, pred, cm, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Test Set)", fontsize=11, weight="bold")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["No Rain", "Rain"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["No Rain", "Rain"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=12, weight="bold")

    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, color="#1a2744", linewidth=2, label=f"XGBoost (AUC={metrics['roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve (Test Set)", fontsize=11, weight="bold")
    axes[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "eval_plots.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[evaluate] Saved {out_path}")