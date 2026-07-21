"""
uncertainty.py — a small feedforward neural network with Monte Carlo Dropout,
implemented from scratch in NumPy (forward pass, backprop, and the Adam
optimizer are all hand-written here — no autodiff framework), so the
mechanism is fully transparent and matches the math derived in the paper
(Eqs. 12-13: predictive mean/variance from T stochastic forward passes).

Wrapped as a class so it can be trained once (train_pipeline) and reused for
live single-input inference (the Streamlit app) without duplicating logic.
"""
import json

import numpy as np

from src.config import MC_WEIGHTS_PATH, UNCERTAINTY_JSON, OUTPUTS_DIR
import os


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _relu(z):
    return np.maximum(0, z)


class MCDropoutNet:
    """2-hidden-layer MLP with inverted dropout kept active at inference
    (Monte Carlo Dropout, Gal & Ghahramani 2016)."""

    def __init__(self, n_in, h1=32, h2=16, drop_rate=0.3, seed=42):
        self.h1, self.h2, self.drop_rate = h1, h2, drop_rate
        self.rng = np.random.default_rng(seed)
        self.params = {
            "W1": self.rng.normal(0, np.sqrt(2 / n_in), (n_in, h1)), "b1": np.zeros(h1),
            "W2": self.rng.normal(0, np.sqrt(2 / h1), (h1, h2)), "b2": np.zeros(h2),
            "W3": self.rng.normal(0, np.sqrt(2 / h2), (h2, 1)), "b3": np.zeros(1),
        }
        self.mu_ = None
        self.sigma_ = None

    # ---------- forward / backward ----------
    def _forward(self, X, rng):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = _relu(z1)
        m1 = (rng.random(a1.shape) > self.drop_rate).astype(np.float64)
        a1d = a1 * m1 / (1 - self.drop_rate)

        z2 = a1d @ p["W2"] + p["b2"]
        a2 = _relu(z2)
        m2 = (rng.random(a2.shape) > self.drop_rate).astype(np.float64)
        a2d = a2 * m2 / (1 - self.drop_rate)

        z3 = a2d @ p["W3"] + p["b3"]
        out = _sigmoid(z3).ravel()
        cache = (X, z1, a1, m1, a1d, z2, a2, m2, a2d, z3)
        return out, cache

    def _backward(self, cache, y, out):
        p = self.params
        X, z1, a1, m1, a1d, z2, a2, m2, a2d, z3 = cache
        n = X.shape[0]
        dz3 = (out - y).reshape(-1, 1) / n
        dW3 = a2d.T @ dz3; db3 = dz3.sum(axis=0)
        da2d = dz3 @ p["W3"].T
        da2 = da2d * m2 / (1 - self.drop_rate)
        dz2 = da2 * (z2 > 0)
        dW2 = a1d.T @ dz2; db2 = dz2.sum(axis=0)
        da1d = dz2 @ p["W2"].T
        da1 = da1d * m1 / (1 - self.drop_rate)
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1; db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    @staticmethod
    def _bce(y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # ---------- training (hand-written Adam, Eqs. 19-20 in the paper) ----------
    def fit(self, X_train, y_train, epochs=40, batch_size=512, lr=0.01, verbose=True):
        self.mu_ = X_train.mean(axis=0)
        self.sigma_ = X_train.std(axis=0) + 1e-8
        X = (X_train - self.mu_) / self.sigma_
        y = y_train

        m_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        n = X.shape[0]
        history = []

        for epoch in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                bidx = idx[start:start + batch_size]
                xb, yb = X[bidx], y[bidx]
                out, cache = self._forward(xb, self.rng)
                grads = self._backward(cache, yb, out)
                t += 1
                for k in self.params:
                    m_state[k] = beta1 * m_state[k] + (1 - beta1) * grads[k]
                    v_state[k] = beta2 * v_state[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = m_state[k] / (1 - beta1 ** t)
                    v_hat = v_state[k] / (1 - beta2 ** t)
                    self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            train_pred, _ = self._forward(X, self.rng)
            loss = self._bce(y, train_pred)
            history.append(loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"[uncertainty] epoch {epoch}: train BCE = {loss:.4f}")
        return history

    # ---------- MC Dropout inference ----------
    def predict_mc(self, X, T=50, seed=None):
        """Returns (mean, std, all_samples) over T stochastic forward passes."""
        rng = np.random.default_rng(seed)
        X_std = (X - self.mu_) / self.sigma_
        preds = np.stack([self._forward(X_std, rng)[0] for _ in range(T)], axis=0)
        return preds.mean(axis=0), preds.std(axis=0), preds

    def predict_mc_single(self, row, T=50, seed=None):
        """Convenience for a single input row (1-D array)."""
        mean, std, samples = self.predict_mc(np.array([row], dtype=np.float64), T=T, seed=seed)
        return float(mean[0]), float(std[0]), samples[:, 0]

    # ---------- persistence ----------
    def save(self, path=MC_WEIGHTS_PATH):
        np.savez(path, W1=self.params["W1"], b1=self.params["b1"],
                  W2=self.params["W2"], b2=self.params["b2"],
                  W3=self.params["W3"], b3=self.params["b3"],
                  mu=self.mu_, sigma=self.sigma_, drop_rate=np.array(self.drop_rate))
        print(f"[uncertainty] Saved weights to {path}")

    @classmethod
    def load(cls, path=MC_WEIGHTS_PATH):
        npz = np.load(path)
        n_in = npz["W1"].shape[0]
        h1, h2 = npz["W1"].shape[1], npz["W2"].shape[1]
        net = cls(n_in, h1=h1, h2=h2, drop_rate=float(npz["drop_rate"]))
        net.params = {"W1": npz["W1"], "b1": npz["b1"], "W2": npz["W2"],
                      "b2": npz["b2"], "W3": npz["W3"], "b3": npz["b3"]}
        net.mu_, net.sigma_ = npz["mu"], npz["sigma"]
        return net


def expected_calibration_error(y_true, probs, n_bins=10):
    """Eq. 25 in the paper: bins predictions, compares empirical accuracy to
    stated confidence per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def evaluate_uncertainty(net: MCDropoutNet, X_test, y_test, T=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc_mean, mc_std, _ = net.predict_mc(X_test, T=T, seed=42)
    ece = expected_calibration_error(y_test, mc_mean, n_bins=10)
    acc = ((mc_mean >= 0.5).astype(int) == y_test).mean()

    results = {"mc_dropout_T": T, "mc_test_accuracy": float(acc), "mc_ece": float(ece),
               "mean_predictive_std": float(mc_std.mean())}
    with open(UNCERTAINTY_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[uncertainty] {json.dumps(results, indent=2)}")

    # Reliability diagram + predictive-std histogram
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.linspace(0, 1, 11)
    bin_acc, bin_conf = [], []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (mc_mean >= lo) & (mc_mean < hi) if i < 9 else (mc_mean >= lo) & (mc_mean <= hi)
        if mask.sum() > 0:
            bin_acc.append(y_test[mask].mean()); bin_conf.append(mc_mean[mask].mean())
        else:
            bin_acc.append(np.nan); bin_conf.append(np.nan)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].plot(bin_conf, bin_acc, "o-", color="#7a3b1e", label=f"MC Dropout (ECE={ece:.3f})")
    axes[0].set_xlabel("Mean Predicted Probability"); axes[0].set_ylabel("Observed Frequency")
    axes[0].set_title("Reliability Diagram (Test Set)", fontsize=10, weight="bold")
    axes[0].legend(fontsize=8)

    axes[1].hist(mc_std, bins=40, color="#1f6f6f")
    axes[1].set_title("Predictive Std. Distribution (T=%d)" % T, fontsize=10, weight="bold")
    axes[1].set_xlabel("Predictive Std. Dev."); axes[1].set_ylabel("Count")

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "mc_dropout_plots.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[uncertainty] Saved {out_path}")
    return results