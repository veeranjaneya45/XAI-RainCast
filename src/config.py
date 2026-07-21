"""
config.py — single source of truth for paths and shared constants.
Every other module imports from here so paths never drift out of sync.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

RAW_CSV = os.path.join(DATA_DIR, "weatherAUS.csv")
RAW_CSV_URL = "https://raw.githubusercontent.com/tvdboom/ATOM/master/examples/datasets/weatherAUS.csv"

# Columns dropped for high missingness (>35% NaN in the raw data)
HIGH_NA_COLS = ["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm"]
CAT_COLS = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
TARGET_COL = "RainTomorrow"

RANDOM_SEED = 42

# Artifact filenames (all live under MODELS_DIR / OUTPUTS_DIR)
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
ARTIFACTS_JSON = os.path.join(MODELS_DIR, "artifacts.json")
MC_WEIGHTS_PATH = os.path.join(MODELS_DIR, "mc_dropout_weights.npz")
METRICS_JSON = os.path.join(OUTPUTS_DIR, "metrics.json")
UNCERTAINTY_JSON = os.path.join(OUTPUTS_DIR, "uncertainty_metrics.json")
SHAP_RANKING_CSV = os.path.join(OUTPUTS_DIR, "shap_feature_ranking.csv")

for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)