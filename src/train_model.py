"""
train_model.py — trains the XGBoost rainfall classifier.
"""
import xgboost as xgb

from src.config import RANDOM_SEED, XGB_MODEL_PATH


def train_xgboost(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def save_model(model, path=XGB_MODEL_PATH):
    model.save_model(path)
    print(f"[train_model] Saved model to {path}")


def load_model(path=XGB_MODEL_PATH):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model