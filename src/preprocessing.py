"""
preprocessing.py — cleaning, encoding, and train/val/test splitting.

Kept as a set of small, reusable functions (not a script) so both the
training pipeline and the live app can reuse the exact same encoding logic
and never drift out of sync with each other.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import HIGH_NA_COLS, CAT_COLS, TARGET_COL, RANDOM_SEED, ARTIFACTS_JSON


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-missingness columns, impute the rest. Returns a clean copy."""
    df = df.drop(columns=HIGH_NA_COLS).copy()
    df = df.dropna(subset=[TARGET_COL])

    num_cols = [c for c in df.columns if c not in CAT_COLS + [TARGET_COL]]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        df[c] = df[c].fillna(df[c].mode()[0])
    return df


def encode(df: pd.DataFrame):
    """Label-encode categoricals. Returns (encoded_df, encoders_dict, defaults_dict)."""
    df = df.copy()
    encoders = {}
    defaults = {}

    num_cols = [c for c in df.columns if c not in CAT_COLS + [TARGET_COL]]
    for c in num_cols:
        defaults[c] = float(df[c].median())

    for c in CAT_COLS:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = {cls: int(i) for i, cls in enumerate(le.classes_)}
        # store the *original string* default (mode), not yet encoded
        defaults[c] = sorted(encoders[c].keys())[0]

    return df, encoders, defaults


def split(df: pd.DataFrame):
    """Stratified 68/12/20 train/val/test split."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataset(raw_df: pd.DataFrame):
    """
    Full pipeline: clean -> encode -> split.
    Also saves encoders/defaults/feature_order to models/artifacts.json so the
    Streamlit app (and any other consumer) can encode new inputs identically.
    """
    df = clean(raw_df)
    df, encoders, defaults = encode(df)
    feature_order = [c for c in df.columns if c != TARGET_COL]

    artifacts = {
        "encoders": encoders,
        "defaults": defaults,
        "feature_order": feature_order,
        "cat_cols": CAT_COLS,
    }
    with open(ARTIFACTS_JSON, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"[preprocessing] Saved encoders/defaults to {ARTIFACTS_JSON}")

    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    print(f"[preprocessing] Split sizes -> train {len(X_train):,} / "
          f"val {len(X_val):,} / test {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, artifacts