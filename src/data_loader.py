"""
data_loader.py — fetches the real public weatherAUS.csv dataset if it isn't
already present locally, and loads it into a DataFrame.
"""
import os
import urllib.request

import pandas as pd

from src.config import RAW_CSV, RAW_CSV_URL


def download_if_missing(dest=RAW_CSV, url=RAW_CSV_URL):
    if os.path.exists(dest):
        print(f"[data_loader] Found existing dataset at {dest}")
        return dest
    print(f"[data_loader] Downloading real dataset from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[data_loader] Saved to {dest}")
    return dest


def load_raw(path=RAW_CSV):
    download_if_missing(path)
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())