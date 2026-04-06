"""
preprocessing.py
----------------
Reusable preprocessing functions for the Spotify popularity prediction pipeline.
All fitting operations (scaler, target encoder) must be called on training data only,
then applied to test data to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and do basic type coercion."""
    df = pd.read_csv(path)
    # The dataset sometimes has an unnamed index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop duplicate track_ids (keep first)
    - Drop rows with null name or artists
    - Convert duration_ms -> duration_min
    - Binary encode 'explicit'
    """
    df = df.copy()

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first")
    print(f"  Dropped {before - len(df)} duplicate track_ids")

    # Drop nulls in key text columns
    df = df.dropna(subset=["track_name", "artists"])
    print(f"  Rows after null drop: {len(df)}")

    # Duration
    df["duration_min"] = df["duration_ms"] / 60_000
    df = df.drop(columns=["duration_ms"])

    # Explicit: True/False -> 1/0
    df["explicit"] = df["explicit"].astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------

def split(df: pd.DataFrame, target: str = "popularity",
          test_size: float = 0.2, random_state: int = 42):
    """
    Stratified 80/20 split on a 5-bin discretisation of the target,
    preserving the bimodal popularity distribution in both partitions.
    """
    bins = pd.cut(df[target], bins=5, labels=False)
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=bins
    )
    print(f"  Train: {len(train)}  |  Test: {len(test)}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Encoding
# ---------------------------------------------------------------------------

def encode(train: pd.DataFrame, test: pd.DataFrame, target: str = "popularity"):
    """
    Apply all categorical encodings.  All encoders are FIT on train only.

    Steps:
      - One-hot encode: key (12), mode (2), time_signature (5)  [drop_first=True]
      - Mean target encode: track_genre (114 classes)
      - Drop raw text columns not used as features
    """
    train = train.copy()
    test = test.copy()

    # --- One-hot encoding ---------------------------------------------------
    ohe_cols = ["key", "mode", "time_signature"]
    train = pd.get_dummies(train, columns=ohe_cols, drop_first=True)
    test = pd.get_dummies(test, columns=ohe_cols, drop_first=True)

    # Align columns: test may be missing a dummy column present in train
    train, test = train.align(test, join="left", axis=1, fill_value=0)

    # --- Mean target encoding for genre ------------------------------------
    genre_means = (
        train.groupby("track_genre")[target].mean().rename("genre_encoded")
    )
    global_mean = train[target].mean()  # fallback for unseen genres

    train["genre_encoded"] = train["track_genre"].map(genre_means)
    test["genre_encoded"] = test["track_genre"].map(genre_means).fillna(global_mean)

    # --- Drop columns not used as features ---------------------------------
    drop_cols = ["track_id", "track_name", "artists", "album_name", "track_genre"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    return train, test, genre_means


# ---------------------------------------------------------------------------
# 5. Feature / target split + scaling
# ---------------------------------------------------------------------------

def scale(train: pd.DataFrame, test: pd.DataFrame, target: str = "popularity"):
    """
    Separate X/y and apply StandardScaler (fit on train only).
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    y_train = train[target].values.astype(float)
    y_test = test[target].values.astype(float)

    X_train = train.drop(columns=[target])
    X_test = test.drop(columns=[target])

    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# 6. Full pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

def run_pipeline(raw_path: str, processed_dir: str = "data/processed"):
    """
    Run the full preprocessing pipeline end-to-end.
    Saves processed arrays and artifacts to processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading...")
    df = load_raw(raw_path)

    print("Cleaning...")
    df = clean(df)

    print("Splitting...")
    train, test = split(df)

    print("Encoding...")
    train_enc, test_enc, genre_means = encode(train, test)

    print("Scaling...")
    X_train, X_test, y_train, y_test, scaler, feature_names = scale(train_enc, test_enc)

    # Save artifacts
    np.save(f"{processed_dir}/X_train.npy", X_train)
    np.save(f"{processed_dir}/X_test.npy", X_test)
    np.save(f"{processed_dir}/y_train.npy", y_train)
    np.save(f"{processed_dir}/y_test.npy", y_test)
    joblib.dump(scaler, f"{processed_dir}/scaler.pkl")
    joblib.dump(feature_names, f"{processed_dir}/feature_names.pkl")
    genre_means.to_csv(f"{processed_dir}/genre_means.csv")

    print(f"\nDone. Saved to {processed_dir}/")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, feature_names