#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 6 — Weather Regime Engine
================================
Fits a Hidden Markov Model over historical meteorological features to
detect three canonical Southern California weather regimes:

  Regime 0 — MARINE_LAYER    (cool, humid, overcast; common Jun–Sep mornings)
  Regime 1 — DRY_CLEAR       (warm/hot, low humidity, clear skies)
  Regime 2 — SANTA_ANA       (very hot, very low humidity, strong offshore wind)

Outputs
-------
  artifacts_part6/
      regime_tape.parquet      — daily regime labels merged with source features
      regime_model.pkl         — serialized GaussianHMM
      regime_meta.json         — state statistics, transition matrix, schema version
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _project_dir() -> Path:
    env_root = os.environ.get("LATEMP_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = _project_dir()
SRC_DIR = PROJECT_DIR / "artifacts_part0"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part6"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.0.0"
N_REGIMES = 3
REGIME_NAMES = {0: "MARINE_LAYER", 1: "DRY_CLEAR", 2: "SANTA_ANA"}

# Features used for HMM fitting
HMM_FEATURES = [
    "temp_high_f",
    "temp_low_f",
    "humidity_mean_pct",
    "wind_speed_max_mph",
    "wind_dir_dominant_deg",
    "pressure_mean_hpa",
    "dew_point_mean_f",
    "cloud_cover_mean_pct",
    "precip_in",
]

# Minimum rows needed to fit the HMM
MIN_ROWS_FOR_HMM = 180


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_historical() -> pd.DataFrame:
    path = SRC_DIR / "historical_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"[Part 6] historical_daily.parquet not found at {path}. Run Part 0 first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature preparation for HMM
# ---------------------------------------------------------------------------
def _wind_cyclical(degrees: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Encode wind direction as sine/cosine components."""
    rad = np.deg2rad(degrees.fillna(0.0))
    return np.sin(rad), np.cos(rad)


def prepare_hmm_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and engineer features for the HMM.
    Returns (feature_df, feature_names) — rows with any NaN are dropped.
    """
    feat = df[["date"]].copy()

    # Core thermodynamic features
    for col in ["temp_high_f", "temp_low_f", "dew_point_mean_f"]:
        feat[col] = df.get(col, pd.Series(np.nan, index=df.index))

    # Diurnal range (key for Santa Ana detection)
    if "temp_high_f" in df.columns and "temp_low_f" in df.columns:
        feat["diurnal_range_f"] = df["temp_high_f"] - df["temp_low_f"]
    else:
        feat["diurnal_range_f"] = np.nan

    # Atmospheric moisture
    for col in ["humidity_mean_pct", "cloud_cover_mean_pct"]:
        feat[col] = df.get(col, pd.Series(np.nan, index=df.index))

    # Pressure
    feat["pressure_mean_hpa"] = df.get("pressure_mean_hpa", pd.Series(np.nan, index=df.index))

    # Wind speed
    feat["wind_speed_max_mph"] = df.get("wind_speed_max_mph", pd.Series(np.nan, index=df.index))

    # Wind direction cyclical encoding
    wind_dir = df.get("wind_dir_dominant_deg", pd.Series(0.0, index=df.index))
    feat["wind_sin"], feat["wind_cos"] = _wind_cyclical(wind_dir)

    # Precipitation (log-transformed to reduce skew)
    precip = df.get("precip_in", pd.Series(0.0, index=df.index)).fillna(0.0).clip(lower=0)
    feat["log_precip"] = np.log1p(precip)

    # Calendar seasonality (helps HMM learn seasonal regime patterns)
    day_of_year = pd.to_datetime(df["date"]).dt.dayofyear
    feat["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    feat["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    feature_cols = [
        "temp_high_f", "temp_low_f", "diurnal_range_f",
        "humidity_mean_pct", "cloud_cover_mean_pct",
        "dew_point_mean_f", "pressure_mean_hpa",
        "wind_speed_max_mph", "wind_sin", "wind_cos",
        "log_precip", "season_sin", "season_cos",
    ]

    # Only keep columns actually present
    feature_cols = [c for c in feature_cols if c in feat.columns]

    feat_sub = feat[["date"] + feature_cols].copy()

    # Drop rows missing more than 30% of features
    threshold = int(len(feature_cols) * 0.7)
    feat_sub = feat_sub.dropna(thresh=threshold + 1)  # +1 for date column

    # Fill remaining NaNs with column median
    for col in feature_cols:
        if col in feat_sub.columns:
            feat_sub[col] = feat_sub[col].fillna(feat_sub[col].median())

    return feat_sub, feature_cols


# ---------------------------------------------------------------------------
# HMM fitting
# ---------------------------------------------------------------------------
def fit_hmm(X: np.ndarray, n_regimes: int = N_REGIMES, n_iter: int = 200) -> object:
    """Fit a GaussianHMM with diagonal covariance."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn is required for Part 6. Install with: pip install hmmlearn")

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-4,
        random_state=42,
        verbose=False,
    )
    model.fit(X)
    return model


# ---------------------------------------------------------------------------
# Regime labeling heuristic
# ---------------------------------------------------------------------------
def _assign_regime_labels(
    model,
    scaler: StandardScaler,
    feature_cols: List[str],
    n_regimes: int = N_REGIMES,
) -> Dict[int, str]:
    """
    Map HMM state indices to semantic regime names using the decoded means.
    Heuristic:
      - Highest temp + lowest humidity + highest wind → SANTA_ANA
      - Lowest temp + highest humidity + highest cloud → MARINE_LAYER
      - Middle → DRY_CLEAR
    """
    # Inverse-transform means back to original scale
    means_scaled = model.means_                       # shape (n_states, n_features)
    means_orig = scaler.inverse_transform(means_scaled)
    means_df = pd.DataFrame(means_orig, columns=feature_cols)

    label_map: Dict[int, str] = {}
    remaining = list(range(n_regimes))

    # Santa Ana: max temp_high + max diurnal_range + min humidity
    if "temp_high_f" in means_df.columns and "humidity_mean_pct" in means_df.columns:
        santa_ana_score = (
            means_df.get("temp_high_f", 0) * 0.4
            + means_df.get("diurnal_range_f", 0) * 0.3
            - means_df.get("humidity_mean_pct", 0) * 0.3
        )
        santa_ana_idx = int(santa_ana_score.idxmax())
        if santa_ana_idx in remaining:
            label_map[santa_ana_idx] = "SANTA_ANA"
            remaining.remove(santa_ana_idx)

    # Marine layer: min temp_high + max humidity + max cloud_cover
    if remaining:
        marine_score = (
            -means_df.get("temp_high_f", 0) * 0.4
            + means_df.get("humidity_mean_pct", 0) * 0.4
            + means_df.get("cloud_cover_mean_pct", 0) * 0.2
        )
        marine_candidates = marine_score.iloc[remaining]
        marine_idx = int(marine_candidates.idxmax())
        label_map[marine_idx] = "MARINE_LAYER"
        remaining.remove(marine_idx)

    # Everything else → DRY_CLEAR
    for idx in remaining:
        label_map[idx] = "DRY_CLEAR"

    return label_map


# ---------------------------------------------------------------------------
# Prediction on new data
# ---------------------------------------------------------------------------
def predict_regimes(
    model,
    scaler: StandardScaler,
    X_scaled: np.ndarray,
    label_map: Dict[int, str],
) -> np.ndarray:
    """Decode HMM states and map to regime name strings."""
    states = model.predict(X_scaled)
    return np.array([label_map.get(int(s), "UNKNOWN") for s in states])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def save_model(model, scaler: StandardScaler, label_map: Dict[int, str],
               feature_cols: List[str]) -> None:
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_map": label_map,
        "feature_cols": feature_cols,
        "schema_version": SCHEMA_VERSION,
    }
    with open(ARTIFACTS_DIR / "regime_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("[Part 6] Saved regime_model.pkl")


def load_model() -> Optional[Dict]:
    path = ARTIFACTS_DIR / "regime_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 6] Project root: {PROJECT_DIR}")

    # Load historical data
    df = load_historical()
    print(f"[Part 6] Loaded {len(df)} historical rows")

    # Prepare features
    feat_df, feature_cols = prepare_hmm_features(df)
    print(f"[Part 6] HMM feature matrix: {len(feat_df)} rows × {len(feature_cols)} features")

    if len(feat_df) < MIN_ROWS_FOR_HMM:
        print(f"[Part 6] ERROR: Need at least {MIN_ROWS_FOR_HMM} rows to fit HMM. Got {len(feat_df)}.")
        return 1

    X_raw = feat_df[feature_cols].values.astype(np.float64)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Fit HMM
    print(f"[Part 6] Fitting GaussianHMM with {N_REGIMES} states...")
    model = fit_hmm(X_scaled, n_regimes=N_REGIMES)
    print(f"  Converged: {model.monitor_.converged}")
    print(f"  Log-likelihood: {model.monitor_.history[-1]:.2f}")

    # Label regimes semantically
    label_map = _assign_regime_labels(model, scaler, feature_cols)
    print(f"[Part 6] Regime label mapping: {label_map}")

    # Predict on all available data
    regimes_raw = model.predict(X_scaled)
    regime_names = np.array([label_map.get(int(s), "UNKNOWN") for s in regimes_raw])

    # Build regime tape
    tape = feat_df[["date"]].copy()
    tape["regime_state"] = regimes_raw
    tape["regime"] = regime_names

    # Compute posterior probabilities
    posteriors = model.predict_proba(X_scaled)
    for i in range(N_REGIMES):
        regime_label = label_map.get(i, f"state_{i}")
        tape[f"prob_{regime_label.lower()}"] = posteriors[:, i]

    tape_path = ARTIFACTS_DIR / "regime_tape.parquet"
    tape.to_parquet(tape_path, index=False)
    print(f"[Part 6] Saved regime_tape.parquet ({len(tape)} rows)")

    # Compute state statistics
    means_scaled = model.means_
    means_orig = scaler.inverse_transform(means_scaled)
    means_df = pd.DataFrame(means_orig, columns=feature_cols)
    state_stats = {}
    for state_idx, regime_label in label_map.items():
        state_stats[regime_label] = {
            "state_index": state_idx,
            "count": int((regimes_raw == state_idx).sum()),
            "pct": float((regimes_raw == state_idx).mean()),
            "means": {col: float(means_df.loc[state_idx, col]) for col in feature_cols},
        }

    # Transition matrix
    trans_mat = model.transmat_.tolist()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "fit_date": pd.Timestamp.now().isoformat(),
        "n_regimes": N_REGIMES,
        "n_rows_fit": len(feat_df),
        "feature_cols": feature_cols,
        "label_map": {str(k): v for k, v in label_map.items()},
        "state_stats": state_stats,
        "transition_matrix": trans_mat,
        "hmm_converged": bool(model.monitor_.converged),
        "hmm_log_likelihood": float(model.monitor_.history[-1]),
    }
    with open(ARTIFACTS_DIR / "regime_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("[Part 6] Saved regime_meta.json")

    # Save model bundle
    save_model(model, scaler, label_map, feature_cols)

    # Print summary
    print("\n=== REGIME DISTRIBUTION ===")
    for regime_label, stats in state_stats.items():
        print(f"  {regime_label}: {stats['count']} days ({stats['pct']:.1%})")
        key_stats = {k: f"{v:.1f}" for k, v in stats["means"].items()
                     if k in ["temp_high_f", "humidity_mean_pct", "wind_speed_max_mph"]}
        print(f"    Means: {key_stats}")

    print(f"\n[Part 6] ✅ Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

