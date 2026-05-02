#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 1 — Meteorological Feature Builder
=========================================
Constructs the model-ready feature matrix from Part 0 observations and
Part 6 regime labels.

Feature Groups
--------------
  1. Lag features         — temp_high at t-1, t-2, t-3, t-5, t-7
  2. Rolling statistics   — mean, std, min, max over 3/7/14/30-day windows
  3. Calendar features    — day-of-year (cyclical), month, season, weekday
  4. Regime features      — one-hot regime + posterior probabilities from Part 6
  5. Atmospheric features — pressure, humidity, dew point, wind (lagged)
  6. Derived features     — diurnal range, heat index proxy, anomaly vs 30-day mean

Targets Created
---------------
  target_h1   — temp_high_f at t+1  (next day)
  target_h3   — temp_high_f at t+3  (3-day ahead)
  target_h5   — temp_high_f at t+5  (5-day ahead)

Artifacts Written
-----------------
  artifacts_part1/
      feature_matrix.parquet        — full feature + target matrix
      feature_meta.json             — column list, split dates, schema version
      train_val_test_split.json     — date boundaries for reproducibility
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
PART0_DIR = PROJECT_DIR / "artifacts_part0"
PART6_DIR = PROJECT_DIR / "artifacts_part6"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part1"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.0.0"

# Forecast horizons (trading days ahead)
HORIZONS = [1, 3, 5]
TARGET_COL = "temp_high_f"

# Split fractions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST is remainder

# Min non-null fraction to retain a feature column
MIN_NONULL_FRAC = 0.60


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_historical() -> pd.DataFrame:
    path = PART0_DIR / "historical_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"historical_daily.parquet not found. Run Part 0 first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def load_regime_tape() -> Optional[pd.DataFrame]:
    path = PART6_DIR / "regime_tape.parquet"
    if not path.exists():
        print("[Part 1] regime_tape.parquet not found — regime features will be omitted.")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        df[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(w, min_periods=max(1, w // 2)).std()
        df[f"{col}_roll{w}_min"] = df[col].shift(1).rolling(w, min_periods=max(1, w // 2)).min()
        df[f"{col}_roll{w}_max"] = df[col].shift(1).rolling(w, min_periods=max(1, w // 2)).max()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])

    # Day of year — cyclical encoding
    doy = dates.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Month — cyclical encoding
    month = dates.dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Raw month (useful for tree-based models)
    df["month"] = month.astype(int)
    df["day_of_year"] = doy.astype(int)
    df["year"] = dates.dt.year.astype(int)

    # Season (meteorological)
    df["season"] = ((month % 12) // 3).astype(int)  # 0=DJF, 1=MAM, 2=JJA, 3=SON
    season_dummies = pd.get_dummies(df["season"], prefix="season").astype(float)
    df = pd.concat([df, season_dummies], axis=1)

    return df


def add_atmospheric_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1, 2, and 3-day lags of key atmospheric variables."""
    atm_cols = [
        "pressure_mean_hpa", "humidity_mean_pct", "dew_point_mean_f",
        "wind_speed_max_mph", "wind_dir_dominant_deg", "cloud_cover_mean_pct",
        "precip_in", "temp_low_f", "feels_like_max_f",
    ]
    for col in atm_cols:
        if col not in df.columns:
            continue
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def add_wind_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode dominant wind direction as sin/cos at lag=1."""
    for lag in [1, 2]:
        col = f"wind_dir_dominant_deg_lag{lag}"
        if col in df.columns:
            rad = np.deg2rad(df[col].fillna(0.0))
            df[f"wind_sin_lag{lag}"] = np.sin(rad)
            df[f"wind_cos_lag{lag}"] = np.cos(rad)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically meaningful derived features."""

    # Diurnal temperature range (large range → drier / Santa Ana signal)
    if "temp_high_f" in df.columns and "temp_low_f" in df.columns:
        df["diurnal_range_f"] = df["temp_high_f"] - df["temp_low_f"]
        for lag in [1, 2, 3]:
            df[f"diurnal_range_lag{lag}"] = df["diurnal_range_f"].shift(lag)

    # Heat index proxy: temp_high × humidity
    if "temp_high_f_lag1" in df.columns and "humidity_mean_pct_lag1" in df.columns:
        df["heat_index_proxy_lag1"] = (
            df["temp_high_f_lag1"] * df["humidity_mean_pct_lag1"] / 100.0
        )

    # Dew point depression (temp_mean - dew_point) — low value → fog/marine layer
    if "temp_mean_f" in df.columns and "dew_point_mean_f" in df.columns:
        df["dew_depression_f"] = df["temp_mean_f"] - df["dew_point_mean_f"]
        df["dew_depression_lag1"] = df["dew_depression_f"].shift(1)
        df["dew_depression_lag2"] = df["dew_depression_f"].shift(2)

    # Santa Ana wind proxy: offshore (easterly) wind + low humidity
    if "wind_dir_dominant_deg_lag1" in df.columns and "humidity_mean_pct_lag1" in df.columns:
        # Offshore winds ~ 60–120 degrees (northeast to east-southeast)
        wind_dir = df["wind_dir_dominant_deg_lag1"].fillna(180.0)
        offshore_mask = ((wind_dir >= 30) & (wind_dir <= 150)).astype(float)
        df["santa_ana_proxy_lag1"] = offshore_mask * (100.0 - df["humidity_mean_pct_lag1"].fillna(50.0))

    # Pressure tendency (change over last 2 days)
    if "pressure_mean_hpa" in df.columns:
        df["pressure_tendency_1d"] = df["pressure_mean_hpa"].diff(1)
        df["pressure_tendency_2d"] = df["pressure_mean_hpa"].diff(2)
        df["pressure_tendency_lag1"] = df["pressure_tendency_1d"].shift(1)

    # 30-day climatological anomaly (departure from recent normal)
    if TARGET_COL in df.columns:
        clim_30 = df[TARGET_COL].shift(1).rolling(30, min_periods=10).mean()
        df["temp_anomaly_vs_30d"] = df[TARGET_COL].shift(1) - clim_30

    return df


def add_regime_features(df: pd.DataFrame, regime_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge lagged numeric regime features from Part 6.

    The raw regime label is intentionally not carried into the model matrix as a
    string/object column. Downstream scalers and tree models should only see
    numeric one-hot columns and posterior probabilities.
    """
    if regime_df is None:
        return df

    regime_sub = regime_df[["date", "regime"]].copy()

    # One-hot encode regimes as float so parquet reloads are stable across
    # pandas/pyarrow versions.
    regime_dummies = pd.get_dummies(regime_sub["regime"], prefix="regime").astype(float)
    regime_sub = pd.concat([regime_sub[["date"]], regime_dummies], axis=1)

    # Add posterior probabilities if available, coerced to numeric.
    prob_cols = [c for c in regime_df.columns if c.startswith("prob_")]
    if prob_cols:
        probs = regime_df[prob_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        regime_sub = regime_sub.join(probs)

    # Shift regime by 1 day (we know yesterday's regime at prediction time).
    regime_sub_lag = regime_sub.copy()
    regime_sub_lag["date"] = regime_sub_lag["date"] + pd.Timedelta(days=1)
    rename_map = {c: f"{c}_lag1" for c in regime_sub_lag.columns if c != "date"}
    regime_sub_lag = regime_sub_lag.rename(columns=rename_map)

    df = df.merge(regime_sub_lag, on="date", how="left")
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward-shifted target values for each horizon."""
    for h in HORIZONS:
        df[f"target_h{h}"] = df[TARGET_COL].shift(-h)
    return df


# ---------------------------------------------------------------------------
# Full feature pipeline
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df_hist: pd.DataFrame,
    df_regime: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df_hist.copy().sort_values("date").reset_index(drop=True)

    print("[Part 1] Building lag features for temp_high_f...")
    df = add_lag_features(df, TARGET_COL, lags=[1, 2, 3, 5, 7, 14])

    print("[Part 1] Building rolling statistics...")
    df = add_rolling_features(df, TARGET_COL, windows=[3, 7, 14, 30])

    print("[Part 1] Adding calendar features...")
    df = add_calendar_features(df)

    print("[Part 1] Adding atmospheric variable lags...")
    df = add_atmospheric_lags(df)
    df = add_wind_cyclical(df)

    print("[Part 1] Adding derived physical features...")
    df = add_derived_features(df)

    print("[Part 1] Adding regime features from Part 6...")
    df = add_regime_features(df, df_regime)

    print("[Part 1] Adding target columns...")
    df = add_target_columns(df)

    # Keep the live tail even when future targets are not yet observed.
    # Downstream model training will explicitly filter to rows with all targets present.
    target_cols = [f"target_h{h}" for h in HORIZONS]

    # Drop feature columns with too many NaNs
    feature_cols = [c for c in df.columns if c not in ["date"] + target_cols]
    before = len(feature_cols)
    valid_feat_cols = [
        c for c in feature_cols
        if df[c].notna().mean() >= MIN_NONULL_FRAC
    ]
    print(f"[Part 1] Dropped {before - len(valid_feat_cols)} low-coverage feature columns")

    # Drop constant (zero-variance) feature columns.  These include zero-padded
    # physical probability columns from Part 6 for unvalidated regime labels,
    # and any other column that carries no information.
    nonconstant_feat_cols = []
    constant_dropped = []
    for c in valid_feat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) > 1:
            nonconstant_feat_cols.append(c)
        else:
            constant_dropped.append(c)
    if constant_dropped:
        print(f"[Part 1] Dropped {len(constant_dropped)} constant feature columns: {constant_dropped}")
    valid_feat_cols = nonconstant_feat_cols

    df = df[["date"] + valid_feat_cols + target_cols]

    n_labeled = int(df[target_cols].notna().all(axis=1).sum())
    print(f"[Part 1] Feature matrix: {len(df)} rows × {len(valid_feat_cols)} features")
    print(f"[Part 1] Fully labeled training rows: {n_labeled} | Live tail rows retained: {len(df) - n_labeled}")
    return df


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------
def compute_splits(df: pd.DataFrame) -> Dict[str, str]:
    """Compute train/validation/test boundaries using only fully labeled rows.

    The feature matrix intentionally retains the live tail where future targets are
    not yet observable. Splits must therefore be based on rows with all horizon
    labels present, while live prediction can still use the newest feature row.
    """
    target_cols = [f"target_h{h}" for h in HORIZONS]
    df_labeled = df.dropna(subset=target_cols).copy()
    n = len(df_labeled)
    if n < 100:
        raise ValueError(f"Not enough fully labeled rows to split: n={n}")

    train_end_idx = int(n * TRAIN_FRAC)
    val_end_idx = int(n * (TRAIN_FRAC + VAL_FRAC))

    dates = df_labeled["date"].sort_values().reset_index(drop=True)

    return {
        "train_start": str(dates.iloc[0].date()),
        "train_end": str(dates.iloc[train_end_idx - 1].date()),
        "val_start": str(dates.iloc[train_end_idx].date()),
        "val_end": str(dates.iloc[val_end_idx - 1].date()),
        "test_start": str(dates.iloc[val_end_idx].date()),
        "test_end": str(dates.iloc[-1].date()),
        "n_train": train_end_idx,
        "n_val": val_end_idx - train_end_idx,
        "n_test": n - val_end_idx,
        "n_labeled": n,
        "n_feature_rows": len(df),
        "live_feature_end": str(df["date"].max().date()),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(df: pd.DataFrame, splits: Dict[str, str], feature_cols: List[str]) -> None:
    path = ARTIFACTS_DIR / "feature_matrix.parquet"
    df.to_parquet(path, index=False)
    print(f"[Part 1] Saved feature_matrix.parquet ({len(df)} rows)")

    with open(ARTIFACTS_DIR / "train_val_test_split.json", "w") as f:
        json.dump(splits, f, indent=2)
    print("[Part 1] Saved train_val_test_split.json")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": pd.Timestamp.now().isoformat(),
        "n_rows": len(df),
        "feature_cols": feature_cols,
        "target_cols": [f"target_h{h}" for h in HORIZONS],
        "horizons": HORIZONS,
        "target_variable": TARGET_COL,
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "splits": splits,
    }
    with open(ARTIFACTS_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[Part 1] Saved feature_meta.json")


def load_feature_matrix() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError("feature_matrix.parquet not found. Run Part 1 first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 1] Project root: {PROJECT_DIR}")

    df_hist = load_historical()
    print(f"[Part 1] Loaded {len(df_hist)} historical rows")

    df_regime = load_regime_tape()
    if df_regime is not None:
        print(f"[Part 1] Loaded {len(df_regime)} regime rows")

    df = build_feature_matrix(df_hist, df_regime)

    target_cols = [f"target_h{h}" for h in HORIZONS]
    feature_cols = [c for c in df.columns if c not in ["date"] + target_cols]
    splits = compute_splits(df)

    print("\n=== SPLIT SUMMARY ===")
    for k, v in splits.items():
        print(f"  {k}: {v}")

    save_artifacts(df, splits, feature_cols)

    print(f"\n[Part 1] ✅ Complete. {len(feature_cols)} features ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


