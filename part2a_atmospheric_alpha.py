#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2A — Atmospheric Alpha Engineering
=========================================
Computes advanced atmospheric features (alpha signals) that provide
predictive power beyond raw observations:

  1. Pressure gradient signals       — surface pressure tendency, anomaly
  2. Marine layer depth proxy        — dew point depression at morning hours
  3. Santa Ana wind index            — offshore wind component × dryness score
  4. ENSO teleconnection proxy       — Niño 3.4 SST anomaly index (monthly)
  5. Pacific SST anomaly proxy       — sea surface temperature state via PDO
  6. Heat wave persistence index     — streak length above threshold
  7. Temperature momentum signals    — rate-of-change, acceleration
  8. Seasonal climatological anomaly — departure from historical day-of-year normal

Note: ENSO/PDO indices are fetched from NOAA PSL if available, otherwise
a proxy is computed from local pressure and temperature patterns.

Artifacts Written
-----------------
  artifacts_part2a/
      alpha_features.parquet         — alpha feature matrix aligned to feature dates
      alpha_meta.json                — feature descriptions and schema version
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

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
PART0_DIR = PROJECT_DIR / "artifacts_part0"
PART1_DIR = PROJECT_DIR / "artifacts_part1"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part2a"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.1.0"

# NOAA ENSO Niño 3.4 monthly anomaly.
# The old source, nina34.data, is absolute SST and must not be thresholded as an anomaly.
# This URL is the anomaly series. The parser below still detects absolute-SST-shaped
# values and converts them to monthly anomalies as a defensive guard.
ENSO_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"

# Heat wave threshold
HEAT_WAVE_THRESHOLD_F = 95.0
MARINE_LAYER_THRESHOLD_F = 10.0  # dew point depression < 10°F → marine layer likely


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_historical() -> pd.DataFrame:
    path = PART0_DIR / "historical_daily.parquet"
    if not path.exists():
        raise FileNotFoundError("historical_daily.parquet not found. Run Part 0 first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared alpha helpers
# ---------------------------------------------------------------------------
def streak_when_true(flag: pd.Series, *, shift: bool = True) -> pd.Series:
    """Count consecutive True/1 days and reset to 0 on False/0 days.

    This is intentionally different from a raw run-length ``cumcount``. The
    old implementation counted both event-runs and non-event-runs, which made
    rare-event features such as heat-wave streaks grow into very large
    "days since last event" counters.

    Parameters
    ----------
    flag:
        Boolean or 0/1 event indicator aligned to observation dates.
    shift:
        If True, return yesterday's completed streak so the feature is known
        at decision time for the current feature_date.
    """
    clean = pd.Series(flag, index=flag.index).fillna(0).astype(int)
    groups = (clean != clean.shift()).cumsum()
    streak = clean.groupby(groups).cumcount() + 1
    streak = streak.where(clean.eq(1), 0).astype(float)
    return streak.shift(1) if shift else streak


# ---------------------------------------------------------------------------
# Alpha 1: Pressure gradient / tendency signals
# ---------------------------------------------------------------------------
def compute_pressure_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Pressure tendency, anomaly vs 30-day mean, acceleration."""
    alpha = df[["date"]].copy()

    if "pressure_mean_hpa" not in df.columns:
        return alpha

    p = df["pressure_mean_hpa"].copy()

    # Tendency (rate of change)
    alpha["alpha_pressure_tend_1d"] = p.diff(1)
    alpha["alpha_pressure_tend_2d"] = p.diff(2)
    alpha["alpha_pressure_tend_3d"] = p.diff(3)

    # Acceleration (2nd derivative)
    alpha["alpha_pressure_accel"] = alpha["alpha_pressure_tend_1d"].diff(1)

    # Anomaly vs 30-day rolling mean (shifted to avoid leakage)
    p_roll30 = p.shift(1).rolling(30, min_periods=10).mean()
    alpha["alpha_pressure_anom_30d"] = p.shift(1) - p_roll30

    # High-pressure persistence: count only consecutive above-threshold days.
    # ``shift=False`` here because this function shifts all alpha columns below.
    above_high = (p > 1015).astype(int)
    alpha["alpha_high_pressure_days"] = streak_when_true(above_high, shift=False)

    # Shift all alpha columns by 1 day (known at decision time)
    for col in [c for c in alpha.columns if c != "date"]:
        alpha[col] = alpha[col].shift(1)

    return alpha


# ---------------------------------------------------------------------------
# Alpha 2: Marine layer depth proxy
# ---------------------------------------------------------------------------
def compute_marine_layer_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Marine layer likelihood signals from humidity, cloud, dew point depression."""
    alpha = df[["date"]].copy()

    dew = df.get("dew_point_mean_f", None)
    temp_low = df.get("temp_low_f", None)
    cloud = df.get("cloud_cover_mean_pct", None)
    humidity = df.get("humidity_mean_pct", None)

    if dew is not None and temp_low is not None:
        dew_depression = temp_low - dew
        alpha["alpha_dew_depression_f"] = dew_depression.shift(1)
        alpha["alpha_marine_layer_flag"] = (dew_depression < MARINE_LAYER_THRESHOLD_F).astype(int).shift(1)

        # Consecutive marine-layer days. Count only true-runs, reset on false-runs.
        ml_flag = (dew_depression < MARINE_LAYER_THRESHOLD_F).astype(int)
        alpha["alpha_marine_streak"] = streak_when_true(ml_flag)

    if cloud is not None:
        # Stratus cloud anomaly vs seasonal norm
        cloud_clim = cloud.shift(1).rolling(30, min_periods=10).mean()
        alpha["alpha_cloud_anom_30d"] = (cloud.shift(1) - cloud_clim)

    if humidity is not None:
        # Humidity rate of change
        alpha["alpha_humidity_tend_1d"] = humidity.diff(1).shift(1)
        alpha["alpha_humidity_tend_3d"] = humidity.diff(3).shift(1)

    return alpha


# ---------------------------------------------------------------------------
# Alpha 3: Santa Ana wind index
# ---------------------------------------------------------------------------
def compute_santa_ana_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Santa Ana wind strength and direction signals."""
    alpha = df[["date"]].copy()

    wind_dir = df.get("wind_dir_dominant_deg", None)
    wind_speed = df.get("wind_speed_max_mph", None)
    humidity = df.get("humidity_mean_pct", None)
    gust = df.get("wind_gust_max_mph", None)

    if wind_dir is not None and wind_speed is not None:
        # Offshore direction component (Santa Ana winds blow from NE/E: 30-120°)
        rad = np.deg2rad(wind_dir.fillna(180.0))
        # Project onto easterly axis
        easterly_component = np.cos(rad - np.deg2rad(90))  # +1 = due east
        offshore_score = (easterly_component + 1) / 2  # 0–1

        speed = wind_speed.fillna(0.0)
        hum = humidity.fillna(50.0) if humidity is not None else pd.Series(50.0, index=df.index)
        dryness = (100.0 - hum) / 100.0

        alpha["alpha_santa_ana_index"] = (offshore_score * speed * dryness).shift(1)
        alpha["alpha_santa_ana_flag"] = (
            ((wind_dir >= 30) & (wind_dir <= 120) & (speed > 20) & (hum < 25)).astype(int).shift(1)
        )

        if gust is not None:
            alpha["alpha_gust_spread"] = (gust - wind_speed).fillna(0.0).shift(1)

        # Wind speed acceleration (sudden increase = incoming offshore event)
        alpha["alpha_wind_accel_1d"] = speed.diff(1).shift(1)

    return alpha


# ---------------------------------------------------------------------------
# Alpha 4: Temperature momentum / trend signals
# ---------------------------------------------------------------------------
def compute_temp_momentum_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of change, trend, and heat wave signals for temperature."""
    alpha = df[["date"]].copy()

    if "temp_high_f" not in df.columns:
        return alpha

    t = df["temp_high_f"]

    # Rate of change
    alpha["alpha_temp_mom_1d"] = t.diff(1).shift(1)
    alpha["alpha_temp_mom_3d"] = t.diff(3).shift(1)
    alpha["alpha_temp_mom_7d"] = t.diff(7).shift(1)

    # Acceleration
    alpha["alpha_temp_accel"] = t.diff(1).diff(1).shift(1)

    # Z-score vs 30-day rolling stats
    t_roll_mean = t.shift(1).rolling(30, min_periods=10).mean()
    t_roll_std = t.shift(1).rolling(30, min_periods=10).std().clip(lower=0.5)
    alpha["alpha_temp_zscore_30d"] = ((t.shift(1) - t_roll_mean) / t_roll_std)

    # Heat wave streak: count only consecutive heat-wave days, reset otherwise.
    hot_flag = (t >= HEAT_WAVE_THRESHOLD_F).astype(int)
    alpha["alpha_heat_wave_streak"] = streak_when_true(hot_flag)

    # Distance from seasonal peak (hottest typical day: ~Aug 15, DOY≈227)
    doy = pd.to_datetime(df["date"]).dt.dayofyear
    alpha["alpha_days_from_peak"] = np.abs(doy - 227)

    return alpha


# ---------------------------------------------------------------------------
# Alpha 5: ENSO index (Niño 3.4 anomaly)
# ---------------------------------------------------------------------------
def fetch_enso_index() -> Optional[pd.DataFrame]:
    """Attempt to fetch NOAA Niño 3.4 monthly anomaly index.

    Defensive behavior:
      - ``nina34.anom.data`` should already contain anomaly values.
      - If a source returns absolute SST-shaped values, usually 25--29°C,
        convert them to anomalies by subtracting each calendar month's
        climatological mean before thresholding.
    """
    try:
        print("[Part 2A] Fetching NOAA ENSO Niño 3.4 anomaly index...")
        resp = requests.get(ENSO_URL, timeout=15)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.split("\n") if l.strip()]
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) == 13:
                try:
                    year = int(parts[0])
                    if 1950 <= year <= 2100:
                        for month, val_str in enumerate(parts[1:], start=1):
                            val = float(val_str)
                            if -99 < val < 99:
                                records.append({
                                    "year": year,
                                    "month": month,
                                    "nino34_value": val,
                                })
                except (ValueError, IndexError):
                    continue
        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})

        max_abs = float(df["nino34_value"].abs().max())
        if max_abs > 10.0:
            # Absolute SST source accidentally supplied. Convert to anomaly.
            clim = df.groupby("month")["nino34_value"].transform("mean")
            df["nino34_anom"] = df["nino34_value"] - clim
            source_type = "absolute_sst_converted_to_monthly_anomaly"
        else:
            df["nino34_anom"] = df["nino34_value"]
            source_type = "anomaly"

        df.attrs["enso_source_type"] = source_type
        print(
            f"  → {len(df)} ENSO monthly records loaded "
            f"({source_type}; anomaly range {df['nino34_anom'].min():.2f} to {df['nino34_anom'].max():.2f})"
        )
        return df[["date", "nino34_anom"]]
    except Exception as exc:
        print(f"  [WARN] ENSO fetch failed: {exc}. Will compute proxy instead.")
        return None


def compute_enso_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a local ENSO proxy from pressure and temperature anomalies."""
    alpha = df[["date"]].copy()

    if "pressure_mean_hpa" in df.columns and "temp_high_f" in df.columns:
        p_anom_90d = df["pressure_mean_hpa"].shift(1) - df["pressure_mean_hpa"].shift(1).rolling(90, min_periods=30).mean()
        t_anom_90d = df["temp_high_f"].shift(1) - df["temp_high_f"].shift(1).rolling(90, min_periods=30).mean()
        # Combine: cold + low pressure = La Niña-like; warm + high = El Niño-like
        alpha["alpha_enso_proxy"] = (0.5 * t_anom_90d + 0.5 * p_anom_90d)

    return alpha


def merge_enso_features(df: pd.DataFrame, enso_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge monthly ENSO anomaly onto daily feature frame."""
    alpha = df[["date"]].copy()

    if enso_df is not None:
        daily_dates = pd.DataFrame({"date": df["date"]})
        daily_dates["year"] = daily_dates["date"].dt.year
        daily_dates["month"] = daily_dates["date"].dt.month
        enso_monthly = enso_df.copy()
        enso_monthly["year"] = enso_monthly["date"].dt.year
        enso_monthly["month"] = enso_monthly["date"].dt.month

        merged = daily_dates.merge(enso_monthly[["year", "month", "nino34_anom"]], on=["year", "month"], how="left")
        alpha["alpha_nino34_anom"] = merged["nino34_anom"].values

        # El Niño / La Niña flags
        alpha["alpha_el_nino_flag"] = (alpha["alpha_nino34_anom"] > 0.5).astype(int)
        alpha["alpha_la_nina_flag"] = (alpha["alpha_nino34_anom"] < -0.5).astype(int)

    return alpha


# ---------------------------------------------------------------------------
# Merge all alpha frames
# ---------------------------------------------------------------------------
def build_alpha_features(df: pd.DataFrame, enso_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    frames = [
        compute_pressure_alphas(df),
        compute_marine_layer_alphas(df),
        compute_santa_ana_alphas(df),
        compute_temp_momentum_alphas(df),
        compute_enso_proxy(df),
    ]
    if enso_df is not None:
        frames.append(merge_enso_features(df, enso_df))

    alpha_all = frames[0].copy()
    for frame in frames[1:]:
        common_date = frame["date"]
        for col in frame.columns:
            if col != "date":
                alpha_all[col] = frame[col].values

    # Drop near-duplicate or all-NaN columns
    alpha_cols = [c for c in alpha_all.columns if c != "date"]
    alpha_cols = [c for c in alpha_cols if alpha_all[c].notna().mean() > 0.3]
    alpha_all = alpha_all[["date"] + alpha_cols]

    print(f"[Part 2A] Built {len(alpha_cols)} alpha features over {len(alpha_all)} rows")
    return alpha_all


# ---------------------------------------------------------------------------
# Merge alphas into Part 1 feature matrix
# ---------------------------------------------------------------------------
def merge_into_feature_matrix(alpha_df: pd.DataFrame) -> None:
    feat_path = PART1_DIR / "feature_matrix.parquet"
    if not feat_path.exists():
        print("[Part 2A] feature_matrix.parquet not found — saving alpha features separately only.")
        return

    df_feat = pd.read_parquet(feat_path)
    df_feat["date"] = pd.to_datetime(df_feat["date"]).dt.normalize()

    # Drop any existing alpha columns to avoid duplication
    existing_alpha = [c for c in df_feat.columns if c.startswith("alpha_")]
    df_feat = df_feat.drop(columns=existing_alpha)

    df_merged = df_feat.merge(alpha_df, on="date", how="left")
    df_merged.to_parquet(feat_path, index=False)
    print(f"[Part 2A] Merged alpha features into feature_matrix.parquet ({len(df_merged)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 2A] Project root: {PROJECT_DIR}")

    df = load_historical()
    print(f"[Part 2A] Loaded {len(df)} historical rows")

    # Fetch ENSO data (non-blocking)
    enso_df = fetch_enso_index()

    # Build all alpha features
    alpha_df = build_alpha_features(df, enso_df)

    # Save alpha features
    alpha_path = ARTIFACTS_DIR / "alpha_features.parquet"
    alpha_df.to_parquet(alpha_path, index=False)
    print(f"[Part 2A] Saved alpha_features.parquet ({len(alpha_df)} rows)")

    # Merge into feature matrix
    merge_into_feature_matrix(alpha_df)

    # Metadata
    alpha_cols = [c for c in alpha_df.columns if c != "date"]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": pd.Timestamp.now().isoformat(),
        "n_alpha_features": len(alpha_cols),
        "alpha_feature_cols": alpha_cols,
        "enso_available": enso_df is not None,
        "enso_source_url": ENSO_URL if enso_df is not None else None,
        "enso_note": "alpha_nino34_anom is an anomaly series; absolute SST inputs are converted defensively",
        "feature_descriptions": {
            "alpha_pressure_tend_Xd": "Pressure tendency over X days (hPa/day)",
            "alpha_pressure_accel": "Pressure rate-of-change acceleration",
            "alpha_pressure_anom_30d": "Pressure anomaly vs 30-day rolling mean",
            "alpha_high_pressure_days": "Consecutive days above 1015 hPa",
            "alpha_dew_depression_f": "Dew point depression (temp_low - dew_point)",
            "alpha_marine_layer_flag": "1 if dew depression < 10°F (marine layer likely)",
            "alpha_marine_streak": "Consecutive marine layer days",
            "alpha_cloud_anom_30d": "Cloud cover anomaly vs 30-day mean",
            "alpha_santa_ana_index": "Offshore wind × dryness composite index",
            "alpha_santa_ana_flag": "1 if classic Santa Ana conditions detected",
            "alpha_gust_spread": "Wind gust minus mean wind speed",
            "alpha_wind_accel_1d": "Wind speed 1-day acceleration",
            "alpha_temp_mom_Xd": "Temperature momentum over X days",
            "alpha_temp_accel": "Temperature acceleration (2nd derivative)",
            "alpha_temp_zscore_30d": "Temperature z-score vs 30-day distribution",
            "alpha_heat_wave_streak": "Consecutive days at or above 95°F",
            "alpha_days_from_peak": "Days from Aug 15 (climatological hottest day)",
            "alpha_enso_proxy": "Local proxy for ENSO state from P/T anomalies",
            "alpha_nino34_anom": "NOAA Niño 3.4 monthly anomaly (if available)",
            "alpha_el_nino_flag": "1 if El Niño conditions (Niño 3.4 > 0.5)",
            "alpha_la_nina_flag": "1 if La Niña conditions (Niño 3.4 < -0.5)",
        },
    }
    with open(ARTIFACTS_DIR / "alpha_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[Part 2A] Saved alpha_meta.json")

    print(f"\n[Part 2A] ✅ Complete. {len(alpha_cols)} alpha features built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

