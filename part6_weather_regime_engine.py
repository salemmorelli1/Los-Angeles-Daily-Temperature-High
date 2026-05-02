#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 6 — Weather Regime Engine
================================
Fits a regime model over historical meteorological features and writes daily
weather-regime artifacts for the LA Temperature Forecasting stack.

Primary backend: GaussianHMM from hmmlearn.
Fallback backend: deterministic KMeans if HMM fitting/decoding fails.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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

SCHEMA_VERSION = "1.1.0"
N_REGIMES = 3
MIN_ROWS_FOR_MODEL = 180


def load_historical() -> pd.DataFrame:
    path = SRC_DIR / "historical_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"[Part 6] historical_daily.parquet not found at {path}. Run Part 0 first."
        )
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError("[Part 6] historical_daily.parquet is missing required column: date")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _wind_cyclical(degrees: pd.Series) -> Tuple[pd.Series, pd.Series]:
    rad = np.deg2rad(pd.to_numeric(degrees, errors="coerce").fillna(0.0))
    return pd.Series(np.sin(rad), index=degrees.index), pd.Series(np.cos(rad), index=degrees.index)


def prepare_regime_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feat = df[["date"]].copy()

    feat["temp_high_f"] = _series_or_nan(df, "temp_high_f")
    feat["temp_low_f"] = _series_or_nan(df, "temp_low_f")
    feat["dew_point_mean_f"] = _series_or_nan(df, "dew_point_mean_f")
    feat["diurnal_range_f"] = feat["temp_high_f"] - feat["temp_low_f"]

    feat["humidity_mean_pct"] = _series_or_nan(df, "humidity_mean_pct")
    feat["cloud_cover_mean_pct"] = _series_or_nan(df, "cloud_cover_mean_pct")
    feat["pressure_mean_hpa"] = _series_or_nan(df, "pressure_mean_hpa")
    feat["wind_speed_max_mph"] = _series_or_nan(df, "wind_speed_max_mph")

    wind_dir = _series_or_nan(df, "wind_dir_dominant_deg").fillna(0.0)
    feat["wind_sin"], feat["wind_cos"] = _wind_cyclical(wind_dir)

    precip = _series_or_nan(df, "precip_in").fillna(0.0).clip(lower=0.0)
    feat["log_precip"] = np.log1p(precip)

    doy = pd.to_datetime(df["date"], errors="coerce").dt.dayofyear.fillna(1)
    feat["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    feature_cols = [
        "temp_high_f",
        "temp_low_f",
        "diurnal_range_f",
        "humidity_mean_pct",
        "cloud_cover_mean_pct",
        "dew_point_mean_f",
        "pressure_mean_hpa",
        "wind_speed_max_mph",
        "wind_sin",
        "wind_cos",
        "log_precip",
        "season_sin",
        "season_cos",
    ]

    threshold = int(len(feature_cols) * 0.70)
    feat_sub = feat[["date"] + feature_cols].copy()
    feat_sub = feat_sub.dropna(thresh=threshold + 1)

    for col in feature_cols:
        arr = pd.to_numeric(feat_sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = arr.median()
        if pd.isna(med):
            med = 0.0
        feat_sub[col] = arr.fillna(med).astype(float)

    return feat_sub.reset_index(drop=True), feature_cols


def fit_hmm(X: np.ndarray, n_regimes: int = N_REGIMES, n_iter: int = 200):
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-4,
        random_state=42,
        verbose=False,
        min_covar=1e-3,
    )
    model.fit(X)
    return model


def fit_kmeans_fallback(X: np.ndarray, n_regimes: int = N_REGIMES) -> KMeans:
    model = KMeans(n_clusters=n_regimes, random_state=42, n_init=25)
    model.fit(X)
    return model


def _soft_probs_from_distances(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    scale = np.nanmedian(d)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    logits = -d / scale
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expv = np.exp(logits)
    denom = expv.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return expv / denom


def _state_means_from_model(model: Any, scaler: StandardScaler, feature_cols: List[str], backend: str) -> pd.DataFrame:
    if backend == "hmm":
        means_scaled = np.asarray(model.means_, dtype=float)
    else:
        means_scaled = np.asarray(model.cluster_centers_, dtype=float)
    means_orig = scaler.inverse_transform(means_scaled)
    return pd.DataFrame(means_orig, columns=feature_cols)


def _assign_regime_labels(means_df: pd.DataFrame, n_regimes: int = N_REGIMES) -> Dict[int, str]:
    label_map: Dict[int, str] = {}
    remaining = list(range(n_regimes))

    temp = means_df.get("temp_high_f", pd.Series(0.0, index=means_df.index))
    humidity = means_df.get("humidity_mean_pct", pd.Series(0.0, index=means_df.index))
    cloud = means_df.get("cloud_cover_mean_pct", pd.Series(0.0, index=means_df.index))
    diurnal = means_df.get("diurnal_range_f", pd.Series(0.0, index=means_df.index))
    wind = means_df.get("wind_speed_max_mph", pd.Series(0.0, index=means_df.index))

    marine_score = (
        -0.35 * temp.rank(pct=True)
        + 0.30 * humidity.rank(pct=True)
        + 0.25 * cloud.rank(pct=True)
        - 0.10 * diurnal.rank(pct=True)
    )
    marine_idx = int(marine_score.idxmax())
    label_map[marine_idx] = "MARINE_LAYER"
    if marine_idx in remaining:
        remaining.remove(marine_idx)

    if remaining:
        santa_score = (
            0.35 * temp.rank(pct=True)
            - 0.35 * humidity.rank(pct=True)
            + 0.20 * diurnal.rank(pct=True)
            + 0.10 * wind.rank(pct=True)
        )
        santa_candidates = santa_score.iloc[remaining]
        santa_idx = int(santa_candidates.idxmax())
        label_map[santa_idx] = "SANTA_ANA"
        if santa_idx in remaining:
            remaining.remove(santa_idx)

    for idx in remaining:
        label_map[idx] = "DRY_CLEAR"

    return label_map


def _transition_matrix_from_states(states: np.ndarray, n_regimes: int = N_REGIMES) -> List[List[float]]:
    counts = np.ones((n_regimes, n_regimes), dtype=float) * 1e-6
    for a, b in zip(states[:-1], states[1:]):
        if 0 <= int(a) < n_regimes and 0 <= int(b) < n_regimes:
            counts[int(a), int(b)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs.tolist()


def save_model(
    model: Any,
    scaler: StandardScaler,
    label_map: Dict[int, str],
    feature_cols: List[str],
    backend: str,
) -> None:
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_map": label_map,
        "feature_cols": feature_cols,
        "backend": backend,
        "schema_version": SCHEMA_VERSION,
    }
    with open(ARTIFACTS_DIR / "regime_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("[Part 6] Saved regime_model.pkl", flush=True)


def main() -> int:
    print(f"[Part 6] Project root: {PROJECT_DIR}", flush=True)

    try:
        df = load_historical()
        print(f"[Part 6] Loaded {len(df)} historical rows", flush=True)

        feat_df, feature_cols = prepare_regime_features(df)
        print(f"[Part 6] Regime feature matrix: {len(feat_df)} rows × {len(feature_cols)} features", flush=True)

        if len(feat_df) < MIN_ROWS_FOR_MODEL:
            print(f"[Part 6] ERROR: Need at least {MIN_ROWS_FOR_MODEL} rows. Got {len(feat_df)}.", flush=True)
            return 1

        X_raw = feat_df[feature_cols].to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        backend = "hmm"
        hmm_error: Optional[str] = None

        try:
            print(f"[Part 6] Fitting GaussianHMM with {N_REGIMES} states...", flush=True)
            model = fit_hmm(X_scaled, n_regimes=N_REGIMES)
            converged = bool(getattr(model.monitor_, "converged", False))
            history = list(getattr(model.monitor_, "history", []))
            log_likelihood = float(history[-1]) if history else None
            states = model.predict(X_scaled)
            posteriors = model.predict_proba(X_scaled)
            print(f"  Converged: {converged}", flush=True)
            if log_likelihood is not None:
                print(f"  Log-likelihood: {log_likelihood:.2f}", flush=True)
        except Exception as exc:
            backend = "kmeans_fallback"
            hmm_error = repr(exc)
            print(f"[Part 6] WARNING: HMM failed; using KMeans fallback. Error: {hmm_error}", flush=True)
            model = fit_kmeans_fallback(X_scaled, n_regimes=N_REGIMES)
            states = model.predict(X_scaled)
            posteriors = _soft_probs_from_distances(model.transform(X_scaled))
            converged = True
            log_likelihood = None

        states = np.asarray(states, dtype=int)
        means_df = _state_means_from_model(model, scaler, feature_cols, backend)
        label_map = _assign_regime_labels(means_df, n_regimes=N_REGIMES)
        print(f"[Part 6] Regime label mapping: {label_map}", flush=True)

        regime_names = np.array([label_map.get(int(s), f"REGIME_{int(s)}") for s in states])

        tape = feat_df[["date"]].copy()
        tape["regime_state"] = states
        tape["regime"] = regime_names

        for idx in range(N_REGIMES):
            label = label_map.get(idx, f"REGIME_{idx}")
            safe_label = label.lower()
            tape[f"prob_{safe_label}"] = posteriors[:, idx]

        for col in ["prob_marine_layer", "prob_dry_clear", "prob_santa_ana"]:
            if col not in tape.columns:
                tape[col] = 0.0

        tape_path = ARTIFACTS_DIR / "regime_tape.parquet"
        tape.to_parquet(tape_path, index=False)
        print(f"[Part 6] Saved regime_tape.parquet ({len(tape)} rows)", flush=True)

        state_stats: Dict[str, Dict[str, Any]] = {}
        for state_idx, regime_label in label_map.items():
            mask = states == int(state_idx)
            state_stats[regime_label] = {
                "state_index": int(state_idx),
                "count": int(mask.sum()),
                "pct": float(mask.mean()) if len(mask) else 0.0,
                "means": {
                    col: float(means_df.loc[int(state_idx), col])
                    for col in feature_cols
                    if col in means_df.columns
                },
            }

        if backend == "hmm" and hasattr(model, "transmat_"):
            transition_matrix = np.asarray(model.transmat_, dtype=float).tolist()
        else:
            transition_matrix = _transition_matrix_from_states(states, N_REGIMES)

        meta = {
            "schema_version": SCHEMA_VERSION,
            "fit_date": pd.Timestamp.now().isoformat(),
            "backend": backend,
            "hmm_error": hmm_error,
            "n_regimes": N_REGIMES,
            "n_rows_fit": int(len(feat_df)),
            "feature_cols": feature_cols,
            "label_map": {str(k): v for k, v in label_map.items()},
            "state_stats": state_stats,
            "transition_matrix": transition_matrix,
            "hmm_converged": bool(converged) if backend == "hmm" else None,
            "hmm_log_likelihood": log_likelihood,
        }

        with open(ARTIFACTS_DIR / "regime_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print("[Part 6] Saved regime_meta.json", flush=True)

        save_model(model, scaler, label_map, feature_cols, backend)

        print(f"\n[Part 6] ✅ Complete using backend={backend}.", flush=True)
        return 0

    except Exception as exc:
        print(f"[Part 6] ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
