#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2B — XGBoost Ensemble Sleeve (Optional, Non-Blocking)
============================================================
Trains one XGBoost regressor per forecast horizon (H=1, H=3, H=5)
as a strong gradient-boosting baseline alongside the LSTM.

Outputs
-------
  Standalone predictions written to artifacts_part2b/
  Ensemble blend written back to artifacts_part2/prediction_log.csv
    (xgb_h1, xgb_h3, xgb_h5 columns appended)

Gate Validation
---------------
  gate_validation_passed: true  ← val MAE better than naive persistence baseline
  bnn_sleeve_recommended: true  ← if XGB significantly outperforms LSTM on val

Artifacts Written
-----------------
  artifacts_part2b/
      xgb_h1.pkl / xgb_h3.pkl / xgb_h5.pkl   — serialized models
      xgb_predictions.parquet                  — val + test predictions
      part2b_summary.json                      — metrics, gate, BNN recommendation
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
PART1_DIR = PROJECT_DIR / "artifacts_part1"
PART2_DIR = PROJECT_DIR / "artifacts_part2"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part2b"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.0.0"
HORIZONS = [1, 3, 5]

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "early_stopping_rounds": 30,
    "verbosity": 0,
}

# Gate: XGB val MAE must beat naive persistence by this margin (°F)
GATE_IMPROVEMENT_F = 0.2

# BNN recommendation: XGB must outperform LSTM val MAE by this margin
BNN_RECOMMENDATION_THRESHOLD_F = 0.3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    path = PART1_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError("feature_matrix.parquet not found. Run Part 1 first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def load_splits() -> Dict:
    path = PART1_DIR / "train_val_test_split.json"
    if not path.exists():
        raise FileNotFoundError("train_val_test_split.json not found.")
    with open(path) as f:
        return json.load(f)


def _feature_cols(df: pd.DataFrame) -> List[str]:
    target_cols = {f"target_h{h}" for h in HORIZONS}
    return [c for c in df.columns if c not in {"date"} | target_cols]


# ---------------------------------------------------------------------------
# Naive persistence baseline
# ---------------------------------------------------------------------------
def naive_persistence_mae(df_val: pd.DataFrame) -> Dict[str, float]:
    """MAE of naive forecast: predict that tomorrow = today."""
    maes = {}
    if "temp_high_f_lag1" not in df_val.columns:
        return maes
    for h in HORIZONS:
        y_true = df_val[f"target_h{h}"].dropna()
        if len(y_true) == 0:
            continue
        persistence = df_val.loc[y_true.index, "temp_high_f_lag1"]
        maes[f"h{h}"] = float(np.mean(np.abs(y_true.values - persistence.values)))
    return maes


# ---------------------------------------------------------------------------
# Feature importance analysis
# ---------------------------------------------------------------------------
def top_features(model, feature_cols: List[str], n: int = 20) -> List[Tuple[str, float]]:
    importances = model.feature_importances_
    pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    return pairs[:n]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_xgb_horizon(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    horizon: int,
) -> object:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is required for Part 2B. Install with: pip install xgboost")

    # Mask out NaN targets
    train_mask = np.isfinite(y_train)
    val_mask = np.isfinite(y_val)

    params = dict(XGB_PARAMS)
    model = XGBRegressor(**params)
    model.fit(
        X_train[train_mask], y_train[train_mask],
        eval_set=[(X_val[val_mask], y_val[val_mask])],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Ensemble blend with LSTM predictions
# ---------------------------------------------------------------------------
def blend_with_lstm(
    xgb_preds: Dict[str, np.ndarray],
    lstm_pred_log_path: Path,
    blend_weight_xgb: float = 0.40,
) -> Optional[pd.DataFrame]:
    """Simple weighted average blend of XGB and LSTM live predictions."""
    if not lstm_pred_log_path.exists():
        print("[Part 2B] LSTM prediction_log.csv not found — skipping blend.")
        return None

    df_log = pd.read_csv(lstm_pred_log_path)
    if df_log.empty:
        return None

    latest = df_log.iloc[-1].copy()
    for h in HORIZONS:
        xgb_val = xgb_preds.get(f"h{h}")
        lstm_val_col = f"target_h{h}"
        if xgb_val is not None and lstm_val_col in df_log.columns:
            lstm_val = float(latest.get(lstm_val_col, np.nan))
            if np.isfinite(lstm_val) and np.isfinite(xgb_val[-1]):
                blend = blend_weight_xgb * xgb_val[-1] + (1 - blend_weight_xgb) * lstm_val
                df_log.loc[df_log.index[-1], f"xgb_h{h}"] = float(xgb_val[-1])
                df_log.loc[df_log.index[-1], f"blend_h{h}"] = float(blend)

    df_log.to_csv(lstm_pred_log_path, index=False)
    print("[Part 2B] Updated prediction_log.csv with xgb and blend columns")
    return df_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 2B] Project root: {PROJECT_DIR}")

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[Part 2B] xgboost not installed. Skipping (non-blocking).")
        return 0

    # Load data
    df = load_data()
    splits = load_splits()
    feature_cols = _feature_cols(df)
    print(f"[Part 2B] {len(df)} rows, {len(feature_cols)} features")

    # Splits
    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])

    df_train = df[df["date"] <= train_end].copy()
    df_val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    df_test = df[df["date"] > val_end].copy()

    X_train = df_train[feature_cols].fillna(0.0).values
    X_val = df_val[feature_cols].fillna(0.0).values
    X_test = df_test[feature_cols].fillna(0.0).values
    X_all = df[feature_cols].fillna(0.0).values

    print(f"[Part 2B] Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Naive persistence baseline
    persistence_maes = naive_persistence_mae(df_val)
    print(f"[Part 2B] Naive persistence val MAE: {persistence_maes}")

    models: Dict[str, object] = {}
    val_metrics: Dict[str, float] = {}
    test_metrics: Dict[str, float] = {}
    xgb_live_preds: Dict[str, np.ndarray] = {}
    feature_importances: Dict[str, List] = {}
    all_val_preds: Dict[str, np.ndarray] = {}

    for h in HORIZONS:
        target_col = f"target_h{h}"
        y_train = df_train[target_col].values
        y_val_h = df_val[target_col].values
        y_test_h = df_test[target_col].values

        print(f"\n[Part 2B] Training XGB for H={h}...")
        model = train_xgb_horizon(X_train, y_train, X_val, y_val_h, h)
        models[f"h{h}"] = model

        # Validation metrics
        val_pred = model.predict(X_val)
        val_mask = np.isfinite(y_val_h)
        val_mae = float(np.mean(np.abs(val_pred[val_mask] - y_val_h[val_mask])))
        val_rmse = float(np.sqrt(np.mean((val_pred[val_mask] - y_val_h[val_mask]) ** 2)))
        val_metrics[f"h{h}_mae_f"] = val_mae
        val_metrics[f"h{h}_rmse_f"] = val_rmse
        all_val_preds[f"h{h}"] = val_pred
        print(f"  Val MAE={val_mae:.2f}°F | RMSE={val_rmse:.2f}°F")

        # Test metrics
        if len(df_test) > 0:
            test_pred = model.predict(X_test)
            test_mask = np.isfinite(y_test_h)
            test_mae = float(np.mean(np.abs(test_pred[test_mask] - y_test_h[test_mask])))
            test_rmse = float(np.sqrt(np.mean((test_pred[test_mask] - y_test_h[test_mask]) ** 2)))
            test_metrics[f"h{h}_mae_f"] = test_mae
            test_metrics[f"h{h}_rmse_f"] = test_rmse
            print(f"  Test MAE={test_mae:.2f}°F | RMSE={test_rmse:.2f}°F")

        # Live prediction (latest available row)
        live_pred = model.predict(X_all[-1:])
        xgb_live_preds[f"h{h}"] = live_pred

        # Feature importance
        feature_importances[f"h{h}"] = [
            {"feature": fc, "importance": float(imp)}
            for fc, imp in top_features(model, feature_cols, n=20)
        ]

        # Save model
        with open(ARTIFACTS_DIR / f"xgb_h{h}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Gate validation
    gate_passed = all(
        val_metrics.get(f"h{h}_mae_f", 999) <= persistence_maes.get(f"h{h}", 999) - GATE_IMPROVEMENT_F
        for h in HORIZONS if f"h{h}" in persistence_maes
    )
    print(f"\n[Part 2B] Gate validation passed: {gate_passed}")

    # BNN recommendation
    lstm_meta_path = PART2_DIR / "part2_meta.json"
    bnn_recommended = False
    if lstm_meta_path.exists():
        with open(lstm_meta_path) as f:
            lstm_meta = json.load(f)
        lstm_val_mae = lstm_meta.get("val_mae_f", None)
        xgb_avg_mae = float(np.mean([val_metrics[f"h{h}_mae_f"] for h in HORIZONS if f"h{h}_mae_f" in val_metrics]))
        if lstm_val_mae is not None:
            improvement = lstm_val_mae - xgb_avg_mae
            bnn_recommended = improvement > BNN_RECOMMENDATION_THRESHOLD_F
            print(f"  LSTM val MAE: {lstm_val_mae:.2f}°F | XGB avg: {xgb_avg_mae:.2f}°F")
            print(f"  BNN sleeve recommended: {bnn_recommended} (XGB advantage: {improvement:.2f}°F)")

    # Save prediction parquet
    val_df = df_val[["date"]].copy().reset_index(drop=True)
    for h in HORIZONS:
        val_df[f"xgb_pred_h{h}"] = all_val_preds[f"h{h}"]
        val_df[f"true_h{h}"] = df_val[f"target_h{h}"].values
    val_df.to_parquet(ARTIFACTS_DIR / "xgb_predictions.parquet", index=False)

    # Blend with LSTM
    lstm_log_path = PART2_DIR / "prediction_log.csv"
    blend_with_lstm(xgb_live_preds, lstm_log_path)

    # Print live predictions
    print("\n=== XGB LIVE PREDICTIONS ===")
    for h in HORIZONS:
        print(f"  H={h}: {float(xgb_live_preds[f'h{h}'][0]):.1f}°F")

    # Summary JSON
    summary = {
        "schema_version": SCHEMA_VERSION,
        "built_at": pd.Timestamp.now().isoformat(),
        "gate_validation_passed": gate_passed,
        "bnn_sleeve_recommended": bnn_recommended,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "persistence_baseline_mae": persistence_maes,
        "live_predictions": {f"h{h}": float(xgb_live_preds[f"h{h}"][0]) for h in HORIZONS},
        "feature_importances": feature_importances,
        "hyperparameters": XGB_PARAMS,
    }
    with open(ARTIFACTS_DIR / "part2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Part 2B] Saved part2b_summary.json")

    print(f"\n[Part 2B] ✅ Complete. Gate passed={gate_passed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
