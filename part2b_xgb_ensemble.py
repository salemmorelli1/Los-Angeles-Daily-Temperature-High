#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2B — XGBoost Ensemble Sleeve + Canonical Forecast Column Publisher
========================================================================
Trains one XGBoost regressor per horizon (H=1, H=3, H=5) as a strong
gradient-boosting baseline alongside the LSTM.

Canonical forecast columns
--------------------------
Part 2B owns the canonical forecast_h1 / forecast_h3 / forecast_h5 columns
and forecast_source / forecast_reason in the prediction log.

Fallback chain (evaluated in order):
  1. blend_h*     — 0.40 * XGB + 0.60 * LSTM  (if both present and pass sanity)
  2. xgb_h*       — XGB alone                  (if LSTM is implausible)
  3. nws_h*       — NWS official forecast       (if XGB not available)
  4. persistence  — last observed temp          (last resort)

A candidate is "plausible" if its deviation from the last observed temperature
is ≤ FORECAST_SANITY_THRESHOLD_F. The LSTM is always checked; if it fails the
deviation check, the blend is discarded and XGB is used alone.

Gate Validation
---------------
  gate_validation_passed  = XGB val MAE beats naive persistence by >0.2°F
  bnn_sleeve_recommended  = XGB outperforms LSTM val MAE by >0.3°F

Artifacts Written
-----------------
  artifacts_part2b/
      xgb_h1.pkl / xgb_h3.pkl / xgb_h5.pkl  — serialized models
      xgb_predictions.parquet                 — val predictions
      part2b_summary.json                     — metrics, gate, BNN flag
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
PART0_DIR = PROJECT_DIR / "artifacts_part0"
PART1_DIR = PROJECT_DIR / "artifacts_part1"
PART2_DIR = PROJECT_DIR / "artifacts_part2"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part2b"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.1.0"
HORIZONS = [1, 3, 5]

XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.80, "colsample_bytree": 0.75, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "n_jobs": -1, "objective": "reg:squarederror", "eval_metric": "mae",
    "early_stopping_rounds": 30, "verbosity": 0,
}

GATE_IMPROVEMENT_F = 0.2           # XGB must beat persistence by this much
BNN_RECOMMENDATION_THRESHOLD_F = 0.3  # XGB must beat LSTM by this much

# A forecast is "plausible" if it deviates by less than this from last observed
FORECAST_SANITY_THRESHOLD_F = 15.0

BLEND_WEIGHT_XGB = 0.40           # blend = 0.40 * XGB + 0.60 * LSTM
LOG_KEY_COLS = ("decision_date", "feature_date", "model")


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
    with open(PART1_DIR / "train_val_test_split.json") as f:
        return json.load(f)


def _feature_cols(df: pd.DataFrame) -> List[str]:
    target_cols = {f"target_h{h}" for h in HORIZONS}
    excluded = {"date"} | target_cols
    cols: List[str] = []
    dropped: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            cols.append(col)
        else:
            c = pd.to_numeric(s, errors="coerce")
            if int(s.notna().sum()) > 0 and c[s.notna()].notna().all():
                cols.append(col)
            else:
                dropped.append(col)
    if dropped:
        print(f"[Part 2B] Dropping non-numeric columns: {dropped}")
    if not cols:
        raise ValueError("No numeric feature columns for Part 2B.")
    return cols


def _clean(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    X = df[cols].copy()
    for c in cols:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def naive_persistence_mae(df_val: pd.DataFrame) -> Dict[str, float]:
    maes: Dict[str, float] = {}
    if "temp_high_f_lag1" not in df_val.columns:
        return maes
    for h in HORIZONS:
        y = df_val[f"target_h{h}"].dropna()
        if len(y) == 0:
            continue
        pers = df_val.loc[y.index, "temp_high_f_lag1"]
        maes[f"h{h}"] = float(np.mean(np.abs(y.values - pers.values)))
    return maes




def heat_event_diagnostics_1d(
    pred: np.ndarray,
    true: np.ndarray,
    threshold_f: float = 85.0,
) -> Dict[str, float]:
    """Upper-tail diagnostics for one horizon."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    heat = mask & (true > threshold_f)
    n_heat = int(heat.sum())
    if n_heat == 0:
        return {
            "threshold_f": float(threshold_f),
            "n_true_heat_days": 0,
            "predicted_heat_hits": 0,
            "hit_rate": None,
            "heat_mae_f": None,
            "heat_bias_f": None,
            "max_true_f": float(np.nanmax(true[mask])) if mask.any() else None,
            "max_pred_f": float(np.nanmax(pred[mask])) if mask.any() else None,
        }
    err = pred[heat] - true[heat]
    return {
        "threshold_f": float(threshold_f),
        "n_true_heat_days": n_heat,
        "predicted_heat_hits": int((pred[heat] > threshold_f).sum()),
        "hit_rate": float((pred[heat] > threshold_f).mean()),
        "heat_mae_f": float(np.mean(np.abs(err))),
        "heat_bias_f": float(np.mean(err)),
        "max_true_f": float(np.max(true[heat])),
        "max_pred_f": float(np.max(pred[mask])) if mask.any() else None,
    }


def load_last_observed_temp() -> Optional[float]:
    """Return the most recent observed temp_high_f from Part 0 historical data."""
    hist_path = PART0_DIR / "historical_daily.parquet"
    if not hist_path.exists():
        return None
    hist = pd.read_parquet(hist_path)
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")
    vals = hist["temp_high_f"].dropna()
    return float(vals.iloc[-1]) if len(vals) > 0 else None


def load_nws_forecast_for_horizons(feature_date: pd.Timestamp) -> Dict[int, Optional[float]]:
    """Return NWS forecast high for each horizon's target date."""
    nws_path = PART0_DIR / "nws_official_forecast.json"
    if not nws_path.exists():
        return {}
    with open(nws_path) as f:
        nws = json.load(f)
    daily = nws.get("daily_high_f", {})
    result: Dict[int, Optional[float]] = {}
    for h in HORIZONS:
        target_date = (feature_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        result[h] = float(daily[target_date]) if target_date in daily else None
    return result


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def top_features(model, feature_cols: List[str], n: int = 20) -> List[Tuple[str, float]]:
    pairs = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    return pairs[:n]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_xgb_horizon(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> object:
    from xgboost import XGBRegressor
    tr_mask = np.isfinite(y_train)
    va_mask = np.isfinite(y_val)
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train[tr_mask], y_train[tr_mask],
        eval_set=[(X_val[va_mask], y_val[va_mask])],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Canonical forecast fallback chain
# ---------------------------------------------------------------------------
def _is_plausible(value: float, last_obs: float) -> bool:
    return np.isfinite(value) and abs(value - last_obs) <= FORECAST_SANITY_THRESHOLD_F


def compute_canonical_forecast(
    xgb_preds: Dict[str, float],
    lstm_preds: Dict[str, float],
    nws_preds: Dict[int, Optional[float]],
    last_obs: Optional[float],
) -> Tuple[Dict[str, float], str, str]:
    """Apply the fallback chain and return (forecast, source, reason).

    Chain: blend → xgb → nws → persistence
    The blend is discarded if the LSTM component fails the sanity check.
    """
    forecast: Dict[str, float] = {}
    sources: List[str] = []
    reasons: List[str] = []

    if last_obs is None:
        last_obs = float("nan")

    for h in HORIZONS:
        key = f"h{h}"
        xgb_val = xgb_preds.get(key)
        lstm_val = lstm_preds.get(key)

        # --- blend ---
        if (xgb_val is not None and np.isfinite(xgb_val) and
                lstm_val is not None and np.isfinite(lstm_val)):
            lstm_ok = _is_plausible(lstm_val, last_obs) if np.isfinite(last_obs) else True
            xgb_ok = _is_plausible(xgb_val, last_obs) if np.isfinite(last_obs) else True
            if lstm_ok and xgb_ok:
                forecast[key] = BLEND_WEIGHT_XGB * xgb_val + (1 - BLEND_WEIGHT_XGB) * lstm_val
                sources.append("blend")
                reasons.append(f"H{h}:blend(xgb={xgb_val:.1f},lstm={lstm_val:.1f})")
                continue
            # LSTM failed sanity — use XGB alone if it passes
            if xgb_ok:
                forecast[key] = xgb_val
                sources.append("xgb")
                reasons.append(f"H{h}:xgb_only(lstm_implausible:{lstm_val:.1f})")
                continue

        # --- xgb alone ---
        if xgb_val is not None and np.isfinite(xgb_val):
            if not np.isfinite(last_obs) or _is_plausible(xgb_val, last_obs):
                forecast[key] = xgb_val
                sources.append("xgb")
                reasons.append(f"H{h}:xgb_only")
                continue

        # --- NWS ---
        nws_val = nws_preds.get(h)
        if nws_val is not None and np.isfinite(nws_val):
            forecast[key] = nws_val
            sources.append("nws")
            reasons.append(f"H{h}:nws_fallback")
            continue

        # --- persistence ---
        if np.isfinite(last_obs):
            forecast[key] = last_obs
            sources.append("persistence")
            reasons.append(f"H{h}:persistence_fallback")
        else:
            forecast[key] = float("nan")
            sources.append("unavailable")
            reasons.append(f"H{h}:no_forecast_available")

    # Derive summary source label
    unique_sources = list(dict.fromkeys(sources))  # ordered unique
    source_label = "+".join(unique_sources) if unique_sources else "unavailable"
    return forecast, source_label, " | ".join(reasons)


# ---------------------------------------------------------------------------
# Prediction log — idempotent upsert (mirrors Part 2 helper)
# ---------------------------------------------------------------------------
def _log_path() -> Path:
    return PART2_DIR / "prediction_log.csv"


def load_prediction_log() -> pd.DataFrame:
    p = _log_path()
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def upsert_log_columns(updates: Dict, decision_date: str, feature_date: str, model: str) -> None:
    """Update specific columns on the matching row. Does not create new rows."""
    df = load_prediction_log()
    if df.empty:
        print("[Part 2B] prediction_log.csv not found — forecast_h* not written.")
        return

    key_vals = {
        "decision_date": str(decision_date).strip(),
        "feature_date": str(feature_date).strip(),
        "model": str(model).strip().upper(),
    }
    match = pd.Series([True] * len(df))
    for k, v in key_vals.items():
        col = df[k].astype(str).str.strip() if k in df.columns else pd.Series([""] * len(df))
        match = match & (col == v)

    if not match.any():
        print(f"[Part 2B] No matching row in prediction_log for {key_vals}; skipping update.")
        return

    idx = df.index[match][-1]
    for col, val in updates.items():
        df.loc[idx, col] = val

    df.to_csv(_log_path(), index=False)
    print(f"[Part 2B] Updated prediction_log row for {decision_date}: "
          f"forecast_source={updates.get('forecast_source')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 2B] Project root: {PROJECT_DIR}")

    try:
        from xgboost import XGBRegressor  # noqa: F401
    except ImportError:
        print("[Part 2B] xgboost not installed — skipping (non-blocking).")
        return 0

    df = load_data()
    splits = load_splits()
    feature_cols = _feature_cols(df)
    print(f"[Part 2B] {len(df)} rows, {len(feature_cols)} features")

    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])
    df_train = df[df["date"] <= train_end].copy()
    df_val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    df_test = df[df["date"] > val_end].copy()

    X_tr = _clean(df_train, feature_cols)
    X_va = _clean(df_val, feature_cols)
    X_te = _clean(df_test, feature_cols)
    X_all = _clean(df, feature_cols)
    print(f"[Part 2B] Train:{len(df_train)} Val:{len(df_val)} Test:{len(df_test)}")

    pers_maes = naive_persistence_mae(df_val)
    print(f"[Part 2B] Persistence val MAE: {pers_maes}")

    models: Dict = {}
    val_metrics: Dict = {}
    test_metrics: Dict = {}
    val_heat_event_diagnostics: Dict = {}
    test_heat_event_diagnostics: Dict = {}
    xgb_live: Dict[str, float] = {}
    val_preds: Dict = {}
    feat_importances: Dict = {}

    for h in HORIZONS:
        tc = f"target_h{h}"
        print(f"\n[Part 2B] Training XGB H={h}...")
        m = train_xgb_horizon(X_tr, df_train[tc].values, X_va, df_val[tc].values)
        models[f"h{h}"] = m

        vp = m.predict(X_va)
        vm = np.isfinite(df_val[tc].values)
        mae_v = float(np.mean(np.abs(vp[vm] - df_val[tc].values[vm])))
        rmse_v = float(np.sqrt(np.mean((vp[vm] - df_val[tc].values[vm]) ** 2)))
        val_metrics[f"h{h}_mae_f"] = mae_v
        val_metrics[f"h{h}_rmse_f"] = rmse_v
        val_heat_event_diagnostics[f"h{h}"] = heat_event_diagnostics_1d(vp, df_val[tc].values)
        val_preds[f"h{h}"] = vp
        print(f"  Val MAE={mae_v:.2f}°F  RMSE={rmse_v:.2f}°F")

        if len(df_test) > 0:
            tp = m.predict(X_te)
            tm = np.isfinite(df_test[tc].values)
            mae_t = float(np.mean(np.abs(tp[tm] - df_test[tc].values[tm])))
            rmse_t = float(np.sqrt(np.mean((tp[tm] - df_test[tc].values[tm]) ** 2)))
            test_metrics[f"h{h}_mae_f"] = mae_t
            test_metrics[f"h{h}_rmse_f"] = rmse_t
            test_heat_event_diagnostics[f"h{h}"] = heat_event_diagnostics_1d(tp, df_test[tc].values)
            print(f"  Test MAE={mae_t:.2f}°F  RMSE={rmse_t:.2f}°F")

        live = float(m.predict(X_all[-1:])[0])
        xgb_live[f"h{h}"] = live
        feat_importances[f"h{h}"] = [
            {"feature": fc, "importance": float(imp)}
            for fc, imp in top_features(m, feature_cols, 20)
        ]
        with open(ARTIFACTS_DIR / f"xgb_h{h}.pkl", "wb") as f:
            pickle.dump(m, f)

    # Gate validation
    gate = all(
        val_metrics.get(f"h{h}_mae_f", 999) <= pers_maes.get(f"h{h}", 999) - GATE_IMPROVEMENT_F
        for h in HORIZONS if f"h{h}" in pers_maes
    )
    print(f"\n[Part 2B] Gate validation passed: {gate}")

    # BNN recommendation
    bnn_rec = False
    p2_meta_path = PART2_DIR / "part2_meta.json"
    if p2_meta_path.exists():
        with open(p2_meta_path) as f:
            p2m = json.load(f)
        lstm_val_mae = p2m.get("val_mae_f")
        xgb_avg = float(np.mean([val_metrics[f"h{h}_mae_f"] for h in HORIZONS
                                  if f"h{h}_mae_f" in val_metrics]))
        if lstm_val_mae is not None:
            bnn_rec = (lstm_val_mae - xgb_avg) > BNN_RECOMMENDATION_THRESHOLD_F
            print(f"  LSTM val MAE={lstm_val_mae:.2f}°F  XGB avg={xgb_avg:.2f}°F  "
                  f"BNN recommended={bnn_rec}")

    # Save val predictions parquet
    val_df = df_val[["date"]].copy().reset_index(drop=True)
    for h in HORIZONS:
        val_df[f"xgb_pred_h{h}"] = val_preds[f"h{h}"]
        val_df[f"true_h{h}"] = df_val[f"target_h{h}"].values
    val_df.to_parquet(ARTIFACTS_DIR / "xgb_predictions.parquet", index=False)

    # -------------------------------------------------------------------
    # Canonical forecast fallback chain
    # -------------------------------------------------------------------
    feature_date = pd.Timestamp(df["date"].max()).normalize()
    decision_date = pd.Timestamp.today().normalize()
    model_key = "LSTM"  # we're updating the LSTM row written by Part 2

    # Load LSTM live preds from log
    log = load_prediction_log()
    lstm_live: Dict[str, float] = {}
    if not log.empty:
        # Find the most recent row for this decision_date + feature_date
        dd_str = decision_date.strftime("%Y-%m-%d")
        fd_str = feature_date.strftime("%Y-%m-%d")
        mask = (
            log["decision_date"].astype(str).str.strip() == dd_str
        )
        if "feature_date" in log.columns:
            mask = mask & (log["feature_date"].astype(str).str.strip() == fd_str)
        sub = log[mask]
        if not sub.empty:
            row = sub.iloc[-1]
            for h in HORIZONS:
                v = pd.to_numeric(pd.Series([row.get(f"target_h{h}", np.nan)]),
                                  errors="coerce").iloc[0]
                if np.isfinite(v):
                    lstm_live[f"h{h}"] = float(v)

    last_obs = load_last_observed_temp()
    nws_preds = load_nws_forecast_for_horizons(feature_date)
    forecast, source, reason = compute_canonical_forecast(
        xgb_live, lstm_live, nws_preds, last_obs
    )

    print("\n=== CANONICAL FORECAST ===")
    print(f"  Source: {source}")
    for h in HORIZONS:
        print(f"  H={h}: {forecast.get(f'h{h}', float('nan')):.1f}°F")
    if last_obs:
        print(f"  Last observed: {last_obs:.1f}°F")

    # Write canonical columns to prediction log row
    updates: Dict = {
        "forecast_source": source,
        "forecast_reason": reason,
    }
    for h in HORIZONS:
        updates[f"xgb_h{h}"] = xgb_live.get(f"h{h}", np.nan)
        updates[f"forecast_h{h}"] = forecast.get(f"h{h}", np.nan)
        if f"h{h}" in lstm_live:
            blend = BLEND_WEIGHT_XGB * xgb_live.get(f"h{h}", 0) + \
                    (1 - BLEND_WEIGHT_XGB) * lstm_live[f"h{h}"]
            updates[f"blend_h{h}"] = blend

    upsert_log_columns(updates, dd_str, fd_str, model_key)

    # Save summary
    summary = {
        "schema_version": SCHEMA_VERSION,
        "built_at": pd.Timestamp.now().isoformat(),
        "gate_validation_passed": gate,
        "bnn_sleeve_recommended": bnn_rec,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_heat_event_diagnostics": val_heat_event_diagnostics,
        "test_heat_event_diagnostics": test_heat_event_diagnostics,
        "persistence_baseline_mae": pers_maes,
        "xgb_live_predictions": xgb_live,
        "canonical_forecast": forecast,
        "forecast_source": source,
        "forecast_reason": reason,
        "feature_importances": feat_importances,
        "hyperparameters": XGB_PARAMS,
    }
    with open(ARTIFACTS_DIR / "part2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Part 2B] Saved part2b_summary.json")
    print(f"\n[Part 2B] ✅  Complete. Gate={gate}  Source={source}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())








