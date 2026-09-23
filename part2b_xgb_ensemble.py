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

Per-horizon forecast policy:
  1. Use a validation-approved XGB/LSTM blend or individual model only when
     recent paired realized errors establish skill over persistence.
  2. Use persistence when learned skill is absent or not established.
  3. Apply an NWS adjustment only when its paired realized-skill gate passes.
  4. Keep shadow learned candidates in the log to continue evaluating skill.

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
      xgb_predictions.parquet                 — validation + test predictions with split column
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

from forecast_protocol import (
    moving_block_bootstrap_mean_ci,
    pacific_today_timestamp,
    purged_labeled_splits,
)

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

SCHEMA_VERSION = "1.4.0-temporal-purge-live-skill"
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

# NWS anchoring is a learned overlay, not an unconditional safety rule. It is
# enabled per horizon only after row-level, realized forecasts show that NWS has
# beaten the independent pre-anchor model by a material MAE margin. This gate
# fails closed when history is missing or incomplete.
NWS_ANCHOR_MIN_REALIZED_SAMPLES = 30
NWS_ANCHOR_LOOKBACK_ROWS = 90
NWS_ANCHOR_MIN_MAE_IMPROVEMENT_F = 0.25

LIVE_SKILL_MIN_REALIZED_SAMPLES = 30
LIVE_SKILL_LOOKBACK_ROWS = 90

# A forecast is "plausible" if it deviates by less than this from last observed
FORECAST_SANITY_THRESHOLD_F = 15.0

DEFAULT_BLEND_WEIGHT_XGB = 0.40     # fallback blend = 0.40 * XGB + 0.60 * LSTM
BLEND_WEIGHT_GRID_STEP = 0.05       # coarse grid to avoid overfitting the validation set
MIN_BLEND_TUNE_ROWS = 50            # common LSTM/XGB validation rows required per horizon
MIN_BLEND_IMPROVEMENT_F = 0.05      # keep default unless tuned blend improves MAE by at least this

# NWS anchoring / heat-event safety overlay.
# These do not replace the model stack in normal conditions. They only keep the
# published canonical forecast from materially diverging from the official NWS
# benchmark, especially on warm/heat-event days where the learned models have
# shown a systematic lower-tail bias.
# NWS anchoring thresholds were tightened after the heat-branch audit showed
# persistent cold live forecasts even after the first anchoring layer.  These
# values keep the model stack active but prevent 4–6°F cold gaps against the
# official NWS benchmark on warm forecast days.
NWS_SOFT_DEVIATION_F = 3.0          # begin partial NWS anchoring sooner
NWS_STRONG_DEVIATION_F = 8.0        # stronger NWS anchoring
NWS_HARD_DEVIATION_F = 18.0         # use NWS directly if exceeded
NWS_SOFT_ANCHOR_WEIGHT = 0.60       # adjusted = 0.60*NWS + 0.40*model
NWS_STRONG_ANCHOR_WEIGHT = 0.80     # adjusted = 0.80*NWS + 0.20*model

# Warm/heat guards: when the official forecast says LA is warm/hot, do not
# publish a canonical forecast that remains materially colder than NWS.
WARM_EVENT_THRESHOLD_F = 78.0
MAX_WARM_NWS_COLD_GAP_F = 3.0
HEAT_EVENT_THRESHOLD_F = 85.0
HEAT_EVENT_MIN_MODEL_F = 83.0       # if NWS says heat, do not publish a very low model value
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

    # Drop zero-variance (constant) feature columns — mirrors the Part 2 guard.
    # Part 2A runs after Part 1, so a constant alpha feature (e.g. alpha_santa_ana_flag
    # before the threshold fix) can reach Part 2B even if Part 1's guard already ran.
    constant: List[str] = []
    nonconstant: List[str] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.nunique(dropna=True) > 1:
            nonconstant.append(col)
        else:
            constant.append(col)
    if constant:
        print(f"[Part 2B] Dropping {len(constant)} constant/zero-variance feature(s): {constant}")
    cols = nonconstant

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
    """Score persistence from the last observation known on feature_date."""
    maes: Dict[str, float] = {}
    if "temp_high_f" not in df_val.columns:
        return maes
    for h in HORIZONS:
        target_col = f"target_h{h}"
        if target_col not in df_val.columns:
            continue
        paired = pd.DataFrame({
            "target": pd.to_numeric(df_val[target_col], errors="coerce"),
            "persistence": pd.to_numeric(df_val["temp_high_f"], errors="coerce"),
        }).dropna()
        if not paired.empty:
            maes[f"h{h}"] = float(
                np.mean(np.abs(paired["target"].to_numpy() - paired["persistence"].to_numpy()))
            )
    return maes


def evaluate_live_model_skill(
    df_log: pd.DataFrame,
    min_samples: int = LIVE_SKILL_MIN_REALIZED_SAMPLES,
    lookback_rows: int = LIVE_SKILL_LOOKBACK_ROWS,
) -> Tuple[Dict[int, bool], Dict[str, Dict[str, object]]]:
    """Fail closed unless recent paired errors show learned-model skill.

    Loss differences are model absolute error minus persistence absolute error;
    negative values favor the learned candidate. The block bootstrap preserves
    short-range serial dependence across adjacent forecast issue dates.
    """
    allowed = {h: False for h in HORIZONS}
    diagnostics: Dict[str, Dict[str, object]] = {}
    for h in HORIZONS:
        real_col = f"realized_h{h}"
        persistence_col = f"persistence_h{h}"
        candidate_col = f"forecast_candidate_h{h}"
        pre_anchor_col = f"forecast_pre_anchor_h{h}"
        if df_log.empty or real_col not in df_log or persistence_col not in df_log:
            diagnostics[f"h{h}"] = {
                "enabled": False, "reason": "paired_history_missing", "n_paired": 0
            }
            continue

        candidate = pd.to_numeric(
            df_log.get(candidate_col, pd.Series(np.nan, index=df_log.index)),
            errors="coerce",
        )
        pre_anchor = pd.to_numeric(
            df_log.get(pre_anchor_col, pd.Series(np.nan, index=df_log.index)),
            errors="coerce",
        )
        candidate = candidate.where(candidate.notna(), pre_anchor)
        paired = pd.DataFrame({
            "candidate": candidate,
            "persistence": pd.to_numeric(df_log[persistence_col], errors="coerce"),
            "realized": pd.to_numeric(df_log[real_col], errors="coerce"),
        })
        if "feature_date" in df_log.columns:
            paired["_feature_date"] = pd.to_datetime(df_log["feature_date"], errors="coerce")
            sort_cols = ["_feature_date"]
            if "decision_date" in df_log.columns:
                paired["_decision_date"] = pd.to_datetime(df_log["decision_date"], errors="coerce")
                sort_cols.append("_decision_date")
            paired = paired.sort_values(sort_cols, kind="stable")
        elif "decision_date" in df_log.columns:
            paired["_decision_date"] = pd.to_datetime(df_log["decision_date"], errors="coerce")
            paired = paired.sort_values("_decision_date", kind="stable")
        paired = paired.dropna(subset=["candidate", "persistence", "realized"]).tail(lookback_rows)
        n_paired = int(len(paired))
        delta = (
            np.abs(paired["candidate"].to_numpy() - paired["realized"].to_numpy())
            - np.abs(paired["persistence"].to_numpy() - paired["realized"].to_numpy())
        )
        if n_paired < min_samples:
            diagnostics[f"h{h}"] = {
                "enabled": False,
                "reason": "insufficient_paired_history",
                "n_paired": n_paired,
                "min_samples": int(min_samples),
                "lookback_rows": int(lookback_rows),
            }
            continue

        ci = moving_block_bootstrap_mean_ci(delta, block_length=5, n_boot=2000, seed=42 + h)
        model_mae = float(np.abs(paired["candidate"] - paired["realized"]).mean())
        persistence_mae = float(np.abs(paired["persistence"] - paired["realized"]).mean())
        enabled = bool(ci["upper"] < 0.0)
        allowed[h] = enabled
        diagnostics[f"h{h}"] = {
            "enabled": enabled,
            "reason": "paired_skill_ci_below_zero" if enabled else "skill_not_established",
            "n_paired": n_paired,
            "lookback_rows": int(lookback_rows),
            "candidate_mae_f": model_mae,
            "persistence_mae_f": persistence_mae,
            "mean_paired_loss_delta_f": float(delta.mean()),
            "loss_delta_ci95_f": [ci["lower"], ci["upper"]],
            "bootstrap_block_length_rows": int(ci["block_length_rows"]),
            "bootstrap_replicates": int(ci["n_boot"]),
        }
    return allowed, diagnostics


def evaluate_xgb_validation_gate(
    val_metrics: Dict[str, float],
    persistence_maes: Dict[str, float],
) -> Tuple[bool, Dict[str, Dict[str, object]]]:
    """Require finite XGB and persistence MAE for every governed horizon.

    ``all([])`` is true in Python, so the previous inline expression could
    report a passing gate when the persistence baseline was absent. A model
    promotion gate must instead fail closed on missing or non-finite evidence.
    """
    diagnostics: Dict[str, Dict[str, object]] = {}
    passed = True

    for h in HORIZONS:
        xgb_mae = pd.to_numeric(
            pd.Series([val_metrics.get(f"h{h}_mae_f", np.nan)]), errors="coerce"
        ).iloc[0]
        persistence_mae = pd.to_numeric(
            pd.Series([persistence_maes.get(f"h{h}", np.nan)]), errors="coerce"
        ).iloc[0]
        evidence_available = bool(np.isfinite(xgb_mae) and np.isfinite(persistence_mae))
        horizon_pass = bool(
            evidence_available
            and float(xgb_mae) <= float(persistence_mae) - GATE_IMPROVEMENT_F
        )
        diagnostics[f"h{h}"] = {
            "passed": horizon_pass,
            "evidence_available": evidence_available,
            "xgb_val_mae_f": float(xgb_mae) if np.isfinite(xgb_mae) else None,
            "persistence_val_mae_f": (
                float(persistence_mae) if np.isfinite(persistence_mae) else None
            ),
            "required_improvement_f": float(GATE_IMPROVEMENT_F),
        }
        passed = passed and horizon_pass

    return bool(passed), diagnostics


def select_gate_approved_xgb_predictions(
    xgb_predictions: Dict[str, float],
    diagnostics: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    """Keep validation-approved XGB horizons even if another horizon is blocked."""
    approved: Dict[str, float] = {}
    for h in HORIZONS:
        key = f"h{h}"
        value = pd.to_numeric(
            pd.Series([xgb_predictions.get(key, np.nan)]), errors="coerce"
        ).iloc[0]
        if diagnostics.get(key, {}).get("passed", False) and np.isfinite(value):
            approved[key] = float(value)
    return approved


def evaluate_nws_anchor_skill(
    df_log: pd.DataFrame,
) -> Tuple[Dict[int, bool], Dict[str, Dict[str, object]]]:
    """Decide per horizon whether realized evidence permits NWS anchoring.

    Only forecasts having pre-anchor, NWS, and realized values on the same row
    enter the paired comparison. The most recent rows are used so an obsolete
    NWS grid or an old operating regime cannot authorize the overlay forever.
    """
    allowed = {h: False for h in HORIZONS}
    diagnostics: Dict[str, Dict[str, object]] = {}

    for h in HORIZONS:
        cols = [f"forecast_pre_anchor_h{h}", f"nws_h{h}", f"realized_h{h}"]
        if df_log.empty or any(col not in df_log.columns for col in cols):
            diagnostics[f"h{h}"] = {
                "enabled": False,
                "reason": "paired_history_missing",
                "n_paired": 0,
            }
            continue

        paired = df_log[cols].apply(pd.to_numeric, errors="coerce").dropna()
        paired = paired.tail(NWS_ANCHOR_LOOKBACK_ROWS)
        n_paired = int(len(paired))
        if n_paired < NWS_ANCHOR_MIN_REALIZED_SAMPLES:
            diagnostics[f"h{h}"] = {
                "enabled": False,
                "reason": "insufficient_paired_history",
                "n_paired": n_paired,
                "min_samples": int(NWS_ANCHOR_MIN_REALIZED_SAMPLES),
            }
            continue

        model_mae = float((paired[cols[0]] - paired[cols[2]]).abs().mean())
        nws_mae = float((paired[cols[1]] - paired[cols[2]]).abs().mean())
        improvement = float(model_mae - nws_mae)
        enabled = bool(improvement >= NWS_ANCHOR_MIN_MAE_IMPROVEMENT_F)
        allowed[h] = enabled
        diagnostics[f"h{h}"] = {
            "enabled": enabled,
            "reason": "nws_better_by_required_margin" if enabled else "nws_not_better",
            "n_paired": n_paired,
            "lookback_rows": int(NWS_ANCHOR_LOOKBACK_ROWS),
            "model_pre_anchor_mae_f": model_mae,
            "nws_mae_f": nws_mae,
            "nws_improvement_f": improvement,
            "required_improvement_f": float(NWS_ANCHOR_MIN_MAE_IMPROVEMENT_F),
        }

    return allowed, diagnostics




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
# Validation-tuned XGB/LSTM blend weights
# ---------------------------------------------------------------------------
def _load_lstm_val_predictions() -> pd.DataFrame:
    """Load Part 2 validation predictions used to tune blend weights.

    Part 2 sequence models cannot predict the first SEQUENCE_LEN - 1 validation
    rows, so this table is usually shorter than the XGB validation table.
    The tuner therefore merges on date and uses only common rows.
    """
    path = PART2_DIR / "val_predictions.parquet"
    if not path.exists():
        print("[Part 2B] val_predictions.parquet not found — using default blend weights.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        print("[Part 2B] val_predictions.parquet has no date column — using default blend weights.")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _blend_weight_for_h(blend_weights_xgb: Optional[Dict[str, float]], h: int) -> float:
    if not blend_weights_xgb:
        return float(DEFAULT_BLEND_WEIGHT_XGB)
    val = blend_weights_xgb.get(f"h{h}", DEFAULT_BLEND_WEIGHT_XGB)
    try:
        val = float(val)
    except Exception:
        val = DEFAULT_BLEND_WEIGHT_XGB
    if not np.isfinite(val):
        val = DEFAULT_BLEND_WEIGHT_XGB
    return float(np.clip(val, 0.0, 1.0))


def tune_blend_weights(df_val: pd.DataFrame, val_preds: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    """Tune horizon-specific XGB blend weights using validation MAE only.

    The old production rule used a fixed 0.40 XGB / 0.60 LSTM blend for every
    horizon. Current artifacts show XGB materially outperforming LSTM on H=1
    and H=3, so the fixed rule can degrade the published canonical forecast.
    This tuner selects a coarse-grid XGB weight per horizon on validation rows
    where both XGB and LSTM predictions are available. Test rows are never used.
    """
    defaults = {f"h{h}": float(DEFAULT_BLEND_WEIGHT_XGB) for h in HORIZONS}
    diagnostics: Dict[str, Dict[str, object]] = {}

    lstm_val = _load_lstm_val_predictions()
    if lstm_val.empty:
        for h in HORIZONS:
            diagnostics[f"h{h}"] = {
                "chosen_weight_xgb": float(DEFAULT_BLEND_WEIGHT_XGB),
                "reason": "lstm_val_predictions_missing",
            }
        return defaults, diagnostics

    xgb_val = df_val[["date"]].copy().reset_index(drop=True)
    for h in HORIZONS:
        xgb_val[f"xgb_pred_h{h}"] = np.asarray(val_preds.get(f"h{h}", np.nan), dtype=float)
        xgb_val[f"true_h{h}"] = pd.to_numeric(df_val[f"target_h{h}"].values, errors="coerce")

    merged = xgb_val.merge(lstm_val, on="date", how="inner", suffixes=("_xgb", "_lstm"))
    if merged.empty:
        print("[Part 2B] No common XGB/LSTM validation dates — using default blend weights.")
        for h in HORIZONS:
            diagnostics[f"h{h}"] = {
                "chosen_weight_xgb": float(DEFAULT_BLEND_WEIGHT_XGB),
                "reason": "no_common_validation_dates",
            }
        return defaults, diagnostics

    weights = defaults.copy()
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, BLEND_WEIGHT_GRID_STEP), 4)

    for h in HORIZONS:
        x_col = f"xgb_pred_h{h}"
        l_col = f"pred_h{h}"
        y_col = f"true_h{h}_xgb" if f"true_h{h}_xgb" in merged.columns else f"true_h{h}"
        if x_col not in merged.columns or l_col not in merged.columns or y_col not in merged.columns:
            diagnostics[f"h{h}"] = {
                "chosen_weight_xgb": float(DEFAULT_BLEND_WEIGHT_XGB),
                "reason": "required_columns_missing",
            }
            continue

        x = pd.to_numeric(merged[x_col], errors="coerce").to_numpy(dtype=float)
        l = pd.to_numeric(merged[l_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(merged[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(l) & np.isfinite(y)
        n = int(mask.sum())
        if n < MIN_BLEND_TUNE_ROWS:
            diagnostics[f"h{h}"] = {
                "chosen_weight_xgb": float(DEFAULT_BLEND_WEIGHT_XGB),
                "reason": f"insufficient_common_rows:{n}",
                "n_common_rows": n,
            }
            continue

        x = x[mask]
        l = l[mask]
        y = y[mask]
        mae_by_weight = {}
        for w in grid:
            pred = float(w) * x + (1.0 - float(w)) * l
            mae_by_weight[float(w)] = float(np.mean(np.abs(pred - y)))

        default_w = float(DEFAULT_BLEND_WEIGHT_XGB)
        default_mae = mae_by_weight.get(default_w)
        if default_mae is None:
            default_pred = default_w * x + (1.0 - default_w) * l
            default_mae = float(np.mean(np.abs(default_pred - y)))

        best_w = min(mae_by_weight, key=mae_by_weight.get)
        best_mae = mae_by_weight[best_w]
        improvement = float(default_mae - best_mae)

        if improvement >= MIN_BLEND_IMPROVEMENT_F:
            chosen_w = float(best_w)
            reason = "validation_mae_tuned"
        else:
            chosen_w = default_w
            reason = "default_retained_small_improvement"

        weights[f"h{h}"] = chosen_w
        diagnostics[f"h{h}"] = {
            "chosen_weight_xgb": chosen_w,
            "best_weight_xgb": float(best_w),
            "default_weight_xgb": default_w,
            "best_mae_f": float(best_mae),
            "default_mae_f": float(default_mae),
            "improvement_vs_default_f": improvement,
            "xgb_only_mae_f": float(mae_by_weight.get(1.0, np.nan)),
            "lstm_only_mae_f": float(mae_by_weight.get(0.0, np.nan)),
            "n_common_rows": n,
            "reason": reason,
        }

    print("[Part 2B] Validation-tuned blend weights:")
    for h in HORIZONS:
        d = diagnostics.get(f"h{h}", {})
        print(f"  H={h}: wxgb={weights[f'h{h}']:.2f} ({d.get('reason', 'unknown')})")

    return weights, diagnostics


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


def _apply_nws_anchor(value: float, nws_value: Optional[float]) -> Tuple[float, str]:
    """Conservatively anchor model forecasts to NWS when divergence is large.

    Three-tier system (values are the module-level constants, not restated
    here so this docstring cannot go stale again):
      dev > NWS_SOFT_DEVIATION_F    -> soft partial anchor (NWS_SOFT_ANCHOR_WEIGHT)
      dev > NWS_STRONG_DEVIATION_F  -> stronger partial anchor (NWS_STRONG_ANCHOR_WEIGHT)
      dev > NWS_HARD_DEVIATION_F    -> NWS becomes the horizon forecast outright
    This directly guards the observed cold-bias / heat-event blind spot.
    """
    if nws_value is None or not np.isfinite(nws_value) or not np.isfinite(value):
        return value, ""

    nws_value = float(nws_value)
    dev = abs(float(value) - nws_value)
    adjusted = float(value)
    tag = ""

    if dev > NWS_HARD_DEVIATION_F:
        adjusted = nws_value
        tag = f"nws_hard_anchor(dev={dev:.1f})"
    elif dev > NWS_STRONG_DEVIATION_F:
        adjusted = NWS_STRONG_ANCHOR_WEIGHT * nws_value + (1.0 - NWS_STRONG_ANCHOR_WEIGHT) * float(value)
        tag = f"nws_strong_anchor(dev={dev:.1f})"
    elif dev > NWS_SOFT_DEVIATION_F:
        adjusted = NWS_SOFT_ANCHOR_WEIGHT * nws_value + (1.0 - NWS_SOFT_ANCHOR_WEIGHT) * float(value)
        tag = f"nws_soft_anchor(dev={dev:.1f})"

    # Warm-day guard: if NWS is warm but the model stack remains too cold,
    # cap the cold gap against NWS.  This is intentionally expressed as
    # NWS-minus-gap instead of an absolute floor so May marine-layer days do
    # not get forced to an unrealistic fixed temperature.
    if nws_value >= WARM_EVENT_THRESHOLD_F:
        min_warm_adjusted = nws_value - MAX_WARM_NWS_COLD_GAP_F
        if adjusted < min_warm_adjusted:
            before = adjusted
            adjusted = min_warm_adjusted
            warm_tag = (
                f"warm_gap_guard(nws={nws_value:.1f},"
                f"max_cold_gap={MAX_WARM_NWS_COLD_GAP_F:.1f},before={before:.1f})"
            )
            tag = f"{tag}+{warm_tag}" if tag else warm_tag

    # Heat-event guard: if NWS indicates a heat day but the model is capped well
    # below the threshold, lift the forecast to at least a near-threshold value.
    if nws_value >= HEAT_EVENT_THRESHOLD_F and adjusted < HEAT_EVENT_MIN_MODEL_F:
        before = adjusted
        adjusted = min(nws_value, max(HEAT_EVENT_MIN_MODEL_F, adjusted))
        heat_tag = f"heat_guard(nws={nws_value:.1f},before={before:.1f})"
        tag = f"{tag}+{heat_tag}" if tag else heat_tag

    return float(adjusted), tag


def _finish_candidate(value: float, source: str, reason: str, h: int, nws_preds: Dict[int, Optional[float]]) -> Tuple[float, str, str]:
    """Apply NWS/heat-event anchoring to a selected model candidate."""
    adjusted, anchor_tag = _apply_nws_anchor(value, nws_preds.get(h))
    if anchor_tag:
        source = f"{source}+nws_anchor"
        reason = f"{reason},{anchor_tag}"
    return adjusted, source, reason


def _select_pre_anchor_candidate(
    h: int,
    xgb_preds: Dict[str, float],
    lstm_preds: Dict[str, float],
    nws_preds: Dict[int, Optional[float]],
    last_obs: Optional[float],
    blend_weights_xgb: Optional[Dict[str, float]] = None,
    learned_allowed: bool = True,
) -> Tuple[float, str, str]:
    """Select the exact fallback candidate before any NWS anchoring overlay."""
    key = f"h{h}"
    xgb_val = xgb_preds.get(key)
    lstm_val = lstm_preds.get(key)
    last_obs_f = float(last_obs) if last_obs is not None else float("nan")

    if not learned_allowed:
        if np.isfinite(last_obs_f):
            return last_obs_f, "persistence", f"H{h}:live_skill_gate_persistence"
        nws_val = nws_preds.get(h)
        if nws_val is not None and np.isfinite(nws_val):
            return float(nws_val), "nws", f"H{h}:live_skill_gate_nws_fallback"
        return float("nan"), "unavailable", f"H{h}:live_skill_gate_no_baseline"

    xgb_ok = bool(
        xgb_val is not None
        and np.isfinite(xgb_val)
        and (not np.isfinite(last_obs_f) or _is_plausible(float(xgb_val), last_obs_f))
    )
    lstm_ok = bool(
        lstm_val is not None
        and np.isfinite(lstm_val)
        and (not np.isfinite(last_obs_f) or _is_plausible(float(lstm_val), last_obs_f))
    )

    if xgb_ok and lstm_ok:
        w_xgb = _blend_weight_for_h(blend_weights_xgb, h)
        value = w_xgb * float(xgb_val) + (1.0 - w_xgb) * float(lstm_val)
        return value, "blend", f"H{h}:blend(wxgb={w_xgb:.2f},xgb={xgb_val:.1f},lstm={lstm_val:.1f})"

    if xgb_ok:
        suffix = (
            f"(lstm_implausible:{lstm_val:.1f})"
            if lstm_val is not None and np.isfinite(lstm_val)
            else ""
        )
        return float(xgb_val), "xgb", f"H{h}:xgb_only{suffix}"

    # Preserve the required Part 2 model as the next independent fallback.
    if lstm_ok:
        suffix = (
            f"(xgb_implausible:{xgb_val:.1f})"
            if xgb_val is not None and np.isfinite(xgb_val)
            else ""
        )
        return float(lstm_val), "lstm", f"H{h}:lstm_only{suffix}"

    nws_val = nws_preds.get(h)
    if nws_val is not None and np.isfinite(nws_val):
        return float(nws_val), "nws", f"H{h}:nws_fallback"

    if np.isfinite(last_obs_f):
        return last_obs_f, "persistence", f"H{h}:persistence_fallback"

    return float("nan"), "unavailable", f"H{h}:no_forecast_available"


def compute_canonical_forecast(
    xgb_preds: Dict[str, float],
    lstm_preds: Dict[str, float],
    nws_preds: Dict[int, Optional[float]],
    last_obs: Optional[float],
    blend_weights_xgb: Optional[Dict[str, float]] = None,
    nws_anchor_allowed: Optional[Dict[int, bool]] = None,
    model_skill_allowed: Optional[Dict[int, bool]] = None,
) -> Tuple[Dict[str, float], str, str]:
    """Apply the fallback chain and return (forecast, source, reason).

    Learned candidates pass a horizon-specific live-skill gate first. The
    canonical fallback is persistence when the paired block-bootstrap interval
    does not establish a learned-model improvement. NWS anchoring is a separate
    realized-skill overlay.
    """
    forecast: Dict[str, float] = {}
    sources: List[str] = []
    reasons: List[str] = []

    for h in HORIZONS:
        key = f"h{h}"
        selected_val, selected_source, selected_reason = _select_pre_anchor_candidate(
            h,
            xgb_preds,
            lstm_preds,
            nws_preds,
            last_obs,
            blend_weights_xgb,
            learned_allowed=bool((model_skill_allowed or {}).get(h, True)),
        )

        # Apply the overlay only when paired live evidence authorizes it.
        anchor_allowed = bool((nws_anchor_allowed or {}).get(h, False))
        if selected_source in {"blend", "xgb", "lstm"} and anchor_allowed:
            selected_val, selected_source, selected_reason = _finish_candidate(
                selected_val, selected_source, selected_reason, h, nws_preds
            )

        forecast[key] = float(selected_val)
        sources.append(selected_source)
        reasons.append(selected_reason)

    unique_sources = list(dict.fromkeys(sources))
    if any("nws_anchor" in src for src in unique_sources):
        base_sources = list(dict.fromkeys(src.replace("+nws_anchor", "") for src in sources))
        source_label = "+".join(base_sources + ["nws_anchor"])
    else:
        source_label = "+".join(unique_sources) if unique_sources else "unavailable"
    return forecast, source_label, " | ".join(reasons)


def build_anchor_audit_fields(
    forecast: Dict[str, float],
    xgb_preds: Dict[str, float],
    lstm_preds: Dict[str, float],
    nws_preds: Dict[int, Optional[float]],
    reason: str,
    blend_weights_xgb: Optional[Dict[str, float]] = None,
    last_obs: Optional[float] = None,
    nws_anchor_allowed: Optional[Dict[int, bool]] = None,
    model_skill_allowed: Optional[Dict[int, bool]] = None,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    """Create transparent NWS-anchor audit columns.

    The canonical forecast may be partially pulled toward NWS. These fields
    preserve the model-only pre-anchor value, the NWS comparator, and the
    adjustment applied by horizon so attribution can distinguish independent
    model skill from NWS-anchored forecasts.
    """
    flat: Dict[str, object] = {}
    details: Dict[str, Dict[str, object]] = {}
    any_anchor = False

    for h in HORIZONS:
        key = f"h{h}"
        nws_val = pd.to_numeric(
            pd.Series([nws_preds.get(h, np.nan)]), errors="coerce"
        ).iloc[0]
        selected, selected_source, _ = _select_pre_anchor_candidate(
            h,
            xgb_preds,
            lstm_preds,
            nws_preds,
            last_obs,
            blend_weights_xgb,
            learned_allowed=bool((model_skill_allowed or {}).get(h, True)),
        )
        candidate, candidate_source, _ = _select_pre_anchor_candidate(
            h, xgb_preds, lstm_preds, nws_preds, last_obs,
            blend_weights_xgb, learned_allowed=True
        )
        if selected_source in {"blend", "xgb", "lstm"}:
            pre_anchor = float(selected)
            pre_source = selected_source
        else:
            pre_anchor = float("nan")
            pre_source = "unavailable"

        final_val = float(forecast.get(key, np.nan))
        h_reason = next((part for part in str(reason).split(" | ") if part.startswith(f"H{h}:")), "")
        anchor_applied = bool(
            ("nws_anchor" in h_reason)
            or ("heat_guard" in h_reason)
            or (("nws_" in h_reason) and ("anchor" in h_reason))
        )
        if np.isfinite(pre_anchor) and np.isfinite(final_val) and abs(final_val - pre_anchor) > 1e-6:
            anchor_applied = anchor_applied or bool(np.isfinite(nws_val))
        any_anchor = any_anchor or anchor_applied

        delta = final_val - pre_anchor if np.isfinite(final_val) and np.isfinite(pre_anchor) else float("nan")
        details[key] = {
            "pre_anchor_source": pre_source,
            "candidate_source": candidate_source,
            "candidate_f": (
                float(candidate)
                if candidate_source in {"blend", "xgb", "lstm"} and np.isfinite(candidate)
                else None
            ),
            "blend_weight_xgb": _blend_weight_for_h(blend_weights_xgb, h) if pre_source == "blend" else None,
            "pre_anchor_f": float(pre_anchor) if np.isfinite(pre_anchor) else None,
            "nws_f": float(nws_val) if np.isfinite(nws_val) else None,
            "final_f": float(final_val) if np.isfinite(final_val) else None,
            "anchor_applied": bool(anchor_applied),
            "anchor_delta_f": float(delta) if np.isfinite(delta) else None,
            "reason": h_reason,
            "anchor_allowed_by_skill_gate": bool((nws_anchor_allowed or {}).get(h, False)),
        }

        flat[f"forecast_pre_anchor_h{h}"] = details[key]["pre_anchor_f"]
        flat[f"forecast_candidate_h{h}"] = details[key]["candidate_f"]
        flat[f"forecast_candidate_source_h{h}"] = details[key]["candidate_source"]
        flat[f"nws_h{h}"] = details[key]["nws_f"]
        flat[f"nws_anchor_applied_h{h}"] = bool(anchor_applied)
        flat[f"nws_anchor_delta_h{h}"] = details[key]["anchor_delta_f"]
        flat[f"nws_anchor_allowed_h{h}"] = details[key]["anchor_allowed_by_skill_gate"]

    flat["nws_anchor_used"] = bool(any_anchor)
    return flat, details


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

    # Keep the full feature matrix for live inference, and purge the final
    # five calendar days from train and validation to prevent target overlap.
    target_cols = [f"target_h{h}" for h in HORIZONS]
    df_train, df_val, df_test = purged_labeled_splits(df, splits, target_cols)

    X_tr = _clean(df_train, feature_cols)
    X_va = _clean(df_val, feature_cols)
    X_te = _clean(df_test, feature_cols)

    # Keep full feature matrix (including live tail) only for live prediction.
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
    test_preds: Dict = {}   # Issue 5 fix: declare test_preds
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
            test_preds[f"h{h}"] = tp   # Issue 5 fix: store test predictions
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
    gate, gate_diagnostics = evaluate_xgb_validation_gate(val_metrics, pers_maes)
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
            bnn_rec = bool(
                gate and (lstm_val_mae - xgb_avg) > BNN_RECOMMENDATION_THRESHOLD_F
            )
            print(f"  LSTM val MAE={lstm_val_mae:.2f}°F  XGB avg={xgb_avg:.2f}°F  "
                  f"BNN recommended={bnn_rec}")

    # Tune horizon-specific XGB/LSTM blend weights using validation rows only.
    # This prevents the canonical forecast from using a stale fixed 40/60 blend
    # when one sleeve is materially stronger on recent validation data.
    blend_weights_xgb, blend_weight_diagnostics = tune_blend_weights(df_val, val_preds)

    # Save val + test predictions parquet (Issue 5 fix: include test split)
    val_df = df_val[["date"]].copy().reset_index(drop=True)
    for h in HORIZONS:
        val_df[f"xgb_pred_h{h}"] = val_preds[f"h{h}"]
        val_df[f"true_h{h}"] = df_val[f"target_h{h}"].values
    val_df["split"] = "val"

    if len(df_test) > 0 and test_preds:
        test_df = df_test[["date"]].copy().reset_index(drop=True)
        for h in HORIZONS:
            test_df[f"xgb_pred_h{h}"] = test_preds.get(f"h{h}", np.full(len(df_test), np.nan))
            test_df[f"true_h{h}"] = df_test[f"target_h{h}"].values
        test_df["split"] = "test"
        all_xgb_df = pd.concat([val_df, test_df], ignore_index=True)
    else:
        all_xgb_df = val_df

    all_xgb_df.to_parquet(ARTIFACTS_DIR / "xgb_predictions.parquet", index=False)

    # -------------------------------------------------------------------
    # Canonical forecast fallback chain
    # -------------------------------------------------------------------
    feature_date = pd.Timestamp(df["date"].max()).normalize()
    decision_date = pacific_today_timestamp()

    # Determine which row Part 2 actually wrote. Part 2 keys its prediction_log
    # row by model=model_type.upper() ("LSTM" or "TRANSFORMER" depending on the
    # --model flag Part 2 was run with). This MUST match that value or
    # upsert_log_columns' exact-match lookup finds no row and silently skips
    # writing forecast_h*/forecast_source (upsert_log_columns never creates
    # rows) -- which then makes Part 3's FORECAST_SOURCE_CHECK HOLD the whole
    # pipeline. Read the live value from part2_meta.json rather than assuming
    # LSTM, since --model transformer is a supported, documented run mode.
    p2_meta_path_for_key = PART2_DIR / "part2_meta.json"
    model_key = "LSTM"
    if p2_meta_path_for_key.exists():
        try:
            with open(p2_meta_path_for_key) as _f:
                model_key = str(json.load(_f).get("model_type", "lstm")).upper().strip() or "LSTM"
        except Exception as exc:
            print(f"[Part 2B] WARNING: could not read model_type from part2_meta.json "
                  f"({exc}); defaulting model_key='LSTM'.")

    # Always define key strings before any conditional block (Issue 1 fix)
    dd_str = decision_date.strftime("%Y-%m-%d")
    fd_str = feature_date.strftime("%Y-%m-%d")

    # Load the base sequence-model's live preds from the log row Part 2 wrote
    # for this decision_date/feature_date/model_key (LSTM or TRANSFORMER).
    log = load_prediction_log()
    lstm_live: Dict[str, float] = {}
    if not log.empty:
        # Find the most recent row for this decision_date + feature_date
        mask = (
            log["decision_date"].astype(str).str.strip() == dd_str
        )
        if "feature_date" in log.columns:
            mask = mask & (log["feature_date"].astype(str).str.strip() == fd_str)
        if "model" in log.columns:
            mask = mask & (log["model"].astype(str).str.strip().str.upper() == model_key)
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
    nws_anchor_allowed, nws_anchor_skill = evaluate_nws_anchor_skill(log)
    model_skill_allowed, model_skill_diagnostics = evaluate_live_model_skill(log)
    governed_xgb_live = select_gate_approved_xgb_predictions(
        xgb_live, gate_diagnostics
    )
    blocked_xgb_horizons = [
        h for h in HORIZONS if f"h{h}" not in governed_xgb_live
    ]
    if blocked_xgb_horizons:
        print(
            f"[Part 2B] XGB validation gate blocked H={blocked_xgb_horizons}; "
            "approved horizons remain eligible for the canonical chain."
        )
    print(f"[Part 2B] NWS anchor allowed by horizon: {nws_anchor_allowed}")

    forecast, source, reason = compute_canonical_forecast(
        governed_xgb_live,
        lstm_live,
        nws_preds,
        last_obs,
        blend_weights_xgb,
        nws_anchor_allowed,
        model_skill_allowed=model_skill_allowed,
    )
    anchor_log_fields, anchor_details = build_anchor_audit_fields(
        forecast,
        governed_xgb_live,
        lstm_live,
        nws_preds,
        reason,
        blend_weights_xgb,
        last_obs,
        nws_anchor_allowed,
        model_skill_allowed,
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
    updates.update(anchor_log_fields)
    for h in HORIZONS:
        updates[f"xgb_h{h}"] = xgb_live.get(f"h{h}", np.nan)
        updates[f"forecast_h{h}"] = forecast.get(f"h{h}", np.nan)
        # Store row-level NWS forecast value for this target date.
        # Part 9 uses these to compute NWS baseline accuracy without
        # relying on the current nws_official_forecast.json (which is
        # overwritten on every daily run and loses historical NWS data).
        updates[f"nws_h{h}"] = nws_preds.get(h, np.nan)
        if f"h{h}" in lstm_live:
            w_xgb = _blend_weight_for_h(blend_weights_xgb, h)
            blend = w_xgb * xgb_live.get(f"h{h}", 0) + \
                    (1.0 - w_xgb) * lstm_live[f"h{h}"]
            updates[f"blend_h{h}"] = blend
            updates[f"blend_weight_xgb_h{h}"] = w_xgb

    upsert_log_columns(updates, dd_str, fd_str, model_key)

    # Save summary
    summary = {
        "schema_version": SCHEMA_VERSION,
        "built_at": pd.Timestamp.now().isoformat(),
        "gate_validation_passed": gate,
        "gate_validation_diagnostics": gate_diagnostics,
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
        "blend_weights_xgb": blend_weights_xgb,
        "blend_weight_diagnostics": blend_weight_diagnostics,
        "blend_weight_policy": {
            "default_weight_xgb": DEFAULT_BLEND_WEIGHT_XGB,
            "grid_step": BLEND_WEIGHT_GRID_STEP,
            "min_tune_rows": MIN_BLEND_TUNE_ROWS,
            "min_improvement_f": MIN_BLEND_IMPROVEMENT_F,
            "tuning_split": "validation_only_common_xgb_lstm_dates",
        },
        "nws_anchor_used": bool(anchor_log_fields.get("nws_anchor_used", False)),
        "nws_anchor_details": anchor_details,
        "nws_anchor_skill_gate": nws_anchor_skill,
        "live_model_skill_gate": model_skill_diagnostics,
        "feature_importances": feat_importances,
        "hyperparameters": XGB_PARAMS,
        "nws_anchor_policy": {
            "soft_deviation_f": NWS_SOFT_DEVIATION_F,
            "strong_deviation_f": NWS_STRONG_DEVIATION_F,
            "hard_deviation_f": NWS_HARD_DEVIATION_F,
            "soft_anchor_weight": NWS_SOFT_ANCHOR_WEIGHT,
            "strong_anchor_weight": NWS_STRONG_ANCHOR_WEIGHT,
            "warm_event_threshold_f": WARM_EVENT_THRESHOLD_F,
            "max_warm_nws_cold_gap_f": MAX_WARM_NWS_COLD_GAP_F,
            "heat_event_threshold_f": HEAT_EVENT_THRESHOLD_F,
            "heat_event_min_model_f": HEAT_EVENT_MIN_MODEL_F,
            "min_realized_samples": NWS_ANCHOR_MIN_REALIZED_SAMPLES,
            "lookback_rows": NWS_ANCHOR_LOOKBACK_ROWS,
            "min_mae_improvement_f": NWS_ANCHOR_MIN_MAE_IMPROVEMENT_F,
        },
    }
    with open(ARTIFACTS_DIR / "part2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Part 2B] Saved part2b_summary.json")
    print(f"\n[Part 2B] ✅  Complete. Gate={gate}  Source={source}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
