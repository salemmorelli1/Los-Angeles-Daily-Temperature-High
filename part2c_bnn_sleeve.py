#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2C — Bayesian Neural Network Uncertainty Sleeve
======================================================
Wraps the trained LSTM backbone with Monte Carlo Dropout to produce uncertainty
estimates for each horizon.

Activation Gate
---------------
Runs when Part 2B reports gate_validation_passed=true and Part 2 used the
LSTM backbone. The bnn_sleeve_recommended flag is recorded by Part 2B, but it
does not block uncertainty/display interval generation. Part 2C is an
uncertainty sleeve, not a model-selection reward for XGB improvement.

Safety contract
---------------
  - Non-blocking by design.
  - Skips gracefully when Part 2 used the Transformer backbone.
  - Uses the same feature_date + H target clock as Part 2.
  - Converts scaled standard deviations back to Fahrenheit correctly.
  - Uses split-conformal intervals with an independent validation evaluation half.
  - Writes diagnostic BNN means as bnn_diagnostic_mean_h*.
  - Labels bnn_predictions.parquet rows as val_cal, val_eval, or test.

Important interval terminology
------------------------------
If live intervals are centered on the raw LSTM/BNN mean, the interval label is
``conformal_calibrated`` when calibration passes. If live intervals are centered
on a canonical forecast that may include XGB blending or NWS anchoring, the
interval label is ``canonical_display_interval`` because that center is no
longer the same model used to fit the conformal residual quantile.
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
PART2B_DIR = PROJECT_DIR / "artifacts_part2b"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part2c"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.2.0"
HORIZONS = [1, 3, 5]
SEQUENCE_LEN = 14
N_MC_SAMPLES = 200
CI_LOWER = 5.0
CI_UPPER = 95.0
CONFORMAL_ALPHA = 0.10  # 90% split-conformal interval target
MIN_CONFORMAL_COVERAGE = 0.85
MIN_TEST_CONFORMAL_COVERAGE = 0.80
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20


# ---------------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------------
def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("PyTorch is required for Part 2C. Install with: pip install torch") from exc


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------
def check_bnn_gate() -> bool:
    path = PART2B_DIR / "part2b_summary.json"
    if not path.exists():
        print("[Part 2C] part2b_summary.json not found. Run Part 2B first.")
        return False

    with open(path) as f:
        summary = json.load(f)

    if not summary.get("gate_validation_passed", False):
        print("[Part 2C] Part 2B gate_validation_passed=False. Skipping BNN sleeve.")
        return False

    if not summary.get("bnn_sleeve_recommended", False):
        print(
            "[Part 2C] bnn_sleeve_recommended=False from Part 2B. "
            "Continuing because Part 2C produces uncertainty/display intervals "
            "for the LSTM/canonical forecast path."
        )

    meta_path = PART2_DIR / "part2_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("model_type", "lstm") != "lstm":
            print("[Part 2C] Part 2 model_type is not LSTM. MC-dropout sleeve is skipped.")
            return False

    return True


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def _build_mc_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    n_outputs: int,
):
    """Rebuild the LSTM architecture with dropout available at inference."""
    torch, nn, _, _ = _try_import_torch()

    class MCDropoutLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ])

        def forward(self, x):
            out, _ = self.lstm(x)
            h = self.dropout(out[:, -1, :])
            h = self.bn(h)
            return torch.cat([head(h) for head in self.heads], dim=1)

        def enable_dropout(self):
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

    return MCDropoutLSTM()


def load_lstm_artifacts():
    torch, _, _, _ = _try_import_torch()

    feat_scaler_path = PART2_DIR / "feature_scaler.pkl"
    tgt_scaler_path = PART2_DIR / "target_scaler.pkl"
    model_path = PART2_DIR / "lstm_model.pt"
    meta_path = PART2_DIR / "part2_meta.json"

    for p in [feat_scaler_path, tgt_scaler_path, model_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Part 2 first.")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("model_type", "lstm") != "lstm":
        raise RuntimeError("Part 2C only supports LSTM artifacts. Transformer run detected.")

    with open(feat_scaler_path, "rb") as f:
        feat_scaler = pickle.load(f)
    with open(tgt_scaler_path, "rb") as f:
        tgt_scaler = pickle.load(f)

    feature_cols = meta["feature_cols"]
    n_features = meta["n_features"]

    model = _build_mc_lstm(n_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, feat_scaler, tgt_scaler, feature_cols, meta


# ---------------------------------------------------------------------------
# Data loading / sequencing
# ---------------------------------------------------------------------------
def load_feature_matrix() -> pd.DataFrame:
    path = PART1_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError("feature_matrix.parquet not found.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def load_splits() -> Dict:
    with open(PART1_DIR / "train_val_test_split.json") as f:
        return json.load(f)


def _target_cols() -> List[str]:
    return [f"target_h{h}" for h in HORIZONS]


def _clean_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)


def build_sequences(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    seq_len: int = SEQUENCE_LEN,
):
    if len(X) < seq_len:
        X_empty = np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        if y is None:
            return X_empty
        return X_empty, np.empty((0, y.shape[1]), dtype=np.float32)

    Xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1:i + 1])
        if y is not None:
            ys.append(y[i])

    Xs = np.array(Xs, dtype=np.float32)
    if y is None:
        return Xs
    return Xs, np.array(ys, dtype=np.float32)


def sequence_dates(dates: pd.Series, seq_len: int = SEQUENCE_LEN) -> pd.Series:
    if len(dates) < seq_len:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(dates).iloc[seq_len - 1:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------
def mc_predict(
    model,
    X_seq: np.ndarray,
    n_samples: int = N_MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch, _, _, _ = _try_import_torch()

    if len(X_seq) == 0:
        empty = np.empty((0, len(HORIZONS)), dtype=np.float32)
        return empty, empty, empty, empty

    # BatchNorm stays in eval mode; only dropout is re-enabled.
    model.eval()
    model.enable_dropout()

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(X_tensor).numpy())

    samples_arr = np.stack(samples, axis=0)
    mean_pred = samples_arr.mean(axis=0)
    std_pred = samples_arr.std(axis=0)
    lower_ci = np.percentile(samples_arr, CI_LOWER, axis=0)
    upper_ci = np.percentile(samples_arr, CI_UPPER, axis=0)

    return mean_pred, std_pred, lower_ci, upper_ci


def scaled_std_to_fahrenheit(std_scaled: np.ndarray, tgt_scaler) -> np.ndarray:
    """For MinMaxScaler, x_scaled = x * scale_ + min_, so std_f = std_scaled / scale_."""
    return std_scaled / tgt_scaler.scale_.reshape(1, -1)


def evaluate_calibration(true_vals: np.ndarray, lower_ci: np.ndarray, upper_ci: np.ndarray) -> Dict[str, float]:
    results: Dict[str, float] = {}
    true_vals = np.asarray(true_vals, dtype=float)
    lower_ci = np.asarray(lower_ci, dtype=float)
    upper_ci = np.asarray(upper_ci, dtype=float)

    if len(true_vals) == 0:
        return results

    for i, h in enumerate(HORIZONS):
        y = true_vals[:, i]
        lo = lower_ci[:, i]
        hi = upper_ci[:, i]
        mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if mask.sum() == 0:
            continue

        in_ci = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
        coverage = float(in_ci.mean())
        mean_width = float((hi[mask] - lo[mask]).mean())
        results[f"h{h}_coverage_90pct"] = coverage
        results[f"h{h}_mean_ci_width_f"] = mean_width
        results[f"h{h}_calibration_error"] = abs(coverage - 0.90)
        results[f"h{h}_n"] = int(mask.sum())

    return results


def conformal_quantiles(
    true_vals: np.ndarray,
    mean_vals: np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
) -> np.ndarray:
    """Finite-sample split-conformal absolute residual quantiles by horizon."""
    true_vals = np.asarray(true_vals, dtype=float)
    mean_vals = np.asarray(mean_vals, dtype=float)
    qs: List[float] = []

    for i, _h in enumerate(HORIZONS):
        resid = np.abs(true_vals[:, i] - mean_vals[:, i])
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            qs.append(float("nan"))
            continue

        n = resid.size
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        qs.append(float(np.sort(resid)[k - 1]))

    return np.asarray(qs, dtype=float)


def apply_conformal_intervals(mean_vals: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q2 = np.asarray(q, dtype=float).reshape(1, -1)
    return mean_vals - q2, mean_vals + q2


def conformal_coverage_pass(cal: Dict[str, float], threshold: float) -> bool:
    if not cal:
        return False
    return all(cal.get(f"h{h}_coverage_90pct", 0.0) >= threshold for h in HORIZONS)


def _make_pred_df(
    dates,
    mean_f: np.ndarray,
    lo_f: np.ndarray,
    hi_f: np.ndarray,
    std_f: np.ndarray,
    true_f: Optional[np.ndarray] = None,
    split: str = "",
) -> pd.DataFrame:
    """Build the persisted BNN diagnostic prediction frame.

    The mean column is intentionally named bnn_diagnostic_mean_h* because the
    live published interval may be centered on the canonical forecast
    (blend/XGB/NWS-anchor), not necessarily on the raw BNN/LSTM mean.
    """
    rows = {
        "date": pd.to_datetime(dates),
        "split": split,
    }

    for i, h in enumerate(HORIZONS):
        rows[f"target_date_h{h}"] = pd.to_datetime(dates) + pd.Timedelta(days=h)
        rows[f"bnn_diagnostic_mean_h{h}"] = mean_f[:, i]
        rows[f"bnn_lo90_h{h}"] = lo_f[:, i]
        rows[f"bnn_hi90_h{h}"] = hi_f[:, i]
        rows[f"bnn_std_h{h}"] = std_f[:, i]

        if true_f is not None:
            rows[f"true_h{h}"] = true_f[:, i]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction-log helpers
# ---------------------------------------------------------------------------
def _prediction_log_path() -> Path:
    return PART2_DIR / "prediction_log.csv"


def load_prediction_log() -> pd.DataFrame:
    path = _prediction_log_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest_log_row_for_feature_date(df_log: pd.DataFrame, feature_date: pd.Timestamp) -> Tuple[Optional[int], Optional[pd.Series]]:
    if df_log.empty:
        return None, None

    fd_str = feature_date.strftime("%Y-%m-%d")
    if "feature_date" in df_log.columns:
        mask = df_log["feature_date"].astype(str).str.strip() == fd_str
        sub = df_log[mask]
        if not sub.empty:
            idx = sub.index[-1]
            return int(idx), df_log.loc[idx]

    idx = df_log.index[-1]
    return int(idx), df_log.loc[idx]


def _canonical_center_from_log(row: Optional[pd.Series]) -> Tuple[Optional[np.ndarray], str, Dict[str, object]]:
    """Return canonical forecast center if forecast_h* values are available."""
    if row is None:
        return None, "lstm_bnn_mean", {"reason": "prediction_log_missing"}

    source = str(row.get("forecast_source", "")).strip()
    vals: List[float] = []
    for h in HORIZONS:
        v = pd.to_numeric(pd.Series([row.get(f"forecast_h{h}", np.nan)]), errors="coerce").iloc[0]
        vals.append(float(v) if np.isfinite(v) else float("nan"))

    arr = np.asarray(vals, dtype=float)
    if np.isfinite(arr).all() and source and "unavailable" not in source.lower():
        return arr, "canonical_forecast", {"forecast_source": source}

    return None, "lstm_bnn_mean", {"reason": "forecast_h_missing_or_unavailable", "forecast_source": source}


def update_prediction_log_with_bnn(
    feature_date: pd.Timestamp,
    live_mean_f: np.ndarray,
    live_std_f: np.ndarray,
    live_center_f: np.ndarray,
    live_lo_f: np.ndarray,
    live_hi_f: np.ndarray,
    cal_pass: bool,
    interval_status: str,
    interval_label: str,
    intervals_publishable: bool,
    intervals_displayable: bool,
) -> None:
    log_path = _prediction_log_path()
    if not log_path.exists():
        print("[Part 2C] prediction_log.csv not found — BNN uncertainty columns not written.")
        return

    df_log = pd.read_csv(log_path)
    if df_log.empty:
        print("[Part 2C] prediction_log.csv is empty — BNN uncertainty columns not written.")
        return

    idx, _row = _latest_log_row_for_feature_date(df_log, feature_date)
    if idx is None:
        print("[Part 2C] No log row found — BNN uncertainty columns not written.")
        return

    df_log.loc[idx, "bnn_available"] = True
    df_log.loc[idx, "bnn_calibrated"] = bool(cal_pass)
    df_log.loc[idx, "bnn_interval_status"] = interval_status
    df_log.loc[idx, "bnn_interval_label"] = interval_label
    df_log.loc[idx, "intervals_publishable"] = bool(intervals_publishable)
    df_log.loc[idx, "intervals_displayable"] = bool(intervals_displayable)

    for i, h in enumerate(HORIZONS):
        df_log.loc[idx, f"bnn_diagnostic_mean_h{h}"] = float(live_mean_f[i])
        df_log.loc[idx, f"bnn_interval_center_h{h}"] = float(live_center_f[i])
        df_log.loc[idx, f"bnn_lo90_h{h}"] = float(live_lo_f[i])
        df_log.loc[idx, f"bnn_hi90_h{h}"] = float(live_hi_f[i])
        df_log.loc[idx, f"bnn_std_h{h}"] = float(live_std_f[i])

    df_log.to_csv(log_path, index=False)
    print("[Part 2C] Updated prediction_log.csv with BNN uncertainty columns")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 2C] Project root: {PROJECT_DIR}")

    if not check_bnn_gate():
        print("[Part 2C] Gate not met. Exiting gracefully (non-blocking).")
        return 0

    print("[Part 2C] Loading LSTM model and scalers from Part 2...")
    model, feat_scaler, tgt_scaler, feature_cols, part2_meta = load_lstm_artifacts()

    df = load_feature_matrix()
    splits = load_splits()
    target_cols = _target_cols()

    labeled = df.dropna(subset=target_cols).copy()
    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])

    df_val = labeled[(labeled["date"] > train_end) & (labeled["date"] <= val_end)].copy()
    df_test = labeled[labeled["date"] > val_end].copy()

    X_val = feat_scaler.transform(_clean_features(df_val, feature_cols)).astype(np.float32)
    y_val = tgt_scaler.transform(df_val[target_cols].values.astype(np.float32)).astype(np.float32)
    X_test = feat_scaler.transform(_clean_features(df_test, feature_cols)).astype(np.float32)
    y_test = tgt_scaler.transform(df_test[target_cols].values.astype(np.float32)).astype(np.float32)
    X_all = feat_scaler.transform(_clean_features(df, feature_cols)).astype(np.float32)

    X_val_seq, y_val_seq = build_sequences(X_val, y_val)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test)
    X_all_seq = build_sequences(X_all)

    # -------------------------------------------------------------------
    # Validation calibration/evaluation
    # -------------------------------------------------------------------
    print(f"[Part 2C] Running MC Dropout ({N_MC_SAMPLES} samples) on validation set...")
    val_mean_s, val_std_s, val_lo_s, val_hi_s = mc_predict(model, X_val_seq)

    val_mean_f = tgt_scaler.inverse_transform(val_mean_s)
    val_lo_raw_f = tgt_scaler.inverse_transform(val_lo_s)
    val_hi_raw_f = tgt_scaler.inverse_transform(val_hi_s)
    val_std_f = scaled_std_to_fahrenheit(val_std_s, tgt_scaler)
    val_true_f = tgt_scaler.inverse_transform(y_val_seq)

    raw_cal = evaluate_calibration(val_true_f, val_lo_raw_f, val_hi_raw_f)

    n_val_seq = len(val_mean_f)
    n_cal = n_val_seq // 2
    if n_val_seq < 20 or n_cal <= 0 or n_cal >= n_val_seq:
        print(
            f"[Part 2C] Not enough validation sequences for split conformal: n={n_val_seq}. "
            "Exiting gracefully."
        )
        return 0

    # Independent split-conformal diagnostic:
    #   val_cal half fits q_eval_f.
    #   val_eval half evaluates q_eval_f without reusing calibration rows.
    # This is an honest diagnostic of calibration stability, not the production
    # interval width used for live forecasts.
    q_eval_f = conformal_quantiles(val_true_f[:n_cal], val_mean_f[:n_cal], alpha=CONFORMAL_ALPHA)
    half_eval_lo_f, half_eval_hi_f = apply_conformal_intervals(val_mean_f[n_cal:], q_eval_f)
    half_split_cal = evaluate_calibration(val_true_f[n_cal:], half_eval_lo_f, half_eval_hi_f)
    half_split_validation_pass = conformal_coverage_pass(half_split_cal, MIN_CONFORMAL_COVERAGE)

    # Production conformal width:
    # Use all validation sequences to fit the deployed conformal radius, then
    # evaluate that deployed width on the independent test split.  The validation
    # coverage below is reported as an in-sample diagnostic, while test coverage
    # is the independent gate for deployability.
    q_live_f = conformal_quantiles(val_true_f, val_mean_f, alpha=CONFORMAL_ALPHA)
    val_prod_lo_f, val_prod_hi_f = apply_conformal_intervals(val_mean_f[n_cal:], q_live_f)
    cal = evaluate_calibration(val_true_f[n_cal:], val_prod_lo_f, val_prod_hi_f)
    validation_calibration_pass = conformal_coverage_pass(cal, MIN_CONFORMAL_COVERAGE)

    # Persisted validation diagnostic intervals should match the deployed/live
    # conformal radius, while the split labels disclose the calibration/eval rows.
    val_lo_conf_f, val_hi_conf_f = apply_conformal_intervals(val_mean_f, q_live_f)

    print("\n=== VALIDATION CALIBRATION DIAGNOSTICS (90% CI) ===")
    for h in HORIZONS:
        cov = cal.get(f"h{h}_coverage_90pct")
        width = cal.get(f"h{h}_mean_ci_width_f")
        n_cov = cal.get(f"h{h}_n")
        hs_cov = half_split_cal.get(f"h{h}_coverage_90pct")
        if cov is not None:
            print(
                f"  H={h}: production-q coverage={cov:.1%}, width={width:.1f}°F, n={n_cov}; "
                f"half-split diagnostic={hs_cov:.1%}" if hs_cov is not None else
                f"  H={h}: production-q coverage={cov:.1%}, width={width:.1f}°F, n={n_cov}"
            )

    # -------------------------------------------------------------------
    # Test set evaluation
    # -------------------------------------------------------------------
    print("\n[Part 2C] Running MC Dropout on test set...")
    test_mean_s, test_std_s, test_lo_s, test_hi_s = mc_predict(model, X_test_seq)
    test_mean_f = tgt_scaler.inverse_transform(test_mean_s)
    test_std_f = scaled_std_to_fahrenheit(test_std_s, tgt_scaler)
    test_true_f = tgt_scaler.inverse_transform(y_test_seq) if len(y_test_seq) else np.empty((0, len(HORIZONS)))

    test_lo_conf_f, test_hi_conf_f = apply_conformal_intervals(test_mean_f, q_live_f)
    test_cal = evaluate_calibration(test_true_f, test_lo_conf_f, test_hi_conf_f) if len(test_true_f) else {}
    test_calibration_pass = conformal_coverage_pass(test_cal, MIN_TEST_CONFORMAL_COVERAGE)

    # The independent test split is the deployability gate.  The validation
    # production-width coverage is retained as a diagnostic because q_live_f is
    # fitted on the full validation set, so the val_eval rows are partly
    # in-sample for that diagnostic.
    cal_pass = bool(test_calibration_pass)
    interval_status = "CONFORMAL_CALIBRATED" if cal_pass else "UNCALIBRATED"

    # -------------------------------------------------------------------
    # Live uncertainty
    # -------------------------------------------------------------------
    print("\n[Part 2C] Running MC Dropout on latest available data...")
    if len(X_all_seq) == 0:
        raise ValueError("Not enough feature rows to build live sequence.")

    live_seq = X_all_seq[-1:, :, :]
    live_mean_s, live_std_s, live_lo_s, live_hi_s = mc_predict(model, live_seq)

    live_mean_f = tgt_scaler.inverse_transform(live_mean_s)[0]
    live_std_f = scaled_std_to_fahrenheit(live_std_s, tgt_scaler)[0]

    feature_date = pd.Timestamp(df["date"].max()).normalize()

    log = load_prediction_log()
    _idx, latest_row = _latest_log_row_for_feature_date(log, feature_date)
    canonical_center, center_source, center_details = _canonical_center_from_log(latest_row)

    if canonical_center is not None:
        live_center_f = canonical_center.astype(float)
        interval_label = "canonical_display_interval"
    else:
        live_center_f = live_mean_f.astype(float)
        interval_label = "conformal_calibrated" if cal_pass else "uncalibrated_lstm_bnn_interval"

    live_lo_f = live_center_f - q_live_f
    live_hi_f = live_center_f + q_live_f
    # Strict publishability is reserved for intervals centered on the same
    # diagnostic BNN/LSTM mean used during conformal calibration.  Canonical
    # display intervals may still be shown as calibrated risk bands, but they
    # are not labeled as strict split-conformal predictive intervals because
    # the live center may include XGB blending or NWS anchoring.
    intervals_publishable = bool(cal_pass and interval_label == "conformal_calibrated")
    intervals_displayable = bool(cal_pass)

    print("\n=== LIVE PREDICTIONS WITH UNCERTAINTY ===")
    print(f"  Interval label: {interval_label}")
    print(f"  Interval status: {interval_status}")
    print(f"  Intervals publishable: {intervals_publishable}")
    print(f"  Intervals displayable: {intervals_displayable}")
    for i, h in enumerate(HORIZONS):
        target_date = feature_date + pd.Timedelta(days=h)
        print(
            f"  H={h} ({target_date.date()}): center={live_center_f[i]:.1f}°F "
            f"[90% interval: {live_lo_f[i]:.1f}°F – {live_hi_f[i]:.1f}°F] "
            f"diagnostic_mean={live_mean_f[i]:.1f}°F std={live_std_f[i]:.2f}°F"
        )

    # -------------------------------------------------------------------
    # Save prediction parquet
    # -------------------------------------------------------------------
    # Validation rows are labeled according to the same split-conformal logic:
    #   val_cal  = first half of validation sequences used to fit the conformal quantile
    #   val_eval = second half of validation sequences used for independent coverage evaluation
    #   test     = independent test sequences
    val_dates = sequence_dates(df_val["date"])
    test_dates = sequence_dates(df_test["date"])

    val_df = _make_pred_df(
        val_dates,
        val_mean_f,
        val_lo_conf_f,
        val_hi_conf_f,
        val_std_f,
        val_true_f,
        split="val",
    )

    n_val_rows = len(val_df)
    n_val_cal = n_val_rows // 2
    if n_val_rows > 0:
        val_df.loc[val_df.index[:n_val_cal], "split"] = "val_cal"
        val_df.loc[val_df.index[n_val_cal:], "split"] = "val_eval"

    test_df = _make_pred_df(
        test_dates,
        test_mean_f,
        test_lo_conf_f,
        test_hi_conf_f,
        test_std_f,
        test_true_f,
        split="test",
    )

    all_df = pd.concat([val_df, test_df], ignore_index=True)
    all_df.to_parquet(ARTIFACTS_DIR / "bnn_predictions.parquet", index=False)

    split_counts = all_df["split"].value_counts(dropna=False).to_dict()
    print(f"\n[Part 2C] Saved bnn_predictions.parquet ({len(all_df)} rows, splits={split_counts})")

    # -------------------------------------------------------------------
    # Update prediction log
    # -------------------------------------------------------------------
    update_prediction_log_with_bnn(
        feature_date=feature_date,
        live_mean_f=live_mean_f,
        live_std_f=live_std_f,
        live_center_f=live_center_f,
        live_lo_f=live_lo_f,
        live_hi_f=live_hi_f,
        cal_pass=cal_pass,
        interval_status=interval_status,
        interval_label=interval_label,
        intervals_publishable=intervals_publishable,
        intervals_displayable=intervals_displayable,
    )

    # -------------------------------------------------------------------
    # Save calibration report and meta
    # -------------------------------------------------------------------
    cal_report = {
        "schema_version": SCHEMA_VERSION,
        "n_mc_samples": N_MC_SAMPLES,
        "ci_lower_pct": CI_LOWER,
        "ci_upper_pct": CI_UPPER,
        "ci_target_coverage": 0.90,
        "interval_method": "split_conformal_full_validation_with_half_split_diagnostic",
        "conformal_alpha": CONFORMAL_ALPHA,
        "min_validation_coverage": MIN_CONFORMAL_COVERAGE,
        "min_test_coverage": MIN_TEST_CONFORMAL_COVERAGE,
        "validation_split": {
            "n_val_sequences": int(n_val_seq),
            "n_val_cal": int(n_cal),
            "n_val_eval": int(n_val_seq - n_cal),
        },
        "conformal_quantile_eval_f_by_horizon": {
            f"h{h}": float(q_eval_f[i]) for i, h in enumerate(HORIZONS)
        },
        "conformal_quantile_live_f_by_horizon": {
            f"h{h}": float(q_live_f[i]) for i, h in enumerate(HORIZONS)
        },
        "raw_mc_dropout_calibration_results": raw_cal,
        "half_split_calibration_results": half_split_cal,
        "half_split_validation_pass": bool(half_split_validation_pass),
        "calibration_results": cal,
        "test_coverage_results": test_cal,
        "validation_calibration_pass": bool(validation_calibration_pass),
        "test_calibration_pass": bool(test_calibration_pass),
        "calibration_pass": bool(cal_pass),
        "interval_status": interval_status,
        "interval_label": interval_label,
        "interval_center_source": center_source,
        "interval_center_details": center_details,
        "intervals_publishable": bool(intervals_publishable),
        "intervals_displayable": bool(intervals_displayable),
        "statistical_note": (
            "half_split_calibration_results are evaluated on the validation evaluation half using only "
            "the first validation half to fit q_eval_f. calibration_results use the deployed full-validation "
            "conformal radius and are therefore a production-width diagnostic; test_coverage_results are the "
            "independent deployability gate used for calibration_pass. If interval_label is "
            "canonical_display_interval, live bounds are centered on the canonical forecast and should be "
            "treated as calibrated display/risk bands rather than strict split-conformal predictive intervals."
        ),
    }
    with open(ARTIFACTS_DIR / "calibration_report.json", "w") as f:
        json.dump(cal_report, f, indent=2, default=str)
    print("[Part 2C] Saved calibration_report.json")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_at": pd.Timestamp.now().isoformat(),
        "part2_model_type": part2_meta.get("model_type", "lstm"),
        "feature_date": feature_date.isoformat(),
        "target_clock": "target_date_h = feature_date + h calendar days",
        "n_mc_samples": N_MC_SAMPLES,
        "sequence_len": SEQUENCE_LEN,
        "dropout_rate": DROPOUT,
        "interval_method": "split_conformal_full_validation_with_half_split_diagnostic",
        "interval_status": interval_status,
        "interval_label": interval_label,
        "interval_center_source": center_source,
        "interval_center_details": center_details,
        "intervals_publishable": bool(intervals_publishable),
        "intervals_displayable": bool(intervals_displayable),
        "validation_split": {
            "n_val_sequences": int(n_val_seq),
            "n_val_cal": int(n_cal),
            "n_val_eval": int(n_val_seq - n_cal),
        },
        "live_predictions": {
            f"h{h}": {
                "target_date": str((feature_date + pd.Timedelta(days=h)).date()),
                "diagnostic_mean_f": float(live_mean_f[i]),
                "interval_center_f": float(live_center_f[i]),
                "lo90_f": float(live_lo_f[i]),
                "hi90_f": float(live_hi_f[i]),
                "std_f": float(live_std_f[i]),
                "interval_label": interval_label,
            }
            for i, h in enumerate(HORIZONS)
        },
        "conformal_quantile_eval_f_by_horizon": {
            f"h{h}": float(q_eval_f[i]) for i, h in enumerate(HORIZONS)
        },
        "conformal_quantile_live_f_by_horizon": {
            f"h{h}": float(q_live_f[i]) for i, h in enumerate(HORIZONS)
        },
        "raw_mc_dropout_calibration_summary": raw_cal,
        "half_split_calibration_summary": half_split_cal,
        "half_split_validation_pass": bool(half_split_validation_pass),
        "calibration_summary": cal,
        "test_coverage_summary": test_cal,
        "validation_calibration_pass": bool(validation_calibration_pass),
        "test_calibration_pass": bool(test_calibration_pass),
        "calibration_pass": bool(cal_pass),
    }
    with open(ARTIFACTS_DIR / "part2c_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("[Part 2C] Saved part2c_meta.json")

    print("\n[Part 2C] ✅ Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


