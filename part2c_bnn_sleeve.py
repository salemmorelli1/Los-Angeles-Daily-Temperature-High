#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2C — Bayesian Neural Network Uncertainty Sleeve
======================================================
Wraps the trained LSTM backbone with Monte Carlo Dropout to produce uncertainty
estimates for each horizon.

Activation Gate
---------------
Only activates when Part 2B reports:
    gate_validation_passed: true
    bnn_sleeve_recommended: true

Safety contract
---------------
  - Non-blocking by design.
  - Skips gracefully when Part 2 used the Transformer backbone.
  - Uses the same feature_date + H target clock as Part 2.
  - Converts scaled standard deviations back to Fahrenheit correctly.
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

SCHEMA_VERSION = "1.1.0"
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
    except ImportError:
        raise ImportError("PyTorch is required for Part 2C. Install with: pip install torch")


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
        print("[Part 2C] bnn_sleeve_recommended=False from Part 2B. Skipping BNN sleeve.")
        return False

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
def _build_mc_lstm(input_size: int, hidden_size: int, num_layers: int,
                   dropout: float, n_outputs: int):
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
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)


def build_sequences(X: np.ndarray, y: Optional[np.ndarray] = None, seq_len: int = SEQUENCE_LEN):
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
    results = {}
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

    return results


def conformal_quantiles(true_vals: np.ndarray, mean_vals: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> np.ndarray:
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
        # Finite-sample conformal quantile. If k exceeds n, use the maximum.
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


def _make_pred_df(dates, mean_f, lo_f, hi_f, std_f, true_f=None) -> pd.DataFrame:
    rows = {"date": pd.to_datetime(dates)}
    for i, h in enumerate(HORIZONS):
        rows[f"target_date_h{h}"] = pd.to_datetime(dates) + pd.Timedelta(days=h)
        rows[f"bnn_mean_h{h}"] = mean_f[:, i]
        rows[f"bnn_lo90_h{h}"] = lo_f[:, i]
        rows[f"bnn_hi90_h{h}"] = hi_f[:, i]
        rows[f"bnn_std_h{h}"] = std_f[:, i]
        if true_f is not None:
            rows[f"true_h{h}"] = true_f[:, i]
    return pd.DataFrame(rows)


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

    # Validation/calibration.
    print(f"[Part 2C] Running MC Dropout ({N_MC_SAMPLES} samples) on validation set...")
    val_mean_s, val_std_s, val_lo_s, val_hi_s = mc_predict(model, X_val_seq)

    val_mean_f = tgt_scaler.inverse_transform(val_mean_s)
    val_lo_f = tgt_scaler.inverse_transform(val_lo_s)
    val_hi_f = tgt_scaler.inverse_transform(val_hi_s)
    val_std_f = scaled_std_to_fahrenheit(val_std_s, tgt_scaler)
    val_true_f = tgt_scaler.inverse_transform(y_val_seq)

    # Raw MC-dropout intervals measure model-weight uncertainty and have been
    # under-covering. We therefore publish split-conformal intervals built from
    # validation residuals around the MC mean.
    raw_cal = evaluate_calibration(val_true_f, val_lo_f, val_hi_f)
    conformal_q_f = conformal_quantiles(val_true_f, val_mean_f, alpha=CONFORMAL_ALPHA)
    val_lo_conf_f, val_hi_conf_f = apply_conformal_intervals(val_mean_f, conformal_q_f)
    cal = evaluate_calibration(val_true_f, val_lo_conf_f, val_hi_conf_f)
    validation_calibration_pass = conformal_coverage_pass(cal, MIN_CONFORMAL_COVERAGE)
    cal_pass = False  # finalized after the independent test-coverage check below
    interval_status = "UNCALIBRATED"

    print("\n=== CALIBRATION (90% CI) ===")
    for h in HORIZONS:
        cov = cal.get(f"h{h}_coverage_90pct")
        width = cal.get(f"h{h}_mean_ci_width_f")
        if cov is not None:
            print(f"  H={h}: coverage={cov:.1%}, mean CI width={width:.1f}°F")

    # Test set.
    print("\n[Part 2C] Running MC Dropout on test set...")
    test_mean_s, test_std_s, test_lo_s, test_hi_s = mc_predict(model, X_test_seq)
    test_mean_f = tgt_scaler.inverse_transform(test_mean_s)
    test_lo_f = tgt_scaler.inverse_transform(test_lo_s)
    test_hi_f = tgt_scaler.inverse_transform(test_hi_s)
    test_std_f = scaled_std_to_fahrenheit(test_std_s, tgt_scaler)
    test_lo_conf_f, test_hi_conf_f = apply_conformal_intervals(test_mean_f, conformal_q_f)
    test_true_f = tgt_scaler.inverse_transform(y_test_seq) if len(y_test_seq) else None
    test_cal = evaluate_calibration(test_true_f, test_lo_conf_f, test_hi_conf_f) if test_true_f is not None else {}
    test_calibration_pass = conformal_coverage_pass(test_cal, MIN_TEST_CONFORMAL_COVERAGE)
    cal_pass = bool(validation_calibration_pass and test_calibration_pass)
    interval_status = "CONFORMAL_CALIBRATED" if cal_pass else "UNCALIBRATED"

    # Live uncertainty.
    print("\n[Part 2C] Running MC Dropout on latest available data...")
    if len(X_all_seq) == 0:
        raise ValueError("Not enough feature rows to build live sequence.")

    live_seq = X_all_seq[-1:, :, :]
    live_mean_s, live_std_s, live_lo_s, live_hi_s = mc_predict(model, live_seq)

    live_mean_f = tgt_scaler.inverse_transform(live_mean_s)[0]
    live_std_f = scaled_std_to_fahrenheit(live_std_s, tgt_scaler)[0]

    # Center publishable intervals on the canonical forecast (blend+NWS-anchor)
    # from Part 2B, not on the raw LSTM mean.  The raw LSTM mean is a diagnostic;
    # the canonical forecast is what governance publishes.  Centering intervals
    # on a different value than the published point forecast is misleading.
    canonical_center_f = live_mean_f.copy()  # fallback: LSTM mean
    canonical_source = "lstm_bnn_mean"

    part2b_summary_path = PART2B_DIR / "part2b_summary.json"
    if part2b_summary_path.exists():
        with open(part2b_summary_path) as _f:
            _summary = json.load(_f)
        _canon = _summary.get("canonical_forecast", {})
        if _canon:
            for i, h in enumerate(HORIZONS):
                _v = _canon.get(f"h{h}")
                if _v is not None and np.isfinite(float(_v)):
                    canonical_center_f[i] = float(_v)
            canonical_source = _summary.get("forecast_source", "part2b_canonical")
            print(f"[Part 2C] Centering live intervals on Part 2B canonical forecast "
                  f"(source={canonical_source})")
    else:
        print("[Part 2C] part2b_summary.json not found — using LSTM mean as interval center.")

    # Publishable intervals: canonical center ± conformal quantile.
    #
    # Statistical note: the conformal quantile is calibrated against LSTM
    # validation residuals (LSTM mean vs realized).  When the canonical center
    # is the LSTM mean, the interval is a valid split-conformal predictive
    # interval.  When the canonical center includes XGB blending or NWS
    # anchoring, the interval is a display approximation — it captures the
    # empirical LSTM error spread but is not a formally calibrated interval for
    # the blended/anchored forecast.  It is labeled "canonical_display_interval"
    # in the meta to make this distinction explicit.
    live_lo_conf_f, live_hi_conf_f = apply_conformal_intervals(
        canonical_center_f.reshape(1, -1), conformal_q_f
    )
    live_lo_conf_f = live_lo_conf_f[0]
    live_hi_conf_f = live_hi_conf_f[0]
    interval_label = (
        "conformal_calibrated"
        if canonical_source == "lstm_bnn_mean"
        else "canonical_display_interval"
    )

    feature_date = pd.Timestamp(df["date"].max()).normalize()
    print("\n=== LIVE PREDICTIONS WITH UNCERTAINTY ===")
    for i, h in enumerate(HORIZONS):
        target_date = feature_date + pd.Timedelta(days=h)
        print(
            f"  H={h} ({target_date.date()}): canonical={canonical_center_f[i]:.1f}°F "
            f"[conformal 90% CI: {live_lo_conf_f[i]:.1f}°F – {live_hi_conf_f[i]:.1f}°F] "
            f"(LSTM diagnostic mean={live_mean_f[i]:.1f}°F ±{live_std_f[i]:.2f}°F)"
        )

    # Save prediction parquet.
    val_dates = sequence_dates(df_val["date"])
    test_dates = sequence_dates(df_test["date"])

    val_df = _make_pred_df(val_dates, val_mean_f, val_lo_conf_f, val_hi_conf_f, val_std_f, val_true_f)
    test_df = _make_pred_df(test_dates, test_mean_f, test_lo_conf_f, test_hi_conf_f, test_std_f, test_true_f)

    all_df = pd.concat([val_df, test_df], ignore_index=True)
    all_df.to_parquet(ARTIFACTS_DIR / "bnn_predictions.parquet", index=False)
    print(f"\n[Part 2C] Saved bnn_predictions.parquet ({len(all_df)} rows)")

    # Update latest prediction row with BNN uncertainty.
    # bnn_lo90_h* / bnn_hi90_h* are canonical-centered (publishable).
    # bnn_diagnostic_mean_h* is the raw LSTM mean (diagnostic only).
    log_path = PART2_DIR / "prediction_log.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        if not df_log.empty:
            df_log.loc[df_log.index[-1], "bnn_available"] = True
            df_log.loc[df_log.index[-1], "bnn_calibrated"] = bool(cal_pass)
            df_log.loc[df_log.index[-1], "bnn_interval_status"] = interval_status
            df_log.loc[df_log.index[-1], "intervals_publishable"] = bool(cal_pass)
            df_log.loc[df_log.index[-1], "bnn_interval_center"] = canonical_source
            df_log.loc[df_log.index[-1], "bnn_interval_label"] = interval_label
            for i, h in enumerate(HORIZONS):
                df_log.loc[df_log.index[-1], f"bnn_lo90_h{h}"] = float(live_lo_conf_f[i])
                df_log.loc[df_log.index[-1], f"bnn_hi90_h{h}"] = float(live_hi_conf_f[i])
                df_log.loc[df_log.index[-1], f"bnn_std_h{h}"] = float(live_std_f[i])
                df_log.loc[df_log.index[-1], f"bnn_diagnostic_mean_h{h}"] = float(live_mean_f[i])
            df_log.to_csv(log_path, index=False)
            print("[Part 2C] Updated prediction_log.csv with BNN uncertainty columns "
                  f"(interval center={canonical_source})")

    cal_report = {
        "schema_version": SCHEMA_VERSION,
        "n_mc_samples": N_MC_SAMPLES,
        "ci_lower_pct": CI_LOWER,
        "ci_upper_pct": CI_UPPER,
        "ci_target_coverage": 0.90,
        "interval_method": "split_conformal_on_validation_residuals",
        "conformal_alpha": CONFORMAL_ALPHA,
        "min_validation_coverage": MIN_CONFORMAL_COVERAGE,
        "min_test_coverage": MIN_TEST_CONFORMAL_COVERAGE,
        "conformal_quantile_f_by_horizon": {f"h{h}": float(conformal_q_f[i]) for i, h in enumerate(HORIZONS)},
        "raw_mc_dropout_calibration_results": raw_cal,
        "calibration_results": cal,
        "test_coverage_results": test_cal,
        "validation_calibration_pass": bool(validation_calibration_pass),
        "test_calibration_pass": bool(test_calibration_pass),
        "calibration_pass": bool(cal_pass),
        "interval_status": interval_status,
        "intervals_publishable": bool(cal_pass),
    }
    with open(ARTIFACTS_DIR / "calibration_report.json", "w") as f:
        json.dump(cal_report, f, indent=2)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_at": pd.Timestamp.now().isoformat(),
        "part2_model_type": part2_meta.get("model_type", "lstm"),
        "feature_date": feature_date.isoformat(),
        "target_clock": "target_date_h = feature_date + h calendar days",
        "n_mc_samples": N_MC_SAMPLES,
        "sequence_len": SEQUENCE_LEN,
        "dropout_rate": DROPOUT,
        "live_predictions": {
            f"h{h}": {
                "target_date": str((feature_date + pd.Timedelta(days=h)).date()),
                "canonical_center_f": float(canonical_center_f[i]),
                "canonical_source": canonical_source,
                "interval_label": interval_label,
                "lo90_f": float(live_lo_conf_f[i]),
                "hi90_f": float(live_hi_conf_f[i]),
                "bnn_diagnostic_mean_f": float(live_mean_f[i]),
                "std_f": float(live_std_f[i]),
            }
            for i, h in enumerate(HORIZONS)
        },
        "interval_method": "split_conformal_on_validation_residuals",
        "conformal_quantile_f_by_horizon": {f"h{h}": float(conformal_q_f[i]) for i, h in enumerate(HORIZONS)},
        "min_validation_coverage": MIN_CONFORMAL_COVERAGE,
        "min_test_coverage": MIN_TEST_CONFORMAL_COVERAGE,
        "raw_mc_dropout_calibration_summary": raw_cal,
        "calibration_summary": cal,
        "test_coverage_summary": test_cal,
        "validation_calibration_pass": bool(validation_calibration_pass),
        "test_calibration_pass": bool(test_calibration_pass),
        "calibration_pass": cal_report["calibration_pass"],
        "interval_status": interval_status,
        "intervals_publishable": bool(cal_pass),
    }
    with open(ARTIFACTS_DIR / "part2c_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[Part 2C] Saved part2c_meta.json")

    print("\n[Part 2C] ✅ Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

