#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2C — Bayesian Neural Network Uncertainty Sleeve
======================================================
Wraps the trained LSTM backbone with Monte Carlo Dropout to produce
calibrated uncertainty estimates (confidence intervals) for each horizon.

Activation Gate
---------------
Only activate this sleeve after Part 2B reports:
    bnn_sleeve_recommended: true

in artifacts_part2b/part2b_summary.json. If the gate is not set, Part 2C
exits gracefully (non-blocking).

Method: MC Dropout (Gal & Ghahramani, 2016)
--------------------------------------------
  - Dropout layers remain ACTIVE at inference time
  - N=200 stochastic forward passes are sampled
  - Mean of samples → point forecast
  - Std of samples → aleatoric + epistemic uncertainty proxy
  - 90% CI = [5th percentile, 95th percentile] of sample distribution

Calibration
-----------
  Empirical coverage check: what fraction of held-out observed highs
  fall within the 90% CI? Good calibration → ~90% coverage.

Artifacts Written
-----------------
  artifacts_part2c/
      bnn_predictions.parquet        — point forecast + CI bounds per date
      calibration_report.json        — coverage per horizon, ECE
      part2c_meta.json               — settings, calibration summary
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

SCHEMA_VERSION = "1.0.0"
HORIZONS = [1, 3, 5]
SEQUENCE_LEN = 14
N_MC_SAMPLES = 200       # Monte Carlo forward passes
CI_LOWER = 5.0           # Percentile for lower CI bound (90% CI)
CI_UPPER = 95.0          # Percentile for upper CI bound
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20           # Must match Part 2 training dropout


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
    recommended = summary.get("bnn_sleeve_recommended", False)
    gate = summary.get("gate_validation_passed", False)
    if not gate:
        print("[Part 2C] Part 2B gate_validation_passed=False. Skipping BNN sleeve.")
        return False
    if not recommended:
        print("[Part 2C] bnn_sleeve_recommended=False from Part 2B. Skipping BNN sleeve.")
        print("  (Re-enable manually by editing part2b_summary.json if desired)")
        return False
    return True


# ---------------------------------------------------------------------------
# Load Part 2 artifacts
# ---------------------------------------------------------------------------
def load_lstm_artifacts():
    torch, nn, _, _ = _try_import_torch()

    # Load scalers
    feat_scaler_path = PART2_DIR / "feature_scaler.pkl"
    tgt_scaler_path = PART2_DIR / "target_scaler.pkl"
    model_path = PART2_DIR / "lstm_model.pt"
    meta_path = PART2_DIR / "part2_meta.json"

    for p in [feat_scaler_path, tgt_scaler_path, model_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Part 2 first.")

    with open(feat_scaler_path, "rb") as f:
        feat_scaler = pickle.load(f)
    with open(tgt_scaler_path, "rb") as f:
        tgt_scaler = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    n_features = meta["n_features"]

    # Rebuild model architecture (must match Part 2)
    model = _build_mc_lstm(n_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    return model, feat_scaler, tgt_scaler, feature_cols


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except ImportError:
        raise ImportError("PyTorch is required for Part 2C. Install with: pip install torch")


def _build_mc_lstm(input_size: int, hidden_size: int, num_layers: int,
                   dropout: float, n_outputs: int):
    """Re-build the LSTM architecture with dropout ALWAYS active (MC mode)."""
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
            """Set all dropout layers to training mode (enables MC sampling)."""
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

    return MCDropoutLSTM()


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------
def mc_predict(
    model,
    X_seq: np.ndarray,
    n_samples: int = N_MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes with dropout enabled.

    Returns
    -------
    mean_pred   : (n_rows, n_horizons)  — point estimate
    std_pred    : (n_rows, n_horizons)  — predictive std dev
    lower_ci    : (n_rows, n_horizons)  — CI_LOWER percentile
    upper_ci    : (n_rows, n_horizons)  — CI_UPPER percentile
    """
    torch, _, _, _ = _try_import_torch()

    model.eval()
    model.enable_dropout()   # keep dropout active

    X_tensor = torch.tensor(X_seq)
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor).numpy()   # (n_rows, n_horizons)
            samples.append(pred)

    samples_arr = np.stack(samples, axis=0)  # (n_samples, n_rows, n_horizons)

    mean_pred = samples_arr.mean(axis=0)
    std_pred = samples_arr.std(axis=0)
    lower_ci = np.percentile(samples_arr, CI_LOWER, axis=0)
    upper_ci = np.percentile(samples_arr, CI_UPPER, axis=0)

    return mean_pred, std_pred, lower_ci, upper_ci


# ---------------------------------------------------------------------------
# Calibration evaluation
# ---------------------------------------------------------------------------
def evaluate_calibration(
    true_vals: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
) -> Dict[str, float]:
    """Compute empirical coverage and expected calibration error."""
    results = {}
    for i, h in enumerate(HORIZONS):
        y = true_vals[:, i]
        lo = lower_ci[:, i]
        hi = upper_ci[:, i]
        mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if mask.sum() == 0:
            continue

        in_ci = ((y[mask] >= lo[mask]) & (y[mask] <= hi[mask]))
        coverage = float(in_ci.mean())
        mean_width = float((hi[mask] - lo[mask]).mean())
        results[f"h{h}_coverage_90pct"] = coverage
        results[f"h{h}_mean_ci_width_f"] = mean_width
        results[f"h{h}_calibration_error"] = abs(coverage - 0.90)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 2C] Project root: {PROJECT_DIR}")

    # Check gate
    if not check_bnn_gate():
        print("[Part 2C] Gate not met. Exiting gracefully (non-blocking).")
        return 0

    torch, _, _, _ = _try_import_torch()

    # Load Part 2 artifacts
    print("[Part 2C] Loading LSTM model and scalers from Part 2...")
    model, feat_scaler, tgt_scaler, feature_cols = load_lstm_artifacts()

    df = load_feature_matrix()
    splits = load_splits()

    target_cols = [f"target_h{h}" for h in HORIZONS]
    val_end = pd.Timestamp(splits["val_end"])
    df_val = df[(df["date"] > pd.Timestamp(splits["train_end"])) & (df["date"] <= val_end)].copy()
    df_test = df[df["date"] > val_end].copy()

    # Scale features
    X_val = feat_scaler.transform(df_val[feature_cols].fillna(0.0).values).astype(np.float32)
    X_test = feat_scaler.transform(df_test[feature_cols].fillna(0.0).values).astype(np.float32)
    X_all = feat_scaler.transform(df[feature_cols].fillna(0.0).values).astype(np.float32)

    # Build sequences
    def _build_seqs(X: np.ndarray) -> np.ndarray:
        return np.array([X[i - SEQUENCE_LEN:i] for i in range(SEQUENCE_LEN, len(X))], dtype=np.float32)

    X_val_seq = _build_seqs(X_val)
    X_test_seq = _build_seqs(X_test)
    X_all_seq = _build_seqs(X_all)

    # -----------------------------------------------------------------------
    # MC Dropout inference on validation set (calibration)
    # -----------------------------------------------------------------------
    print(f"[Part 2C] Running MC Dropout ({N_MC_SAMPLES} samples) on validation set...")
    val_mean_s, val_std_s, val_lo_s, val_hi_s = mc_predict(model, X_val_seq)

    # Inverse-scale
    val_mean_f = tgt_scaler.inverse_transform(val_mean_s)
    val_lo_f = tgt_scaler.inverse_transform(val_lo_s)
    val_hi_f = tgt_scaler.inverse_transform(val_hi_s)
    val_std_f = val_std_s * tgt_scaler.scale_  # approximate std in °F

    val_target_raw = df_val[target_cols].iloc[SEQUENCE_LEN:].values.astype(np.float32)
    val_true_f = tgt_scaler.inverse_transform(
        tgt_scaler.transform(np.nan_to_num(val_target_raw))
    )

    # Calibration
    cal = evaluate_calibration(val_true_f, val_lo_f, val_hi_f)
    print("\n=== CALIBRATION (90% CI) ===")
    for h in HORIZONS:
        cov = cal.get(f"h{h}_coverage_90pct", None)
        width = cal.get(f"h{h}_mean_ci_width_f", None)
        if cov is not None:
            print(f"  H={h}: coverage={cov:.1%}, mean CI width={width:.1f}°F")

    # -----------------------------------------------------------------------
    # MC Dropout inference on test set
    # -----------------------------------------------------------------------
    print("\n[Part 2C] Running MC Dropout on test set...")
    test_mean_s, test_std_s, test_lo_s, test_hi_s = mc_predict(model, X_test_seq)
    test_mean_f = tgt_scaler.inverse_transform(test_mean_s)
    test_lo_f = tgt_scaler.inverse_transform(test_lo_s)
    test_hi_f = tgt_scaler.inverse_transform(test_hi_s)

    # -----------------------------------------------------------------------
    # Live prediction with uncertainty
    # -----------------------------------------------------------------------
    print("\n[Part 2C] Running MC Dropout on latest available data...")
    live_seq = X_all_seq[-1:, :, :]  # last available sequence
    live_mean_s, live_std_s, live_lo_s, live_hi_s = mc_predict(model, live_seq)
    live_mean_f = tgt_scaler.inverse_transform(live_mean_s)[0]
    live_lo_f = tgt_scaler.inverse_transform(live_lo_s)[0]
    live_hi_f = tgt_scaler.inverse_transform(live_hi_s)[0]
    live_std_f = live_std_s[0] * tgt_scaler.scale_

    print("\n=== LIVE PREDICTIONS WITH UNCERTAINTY ===")
    for i, h in enumerate(HORIZONS):
        print(f"  H={h}: {live_mean_f[i]:.1f}°F  "
              f"[90% CI: {live_lo_f[i]:.1f}°F – {live_hi_f[i]:.1f}°F]  "
              f"(±{live_std_f[i]:.2f}°F)")

    # -----------------------------------------------------------------------
    # Save prediction parquet
    # -----------------------------------------------------------------------
    val_dates = df_val["date"].iloc[SEQUENCE_LEN:].reset_index(drop=True)
    test_dates = df_test["date"].iloc[SEQUENCE_LEN:].reset_index(drop=True)

    def _make_pred_df(dates, mean_f, lo_f, hi_f, std_f, true_f=None) -> pd.DataFrame:
        rows = {"date": dates}
        for i, h in enumerate(HORIZONS):
            rows[f"bnn_mean_h{h}"] = mean_f[:, i]
            rows[f"bnn_lo90_h{h}"] = lo_f[:, i]
            rows[f"bnn_hi90_h{h}"] = hi_f[:, i]
            rows[f"bnn_std_h{h}"] = std_f[:, i] if std_f.ndim == 2 else std_f
            if true_f is not None:
                rows[f"true_h{h}"] = true_f[:, i]
        return pd.DataFrame(rows)

    val_df = _make_pred_df(val_dates, val_mean_f, val_lo_f, val_hi_f, val_std_f, val_true_f)
    test_df = _make_pred_df(test_dates, test_mean_f, test_lo_f, test_hi_f,
                            test_std_f * np.ones_like(test_mean_f))

    all_df = pd.concat([val_df, test_df], ignore_index=True)
    all_df.to_parquet(ARTIFACTS_DIR / "bnn_predictions.parquet", index=False)
    print(f"\n[Part 2C] Saved bnn_predictions.parquet ({len(all_df)} rows)")

    # Update prediction_log with BNN uncertainty
    log_path = PART2_DIR / "prediction_log.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        for i, h in enumerate(HORIZONS):
            df_log.loc[df_log.index[-1], f"bnn_lo90_h{h}"] = float(live_lo_f[i])
            df_log.loc[df_log.index[-1], f"bnn_hi90_h{h}"] = float(live_hi_f[i])
            df_log.loc[df_log.index[-1], f"bnn_std_h{h}"] = float(live_std_f[i])
        df_log.to_csv(log_path, index=False)
        print("[Part 2C] Updated prediction_log.csv with BNN uncertainty columns")

    # Calibration report
    cal_report = {
        "schema_version": SCHEMA_VERSION,
        "n_mc_samples": N_MC_SAMPLES,
        "ci_lower_pct": CI_LOWER,
        "ci_upper_pct": CI_UPPER,
        "ci_target_coverage": 0.90,
        "calibration_results": cal,
        "calibration_pass": all(
            cal.get(f"h{h}_calibration_error", 1.0) < 0.10
            for h in HORIZONS
        ),
    }
    with open(ARTIFACTS_DIR / "calibration_report.json", "w") as f:
        json.dump(cal_report, f, indent=2)

    # Meta
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_at": pd.Timestamp.now().isoformat(),
        "n_mc_samples": N_MC_SAMPLES,
        "sequence_len": SEQUENCE_LEN,
        "dropout_rate": DROPOUT,
        "live_predictions": {
            f"h{h}": {
                "mean_f": float(live_mean_f[i]),
                "lo90_f": float(live_lo_f[i]),
                "hi90_f": float(live_hi_f[i]),
                "std_f": float(live_std_f[i]),
            }
            for i, h in enumerate(HORIZONS)
        },
        "calibration_summary": cal,
        "calibration_pass": cal_report["calibration_pass"],
    }
    with open(ARTIFACTS_DIR / "part2c_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[Part 2C] Saved part2c_meta.json")

    print(f"\n[Part 2C] ✅ Complete. Calibration pass={cal_report['calibration_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
