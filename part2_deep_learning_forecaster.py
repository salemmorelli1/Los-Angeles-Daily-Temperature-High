#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 — Deep Learning Forecaster (LSTM)
==========================================
Trains a multi-horizon LSTM to predict LA daily temperature highs at
H=1, H=3, and H=5 days ahead. Includes a Transformer option (--model transformer).

Architecture
------------
  Input  : sequence of last N days of features (default N=14)
  Hidden : 2-layer stacked LSTM with dropout
  Output : 3 heads — one per horizon (H=1, H=3, H=5)

Training
--------
  - MSE loss (sum across heads, weighted by horizon)
  - Adam optimizer with ReduceLROnPlateau scheduler
  - Early stopping on validation MAE (patience=20)
  - MinMax scaler fit on train split only (no leakage)

Artifacts Written
-----------------
  artifacts_part2/
      lstm_model.pt                  — trained PyTorch model weights
      feature_scaler.pkl             — fitted MinMaxScaler
      target_scaler.pkl              — fitted target scaler
      training_history.json          — loss/MAE per epoch
      val_predictions.parquet        — out-of-sample val set predictions
      prediction_log.csv             — today's predictions (appended daily)
      part2_meta.json                — hyperparameters, val metrics, schema version
"""

from __future__ import annotations

import argparse
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
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part2"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.0.0"
HORIZONS = [1, 3, 5]
SEQUENCE_LEN = 14          # Look back 14 days
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 20
LR = 1e-3
HORIZON_WEIGHTS = {1: 1.0, 3: 0.8, 5: 0.6}


# ---------------------------------------------------------------------------
# PyTorch imports (with graceful fallback)
# ---------------------------------------------------------------------------
def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for Part 2.\n"
            "Install with: pip install torch\n"
            "Or run: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


# ---------------------------------------------------------------------------
# Data loading and preparation
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
        raise FileNotFoundError("train_val_test_split.json not found. Run Part 1 first.")
    with open(path) as f:
        return json.load(f)


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    target_cols = {f"target_h{h}" for h in HORIZONS}
    return [c for c in df.columns if c not in {"date"} | target_cols]


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tabular data to overlapping sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------
def build_lstm_model(input_size: int, hidden_size: int, num_layers: int,
                     dropout: float, n_outputs: int):
    torch, nn, _, _ = _try_import_torch()

    class TemperatureLSTM(nn.Module):
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
            # Separate heads per horizon
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
            h = self.dropout(out[:, -1, :])           # take last timestep
            h = self.bn(h)
            return torch.cat([head(h) for head in self.heads], dim=1)  # (B, n_outputs)

    return TemperatureLSTM()


# ---------------------------------------------------------------------------
# Transformer Model (alternative backbone)
# ---------------------------------------------------------------------------
def build_transformer_model(input_size: int, d_model: int, nhead: int,
                             num_encoder_layers: int, dropout: float, n_outputs: int):
    torch, nn, _, _ = _try_import_torch()
    import math

    class TemperatureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ])

        def forward(self, x):
            x = self.input_proj(x)
            encoded = self.encoder(x)
            h = self.dropout(encoded[:, -1, :])
            return torch.cat([head(h) for head in self.heads], dim=1)

    return TemperatureTransformer()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
) -> Dict:
    torch, nn, DataLoader, TensorDataset = _try_import_torch()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, verbose=False
    )
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    w = np.array([HORIZON_WEIGHTS[h] for h in HORIZONS], dtype=np.float32)
    w_tensor = torch.tensor(w).to(device)

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = (criterion(pred, yb) * w_tensor).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = (criterion(pred, yb) * w_tensor).mean()
                val_losses.append(loss.item())
                mae = torch.abs(pred - yb).mean().item()
                val_maes.append(mae)

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_mae = float(np.mean(val_maes))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        scheduler.step(val_mae)

        if val_mae < best_val_mae - 0.01:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{max_epochs} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_mae={val_mae:.4f}°F")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best val_mae={best_val_mae:.4f}°F)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")
    return history, model, best_val_mae


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_seq: np.ndarray, y_seq: np.ndarray) -> Dict:
    torch, _, _, _ = _try_import_torch()
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_seq)).numpy()

    metrics = {}
    for i, h in enumerate(HORIZONS):
        y_true = y_seq[:, i]
        y_pred = preds[:, i]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            continue
        mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        bias = float(np.mean(y_pred[mask] - y_true[mask]))
        metrics[f"h{h}_mae"] = mae
        metrics[f"h{h}_rmse"] = rmse
        metrics[f"h{h}_bias"] = bias
    return metrics


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------
def append_prediction_log(
    predictions: Dict,
    decision_date: pd.Timestamp,
    feature_date: pd.Timestamp,
) -> None:
    """Append today's LSTM predictions to the canonical prediction log."""
    log_path = ARTIFACTS_DIR / "prediction_log.csv"

    row = {
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "feature_date": feature_date.strftime("%Y-%m-%d"),
        "model": "LSTM",
        "target_h1": predictions.get("h1"),
        "target_h3": predictions.get("h3"),
        "target_h5": predictions.get("h5"),
        "realized_h1": None,
        "realized_h3": None,
        "realized_h5": None,
        "written_at": pd.Timestamp.now().isoformat(),
    }

    if log_path.exists():
        df_log = pd.read_csv(log_path)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(log_path, index=False)
    print(f"[Part 2] Appended prediction to prediction_log.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(model_type: str = "lstm") -> int:
    torch, nn, DataLoader, TensorDataset = _try_import_torch()
    print(f"[Part 2] Model type: {model_type.upper()}")
    print(f"[Part 2] Project root: {PROJECT_DIR}")

    # Load data
    df = load_data()
    splits = load_splits()
    feature_cols = _get_feature_cols(df)
    target_cols = [f"target_h{h}" for h in HORIZONS]

    print(f"[Part 2] {len(df)} rows, {len(feature_cols)} features")

    # Split by date
    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])

    df_train = df[df["date"] <= train_end].copy()
    df_val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    df_test = df[df["date"] > val_end].copy()

    print(f"[Part 2] Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Scale features
    from sklearn.preprocessing import MinMaxScaler
    feat_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()

    X_train_raw = df_train[feature_cols].fillna(0.0).values.astype(np.float32)
    y_train_raw = df_train[target_cols].fillna(method="ffill").values.astype(np.float32)

    X_train = feat_scaler.fit_transform(X_train_raw).astype(np.float32)
    y_train = tgt_scaler.fit_transform(y_train_raw).astype(np.float32)

    X_val = feat_scaler.transform(df_val[feature_cols].fillna(0.0).values).astype(np.float32)
    y_val_raw = df_val[target_cols].fillna(method="ffill").values.astype(np.float32)
    y_val = tgt_scaler.transform(y_val_raw).astype(np.float32)

    X_test = feat_scaler.transform(df_test[feature_cols].fillna(0.0).values).astype(np.float32)
    y_test_raw = df_test[target_cols].fillna(method="ffill").values.astype(np.float32)
    y_test = tgt_scaler.transform(y_test_raw).astype(np.float32)

    # Build sequences
    X_tr_seq, y_tr_seq = build_sequences(X_train, y_train, SEQUENCE_LEN)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, SEQUENCE_LEN)
    X_te_seq, y_te_seq = build_sequences(X_test, y_test, SEQUENCE_LEN)

    print(f"[Part 2] Sequence shapes — Train: {X_tr_seq.shape} | Val: {X_val_seq.shape}")

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    input_size = X_tr_seq.shape[2]
    if model_type == "transformer":
        model = build_transformer_model(
            input_size=input_size, d_model=128, nhead=4,
            num_encoder_layers=2, dropout=DROPOUT, n_outputs=len(HORIZONS)
        )
    else:
        model = build_lstm_model(
            input_size=input_size, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout=DROPOUT, n_outputs=len(HORIZONS)
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Part 2] Model parameters: {n_params:,}")

    # Train
    print("[Part 2] Training...")
    history, model, best_val_mae = train_model(model, train_loader, val_loader)

    # Evaluate on test set
    test_metrics = evaluate(model, X_te_seq, y_te_seq)
    # Inverse scale to get real °F errors
    dummy_y = np.zeros((len(y_te_seq), len(HORIZONS)))
    dummy_pred = model(torch.tensor(X_te_seq)).detach().numpy()
    pred_unscaled = tgt_scaler.inverse_transform(dummy_pred)
    true_unscaled = tgt_scaler.inverse_transform(y_te_seq)

    real_metrics = {}
    for i, h in enumerate(HORIZONS):
        mask = np.isfinite(true_unscaled[:, i]) & np.isfinite(pred_unscaled[:, i])
        mae = float(np.mean(np.abs(true_unscaled[mask, i] - pred_unscaled[mask, i])))
        rmse = float(np.sqrt(np.mean((true_unscaled[mask, i] - pred_unscaled[mask, i]) ** 2)))
        real_metrics[f"h{h}_mae_f"] = mae
        real_metrics[f"h{h}_rmse_f"] = rmse
        print(f"  Test H={h}: MAE={mae:.2f}°F | RMSE={rmse:.2f}°F")

    # Save val predictions for attribution
    val_preds_raw = model(torch.tensor(X_val_seq)).detach().numpy()
    val_preds_f = tgt_scaler.inverse_transform(val_preds_raw)
    val_true_f = tgt_scaler.inverse_transform(y_val_seq)
    val_dates = df_val["date"].iloc[SEQUENCE_LEN:].reset_index(drop=True)

    val_df = pd.DataFrame({"date": val_dates})
    for i, h in enumerate(HORIZONS):
        val_df[f"pred_h{h}"] = val_preds_f[:, i]
        val_df[f"true_h{h}"] = val_true_f[:, i]
    val_df.to_parquet(ARTIFACTS_DIR / "val_predictions.parquet", index=False)

    # Live prediction on latest available data
    all_X = feat_scaler.transform(df[feature_cols].fillna(0.0).values).astype(np.float32)
    latest_seq = all_X[-SEQUENCE_LEN:][np.newaxis, :, :]  # (1, seq_len, n_features)
    model.eval()
    with torch.no_grad():
        live_pred_scaled = model(torch.tensor(latest_seq)).numpy()
    live_pred_f = tgt_scaler.inverse_transform(live_pred_scaled)[0]

    decision_date = pd.Timestamp.today().normalize()
    feature_date = pd.Timestamp(df["date"].max())

    live_predictions = {f"h{h}": float(live_pred_f[i]) for i, h in enumerate(HORIZONS)}
    print("\n=== LIVE PREDICTIONS ===")
    for h in HORIZONS:
        print(f"  H={h} day(s): {live_predictions[f'h{h}']:.1f}°F")

    append_prediction_log(live_predictions, decision_date, feature_date)

    # Save model and scalers
    torch.save(model.state_dict(), ARTIFACTS_DIR / "lstm_model.pt")
    with open(ARTIFACTS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(ARTIFACTS_DIR / "target_scaler.pkl", "wb") as f:
        pickle.dump(tgt_scaler, f)
    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "trained_at": pd.Timestamp.now().isoformat(),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "sequence_len": SEQUENCE_LEN,
        "horizons": HORIZONS,
        "hyperparameters": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
        },
        "val_mae_f": best_val_mae,
        "test_metrics": real_metrics,
        "live_predictions": live_predictions,
        "decision_date": decision_date.isoformat(),
    }
    with open(ARTIFACTS_DIR / "part2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Part 2] ✅ Complete. Best val MAE = {best_val_mae:.2f}°F")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    args = parser.parse_args()
    raise SystemExit(main(model_type=args.model))
