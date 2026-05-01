#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 — Deep Learning Forecaster (LSTM / Transformer)
=======================================================
Trains or loads a multi-horizon neural forecaster for LA daily temperature highs
at H=1, H=3, and H=5 days ahead.

Key production contracts
------------------------
  1. Feature rows are dated by the last fully known observation date.
  2. Targets are interpreted as temp_high_f at feature_date + H calendar days.
  3. Training uses only rows where all horizon targets are observed.
  4. Live inference uses the newest feature row, even if future targets are NaN.
  5. Validation metrics saved as *_f are computed in real Fahrenheit units.

Artifacts Written
-----------------
  artifacts_part2/
      lstm_model.pt / transformer_model.pt  — trained PyTorch weights
      feature_scaler.pkl                    — fitted MinMaxScaler
      target_scaler.pkl                     — fitted target scaler
      training_history.json                 — scaled loss/MAE per epoch
      val_predictions.parquet               — out-of-sample validation predictions
      prediction_log.csv                    — live predictions, appended daily
      part2_meta.json                       — hyperparameters, metrics, schema version
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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

SCHEMA_VERSION = "1.1.0"
HORIZONS = [1, 3, 5]
SEQUENCE_LEN = 14
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 20
LR = 1e-3
HORIZON_WEIGHTS = {1: 1.0, 3: 0.8, 5: 0.6}
MODEL_FILES = {"lstm": "lstm_model.pt", "transformer": "transformer_model.pt"}


# ---------------------------------------------------------------------------
# PyTorch imports
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
        raise FileNotFoundError("train_val_test_split.json not found. Run Part 1 first.")
    with open(path) as f:
        return json.load(f)


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return model-eligible numeric feature columns only.

    Part 6/Part 1 can add descriptive string columns such as regime_name
    (for example, SANTA_ANA / MARINE_LAYER / DRY_CLEAR). Those are useful
    for diagnostics, but neural-network scalers require numeric inputs.
    Keep numeric/bool columns and drop true categorical text fields.
    """
    target_cols = {f"target_h{h}" for h in HORIZONS}
    excluded = {"date"} | target_cols

    feature_cols: List[str] = []
    dropped_non_numeric: List[str] = []

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            feature_cols.append(col)
            continue

        coerced = pd.to_numeric(series, errors="coerce")
        non_null = series.notna()
        if int(non_null.sum()) > 0 and coerced[non_null].notna().all():
            feature_cols.append(col)
        else:
            dropped_non_numeric.append(col)

    if dropped_non_numeric:
        print(f"[Part 2] Dropping non-numeric feature columns: {dropped_non_numeric}")

    if not feature_cols:
        raise ValueError("No numeric feature columns available for Part 2 training/inference.")

    return feature_cols

def _target_cols() -> List[str]:
    return [f"target_h{h}" for h in HORIZONS]


def _model_path(model_type: str) -> Path:
    return ARTIFACTS_DIR / MODEL_FILES.get(model_type, MODEL_FILES["lstm"])


def _clean_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Coerce selected features to a finite float32 matrix."""
    X = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32)

def _build_labeled_splits(df: pd.DataFrame, splits: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_cols = _target_cols()
    labeled = df.dropna(subset=target_cols).copy()

    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])

    df_train = labeled[labeled["date"] <= train_end].copy()
    df_val = labeled[(labeled["date"] > train_end) & (labeled["date"] <= val_end)].copy()
    df_test = labeled[labeled["date"] > val_end].copy()

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tabular rows to sequences ending on the same row as the target.

    For row i, the model sees X[i-seq_len+1 : i+1] and learns y[i].
    Therefore H=1 means temp_high at feature_date + 1, not an accidental
    extra day caused by excluding the current feature row.
    """
    if len(X) < seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0, y.shape[1]), dtype=np.float32),
        )

    Xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1:i + 1])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def sequence_dates(dates: pd.Series, seq_len: int) -> pd.Series:
    """Dates aligned to build_sequences output."""
    if len(dates) < seq_len:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(dates).iloc[seq_len - 1:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Models
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

    return TemperatureLSTM()


def build_transformer_model(input_size: int, d_model: int, nhead: int,
                            num_encoder_layers: int, dropout: float, n_outputs: int):
    torch, nn, _, _ = _try_import_torch()

    class TemperatureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
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


def build_model(model_type: str, input_size: int):
    if model_type == "transformer":
        return build_transformer_model(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            dropout=DROPOUT,
            n_outputs=len(HORIZONS),
        )
    return build_lstm_model(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        n_outputs=len(HORIZONS),
    )


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
) -> Tuple[Dict, object, float]:
    torch, nn, _, _ = _try_import_torch()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    criterion = nn.MSELoss(reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_mae_scaled": []}
    best_val_mae_scaled = float("inf")
    best_state = None
    no_improve = 0

    w = np.array([HORIZON_WEIGHTS[h] for h in HORIZONS], dtype=np.float32)
    w_tensor = torch.tensor(w, device=device).view(1, -1)

    for epoch in range(1, max_epochs + 1):
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
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_loss = (criterion(pred, yb) * w_tensor).mean()
                val_losses.append(float(val_loss.item()))
                val_maes.append(float(torch.abs(pred - yb).mean().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_mae_scaled = float(np.mean(val_maes)) if val_maes else float("inf")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_scaled"].append(val_mae_scaled)

        scheduler.step(val_mae_scaled)

        if val_mae_scaled < best_val_mae_scaled - 1e-5:
            best_val_mae_scaled = val_mae_scaled
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{max_epochs} | "
                f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
                f"val_mae_scaled={val_mae_scaled:.5f}"
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best scaled val MAE={best_val_mae_scaled:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")
    return history, model, best_val_mae_scaled


def predict_scaled(model, X_seq: np.ndarray) -> np.ndarray:
    torch, _, _, _ = _try_import_torch()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_seq, dtype=torch.float32)).numpy()


def metrics_fahrenheit(pred_f: np.ndarray, true_f: np.ndarray, prefix: str = "") -> Dict[str, float]:
    metrics = {}
    for i, h in enumerate(HORIZONS):
        y_true = true_f[:, i]
        y_pred = pred_f[:, i]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            continue
        mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        bias = float(np.mean(y_pred[mask] - y_true[mask]))
        metrics[f"{prefix}h{h}_mae_f"] = mae
        metrics[f"{prefix}h{h}_rmse_f"] = rmse
        metrics[f"{prefix}h{h}_bias_f"] = bias
    return metrics


def average_horizon_mae(metrics: Dict[str, float], prefix: str = "") -> float:
    vals = [metrics.get(f"{prefix}h{h}_mae_f") for h in HORIZONS]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def append_prediction_log(
    predictions: Dict[str, float],
    decision_date: pd.Timestamp,
    feature_date: pd.Timestamp,
    model_type: str,
) -> None:
    """Append live predictions to the Part 2 prediction log.

    Target dates are defined from feature_date, not decision_date.
    """
    log_path = ARTIFACTS_DIR / "prediction_log.csv"

    row = {
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "feature_date": feature_date.strftime("%Y-%m-%d"),
        "target_date_h1": (feature_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "target_date_h3": (feature_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        "target_date_h5": (feature_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "model": model_type.upper(),
        "target_h1": predictions.get("h1"),
        "target_h3": predictions.get("h3"),
        "target_h5": predictions.get("h5"),
        "realized_h1": np.nan,
        "realized_h3": np.nan,
        "realized_h5": np.nan,
        "written_at": pd.Timestamp.now().isoformat(),
    }

    if log_path.exists():
        df_log = pd.read_csv(log_path)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(log_path, index=False)
    print("[Part 2] Appended prediction to prediction_log.csv")


def save_training_artifacts(
    model,
    model_type: str,
    feat_scaler,
    tgt_scaler,
    history: Dict,
    meta: Dict,
) -> None:
    torch, _, _, _ = _try_import_torch()
    torch.save(model.state_dict(), _model_path(model_type))

    # Maintain legacy LSTM filename for Part 2C compatibility.
    if model_type == "lstm":
        torch.save(model.state_dict(), ARTIFACTS_DIR / "lstm_model.pt")

    with open(ARTIFACTS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(ARTIFACTS_DIR / "target_scaler.pkl", "wb") as f:
        pickle.dump(tgt_scaler, f)
    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(ARTIFACTS_DIR / "part2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_training_artifacts(model_type: str, input_size: int):
    torch, _, _, _ = _try_import_torch()

    feat_scaler_path = ARTIFACTS_DIR / "feature_scaler.pkl"
    tgt_scaler_path = ARTIFACTS_DIR / "target_scaler.pkl"
    meta_path = ARTIFACTS_DIR / "part2_meta.json"
    model_path = _model_path(model_type)

    if not model_path.exists() and model_type == "lstm":
        model_path = ARTIFACTS_DIR / "lstm_model.pt"

    for p in [feat_scaler_path, tgt_scaler_path, meta_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Part 2 training first.")

    with open(feat_scaler_path, "rb") as f:
        feat_scaler = pickle.load(f)
    with open(tgt_scaler_path, "rb") as f:
        tgt_scaler = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    saved_model_type = meta.get("model_type", model_type)
    if saved_model_type != model_type:
        raise ValueError(
            f"Requested model_type={model_type}, but saved model_type={saved_model_type}. "
            "Use matching --model or retrain."
        )

    model = build_model(model_type, input_size)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, feat_scaler, tgt_scaler, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(model_type: str = "lstm", mode: str = "train") -> int:
    torch, _, DataLoader, TensorDataset = _try_import_torch()

    print(f"[Part 2] Model type: {model_type.upper()}")
    print(f"[Part 2] Mode: {mode}")
    print(f"[Part 2] Project root: {PROJECT_DIR}")

    df = load_data()
    splits = load_splits()
    feature_cols = _get_feature_cols(df)
    target_cols = _target_cols()

    print(f"[Part 2] {len(df)} feature rows, {len(feature_cols)} features")

    df_train, df_val, df_test = _build_labeled_splits(df, splits)
    print(f"[Part 2] Fully labeled — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"[Part 2] Live feature date: {pd.Timestamp(df['date'].max()).date()}")

    if len(df_train) < SEQUENCE_LEN or len(df_val) < SEQUENCE_LEN:
        raise ValueError("Not enough rows to build train/validation sequences.")

    input_size = len(feature_cols)

    # -----------------------------------------------------------------------
    # Training branch
    # -----------------------------------------------------------------------
    if mode == "train":
        from sklearn.preprocessing import MinMaxScaler

        feat_scaler = MinMaxScaler()
        tgt_scaler = MinMaxScaler()

        X_train_raw = _clean_feature_frame(df_train, feature_cols)
        y_train_raw = df_train[target_cols].values.astype(np.float32)

        X_train = feat_scaler.fit_transform(X_train_raw).astype(np.float32)
        y_train = tgt_scaler.fit_transform(y_train_raw).astype(np.float32)

        X_val = feat_scaler.transform(_clean_feature_frame(df_val, feature_cols)).astype(np.float32)
        y_val_raw = df_val[target_cols].values.astype(np.float32)
        y_val = tgt_scaler.transform(y_val_raw).astype(np.float32)

        X_test = feat_scaler.transform(_clean_feature_frame(df_test, feature_cols)).astype(np.float32)
        y_test_raw = df_test[target_cols].values.astype(np.float32)
        y_test = tgt_scaler.transform(y_test_raw).astype(np.float32)

        X_tr_seq, y_tr_seq = build_sequences(X_train, y_train, SEQUENCE_LEN)
        X_val_seq, y_val_seq = build_sequences(X_val, y_val, SEQUENCE_LEN)
        X_te_seq, y_te_seq = build_sequences(X_test, y_test, SEQUENCE_LEN)

        print(f"[Part 2] Sequence shapes — Train: {X_tr_seq.shape} | Val: {X_val_seq.shape} | Test: {X_te_seq.shape}")

        train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
        val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(model_type, input_size)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Part 2] Model parameters: {n_params:,}")

        print("[Part 2] Training...")
        history, model, best_val_mae_scaled = train_model(model, train_loader, val_loader)

        # Validation metrics in real Fahrenheit units.
        val_preds_s = predict_scaled(model, X_val_seq)
        val_preds_f = tgt_scaler.inverse_transform(val_preds_s)
        val_true_f = tgt_scaler.inverse_transform(y_val_seq)
        val_metrics = metrics_fahrenheit(val_preds_f, val_true_f)
        val_mae_f = average_horizon_mae(val_metrics)

        print("\n=== VALIDATION METRICS ===")
        for h in HORIZONS:
            print(
                f"  Val H={h}: MAE={val_metrics.get(f'h{h}_mae_f', float('nan')):.2f}°F | "
                f"RMSE={val_metrics.get(f'h{h}_rmse_f', float('nan')):.2f}°F"
            )

        # Test metrics in real Fahrenheit units.
        test_metrics = {}
        if len(X_te_seq) > 0:
            test_preds_s = predict_scaled(model, X_te_seq)
            test_preds_f = tgt_scaler.inverse_transform(test_preds_s)
            test_true_f = tgt_scaler.inverse_transform(y_te_seq)
            test_metrics = metrics_fahrenheit(test_preds_f, test_true_f)

            print("\n=== TEST METRICS ===")
            for h in HORIZONS:
                print(
                    f"  Test H={h}: MAE={test_metrics.get(f'h{h}_mae_f', float('nan')):.2f}°F | "
                    f"RMSE={test_metrics.get(f'h{h}_rmse_f', float('nan')):.2f}°F"
                )

        # Save validation predictions.
        val_dates = sequence_dates(df_val["date"], SEQUENCE_LEN)
        val_df = pd.DataFrame({"date": val_dates})
        for i, h in enumerate(HORIZONS):
            val_df[f"pred_h{h}"] = val_preds_f[:, i]
            val_df[f"true_h{h}"] = val_true_f[:, i]
            val_df[f"target_date_h{h}"] = val_dates + pd.Timedelta(days=h)
        val_df.to_parquet(ARTIFACTS_DIR / "val_predictions.parquet", index=False)

        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "trained_at": pd.Timestamp.now().isoformat(),
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "sequence_len": SEQUENCE_LEN,
            "horizons": HORIZONS,
            "target_clock": "target_date_h = feature_date + h calendar days",
            "model_file": _model_path(model_type).name,
            "hyperparameters": {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "lr": LR,
                "horizon_weights": HORIZON_WEIGHTS,
            },
            "best_val_mae_scaled": best_val_mae_scaled,
            "val_mae_f": val_mae_f,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "split_summary": {
                "n_train": len(df_train),
                "n_val": len(df_val),
                "n_test": len(df_test),
                "train_end": splits.get("train_end"),
                "val_end": splits.get("val_end"),
                "test_end": splits.get("test_end"),
            },
        }

        save_training_artifacts(model, model_type, feat_scaler, tgt_scaler, history, meta)

    # -----------------------------------------------------------------------
    # Predict-only branch
    # -----------------------------------------------------------------------
    else:
        model, feat_scaler, tgt_scaler, meta = load_training_artifacts(model_type, input_size)

    # -----------------------------------------------------------------------
    # Live prediction
    # -----------------------------------------------------------------------
    all_X = feat_scaler.transform(_clean_feature_frame(df, feature_cols)).astype(np.float32)
    if len(all_X) < SEQUENCE_LEN:
        raise ValueError("Not enough rows to build live sequence.")

    latest_seq = all_X[-SEQUENCE_LEN:][np.newaxis, :, :]
    live_pred_scaled = predict_scaled(model, latest_seq)
    live_pred_f = tgt_scaler.inverse_transform(live_pred_scaled)[0]

    decision_date = pd.Timestamp.today().normalize()
    feature_date = pd.Timestamp(df["date"].max()).normalize()

    live_predictions = {f"h{h}": float(live_pred_f[i]) for i, h in enumerate(HORIZONS)}

    print("\n=== LIVE PREDICTIONS ===")
    print(f"  Feature date: {feature_date.date()} | Decision date: {decision_date.date()}")
    for i, h in enumerate(HORIZONS):
        target_date = feature_date + pd.Timedelta(days=h)
        print(f"  H={h} ({target_date.date()}): {live_predictions[f'h{h}']:.1f}°F")

    append_prediction_log(live_predictions, decision_date, feature_date, model_type)

    # Update meta with latest live prediction after append.
    meta_path = ARTIFACTS_DIR / "part2_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "sequence_len": SEQUENCE_LEN,
            "horizons": HORIZONS,
        }

    meta.update({
        "last_prediction_at": pd.Timestamp.now().isoformat(),
        "decision_date": decision_date.isoformat(),
        "feature_date": feature_date.isoformat(),
        "target_clock": "target_date_h = feature_date + h calendar days",
        "live_predictions": live_predictions,
    })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Part 2] ✅ Complete. Live forecast appended.")
    if meta.get("val_mae_f") is not None:
        print(f"[Part 2] Validation average MAE = {meta['val_mae_f']:.2f}°F")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train = fit model and predict; predict = load existing model and append live forecast",
    )
    parser.add_argument("--predict-only", action="store_true", help="Alias for --mode=predict")
    args = parser.parse_args()
    selected_mode = "predict" if args.predict_only else args.mode
    raise SystemExit(main(model_type=args.model, mode=selected_mode))


