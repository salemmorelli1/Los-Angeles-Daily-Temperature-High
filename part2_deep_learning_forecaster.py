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
  6. Raw model output is clipped to [CLIP_MIN, CLIP_MAX] before inverse-scaling.
     Any clipping is flagged in the prediction log as lstm_output_clipped=True.
  7. The prediction log is upserted by (decision_date, feature_date, model).
     Duplicate rows are never created; reruns overwrite the existing row.
  8. Part 2 writes preliminary forecast_h* columns (forecast_source=lstm_preliminary).
     Part 2B overwrites these with the canonical fallback-chain forecast.

Artifacts Written
-----------------
  artifacts_part2/
      lstm_model.pt / transformer_model.pt  — trained PyTorch weights
      feature_scaler.pkl                    — fitted MinMaxScaler
      target_scaler.pkl                     — fitted target scaler
      training_history.json                 — scaled loss/MAE per epoch
      val_predictions.parquet               — out-of-sample validation predictions
      prediction_log.csv                    — live predictions, upserted daily
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

SCHEMA_VERSION = "1.2.0"
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

# Scaled output is clipped to this range before inverse-scaling.
# Values outside [0, 1] produce temperatures outside the training range.
CLIP_MIN = 0.0
CLIP_MAX = 1.0

# Upsert key for prediction log
LOG_KEY_COLS = ("decision_date", "feature_date", "model")


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
            "Install: pip install torch --index-url https://download.pytorch.org/whl/cpu"
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

    Drops string/object columns that cannot be fed to the scaler or model.
    Bool columns (e.g. regime one-hots) are retained and cast to float32.
    """
    target_cols = {f"target_h{h}" for h in HORIZONS}
    excluded = {"date"} | target_cols

    feature_cols: List[str] = []
    dropped: List[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            feature_cols.append(col)
        else:
            coerced = pd.to_numeric(series, errors="coerce")
            non_null = series.notna()
            if int(non_null.sum()) > 0 and coerced[non_null].notna().all():
                feature_cols.append(col)
            else:
                dropped.append(col)

    if dropped:
        print(f"[Part 2] Dropping non-numeric columns: {dropped}")
    if not feature_cols:
        raise ValueError("No numeric feature columns available for Part 2.")
    return feature_cols


def _target_cols() -> List[str]:
    return [f"target_h{h}" for h in HORIZONS]


def _model_path(model_type: str) -> Path:
    return ARTIFACTS_DIR / MODEL_FILES.get(model_type, MODEL_FILES["lstm"])


def _clean_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def _build_labeled_splits(
    df: pd.DataFrame, splits: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df.dropna(subset=_target_cols()).copy()
    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])
    return (
        labeled[labeled["date"] <= train_end].copy(),
        labeled[(labeled["date"] > train_end) & (labeled["date"] <= val_end)].copy(),
        labeled[labeled["date"] > val_end].copy(),
    )


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) < seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0, y.shape[1]), dtype=np.float32),
        )
    Xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1 : i + 1])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def sequence_dates(dates: pd.Series, seq_len: int) -> pd.Series:
    if len(dates) < seq_len:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(dates).iloc[seq_len - 1 :].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_lstm_model(input_size, hidden_size, num_layers, dropout, n_outputs):
    torch, nn, _, _ = _try_import_torch()

    class TemperatureLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, 64), nn.ReLU(),
                    nn.Dropout(dropout * 0.5), nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ])

        def forward(self, x):
            out, _ = self.lstm(x)
            h = self.dropout(out[:, -1, :])
            h = self.bn(h)
            return torch.cat([head(h) for head in self.heads], dim=1)

    return TemperatureLSTM()


def build_transformer_model(input_size, d_model, nhead, num_encoder_layers, dropout, n_outputs):
    torch, nn, _, _ = _try_import_torch()

    class TemperatureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, 64), nn.ReLU(),
                    nn.Dropout(dropout * 0.5), nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ])

        def forward(self, x):
            x = self.input_proj(x)
            h = self.dropout(self.encoder(x)[:, -1, :])
            return torch.cat([head(h) for head in self.heads], dim=1)

    return TemperatureTransformer()


def build_model(model_type: str, input_size: int):
    if model_type == "transformer":
        return build_transformer_model(input_size, 128, 4, 2, DROPOUT, len(HORIZONS))
    return build_lstm_model(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader) -> Tuple[Dict, object, float]:
    torch, nn, _, _ = _try_import_torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    criterion = nn.MSELoss(reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    w = torch.tensor(
        [HORIZON_WEIGHTS[h] for h in HORIZONS], dtype=torch.float32, device=device
    ).view(1, -1)

    history: Dict = {"train_loss": [], "val_loss": [], "val_mae_scaled": []}
    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(Xb), yb) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_losses.append(float((criterion(pred, yb) * w).mean().item()))
                val_maes.append(float(torch.abs(pred - yb).mean().item()))

        tl = float(np.mean(train_losses)) if train_losses else float("nan")
        vl = float(np.mean(val_losses)) if val_losses else float("inf")
        vm = float(np.mean(val_maes)) if val_maes else float("inf")

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_mae_scaled"].append(vm)
        scheduler.step(vm)

        if vm < best_val_mae - 1e-5:
            best_val_mae = vm
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{MAX_EPOCHS} | "
                  f"train={tl:.5f} | val={vl:.5f} | val_mae_scaled={vm:.5f}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best={best_val_mae:.5f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return history, model.to("cpu"), best_val_mae


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def predict_scaled(model, X_seq: np.ndarray) -> np.ndarray:
    torch, _, _, _ = _try_import_torch()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_seq, dtype=torch.float32)).numpy()


def predict_fahrenheit(
    model, X_seq: np.ndarray, tgt_scaler
) -> Tuple[np.ndarray, bool]:
    """Run inference, clip to [CLIP_MIN, CLIP_MAX], inverse-scale to °F.

    Returns
    -------
    preds_f    : ndarray (n, n_horizons) in Fahrenheit
    was_clipped: True if any raw output was outside [CLIP_MIN, CLIP_MAX]
    """
    raw = predict_scaled(model, X_seq)
    clipped = np.clip(raw, CLIP_MIN, CLIP_MAX)
    was_clipped = bool(not np.allclose(raw, clipped, atol=1e-4))
    if was_clipped:
        diff = np.abs(raw - clipped)
        print(f"[Part 2] ⚠️  Output clipped. Max exceedance: {diff.max():.4f}")
    return tgt_scaler.inverse_transform(clipped), was_clipped


def metrics_fahrenheit(pred_f: np.ndarray, true_f: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, h in enumerate(HORIZONS):
        mask = np.isfinite(true_f[:, i]) & np.isfinite(pred_f[:, i])
        if not mask.any():
            continue
        err = pred_f[mask, i] - true_f[mask, i]
        out[f"h{h}_mae_f"] = float(np.mean(np.abs(err)))
        out[f"h{h}_rmse_f"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"h{h}_bias_f"] = float(np.mean(err))
    return out


def average_horizon_mae(metrics: Dict[str, float]) -> float:
    vals = [metrics[f"h{h}_mae_f"] for h in HORIZONS if f"h{h}_mae_f" in metrics]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Prediction log — idempotent upsert
# ---------------------------------------------------------------------------
def _log_path() -> Path:
    return ARTIFACTS_DIR / "prediction_log.csv"


def load_prediction_log() -> pd.DataFrame:
    p = _log_path()
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def upsert_log_row(row: Dict) -> None:
    """Write `row` to the prediction log, overwriting any existing row with the same key.

    Key: (decision_date, feature_date, model).
    Columns not present in `row` (e.g. realized_h*, blend_h* written by other parts)
    are preserved on existing rows.
    """
    df = load_prediction_log()
    if df.empty:
        pd.DataFrame([row]).to_csv(_log_path(), index=False)
        print("[Part 2] Created prediction_log.csv with first row.")
        return

    key_vals = {k: str(row.get(k, "")).strip() for k in LOG_KEY_COLS}
    match = pd.Series([True] * len(df))
    for k, v in key_vals.items():
        col = df[k].astype(str).str.strip() if k in df.columns else pd.Series([""] * len(df))
        match = match & (col == v)

    if match.any():
        idx = df.index[match][-1]
        for col, val in row.items():
            df.loc[idx, col] = val
        print(f"[Part 2] Upserted prediction row for {row.get('decision_date')}.")
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print(f"[Part 2] Appended new prediction row for {row.get('decision_date')}.")

    df.to_csv(_log_path(), index=False)


def write_prediction_row(
    preds_f: np.ndarray,
    was_clipped: bool,
    decision_date: pd.Timestamp,
    feature_date: pd.Timestamp,
    model_type: str,
) -> None:
    """Write LSTM output + preliminary forecast_h* to the prediction log.

    forecast_h* is set here as a preliminary value (source=lstm_preliminary).
    Part 2B will overwrite forecast_h* with the canonical fallback-chain value.
    """
    row: Dict = {
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "feature_date": feature_date.strftime("%Y-%m-%d"),
        "target_date_h1": (feature_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "target_date_h3": (feature_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        "target_date_h5": (feature_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "model": model_type.upper(),
        "lstm_output_clipped": bool(was_clipped),
        "written_at": pd.Timestamp.now().isoformat(),
    }
    for i, h in enumerate(HORIZONS):
        val = float(preds_f[i])
        row[f"target_h{h}"] = val
        row[f"forecast_h{h}"] = val  # preliminary; Part 2B will overwrite
    row["forecast_source"] = "lstm_preliminary"
    row["forecast_reason"] = "awaiting_Part2B_fallback_chain"

    # Preserve any realized_h* values already written by Part 9 on this key
    existing = load_prediction_log()
    if not existing.empty:
        key_vals = {k: str(row.get(k, "")).strip() for k in LOG_KEY_COLS}
        match = pd.Series([True] * len(existing))
        for k, v in key_vals.items():
            col = existing[k].astype(str).str.strip() if k in existing.columns else pd.Series([""] * len(existing))
            match = match & (col == v)
        if match.any():
            idx = existing.index[match][-1]
            for col in [f"realized_h{h}" for h in HORIZONS]:
                if col in existing.columns and pd.notna(existing.at[idx, col]):
                    row[col] = existing.at[idx, col]

    upsert_log_row(row)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_training_artifacts(model, model_type, feat_scaler, tgt_scaler, history, meta):
    torch, _, _, _ = _try_import_torch()
    torch.save(model.state_dict(), _model_path(model_type))
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
    feat_path = ARTIFACTS_DIR / "feature_scaler.pkl"
    tgt_path = ARTIFACTS_DIR / "target_scaler.pkl"
    meta_path = ARTIFACTS_DIR / "part2_meta.json"
    mpath = _model_path(model_type)
    if not mpath.exists() and model_type == "lstm":
        mpath = ARTIFACTS_DIR / "lstm_model.pt"
    for p in [feat_path, tgt_path, meta_path, mpath]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Part 2 --mode=train first.")
    with open(feat_path, "rb") as f:
        feat_scaler = pickle.load(f)
    with open(tgt_path, "rb") as f:
        tgt_scaler = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    saved = meta.get("model_type", model_type)
    if saved != model_type:
        raise ValueError(f"Saved model_type={saved} != requested {model_type}. Retrain or use --model={saved}.")
    model = build_model(model_type, input_size)
    model.load_state_dict(torch.load(mpath, map_location="cpu"))
    model.eval()
    return model, feat_scaler, tgt_scaler, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(model_type: str = "lstm", mode: str = "train") -> int:
    torch, _, DataLoader, TensorDataset = _try_import_torch()
    print(f"[Part 2] model={model_type.upper()}  mode={mode}  root={PROJECT_DIR}")

    df = load_data()
    splits = load_splits()
    feature_cols = _get_feature_cols(df)
    target_cols = _target_cols()
    print(f"[Part 2] {len(df)} feature rows, {len(feature_cols)} features")

    df_train, df_val, df_test = _build_labeled_splits(df, splits)
    print(f"[Part 2] Labeled — Train:{len(df_train)} Val:{len(df_val)} Test:{len(df_test)}")
    print(f"[Part 2] Live feature date: {df['date'].max().date()}")

    if len(df_train) < SEQUENCE_LEN or len(df_val) < SEQUENCE_LEN:
        raise ValueError("Not enough rows to build sequences.")

    input_size = len(feature_cols)

    # -------------------------------------------------------------------
    # Training branch
    # -------------------------------------------------------------------
    if mode == "train":
        from sklearn.preprocessing import MinMaxScaler

        feat_sc = MinMaxScaler()
        tgt_sc = MinMaxScaler()

        X_tr = feat_sc.fit_transform(_clean_feature_frame(df_train, feature_cols)).astype(np.float32)
        y_tr = tgt_sc.fit_transform(df_train[target_cols].values.astype(np.float32)).astype(np.float32)
        X_va = feat_sc.transform(_clean_feature_frame(df_val, feature_cols)).astype(np.float32)
        y_va = tgt_sc.transform(df_val[target_cols].values.astype(np.float32)).astype(np.float32)
        X_te = feat_sc.transform(_clean_feature_frame(df_test, feature_cols)).astype(np.float32)
        y_te = tgt_sc.transform(df_test[target_cols].values.astype(np.float32)).astype(np.float32)

        Xtr_seq, ytr_seq = build_sequences(X_tr, y_tr, SEQUENCE_LEN)
        Xva_seq, yva_seq = build_sequences(X_va, y_va, SEQUENCE_LEN)
        Xte_seq, yte_seq = build_sequences(X_te, y_te, SEQUENCE_LEN)
        print(f"[Part 2] Seqs — Tr:{Xtr_seq.shape} Va:{Xva_seq.shape} Te:{Xte_seq.shape}")

        tr_ds = TensorDataset(torch.tensor(Xtr_seq), torch.tensor(ytr_seq))
        va_ds = TensorDataset(torch.tensor(Xva_seq), torch.tensor(yva_seq))
        tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        va_ldr = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(model_type, input_size)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Part 2] Parameters: {n_p:,}")
        print("[Part 2] Training...")

        history, model, best_scaled_mae = train_model(model, tr_ldr, va_ldr)

        # Metrics in °F — clipping applied consistently with inference
        vp_f, _ = predict_fahrenheit(model, Xva_seq, tgt_sc)
        vt_f = tgt_sc.inverse_transform(np.clip(yva_seq, CLIP_MIN, CLIP_MAX))
        val_metrics = metrics_fahrenheit(vp_f, vt_f)
        val_mae_f = average_horizon_mae(val_metrics)

        print("\n=== VALIDATION METRICS ===")
        for h in HORIZONS:
            print(f"  H={h}: MAE={val_metrics.get(f'h{h}_mae_f', float('nan')):.2f}°F  "
                  f"RMSE={val_metrics.get(f'h{h}_rmse_f', float('nan')):.2f}°F  "
                  f"Bias={val_metrics.get(f'h{h}_bias_f', float('nan')):+.2f}°F")

        test_metrics: Dict = {}
        if len(Xte_seq) > 0:
            tp_f, _ = predict_fahrenheit(model, Xte_seq, tgt_sc)
            tt_f = tgt_sc.inverse_transform(np.clip(yte_seq, CLIP_MIN, CLIP_MAX))
            test_metrics = metrics_fahrenheit(tp_f, tt_f)
            print("\n=== TEST METRICS ===")
            for h in HORIZONS:
                print(f"  H={h}: MAE={test_metrics.get(f'h{h}_mae_f', float('nan')):.2f}°F  "
                      f"RMSE={test_metrics.get(f'h{h}_rmse_f', float('nan')):.2f}°F  "
                      f"Bias={test_metrics.get(f'h{h}_bias_f', float('nan')):+.2f}°F")

        val_dates = sequence_dates(df_val["date"], SEQUENCE_LEN)
        val_df = pd.DataFrame({"date": val_dates})
        for i, h in enumerate(HORIZONS):
            val_df[f"pred_h{h}"] = vp_f[:, i]
            val_df[f"true_h{h}"] = vt_f[:, i]
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
            "output_clipping": {"clip_min": CLIP_MIN, "clip_max": CLIP_MAX},
            "model_file": _model_path(model_type).name,
            "hyperparameters": {
                "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
                "dropout": DROPOUT, "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "lr": LR, "horizon_weights": HORIZON_WEIGHTS,
            },
            "best_val_mae_scaled_01": best_scaled_mae,
            "val_mae_f": val_mae_f,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "split_summary": {
                "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
                "train_end": splits.get("train_end"),
                "val_end": splits.get("val_end"),
                "test_end": splits.get("test_end"),
            },
        }
        save_training_artifacts(model, model_type, feat_sc, tgt_sc, history, meta)
        feat_scaler, tgt_scaler = feat_sc, tgt_sc

    # -------------------------------------------------------------------
    # Predict-only branch
    # -------------------------------------------------------------------
    else:
        model, feat_scaler, tgt_scaler, _ = load_training_artifacts(model_type, input_size)

    # -------------------------------------------------------------------
    # Live prediction
    # -------------------------------------------------------------------
    all_X = feat_scaler.transform(_clean_feature_frame(df, feature_cols)).astype(np.float32)
    if len(all_X) < SEQUENCE_LEN:
        raise ValueError("Not enough rows for live sequence.")

    live_seq = all_X[-SEQUENCE_LEN:][np.newaxis, :, :]
    live_f, was_clipped = predict_fahrenheit(model, live_seq, tgt_scaler)
    live_f = live_f[0]

    decision_date = pd.Timestamp.today().normalize()
    feature_date = pd.Timestamp(df["date"].max()).normalize()

    print("\n=== LIVE PREDICTIONS ===")
    print(f"  Feature date: {feature_date.date()}  Decision date: {decision_date.date()}")
    if was_clipped:
        print("  ⚠️  Raw LSTM output exceeded [0,1] and was clipped before inverse-scaling.")
    for i, h in enumerate(HORIZONS):
        print(f"  H={h} ({(feature_date + pd.Timedelta(days=h)).date()}): {live_f[i]:.1f}°F")

    write_prediction_row(live_f, was_clipped, decision_date, feature_date, model_type)

    # Patch meta with live info
    meta_path = ARTIFACTS_DIR / "part2_meta.json"
    meta: Dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta.update({
        "last_prediction_at": pd.Timestamp.now().isoformat(),
        "decision_date": decision_date.isoformat(),
        "feature_date": feature_date.isoformat(),
        "lstm_output_clipped": bool(was_clipped),
        "live_predictions": {f"h{h}": float(live_f[i]) for i, h in enumerate(HORIZONS)},
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Part 2] ✅  Complete (clipped={was_clipped}).")
    if meta.get("val_mae_f"):
        print(f"[Part 2] Val avg MAE = {meta['val_mae_f']:.2f}°F")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(model_type=args.model, mode="predict" if args.predict_only else args.mode))





