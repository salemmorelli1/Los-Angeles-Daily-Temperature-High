#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 — Deep Learning Forecaster (LSTM / Transformer)
=======================================================
Trains or loads a multi-horizon neural forecaster for LA daily temperature highs
at H=1, H=3, and H=5 calendar days ahead.

Heat-branch experiment status
-----------------------------
The asymmetric heat-branch training objective is retained in the file for
controlled experiments, but it is disabled by default for production after the
latest audit showed that it worsened global LSTM MAE and still produced 0/5
test-set heat-event hits.

When HEAT_BRANCH_EXPERIMENT = False, training uses the ordinary unweighted
horizon-weighted MSE objective. Validation and test metrics remain computed in
Fahrenheit units.

Key production contracts
------------------------
  1. Feature rows are dated by the last fully known observation date.
  2. Targets are interpreted as temp_high_f at feature_date + H calendar days.
  3. Training uses only rows where all horizon targets are observed.
  4. Live inference uses the newest feature row, even if future targets are NaN.
  5. Validation/test metrics saved as *_f are computed in Fahrenheit units.
  6. Raw model output is clipped to [CLIP_MIN, CLIP_MAX] before inverse-scaling.
  7. The prediction log is upserted by (decision_date, feature_date, model).
  8. Part 2 writes preliminary forecast_h* columns; Part 2B overwrites them.

Artifacts Written
-----------------
  artifacts_part2/
      lstm_model.pt / transformer_model.pt
      feature_scaler.pkl
      target_scaler.pkl
      training_history.json
      val_predictions.parquet
      prediction_log.csv
      part2_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
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

SCHEMA_VERSION = "1.3.1-heat-branch-disabled"
HORIZONS = [1, 3, 5]
SEQUENCE_LEN = 14
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 15
LR = 1e-3
RANDOM_SEED = 42
HORIZON_WEIGHTS = {1: 1.0, 3: 0.8, 5: 0.6}
MODEL_FILES = {"lstm": "lstm_model.pt", "transformer": "transformer_model.pt"}

CLIP_MIN = 0.0
CLIP_MAX = 1.0

# ---------------------------------------------------------------------------
# Heat-branch experiment controls
# ---------------------------------------------------------------------------
# Disabled by default after the heat branch failed the audit: global LSTM MAE worsened
# and the test-set heat-event hit-rate remained 0/5. Keep the code path only for
# future named experiments, not production retrains.
HEAT_BRANCH_EXPERIMENT = False
WARM_EVENT_F = 78.0
HEAT_EVENT_F = 85.0
WARM_WEIGHT = 2.0
HEAT_WEIGHT = 5.0
WARM_UNDERPRED_PENALTY = 1.25
HEAT_UNDERPRED_PENALTY = 2.00

LOG_KEY_COLS = ("decision_date", "feature_date", "model")


# ---------------------------------------------------------------------------
# PyTorch imports / reproducibility
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


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch, _, _, _ = _try_import_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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


def _target_cols() -> List[str]:
    return [f"target_h{h}" for h in HORIZONS]


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return model-eligible numeric, nonconstant feature columns."""
    target_cols = set(_target_cols())
    excluded = {"date"} | target_cols

    feature_cols: List[str] = []
    dropped_non_numeric: List[str] = []

    for col in df.columns:
        if col in excluded:
            continue

        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            feature_cols.append(col)
            continue

        coerced = pd.to_numeric(s, errors="coerce")
        non_null = s.notna()
        if int(non_null.sum()) > 0 and coerced[non_null].notna().all():
            feature_cols.append(col)
        else:
            dropped_non_numeric.append(col)

    if dropped_non_numeric:
        print(f"[Part 2] Dropping non-numeric columns: {dropped_non_numeric}")

    nonconstant: List[str] = []
    dropped_constant: List[str] = []

    for col in feature_cols:
        s_num = pd.to_numeric(df[col], errors="coerce")
        if s_num.nunique(dropna=True) > 1:
            nonconstant.append(col)
        else:
            dropped_constant.append(col)

    if dropped_constant:
        print(
            f"[Part 2] Dropping {len(dropped_constant)} constant/zero-variance "
            f"feature columns: {dropped_constant}"
        )

    if not nonconstant:
        raise ValueError("No numeric nonconstant feature columns available for Part 2.")

    return nonconstant


def _clean_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return (
        X.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _build_labeled_splits(
    df: pd.DataFrame,
    splits: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df.dropna(subset=_target_cols()).copy()
    train_end = pd.Timestamp(splits["train_end"])
    val_end = pd.Timestamp(splits["val_end"])

    df_train = labeled[labeled["date"] <= train_end].copy()
    df_val = labeled[(labeled["date"] > train_end) & (labeled["date"] <= val_end)].copy()
    df_test = labeled[labeled["date"] > val_end].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError(
            f"Empty split detected: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
        )

    return df_train, df_val, df_test


def _model_path(model_type: str) -> Path:
    return ARTIFACTS_DIR / MODEL_FILES.get(model_type, MODEL_FILES["lstm"])


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
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

    Xs = []
    ys = []

    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1 : i + 1])
        if y is not None:
            ys.append(y[i])

    Xs = np.array(Xs, dtype=np.float32)

    if y is None:
        return Xs

    return Xs, np.array(ys, dtype=np.float32)


def sequence_dates(dates: pd.Series, seq_len: int = SEQUENCE_LEN) -> pd.Series:
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
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.5),
                        nn.Linear(64, 1),
                    )
                    for _ in range(n_outputs)
                ]
            )

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
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.5),
                        nn.Linear(64, 1),
                    )
                    for _ in range(n_outputs)
                ]
            )

        def forward(self, x):
            x = self.input_proj(x)
            h = self.dropout(self.encoder(x)[:, -1, :])
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
# Heat-branch loss
# ---------------------------------------------------------------------------
def _make_sample_weights(
    y_scaled: np.ndarray,
    warm_threshold_scaled: np.ndarray,
    heat_threshold_scaled: np.ndarray,
) -> np.ndarray:
    """Return per-sequence weights for warm/heat-event training."""
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    warm_threshold_scaled = np.asarray(warm_threshold_scaled, dtype=np.float32).reshape(1, -1)
    heat_threshold_scaled = np.asarray(heat_threshold_scaled, dtype=np.float32).reshape(1, -1)

    warm_event = (y_scaled >= warm_threshold_scaled).any(axis=1)
    heat_event = (y_scaled >= heat_threshold_scaled).any(axis=1)

    weights = np.ones(len(y_scaled), dtype=np.float32)
    weights[warm_event] = float(WARM_WEIGHT)
    weights[heat_event] = float(HEAT_WEIGHT)
    return weights


def _asymmetric_warm_heat_loss(
    pred,
    target,
    sample_weight,
    warm_threshold_scaled,
    heat_threshold_scaled,
):
    """Weighted MSE with extra penalty for under-predicting warm/heat targets."""
    torch, _, _, _ = _try_import_torch()

    device = pred.device
    dtype = pred.dtype

    warm_thr = torch.tensor(warm_threshold_scaled, dtype=dtype, device=device).view(1, -1)
    heat_thr = torch.tensor(heat_threshold_scaled, dtype=dtype, device=device).view(1, -1)

    horizon_w = torch.tensor(
        [HORIZON_WEIGHTS[h] for h in HORIZONS],
        dtype=dtype,
        device=device,
    ).view(1, -1)

    if sample_weight is None:
        sample_weight = torch.ones(pred.shape[0], dtype=dtype, device=device)
    else:
        sample_weight = sample_weight.to(device=device, dtype=dtype)

    sample_weight = sample_weight.view(-1, 1)

    sq_err = (pred - target) ** 2

    warm_mask = target >= warm_thr
    heat_mask = target >= heat_thr
    under_mask = pred < target

    asym = torch.ones_like(sq_err)
    asym = torch.where(
        warm_mask & under_mask,
        torch.full_like(asym, float(WARM_UNDERPRED_PENALTY)),
        asym,
    )
    asym = torch.where(
        heat_mask & under_mask,
        torch.full_like(asym, float(HEAT_UNDERPRED_PENALTY)),
        asym,
    )

    return (sq_err * horizon_w * sample_weight * asym).mean()


def train_model(
    model,
    train_loader,
    val_loader,
    warm_threshold_scaled=None,
    heat_threshold_scaled=None,
) -> Tuple[object, Dict]:
    """Train model using heat-branch objective and unweighted validation loss."""
    import copy

    torch, _, _, _ = _try_import_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    horizon_w = torch.tensor(
        [HORIZON_WEIGHTS[h] for h in HORIZONS],
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    use_heat_branch_loss = bool(
        HEAT_BRANCH_EXPERIMENT
        and warm_threshold_scaled is not None
        and heat_threshold_scaled is not None
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history: Dict = {
        "train_loss": [],
        "val_loss": [],
        "val_mae_scaled": [],
        "lr": [],
        "heat_branch_experiment": bool(use_heat_branch_loss),
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, wb = batch
                wb = wb.to(device)
            else:
                xb, yb = batch
                wb = None

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)

            if use_heat_branch_loss:
                loss = _asymmetric_warm_heat_loss(
                    pred=pred,
                    target=yb,
                    sample_weight=wb,
                    warm_threshold_scaled=warm_threshold_scaled,
                    heat_threshold_scaled=heat_threshold_scaled,
                )
            else:
                # Production/default objective.  Do not apply warm/heat sample
                # weights unless HEAT_BRANCH_EXPERIMENT is explicitly enabled,
                # because the latest heat-branch audit showed that the weighted
                # asymmetric objective worsened global LSTM performance.
                sq_err = (pred - yb) ** 2
                loss = (sq_err * horizon_w).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        val_maes = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    xb, yb, _wb = batch
                else:
                    xb, yb = batch

                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)

                sq_err = (pred - yb) ** 2
                val_loss = (sq_err * horizon_w).mean()
                val_losses.append(float(val_loss.detach().cpu().item()))
                val_maes.append(float(torch.abs(pred - yb).mean().detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_mae_scaled = float(np.mean(val_maes)) if val_maes else float("nan")

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_scaled"].append(val_mae_scaled)
        history["lr"].append(current_lr)

        print(
            f"[Part 2] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mae_scaled={val_mae_scaled:.6f} | "
            f"lr={current_lr:.6g}"
        )

        if np.isfinite(val_loss) and val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"[Part 2] Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to("cpu")
    model.eval()

    history["best_val_loss"] = float(best_val_loss)
    history["epochs_ran"] = int(len(history["train_loss"]))
    history["patience"] = int(PATIENCE)

    return model, history


# ---------------------------------------------------------------------------
# Prediction / metrics
# ---------------------------------------------------------------------------
def predict_scaled(model, X_seq: np.ndarray) -> np.ndarray:
    torch, _, _, _ = _try_import_torch()

    if len(X_seq) == 0:
        return np.empty((0, len(HORIZONS)), dtype=np.float32)

    model.eval()
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_seq), BATCH_SIZE):
            batch = torch.tensor(X_seq[start : start + BATCH_SIZE], dtype=torch.float32)
            preds.append(model(batch).cpu().numpy())

    return np.vstack(preds).astype(np.float32)


def inverse_clip_predictions(pred_scaled: np.ndarray, tgt_scaler) -> Tuple[np.ndarray, bool]:
    pred_scaled = np.asarray(pred_scaled, dtype=np.float32)
    clipped = np.clip(pred_scaled, CLIP_MIN, CLIP_MAX)
    was_clipped = bool(np.any(np.abs(clipped - pred_scaled) > 1e-8))
    pred_f = tgt_scaler.inverse_transform(clipped).astype(np.float32)
    return pred_f, was_clipped


def heat_event_diagnostics(
    pred: np.ndarray,
    true: np.ndarray,
    threshold_f: float = HEAT_EVENT_F,
) -> Dict[str, object]:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    mask = np.isfinite(pred) & np.isfinite(true)
    heat = mask & (true >= threshold_f)
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
        "predicted_heat_hits": int((pred[heat] >= threshold_f).sum()),
        "hit_rate": float((pred[heat] >= threshold_f).mean()),
        "heat_mae_f": float(np.mean(np.abs(err))),
        "heat_bias_f": float(np.mean(err)),
        "max_true_f": float(np.max(true[heat])),
        "max_pred_f": float(np.max(pred[mask])) if mask.any() else None,
    }


def evaluate_predictions(pred_f: np.ndarray, true_f: np.ndarray, prefix: str = "") -> Dict[str, object]:
    pred_f = np.asarray(pred_f, dtype=float)
    true_f = np.asarray(true_f, dtype=float)

    out: Dict[str, object] = {}
    mae_values = []

    for i, h in enumerate(HORIZONS):
        pred = pred_f[:, i]
        true = true_f[:, i]
        mask = np.isfinite(pred) & np.isfinite(true)

        if not mask.any():
            continue

        err = pred[mask] - true[mask]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        out[f"{prefix}h{h}_mae_f"] = mae
        out[f"{prefix}h{h}_rmse_f"] = rmse
        out[f"{prefix}h{h}_bias_f"] = bias
        out[f"{prefix}h{h}_n"] = int(mask.sum())
        out[f"{prefix}h{h}_heat_event_diagnostics"] = heat_event_diagnostics(pred, true)

        mae_values.append(mae)

    if mae_values:
        out[f"{prefix}avg_mae_f"] = float(np.mean(mae_values))

    return out


def make_prediction_frame(
    dates: pd.Series,
    pred_f: np.ndarray,
    true_f: np.ndarray,
    split: str,
) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(dates), "split": split})

    for i, h in enumerate(HORIZONS):
        out[f"pred_h{h}"] = pred_f[:, i] if len(pred_f) else []
        out[f"true_h{h}"] = true_f[:, i] if len(true_f) else []
        out[f"target_date_h{h}"] = pd.to_datetime(out["date"]) + pd.Timedelta(days=h)
        out[f"error_h{h}"] = out[f"pred_h{h}"] - out[f"true_h{h}"]
        out[f"abs_error_h{h}"] = out[f"error_h{h}"].abs()

    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
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

    with open(ARTIFACTS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)

    with open(ARTIFACTS_DIR / "target_scaler.pkl", "wb") as f:
        pickle.dump(tgt_scaler, f)

    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(ARTIFACTS_DIR / "part2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Part 2] Saved {model_type} model + scalers + metadata.")


def load_training_artifacts(model_type: str, input_size: int):
    torch, _, _, _ = _try_import_torch()

    feat_path = ARTIFACTS_DIR / "feature_scaler.pkl"
    tgt_path = ARTIFACTS_DIR / "target_scaler.pkl"
    meta_path = ARTIFACTS_DIR / "part2_meta.json"
    mpath = _model_path(model_type)

    for p in [feat_path, tgt_path, meta_path, mpath]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Part 2 --mode=train first.")

    with open(feat_path, "rb") as f:
        feat_scaler = pickle.load(f)

    with open(tgt_path, "rb") as f:
        tgt_scaler = pickle.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    saved_type = meta.get("model_type", model_type)
    if saved_type != model_type:
        raise ValueError(f"Saved model_type={saved_type} != requested {model_type}.")

    saved_features = meta.get("feature_cols", [])
    if len(saved_features) != input_size:
        raise ValueError(
            f"Feature count mismatch: current={input_size}, saved={len(saved_features)}. "
            "Run Part 2 with --mode=train after feature-schema changes."
        )

    model = build_model(model_type, input_size)
    model.load_state_dict(torch.load(mpath, map_location="cpu"))
    model.eval()

    return model, feat_scaler, tgt_scaler, meta


# ---------------------------------------------------------------------------
# Prediction log — idempotent upsert
# ---------------------------------------------------------------------------
def _log_path() -> Path:
    return ARTIFACTS_DIR / "prediction_log.csv"


def load_prediction_log() -> pd.DataFrame:
    p = _log_path()
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def upsert_log_row(row: Dict) -> None:
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
        row[f"forecast_h{h}"] = val

    row["forecast_source"] = "lstm_preliminary"
    row["forecast_reason"] = "awaiting_Part2B_fallback_chain"

    upsert_log_row(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_live_sequence(
    df: pd.DataFrame,
    feature_cols: List[str],
    feat_scaler,
) -> Tuple[np.ndarray, pd.Timestamp]:
    X_all = feat_scaler.transform(_clean_feature_frame(df, feature_cols)).astype(np.float32)
    X_all_seq = build_sequences(X_all, y=None, seq_len=SEQUENCE_LEN)

    if len(X_all_seq) == 0:
        raise ValueError("Not enough feature rows for live prediction sequence.")

    feature_date = pd.Timestamp(df["date"].max()).normalize()
    return X_all_seq[-1:, :, :], feature_date


def main(model_type: str = "lstm", mode: str = "train") -> int:
    set_random_seeds(RANDOM_SEED)
    torch, _, DataLoader, TensorDataset = _try_import_torch()

    model_type = str(model_type).lower().strip()
    if model_type not in {"lstm", "transformer"}:
        raise ValueError("model_type must be 'lstm' or 'transformer'.")

    mode = str(mode).lower().strip()
    if mode not in {"train", "predict"}:
        raise ValueError("mode must be 'train' or 'predict'.")

    print(f"[Part 2] model={model_type.upper()}  mode={mode}  root={PROJECT_DIR}")

    df = load_data()
    splits = load_splits()
    feature_cols = _get_feature_cols(df)
    target_cols = _target_cols()

    print(f"[Part 2] {len(df)} feature rows, {len(feature_cols)} features")

    df_train, df_val, df_test = _build_labeled_splits(df, splits)

    print(f"[Part 2] Fully labeled — Train:{len(df_train)} Val:{len(df_val)} Test:{len(df_test)}")
    print(f"[Part 2] Live feature date: {df['date'].max().date()}")

    if len(df_train) < SEQUENCE_LEN or len(df_val) < SEQUENCE_LEN:
        raise ValueError("Not enough rows to build sequences.")

    input_size = len(feature_cols)

    if mode == "train":
        from sklearn.preprocessing import MinMaxScaler

        feat_scaler = MinMaxScaler()
        tgt_scaler = MinMaxScaler()

        X_train = feat_scaler.fit_transform(_clean_feature_frame(df_train, feature_cols)).astype(np.float32)
        y_train = tgt_scaler.fit_transform(df_train[target_cols].values.astype(np.float32)).astype(np.float32)

        X_val = feat_scaler.transform(_clean_feature_frame(df_val, feature_cols)).astype(np.float32)
        y_val = tgt_scaler.transform(df_val[target_cols].values.astype(np.float32)).astype(np.float32)

        X_test = feat_scaler.transform(_clean_feature_frame(df_test, feature_cols)).astype(np.float32)
        y_test = tgt_scaler.transform(df_test[target_cols].values.astype(np.float32)).astype(np.float32)

        warm_thresh_f = np.full((1, len(HORIZONS)), WARM_EVENT_F, dtype=np.float32)
        heat_thresh_f = np.full((1, len(HORIZONS)), HEAT_EVENT_F, dtype=np.float32)

        warm_threshold_scaled = tgt_scaler.transform(warm_thresh_f)[0].astype(np.float32)
        heat_threshold_scaled = tgt_scaler.transform(heat_thresh_f)[0].astype(np.float32)

        print("[Part 2] Heat-branch thresholds:")
        for i, h in enumerate(HORIZONS):
            print(
                f"  H={h}: warm={WARM_EVENT_F:.1f}°F "
                f"(scaled={warm_threshold_scaled[i]:.4f}), "
                f"heat={HEAT_EVENT_F:.1f}°F "
                f"(scaled={heat_threshold_scaled[i]:.4f})"
            )

        X_train_seq, y_train_seq = build_sequences(X_train, y_train, SEQUENCE_LEN)
        X_val_seq, y_val_seq = build_sequences(X_val, y_val, SEQUENCE_LEN)
        X_test_seq, y_test_seq = build_sequences(X_test, y_test, SEQUENCE_LEN)

        print(
            f"[Part 2] Sequences — Train:{X_train_seq.shape} "
            f"Val:{X_val_seq.shape} Test:{X_test_seq.shape}"
        )

        train_sample_weights = _make_sample_weights(
            y_train_seq,
            warm_threshold_scaled=warm_threshold_scaled,
            heat_threshold_scaled=heat_threshold_scaled,
        )

        n_warm = int((train_sample_weights >= WARM_WEIGHT).sum())
        n_heat = int((train_sample_weights >= HEAT_WEIGHT).sum())

        if HEAT_BRANCH_EXPERIMENT:
            print(
                f"[Part 2] Weighted heat-branch training sequences: "
                f"warm_or_hot={n_warm}/{len(train_sample_weights)}, "
                f"heat={n_heat}/{len(train_sample_weights)}"
            )
            train_ds = TensorDataset(
                torch.tensor(X_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32),
                torch.tensor(train_sample_weights, dtype=torch.float32),
            )
        else:
            print(
                "[Part 2] Heat-branch training disabled. "
                f"Diagnostic warm_or_hot={n_warm}/{len(train_sample_weights)}, "
                f"heat={n_heat}/{len(train_sample_weights)}; sample weights not applied."
            )
            train_ds = TensorDataset(
                torch.tensor(X_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32),
            )

        val_ds = TensorDataset(
            torch.tensor(X_val_seq, dtype=torch.float32),
            torch.tensor(y_val_seq, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(model_type, input_size)

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            warm_threshold_scaled=warm_threshold_scaled,
            heat_threshold_scaled=heat_threshold_scaled,
        )

        val_pred_scaled = predict_scaled(model, X_val_seq)
        test_pred_scaled = predict_scaled(model, X_test_seq)

        val_pred_f, val_clipped = inverse_clip_predictions(val_pred_scaled, tgt_scaler)
        test_pred_f, test_clipped = inverse_clip_predictions(test_pred_scaled, tgt_scaler)

        val_true_f = tgt_scaler.inverse_transform(y_val_seq)
        test_true_f = tgt_scaler.inverse_transform(y_test_seq)

        val_metrics = evaluate_predictions(val_pred_f, val_true_f, prefix="val_")
        test_metrics = evaluate_predictions(test_pred_f, test_true_f, prefix="test_")

        val_mae_f = float(val_metrics.get("val_avg_mae_f", np.nan))
        test_mae_f = float(test_metrics.get("test_avg_mae_f", np.nan))

        val_dates = sequence_dates(df_val["date"], SEQUENCE_LEN)
        test_dates = sequence_dates(df_test["date"], SEQUENCE_LEN)

        val_df = make_prediction_frame(val_dates, val_pred_f, val_true_f, split="val")
        test_df = make_prediction_frame(test_dates, test_pred_f, test_true_f, split="test")
        all_pred_df = pd.concat([val_df, test_df], ignore_index=True)

        all_pred_df.to_parquet(ARTIFACTS_DIR / "val_predictions.parquet", index=False)

        print(f"[Part 2] Saved val_predictions.parquet ({len(all_pred_df)} rows, val+test).")

        meta: Dict = {
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_features": int(input_size),
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "horizons": HORIZONS,
            "sequence_len": int(SEQUENCE_LEN),
            "val_mae_f": val_mae_f,
            "test_mae_f": test_mae_f,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_output_clipped": bool(val_clipped),
            "test_output_clipped": bool(test_clipped),
            "hyperparameters": {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "lr": LR,
                "horizon_weights": HORIZON_WEIGHTS,
                "random_seed": RANDOM_SEED,
                "heat_branch_experiment": bool(HEAT_BRANCH_EXPERIMENT),
                "warm_event_f": float(WARM_EVENT_F),
                "heat_event_f": float(HEAT_EVENT_F),
                "warm_weight": float(WARM_WEIGHT),
                "heat_weight": float(HEAT_WEIGHT),
                "warm_underpred_penalty": float(WARM_UNDERPRED_PENALTY),
                "heat_underpred_penalty": float(HEAT_UNDERPRED_PENALTY),
            },
            "heat_experiment": {
                "enabled": bool(HEAT_BRANCH_EXPERIMENT),
                "decision": "disabled_after_failed_audit" if not HEAT_BRANCH_EXPERIMENT else "experimental_enabled",
                "warm_event_f": float(WARM_EVENT_F),
                "heat_event_f": float(HEAT_EVENT_F),
                "warm_weight": float(WARM_WEIGHT),
                "heat_weight": float(HEAT_WEIGHT),
                "warm_underpred_penalty": float(WARM_UNDERPRED_PENALTY),
                "heat_underpred_penalty": float(HEAT_UNDERPRED_PENALTY),
                "patience": int(PATIENCE),
                "loss": "asymmetric_warm_heat_mse",
                "validation_loss": "unweighted_horizon_weighted_mse",
                "warm_threshold_scaled_by_horizon": {
                    f"h{h}": float(warm_threshold_scaled[i])
                    for i, h in enumerate(HORIZONS)
                },
                "heat_threshold_scaled_by_horizon": {
                    f"h{h}": float(heat_threshold_scaled[i])
                    for i, h in enumerate(HORIZONS)
                },
                "n_warm_or_hot_train_sequences": int(n_warm),
                "n_heat_train_sequences": int(n_heat),
            },
        }

        save_training_artifacts(model, model_type, feat_scaler, tgt_scaler, history, meta)

    else:
        model, feat_scaler, tgt_scaler, meta = load_training_artifacts(model_type, input_size)
        saved_cols = meta.get("feature_cols", feature_cols)

        if saved_cols != feature_cols:
            raise ValueError(
                "Current feature_cols differ from saved feature_cols. "
                "Run Part 2 with --mode=train after feature-schema changes."
            )

    X_live_seq, feature_date = _build_live_sequence(df, feature_cols, feat_scaler)

    live_scaled = predict_scaled(model, X_live_seq)
    live_f, live_clipped = inverse_clip_predictions(live_scaled, tgt_scaler)
    live_pred = live_f[0]

    print("\n=== LIVE PART 2 PRELIMINARY FORECAST ===")
    for i, h in enumerate(HORIZONS):
        target_date = feature_date + pd.Timedelta(days=h)
        print(f"  H={h} ({target_date.date()}): {live_pred[i]:.1f}°F")

    write_prediction_row(
        preds_f=live_pred,
        was_clipped=bool(live_clipped),
        decision_date=pd.Timestamp.today().normalize(),
        feature_date=feature_date,
        model_type=model_type,
    )

    print("\n[Part 2] ✅ Complete.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Part 2 — Deep Learning Forecaster")
    parser.add_argument(
        "--model",
        default="lstm",
        choices=["lstm", "transformer"],
        help="Model backbone to train/load.",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "predict"],
        help="train = retrain model and write live prediction; predict = load existing model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(model_type=args.model, mode=args.mode))

