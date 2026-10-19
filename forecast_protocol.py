"""Shared temporal and clock rules for the LA temperature forecast pipeline."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
PURGE_DAYS = 5
TRAINING_PROTOCOL_VERSION = "purged-frozen-regime-v3"
MIN_PAIRED_SKILL_SAMPLES = 30


def file_sha256(path: Path) -> str:
    """Identify the exact fitted artifact, independently of its filename."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_lineage_errors(root: Path) -> list[str]:
    """Validate the checkpoint, preprocessing bundle and saved fit boundary."""
    try:
        meta = json.loads((root / "artifacts_part2/part2_meta.json").read_text())
        regime_meta = json.loads((root / "artifacts_part6/regime_meta.json").read_text())
        if meta.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
            return ["checkpoint_requires_protocol_retrain"]
        cutoff = pd.Timestamp(meta["training_splits"]["train_effective_end"])
        fit_end = pd.Timestamp(regime_meta["fit_end_date"])
        if pd.isna(cutoff) or pd.isna(fit_end) or fit_end > cutoff:
            return ["regime_fit_crosses_purged_training_boundary"]
        paths = {
            "regime_model_sha256": root / "artifacts_part6/regime_model.pkl",
            "checkpoint_sha256": root / f"artifacts_part2/{meta['model_type']}_model.pt",
            "feature_scaler_sha256": root / "artifacts_part2/feature_scaler.pkl",
            "target_scaler_sha256": root / "artifacts_part2/target_scaler.pkl",
            "val_predictions_sha256": root / "artifacts_part2/val_predictions.parquet",
        }
        return [
            f"{key}_mismatch" for key, path in paths.items()
            if not meta.get(key) or file_sha256(path) != meta[key]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"training_lineage_unreadable:{type(exc).__name__}"]


def forecast_is_published(row: pd.Series) -> bool:
    """Published forecast inputs are immutable; outcomes may still be backfilled.

    Legacy rows use their recorded governance time, without inventing a new
    issue time or relabeling their historical provenance.
    """
    if str(row.get("publish_mode", "")).upper() not in {"NORMAL", "CAUTION"}:
        return False
    return any(
        pd.notna(row.get(key)) and str(row.get(key, "")).strip() != ""
        for key in ["forecast_finalized_at", "governance_run_at"]
    )


def protocol_published_history(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Select first-issued paper forecasts from one method, without relabeling."""
    versions = df.get("training_protocol_version", pd.Series(index=df.index, dtype=object))
    published = pd.Series([forecast_is_published(row) for _, row in df.iterrows()], index=df.index, dtype=bool)
    return df.loc[versions.eq(version) & published].copy()


def pacific_today() -> date:
    """Return today's calendar date in the forecast's Los Angeles timezone."""
    return datetime.now(PACIFIC_TZ).date()


def pacific_today_timestamp() -> pd.Timestamp:
    """Return a timezone-naive midnight timestamp for the Los Angeles date."""
    return pd.Timestamp(pacific_today()).normalize()


def purged_labeled_splits(
    df: pd.DataFrame,
    splits: Dict,
    target_cols: Sequence[str],
    purge_days: int = PURGE_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological labeled splits with a calendar-horizon purge.

    The stored boundaries define the original 70/15/15 row splits. Remove the
    final purge_days calendar dates from training and validation so target
    dates cannot overlap the following split. The test interval stays intact.
    """
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative")
    labeled = df.dropna(subset=list(target_cols)).copy()
    dates = pd.to_datetime(labeled["date"], errors="coerce").dt.normalize()
    train_end = pd.Timestamp(splits["train_end"]).normalize()
    val_end = pd.Timestamp(splits["val_end"]).normalize()
    train_cutoff = train_end - pd.Timedelta(days=purge_days)
    val_cutoff = val_end - pd.Timedelta(days=purge_days)

    train = labeled.loc[dates <= train_cutoff].copy()
    val = labeled.loc[(dates > train_end) & (dates <= val_cutoff)].copy()
    test_mask = dates > val_end
    if "test_end" in splits:
        test_mask &= dates <= pd.Timestamp(splits["test_end"]).normalize()
    test = labeled.loc[test_mask].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Purged split is empty: "
            f"train={len(train)}, val={len(val)}, test={len(test)}, "
            f"purge_days={purge_days}"
        )
    return train, val, test


def moving_block_bootstrap_mean_ci(
    values: Sequence[float],
    block_length: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Deterministic percentile CI for a mean using contiguous row blocks."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"lower": float("nan"), "upper": float("nan"), "n": 0}
    if block_length < 1 or n_boot < 1:
        raise ValueError("block_length and n_boot must be positive")

    n = int(arr.size)
    block = min(int(block_length), n)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "n": n,
        "block_length_rows": block,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
