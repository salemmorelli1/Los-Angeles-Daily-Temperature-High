#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 9 — Live Accuracy Attribution
===================================
Backfills realized temperature highs into the prediction log, then computes
rolling accuracy metrics versus two baselines:

  1. NWS Official Forecast          — operational NWS point forecast
  2. Climatological Persistence     — last known observed high at feature_date
  3. Climatological Normal          — historical average for target calendar date

Production clock
----------------
For every row, the forecast target date is:
    target_date_h = feature_date + h calendar days

If older prediction logs do not have explicit target_date_h* columns, the code
falls back to feature_date + h. It only falls back to decision_date + h if
feature_date is missing.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

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
PART2_DIR = PROJECT_DIR / "artifacts_part2"
PART3_DIR = PROJECT_DIR / "artifacts_part3"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part9"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.1.0"
HORIZONS = [1, 3, 5]

OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT = 33.9425
LON = -118.4081


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_prediction_log() -> Optional[pd.DataFrame]:
    """Load prediction log from Part 2 first, then Part 3 as fallback."""
    for p in [PART2_DIR / "prediction_log.csv", PART3_DIR / "prediction_log.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce").dt.normalize()
                if "feature_date" in df.columns:
                    df["feature_date"] = pd.to_datetime(df["feature_date"], errors="coerce").dt.normalize()
                else:
                    df["feature_date"] = df["decision_date"]
                for h in HORIZONS:
                    col = f"target_date_h{h}"
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
                return df.sort_values(["feature_date", "decision_date"]).reset_index(drop=True)
    return None


def load_historical() -> pd.DataFrame:
    path = PART0_DIR / "historical_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def load_nws_forecast() -> Dict:
    path = PART0_DIR / "nws_official_forecast.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Target-date helpers
# ---------------------------------------------------------------------------
def target_date_for_row(row: pd.Series, h: int) -> pd.Timestamp:
    explicit_col = f"target_date_h{h}"
    if explicit_col in row.index and pd.notna(row.get(explicit_col)):
        return pd.Timestamp(row[explicit_col]).normalize()

    if "feature_date" in row.index and pd.notna(row.get("feature_date")):
        return pd.Timestamp(row["feature_date"]).normalize() + pd.Timedelta(days=h)

    return pd.Timestamp(row["decision_date"]).normalize() + pd.Timedelta(days=h)


def add_missing_target_date_columns(df_log: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        col = f"target_date_h{h}"
        if col not in df_log.columns:
            df_log[col] = [target_date_for_row(row, h) for _, row in df_log.iterrows()]
        else:
            df_log[col] = pd.to_datetime(df_log[col], errors="coerce")
            missing = df_log[col].isna()
            if missing.any():
                df_log.loc[missing, col] = [
                    target_date_for_row(row, h) for _, row in df_log.loc[missing].iterrows()
                ]
    return df_log


# ---------------------------------------------------------------------------
# Realized temperature backfill
# ---------------------------------------------------------------------------
def fetch_realized_temps(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch observed daily high temperatures from Open-Meteo archive."""
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame()

    start_date = pd.Timestamp(start_date).normalize()
    end_date = min(pd.Timestamp(end_date).normalize(), pd.Timestamp.today().normalize() - pd.Timedelta(days=1))

    # Allow a one-day range; only skip when start is strictly after end.
    if start_date > end_date:
        return pd.DataFrame()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"[Part 9] Fetching realized temps: {start_str} → {end_str}")
    try:
        resp = requests.get(
            OM_ARCHIVE_URL,
            params={
                "latitude": LAT,
                "longitude": LON,
                "start_date": start_str,
                "end_date": end_str,
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": "America/Los_Angeles",
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        daily = payload.get("daily", {})
        if not daily:
            return pd.DataFrame()

        df = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
        df = df.rename(columns={"temperature_2m_max": "realized_high_f"})
        return df[["date", "realized_high_f"]].dropna()
    except Exception as exc:
        print(f"  [WARN] Realized temp fetch failed: {exc}")
        return pd.DataFrame()


def backfill_realized(
    df_log: pd.DataFrame,
    df_realized: pd.DataFrame,
) -> pd.DataFrame:
    """Match each prediction row's feature-date-based target date to realized high."""
    df_log = add_missing_target_date_columns(df_log)

    if df_realized.empty:
        return df_log

    realized_map = {
        pd.Timestamp(k).normalize(): float(v)
        for k, v in zip(df_realized["date"], df_realized["realized_high_f"])
    }
    today = pd.Timestamp.today().normalize()

    for idx, row in df_log.iterrows():
        for h in HORIZONS:
            target_date = target_date_for_row(row, h)
            realized_col = f"realized_h{h}"

            if realized_col not in df_log.columns:
                df_log[realized_col] = np.nan

            if target_date > today:
                continue

            current = pd.to_numeric(pd.Series([df_log.at[idx, realized_col]]), errors="coerce").iloc[0]
            if np.isfinite(current):
                continue

            realized = realized_map.get(target_date)
            if realized is not None and np.isfinite(realized):
                df_log.at[idx, realized_col] = float(realized)

    return df_log


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def compute_persistence_baseline(df_log: pd.DataFrame, df_hist: pd.DataFrame) -> pd.DataFrame:
    """Add persistence forecast columns using the last known high at feature_date."""
    if df_hist.empty or "temp_high_f" not in df_hist.columns:
        return df_log

    hist_map = {
        pd.Timestamp(k).normalize(): float(v)
        for k, v in zip(df_hist["date"], df_hist["temp_high_f"])
        if pd.notna(v)
    }

    for idx, row in df_log.iterrows():
        fd = row.get("feature_date", row.get("decision_date"))
        if pd.isna(fd):
            continue
        feature_date = pd.Timestamp(fd).normalize()

        # Find most recent available observation at or before feature_date.
        for lookback in range(0, 10):
            check_date = feature_date - pd.Timedelta(days=lookback)
            if check_date in hist_map:
                val = hist_map[check_date]
                df_log.at[idx, "persistence_temp_f"] = val
                for h in HORIZONS:
                    df_log.at[idx, f"persistence_h{h}"] = val
                break

    return df_log


def compute_climatological_normal(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Compute day-of-year average high temperature."""
    if df_hist.empty or "temp_high_f" not in df_hist.columns:
        return pd.DataFrame()

    df = df_hist.copy()
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear
    clim = df.groupby("doy")["temp_high_f"].mean().reset_index()
    clim.columns = ["doy", "clim_normal_f"]
    return clim


def climatology_for_dates(dates: pd.Series, clim_map: Dict[int, float]) -> pd.Series:
    doys = pd.to_datetime(dates).dt.dayofyear
    return doys.map(lambda d: clim_map.get(int(d), np.nan))


# ---------------------------------------------------------------------------
# NWS benchmark
# ---------------------------------------------------------------------------
def compute_nws_accuracy(
    df_log: pd.DataFrame,
    nws_forecast: Dict,
) -> Dict[str, float]:
    """Compare NWS official forecast (H=1 only) to realized high where dates overlap."""
    daily_highs = nws_forecast.get("daily_high_f", {})
    if not daily_highs:
        return {}

    nws_by_date = {
        pd.Timestamp(k).normalize(): float(v)
        for k, v in daily_highs.items()
        if pd.notna(v)
    }

    errors = []
    for _, row in df_log.iterrows():
        target_date = target_date_for_row(row, 1)
        nws_high = nws_by_date.get(target_date)
        realized = pd.to_numeric(pd.Series([row.get("realized_h1", np.nan)]), errors="coerce").iloc[0]
        if nws_high is not None and np.isfinite(nws_high) and np.isfinite(realized):
            errors.append(realized - nws_high)

    if not errors:
        return {}

    errors = np.array(errors, dtype=float)
    return {
        "nws_h1_mae_f": float(np.mean(np.abs(errors))),
        "nws_h1_rmse_f": float(np.sqrt(np.mean(errors ** 2))),
        "nws_h1_bias_f": float(np.mean(errors)),
        "nws_n_days": int(len(errors)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(df_log: pd.DataFrame, clim_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute MAE, RMSE, bias, skill score per horizon."""
    df_log = add_missing_target_date_columns(df_log)
    clim_map = dict(zip(clim_df["doy"], clim_df["clim_normal_f"])) if not clim_df.empty else {}
    metrics: Dict[str, Any] = {}

    for h in HORIZONS:
        pred_col = f"target_h{h}"
        real_col = f"realized_h{h}"
        pers_col = f"persistence_h{h}" if f"persistence_h{h}" in df_log.columns else "persistence_temp_f"

        if pred_col not in df_log.columns or real_col not in df_log.columns:
            metrics[f"h{h}"] = {"n_samples": 0}
            continue

        pred = pd.to_numeric(df_log[pred_col], errors="coerce")
        real = pd.to_numeric(df_log[real_col], errors="coerce")
        pers = pd.to_numeric(df_log.get(pers_col, pd.Series(np.nan, index=df_log.index)), errors="coerce")

        mask = pred.notna() & real.notna()
        n = int(mask.sum())
        if n == 0:
            metrics[f"h{h}"] = {"n_samples": 0}
            continue

        errors = real[mask] - pred[mask]
        abs_errors = errors.abs()

        mae = float(abs_errors.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        bias = float(errors.mean())

        # Persistence skill.
        pers_mask = mask & pers.notna()
        if int(pers_mask.sum()) > 0:
            pers_day_errors = (real[pers_mask] - pers[pers_mask]).abs()
            pers_mae = float(pers_day_errors.mean())
            skill_vs_persistence = float(1 - mae / pers_mae) if pers_mae > 0 else 0.0

            model_day_errors = (real[pers_mask] - pred[pers_mask]).abs()
            hit_rate = float((model_day_errors < pers_day_errors).mean())
        else:
            pers_mae = None
            skill_vs_persistence = None
            hit_rate = None

        # Climatology skill by actual target date.
        target_dates = pd.to_datetime(df_log.loc[mask, f"target_date_h{h}"])
        clim_preds = climatology_for_dates(target_dates, clim_map)
        clim_mask = clim_preds.notna()
        if int(clim_mask.sum()) > 0:
            clim_errors = (real[mask].reset_index(drop=True)[clim_mask.values] - clim_preds[clim_mask].reset_index(drop=True)).abs()
            clim_mae = float(clim_errors.mean())
            skill_vs_clim = float(1 - mae / clim_mae) if clim_mae > 0 else 0.0
        else:
            clim_mae = None
            skill_vs_clim = None

        # BNN coverage.
        lo_col = f"bnn_lo90_h{h}"
        hi_col = f"bnn_hi90_h{h}"
        coverage = None
        if lo_col in df_log.columns and hi_col in df_log.columns:
            lo = pd.to_numeric(df_log[lo_col], errors="coerce")
            hi = pd.to_numeric(df_log[hi_col], errors="coerce")
            cov_mask = mask & lo.notna() & hi.notna()
            if int(cov_mask.sum()) > 0:
                in_ci = (real[cov_mask] >= lo[cov_mask]) & (real[cov_mask] <= hi[cov_mask])
                coverage = float(in_ci.mean())

        metrics[f"h{h}"] = {
            "n_samples": n,
            "mae_f": round(mae, 3),
            "rmse_f": round(rmse, 3),
            "bias_f": round(bias, 3),
            "persistence_mae_f": round(pers_mae, 3) if pers_mae is not None else None,
            "skill_vs_persistence": round(skill_vs_persistence, 4) if skill_vs_persistence is not None else None,
            "clim_mae_f": round(clim_mae, 3) if clim_mae is not None else None,
            "skill_vs_climatology": round(skill_vs_clim, 4) if skill_vs_clim is not None else None,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "bnn_coverage_90pct": round(coverage, 4) if coverage is not None else None,
        }

    return metrics


def compute_rolling_skill(df_tape: pd.DataFrame) -> pd.DataFrame:
    """Compute 30-day and 90-day rolling MAE per horizon."""
    rows = []
    df_tape = add_missing_target_date_columns(df_tape).sort_values(["feature_date", "decision_date"])

    for i in range(len(df_tape)):
        row = {
            "date": df_tape["decision_date"].iloc[i],
            "feature_date": df_tape["feature_date"].iloc[i],
        }

        for h in HORIZONS:
            pred_col = f"target_h{h}"
            real_col = f"realized_h{h}"
            if pred_col not in df_tape.columns or real_col not in df_tape.columns:
                continue

            for window, label in [(30, "30d"), (90, "90d")]:
                start_idx = max(0, i - window + 1)
                slice_df = df_tape.iloc[start_idx:i + 1]
                pred = pd.to_numeric(slice_df[pred_col], errors="coerce")
                real = pd.to_numeric(slice_df[real_col], errors="coerce")
                mask = pred.notna() & real.notna()
                if int(mask.sum()) >= 5:
                    mae = float((real[mask] - pred[mask]).abs().mean())
                    row[f"h{h}_mae_{label}"] = round(mae, 3)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 9] Project root: {PROJECT_DIR}")

    df_log = load_prediction_log()
    if df_log is None or df_log.empty:
        print("[Part 9] Prediction log is empty. Run Part 2 first.")
        return 1

    df_log = add_missing_target_date_columns(df_log)
    print(f"[Part 9] Loaded {len(df_log)} prediction rows")

    df_hist = load_historical()
    nws_forecast = load_nws_forecast()
    clim_df = compute_climatological_normal(df_hist)

    target_date_cols = [f"target_date_h{h}" for h in HORIZONS]
    min_target = pd.to_datetime(df_log[target_date_cols].stack(), errors="coerce").dropna().min()
    max_target = pd.to_datetime(df_log[target_date_cols].stack(), errors="coerce").dropna().max()

    df_realized = fetch_realized_temps(min_target, max_target)

    df_log = backfill_realized(df_log, df_realized)
    df_log = compute_persistence_baseline(df_log, df_hist)

    print("[Part 9] Computing accuracy metrics...")
    metrics = compute_metrics(df_log, clim_df)
    nws_metrics = compute_nws_accuracy(df_log, nws_forecast)

    print("\n=== ATTRIBUTION SUMMARY ===")
    for h in HORIZONS:
        m = metrics.get(f"h{h}", {})
        n = m.get("n_samples", 0)
        if n == 0:
            print(f"  H={h}: No realized samples yet")
            continue

        parts = [
            f"n={n}",
            f"MAE={m.get('mae_f'):.2f}°F",
            f"Bias={m.get('bias_f'):+.2f}°F",
        ]
        if m.get("skill_vs_persistence") is not None:
            parts.append(f"Skill={m['skill_vs_persistence']:+.3f}")
        if m.get("bnn_coverage_90pct") is not None:
            parts.append(f"BNN_cov={m['bnn_coverage_90pct']:.0%}")
        print(f"  H={h}: {' | '.join(parts)}")

    if nws_metrics:
        print(f"\n  NWS H=1: MAE={nws_metrics['nws_h1_mae_f']:.2f}°F (n={nws_metrics['nws_n_days']})")

    tape_path = ARTIFACTS_DIR / "attribution_tape.parquet"
    df_log.to_parquet(tape_path, index=False)
    print(f"\n[Part 9] Saved attribution_tape.parquet ({len(df_log)} rows)")

    skill_df = compute_rolling_skill(df_log)
    skill_df.to_parquet(ARTIFACTS_DIR / "skill_history.parquet", index=False)
    print(f"[Part 9] Saved skill_history.parquet ({len(skill_df)} rows)")

    skill_values = [
        metrics[f"h{h}"]["skill_vs_persistence"]
        for h in HORIZONS
        if metrics.get(f"h{h}", {}).get("skill_vs_persistence") is not None
    ]

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.now().isoformat(),
        "target_clock": "target_date_h = feature_date + h calendar days",
        "n_prediction_rows": len(df_log),
        "realized_samples_by_horizon": {
            f"h{h}": int(pd.to_numeric(df_log.get(f"realized_h{h}", pd.Series(dtype=float)), errors="coerce").notna().sum())
            for h in HORIZONS
        },
        "metrics_by_horizon": metrics,
        "nws_baseline_metrics": nws_metrics,
        "climatological_baseline": {
            f"h{h}": {
                "clim_mae_f": metrics.get(f"h{h}", {}).get("clim_mae_f"),
                "skill_vs_climatology": metrics.get(f"h{h}", {}).get("skill_vs_climatology"),
            }
            for h in HORIZONS
        },
        "summary": {
            "best_horizon": min(
                [h for h in HORIZONS if metrics.get(f"h{h}", {}).get("n_samples", 0) > 0],
                key=lambda h: metrics.get(f"h{h}", {}).get("mae_f", 999),
                default=None,
            ),
            "overall_skill": float(np.mean(skill_values)) if skill_values else None,
        },
    }

    with open(ARTIFACTS_DIR / "live_attribution_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[Part 9] Saved live_attribution_report.json")

    # Write updated prediction log back to source.
    log_path = PART2_DIR / "prediction_log.csv"
    if log_path.exists():
        df_log.to_csv(log_path, index=False)
        print("[Part 9] Updated prediction_log.csv with realized temperatures")

    print("\n[Part 9] ✅ Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
