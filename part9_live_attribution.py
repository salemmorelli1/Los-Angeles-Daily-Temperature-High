#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 9 — Live Accuracy Attribution
=====================================
Backfills realized temperature highs into the prediction log, then computes
rolling accuracy metrics versus two baselines:

  1. NWS Official Forecast          — the operational NWS point forecast
  2. Climatological Persistence     — naive "tomorrow = today" forecast
  3. Climatological Normal          — historical average for that calendar date

Metrics Computed (per horizon H=1, H=3, H=5)
---------------------------------------------
  MAE        — Mean Absolute Error (°F)
  RMSE       — Root Mean Squared Error (°F)
  Bias       — Mean signed error (positive = model runs warm)
  Hit Rate   — Fraction where model beats persistence on this day
  Skill Score — 1 - (model_MAE / persistence_MAE)  (>0 = better than persistence)
  Coverage   — Fraction of observed values within BNN 90% CI (if available)

Artifacts Written
-----------------
  artifacts_part9/
      live_attribution_report.json    — full metrics breakdown
      attribution_tape.parquet        — row-level actuals + error scores
      skill_history.parquet           — rolling 30-day and 90-day skill scores
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

SCHEMA_VERSION = "1.0.0"
HORIZONS = [1, 3, 5]

# Open-Meteo archive for realized temperature backfill
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT = 33.9425
LON = -118.4081


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_prediction_log() -> Optional[pd.DataFrame]:
    """Load the canonical prediction log from Part 2 or Part 3."""
    for p in [PART2_DIR / "prediction_log.csv", PART3_DIR / "prediction_log.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
                df["feature_date"] = pd.to_datetime(df.get("feature_date", df["decision_date"]), errors="coerce")
                return df.sort_values("decision_date").reset_index(drop=True)
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
# Realized temperature backfill
# ---------------------------------------------------------------------------
def fetch_realized_temps(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch observed daily high temperatures from Open-Meteo archive."""
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = min(end_date, pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if pd.Timestamp(start_str) >= pd.Timestamp(end_str):
        return pd.DataFrame()

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
    """
    Match each prediction row's target date with the realized temperature.
    Target date for H-day horizon = decision_date + H calendar days.
    """
    if df_realized.empty:
        return df_log

    realized_map = dict(zip(df_realized["date"], df_realized["realized_high_f"]))
    today = pd.Timestamp.today().normalize()

    for idx, row in df_log.iterrows():
        dd = pd.Timestamp(row["decision_date"]).normalize()
        for h in HORIZONS:
            target_date = dd + pd.Timedelta(days=h)
            realized_col = f"realized_h{h}"
            if realized_col not in df_log.columns:
                df_log[realized_col] = np.nan

            if target_date > today:
                continue  # Not yet observable

            realized = realized_map.get(target_date)
            if realized is not None and not np.isfinite(float(df_log.at[idx, realized_col] or np.nan)):
                df_log.at[idx, realized_col] = float(realized)

    return df_log


# ---------------------------------------------------------------------------
# Climatological persistence baseline
# ---------------------------------------------------------------------------
def compute_persistence_baseline(df_log: pd.DataFrame, df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Add persistence forecast columns: the last observed temperature at decision time.
    """
    if df_hist.empty or "temp_high_f" not in df_hist.columns:
        return df_log

    hist_map = dict(zip(df_hist["date"], df_hist["temp_high_f"]))

    for idx, row in df_log.iterrows():
        dd = pd.Timestamp(row["decision_date"]).normalize()
        last_obs_date = dd - pd.Timedelta(days=1)  # last known at decision time

        # Find most recent available observation
        for lookback in range(0, 5):
            check_date = last_obs_date - pd.Timedelta(days=lookback)
            if check_date in hist_map:
                df_log.at[idx, "persistence_temp_f"] = float(hist_map[check_date])
                break

    return df_log


# ---------------------------------------------------------------------------
# Climatological normal baseline
# ---------------------------------------------------------------------------
def compute_climatological_normal(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-year rolling day-of-year average temperature."""
    if df_hist.empty or "temp_high_f" not in df_hist.columns:
        return pd.DataFrame()

    df = df_hist.copy()
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear
    clim = df.groupby("doy")["temp_high_f"].mean().reset_index()
    clim.columns = ["doy", "clim_normal_f"]
    return clim


# ---------------------------------------------------------------------------
# NWS forecast accuracy
# ---------------------------------------------------------------------------
def compute_nws_accuracy(
    df_log: pd.DataFrame,
    nws_forecast: Dict,
) -> Dict[str, float]:
    """Compare NWS official forecast (H=1 only) vs realized temperature."""
    daily_highs = nws_forecast.get("daily_high_f", {})
    if not daily_highs:
        return {}

    errors = []
    for date_str, nws_high in daily_highs.items():
        target_date = pd.Timestamp(date_str).normalize()
        # Find matching realized from prediction log
        for idx, row in df_log.iterrows():
            dd = pd.Timestamp(row["decision_date"]).normalize()
            if dd + pd.Timedelta(days=1) == target_date:
                realized = float(row.get("realized_h1", np.nan))
                if np.isfinite(realized) and np.isfinite(float(nws_high)):
                    errors.append(abs(realized - float(nws_high)))
                break

    if not errors:
        return {}

    return {
        "nws_h1_mae_f": float(np.mean(errors)),
        "nws_h1_rmse_f": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        "nws_n_days": len(errors),
    }


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(df_log: pd.DataFrame, clim_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute MAE, RMSE, bias, skill score per horizon."""
    clim_map = dict(zip(clim_df["doy"], clim_df["clim_normal_f"])) if not clim_df.empty else {}
    metrics: Dict[str, Any] = {}

    for h in HORIZONS:
        pred_col = f"target_h{h}"
        real_col = f"realized_h{h}"

        if pred_col not in df_log.columns or real_col not in df_log.columns:
            continue

        pred = pd.to_numeric(df_log[pred_col], errors="coerce")
        real = pd.to_numeric(df_log[real_col], errors="coerce")
        pers = pd.to_numeric(df_log.get("persistence_temp_f", pd.Series(np.nan, index=df_log.index)), errors="coerce")

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

        # Skill score vs persistence
        pers_mask = mask & pers.notna()
        if pers_mask.sum() > 0:
            pers_errors = (real[pers_mask] - pers[pers_mask]).abs()
            pers_mae = float(pers_errors.mean())
            skill_vs_persistence = float(1 - mae / pers_mae) if pers_mae > 0 else 0.0
        else:
            pers_mae = None
            skill_vs_persistence = None

        # Skill score vs climatological normal
        doys = pd.to_datetime(df_log.loc[mask, "decision_date"]).dt.dayofyear + h
        clim_preds = doys.map(lambda d: clim_map.get(int(d % 365) or 365, np.nan))
        clim_mask = clim_preds.notna()
        if clim_mask.sum() > 0:
            clim_errors = (real[mask][clim_mask.values] - clim_preds[clim_mask]).abs()
            clim_mae = float(clim_errors.mean())
            skill_vs_clim = float(1 - mae / clim_mae) if clim_mae > 0 else 0.0
        else:
            clim_mae = None
            skill_vs_clim = None

        # Hit rate (model beats persistence on each day)
        if pers_mask.sum() > 0:
            model_day_errors = abs_errors[pers_mask]
            pers_day_errors = (real[pers_mask] - pers[pers_mask]).abs()
            hit_rate = float((model_day_errors < pers_day_errors).mean())
        else:
            hit_rate = None

        # BNN coverage (if available)
        lo_col = f"bnn_lo90_h{h}"
        hi_col = f"bnn_hi90_h{h}"
        coverage = None
        if lo_col in df_log.columns and hi_col in df_log.columns:
            lo = pd.to_numeric(df_log[lo_col], errors="coerce")
            hi = pd.to_numeric(df_log[hi_col], errors="coerce")
            cov_mask = mask & lo.notna() & hi.notna()
            if cov_mask.sum() > 0:
                in_ci = (real[cov_mask] >= lo[cov_mask]) & (real[cov_mask] <= hi[cov_mask])
                coverage = float(in_ci.mean())

        metrics[f"h{h}"] = {
            "n_samples": n,
            "mae_f": round(mae, 3),
            "rmse_f": round(rmse, 3),
            "bias_f": round(bias, 3),
            "persistence_mae_f": round(pers_mae, 3) if pers_mae else None,
            "skill_vs_persistence": round(skill_vs_persistence, 4) if skill_vs_persistence is not None else None,
            "clim_mae_f": round(clim_mae, 3) if clim_mae else None,
            "skill_vs_climatology": round(skill_vs_clim, 4) if skill_vs_clim is not None else None,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "bnn_coverage_90pct": round(coverage, 4) if coverage is not None else None,
        }

    return metrics


# ---------------------------------------------------------------------------
# Rolling skill history
# ---------------------------------------------------------------------------
def compute_rolling_skill(df_tape: pd.DataFrame) -> pd.DataFrame:
    """Compute 30-day and 90-day rolling MAE per horizon."""
    rows = []
    df_tape = df_tape.sort_values("decision_date")

    for i in range(len(df_tape)):
        row = {"date": df_tape["decision_date"].iloc[i]}

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
                if mask.sum() >= 5:
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

    print(f"[Part 9] Loaded {len(df_log)} prediction rows")

    df_hist = load_historical()
    nws_forecast = load_nws_forecast()
    clim_df = compute_climatological_normal(df_hist)

    # Backfill realized temperatures
    min_dd = df_log["decision_date"].dropna().min()
    max_target = df_log["decision_date"].dropna().max() + pd.Timedelta(days=max(HORIZONS) + 2)
    df_realized = fetch_realized_temps(min_dd, max_target)

    df_log = backfill_realized(df_log, df_realized)
    df_log = compute_persistence_baseline(df_log, df_hist)

    # Compute metrics
    print("[Part 9] Computing accuracy metrics...")
    metrics = compute_metrics(df_log, clim_df)
    nws_metrics = compute_nws_accuracy(df_log, nws_forecast)

    # Print summary
    print("\n=== ATTRIBUTION SUMMARY ===")
    for h in HORIZONS:
        m = metrics.get(f"h{h}", {})
        n = m.get("n_samples", 0)
        if n == 0:
            print(f"  H={h}: No realized samples yet")
            continue
        mae = m.get("mae_f")
        bias = m.get("bias_f")
        skill = m.get("skill_vs_persistence")
        cov = m.get("bnn_coverage_90pct")

        parts = [f"n={n}", f"MAE={mae:.2f}°F", f"Bias={bias:+.2f}°F"]
        if skill is not None:
            parts.append(f"Skill={skill:+.3f}")
        if cov is not None:
            parts.append(f"BNN_cov={cov:.0%}")
        print(f"  H={h}: {' | '.join(parts)}")

    if nws_metrics:
        print(f"\n  NWS H=1: MAE={nws_metrics['nws_h1_mae_f']:.2f}°F (n={nws_metrics['nws_n_days']})")

    # Save attribution tape
    tape_path = ARTIFACTS_DIR / "attribution_tape.parquet"
    df_log.to_parquet(tape_path, index=False)
    print(f"\n[Part 9] Saved attribution_tape.parquet ({len(df_log)} rows)")

    # Save skill history
    skill_df = compute_rolling_skill(df_log)
    skill_df.to_parquet(ARTIFACTS_DIR / "skill_history.parquet", index=False)
    print(f"[Part 9] Saved skill_history.parquet ({len(skill_df)} rows)")

    # Save full report
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_prediction_rows": len(df_log),
        "realized_samples_by_horizon": {
            f"h{h}": int(pd.to_numeric(df_log.get(f"realized_h{h}", pd.Series()), errors="coerce").notna().sum())
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
            "overall_skill": float(np.mean([
                metrics[f"h{h}"]["skill_vs_persistence"]
                for h in HORIZONS
                if metrics.get(f"h{h}", {}).get("skill_vs_persistence") is not None
            ])) if any(
                metrics.get(f"h{h}", {}).get("skill_vs_persistence") is not None
                for h in HORIZONS
            ) else None,
        },
    }

    with open(ARTIFACTS_DIR / "live_attribution_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[Part 9] Saved live_attribution_report.json")

    # Write updated prediction log back to source
    log_path = PART2_DIR / "prediction_log.csv"
    if log_path.exists():
        df_log.to_csv(log_path, index=False)
        print("[Part 9] Updated prediction_log.csv with realized temperatures")

    print(f"\n[Part 9] ✅ Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
