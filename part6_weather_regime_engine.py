#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 9 — Live Accuracy Attribution
=====================================
Backfills realized temperature highs and computes rolling accuracy metrics
versus three baselines: NWS official forecast, persistence, and climatology.

Production clock
----------------
  target_date_h = feature_date + h calendar days

Attribution is computed against forecast_h* (the canonical fallback-chain
forecast written by Part 2B), falling back to target_h* (raw LSTM) when
forecast_h* is absent or NaN.

Minimum-sample guards
---------------------
Skill scores and comparative metrics are only displayed and reported once
enough unique realized samples exist:
  SKILL_MIN_SAMPLES   = 30  — minimum for displaying skill / hit-rate
  PROMOTE_MIN_SAMPLES = 60  — minimum for model promotion decisions

Idempotency
-----------
The prediction log is updated in-place (no new rows created). The
attribution_tape and skill_history are rebuilt from scratch each run.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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

SCHEMA_VERSION = "1.2.0"
HORIZONS = [1, 3, 5]
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT = 33.9425
LON = -118.4081

SKILL_MIN_SAMPLES = 30    # report skill only above this
PROMOTE_MIN_SAMPLES = 60  # model promotion threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_prediction_log() -> Optional[pd.DataFrame]:
    for p in [PART2_DIR / "prediction_log.csv", PART3_DIR / "prediction_log.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                df["decision_date"] = pd.to_datetime(
                    df["decision_date"], errors="coerce"
                ).dt.normalize()
                if "feature_date" in df.columns:
                    df["feature_date"] = pd.to_datetime(
                        df["feature_date"], errors="coerce"
                    ).dt.normalize()
                else:
                    df["feature_date"] = df["decision_date"]
                for h in HORIZONS:
                    col = f"target_date_h{h}"
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
                return df.sort_values(
                    ["feature_date", "decision_date"]
                ).reset_index(drop=True)
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
# Canonical prediction column
# ---------------------------------------------------------------------------
def _prediction_col_for(df_log: pd.DataFrame, h: int) -> str:
    """Return the best available prediction column for horizon h.

    Preference: forecast_h* (canonical fallback chain from Part 2B)
                then target_h* (raw LSTM from Part 2)
    """
    fc = f"forecast_h{h}"
    if fc in df_log.columns and df_log[fc].notna().any():
        return fc
    return f"target_h{h}"


# ---------------------------------------------------------------------------
# Target-date helpers
# ---------------------------------------------------------------------------
def target_date_for_row(row: pd.Series, h: int) -> pd.Timestamp:
    explicit = f"target_date_h{h}"
    if explicit in row.index and pd.notna(row.get(explicit)):
        return pd.Timestamp(row[explicit]).normalize()
    if "feature_date" in row.index and pd.notna(row.get("feature_date")):
        return pd.Timestamp(row["feature_date"]).normalize() + pd.Timedelta(days=h)
    return pd.Timestamp(row["decision_date"]).normalize() + pd.Timedelta(days=h)


def add_missing_target_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        col = f"target_date_h{h}"
        if col not in df.columns:
            df[col] = [target_date_for_row(row, h) for _, row in df.iterrows()]
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            missing = df[col].isna()
            if missing.any():
                df.loc[missing, col] = [
                    target_date_for_row(row, h) for _, row in df.loc[missing].iterrows()
                ]
    return df


# ---------------------------------------------------------------------------
# Realized temperature backfill
# ---------------------------------------------------------------------------
def fetch_realized_temps(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    start = pd.Timestamp(start).normalize()
    # Cap at yesterday — future dates are never in the archive
    end = min(
        pd.Timestamp(end).normalize(),
        pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
    )
    if start > end:
        return pd.DataFrame()

    print(f"[Part 9] Fetching realized temps: {start.date()} → {end.date()}")
    try:
        resp = requests.get(
            OM_ARCHIVE_URL,
            params={
                "latitude": LAT, "longitude": LON,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": "America/Los_Angeles",
            },
            timeout=20,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        if not daily:
            return pd.DataFrame()
        df = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
        df = df.rename(columns={"temperature_2m_max": "realized_high_f"})
        return df[["date", "realized_high_f"]].dropna()
    except Exception as exc:
        print(f"  [WARN] Realized temp fetch failed: {exc}")
        return pd.DataFrame()


def backfill_realized(df_log: pd.DataFrame, df_realized: pd.DataFrame) -> pd.DataFrame:
    """Match each row's target date to the realized temperature archive."""
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
            real_col = f"realized_h{h}"
            if real_col not in df_log.columns:
                df_log[real_col] = np.nan
            if target_date > today:
                continue
            current = pd.to_numeric(
                pd.Series([df_log.at[idx, real_col]]), errors="coerce"
            ).iloc[0]
            if np.isfinite(current):
                continue
            val = realized_map.get(target_date)
            if val is not None and np.isfinite(val):
                df_log.at[idx, real_col] = val
    return df_log


# ---------------------------------------------------------------------------
# Persistence baseline — horizon-aware
# ---------------------------------------------------------------------------
def compute_persistence_baseline(df_log: pd.DataFrame, df_hist: pd.DataFrame) -> pd.DataFrame:
    """Add persistence_h* columns.

    persistence_h{k} = last observed temp at feature_date (same value for all h,
    since persistence simply says 'tomorrow will equal today').
    Also stored as persistence_temp_f for backward compatibility.
    """
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
        fd = pd.Timestamp(fd).normalize()
        for lookback in range(0, 10):
            check = fd - pd.Timedelta(days=lookback)
            if check in hist_map:
                val = hist_map[check]
                df_log.at[idx, "persistence_temp_f"] = val
                for h in HORIZONS:
                    df_log.at[idx, f"persistence_h{h}"] = val
                break
    return df_log


# ---------------------------------------------------------------------------
# Climatological normal
# ---------------------------------------------------------------------------
def compute_climatological_normal(df_hist: pd.DataFrame) -> pd.DataFrame:
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
def compute_nws_accuracy(df_log: pd.DataFrame, nws_forecast: Dict) -> Dict[str, float]:
    daily = nws_forecast.get("daily_high_f", {})
    if not daily:
        return {}
    nws_map = {pd.Timestamp(k).normalize(): float(v) for k, v in daily.items() if pd.notna(v)}

    errors: List[float] = []
    for _, row in df_log.iterrows():
        tdate = target_date_for_row(row, 1)
        nws_val = nws_map.get(tdate)
        realized = pd.to_numeric(
            pd.Series([row.get("realized_h1", np.nan)]), errors="coerce"
        ).iloc[0]
        if nws_val is not None and np.isfinite(nws_val) and np.isfinite(realized):
            errors.append(realized - nws_val)

    if not errors:
        return {}
    err = np.array(errors, dtype=float)
    return {
        "nws_h1_mae_f": float(np.mean(np.abs(err))),
        "nws_h1_rmse_f": float(np.sqrt(np.mean(err ** 2))),
        "nws_h1_bias_f": float(np.mean(err)),
        "nws_n_days": int(len(err)),
    }


# ---------------------------------------------------------------------------
# Core metrics — with minimum-sample guards
# ---------------------------------------------------------------------------
def compute_metrics(df_log: pd.DataFrame, clim_df: pd.DataFrame) -> Dict[str, Any]:
    df_log = add_missing_target_date_columns(df_log)
    clim_map = dict(zip(clim_df["doy"], clim_df["clim_normal_f"])) if not clim_df.empty else {}
    metrics: Dict[str, Any] = {}

    for h in HORIZONS:
        pred_col = _prediction_col_for(df_log, h)
        real_col = f"realized_h{h}"
        pers_col = f"persistence_h{h}" if f"persistence_h{h}" in df_log.columns else "persistence_temp_f"

        if pred_col not in df_log.columns or real_col not in df_log.columns:
            metrics[f"h{h}"] = {"n_samples": 0, "pred_col_used": pred_col}
            continue

        pred = pd.to_numeric(df_log[pred_col], errors="coerce")
        real = pd.to_numeric(df_log[real_col], errors="coerce")
        pers = pd.to_numeric(
            df_log.get(pers_col, pd.Series(np.nan, index=df_log.index)), errors="coerce"
        )

        mask = pred.notna() & real.notna()
        n = int(mask.sum())

        if n == 0:
            metrics[f"h{h}"] = {"n_samples": 0, "pred_col_used": pred_col}
            continue

        errors = real[mask] - pred[mask]
        abs_err = errors.abs()
        mae = float(abs_err.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        bias = float(errors.mean())

        row: Dict[str, Any] = {
            "n_samples": n,
            "pred_col_used": pred_col,
            "mae_f": round(mae, 3),
            "rmse_f": round(rmse, 3),
            "bias_f": round(bias, 3),
        }

        # Skill metrics — only when n >= SKILL_MIN_SAMPLES
        if n >= SKILL_MIN_SAMPLES:
            pers_mask = mask & pers.notna()
            if int(pers_mask.sum()) >= SKILL_MIN_SAMPLES:
                pers_err = (real[pers_mask] - pers[pers_mask]).abs()
                pers_mae = float(pers_err.mean())
                row["persistence_mae_f"] = round(pers_mae, 3)
                row["skill_vs_persistence"] = round(
                    float(1 - mae / pers_mae) if pers_mae > 0 else 0.0, 4
                )
                model_day_err = (real[pers_mask] - pred[pers_mask]).abs()
                row["hit_rate"] = round(float((model_day_err < pers_err).mean()), 4)
            else:
                row["persistence_mae_f"] = None
                row["skill_vs_persistence"] = None
                row["hit_rate"] = None

            target_dates = pd.to_datetime(df_log.loc[mask, f"target_date_h{h}"])
            clim_p = climatology_for_dates(target_dates, clim_map)
            clim_m = clim_p.notna()
            if int(clim_m.sum()) >= SKILL_MIN_SAMPLES:
                clim_err = (
                    real[mask].reset_index(drop=True)[clim_m.values]
                    - clim_p[clim_m].reset_index(drop=True)
                ).abs()
                clim_mae = float(clim_err.mean())
                row["clim_mae_f"] = round(clim_mae, 3)
                row["skill_vs_climatology"] = round(
                    float(1 - mae / clim_mae) if clim_mae > 0 else 0.0, 4
                )
            else:
                row["clim_mae_f"] = None
                row["skill_vs_climatology"] = None
        else:
            row["note"] = (
                f"Skill scores withheld: n={n} < SKILL_MIN_SAMPLES={SKILL_MIN_SAMPLES}. "
                f"Need {SKILL_MIN_SAMPLES - n} more realized samples."
            )

        # BNN coverage (always compute if available; no minimum guard)
        lo_col, hi_col = f"bnn_lo90_h{h}", f"bnn_hi90_h{h}"
        if lo_col in df_log.columns and hi_col in df_log.columns:
            lo = pd.to_numeric(df_log[lo_col], errors="coerce")
            hi = pd.to_numeric(df_log[hi_col], errors="coerce")
            cov_mask = mask & lo.notna() & hi.notna()
            if int(cov_mask.sum()) > 0:
                in_ci = (real[cov_mask] >= lo[cov_mask]) & (real[cov_mask] <= hi[cov_mask])
                row["bnn_coverage_90pct"] = round(float(in_ci.mean()), 4)
                row["bnn_n_samples"] = int(cov_mask.sum())

        metrics[f"h{h}"] = row

    return metrics


# ---------------------------------------------------------------------------
# Rolling skill history
# ---------------------------------------------------------------------------
def compute_rolling_skill(df_tape: pd.DataFrame) -> pd.DataFrame:
    df_tape = add_missing_target_date_columns(df_tape).sort_values(
        ["feature_date", "decision_date"]
    ).reset_index(drop=True)
    rows = []
    for i in range(len(df_tape)):
        row: Dict = {
            "date": df_tape["decision_date"].iloc[i],
            "feature_date": df_tape["feature_date"].iloc[i],
        }
        for h in HORIZONS:
            pred_col = _prediction_col_for(df_tape, h)
            real_col = f"realized_h{h}"
            if pred_col not in df_tape.columns or real_col not in df_tape.columns:
                continue
            for window, label in [(30, "30d"), (90, "90d")]:
                slc = df_tape.iloc[max(0, i - window + 1) : i + 1]
                pred = pd.to_numeric(slc[pred_col], errors="coerce")
                real = pd.to_numeric(slc[real_col], errors="coerce")
                m = pred.notna() & real.notna()
                if int(m.sum()) >= 5:
                    row[f"h{h}_mae_{label}"] = round(float((real[m] - pred[m]).abs().mean()), 3)
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

    # Determine forecast column
    for h in HORIZONS:
        col = _prediction_col_for(df_log, h)
        print(f"  H={h}: using column '{col}' for attribution")

    df_hist = load_historical()
    nws_forecast = load_nws_forecast()
    clim_df = compute_climatological_normal(df_hist)

    # Fetch realized temps — cap end date at yesterday
    target_date_cols = [f"target_date_h{h}" for h in HORIZONS]
    all_target_dates = pd.to_datetime(
        df_log[target_date_cols].stack(), errors="coerce"
    ).dropna()
    if all_target_dates.empty:
        print("[Part 9] No target dates found — cannot backfill.")
        return 1

    min_target = all_target_dates.min()
    max_target = min(
        all_target_dates.max(),
        pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
    )
    df_realized = fetch_realized_temps(min_target, max_target)

    df_log = backfill_realized(df_log, df_realized)
    df_log = compute_persistence_baseline(df_log, df_hist)

    # Metrics
    print("[Part 9] Computing attribution metrics...")
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
        parts = [f"n={n}", f"MAE={m['mae_f']:.2f}°F", f"Bias={m['bias_f']:+.2f}°F"]
        if m.get("skill_vs_persistence") is not None:
            parts.append(f"Skill={m['skill_vs_persistence']:+.4f}")
        if m.get("bnn_coverage_90pct") is not None:
            parts.append(f"BNN_cov={m['bnn_coverage_90pct']:.0%}")
        if "note" in m:
            parts.append(f"[{m['note']}]")
        print(f"  H={h} ({m.get('pred_col_used', '?')}): {' | '.join(parts)}")

    if nws_metrics:
        n_nws = nws_metrics.get("nws_n_days", 0)
        nws_note = "" if n_nws >= SKILL_MIN_SAMPLES else f" ⚠️  n={n_nws} < {SKILL_MIN_SAMPLES}"
        print(f"\n  NWS H=1: MAE={nws_metrics['nws_h1_mae_f']:.2f}°F (n={n_nws}){nws_note}")

    # Save attribution tape
    df_log.to_parquet(ARTIFACTS_DIR / "attribution_tape.parquet", index=False)
    print(f"\n[Part 9] Saved attribution_tape.parquet ({len(df_log)} rows)")

    # Skill history
    skill_df = compute_rolling_skill(df_log)
    skill_df.to_parquet(ARTIFACTS_DIR / "skill_history.parquet", index=False)
    print(f"[Part 9] Saved skill_history.parquet ({len(skill_df)} rows)")

    # Report
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
        "skill_min_samples": SKILL_MIN_SAMPLES,
        "promote_min_samples": PROMOTE_MIN_SAMPLES,
        "realized_samples_by_horizon": {
            f"h{h}": int(
                pd.to_numeric(
                    df_log.get(f"realized_h{h}", pd.Series(dtype=float)), errors="coerce"
                ).notna().sum()
            )
            for h in HORIZONS
        },
        "metrics_by_horizon": metrics,
        "nws_baseline_metrics": nws_metrics,
        "summary": {
            "best_horizon": min(
                [h for h in HORIZONS if metrics.get(f"h{h}", {}).get("n_samples", 0) > 0],
                key=lambda h: metrics.get(f"h{h}", {}).get("mae_f", 999),
                default=None,
            ),
            "overall_skill": float(np.mean(skill_values)) if skill_values else None,
            "skill_trustworthy": all(
                metrics.get(f"h{h}", {}).get("n_samples", 0) >= SKILL_MIN_SAMPLES
                for h in HORIZONS
            ),
        },
    }
    with open(ARTIFACTS_DIR / "live_attribution_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[Part 9] Saved live_attribution_report.json")

    # Update prediction log with realized values
    log_path = PART2_DIR / "prediction_log.csv"
    if log_path.exists():
        df_log.to_csv(log_path, index=False)
        print("[Part 9] Updated prediction_log.csv with realized temperatures")

    print("\n[Part 9] ✅  Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

