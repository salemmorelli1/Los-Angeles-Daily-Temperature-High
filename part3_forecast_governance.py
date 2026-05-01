#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3 — Forecast Governance
==============================
Runs pre-publish checks on today's forecast before it is logged as
authoritative. Mirrors the PriceCall governance pattern but scoped to
temperature forecasting (no trading logic).

Governance Checks
-----------------
  1. DATA_FRESHNESS      — historical data is current (≤1 day stale)
  2. MODEL_FRESHNESS     — LSTM model trained within N days
  3. PREDICTION_BOUNDS   — forecasted temp within physically plausible range
  4. PREDICTION_SPREAD   — multi-horizon spread is reasonable (not degenerate)
  5. CONFIDENCE_GATE     — BNN uncertainty width within acceptable range
  6. PERSISTENCE_SANITY  — forecast not too far from naive persistence
  7. SCHEMA_INTEGRITY    — prediction_log has required columns

Publish Modes
-------------
  NORMAL          → all checks pass; forecast is authoritative
  CAUTION         → minor flags; forecast published with warnings
  HOLD            → critical failure; forecast withheld, alert raised

Artifacts Written
-----------------
  artifacts_part3/
      prediction_log.csv         — canonical log (appended by Part 2 / 2B / 2C)
      governance_report.json     — per-check results, publish mode, run timestamp
      governance_history.parquet — rolling history of governance decisions
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
PART0_DIR = PROJECT_DIR / "artifacts_part0"
PART2_DIR = PROJECT_DIR / "artifacts_part2"
PART2B_DIR = PROJECT_DIR / "artifacts_part2b"
PART2C_DIR = PROJECT_DIR / "artifacts_part2c"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part3"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.0.0"

# Governance thresholds
MAX_DATA_STALENESS_DAYS = 2
MAX_MODEL_AGE_DAYS = 30
LA_TEMP_MIN_F = 32.0           # physically implausible below freezing
LA_TEMP_MAX_F = 125.0          # all-time record is 121°F (Woodland Hills)
MAX_PERSISTENCE_DEVIATION_F = 25.0   # flag if forecast deviates >25°F from persistence
MAX_CI_WIDTH_F = 40.0          # BNN CI wider than 40°F = low confidence
MAX_MULTI_HORIZON_SPREAD_F = 35.0   # H1–H5 range > 35°F is suspicious

REQUIRED_PREDICTION_LOG_COLS = [
    "decision_date", "feature_date", "model",
    "target_h1", "target_h3", "target_h5",
]

HORIZONS = [1, 3, 5]


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------
class GovernanceCheck:
    def __init__(self, name: str, level: str = "CRITICAL"):
        self.name = name
        self.level = level  # CRITICAL → HOLD if failed | WARN → CAUTION if failed
        self.passed = True
        self.message = "OK"
        self.details: Dict[str, Any] = {}

    def fail(self, message: str, **details) -> "GovernanceCheck":
        self.passed = False
        self.message = message
        self.details = details
        return self

    def warn(self, message: str, **details) -> "GovernanceCheck":
        self.passed = False
        self.level = "WARN"
        self.message = message
        self.details = details
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "level": self.level,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Individual governance checks
# ---------------------------------------------------------------------------
def check_data_freshness() -> GovernanceCheck:
    chk = GovernanceCheck("DATA_FRESHNESS", level="CRITICAL")
    meta_path = PART0_DIR / "data_meta.json"
    if not meta_path.exists():
        return chk.fail("data_meta.json not found — Part 0 has not run")

    with open(meta_path) as f:
        meta = json.load(f)

    fetched_at = pd.Timestamp(meta.get("fetched_at", "1970-01-01"))
    age_hours = (pd.Timestamp.now() - fetched_at).total_seconds() / 3600

    hist_end = pd.Timestamp(meta.get("historical_end", "1970-01-01"))
    today = pd.Timestamp.today().normalize()
    staleness_days = (today - hist_end).days

    chk.details = {
        "fetched_at": str(fetched_at),
        "age_hours": round(age_hours, 1),
        "historical_end": str(hist_end.date()),
        "staleness_days": staleness_days,
    }

    if staleness_days > MAX_DATA_STALENESS_DAYS:
        return chk.warn(
            f"Historical data is {staleness_days} days stale (max={MAX_DATA_STALENESS_DAYS})",
            **chk.details,
        )
    return chk


def check_model_freshness() -> GovernanceCheck:
    chk = GovernanceCheck("MODEL_FRESHNESS", level="WARN")
    meta_path = PART2_DIR / "part2_meta.json"
    if not meta_path.exists():
        return chk.fail("part2_meta.json not found — Part 2 has not run")

    with open(meta_path) as f:
        meta = json.load(f)

    trained_at = pd.Timestamp(meta.get("trained_at", "1970-01-01"))
    age_days = (pd.Timestamp.now() - trained_at).days
    chk.details = {"trained_at": str(trained_at), "age_days": age_days}

    if age_days > MAX_MODEL_AGE_DAYS:
        return chk.warn(
            f"LSTM model is {age_days} days old (max={MAX_MODEL_AGE_DAYS}). Consider retraining.",
            **chk.details,
        )
    return chk


def check_prediction_bounds(latest_row: pd.Series) -> GovernanceCheck:
    chk = GovernanceCheck("PREDICTION_BOUNDS", level="CRITICAL")
    violations = []
    for h in HORIZONS:
        val = float(latest_row.get(f"target_h{h}", np.nan))
        if not np.isfinite(val):
            violations.append(f"H={h}: NaN prediction")
        elif val < LA_TEMP_MIN_F:
            violations.append(f"H={h}: {val:.1f}°F below minimum {LA_TEMP_MIN_F}°F")
        elif val > LA_TEMP_MAX_F:
            violations.append(f"H={h}: {val:.1f}°F above maximum {LA_TEMP_MAX_F}°F")

    chk.details = {
        "predictions": {f"h{h}": float(latest_row.get(f"target_h{h}", np.nan)) for h in HORIZONS},
        "bounds": {"min_f": LA_TEMP_MIN_F, "max_f": LA_TEMP_MAX_F},
    }
    if violations:
        return chk.fail(f"Prediction bounds violated: {'; '.join(violations)}", **chk.details)
    return chk


def check_prediction_spread(latest_row: pd.Series) -> GovernanceCheck:
    chk = GovernanceCheck("PREDICTION_SPREAD", level="WARN")
    preds = [float(latest_row.get(f"target_h{h}", np.nan)) for h in HORIZONS]
    preds_valid = [p for p in preds if np.isfinite(p)]

    if len(preds_valid) < 2:
        return chk.warn("Insufficient valid predictions to check spread")

    spread = max(preds_valid) - min(preds_valid)
    chk.details = {"spread_f": round(spread, 1), "max_spread_f": MAX_MULTI_HORIZON_SPREAD_F}

    if spread > MAX_MULTI_HORIZON_SPREAD_F:
        return chk.warn(
            f"Multi-horizon spread {spread:.1f}°F exceeds threshold {MAX_MULTI_HORIZON_SPREAD_F}°F",
            **chk.details,
        )
    return chk


def check_persistence_sanity(latest_row: pd.Series, df_hist: pd.DataFrame) -> GovernanceCheck:
    chk = GovernanceCheck("PERSISTENCE_SANITY", level="WARN")

    if df_hist.empty or "temp_high_f" not in df_hist.columns:
        return chk.warn("Cannot check persistence: no historical data")

    last_observed = float(df_hist["temp_high_f"].dropna().iloc[-1])
    deviations = {}
    violations = []

    for h in HORIZONS:
        pred = float(latest_row.get(f"target_h{h}", np.nan))
        if not np.isfinite(pred):
            continue
        dev = abs(pred - last_observed)
        deviations[f"h{h}"] = round(dev, 1)
        if dev > MAX_PERSISTENCE_DEVIATION_F:
            violations.append(f"H={h}: deviation {dev:.1f}°F from last obs {last_observed:.1f}°F")

    chk.details = {
        "last_observed_f": round(last_observed, 1),
        "deviations_f": deviations,
        "max_deviation_f": MAX_PERSISTENCE_DEVIATION_F,
    }

    if violations:
        return chk.warn(f"Persistence sanity flags: {'; '.join(violations)}", **chk.details)
    return chk


def check_confidence_gate(latest_row: pd.Series) -> GovernanceCheck:
    chk = GovernanceCheck("CONFIDENCE_GATE", level="WARN")

    bnn_meta_path = PART2C_DIR / "part2c_meta.json"
    if not bnn_meta_path.exists():
        chk.details = {"bnn_available": False}
        chk.message = "BNN sleeve not run — uncertainty estimates unavailable"
        return chk

    with open(bnn_meta_path) as f:
        bnn_meta = json.load(f)

    live_preds = bnn_meta.get("live_predictions", {})
    wide_intervals = []
    ci_widths = {}

    for h in HORIZONS:
        h_key = f"h{h}"
        if h_key not in live_preds:
            continue
        lo = live_preds[h_key].get("lo90_f", np.nan)
        hi = live_preds[h_key].get("hi90_f", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            width = hi - lo
            ci_widths[f"h{h}"] = round(width, 1)
            if width > MAX_CI_WIDTH_F:
                wide_intervals.append(f"H={h}: CI width {width:.1f}°F")

    chk.details = {"ci_widths_f": ci_widths, "max_ci_width_f": MAX_CI_WIDTH_F}

    if wide_intervals:
        return chk.warn(f"Wide confidence intervals: {'; '.join(wide_intervals)}", **chk.details)
    return chk


def check_schema_integrity(df_log: Optional[pd.DataFrame]) -> GovernanceCheck:
    chk = GovernanceCheck("SCHEMA_INTEGRITY", level="CRITICAL")
    if df_log is None or df_log.empty:
        return chk.fail("Prediction log is missing or empty")

    missing = [c for c in REQUIRED_PREDICTION_LOG_COLS if c not in df_log.columns]
    chk.details = {"required_cols": REQUIRED_PREDICTION_LOG_COLS, "missing_cols": missing}

    if missing:
        return chk.fail(f"Prediction log missing required columns: {missing}", **chk.details)
    return chk


# ---------------------------------------------------------------------------
# Publish mode determination
# ---------------------------------------------------------------------------
def determine_publish_mode(checks: List[GovernanceCheck]) -> str:
    critical_failures = [c for c in checks if not c.passed and c.level == "CRITICAL"]
    warnings = [c for c in checks if not c.passed and c.level == "WARN"]

    if critical_failures:
        return "HOLD"
    if warnings:
        return "CAUTION"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Prediction log helpers
# ---------------------------------------------------------------------------
def load_or_create_prediction_log() -> Optional[pd.DataFrame]:
    log_path = PART2_DIR / "prediction_log.csv"
    if not log_path.exists():
        alt_path = ARTIFACTS_DIR / "prediction_log.csv"
        if not alt_path.exists():
            return None
        log_path = alt_path

    df = pd.read_csv(log_path)
    return df


def load_historical() -> pd.DataFrame:
    path = PART0_DIR / "historical_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Governance history
# ---------------------------------------------------------------------------
def append_governance_history(report: Dict[str, Any]) -> None:
    hist_path = ARTIFACTS_DIR / "governance_history.parquet"
    row = {
        "run_at": report["run_at"],
        "publish_mode": report["publish_mode"],
        "checks_passed": sum(1 for c in report["checks"] if c["passed"]),
        "checks_total": len(report["checks"]),
        "critical_failures": sum(
            1 for c in report["checks"] if not c["passed"] and c["level"] == "CRITICAL"
        ),
        "warnings": sum(
            1 for c in report["checks"] if not c["passed"] and c["level"] == "WARN"
        ),
        "decision_date": report.get("decision_date", ""),
    }
    df_new = pd.DataFrame([row])
    if hist_path.exists():
        df_hist = pd.read_parquet(hist_path)
        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df_hist = df_new
    df_hist.to_parquet(hist_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 3] Project root: {PROJECT_DIR}")

    df_log = load_or_create_prediction_log()
    df_hist = load_historical()

    latest_row = df_log.iloc[-1] if df_log is not None and not df_log.empty else pd.Series()
    decision_date = str(latest_row.get("decision_date", pd.Timestamp.today().date()))

    print(f"[Part 3] Evaluating governance for decision_date={decision_date}")

    # Run all checks
    checks: List[GovernanceCheck] = [
        check_data_freshness(),
        check_model_freshness(),
        check_schema_integrity(df_log),
        check_prediction_bounds(latest_row),
        check_prediction_spread(latest_row),
        check_persistence_sanity(latest_row, df_hist),
        check_confidence_gate(latest_row),
    ]

    # Determine publish mode
    publish_mode = determine_publish_mode(checks)

    # Print results
    print("\n=== GOVERNANCE CHECKS ===")
    for chk in checks:
        status = "✅ PASS" if chk.passed else (f"⚠️  WARN" if chk.level == "WARN" else "❌ FAIL")
        print(f"  [{status}] {chk.name}: {chk.message}")

    print(f"\n=== PUBLISH MODE: {publish_mode} ===")
    if publish_mode == "HOLD":
        print("  ⛔ Forecast is being WITHHELD due to critical governance failures.")
    elif publish_mode == "CAUTION":
        print("  ⚠️  Forecast published with CAUTION flags. Review warnings before use.")
    else:
        print("  ✅ Forecast is NORMAL — all governance checks passed.")

    # Update prediction log with governance decision
    if df_log is not None and not df_log.empty:
        log_path = PART2_DIR / "prediction_log.csv"
        if not log_path.exists():
            log_path = ARTIFACTS_DIR / "prediction_log.csv"

        df_log.loc[df_log.index[-1], "publish_mode"] = publish_mode
        df_log.loc[df_log.index[-1], "governance_run_at"] = pd.Timestamp.now().isoformat()
        df_log.to_csv(log_path, index=False)
        print(f"[Part 3] Updated prediction_log.csv with publish_mode={publish_mode}")

    # Write governance report
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_at": pd.Timestamp.now().isoformat(),
        "decision_date": decision_date,
        "publish_mode": publish_mode,
        "checks_passed": sum(1 for c in checks if c.passed),
        "checks_total": len(checks),
        "checks": [c.to_dict() for c in checks],
        "thresholds": {
            "max_data_staleness_days": MAX_DATA_STALENESS_DAYS,
            "max_model_age_days": MAX_MODEL_AGE_DAYS,
            "la_temp_min_f": LA_TEMP_MIN_F,
            "la_temp_max_f": LA_TEMP_MAX_F,
            "max_persistence_deviation_f": MAX_PERSISTENCE_DEVIATION_F,
            "max_ci_width_f": MAX_CI_WIDTH_F,
        },
    }

    with open(ARTIFACTS_DIR / "governance_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[Part 3] Saved governance_report.json")

    append_governance_history(report)
    print("[Part 3] Updated governance_history.parquet")

    print(f"\n[Part 3] ✅ Complete. Mode={publish_mode}")

    # Return non-zero if HOLD
    return 1 if publish_mode == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
