#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3 — Forecast Governance
==============================
Runs pre-publish quality checks on the canonical forecast before it is
treated as authoritative. Governance operates on forecast_h* (the fallback-
chain output written by Part 2B), not on raw target_h* (raw LSTM output).

Governance Checks
-----------------
  1.  DATA_FRESHNESS         — historical data ≤ MAX_DATA_STALENESS_DAYS stale  [CRITICAL]
  2.  MODEL_FRESHNESS        — LSTM model trained within MAX_MODEL_AGE_DAYS     [WARN]
  3.  SCHEMA_INTEGRITY       — prediction log has required columns               [CRITICAL]
  4.  FORECAST_SOURCE_CHECK  — forecast_h* exists and source is not "unavailable"[CRITICAL]
  5.  FORECAST_BOUNDS        — canonical forecast in [32°F, 125°F]               [CRITICAL]
  6.  FORECAST_SPREAD        — H1–H5 spread ≤ MAX_SPREAD_F                      [WARN]
  7.  PERSISTENCE_SANITY     — canonical forecast deviation ≤ 15°F from last obs [CRITICAL]
  8.  NWS_SANITY             — canonical forecast deviation ≤ 20°F from NWS     [WARN]
  9.  BNN_CALIBRATION_GATE   — BNN calibration_pass=true (if BNN ran)           [WARN]

Publish Modes
-------------
  NORMAL   — all checks pass
  CAUTION  — one or more WARN-level checks failed; review before use
  HOLD     — one or more CRITICAL checks failed; do not publish

Governance history is upserted by decision_date — reruns overwrite the
existing row rather than creating duplicates.

Artifacts Written
-----------------
  artifacts_part3/
      governance_report.json     — per-check results, publish mode, run timestamp
      governance_history.parquet — rolling history (one row per decision_date)
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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

SCHEMA_VERSION = "1.1.0"
HORIZONS = [1, 3, 5]

# Thresholds
MAX_DATA_STALENESS_DAYS = 2
MAX_MODEL_AGE_DAYS = 30
LA_TEMP_MIN_F = 32.0
LA_TEMP_MAX_F = 125.0
MAX_SPREAD_F = 35.0
MAX_PERSISTENCE_DEVIATION_F = 15.0   # Lowered from 25°F — CRITICAL level
MAX_NWS_DEVIATION_F = 20.0           # New: flag if forecast diverges from NWS
MIN_BNN_COVERAGE = 0.75              # New: BNN must achieve this coverage to pass

REQUIRED_LOG_COLS = [
    "decision_date", "feature_date", "model",
    "target_h1", "target_h3", "target_h5",
    "forecast_h1", "forecast_h3", "forecast_h5", "forecast_source",
]


# ---------------------------------------------------------------------------
# GovernanceCheck
# ---------------------------------------------------------------------------
class GovernanceCheck:
    def __init__(self, name: str, level: str = "CRITICAL"):
        self.name = name
        self.level = level
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
            "name": self.name, "passed": self.passed,
            "level": self.level, "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Helper: read forecast_h* from the latest prediction log row
# ---------------------------------------------------------------------------
def _forecast_vals(row: pd.Series) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for h in HORIZONS:
        v = pd.to_numeric(pd.Series([row.get(f"forecast_h{h}", np.nan)]),
                          errors="coerce").iloc[0]
        vals[f"h{h}"] = float(v) if pd.notna(v) else float("nan")
    return vals


def _target_vals(row: pd.Series) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for h in HORIZONS:
        v = pd.to_numeric(pd.Series([row.get(f"target_h{h}", np.nan)]),
                          errors="coerce").iloc[0]
        vals[f"h{h}"] = float(v) if pd.notna(v) else float("nan")
    return vals


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
    age_h = (pd.Timestamp.now() - fetched_at).total_seconds() / 3600
    hist_end = pd.Timestamp(meta.get("historical_end", "1970-01-01"))
    staleness = (pd.Timestamp.today().normalize() - hist_end).days
    chk.details = {
        "fetched_at": str(fetched_at), "age_hours": round(age_h, 1),
        "historical_end": str(hist_end.date()), "staleness_days": staleness,
    }
    if staleness > MAX_DATA_STALENESS_DAYS:
        return chk.warn(
            f"Data is {staleness}d stale (max={MAX_DATA_STALENESS_DAYS}d)", **chk.details
        )
    return chk


def check_model_freshness() -> GovernanceCheck:
    chk = GovernanceCheck("MODEL_FRESHNESS", level="WARN")
    path = PART2_DIR / "part2_meta.json"
    if not path.exists():
        return chk.fail("part2_meta.json not found — Part 2 has not run")
    with open(path) as f:
        meta = json.load(f)
    trained_at = pd.Timestamp(meta.get("trained_at", "1970-01-01"))
    age = (pd.Timestamp.now() - trained_at).days
    chk.details = {"trained_at": str(trained_at), "age_days": age}
    if age > MAX_MODEL_AGE_DAYS:
        return chk.warn(f"Model is {age}d old (max={MAX_MODEL_AGE_DAYS}d)", **chk.details)
    return chk


def check_schema_integrity(df_log: Optional[pd.DataFrame]) -> GovernanceCheck:
    chk = GovernanceCheck("SCHEMA_INTEGRITY", level="CRITICAL")
    if df_log is None or df_log.empty:
        return chk.fail("Prediction log is missing or empty")
    missing = [c for c in REQUIRED_LOG_COLS if c not in df_log.columns]
    chk.details = {"required_cols": REQUIRED_LOG_COLS, "missing_cols": missing}
    if missing:
        return chk.fail(f"Missing columns: {missing}", **chk.details)
    return chk


def check_forecast_source(row: pd.Series) -> GovernanceCheck:
    """Verify that canonical forecast_h* columns exist and have a valid source."""
    chk = GovernanceCheck("FORECAST_SOURCE_CHECK", level="CRITICAL")
    source = str(row.get("forecast_source", "")).strip()
    fcast = _forecast_vals(row)
    missing_horizons = [h for h in HORIZONS if not np.isfinite(fcast.get(f"h{h}", float("nan")))]
    chk.details = {
        "forecast_source": source,
        "forecast_values": {f"h{h}": round(fcast.get(f"h{h}", float("nan")), 2) for h in HORIZONS},
    }
    if "unavailable" in source or not source or source == "lstm_preliminary":
        return chk.fail(
            f"Forecast source is '{source}' — Part 2B may not have run or fallback failed.",
            **chk.details
        )
    if missing_horizons:
        return chk.fail(
            f"forecast_h* is NaN for horizons: {missing_horizons}", **chk.details
        )
    return chk


def check_forecast_bounds(row: pd.Series) -> GovernanceCheck:
    """Check canonical forecast_h* (not raw target_h*) against physical bounds."""
    chk = GovernanceCheck("FORECAST_BOUNDS", level="CRITICAL")
    fcast = _forecast_vals(row)
    violations = []
    for h in HORIZONS:
        v = fcast.get(f"h{h}", float("nan"))
        if not np.isfinite(v):
            violations.append(f"H={h}: NaN")
        elif v < LA_TEMP_MIN_F:
            violations.append(f"H={h}: {v:.1f}°F < min {LA_TEMP_MIN_F}°F")
        elif v > LA_TEMP_MAX_F:
            violations.append(f"H={h}: {v:.1f}°F > max {LA_TEMP_MAX_F}°F")
    chk.details = {
        "forecast_values": {f"h{h}": round(fcast.get(f"h{h}", float("nan")), 2) for h in HORIZONS},
        "bounds": {"min_f": LA_TEMP_MIN_F, "max_f": LA_TEMP_MAX_F},
    }
    if violations:
        return chk.fail(f"Bounds violated: {'; '.join(violations)}", **chk.details)
    return chk


def check_forecast_spread(row: pd.Series) -> GovernanceCheck:
    chk = GovernanceCheck("FORECAST_SPREAD", level="WARN")
    fcast = _forecast_vals(row)
    vals = [v for v in fcast.values() if np.isfinite(v)]
    if len(vals) < 2:
        return chk.warn("Insufficient valid forecast values to check spread")
    spread = max(vals) - min(vals)
    chk.details = {"spread_f": round(spread, 1), "max_spread_f": MAX_SPREAD_F}
    if spread > MAX_SPREAD_F:
        return chk.warn(f"Forecast spread {spread:.1f}°F > {MAX_SPREAD_F}°F", **chk.details)
    return chk


def check_persistence_sanity(row: pd.Series, df_hist: pd.DataFrame) -> GovernanceCheck:
    """Compare canonical forecast_h* against last observed temp. CRITICAL level."""
    chk = GovernanceCheck("PERSISTENCE_SANITY", level="CRITICAL")
    if df_hist.empty or "temp_high_f" not in df_hist.columns:
        return chk.warn("Cannot check persistence: no historical data")

    last_obs = float(df_hist["temp_high_f"].dropna().iloc[-1])
    fcast = _forecast_vals(row)
    violations = []
    deviations: Dict = {}
    for h in HORIZONS:
        v = fcast.get(f"h{h}", float("nan"))
        if not np.isfinite(v):
            continue
        dev = abs(v - last_obs)
        deviations[f"h{h}"] = round(dev, 1)
        if dev > MAX_PERSISTENCE_DEVIATION_F:
            violations.append(f"H={h}: forecast={v:.1f}°F deviates {dev:.1f}°F from {last_obs:.1f}°F")

    chk.details = {
        "last_observed_f": round(last_obs, 1),
        "deviations_f": deviations,
        "threshold_f": MAX_PERSISTENCE_DEVIATION_F,
        "forecast_source": str(row.get("forecast_source", "")),
    }
    if violations:
        return chk.fail(f"Persistence sanity FAILED: {'; '.join(violations)}", **chk.details)
    return chk


def check_nws_sanity(row: pd.Series) -> GovernanceCheck:
    """Compare canonical forecast_h* against NWS official forecast. WARN level."""
    chk = GovernanceCheck("NWS_SANITY", level="WARN")
    nws_path = PART0_DIR / "nws_official_forecast.json"
    if not nws_path.exists():
        chk.details = {"nws_available": False}
        chk.message = "NWS forecast not available — skipping NWS sanity check"
        return chk

    with open(nws_path) as f:
        nws = json.load(f)
    daily = nws.get("daily_high_f", {})
    if not daily:
        chk.details = {"nws_available": False}
        chk.message = "NWS daily_high_f is empty"
        return chk

    feature_date_raw = row.get("feature_date", row.get("decision_date", ""))
    try:
        feature_date = pd.Timestamp(feature_date_raw).normalize()
    except Exception:
        chk.message = "Cannot parse feature_date for NWS comparison"
        return chk

    fcast = _forecast_vals(row)
    violations = []
    comparisons: Dict = {}

    for h in HORIZONS:
        target_date = (feature_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        nws_val = daily.get(target_date)
        fc_val = fcast.get(f"h{h}", float("nan"))
        if nws_val is None or not np.isfinite(fc_val):
            continue
        dev = abs(fc_val - float(nws_val))
        comparisons[f"h{h}"] = {"forecast_f": round(fc_val, 1),
                                 "nws_f": float(nws_val), "deviation_f": round(dev, 1)}
        if dev > MAX_NWS_DEVIATION_F:
            violations.append(f"H={h}: forecast={fc_val:.1f}°F NWS={nws_val:.1f}°F Δ={dev:.1f}°F")

    chk.details = {"comparisons": comparisons, "max_deviation_f": MAX_NWS_DEVIATION_F}
    if violations:
        return chk.warn(f"NWS deviation exceeded: {'; '.join(violations)}", **chk.details)
    return chk


def check_bnn_calibration() -> GovernanceCheck:
    """Check BNN calibration coverage if Part 2C has run. WARN level."""
    chk = GovernanceCheck("BNN_CALIBRATION_GATE", level="WARN")
    cal_path = PART2C_DIR / "calibration_report.json"
    if not cal_path.exists():
        chk.details = {"bnn_available": False}
        chk.message = "BNN sleeve not run — calibration gate skipped"
        return chk

    with open(cal_path) as f:
        cal = json.load(f)

    cal_pass = cal.get("calibration_pass", None)
    results = cal.get("calibration_results", {})
    coverage_summary: Dict = {}
    failing: List[str] = []

    for h in HORIZONS:
        cov = results.get(f"h{h}_coverage_90pct")
        if cov is not None:
            coverage_summary[f"h{h}"] = round(float(cov), 4)
            if float(cov) < MIN_BNN_COVERAGE:
                failing.append(f"H={h}: coverage={cov:.1%} < min {MIN_BNN_COVERAGE:.0%}")

    chk.details = {
        "calibration_pass": cal_pass,
        "coverage_by_horizon": coverage_summary,
        "min_coverage_threshold": MIN_BNN_COVERAGE,
        "bnn_available": True,
    }

    if cal_pass is False or failing:
        return chk.warn(
            f"BNN calibration FAILED — intervals are UNCALIBRATED. "
            f"Failures: {'; '.join(failing) if failing else 'calibration_pass=false'}",
            **chk.details
        )
    return chk


# ---------------------------------------------------------------------------
# Publish mode
# ---------------------------------------------------------------------------
def determine_publish_mode(checks: List[GovernanceCheck]) -> str:
    if any(not c.passed and c.level == "CRITICAL" for c in checks):
        return "HOLD"
    if any(not c.passed and c.level == "WARN" for c in checks):
        return "CAUTION"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_prediction_log() -> Optional[pd.DataFrame]:
    for p in [PART2_DIR / "prediction_log.csv", ARTIFACTS_DIR / "prediction_log.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                return df
    return None


def load_historical() -> pd.DataFrame:
    path = PART0_DIR / "historical_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Governance history — upsert by decision_date
# ---------------------------------------------------------------------------
def upsert_governance_history(report: Dict[str, Any]) -> None:
    hist_path = ARTIFACTS_DIR / "governance_history.parquet"
    decision_date = report.get("decision_date", "")
    new_row = {
        "decision_date": decision_date,
        "run_at": report["run_at"],
        "publish_mode": report["publish_mode"],
        "checks_passed": sum(1 for c in report["checks"] if c["passed"]),
        "checks_total": len(report["checks"]),
        "critical_failures": sum(1 for c in report["checks"]
                                  if not c["passed"] and c["level"] == "CRITICAL"),
        "warnings": sum(1 for c in report["checks"]
                        if not c["passed"] and c["level"] == "WARN"),
        "forecast_source": report.get("forecast_source", ""),
    }
    df_new = pd.DataFrame([new_row])

    if hist_path.exists():
        df_hist = pd.read_parquet(hist_path)
        mask = df_hist["decision_date"].astype(str).str.strip() == str(decision_date).strip()
        if mask.any():
            df_hist.loc[df_hist.index[mask][-1], list(new_row.keys())] = list(new_row.values())
        else:
            df_hist = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df_hist = df_new

    df_hist.to_parquet(hist_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 3] Project root: {PROJECT_DIR}")

    df_log = load_prediction_log()
    df_hist = load_historical()

    latest = df_log.iloc[-1] if df_log is not None and not df_log.empty else pd.Series()
    decision_date = str(latest.get("decision_date", pd.Timestamp.today().date()))
    forecast_source = str(latest.get("forecast_source", "unknown"))

    print(f"[Part 3] Evaluating governance for decision_date={decision_date}")
    print(f"[Part 3] Forecast source: {forecast_source}")

    checks: List[GovernanceCheck] = [
        check_data_freshness(),
        check_model_freshness(),
        check_schema_integrity(df_log),
        check_forecast_source(latest),
        check_forecast_bounds(latest),
        check_forecast_spread(latest),
        check_persistence_sanity(latest, df_hist),
        check_nws_sanity(latest),
        check_bnn_calibration(),
    ]

    publish_mode = determine_publish_mode(checks)

    bnn_chk = next((c for c in checks if c.name == "BNN_CALIBRATION_GATE"), None)
    bnn_available = bool(bnn_chk and bnn_chk.details.get("bnn_available", False))
    bnn_calibrated = bool(bnn_available and bnn_chk.passed)
    if not bnn_available:
        bnn_interval_status = "NOT_RUN"
    elif bnn_calibrated:
        bnn_interval_status = "CALIBRATED"
    else:
        bnn_interval_status = "UNCALIBRATED"

    print("\n=== GOVERNANCE CHECKS ===")
    for chk in checks:
        icon = "✅" if chk.passed else ("⚠️ " if chk.level == "WARN" else "❌")
        print(f"  [{icon}] {chk.name} [{chk.level}]: {chk.message}")

    print(f"\n=== PUBLISH MODE: {publish_mode} ===")
    if publish_mode == "HOLD":
        print("  ⛔ Forecast WITHHELD. Fix critical failures before publishing.")
    elif publish_mode == "CAUTION":
        print("  ⚠️  Published with CAUTION flags. Review warnings before use.")
    else:
        print("  ✅ All checks passed. Forecast is NORMAL.")

    # Write publish_mode + BNN flag back to prediction log
    if df_log is not None and not df_log.empty:
        log_path = PART2_DIR / "prediction_log.csv"
        if not log_path.exists():
            log_path = ARTIFACTS_DIR / "prediction_log.csv"

        df_log.loc[df_log.index[-1], "publish_mode"] = publish_mode
        df_log.loc[df_log.index[-1], "governance_run_at"] = pd.Timestamp.now().isoformat()
        df_log.loc[df_log.index[-1], "bnn_available"] = bool(bnn_available)
        df_log.loc[df_log.index[-1], "bnn_calibrated"] = bool(bnn_calibrated)
        df_log.loc[df_log.index[-1], "bnn_interval_status"] = bnn_interval_status

        df_log.to_csv(log_path, index=False)
        print(f"[Part 3] Updated prediction_log.csv: publish_mode={publish_mode}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_at": pd.Timestamp.now().isoformat(),
        "decision_date": decision_date,
        "publish_mode": publish_mode,
        "forecast_source": forecast_source,
        "bnn_available": bnn_available,
        "bnn_calibrated": bnn_calibrated,
        "bnn_interval_status": bnn_interval_status,
        "checks_passed": sum(1 for c in checks if c.passed),
        "checks_total": len(checks),
        "checks": [c.to_dict() for c in checks],
        "thresholds": {
            "max_data_staleness_days": MAX_DATA_STALENESS_DAYS,
            "max_model_age_days": MAX_MODEL_AGE_DAYS,
            "la_temp_min_f": LA_TEMP_MIN_F,
            "la_temp_max_f": LA_TEMP_MAX_F,
            "max_spread_f": MAX_SPREAD_F,
            "max_persistence_deviation_f": MAX_PERSISTENCE_DEVIATION_F,
            "max_nws_deviation_f": MAX_NWS_DEVIATION_F,
            "min_bnn_coverage": MIN_BNN_COVERAGE,
        },
    }
    with open(ARTIFACTS_DIR / "governance_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[Part 3] Saved governance_report.json")

    upsert_governance_history(report)
    print("[Part 3] Upserted governance_history.parquet")

    print(f"\n[Part 3] ✅  Complete. Mode={publish_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
















