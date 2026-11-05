#!/usr/bin/env python3
"""Explain the frozen forecast and its gates without fetching, fitting or writing."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_protocol import (
    MIN_PAIRED_SKILL_SAMPLES, TRAINING_PROTOCOL_VERSION, file_sha256,
    forecast_is_published, training_lineage_errors,
)
from part9_live_attribution import build_report_core

ROOT = Path(__file__).resolve().parent
SCHEMA = "LA_FORECAST_DIAGNOSTICS_V1"
HORIZONS = (1, 3, 5)


def finite(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def flag(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def day(value) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("Missing forecast calendar date")
    return parsed.strftime("%Y-%m-%d")


def build_diagnostics(frame, governance, ensemble, uncertainty, calibration, attribution) -> dict:
    """Project recorded decisions, keeping decision-time and backfilled counts apart."""
    if frame.empty:
        raise ValueError("Forecast ledger is empty")
    frame = frame.sort_values(["feature_date", "decision_date"], kind="stable")
    if frame["feature_date"].duplicated().any() or frame["decision_date"].duplicated().any():
        raise ValueError("Duplicate forecast ledger dates")
    row = frame.iloc[-1]
    feature, decision = day(row["feature_date"]), day(row["decision_date"])
    for name in ("decision_date", "publish_mode", "forecast_source"):
        if str(row.get(name)) != governance.get(name):
            raise ValueError(f"Forecast and governance disagree: {name}")
    if ensemble.get("forecast_source") != row.get("forecast_source"):
        raise ValueError("Forecast and ensemble source disagree")
    available = forecast_is_published(row)
    checkpoint = row.get("checkpoint_sha256")
    uncertainty_matches = (
        uncertainty.get("feature_date", "")[:10] == feature
        and isinstance(checkpoint, str) and bool(checkpoint)
        and uncertainty.get("checkpoint_sha256") == checkpoint
        and calibration.get("checkpoint_sha256") == checkpoint
    )
    horizons = {}
    for h in HORIZONS:
        key = f"h{h}"
        target = day(row[f"target_date_{key}"])
        if target != day(pd.Timestamp(feature) + pd.Timedelta(days=h)):
            raise ValueError(f"Forecast target clock disagrees for H={h}")
        value = finite(row.get(f"forecast_{key}"))
        if available and value is None:
            raise ValueError(f"Published canonical forecast missing for H={h}")
        recorded = ensemble.get("canonical_forecast", {}).get(key)
        if recorded is not None and finite(recorded) != value:
            raise ValueError(f"Forecast and ensemble disagree for H={h}")
        gate = ensemble.get("live_model_skill_gate", {}).get(key, {})
        lo, hi = finite(row.get(f"bnn_lo90_{key}")), finite(row.get(f"bnn_hi90_{key}"))
        hflag = calibration.get("horizon_display_flags", {}).get(str(h), {})
        bounds = uncertainty.get("live_predictions", {}).get(key, {})
        bounds_match = finite(bounds.get("lo90_f")) == lo and finite(bounds.get("hi90_f")) == hi
        displayable = bool(
            available and flag(row.get("bnn_available"))
            and governance.get("bnn_available") is True
            and flag(row.get(f"bnn_displayable_{key}"))
            and uncertainty_matches and bounds_match
            and uncertainty.get("calibration_pass") is True
            and calibration.get("calibration_pass") is True
            and hflag.get("displayable") is True
            and lo is not None and hi is not None and lo <= value <= hi
        )
        horizons[key] = {
            "target_date": target, "forecast_f": value if available else None,
            "paper_candidate_f": finite(row.get(f"forecast_candidate_{key}")),
            "decision_time_model_gate": {
                "enabled": gate.get("enabled") is True,
                "reason": gate.get("reason", "evidence_unavailable"),
                "n_paired": gate.get("n_paired", 0),
                "min_samples": MIN_PAIRED_SKILL_SAMPLES,
            },
            "realized_samples": attribution.get("realized_samples_by_horizon", {}).get(key, 0),
            "interval": {
                "displayable": displayable,
                "lo_f": lo if displayable else None,
                "hi_f": hi if displayable else None,
                "label": str(row.get("bnn_interval_label", "unavailable")),
                "reason": "displayable" if displayable else (
                    hflag.get("suppression_reason") or "interval_contract_not_satisfied"
                ),
            },
        }
    published = frame.loc[frame.apply(forecast_is_published, axis=1)]
    dates = sorted({pd.Timestamp(value).normalize() for value in published["feature_date"]})
    gaps = [
        {"previous_feature_date": day(a), "next_feature_date": day(b),
         "missing_days": int((b - a).days - 1)}
        for a, b in zip(dates, dates[1:]) if (b - a).days > 1
    ]
    checks = governance.get("checks", [])
    warnings = [dict(c) for c in checks if c.get("passed") is not True]
    reason_labels = {
        "recent_live_undercoverage": "recent live coverage below its required floor",
        "large_anchor_delta": "large shift from the model interval center",
        "non_finite_center": "interval center unavailable",
        "interval_contract_not_satisfied": "interval checks not satisfied",
    }
    for warning in warnings:
        if warning.get("name") == "BNN_CALIBRATION_GATE":
            warning["display_message"] = "Uncertainty display by horizon: " + "; ".join(
                f"H={h} " + ("available" if horizons[f"h{h}"]["interval"]["displayable"]
                              else "withheld (" + reason_labels.get(
                                  horizons[f"h{h}"]["interval"]["reason"], "interval checks not satisfied"
                              ) + ")")
                for h in HORIZONS
            ) + ". The aggregate uncertainty check remains a warning."
    return {
        "schema_version": SCHEMA, "integrity_status": "PASS",
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "decision_date": decision, "feature_date": feature,
        "publish_mode": governance["publish_mode"], "published": available,
        "forecast_source": str(row.get("forecast_source", "unknown")),
        "forecast_reason": str(row.get("forecast_reason", "")),
        "target_clock": "target_date_h = feature_date + h calendar days",
        "checks_passed": sum(c.get("passed") is True for c in checks),
        "checks_total": len(checks),
        "warnings": warnings,
        "horizons": horizons,
        "skill_status": attribution.get("summary", {}).get("skill_status", "UNAVAILABLE"),
        "cohort": attribution.get("cohort", {}),
        "skill_min_samples": MIN_PAIRED_SKILL_SAMPLES,
        "sample_size_note": "30 paired outcomes alone do not authorize a model: validation and paired-loss CI gates must also pass.",
        "history_gaps": gaps,
        "history_note": "Blank dates have no recorded published forecast; do not interpolate or backdate forecasts.",
    }


def inspect_repository(root: Path) -> dict:
    errors = training_lineage_errors(root)
    if errors:
        raise ValueError(f"Training lineage invalid: {errors}")
    paths = {name: root / path for name, path in {
        "ledger": "artifacts_part2/prediction_log.csv",
        "governed": "artifacts_part3/prediction_log.csv",
        "governance": "artifacts_part3/governance_report.json",
        "ensemble": "artifacts_part2b/part2b_summary.json",
        "uncertainty": "artifacts_part2c/part2c_meta.json",
        "calibration": "artifacts_part2c/calibration_report.json",
        "history": "artifacts_part0/historical_daily.parquet",
    }.items()}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes["ledger"] != hashes["governed"]:
        raise ValueError("Canonical and governed prediction logs disagree")
    frame = pd.read_csv(paths["ledger"], float_precision="round_trip")
    values = {name: json.loads(paths[name].read_text(encoding="utf-8"))
              for name in ("governance", "ensemble", "uncertainty", "calibration")}
    report = build_report_core(frame, pd.read_parquet(paths["history"]), {})
    result = build_diagnostics(frame, **values, attribution=report)
    if any(file_sha256(paths[name]) != digest for name, digest in hashes.items()):
        raise ValueError("Source artifacts changed during the audit")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = inspect_repository(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"integrity_status": "FAIL", "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
