#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_daily_forecast.py — Canonical Daily Production Runner
===========================================================
Executes the full LA Temperature Forecasting pipeline in the correct order.

Authoritative Execution Order
------------------------------
  Part 0  →  Part 6  →  Part 1  →  Part 2A  →  Part 2  →  Part 2B*  →  Part 2C*  →  Part 3  →  Part 9

  * Part 2B and Part 2C are optional/experimental sleeves.
    Part 2B is non-blocking (pipeline continues if it fails).
    Part 2C only activates if Part 2B's bnn_sleeve_recommended=true.

Usage
-----
  python run_daily_forecast.py                # Normal daily run
  python run_daily_forecast.py --retrain      # Force model retrain (Part 2)
  python run_daily_forecast.py --direct       # Skip any orchestrator checks
  python run_daily_forecast.py --skip-train   # Run inference only (no retrain)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LATEMP_ROOT", str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# Canonical file map
# ---------------------------------------------------------------------------
CANONICAL_FILES: Dict[str, str] = {
    "PART0":  "part0_data_infrastructure.py",
    "PART1":  "part1_feature_builder.py",
    "PART2":  "part2_deep_learning_forecaster.py",
    "PART2A": "part2a_atmospheric_alpha.py",
    "PART2B": "part2b_xgb_ensemble.py",
    "PART2C": "part2c_bnn_sleeve.py",
    "PART3":  "part3_forecast_governance.py",
    "PART6":  "part6_weather_regime_engine.py",
    "PART9":  "part9_live_attribution.py",
}

# Parts required for a valid run (2B and 2C are optional)
REQUIRED_PARTS = ["PART0", "PART1", "PART2", "PART2A", "PART3", "PART6", "PART9"]

# Execution order
PIPELINE_ORDER = ["PART0", "PART6", "PART1", "PART2A", "PART2", "PART2B", "PART2C", "PART3", "PART9"]

OPTIONAL_PARTS = {"PART2B", "PART2C"}


# ---------------------------------------------------------------------------
# File audit
# ---------------------------------------------------------------------------
def audit_files() -> Tuple[List[str], List[Tuple[str, bool]]]:
    missing = []
    status = []
    for label, filename in CANONICAL_FILES.items():
        path = (PROJECT_DIR / filename).resolve()
        exists = path.exists()
        status.append((label, exists))
        if not exists and label in REQUIRED_PARTS:
            missing.append(filename)
    return missing, status


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------
def run_part(label: str, extra_args: Optional[List[str]] = None) -> int:
    filename = CANONICAL_FILES.get(label)
    if filename is None:
        print(f"[Runner] Unknown part: {label}")
        return 1

    script = (PROJECT_DIR / filename).resolve()
    if not script.exists():
        if label in OPTIONAL_PARTS:
            print(f"[Runner] {label} ({filename}) not found — skipping (optional).")
            return 0
        print(f"[Runner] ERROR: {label} ({filename}) not found.")
        return 1

    cmd = [sys.executable, str(script)] + (extra_args or [])
    env = os.environ.copy()
    env["LATEMP_ROOT"] = str(PROJECT_DIR)

    print(f"\n{'='*60}")
    print(f"[Runner] Running {label}: {filename}")
    print(f"{'='*60}")

    proc = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env, check=False)
    print(f"[Runner] {label} exit code: {proc.returncode}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="LA Temperature Forecast daily runner")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--skip-train", action="store_true", help="Load existing Part 2 model and append a fresh live prediction")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--direct", action="store_true", help="Skip pre-flight checks")
    args = parser.parse_args()

    print(f"[Runner] LA Temperature Forecast — Daily Runner")
    print(f"[Runner] Project root: {PROJECT_DIR}")
    print(f"[Runner] Date: {__import__('datetime').date.today()}")

    # File audit
    print("\n=== FILE AUDIT ===")
    missing, status = audit_files()
    for label, exists in status:
        marker = "✅" if exists else ("⚪" if label in OPTIONAL_PARTS else "❌")
        print(f"  {marker} {label}: {CANONICAL_FILES[label]}")

    if missing:
        print(f"\n[Runner] ERROR: Required files missing: {missing}")
        return 1

    # Decide whether Part 2 should train or load an existing model.
    model_meta_path = PROJECT_DIR / "artifacts_part2" / "part2_meta.json"
    model_file = PROJECT_DIR / "artifacts_part2" / (
        "transformer_model.pt" if args.model == "transformer" else "lstm_model.pt"
    )

    needs_retrain = bool(args.retrain)
    if args.skip_train:
        needs_retrain = False
        print("\n[Runner] --skip-train set — Part 2 will run predict-only.")
    elif model_meta_path.exists() and model_file.exists():
        import datetime
        with open(model_meta_path) as f:
            meta = json.load(f)
        saved_model_type = meta.get("model_type", args.model)
        if saved_model_type != args.model:
            print(
                f"\n[Runner] Saved model_type={saved_model_type}, requested={args.model} — will retrain."
            )
            needs_retrain = True
        else:
            trained_raw = meta.get("trained_at", "2000-01-01")
            trained_at = __import__("datetime").datetime.fromisoformat(trained_raw)
            age_days = (datetime.datetime.now() - trained_at).days
            if age_days > 7:
                print(f"\n[Runner] Model is {age_days} days old — will retrain.")
                needs_retrain = True
            else:
                print(f"\n[Runner] Model is {age_days} days old — Part 2 will run predict-only.")
                needs_retrain = False
    else:
        print("\n[Runner] No saved model artifacts found — will train.")
        needs_retrain = True

    print(f"\n=== EXECUTION ORDER ===")
    print("Part 0 → Part 6 → Part 1 → Part 2A → Part 2 → Part 2B* → Part 2C* → Part 3 → Part 9")
    print("(* optional sleeves)")

    # Execute pipeline
    for label in PIPELINE_ORDER:
        extra = []
        if label == "PART2":
            part2_mode = "train" if needs_retrain else "predict"
            extra = [f"--model={args.model}", f"--mode={part2_mode}"]
            print(f"\n[Runner] Part 2 mode: {part2_mode}")

        rc = run_part(label, extra_args=extra)

        # Optional parts: non-blocking
        if label in OPTIONAL_PARTS:
            if rc != 0:
                print(f"[Runner] ⚠️  {label} failed (non-blocking) — continuing.")
            continue

        # Part 3 always returns 0 when it completes. HOLD/CAUTION/NORMAL are
        # forecast governance states stored in artifacts, not runner failures.

        # Other required parts: fail fast
        if rc != 0:
            print(f"\n[Runner] ❌ {label} failed with exit code {rc}. Halting pipeline.")
            return rc

    print("\n" + "="*60)
    print("✅ Daily forecast pipeline completed successfully.")
    print("="*60)

    # Print today's predictions summary
    try:
        import pandas as pd
        log_path = PROJECT_DIR / "artifacts_part2" / "prediction_log.csv"
        if log_path.exists():
            df = pd.read_csv(log_path)
            if not df.empty:
                latest = df.iloc[-1]
                print(f"\n=== TODAY'S FORECAST ===")
                print(f"  Decision date: {latest.get('decision_date', '?')}")
                source = latest.get("forecast_source", latest.get("model", "UNKNOWN"))
                for h in [1, 3, 5]:
                    val = latest.get(f"forecast_h{h}", latest.get(f"target_h{h}"))
                    if val and str(val) != "nan":
                        print(f"  H={h}: {float(val):.1f}°F")
                mode = latest.get("publish_mode", "UNKNOWN")
                print(f"  Forecast source: {source}")
                print(f"  Governance mode: {mode}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
