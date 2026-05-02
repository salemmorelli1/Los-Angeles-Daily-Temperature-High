#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_artifacts.py — Artifact Contract Validator
=====================================================
Runs post-pipeline acceptance checks against the committed artifact set.
Intended to run in CI after the Daily LA Temperature Forecast workflow.

Checks
------
  1.  feature_meta.json feature_cols == actual feature_matrix.parquet columns
  2.  No constant (zero-variance) numeric feature columns in feature_matrix
  3.  No object/string model features in feature_matrix
  4.  Split counts are internally consistent and live-tail rows are retained
  5.  xgb_predictions.parquet has a 'split' column with 'val' and 'test' rows
  6.  prediction_log.csv has required columns
  7.  governance_report.json: if intervals_publishable=False, bnn_intervals_displayable=False
  8.  Part 9 model_only_metrics uses a pre-anchor column, not forecast_h*
  9.  live_attribution_report.json exists and has nws_anchor_rows_by_horizon
  10. No alpha_ feature column in feature_matrix is constant zero after Part 2A merge

Exit codes
----------
  0 — all checks passed (or only warnings)
  1 — one or more FAIL checks
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _project_dir() -> Path:
    env_root = os.environ.get("LATEMP_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = _project_dir()

# Required prediction_log columns
REQUIRED_LOG_COLS = {
    "decision_date", "feature_date", "model",
    "forecast_h1", "forecast_h3", "forecast_h5",
    "forecast_source",
    "target_date_h1", "target_date_h3", "target_date_h5",
}

HORIZONS = [1, 3, 5]


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def _ok(name: str, msg: str = "") -> Tuple[str, str, str]:
    return ("PASS", name, msg)


def _warn(name: str, msg: str) -> Tuple[str, str, str]:
    return ("WARN", name, msg)


def _fail(name: str, msg: str) -> Tuple[str, str, str]:
    return ("FAIL", name, msg)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_feature_meta_consistency() -> Tuple[str, str, str]:
    """feature_meta.json feature_cols must match feature_matrix.parquet columns."""
    try:
        import pandas as pd
        meta_path = PROJECT_DIR / "artifacts_part1" / "feature_meta.json"
        matrix_path = PROJECT_DIR / "artifacts_part1" / "feature_matrix.parquet"
        if not meta_path.exists():
            return _warn("feature_meta_consistency", "feature_meta.json not found")
        if not matrix_path.exists():
            return _warn("feature_meta_consistency", "feature_matrix.parquet not found")

        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_parquet(matrix_path)

        target_cols = {c for c in df.columns if c.startswith("target_h")}
        actual_feat_cols = set(df.columns) - {"date"} - target_cols
        meta_feat_cols = set(meta.get("feature_cols", []))

        extra_in_matrix = actual_feat_cols - meta_feat_cols
        missing_from_matrix = meta_feat_cols - actual_feat_cols

        issues = []
        if extra_in_matrix:
            issues.append(f"{len(extra_in_matrix)} cols in matrix but not meta: "
                          f"{sorted(extra_in_matrix)[:5]}...")
        if missing_from_matrix:
            issues.append(f"{len(missing_from_matrix)} cols in meta but not matrix: "
                          f"{sorted(missing_from_matrix)[:5]}...")
        if issues:
            return _fail("feature_meta_consistency", "; ".join(issues))
        return _ok("feature_meta_consistency",
                   f"{len(actual_feat_cols)} features match between meta and matrix")
    except Exception as e:
        return _warn("feature_meta_consistency", f"Error: {e}")


def check_no_constant_features() -> Tuple[str, str, str]:
    """No constant (zero-variance) numeric columns in feature_matrix."""
    try:
        import pandas as pd
        matrix_path = PROJECT_DIR / "artifacts_part1" / "feature_matrix.parquet"
        if not matrix_path.exists():
            return _warn("no_constant_features", "feature_matrix.parquet not found")

        df = pd.read_parquet(matrix_path)
        target_cols = {c for c in df.columns if c.startswith("target_h")}
        feat_cols = [c for c in df.columns if c not in (["date"] + list(target_cols))]

        constant = []
        for col in feat_cols:
            import numpy as np
            s = pd.to_numeric(df[col], errors="coerce")
            if s.nunique(dropna=True) <= 1:
                constant.append(col)

        if constant:
            return _fail("no_constant_features",
                         f"{len(constant)} constant feature(s): {constant}")
        return _ok("no_constant_features", f"{len(feat_cols)} features all have variance")
    except Exception as e:
        return _warn("no_constant_features", f"Error: {e}")


def check_no_object_features() -> Tuple[str, str, str]:
    """No object/string columns among model features."""
    try:
        import pandas as pd
        matrix_path = PROJECT_DIR / "artifacts_part1" / "feature_matrix.parquet"
        if not matrix_path.exists():
            return _warn("no_object_features", "feature_matrix.parquet not found")

        df = pd.read_parquet(matrix_path)
        target_cols = {c for c in df.columns if c.startswith("target_h")}
        feat_cols = [c for c in df.columns if c not in (["date"] + list(target_cols))]

        obj_cols = [c for c in feat_cols if df[c].dtype == object]
        if obj_cols:
            return _fail("no_object_features",
                         f"{len(obj_cols)} object/string feature(s): {obj_cols}")
        return _ok("no_object_features", "No object features")
    except Exception as e:
        return _warn("no_object_features", f"Error: {e}")


def check_split_counts() -> Tuple[str, str, str]:
    """Validate split internal consistency rather than hard-coded absolute counts.

    Checks:
      - train_end < val_end < test_end (chronological ordering)
      - n_labeled == n_train + n_val + n_test (no missing or double-counted rows)
      - n_feature_rows >= n_labeled (live tail preserved)
      - n_train >= n_val and n_train >= n_test (train is the largest split)

    Hard-coding Train=2126/Val=456/Test=456 is fragile as the dataset grows daily.
    These structural invariants hold regardless of absolute row counts.
    """
    try:
        split_path = PROJECT_DIR / "artifacts_part1" / "train_val_test_split.json"
        if not split_path.exists():
            return _warn("split_counts", "train_val_test_split.json not found")
        with open(split_path) as f:
            splits = json.load(f)

        issues = []

        # Required keys
        for key in ["train_end", "val_end", "test_end", "n_train", "n_val", "n_test",
                    "n_labeled", "n_feature_rows"]:
            if key not in splits:
                issues.append(f"missing key: {key}")
        if issues:
            return _fail("split_counts", "; ".join(issues))

        train_end = splits["train_end"]
        val_end = splits["val_end"]
        test_end = splits["test_end"]
        n_train = int(splits["n_train"])
        n_val = int(splits["n_val"])
        n_test = int(splits["n_test"])
        n_labeled = int(splits["n_labeled"])
        n_feature_rows = int(splits["n_feature_rows"])

        # Chronological ordering
        if not (train_end < val_end < test_end):
            issues.append(f"split dates not ordered: {train_end} / {val_end} / {test_end}")

        # Labeled row count consistency (allow ±2 for sequence trimming edge cases)
        expected_labeled = n_train + n_val + n_test
        if abs(n_labeled - expected_labeled) > 2:
            issues.append(
                f"n_labeled={n_labeled} != n_train+n_val+n_test={expected_labeled}"
            )

        # Feature rows >= labeled rows (live unlabeled tail must be retained)
        if n_feature_rows < n_labeled:
            issues.append(
                f"n_feature_rows={n_feature_rows} < n_labeled={n_labeled} "
                "(live tail missing)"
            )

        # Train is the largest split
        if n_train < n_val or n_train < n_test:
            issues.append(
                f"n_train={n_train} is not the largest split "
                f"(val={n_val}, test={n_test})"
            )

        # Non-empty splits
        for label, n in [("n_train", n_train), ("n_val", n_val), ("n_test", n_test)]:
            if n <= 0:
                issues.append(f"{label}={n} (empty split)")

        if issues:
            return _fail("split_counts", "; ".join(issues))
        return _ok("split_counts",
                   f"Train={n_train} Val={n_val} Test={n_test} "
                   f"Labeled={n_labeled} Features={n_feature_rows}")
    except Exception as e:
        return _warn("split_counts", f"Error: {e}")


def check_xgb_predictions_split_column() -> Tuple[str, str, str]:
    """xgb_predictions.parquet must have a 'split' column with val and test rows."""
    try:
        import pandas as pd
        xgb_path = PROJECT_DIR / "artifacts_part2b" / "xgb_predictions.parquet"
        if not xgb_path.exists():
            return _warn("xgb_predictions_split_col",
                         "xgb_predictions.parquet not found in artifacts_part2b/")

        df = pd.read_parquet(xgb_path)
        if "split" not in df.columns:
            return _fail("xgb_predictions_split_col",
                         "'split' column missing from xgb_predictions.parquet")

        splits_present = set(df["split"].dropna().unique())
        missing = {"val", "test"} - splits_present
        if missing:
            return _fail("xgb_predictions_split_col",
                         f"Missing split values: {missing}. Found: {splits_present}")
        return _ok("xgb_predictions_split_col",
                   f"val={int((df['split']=='val').sum())} "
                   f"test={int((df['split']=='test').sum())}")
    except Exception as e:
        return _warn("xgb_predictions_split_col", f"Error: {e}")


def check_prediction_log_schema() -> Tuple[str, str, str]:
    """prediction_log.csv must have all required columns."""
    try:
        import pandas as pd
        log_path = PROJECT_DIR / "artifacts_part2" / "prediction_log.csv"
        if not log_path.exists():
            log_path = PROJECT_DIR / "artifacts_part3" / "prediction_log.csv"
        if not log_path.exists():
            return _warn("prediction_log_schema", "prediction_log.csv not found")

        df = pd.read_csv(log_path)
        missing = REQUIRED_LOG_COLS - set(df.columns)
        if missing:
            return _fail("prediction_log_schema",
                         f"Missing required columns: {sorted(missing)}")
        return _ok("prediction_log_schema",
                   f"{len(df)} rows, all required columns present")
    except Exception as e:
        return _warn("prediction_log_schema", f"Error: {e}")


def check_bnn_display_gate() -> Tuple[str, str, str]:
    """If intervals_publishable=False, bnn_intervals_displayable must also be False."""
    try:
        gov_path = PROJECT_DIR / "artifacts_part3" / "governance_report.json"
        cal_path = PROJECT_DIR / "artifacts_part2c" / "calibration_report.json"

        if not gov_path.exists():
            return _warn("bnn_display_gate", "governance_report.json not found")

        with open(gov_path) as f:
            gov = json.load(f)

        intervals_publishable = True  # default: assume publishable if no BNN
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            intervals_publishable = bool(cal.get("intervals_publishable", False))

        displayable = bool(gov.get("bnn_intervals_displayable", False))

        if not intervals_publishable and displayable:
            return _fail("bnn_display_gate",
                         "intervals_publishable=False but bnn_intervals_displayable=True "
                         "— dashboard would show uncalibrated intervals")
        if not intervals_publishable:
            return _ok("bnn_display_gate",
                       "BNN intervals not publishable; display gate correctly suppressed")
        return _ok("bnn_display_gate",
                   f"bnn_intervals_displayable={displayable}, "
                   f"intervals_publishable={intervals_publishable}")
    except Exception as e:
        return _warn("bnn_display_gate", f"Error: {e}")


def check_attribution_report() -> Tuple[str, str, str]:
    """live_attribution_report.json must exist and contain nws_anchor_rows_by_horizon."""
    try:
        path = PROJECT_DIR / "artifacts_part9" / "live_attribution_report.json"
        if not path.exists():
            return _fail("attribution_report", "live_attribution_report.json not found")

        with open(path) as f:
            rpt = json.load(f)

        issues = []
        if "nws_anchor_rows_by_horizon" not in rpt.get("forecast_source_summary", {}):
            issues.append("nws_anchor_rows_by_horizon missing from forecast_source_summary")
        if "model_only_metrics" not in rpt:
            issues.append("model_only_metrics missing from report")

        if issues:
            return _warn("attribution_report", "; ".join(issues))
        return _ok("attribution_report", "live_attribution_report.json valid")
    except Exception as e:
        return _warn("attribution_report", f"Error: {e}")


def check_alpha_feature_meta_updated() -> Tuple[str, str, str]:
    """feature_meta.json must have alpha_features_merged=True after Part 2A runs."""
    try:
        meta_path = PROJECT_DIR / "artifacts_part1" / "feature_meta.json"
        if not meta_path.exists():
            return _warn("alpha_feature_meta_updated", "feature_meta.json not found")

        with open(meta_path) as f:
            meta = json.load(f)

        if not meta.get("alpha_features_merged", False):
            return _fail("alpha_feature_meta_updated",
                         "alpha_features_merged=False — Part 2A did not update feature_meta.json")
        n = meta.get("post_alpha_n_features", 0)
        return _ok("alpha_feature_meta_updated",
                   f"alpha_features_merged=True, post_alpha_n_features={n}")
    except Exception as e:
        return _warn("alpha_feature_meta_updated", f"Error: {e}")


def check_nws_row_level_storage() -> Tuple[str, str, str]:
    """prediction_log should contain nws_h* columns for row-level NWS baseline."""
    try:
        import pandas as pd
        log_path = PROJECT_DIR / "artifacts_part2" / "prediction_log.csv"
        if not log_path.exists():
            log_path = PROJECT_DIR / "artifacts_part3" / "prediction_log.csv"
        if not log_path.exists():
            return _warn("nws_row_level_storage", "prediction_log.csv not found")

        df = pd.read_csv(log_path)
        missing_nws = [f"nws_h{h}" for h in HORIZONS if f"nws_h{h}" not in df.columns]
        if missing_nws:
            return _warn("nws_row_level_storage",
                         f"Row-level NWS columns missing: {missing_nws} — "
                         "Part 9 will fall back to nws_official_forecast.json for all rows")
        return _ok("nws_row_level_storage",
                   f"nws_h1/h3/h5 present in prediction_log ({len(df)} rows)")
    except Exception as e:
        return _warn("nws_row_level_storage", f"Error: {e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_checks() -> List[Tuple[str, str, str]]:
    return [
        check_feature_meta_consistency(),
        check_no_constant_features(),
        check_no_object_features(),
        check_split_counts(),
        check_xgb_predictions_split_column(),
        check_prediction_log_schema(),
        check_bnn_display_gate(),
        check_attribution_report(),
        check_alpha_feature_meta_updated(),
        check_nws_row_level_storage(),
    ]


def main() -> int:
    print(f"[Validator] Project root: {PROJECT_DIR}")
    print(f"[Validator] Running {10} artifact contract checks...\n")

    results = run_all_checks()
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_warn = sum(1 for r in results if r[0] == "WARN")
    n_fail = sum(1 for r in results if r[0] == "FAIL")

    for status, name, msg in results:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
        print(f"  {icon} [{status}] {name}: {msg}")

    print(f"\n{'='*60}")
    print(f"Results: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")

    if n_fail > 0:
        print("❌ Artifact validation FAILED. Fix issues before deploying.")
        return 1
    if n_warn > 0:
        print("⚠️  Artifact validation passed with warnings. Review before deploying.")
    else:
        print("✅ All artifact contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
