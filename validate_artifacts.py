#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_artifacts.py — Artifact Contract Validator
=====================================================
Runs post-pipeline acceptance checks against the committed artifact set.

Checks
------
  1.  feature_meta.json feature_cols == actual feature_matrix.parquet columns
  2.  No constant zero-variance numeric feature columns in feature_matrix
  3.  No object/string model features in feature_matrix
  4.  Split counts are internally consistent and live-tail rows are retained
  5.  xgb_predictions.parquet has a split column with val/test rows
  6.  bnn_predictions.parquet has a split column with val_cal/val_eval/test rows
  7.  prediction_log.csv has required columns
  8.  If BNN intervals are not publishable, governance suppresses display
  9.  Part 9 model_only_metrics uses pre-anchor columns, not forecast_h*
  10. live_attribution_report.json has NWS-anchor tracking
  11. feature_meta.json confirms alpha features were merged
  12. prediction_log.csv stores row-level nws_h* values
  13. bnn_predictions.parquet uses bnn_diagnostic_mean_h*, not bnn_mean_h*
  14. part2_meta.json records heat_event_f and heat_weight
  15. No redundant physical/regime probability duplicate features remain

Exit codes
----------
  0 — all checks passed, or only warnings
  1 — one or more FAIL checks
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple


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
HORIZONS = [1, 3, 5]

REQUIRED_LOG_COLS = {
    "decision_date",
    "feature_date",
    "model",
    "forecast_h1",
    "forecast_h3",
    "forecast_h5",
    "forecast_source",
    "target_date_h1",
    "target_date_h3",
    "target_date_h5",
}


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def _ok(name: str, msg: str = "") -> Tuple[str, str, str]:
    return ("PASS", name, msg)


def _warn(name: str, msg: str) -> Tuple[str, str, str]:
    return ("WARN", name, msg)


def _fail(name: str, msg: str) -> Tuple[str, str, str]:
    return ("FAIL", name, msg)


def _load_prediction_log_path() -> Path | None:
    candidates = [
        PROJECT_DIR / "artifacts_part2" / "prediction_log.csv",
        PROJECT_DIR / "artifacts_part3" / "prediction_log.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


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
            issues.append(
                f"{len(extra_in_matrix)} cols in matrix but not meta: "
                f"{sorted(extra_in_matrix)[:8]}"
            )
        if missing_from_matrix:
            issues.append(
                f"{len(missing_from_matrix)} cols in meta but not matrix: "
                f"{sorted(missing_from_matrix)[:8]}"
            )

        if issues:
            return _fail("feature_meta_consistency", "; ".join(issues))

        return _ok(
            "feature_meta_consistency",
            f"{len(actual_feat_cols)} features match between meta and matrix",
        )

    except Exception as e:
        return _warn("feature_meta_consistency", f"Error: {e}")


def check_no_constant_features() -> Tuple[str, str, str]:
    """No constant numeric features in feature_matrix."""
    try:
        import pandas as pd

        matrix_path = PROJECT_DIR / "artifacts_part1" / "feature_matrix.parquet"
        if not matrix_path.exists():
            return _warn("no_constant_features", "feature_matrix.parquet not found")

        df = pd.read_parquet(matrix_path)
        target_cols = {c for c in df.columns if c.startswith("target_h")}
        feat_cols = [c for c in df.columns if c not in ["date"] + list(target_cols)]

        constant = []
        for col in feat_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.nunique(dropna=True) <= 1:
                constant.append(col)

        if constant:
            return _fail(
                "no_constant_features",
                f"{len(constant)} constant feature(s): {constant}",
            )

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
        feat_cols = [c for c in df.columns if c not in ["date"] + list(target_cols)]

        obj_cols = [c for c in feat_cols if df[c].dtype == object]
        if obj_cols:
            return _fail(
                "no_object_features",
                f"{len(obj_cols)} object/string feature(s): {obj_cols}",
            )

        return _ok("no_object_features", "No object/string model features")

    except Exception as e:
        return _warn("no_object_features", f"Error: {e}")


def check_split_counts() -> Tuple[str, str, str]:
    """Validate split internal consistency without hard-coded row counts."""
    try:
        split_path = PROJECT_DIR / "artifacts_part1" / "train_val_test_split.json"
        if not split_path.exists():
            return _warn("split_counts", "train_val_test_split.json not found")

        with open(split_path) as f:
            splits = json.load(f)

        required = [
            "train_end",
            "val_end",
            "test_end",
            "n_train",
            "n_val",
            "n_test",
            "n_labeled",
            "n_feature_rows",
        ]

        missing = [k for k in required if k not in splits]
        if missing:
            return _fail("split_counts", f"Missing split key(s): {missing}")

        train_end = str(splits["train_end"])
        val_end = str(splits["val_end"])
        test_end = str(splits["test_end"])

        n_train = int(splits["n_train"])
        n_val = int(splits["n_val"])
        n_test = int(splits["n_test"])
        n_labeled = int(splits["n_labeled"])
        n_feature_rows = int(splits["n_feature_rows"])

        issues = []

        if not (train_end < val_end < test_end):
            issues.append(f"split dates not ordered: {train_end} / {val_end} / {test_end}")

        expected_labeled = n_train + n_val + n_test
        if abs(n_labeled - expected_labeled) > 2:
            issues.append(
                f"n_labeled={n_labeled} != n_train+n_val+n_test={expected_labeled}"
            )

        if n_feature_rows < n_labeled:
            issues.append(
                f"n_feature_rows={n_feature_rows} < n_labeled={n_labeled}; live tail missing"
            )

        if n_train < n_val or n_train < n_test:
            issues.append(
                f"n_train={n_train} is not largest split; val={n_val}, test={n_test}"
            )

        for label, value in [("n_train", n_train), ("n_val", n_val), ("n_test", n_test)]:
            if value <= 0:
                issues.append(f"{label}={value}; empty split")

        if issues:
            return _fail("split_counts", "; ".join(issues))

        return _ok(
            "split_counts",
            f"Train={n_train} Val={n_val} Test={n_test} "
            f"Labeled={n_labeled} Features={n_feature_rows}",
        )

    except Exception as e:
        return _warn("split_counts", f"Error: {e}")


def check_xgb_predictions_split_column() -> Tuple[str, str, str]:
    """xgb_predictions.parquet must have a split column with val/test rows."""
    try:
        import pandas as pd

        path = PROJECT_DIR / "artifacts_part2b" / "xgb_predictions.parquet"
        if not path.exists():
            return _warn(
                "xgb_predictions_split_col",
                "xgb_predictions.parquet not found; Part 2B may not have run",
            )

        df = pd.read_parquet(path)

        if "split" not in df.columns:
            return _fail(
                "xgb_predictions_split_col",
                "'split' column missing from xgb_predictions.parquet",
            )

        observed = set(df["split"].dropna().astype(str).unique())
        required = {"val", "test"}
        missing = required - observed

        if missing:
            return _fail(
                "xgb_predictions_split_col",
                f"Missing split value(s): {sorted(missing)}; observed={sorted(observed)}",
            )

        counts = df["split"].value_counts(dropna=False).to_dict()
        return _ok("xgb_predictions_split_col", f"split counts={counts}")

    except Exception as e:
        return _warn("xgb_predictions_split_col", f"Error: {e}")


def check_bnn_predictions_split_column() -> Tuple[str, str, str]:
    """bnn_predictions.parquet must have split labels val_cal/val_eval/test."""
    try:
        import pandas as pd

        path = PROJECT_DIR / "artifacts_part2c" / "bnn_predictions.parquet"
        if not path.exists():
            return _warn(
                "bnn_predictions_split_col",
                "bnn_predictions.parquet not found; Part 2C may not have run",
            )

        df = pd.read_parquet(path)

        if "split" not in df.columns:
            return _fail(
                "bnn_predictions_split_col",
                "'split' column missing from bnn_predictions.parquet",
            )

        observed = set(df["split"].dropna().astype(str).unique())
        required = {"val_cal", "val_eval", "test"}
        missing = required - observed

        if missing:
            return _fail(
                "bnn_predictions_split_col",
                f"Missing split value(s): {sorted(missing)}; observed={sorted(observed)}",
            )

        counts = df["split"].value_counts(dropna=False).to_dict()
        return _ok("bnn_predictions_split_col", f"split counts={counts}")

    except Exception as e:
        return _warn("bnn_predictions_split_col", f"Error: {e}")


def check_prediction_log_schema() -> Tuple[str, str, str]:
    """prediction_log.csv must have required columns."""
    try:
        import pandas as pd

        log_path = _load_prediction_log_path()
        if log_path is None:
            return _warn("prediction_log_schema", "prediction_log.csv not found")

        df = pd.read_csv(log_path)

        missing = REQUIRED_LOG_COLS - set(df.columns)
        if missing:
            return _fail(
                "prediction_log_schema",
                f"Missing required columns: {sorted(missing)}",
            )

        return _ok(
            "prediction_log_schema",
            f"{len(df)} rows, all required columns present",
        )

    except Exception as e:
        return _warn("prediction_log_schema", f"Error: {e}")


def check_bnn_display_gate() -> Tuple[str, str, str]:
    """If intervals_publishable=False, bnn_intervals_displayable must be False."""
    try:
        gov_path = PROJECT_DIR / "artifacts_part3" / "governance_report.json"
        cal_path = PROJECT_DIR / "artifacts_part2c" / "calibration_report.json"

        if not gov_path.exists():
            return _warn("bnn_display_gate", "governance_report.json not found")

        with open(gov_path) as f:
            gov = json.load(f)

        intervals_publishable = True
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            intervals_publishable = bool(cal.get("intervals_publishable", False))

        displayable = bool(gov.get("bnn_intervals_displayable", False))

        if not intervals_publishable and displayable:
            return _fail(
                "bnn_display_gate",
                "intervals_publishable=False but bnn_intervals_displayable=True",
            )

        if not intervals_publishable:
            return _ok(
                "bnn_display_gate",
                "BNN intervals not publishable; display gate correctly suppressed",
            )

        return _ok(
            "bnn_display_gate",
            f"bnn_intervals_displayable={displayable}, "
            f"intervals_publishable={intervals_publishable}",
        )

    except Exception as e:
        return _warn("bnn_display_gate", f"Error: {e}")


def check_model_only_metrics_uses_pre_anchor() -> Tuple[str, str, str]:
    """Part 9 model_only_metrics must not use forecast_h* anchored columns."""
    try:
        path = PROJECT_DIR / "artifacts_part9" / "live_attribution_report.json"
        if not path.exists():
            return _warn(
                "model_only_metrics_pre_anchor",
                "live_attribution_report.json not found",
            )

        with open(path) as f:
            rpt = json.load(f)

        model_only = rpt.get("model_only_metrics", {})
        if not model_only:
            return _warn(
                "model_only_metrics_pre_anchor",
                "model_only_metrics missing or empty",
            )

        bad = {}
        ok_cols = {}

        for h in HORIZONS:
            key = f"h{h}"
            pred_col = str(model_only.get(key, {}).get("pred_col_used", ""))

            if pred_col.startswith("forecast_h"):
                bad[key] = pred_col
            else:
                ok_cols[key] = pred_col

        if bad:
            return _fail(
                "model_only_metrics_pre_anchor",
                f"model_only_metrics uses anchored forecast_h* column(s): {bad}",
            )

        return _ok(
            "model_only_metrics_pre_anchor",
            f"model-only pred_col_used values={ok_cols}",
        )

    except Exception as e:
        return _warn("model_only_metrics_pre_anchor", f"Error: {e}")


def check_attribution_report() -> Tuple[str, str, str]:
    """live_attribution_report.json must contain NWS-anchor tracking."""
    try:
        path = PROJECT_DIR / "artifacts_part9" / "live_attribution_report.json"
        if not path.exists():
            return _warn("attribution_report", "live_attribution_report.json not found")

        with open(path) as f:
            rpt = json.load(f)

        issues = []

        summary = rpt.get("forecast_source_summary", {})
        if "nws_anchor_rows_by_horizon" not in summary:
            issues.append("nws_anchor_rows_by_horizon missing from forecast_source_summary")

        if "model_only_metrics" not in rpt:
            issues.append("model_only_metrics missing from report")

        if issues:
            return _warn("attribution_report", "; ".join(issues))

        return _ok("attribution_report", "live_attribution_report.json valid")

    except Exception as e:
        return _warn("attribution_report", f"Error: {e}")


def check_alpha_feature_meta_updated() -> Tuple[str, str, str]:
    """feature_meta.json must confirm Part 2A alpha merge."""
    try:
        meta_path = PROJECT_DIR / "artifacts_part1" / "feature_meta.json"
        if not meta_path.exists():
            return _warn("alpha_feature_meta_updated", "feature_meta.json not found")

        with open(meta_path) as f:
            meta = json.load(f)

        if not meta.get("alpha_features_merged", False):
            return _fail(
                "alpha_feature_meta_updated",
                "alpha_features_merged=False; Part 2A did not update feature_meta.json",
            )

        n = meta.get("post_alpha_n_features", len(meta.get("feature_cols", [])))
        return _ok(
            "alpha_feature_meta_updated",
            f"alpha_features_merged=True, post_alpha_n_features={n}",
        )

    except Exception as e:
        return _warn("alpha_feature_meta_updated", f"Error: {e}")


def check_nws_row_level_storage() -> Tuple[str, str, str]:
    """prediction_log.csv should contain row-level nws_h* values."""
    try:
        import pandas as pd

        log_path = _load_prediction_log_path()
        if log_path is None:
            return _warn("nws_row_level_storage", "prediction_log.csv not found")

        df = pd.read_csv(log_path)

        missing = [f"nws_h{h}" for h in HORIZONS if f"nws_h{h}" not in df.columns]
        if missing:
            return _warn(
                "nws_row_level_storage",
                f"Row-level NWS columns missing: {missing}",
            )

        return _ok(
            "nws_row_level_storage",
            f"nws_h1/h3/h5 present in prediction_log ({len(df)} rows)",
        )

    except Exception as e:
        return _warn("nws_row_level_storage", f"Error: {e}")


def check_bnn_parquet_column_names() -> Tuple[str, str, str]:
    """bnn_predictions.parquet must use bnn_diagnostic_mean_h*, not bnn_mean_h*."""
    try:
        import pandas as pd

        path = PROJECT_DIR / "artifacts_part2c" / "bnn_predictions.parquet"
        if not path.exists():
            return _warn(
                "bnn_parquet_column_names",
                "bnn_predictions.parquet not found; Part 2C may not have run",
            )

        df = pd.read_parquet(path)

        old_style = [c for c in df.columns if c.startswith("bnn_mean_h")]
        new_style = [c for c in df.columns if c.startswith("bnn_diagnostic_mean_h")]

        if old_style:
            return _fail(
                "bnn_parquet_column_names",
                f"Old-style bnn_mean_h* columns found: {old_style}",
            )

        required = {f"bnn_diagnostic_mean_h{h}" for h in HORIZONS}
        missing = required - set(new_style)
        if missing:
            return _fail(
                "bnn_parquet_column_names",
                f"Missing diagnostic mean column(s): {sorted(missing)}",
            )

        return _ok(
            "bnn_parquet_column_names",
            f"diagnostic mean columns present: {sorted(required)}",
        )

    except Exception as e:
        return _warn("bnn_parquet_column_names", f"Error: {e}")


def check_heat_weight_in_meta() -> Tuple[str, str, str]:
    """part2_meta.json hyperparameters must include heat_event_f and heat_weight."""
    try:
        meta_path = PROJECT_DIR / "artifacts_part2" / "part2_meta.json"
        if not meta_path.exists():
            return _warn("heat_weight_in_meta", "part2_meta.json not found")

        with open(meta_path) as f:
            meta = json.load(f)

        hp = meta.get("hyperparameters", {})
        missing = [k for k in ["heat_event_f", "heat_weight"] if k not in hp]

        if missing:
            return _warn(
                "heat_weight_in_meta",
                f"Heat weighting params missing from hyperparameters: {missing}",
            )

        return _ok(
            "heat_weight_in_meta",
            f"heat_event_f={hp['heat_event_f']} heat_weight={hp['heat_weight']}",
        )

    except Exception as e:
        return _warn("heat_weight_in_meta", f"Error: {e}")


def check_no_redundant_regime_probability_features() -> Tuple[str, str, str]:
    """Physical probability features must not duplicate prob_regime_* columns.

    This specifically guards against the old issue where, for example,
    prob_marine_layer_lag1 was a perfect copy of prob_regime_0_lag1.
    """
    try:
        import numpy as np
        import pandas as pd

        matrix_path = PROJECT_DIR / "artifacts_part1" / "feature_matrix.parquet"
        if not matrix_path.exists():
            return _warn(
                "no_redundant_regime_probability_features",
                "feature_matrix.parquet not found",
            )

        df = pd.read_parquet(matrix_path)

        target_cols = {c for c in df.columns if c.startswith("target_h")}
        feat_cols = [c for c in df.columns if c not in ["date"] + list(target_cols)]

        physical_cols = [
            c for c in feat_cols
            if c in {
                "prob_marine_layer_lag1",
                "prob_dry_clear_lag1",
                "prob_santa_ana_lag1",
            }
        ]

        regime_cols = [
            c for c in feat_cols
            if c.startswith("prob_regime_") and c.endswith("_lag1")
        ]

        duplicate_pairs = []

        for pc in physical_cols:
            ps = pd.to_numeric(df[pc], errors="coerce")
            for rc in regime_cols:
                rs = pd.to_numeric(df[rc], errors="coerce")
                mask = ps.notna() & rs.notna()
                if int(mask.sum()) == 0:
                    continue

                max_abs_diff = float(np.nanmax(np.abs(ps[mask].values - rs[mask].values)))
                if max_abs_diff <= 1e-12:
                    duplicate_pairs.append((pc, rc))

        if duplicate_pairs:
            return _fail(
                "no_redundant_regime_probability_features",
                f"Perfect duplicate probability feature pair(s): {duplicate_pairs}",
            )

        return _ok(
            "no_redundant_regime_probability_features",
            "No physical prob_* feature is a perfect duplicate of prob_regime_*",
        )

    except Exception as e:
        return _warn("no_redundant_regime_probability_features", f"Error: {e}")


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
        check_bnn_predictions_split_column(),
        check_prediction_log_schema(),
        check_bnn_display_gate(),
        check_model_only_metrics_uses_pre_anchor(),
        check_attribution_report(),
        check_alpha_feature_meta_updated(),
        check_nws_row_level_storage(),
        check_bnn_parquet_column_names(),
        check_heat_weight_in_meta(),
        check_no_redundant_regime_probability_features(),
    ]


def main() -> int:
    checks = run_all_checks()

    print(f"[Validator] Project root: {PROJECT_DIR}")
    print(f"[Validator] Running {len(checks)} artifact contract checks...\n")

    n_pass = sum(1 for status, _, _ in checks if status == "PASS")
    n_warn = sum(1 for status, _, _ in checks if status == "WARN")
    n_fail = sum(1 for status, _, _ in checks if status == "FAIL")

    for status, name, msg in checks:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
        print(f"  {icon} [{status}] {name}: {msg}")

    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")

    if n_fail > 0:
        print("❌ Artifact validation FAILED. Fix issues before deploying.")
        return 1

    if n_warn > 0:
        print("⚠️  Artifact validation passed with warnings. Review before deploying.")
        return 0

    print("✅ All artifact contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

