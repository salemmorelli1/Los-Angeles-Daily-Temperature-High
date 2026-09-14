from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import part2b_xgb_ensemble as ensemble
import part3_forecast_governance as governance
import part9_live_attribution as attribution


class EnsembleGateTests(unittest.TestCase):
    def test_xgb_gate_fails_closed_without_persistence_evidence(self) -> None:
        metrics = {f"h{h}_mae_f": 2.0 for h in ensemble.HORIZONS}

        passed, diagnostics = ensemble.evaluate_xgb_validation_gate(metrics, {})

        self.assertFalse(passed)
        self.assertTrue(all(not row["evidence_available"] for row in diagnostics.values()))

    def test_nws_anchor_requires_better_realized_mae(self) -> None:
        rows = []
        for _ in range(ensemble.NWS_ANCHOR_MIN_REALIZED_SAMPLES):
            row = {}
            for h in ensemble.HORIZONS:
                row[f"realized_h{h}"] = 70.0
                row[f"forecast_pre_anchor_h{h}"] = 71.0
                row[f"nws_h{h}"] = 78.0
            rows.append(row)

        allowed, diagnostics = ensemble.evaluate_nws_anchor_skill(pd.DataFrame(rows))

        self.assertEqual(allowed, {1: False, 3: False, 5: False})
        self.assertTrue(all(row["reason"] == "nws_not_better" for row in diagnostics.values()))

    def test_lstm_is_used_before_nws_when_xgb_is_unavailable(self) -> None:
        forecast, source, _ = ensemble.compute_canonical_forecast(
            xgb_preds={},
            lstm_preds={"h1": 71.0, "h3": 72.0, "h5": 73.0},
            nws_preds={1: 80.0, 3: 81.0, 5: 82.0},
            last_obs=70.0,
            nws_anchor_allowed={1: False, 3: False, 5: False},
        )

        self.assertEqual(source, "lstm")
        self.assertEqual(forecast, {"h1": 71.0, "h3": 72.0, "h5": 73.0})

    def test_anchor_audit_uses_exact_plausible_candidate(self) -> None:
        xgb = {"h1": 100.0, "h3": 100.0, "h5": 100.0}
        lstm = {"h1": 71.0, "h3": 72.0, "h5": 73.0}
        nws = {1: 80.0, 3: 81.0, 5: 82.0}
        allowed = {1: False, 3: False, 5: False}
        forecast, _, reason = ensemble.compute_canonical_forecast(
            xgb, lstm, nws, 70.0, nws_anchor_allowed=allowed
        )

        flat, details = ensemble.build_anchor_audit_fields(
            forecast,
            xgb,
            lstm,
            nws,
            reason,
            last_obs=70.0,
            nws_anchor_allowed=allowed,
        )

        self.assertEqual(details["h1"]["pre_anchor_source"], "lstm")
        self.assertEqual(flat["forecast_pre_anchor_h1"], 71.0)
        self.assertFalse(flat["nws_anchor_applied_h1"])


class GovernanceExitTests(unittest.TestCase):
    def test_hold_returns_nonzero_to_daily_runner(self) -> None:
        row = {
            "decision_date": "2026-09-14",
            "feature_date": "2026-09-13",
            "model": "LSTM",
            "target_h1": 70.0,
            "target_h3": 71.0,
            "target_h5": 72.0,
            "forecast_h1": 70.0,
            "forecast_h3": 71.0,
            "forecast_h5": 72.0,
            "forecast_source": "blend",
        }
        log = pd.DataFrame([row])
        hist = pd.DataFrame({"date": [pd.Timestamp("2026-09-13")], "temp_high_f": [70.0]})
        failure = governance.GovernanceCheck("TEST_FAILURE", level="CRITICAL").fail("forced")
        success = governance.GovernanceCheck("TEST_OK", level="WARN")

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            part2_dir = root / "artifacts_part2"
            part3_dir = root / "artifacts_part3"
            part2_dir.mkdir()
            part3_dir.mkdir()
            log.to_csv(part2_dir / "prediction_log.csv", index=False)

            stack.enter_context(patch.object(governance, "PART2_DIR", part2_dir))
            stack.enter_context(patch.object(governance, "ARTIFACTS_DIR", part3_dir))
            stack.enter_context(patch.object(governance, "load_prediction_log", return_value=log.copy()))
            stack.enter_context(patch.object(governance, "load_historical", return_value=hist))
            stack.enter_context(patch.object(governance, "check_data_freshness", return_value=failure))
            for name in [
                "check_model_freshness",
                "check_schema_integrity",
                "check_forecast_source",
                "check_forecast_bounds",
                "check_forecast_spread",
                "check_persistence_sanity",
                "check_nws_sanity",
                "check_bnn_calibration",
            ]:
                stack.enter_context(patch.object(governance, name, return_value=success))
            stack.enter_context(patch.object(governance, "upsert_governance_history"))

            self.assertEqual(governance.main(), 1)

    def test_stale_bnn_artifact_is_ignored_when_stack_does_not_expect_it(self) -> None:
        with patch.object(
            governance,
            "_bnn_expected_from_current_stack",
            return_value={"expected": False, "reason": "part2b_gate_failed"},
        ):
            check = governance.check_bnn_calibration()

        self.assertTrue(check.passed)
        self.assertFalse(check.details["bnn_available"])

    def test_expected_bnn_artifact_must_match_current_feature_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            part2c_dir = Path(tmp)
            (part2c_dir / "calibration_report.json").write_text("{}", encoding="utf-8")
            (part2c_dir / "part2c_meta.json").write_text(
                json.dumps({"feature_date": "2026-09-12"}), encoding="utf-8"
            )
            latest = pd.Series({"feature_date": "2026-09-13"})
            with (
                patch.object(governance, "PART2C_DIR", part2c_dir),
                patch.object(
                    governance,
                    "_bnn_expected_from_current_stack",
                    return_value={"expected": True, "reason": "expected"},
                ),
            ):
                check = governance.check_bnn_calibration(latest)

        self.assertFalse(check.passed)
        self.assertEqual(check.message, "BNN artifact is stale for the current forecast row")
        self.assertFalse(check.details["bnn_available"])


class AttributionIdempotencyTests(unittest.TestCase):
    def test_unchanged_report_preserves_generated_timestamp(self) -> None:
        core = {"schema_version": "test", "n_prediction_rows": 7}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "test",
                        "generated_at": "2026-09-01T00:00:00",
                        "n_prediction_rows": 7,
                    }
                ),
                encoding="utf-8",
            )

            report = attribution._report_with_stable_generated_at(core, path)

        self.assertEqual(report["generated_at"], "2026-09-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
