from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import forecast_protocol as protocol
import part1_feature_builder as features
import part2_deep_learning_forecaster as forecaster
import part2b_xgb_ensemble as ensemble
import part2c_bnn_sleeve as uncertainty
import part3_forecast_governance as governance
import part6_weather_regime_engine as regimes
import part9_live_attribution as attribution
import run_daily_forecast as runner
import validate_artifacts as validator


class TrainingBoundaryTests(unittest.TestCase):
    def test_regime_cutoff_matches_purged_forecaster_with_calendar_gaps(self):
        dates = pd.date_range("2020-01-01", periods=500).delete([27, 233, 270])
        history = pd.DataFrame(
            {"date": dates, "temp_high_f": np.arange(len(dates)) + 60.0}
        )
        history.loc[252, "temp_high_f"] = np.nan
        splits = features.compute_splits(features.add_target_columns(history))
        train, val, _ = protocol.purged_labeled_splits(
            features.add_target_columns(history),
            splits,
            ["target_h1", "target_h3", "target_h5"],
        )
        cutoff = regimes.regime_training_cutoff(history)
        self.assertEqual(cutoff, pd.Timestamp(splits["train_effective_end"]))
        self.assertLessEqual(train["date"].max(), cutoff)
        self.assertLess(cutoff, val["date"].min())

    def test_saved_test_end_excludes_new_dates_from_checkpoint_evaluation(self):
        history = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=500), "temp_high_f": 70.0}
        )
        labeled = features.add_target_columns(history)
        splits = features.compute_splits(labeled.iloc[:400])
        _, _, test = protocol.purged_labeled_splits(
            labeled, splits, ["target_h1", "target_h3", "target_h5"]
        )
        self.assertEqual(test["date"].max(), pd.Timestamp(splits["test_end"]))


class EvidenceTests(unittest.TestCase):
    def test_new_protocol_does_not_relabel_legacy_evidence(self):
        rows = []
        for i in range(40):
            row = {
                "training_protocol_version": protocol.TRAINING_PROTOCOL_VERSION
                if i < 2
                else None,
                "publish_mode": "CAUTION",
                "forecast_finalized_at": "2020-01-02T15:00:00Z",
            }
            for h in [1, 3, 5]:
                row.update(
                    {
                        f"forecast_candidate_h{h}": 70.0,
                        f"realized_h{h}": 70.0,
                        f"persistence_h{h}": 71.0,
                    }
                )
            rows.append(row)
        log = pd.DataFrame(rows)
        original = log.copy(deep=True)
        allowed, diagnostics = ensemble.evaluate_live_model_skill(
            log, protocol_version=protocol.TRAINING_PROTOCOL_VERSION
        )
        self.assertFalse(any(allowed.values()))
        self.assertEqual(diagnostics["h1"]["n_paired"], 2)
        pd.testing.assert_frame_equal(log, original)

    def test_held_forecasts_cannot_authorize_the_new_method(self):
        row = {
            "training_protocol_version": protocol.TRAINING_PROTOCOL_VERSION,
            "publish_mode": "HOLD",
            "governance_run_at": "2020-01-02T15:00:00Z",
        }
        for h in [1, 3, 5]:
            row.update(
                {
                    f"forecast_candidate_h{h}": 70.0,
                    f"forecast_pre_anchor_h{h}": 75.0,
                    f"nws_h{h}": 70.0,
                    f"realized_h{h}": 70.0,
                    f"persistence_h{h}": 71.0,
                }
            )
        log = pd.DataFrame([row] * 40)
        allowed, diagnostics = ensemble.evaluate_live_model_skill(
            log, protocol_version=protocol.TRAINING_PROTOCOL_VERSION
        )
        self.assertFalse(any(allowed.values()))
        self.assertEqual(diagnostics["h1"]["n_paired"], 0)
        allowed, _ = ensemble.evaluate_nws_anchor_skill(
            log, protocol_version=protocol.TRAINING_PROTOCOL_VERSION
        )
        self.assertFalse(any(allowed.values()))

    def test_infinite_values_cannot_inflate_live_skill_sample_count(self):
        for bad_column in ["forecast_candidate", "persistence", "realized"]:
            with self.subTest(column=bad_column):
                rows = []
                for i in range(40):
                    row = {
                        "feature_date": pd.Timestamp("2020-01-01")
                        + pd.Timedelta(days=i)
                    }
                    for h in [1, 3, 5]:
                        row.update(
                            {
                                f"forecast_candidate_h{h}": 70.0,
                                f"realized_h{h}": 70.0,
                                f"persistence_h{h}": 71.0,
                            }
                        )
                        if i >= 2:
                            row[f"{bad_column}_h{h}"] = np.inf
                    rows.append(row)
                allowed, diagnostics = ensemble.evaluate_live_model_skill(
                    pd.DataFrame(rows)
                )
                self.assertEqual(allowed, {1: False, 3: False, 5: False})
                self.assertEqual(diagnostics["h1"]["n_paired"], 2)

    def test_nws_anchor_cannot_count_infinite_model_errors_as_evidence(self):
        rows = []
        for i in range(40):
            row = {}
            for h in [1, 3, 5]:
                row.update(
                    {
                        f"forecast_pre_anchor_h{h}": 75.0 if i < 2 else np.inf,
                        f"nws_h{h}": 70.0,
                        f"realized_h{h}": 70.0,
                    }
                )
            rows.append(row)
        allowed, diagnostics = ensemble.evaluate_nws_anchor_skill(pd.DataFrame(rows))
        self.assertFalse(any(allowed.values()))
        self.assertEqual(diagnostics["h1"]["n_paired"], 2)

    def test_blend_selection_never_reads_saved_test_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=3),
                    "split": ["val", "test", "test"],
                    "pred_h1": [71.0, 500.0, 600.0],
                }
            )
            frame.to_parquet(root / "val_predictions.parquet", index=False)
            with patch.object(ensemble, "PART2_DIR", root):
                selected = ensemble._load_lstm_val_predictions()
                frame.loc[frame["split"].eq("test"), "pred_h1"] = -1000.0
                frame.to_parquet(root / "val_predictions.parquet", index=False)
                pd.testing.assert_frame_equal(
                    selected, ensemble._load_lstm_val_predictions()
                )
            self.assertEqual(selected["pred_h1"].tolist(), [71.0])

    def test_blend_table_without_split_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"date": ["2020-01-01"], "pred_h1": [71.0]}).to_parquet(
                root / "val_predictions.parquet"
            )
            with patch.object(ensemble, "PART2_DIR", root):
                self.assertTrue(ensemble._load_lstm_val_predictions().empty)

    def test_bnn_cannot_attach_current_intervals_to_an_unmatched_old_row(self):
        log = pd.DataFrame({"feature_date": ["2020-01-01"], "forecast_h1": [70.0]})
        self.assertEqual(
            uncertainty._latest_log_row_for_feature_date(
                log, pd.Timestamp("2020-01-02")
            ),
            (None, None),
        )

    def test_nonfinite_interval_center_is_not_displayable_or_publishable(self):
        publish, display, flags, _ = uncertainty._compute_display_flags(
            True,
            "conformal_calibrated",
            np.array([np.nan, 70.0, 70.0]),
            np.array([70.0] * 3),
        )
        self.assertFalse(publish)
        self.assertFalse(display)
        self.assertFalse(flags[1]["displayable"])
        self.assertEqual(flags[1]["suppression_reason"], "non_finite_center")

    def test_persistence_backfill_preserves_issue_time_baseline(self):
        log = pd.DataFrame(
            {
                "feature_date": ["2020-01-01"],
                "persistence_h1": [71.0],
                "persistence_h3": [71.0],
                "persistence_h5": [np.nan],
            }
        )
        revised_history = pd.DataFrame(
            {"date": [pd.Timestamp("2020-01-01")], "temp_high_f": [80.0]}
        )
        result = attribution.compute_persistence_baseline(log, revised_history)
        self.assertEqual(
            result.loc[
                0, ["persistence_h1", "persistence_h3", "persistence_h5"]
            ].tolist(),
            [71.0] * 3,
        )

    def test_realized_backfill_requires_a_complete_pacific_day(self):
        row = {"feature_date": pd.Timestamp("2020-01-01")}
        realized = pd.DataFrame(
            {"date": [pd.Timestamp("2020-01-02")], "realized_high_f": [80.0]}
        )
        with patch.object(
            attribution,
            "pacific_today_timestamp",
            return_value=pd.Timestamp("2020-01-02"),
        ):
            result = attribution.backfill_realized(pd.DataFrame([row]), realized)
        self.assertTrue(pd.isna(result.loc[0, "realized_h1"]))


class PublicationTests(unittest.TestCase):
    def test_current_calendar_is_required_even_with_other_checks_passing(self):
        row = pd.Series(
            {
                "decision_date": "2020-01-02",
                "feature_date": "2020-01-01",
                "written_at": "2020-01-02T15:00:00+00:00",
                **{
                    f"target_date_h{h}": str(
                        (pd.Timestamp("2020-01-01") + pd.Timedelta(days=h)).date()
                    )
                    for h in [1, 3, 5]
                },
            }
        )
        hist = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]})
        with patch.object(
            governance,
            "pacific_today_timestamp",
            return_value=pd.Timestamp("2020-01-02"),
        ):
            self.assertTrue(governance.check_forecast_dates(row, hist).passed)
            for field, value in [
                ("decision_date", "2020-01-01"),
                ("feature_date", "2019-12-31"),
                ("target_date_h3", "2020-01-03"),
                ("written_at", "2020-01-03T15:00:00Z"),
            ]:
                with self.subTest(field=field):
                    invalid = row.copy()
                    invalid[field] = value
                    check = governance.check_forecast_dates(invalid, hist)
                    self.assertFalse(check.passed)
                    self.assertEqual(governance.determine_publish_mode([check]), "HOLD")

    def test_published_values_are_not_rewritten_by_forecasting_sleeves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "decision_date": "2020-01-02",
                "feature_date": "2020-01-01",
                "model": "LSTM",
                "forecast_h1": 71.0,
                "publish_mode": "CAUTION",
                "governance_run_at": "2020-01-02T15:00:00Z",
            }
            path = root / "prediction_log.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            original = path.read_bytes()
            with patch.object(forecaster, "ARTIFACTS_DIR", root):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    forecaster.upsert_log_row({**row, "forecast_h1": 90.0})
            with patch.object(ensemble, "PART2_DIR", root):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    ensemble.upsert_log_columns(
                        {"forecast_h1": 90.0},
                        row["decision_date"],
                        row["feature_date"],
                        "LSTM",
                    )
            with patch.object(uncertainty, "PART2_DIR", root):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    uncertainty.update_prediction_log_with_bnn(
                        pd.Timestamp("2020-01-01"),
                        *[np.array([70.0] * 3)] * 5,
                        True,
                        "CALIBRATED",
                        "conformal_calibrated",
                        True,
                        True,
                    )
            self.assertEqual(path.read_bytes(), original)

    def test_same_day_runner_only_refreshes_realizations(self):
        with (
            patch.object(runner, "published_forecast_exists", return_value=True),
            patch.object(runner, "audit_files", return_value=([], [])),
            patch.object(runner, "run_part", return_value=0) as run,
            patch("sys.argv", ["run_daily_forecast.py"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 0)
        run.assert_called_once_with("PART9")

    def test_forced_retrain_refits_the_regime_and_forecaster_together(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "PROJECT_DIR", Path(tmp)),
            patch.object(runner, "published_forecast_exists", return_value=False),
            patch.object(runner, "training_lineage_errors", return_value=[]),
            patch.object(runner, "audit_files", return_value=([], [])),
            patch.object(runner, "run_part", return_value=0) as run,
            patch("sys.argv", ["run_daily_forecast.py", "--retrain"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 0)
        self.assertIn(
            unittest.mock.call("PART6", extra_args=["--mode=train"]), run.call_args_list
        )
        self.assertIn(
            unittest.mock.call("PART2", extra_args=["--model=lstm", "--mode=train"]),
            run.call_args_list,
        )

    def test_predict_only_runner_refuses_unqualified_checkpoint(self):
        with (
            patch.object(runner, "published_forecast_exists", return_value=False),
            patch.object(
                runner,
                "training_lineage_errors",
                return_value=["checkpoint_requires_protocol_retrain"],
            ),
            patch.object(runner, "audit_files", return_value=([], [])),
            patch.object(runner, "run_part") as run,
            patch("sys.argv", ["run_daily_forecast.py", "--skip-train"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runner.main(), 1)
        run.assert_not_called()

    def test_schema_drift_refits_encoder_before_training_new_forecaster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts_part2").mkdir()
            (root / "artifacts_part2/lstm_model.pt").touch()
            (root / "artifacts_part2/part2_meta.json").write_text(
                json.dumps(
                    {
                        "model_type": "lstm",
                        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                )
            )
            with (
                patch.object(runner, "PROJECT_DIR", root),
                patch.object(runner, "published_forecast_exists", return_value=False),
                patch.object(runner, "training_lineage_errors", return_value=[]),
                patch.object(
                    runner, "_feature_cols_changed", return_value=(True, "changed")
                ),
                patch.object(runner, "audit_files", return_value=([], [])),
                patch.object(runner, "run_part", return_value=0) as run,
                patch("sys.argv", ["run_daily_forecast.py"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 0)
        calls = run.call_args_list
        refit = calls.index(unittest.mock.call("PART6", extra_args=["--mode=train"]))
        train = calls.index(
            unittest.mock.call("PART2", extra_args=["--model=lstm", "--mode=train"])
        )
        self.assertLess(refit, train)

    def test_legacy_artifacts_warn_and_changed_new_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts_part2").mkdir()
            path = root / "artifacts_part2/part2_meta.json"
            path.write_text(json.dumps({"model_type": "lstm"}))
            with patch.object(validator, "PROJECT_DIR", root):
                self.assertEqual(validator.check_training_lineage_contract()[0], "WARN")
                path.write_text(
                    json.dumps(
                        {
                            "training_protocol_version": protocol.TRAINING_PROTOCOL_VERSION
                        }
                    )
                )
                self.assertEqual(validator.check_training_lineage_contract()[0], "FAIL")


if __name__ == "__main__":
    unittest.main()
