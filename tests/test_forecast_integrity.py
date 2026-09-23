from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import part1_feature_builder as feature_builder
import part2b_xgb_ensemble as ensemble
import part2c_bnn_sleeve as bnn
import part6_weather_regime_engine as regimes
import part9_live_attribution as attribution
from forecast_protocol import purged_labeled_splits


class CalendarTargetTests(unittest.TestCase):
    def test_targets_follow_calendar_dates_across_a_missing_day(self) -> None:
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04"]),
            "temp_high_f": [10.0, 30.0, 40.0],
        })

        result = feature_builder.add_target_columns(frame)

        self.assertTrue(pd.isna(result.loc[0, "target_h1"]))
        self.assertEqual(result.loc[0, "target_h3"], 40.0)
        self.assertEqual(result.loc[1, "target_h1"], 40.0)


class PurgedSplitTests(unittest.TestCase):
    def test_purge_removes_horizon_overlap_at_train_and_validation_edges(self) -> None:
        dates = pd.date_range("2025-01-01", "2025-03-01", freq="D")
        frame = pd.DataFrame({
            "date": dates,
            "target_h1": 1.0,
            "target_h3": 3.0,
            "target_h5": 5.0,
        })
        train_end = pd.Timestamp("2025-01-20")
        val_end = pd.Timestamp("2025-02-10")
        train, val, test = purged_labeled_splits(
            frame,
            {"train_end": train_end, "val_end": val_end},
            ["target_h1", "target_h3", "target_h5"],
        )

        self.assertEqual(pd.Timestamp(train["date"].max()), train_end - pd.Timedelta(days=5))
        self.assertEqual(pd.Timestamp(val["date"].min()), train_end + pd.Timedelta(days=1))
        self.assertEqual(pd.Timestamp(val["date"].max()), val_end - pd.Timedelta(days=5))
        self.assertEqual(pd.Timestamp(test["date"].min()), val_end + pd.Timedelta(days=1))


class PersistenceAndLiveGateTests(unittest.TestCase):
    def test_validation_persistence_uses_last_known_feature_date_observation(self) -> None:
        frame = pd.DataFrame({
            "temp_high_f": [70.0],
            "temp_high_f_lag1": [99.0],
            "target_h1": [80.0],
            "target_h3": [81.0],
            "target_h5": [82.0],
        })

        maes = ensemble.naive_persistence_mae(frame)

        self.assertEqual(maes, {"h1": 10.0, "h3": 11.0, "h5": 12.0})

    def test_xgb_validation_gate_is_applied_per_horizon(self) -> None:
        selected = ensemble.select_gate_approved_xgb_predictions(
            {"h1": 75.0, "h3": 74.0, "h5": 73.0},
            {
                "h1": {"passed": True},
                "h3": {"passed": False},
                "h5": {"passed": True},
            },
        )

        self.assertEqual(selected, {"h1": 75.0, "h5": 73.0})

    def test_live_skill_gate_is_per_horizon_and_fails_closed(self) -> None:
        n = 40
        realized = np.arange(n, dtype=float) + 70.0
        frame = pd.DataFrame({
            "feature_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "decision_date": pd.date_range("2025-01-02", periods=n, freq="D"),
        })
        frame["realized_h1"] = realized
        frame["persistence_h1"] = realized + 2.0
        frame["forecast_candidate_h1"] = realized
        frame["realized_h3"] = realized
        frame["persistence_h3"] = realized + 1.0
        frame["forecast_candidate_h3"] = realized + 2.0
        frame["realized_h5"] = realized
        frame["persistence_h5"] = realized + 1.0
        frame["forecast_candidate_h5"] = np.r_[realized[:5], [np.nan] * (n - 5)]

        allowed, diagnostics = ensemble.evaluate_live_model_skill(frame)

        self.assertTrue(allowed[1])
        self.assertFalse(allowed[3])
        self.assertFalse(allowed[5])
        self.assertEqual(diagnostics["h5"]["reason"], "insufficient_paired_history")

    def test_undercovered_live_intervals_are_suppressed_per_horizon(self) -> None:
        frame = pd.DataFrame({
            "realized_h1": np.r_[np.zeros(21), np.ones(9)],
            "bnn_lo90_h1": np.zeros(30),
            "bnn_hi90_h1": np.zeros(30),
            "realized_h3": np.zeros(30),
            "bnn_lo90_h3": np.full(30, -1.0),
            "bnn_hi90_h3": np.full(30, 1.0),
            "realized_h5": np.zeros(30),
            "bnn_lo90_h5": np.full(30, -1.0),
            "bnn_hi90_h5": np.full(30, 1.0),
        })

        coverage = bnn.evaluate_live_interval_coverage(frame)
        _, aggregate, flags, _ = bnn._compute_display_flags(
            cal_pass=True,
            interval_label="conformal_calibrated",
            live_center_f=np.array([70.0, 70.0, 70.0]),
            live_mean_f=np.array([70.0, 70.0, 70.0]),
            live_coverage_gate=coverage,
        )

        self.assertFalse(flags[1]["displayable"])
        self.assertTrue(flags[1]["suppressed"])
        self.assertTrue(flags[3]["displayable"])
        self.assertFalse(aggregate)


class CausalRegimeTests(unittest.TestCase):
    def test_imputation_medians_come_only_from_the_training_window(self) -> None:
        frame = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=4, freq="D"),
            "x": [1.0, 3.0, 1000.0, np.nan],
        })

        imputed, medians = regimes.impute_regime_features(frame, ["x"], n_fit_rows=3)

        self.assertEqual(medians["x"], 3.0)
        self.assertEqual(imputed.loc[3, "x"], 3.0)

    def test_filtered_probabilities_do_not_depend_on_future_observations(self) -> None:
        class FakeHMM:
            n_components = 2
            startprob_ = np.array([0.5, 0.5])
            transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

            def _compute_log_likelihood(self, x: np.ndarray) -> np.ndarray:
                signal = x[:, 0]
                return np.column_stack([-signal**2, -(signal - 1.0) ** 2])

        model = FakeHMM()
        x1 = np.array([[0.2], [0.8], [0.1]])
        x2 = np.array([[0.2], [0.8], [0.9]])

        states1, probs1 = regimes.causal_hmm_filter(model, x1)
        states2, probs2 = regimes.causal_hmm_filter(model, x2)

        self.assertTrue(np.allclose(probs1[:2], probs2[:2]))
        self.assertEqual(states1.shape, (3,))
        self.assertEqual(probs1.shape, (3, 2))


class PairedAttributionTests(unittest.TestCase):
    def test_skill_uses_same_rows_for_model_and_persistence(self) -> None:
        n = 40
        realized = np.arange(n, dtype=float) + 70.0
        frame = pd.DataFrame({
            "feature_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "decision_date": pd.date_range("2025-01-01", periods=n, freq="D"),
        })
        for h in attribution.HORIZONS:
            frame[f"realized_h{h}"] = realized
            frame[f"forecast_h{h}"] = np.r_[realized[:30] - 1.0, realized[30:] + 10.0]
            frame[f"persistence_h{h}"] = np.r_[realized[:30] + 2.0, [np.nan] * 10]

        metrics = attribution.compute_metrics(
            frame,
            pd.DataFrame(columns=["doy", "clim_normal_f"]),
        )

        self.assertEqual(metrics["h1"]["n_samples"], 40)
        self.assertEqual(metrics["h1"]["n_paired_persistence"], 30)
        self.assertEqual(metrics["h1"]["paired_model_mae_f"], 1.0)
        self.assertEqual(metrics["h1"]["persistence_mae_f"], 2.0)
        self.assertEqual(metrics["h1"]["skill_vs_persistence"], 0.5)
        self.assertEqual(metrics["h1"]["mae_f"], 3.25)


if __name__ == "__main__":
    unittest.main()
