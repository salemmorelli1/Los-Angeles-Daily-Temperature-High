"""CPU integration tests on synthetic data; no live forecasts or network calls."""

from __future__ import annotations

import contextlib
import importlib.util
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
import part6_weather_regime_engine as regimes


def synthetic_history(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(713)
    high = 70 + 8 * np.sin(np.arange(n) / 30) + rng.normal(0, 2, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "temp_high_f": high,
            "temp_low_f": high - rng.uniform(8, 18, n),
            "dew_point_mean_f": rng.normal(52, 3, n),
            "humidity_mean_pct": rng.uniform(40, 85, n),
            "cloud_cover_mean_pct": rng.uniform(0, 100, n),
            "pressure_mean_hpa": rng.normal(1013, 5, n),
            "wind_speed_max_mph": rng.uniform(3, 20, n),
            "wind_dir_dominant_deg": rng.uniform(0, 360, n),
            "precip_in": rng.uniform(0, 0.1, n),
        }
    )


class XGBoostRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Install xgboost-cpu")
    def test_cpu_xgboost_wrapper_fits_with_early_stopping(self):
        rng = np.random.default_rng(84)
        x = rng.normal(size=(160, 5)).astype(np.float32)
        y = (70 + x[:, 0] * 2 + rng.normal(0, 0.5, len(x))).astype(np.float32)
        params = {**ensemble.XGB_PARAMS, "n_estimators": 24, "n_jobs": 1}
        with patch.object(ensemble, "XGB_PARAMS", params):
            model = ensemble.train_xgb_horizon(x[:120], y[:120], x[120:], y[120:])
        prediction = model.predict(x[120:])
        self.assertEqual(prediction.shape, (40,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLessEqual(len(model.evals_result()["validation_0"]["mae"]), 24)


class RegimeRuntimeTests(unittest.TestCase):
    def test_kmeans_fallback_bundle_is_frozen_when_new_data_arrives(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(tmp)
            history = synthetic_history(400)
            with (
                patch.object(regimes, "ARTIFACTS_DIR", root),
                patch.object(regimes, "load_historical", return_value=history),
                patch.object(
                    regimes, "fit_hmm", side_effect=RuntimeError("exercise fallback")
                ),
            ):
                self.assertEqual(regimes.main("train"), 0)
            old = pd.read_parquet(root / "regime_tape.parquet")
            digest = protocol.file_sha256(root / "regime_model.pkl")
            new_history = synthetic_history(410)
            new_history.iloc[:400] = history
            with (
                patch.object(regimes, "ARTIFACTS_DIR", root),
                patch.object(regimes, "load_historical", return_value=new_history),
                patch.object(
                    regimes, "fit_hmm", side_effect=AssertionError("must not refit")
                ),
                patch.object(
                    regimes,
                    "fit_kmeans_fallback",
                    side_effect=AssertionError("must not refit"),
                ),
            ):
                self.assertEqual(regimes.main("predict"), 0)
            new = pd.read_parquet(root / "regime_tape.parquet")
            pd.testing.assert_frame_equal(
                old, new.iloc[: len(old)].reset_index(drop=True)
            )
            self.assertEqual(protocol.file_sha256(root / "regime_model.pkl"), digest)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("hmmlearn"),
    "Install the CPU integration dependencies",
)
class ModelRuntimeTests(unittest.TestCase):
    def test_fitted_stack_survives_new_day_and_rejects_replaced_preprocessing(self):
        import torch
        from hmmlearn.hmm import GaussianHMM

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.ExitStack() as stack,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(tmp)
            for name in [
                "artifacts_part1",
                "artifacts_part2",
                "artifacts_part2c",
                "artifacts_part6",
            ]:
                (root / name).mkdir()
            for module, values in [
                (regimes, {"ARTIFACTS_DIR": root / "artifacts_part6"}),
                (features, {"ARTIFACTS_DIR": root / "artifacts_part1"}),
                (
                    forecaster,
                    {
                        "PROJECT_DIR": root,
                        "PART1_DIR": root / "artifacts_part1",
                        "ARTIFACTS_DIR": root / "artifacts_part2",
                        "MAX_EPOCHS": 2,
                        "PATIENCE": 1,
                        "HIDDEN_SIZE": 16,
                        "NUM_LAYERS": 1,
                    },
                ),
                (
                    uncertainty,
                    {
                        "PROJECT_DIR": root,
                        "PART1_DIR": root / "artifacts_part1",
                        "PART2_DIR": root / "artifacts_part2",
                        "ARTIFACTS_DIR": root / "artifacts_part2c",
                        "HIDDEN_SIZE": 16,
                        "NUM_LAYERS": 1,
                        "N_MC_SAMPLES": 4,
                    },
                ),
            ]:
                for key, value in values.items():
                    stack.enter_context(patch.object(module, key, value))
            history = synthetic_history(400)
            with (
                patch.object(regimes, "load_historical", return_value=history),
                patch.object(
                    regimes,
                    "fit_hmm",
                    side_effect=lambda x, n_regimes: GaussianHMM(
                        n_components=n_regimes,
                        covariance_type="diag",
                        random_state=42,
                        n_iter=10,
                    ).fit(x),
                ),
            ):
                self.assertEqual(regimes.main("train"), 0)
            old_tape = pd.read_parquet(root / "artifacts_part6/regime_tape.parquet")
            regime_hash = protocol.file_sha256(
                root / "artifacts_part6/regime_model.pkl"
            )
            matrix = features.build_feature_matrix(history, old_tape)
            columns = [
                c
                for c in matrix
                if c not in ["date", "target_h1", "target_h3", "target_h5"]
            ]
            features.save_artifacts(matrix, features.compute_splits(matrix), columns)
            self.assertEqual(forecaster.main("lstm", "train"), 0)
            meta = json.loads((root / "artifacts_part2/part2_meta.json").read_text())
            self.assertEqual(protocol.training_lineage_errors(root), [])
            fit_meta = json.loads(
                (root / "artifacts_part6/regime_meta.json").read_text()
            )
            self.assertEqual(fit_meta["backend"], "hmm")
            self.assertLessEqual(
                pd.Timestamp(fit_meta["fit_end_date"]),
                pd.Timestamp(meta["training_splits"]["train_effective_end"]),
            )

            future = synthetic_history(401)
            future.iloc[:400] = history
            with (
                patch.object(regimes, "load_historical", return_value=future),
                patch.object(
                    regimes,
                    "fit_hmm",
                    side_effect=AssertionError("must reuse fitted encoder"),
                ),
            ):
                self.assertEqual(regimes.main("predict"), 0)
            new_tape = pd.read_parquet(root / "artifacts_part6/regime_tape.parquet")
            pd.testing.assert_frame_equal(
                old_tape, new_tape.iloc[:400].reset_index(drop=True)
            )
            matrix = features.build_feature_matrix(future, new_tape)
            features.save_artifacts(matrix, features.compute_splits(matrix), columns)
            self.assertEqual(forecaster.main("lstm", "predict"), 0)
            self.assertEqual(
                protocol.file_sha256(root / "artifacts_part6/regime_model.pkl"),
                regime_hash,
            )
            self.assertEqual(protocol.training_lineage_errors(root), [])

            with (
                patch.object(uncertainty, "check_bnn_gate", return_value=True),
                patch.object(
                    uncertainty,
                    "load_splits",
                    side_effect=AssertionError("must use checkpoint split"),
                ),
            ):
                self.assertEqual(uncertainty.main(), 0)
            cal = json.loads(
                (root / "artifacts_part2c/calibration_report.json").read_text()
            )
            self.assertEqual(cal["training_splits"], meta["training_splits"])
            self.assertEqual(cal["checkpoint_sha256"], meta["checkpoint_sha256"])
            bnn = pd.read_parquet(root / "artifacts_part2c/bnn_predictions.parquet")
            self.assertLessEqual(
                pd.to_datetime(bnn["date"]).max(),
                pd.Timestamp(meta["training_splits"]["test_end"]),
            )

            path = root / "artifacts_part2/feature_scaler.pkl"
            path.write_bytes(path.read_bytes() + b"changed preprocessing bundle")
            self.assertIn(
                "feature_scaler_sha256_mismatch", protocol.training_lineage_errors(root)
            )
            with self.assertRaisesRegex(ValueError, "lineage"):
                forecaster.load_training_artifacts("lstm", len(columns))


if __name__ == "__main__":
    unittest.main()
