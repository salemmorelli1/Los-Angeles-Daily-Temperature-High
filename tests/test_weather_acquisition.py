"""Rate-limit recovery and completed-day outcome acquisition regressions."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

import part0_data_infrastructure as data
import part9_live_attribution as attribution
import weather_http as http


def response(status, retry_after=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://archive-api.open-meteo.com/v1/archive"
    result._content = b'{"daily":{}}'
    result._content_consumed = True
    if retry_after is not None:
        result.headers["Retry-After"] = retry_after
    result.close = Mock(wraps=result.close)
    return result


class WeatherHTTPTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_retry_after_accepts_seconds_and_http_date(self):
        now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(http.retry_after_seconds("45", now), 45)
        self.assertEqual(http.retry_after_seconds("Fri, 27 Nov 2026 12:01:10 GMT", now), 70)
        self.assertEqual(http.retry_after_seconds("Fri, 27 Nov 2026 11:59:00 GMT", now), 0)
        for invalid in (None, "", "NaN", "-1", "0.5", "tomorrow"):
            self.assertIsNone(http.retry_after_seconds(invalid, now))

    def test_429_honors_server_wait_and_closes_failed_response(self):
        limited, success = response(429, "45"), response(200)
        with patch.object(http.requests, "get", side_effect=[limited, success]) as get, patch.object(http.time, "sleep") as sleep:
            self.assertIs(http.get_weather("https://example.invalid", params={"daily": "temperature_2m_max"}), success)
        sleep.assert_called_once_with(45)
        limited.close.assert_called_once()
        success.close.assert_not_called()
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.kwargs["params"], {"daily": "temperature_2m_max"})

    def test_http_date_wait_is_honored(self):
        limited = response(503, "Fri, 27 Nov 2026 12:01:10 GMT")
        with patch.object(http, "retry_after_seconds", wraps=lambda value: http_retry_at_noon(value)), patch.object(http.requests, "get", side_effect=[limited, response(200)]), patch.object(http.time, "sleep") as sleep:
            http.get_weather("https://example.invalid")
        sleep.assert_called_once_with(70)

    def test_long_provider_wait_fails_without_an_early_retry(self):
        limited = response(429, "3600")
        with patch.object(http.requests, "get", return_value=limited) as get, patch.object(http.time, "sleep") as sleep:
            with self.assertRaisesRegex(requests.RequestException, "retry budget"):
                http.get_weather("https://example.invalid")
        get.assert_called_once()
        sleep.assert_not_called()
        limited.close.assert_called_once()

    def test_persistent_429_stops_and_never_returns_cached_success(self):
        limited = [response(429) for _ in range(4)]
        with patch.object(http.requests, "get", side_effect=limited) as get, patch.object(http.time, "sleep") as sleep:
            with self.assertRaises(requests.HTTPError):
                http.get_weather("https://example.invalid")
        self.assertEqual(get.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30, 60, 60])
        for item in limited:
            item.close.assert_called_once()

    def test_permanent_client_errors_and_tls_errors_are_not_retried(self):
        for error in (response(400), response(401), response(404), requests.exceptions.SSLError("certificate verification failed")):
            with self.subTest(error=error):
                args = {"side_effect": error} if isinstance(error, Exception) else {"return_value": error}
                with patch.object(http.requests, "get", **args) as get, patch.object(http.time, "sleep") as sleep:
                    with self.assertRaises(requests.RequestException):
                        http.get_weather("https://example.invalid")
                get.assert_called_once()
                sleep.assert_not_called()

    def test_transient_network_and_503_recover_with_bounded_waits(self):
        with patch.object(http.requests, "get", side_effect=[requests.Timeout(), requests.ConnectionError(), response(503, "25"), response(200)]) as get, patch.object(http.time, "sleep") as sleep:
            self.assertEqual(http.get_weather("https://example.invalid").status_code, 200)
        self.assertEqual(get.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 25])

    def test_retries_use_real_http_status_and_header_parsing(self):
        class Handler(BaseHTTPRequestHandler):
            calls = 0

            def do_GET(self):
                Handler.calls += 1
                self.send_response(429 if Handler.calls == 1 else 200)
                self.send_header("Retry-After", "31")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.object(http.time, "sleep") as sleep, http.get_weather(f"http://127.0.0.1:{server.server_port}/archive") as result:
                    self.assertEqual(result.json(), {"ok": True})
                sleep.assert_called_once_with(31)
                self.assertEqual(Handler.calls, 2)
            finally:
                server.shutdown()
                worker.join()

    def test_part0_outage_preserves_every_old_artifact_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"date": pd.to_datetime(["2026-11-24"]), "temp_high_f": [69.7]}).to_parquet(root / "historical_daily.parquet")
            (root / "data_meta.json").write_text('{"historical_end":"2026-11-24"}')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch.object(data, "ARTIFACTS_DIR", root), patch.object(data, "pacific_today", return_value=pd.Timestamp("2026-11-26").date()), patch.object(http.requests, "get", side_effect=[response(429) for _ in range(4)]), patch.object(http.time, "sleep"), patch.object(data, "save_artifacts") as save:
                with self.assertRaises(requests.HTTPError):
                    data.main()
            save.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})


_parse_retry_after = http.retry_after_seconds


def http_retry_at_noon(value):
    return _parse_retry_after(value, datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc))


class OutcomeAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.meta = root / "data_meta.json"
        self.meta.write_text(json.dumps({
            "schema_version": "1.1.0-pacific-complete-day", "lat": attribution.LAT,
            "lon": attribution.LON, "historical_start": "2018-01-01", "historical_end": "2026-01-09",
        }))
        self.stack.enter_context(patch.object(attribution, "PART0_DIR", root))
        self.stack.enter_context(patch.object(attribution, "pacific_today_timestamp", return_value=pd.Timestamp("2026-01-10")))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.frame = attribution.add_missing_target_date_columns(pd.DataFrame([{
            "feature_date": pd.Timestamp("2026-01-01"), "decision_date": pd.Timestamp("2026-01-02"),
            "forecast_h1": 70.0, "realized_h1": 70.0,
        }]))

    def test_all_known_outcomes_skip_archive_even_without_a_cache(self):
        for h in attribution.HORIZONS:
            self.frame[f"realized_h{h}"] = 70.0
        before = self.frame.copy(deep=True)
        with patch.object(attribution, "fetch_realized_temps") as fetch:
            result = attribution.refresh_realizations(self.frame, pd.DataFrame())
        fetch.assert_not_called()
        pd.testing.assert_frame_equal(result, before)

    def test_local_cache_fills_only_blanks_and_requests_only_remaining_dates(self):
        history = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-04"]), "temp_high_f": [999.0, 72.0]})
        remote = pd.DataFrame({"date": pd.to_datetime(["2026-01-06"]), "realized_high_f": [74.0]})
        with patch.object(attribution, "fetch_realized_temps", return_value=remote) as fetch:
            result = attribution.refresh_realizations(self.frame, history)
        fetch.assert_called_once_with(pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-06"))
        self.assertEqual([result.at[0, f"realized_h{h}"] for h in attribution.HORIZONS], [70, 72, 74])
        self.assertEqual(result.at[0, "forecast_h1"], 70)
        with patch.object(attribution, "fetch_realized_temps") as fetch:
            attribution.refresh_realizations(result, history)
        fetch.assert_not_called()

    def test_today_and_future_days_never_create_outcomes_or_requests(self):
        frame = attribution.add_missing_target_date_columns(pd.DataFrame([{"feature_date": pd.Timestamp("2026-01-09"), "decision_date": pd.Timestamp("2026-01-10")}]))
        history = pd.DataFrame({"date": pd.date_range("2026-01-10", periods=5), "temp_high_f": 80.0})
        with patch.object(attribution, "fetch_realized_temps") as fetch:
            result = attribution.refresh_realizations(frame, history)
        fetch.assert_not_called()
        self.assertTrue(result[[f"realized_h{h}" for h in attribution.HORIZONS]].isna().all().all())

    def test_foreign_cache_identity_cannot_supply_outcomes(self):
        metadata = json.loads(self.meta.read_text())
        metadata["lat"] = 40.0
        self.meta.write_text(json.dumps(metadata))
        history = pd.DataFrame({"date": pd.to_datetime(["2026-01-04"]), "temp_high_f": [72.0]})
        with patch.object(attribution, "fetch_realized_temps", return_value=pd.DataFrame()) as fetch:
            result = attribution.refresh_realizations(self.frame, history)
        fetch.assert_called_once_with(pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-06"))
        self.assertTrue(pd.isna(result.at[0, "realized_h3"]))

    def test_missing_cache_metadata_requires_remote_source(self):
        self.meta.unlink()
        history = pd.DataFrame({"date": pd.to_datetime(["2026-01-04"]), "temp_high_f": [72.0]})
        self.assertTrue(attribution.cached_realized_temps(history).empty)

    def test_duplicate_conflicting_cache_values_are_ignored(self):
        history = pd.DataFrame({"date": pd.to_datetime(["2026-01-04", "2026-01-04"]), "temp_high_f": [72.0, 75.0]})
        self.assertTrue(attribution.cached_realized_temps(history).empty)

    def test_nonfinite_cache_and_remote_outcomes_remain_missing_after_outage(self):
        self.frame["realized_h3"] = np.inf
        history = pd.DataFrame({"date": pd.to_datetime(["2026-01-04", "2026-01-06"]), "temp_high_f": [np.nan, np.inf]})
        before = self.frame.copy(deep=True)
        with patch.object(attribution, "fetch_realized_temps", return_value=pd.DataFrame()) as fetch:
            result = attribution.refresh_realizations(self.frame, history)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(result.at[0, "realized_h1"], before.at[0, "realized_h1"])
        self.assertTrue(pd.isna(result.at[0, "realized_h5"]))
        self.assertEqual(list(attribution.pending_realization_dates(result)), [pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-06")])


if __name__ == "__main__":
    unittest.main()
