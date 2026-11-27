"""Bounded weather GET retries; unavailable inputs remain unavailable."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def retry_after_seconds(value: str | None, now: datetime | None = None) -> float | None:
    """Read HTTP Retry-After (whole seconds or HTTP date), never a negative wait."""
    if not value:
        return None
    value = value.strip()
    if value.isascii() and value.isdecimal():
        return float(value)
    try:
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - (now or datetime.now(timezone.utc))).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def get_weather(url, *, params=None, headers=None, timeout=30,
                attempts=4, max_total_wait=180) -> requests.Response:
    """Retry transient failures with at most four requests and 180s of waits.

    A longer Retry-After fails without sending an early retry. No stale response
    or fabricated data substitutes for a failed request. The caller owns a
    successful response; failed responses are closed before retrying/raising.
    """
    if attempts < 1 or max_total_wait < 0:
        raise ValueError("Invalid weather retry limits")
    waited = 0.0
    host = urlsplit(url).netloc
    for attempt in range(1, attempts + 1):
        response = None
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            status = response.status_code if response is not None else None
            retry_after = retry_after_seconds(response.headers.get("Retry-After")) if response is not None else None
            if response is not None:
                response.close()
            transient = status in RETRYABLE_STATUS or (
                status is None and isinstance(error, (requests.Timeout, requests.ConnectionError))
                and not isinstance(error, requests.exceptions.SSLError)
            )
            if not transient or attempt == attempts:
                if transient:
                    print(f"[Weather HTTP] {host}: {status or type(error).__name__} after {attempt} attempts; inputs unavailable. Retry the workflow later.")
                raise
            base = 30 if status == 429 else 5
            delay = max(min(base * 2 ** (attempt - 1), 60), retry_after or 0.0)
            if delay > max_total_wait - waited:
                raise requests.RequestException(
                    f"{host}: HTTP {status}; Retry-After/backoff exceeds the remaining "
                    f"{max_total_wait - waited:g}s retry budget. Inputs unavailable; retry later."
                ) from error
            print(f"[Weather HTTP] {host}: {status or type(error).__name__} on attempt {attempt}/{attempts}; waiting {delay:g}s before retry.")
            time.sleep(delay)
            waited += delay
    raise AssertionError("Unreachable weather retry state")
