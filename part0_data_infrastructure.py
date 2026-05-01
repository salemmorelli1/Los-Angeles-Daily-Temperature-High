#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 0 — Data Infrastructure
==============================
Fetches, caches, and normalizes weather data for the LA Temperature
Forecasting stack.

Data Sources (all free, no API key required)
---------------------------------------------
  1. Open-Meteo Archive API  — historical daily observations back to 1940
  2. Open-Meteo Forecast API — 7-day ahead gridded forecast
  3. NWS API (api.weather.gov) — official NWS point forecast (benchmark)
  4. NWS KLAX station observations — recent hourly obs

Artifacts Written
-----------------
  artifacts_part0/
      historical_daily.parquet     — full historical training record
      recent_observations.parquet  — last 30 days of KLAX hourly obs
      nws_official_forecast.json   — NWS 7-day official forecast (benchmark)
      openmeteo_forecast.parquet   — Open-Meteo 7-day gridded forecast
      data_meta.json               — fetch timestamps, row counts, schema version
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"

# KLAX coordinates (Los Angeles International Airport)
LAT = 33.9425
LON = -118.4081

# NWS grid point for KLAX  (office=LOX, gridX=155, gridY=49)
NWS_OFFICE = "LOX"
NWS_GRID_X = 155
NWS_GRID_Y = 49
NWS_STATION = "KLAX"

# Open-Meteo endpoints
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# NWS endpoints
NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "LA-Temp-Forecast/1.0 (research; contact: user@example.com)",
    "Accept": "application/geo+json",
}

# Training window: pull data from 2018-01-01 onward
HISTORY_START = "2018-01-01"

# Open-Meteo daily variables
OM_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "precipitation_hours",
    "sunshine_duration",
]

# Open-Meteo hourly variables (used for daily aggregation / pressure)
OM_HOURLY_VARS = [
    "surface_pressure",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]

RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def resolve_project_dir() -> Path:
    env_root = os.environ.get("LATEMP_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = resolve_project_dir()
ARTIFACTS_DIR = PROJECT_DIR / "artifacts_part0"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LATEMP_ROOT", str(PROJECT_DIR))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None,
         timeout: int = 30) -> requests.Response:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"  [WARN] Request failed (attempt {attempt + 1}): {exc}. Retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    raise RuntimeError("Should not reach here")


# ---------------------------------------------------------------------------
# Open-Meteo: historical daily data
# ---------------------------------------------------------------------------
def fetch_historical_daily(
    start_date: str = HISTORY_START,
    end_date: Optional[str] = None,
    lat: float = LAT,
    lon: float = LON,
) -> pd.DataFrame:
    """Pull daily temperature/weather observations from Open-Meteo archive."""
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[Part 0] Fetching Open-Meteo historical daily: {start_date} → {end_date}")

    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(OM_DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
    }

    resp = _get(OM_ARCHIVE_URL, params=params)
    payload = resp.json()

    daily = payload.get("daily", {})
    if not daily:
        raise RuntimeError("Open-Meteo archive returned empty daily block.")

    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.drop(columns=["time"])
    df = df.sort_values("date").reset_index(drop=True)

    # Canonical column renames
    col_map = {
        "temperature_2m_max": "temp_high_f",
        "temperature_2m_min": "temp_low_f",
        "temperature_2m_mean": "temp_mean_f",
        "apparent_temperature_max": "feels_like_max_f",
        "precipitation_sum": "precip_in",
        "wind_speed_10m_max": "wind_speed_max_mph",
        "wind_gusts_10m_max": "wind_gust_max_mph",
        "wind_direction_10m_dominant": "wind_dir_dominant_deg",
        "shortwave_radiation_sum": "solar_radiation_mj",
        "et0_fao_evapotranspiration": "et0_in",
        "precipitation_hours": "precip_hours",
        "sunshine_duration": "sunshine_seconds",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    print(f"  → {len(df)} daily rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Open-Meteo: hourly data for pressure/humidity (aggregated to daily)
# ---------------------------------------------------------------------------
def fetch_hourly_aggregated(
    start_date: str = HISTORY_START,
    end_date: Optional[str] = None,
    lat: float = LAT,
    lon: float = LON,
) -> pd.DataFrame:
    """Pull hourly data and aggregate to daily mean/range for pressure, humidity, dew point."""
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[Part 0] Fetching Open-Meteo hourly for aggregation: {start_date} → {end_date}")

    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OM_HOURLY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Los_Angeles",
    }

    resp = _get(OM_ARCHIVE_URL, params=params)
    payload = resp.json()

    hourly = payload.get("hourly", {})
    if not hourly:
        print("  [WARN] Open-Meteo returned empty hourly block — skipping.")
        return pd.DataFrame()

    df_h = pd.DataFrame(hourly)
    df_h["datetime"] = pd.to_datetime(df_h["time"])
    df_h["date"] = df_h["datetime"].dt.normalize()
    df_h = df_h.drop(columns=["time"])

    # Daily aggregations
    agg = df_h.groupby("date").agg(
        pressure_mean_hpa=("surface_pressure", "mean"),
        pressure_min_hpa=("surface_pressure", "min"),
        pressure_max_hpa=("surface_pressure", "max"),
        humidity_mean_pct=("relative_humidity_2m", "mean"),
        humidity_min_pct=("relative_humidity_2m", "min"),
        dew_point_mean_f=("dew_point_2m", "mean"),
        dew_point_max_f=("dew_point_2m", "max"),
        cloud_cover_mean_pct=("cloud_cover", "mean"),
        wind_speed_mean_mph=("wind_speed_10m", "mean"),
    ).reset_index()

    print(f"  → {len(agg)} daily aggregated rows from hourly data")
    return agg


# ---------------------------------------------------------------------------
# Open-Meteo: 7-day forecast
# ---------------------------------------------------------------------------
def fetch_openmeteo_forecast(lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    """Pull the current Open-Meteo 7-day forecast for LA."""
    print("[Part 0] Fetching Open-Meteo 7-day forecast...")

    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(OM_DAILY_VARS),
        "forecast_days": 7,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
    }

    resp = _get(OM_FORECAST_URL, params=params)
    payload = resp.json()

    daily = payload.get("daily", {})
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.drop(columns=["time"])

    col_map = {
        "temperature_2m_max": "temp_high_f",
        "temperature_2m_min": "temp_low_f",
        "temperature_2m_mean": "temp_mean_f",
        "apparent_temperature_max": "feels_like_max_f",
        "precipitation_sum": "precip_in",
        "wind_speed_10m_max": "wind_speed_max_mph",
        "wind_gusts_10m_max": "wind_gust_max_mph",
        "wind_direction_10m_dominant": "wind_dir_dominant_deg",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    print(f"  → {len(df)} forecast days")
    return df


# ---------------------------------------------------------------------------
# NWS: official point forecast (benchmark)
# ---------------------------------------------------------------------------
def fetch_nws_official_forecast(
    office: str = NWS_OFFICE,
    grid_x: int = NWS_GRID_X,
    grid_y: int = NWS_GRID_Y,
) -> Dict[str, Any]:
    """Pull the NWS official 7-day text forecast for the LA grid point."""
    url = f"{NWS_BASE}/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
    print(f"[Part 0] Fetching NWS official forecast: {url}")

    try:
        resp = _get(url, headers=NWS_HEADERS)
        payload = resp.json()

        periods = payload.get("properties", {}).get("periods", [])

        # Aggregate hourly to daily high temps
        records: Dict[str, List[float]] = {}
        for period in periods:
            start = pd.to_datetime(period.get("startTime", "")).date()
            temp_f = period.get("temperature", None)
            if temp_f is not None and start:
                day_str = str(start)
                records.setdefault(day_str, []).append(float(temp_f))

        daily_highs = {
            day: max(temps) for day, temps in sorted(records.items())
        }

        result = {
            "source": "NWS",
            "office": office,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "fetched_at": pd.Timestamp.now().isoformat(),
            "daily_high_f": daily_highs,
            "raw_period_count": len(periods),
        }
        print(f"  → {len(daily_highs)} forecast days from NWS")
        return result

    except Exception as exc:
        print(f"  [WARN] NWS forecast fetch failed: {exc}")
        return {
            "source": "NWS",
            "fetched_at": pd.Timestamp.now().isoformat(),
            "daily_high_f": {},
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# NWS: recent KLAX station observations
# ---------------------------------------------------------------------------
def fetch_klax_observations(station: str = NWS_STATION, days_back: int = 30) -> pd.DataFrame:
    """Pull recent hourly observations from KLAX via NWS API."""
    url = f"{NWS_BASE}/stations/{station}/observations"
    start_iso = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
    print(f"[Part 0] Fetching KLAX observations: last {days_back} days")

    try:
        resp = _get(url, params={"start": start_iso, "limit": 500}, headers=NWS_HEADERS)
        payload = resp.json()

        features = payload.get("features", [])
        records = []
        for feat in features:
            props = feat.get("properties", {})
            obs_time = pd.to_datetime(props.get("timestamp", ""), errors="coerce")
            if pd.isna(obs_time):
                continue

            def _val(key: str) -> Optional[float]:
                v = props.get(key, {})
                if isinstance(v, dict):
                    return v.get("value")
                return None

            # Convert Celsius to Fahrenheit where needed
            temp_c = _val("temperature")
            dew_c = _val("dewpoint")
            temp_f = (temp_c * 9 / 5 + 32) if temp_c is not None else None
            dew_f = (dew_c * 9 / 5 + 32) if dew_c is not None else None

            wind_mps = _val("windSpeed")
            wind_mph = (wind_mps * 2.237) if wind_mps is not None else None

            records.append({
                "datetime": obs_time.tz_localize(None) if obs_time.tzinfo else obs_time,
                "temp_f": temp_f,
                "dew_point_f": dew_f,
                "humidity_pct": _val("relativeHumidity"),
                "pressure_hpa": _val("seaLevelPressure"),
                "wind_speed_mph": wind_mph,
                "wind_dir_deg": _val("windDirection"),
                "visibility_m": _val("visibility"),
            })

        df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
        print(f"  → {len(df)} KLAX observation records")
        return df

    except Exception as exc:
        print(f"  [WARN] KLAX observations fetch failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Merge and build master historical DataFrame
# ---------------------------------------------------------------------------
def build_master_historical(
    df_daily: pd.DataFrame,
    df_hourly_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily and hourly-aggregated DataFrames on date."""
    df = df_daily.copy()

    if not df_hourly_agg.empty:
        df = df.merge(df_hourly_agg, on="date", how="left")

    # Ensure date column is normalized datetime
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill small gaps (up to 2 days) for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill(limit=2)

    print(f"[Part 0] Master historical record: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_artifacts(
    df_historical: pd.DataFrame,
    df_obs: pd.DataFrame,
    nws_forecast: Dict[str, Any],
    df_om_forecast: pd.DataFrame,
) -> None:
    """Write all Part 0 artifacts to disk."""
    hist_path = ARTIFACTS_DIR / "historical_daily.parquet"
    df_historical.to_parquet(hist_path, index=False)
    print(f"[Part 0] Saved historical_daily.parquet ({len(df_historical)} rows)")

    if not df_obs.empty:
        obs_path = ARTIFACTS_DIR / "recent_observations.parquet"
        df_obs.to_parquet(obs_path, index=False)
        print(f"[Part 0] Saved recent_observations.parquet ({len(df_obs)} rows)")

    nws_path = ARTIFACTS_DIR / "nws_official_forecast.json"
    with open(nws_path, "w") as f:
        json.dump(nws_forecast, f, indent=2, default=str)
    print("[Part 0] Saved nws_official_forecast.json")

    if not df_om_forecast.empty:
        fc_path = ARTIFACTS_DIR / "openmeteo_forecast.parquet"
        df_om_forecast.to_parquet(fc_path, index=False)
        print("[Part 0] Saved openmeteo_forecast.parquet")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": pd.Timestamp.now().isoformat(),
        "historical_rows": len(df_historical),
        "historical_start": str(df_historical["date"].min().date()),
        "historical_end": str(df_historical["date"].max().date()),
        "historical_columns": list(df_historical.columns),
        "obs_rows": len(df_obs),
        "nws_forecast_days": len(nws_forecast.get("daily_high_f", {})),
        "om_forecast_days": len(df_om_forecast),
        "lat": LAT,
        "lon": LON,
        "station": NWS_STATION,
    }
    with open(ARTIFACTS_DIR / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[Part 0] Saved data_meta.json")


def load_historical() -> pd.DataFrame:
    """Load the cached historical daily parquet. Returns empty DataFrame if missing."""
    path = ARTIFACTS_DIR / "historical_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_nws_forecast() -> Dict[str, Any]:
    path = ARTIFACTS_DIR / "nws_official_forecast.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_om_forecast() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "openmeteo_forecast.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Incremental update: only fetch new dates
# ---------------------------------------------------------------------------
def incremental_update() -> pd.DataFrame:
    """Fetch only missing dates since the last cached record."""
    df_existing = load_historical()
    if df_existing.empty:
        print("[Part 0] No cache found — running full fetch.")
        return pd.DataFrame()

    last_date = pd.Timestamp(df_existing["date"].max()).date()
    today = date.today()
    if last_date >= today - timedelta(days=1):
        print(f"[Part 0] Cache is up-to-date (last: {last_date}). Skipping historical refetch.")
        return df_existing

    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[Part 0] Incremental fetch from {start}...")
    return pd.DataFrame()  # Signal to caller to fetch from `start`


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[Part 0] Project root: {PROJECT_DIR}")
    print(f"[Part 0] Artifacts dir: {ARTIFACTS_DIR}")

    # Check for incremental update opportunity
    df_existing = load_historical()
    if not df_existing.empty:
        last_date = pd.Timestamp(df_existing["date"].max()).date()
        today = date.today()
        if last_date >= today - timedelta(days=1):
            print(f"[Part 0] Historical cache is current (last: {last_date}).")
            start_date = HISTORY_START
            do_full = False
        else:
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            do_full = True
    else:
        start_date = HISTORY_START
        do_full = True

    # Fetch historical data
    if do_full or df_existing.empty:
        df_daily = fetch_historical_daily(start_date=start_date)
        df_hourly = fetch_hourly_aggregated(start_date=start_date)
        df_new = build_master_historical(df_daily, df_hourly)

        if not df_existing.empty and not df_new.empty:
            df_historical = pd.concat([df_existing, df_new], ignore_index=True)
            df_historical = df_historical.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        else:
            df_historical = df_new if not df_new.empty else df_existing
    else:
        df_historical = df_existing

    # Always fetch live data
    df_obs = fetch_klax_observations(days_back=30)
    nws_forecast = fetch_nws_official_forecast()
    df_om_forecast = fetch_openmeteo_forecast()

    # Save everything
    save_artifacts(df_historical, df_obs, nws_forecast, df_om_forecast)

    print(f"\n[Part 0] ✅ Complete. {len(df_historical)} historical rows ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
