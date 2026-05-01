# 🌡️ LA Temperature Forecast

A production-grade deep learning pipeline for forecasting the **Los Angeles daily temperature high** at H=1, H=3, and H=5 days ahead.

Built on the same modular architecture as the PriceCall stack — with trading strategy and execution components replaced by meteorological physics, regime detection, and forecast verification against NWS official forecasts.



## Corrected production contracts

This version applies the pipeline-contract fixes from the code audit:

- Forecast target dates are now `target_date_h = feature_date + h calendar days`.
- Part 1 keeps the live feature tail even when future targets are not yet observable.
- Part 2 trains only on fully labeled rows, but live prediction uses the newest feature row.
- Part 2 validation metrics labeled `_f` are computed in real Fahrenheit units, not scaled units.
- Part 2 supports `--mode train` and `--mode predict`; the daily runner now uses predict-only when retraining is not needed.
- Part 2C skips Transformer artifacts safely and fixes MC-dropout uncertainty scaling.
- Part 9 backfills realized values against explicit target-date columns instead of `decision_date + h`.

**No API key required.** All data sources are free and open:
- [Open-Meteo](https://open-meteo.com/) — historical archive + 7-day forecast
- [NWS API](https://api.weather.gov/) — official NWS point forecast (benchmark)

---

## 📐 Architecture

```
Part 0  Data Infrastructure      (Open-Meteo archive + NWS API)
  ↓
Part 6  Weather Regime Engine     (HMM: Marine Layer / Dry Clear / Santa Ana)
  ↓
Part 1  Feature Builder           (lags, rolling stats, calendar, regime)
  ↓
Part 2A Atmospheric Alpha         (pressure gradients, ENSO, Santa Ana index)
  ↓
Part 2  Deep Learning Forecaster  (LSTM or Transformer, H=1/3/5)
  ↓
Part 2B XGBoost Ensemble*         (gradient boosting sleeve, optional)
  ↓
Part 2C BNN Uncertainty*          (Monte Carlo Dropout, confidence intervals)
  ↓
Part 3  Forecast Governance       (staleness gates, bounds checks, publish mode)
  ↓
Part 9  Live Attribution          (MAE/RMSE/Skill vs NWS + climatology)
```

`*` Optional, non-blocking sleeves. Part 2C activates only when Part 2B reports `bnn_sleeve_recommended: true`.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# PyTorch (CPU-only, faster download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Run the full pipeline

```bash
python run_daily_forecast.py
```

### 3. Run with Transformer backbone instead of LSTM

```bash
python run_daily_forecast.py --model transformer
```

### 4. Force model retrain

```bash
python run_daily_forecast.py --retrain
```
### 5. Run Part 2 directly

```bash
# Train/retrain and append live prediction
python part2_deep_learning_forecaster.py --model lstm --mode train

# Load existing model and append live prediction only
python part2_deep_learning_forecaster.py --model lstm --mode predict
```


---

## 📁 Project Structure

```
LA_Temp_Forecast/
├── part0_data_infrastructure.py    # Data fetching and caching
├── part1_feature_builder.py        # Feature engineering pipeline
├── part2_deep_learning_forecaster.py  # LSTM / Transformer model
├── part2a_atmospheric_alpha.py     # Advanced atmospheric features
├── part2b_xgb_ensemble.py          # XGBoost baseline (optional)
├── part2c_bnn_sleeve.py            # Bayesian uncertainty (optional)
├── part3_forecast_governance.py    # Governance and quality gates
├── part6_weather_regime_engine.py  # HMM regime detection
├── part9_live_attribution.py       # Accuracy metrics and backfill
├── run_daily_forecast.py           # Master daily runner
├── requirements.txt
├── README.md
└── .gitignore
```

**Generated at runtime** (not committed to git):
```
artifacts_part0/   historical_daily.parquet, nws_official_forecast.json, ...
artifacts_part1/   feature_matrix.parquet, train_val_test_split.json
artifacts_part2/   lstm_model.pt, prediction_log.csv, ...
artifacts_part2a/  alpha_features.parquet
artifacts_part2b/  xgb_h1.pkl, xgb_h3.pkl, xgb_h5.pkl, part2b_summary.json
artifacts_part2c/  bnn_predictions.parquet, calibration_report.json
artifacts_part3/   governance_report.json, governance_history.parquet
artifacts_part6/   regime_tape.parquet, regime_model.pkl
artifacts_part9/   live_attribution_report.json, attribution_tape.parquet
```

---

## 🌡️ Target Variable

**`temp_high_f`** — daily maximum temperature in °F at grid point for Los Angeles (LAX, 33.94°N, 118.41°W), sourced from Open-Meteo using the ERA5-Land reanalysis + ECMWF forecast blend.

---

## 📡 Data Sources

| Source | What it provides | Key endpoint |
|--------|-----------------|--------------|
| **Open-Meteo Archive** | Historical daily obs 2018–present | `archive-api.open-meteo.com/v1/archive` |
| **Open-Meteo Forecast** | 7-day gridded forecast | `api.open-meteo.com/v1/forecast` |
| **NWS API** | Official NWS 7-day text forecast (benchmark) | `api.weather.gov/gridpoints/LOX/155,49/forecast` |
| **NWS KLAX Obs** | Recent KLAX hourly observations | `api.weather.gov/stations/KLAX/observations` |
| **NOAA ENSO** | Monthly Niño 3.4 anomaly index | `psl.noaa.gov/data/correlation/nina34.data` |

All sources are **free, open, and require no API key.**

---

## 🔬 Model Details

### Part 2 — LSTM Forecaster
- **Architecture**: 2-layer stacked LSTM → BatchNorm → 3 independent forecast heads (H=1, H=3, H=5)
- **Input**: 14-day look-back window of ~80+ engineered features
- **Training**: Adam + ReduceLROnPlateau scheduler, early stopping (patience=20), MSE loss weighted by horizon
- **Typical performance**: H=1 MAE ≈ 2–4°F (vs NWS H=1 ≈ 2–3°F)

### Part 6 — Weather Regime Engine
Three canonical Southern California regimes detected by Gaussian HMM:

| Regime | Temp | Humidity | Wind | When |
|--------|------|----------|------|------|
| **MARINE_LAYER** | Cool (65–75°F) | High (80–95%) | Light W | Jun–Sep mornings |
| **DRY_CLEAR** | Warm (75–90°F) | Moderate (40–60%) | Variable | Spring/Fall |
| **SANTA_ANA** | Hot (90–110°F+) | Very low (<25%) | Strong NE/E | Oct–Mar |

### Part 2C — BNN Uncertainty
Monte Carlo Dropout with N=200 samples produces calibrated 90% confidence intervals. Good calibration means ~90% of observed temperature highs fall within the CI bounds.

---

## 📊 Governance Modes

| Mode | Meaning |
|------|---------|
| `NORMAL` | All checks pass — forecast is authoritative |
| `CAUTION` | Minor flags (stale data, wide CI) — forecast published with warnings |
| `HOLD` | Critical failure — forecast withheld, investigate immediately |

---

## 🔄 Part Activation Sequence

For first-time setup, all parts run in sequence. For daily operation:
- **Part 0** always runs (incremental data update)
- **Part 6** re-runs weekly or when triggered
- **Part 1** runs after Part 0 or Part 2A
- **Part 2** re-trains automatically when model age > 7 days
- **Part 2B** always runs (fast, non-blocking)
- **Part 2C** runs only if `bnn_sleeve_recommended: true` in Part 2B summary
- **Part 3** and **Part 9** always run

---

## 🤝 Compared to PriceCall

| PriceCall | LA Temp Forecast | Notes |
|-----------|-----------------|-------|
| Part 7 Portfolio Construction | — | Dropped (no allocation needed) |
| Part 8 Execution Model | — | Dropped (no trades) |
| Part 10 Trading Bot | — | Dropped (no orders) |
| FRED / yfinance data | Open-Meteo + NWS | Different data sources |
| VOO/IEF price predictions | Temp high H=1/3/5 | Different target variable |
| Regime: risk-on/off | Regime: marine/dry/santa_ana | Same HMM approach |
| Alpha: financial factors | Alpha: pressure, ENSO, wind | Different physics |
| BNN confidence | BNN temperature CI | Same MC Dropout method |

---

## 📝 Environment Variable

Set `LATEMP_ROOT` to override the project directory:

```bash
export LATEMP_ROOT=/path/to/your/project
python run_daily_forecast.py
```

---

## 📄 License

MIT
