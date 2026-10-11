# Forecast provenance audit — October 11, 2026

Repository: `salemmorelli1/Los-Angeles-Daily-Temperature-High`.
Audited main: `f08b023a7559296fb8dd24af6314d9b0a8b62887`.
Original tree: `7678d146895ff071e030e0295eb43d2b09cf6999`.

## Findings and corrections

| Finding | Evidence at the audited commit | Correction |
| --- | --- | --- |
| Regime fitting crossed the forecast training boundary. | `regime_meta.json` fitted through **2024-02-21**. Part 1's effective training end was **2024-02-13**, and validation began **2024-02-19**. Eight dates after the purged boundary, including three validation dates, entered the regime fit. | Derive the regime fit cutoff from Part 1's calendar labels and actual purged splits. Fit its imputer, scaler and HMM/KMeans only within that cutoff. |
| Daily preprocessing changed beneath a reused LSTM. | The October 11 regime fit was new, while the LSTM was trained October 6. Checking feature names did not detect changes to their fitted meaning. | Train the regime and forecaster together; reuse the exact regime bundle during predict-only days. Hash-check the bundle, checkpoint, scalers and saved validation table. |
| Invalid errors could authorize forecasts. | A reproducible 40-row fixture containing only **two finite pairs** passed every live skill gate. The sample count included infinities that the bootstrap subsequently removed. | Count and score the same finite pairs in persistence, learned-model and NWS gates. Reject nonfinite interval centers and invalid historical bounds. |
| Held-out roles were not enforced when reusing a checkpoint. | The LSTM table contains both validation and test predictions, but its loader did not filter `split`. BNN reevaluated the day's changing percentage splits. The current October 11 blend join contained 461 validation rows and **zero** test rows: test contamination was a reproducible latent defect, not an observed occurrence in that snapshot. | Load only explicit saved validation rows for blend selection. Use the checkpoint's saved boundaries for BNN calibration, including its fixed test endpoint. |
| Forecast freshness was inferred from other artifacts. | Fresh Part 0 metadata did not establish that the log row's decision, feature, target and issue dates matched the current forecast. BNN could fall back to the last unrelated log row. | Add critical calendar and model-lineage governance checks. Match BNN to the exact feature date and checkpoint. Unmatched metadata suppresses BNN availability. |
| Reruns could amend already published predictions and baselines. | Part 2/2B/2C upserts overwrote published values. Attribution recalculated recorded persistence using later historical files. | Preserve published forecast values, issue times, uncertainty fields and recorded baselines. A same-day runner invocation refreshes realized outcomes only. Today's incomplete temperature cannot become a realized daily high. |
| The legacy XGBoost wheel fails Python 3.12 environment verification. | The installed 1.7.6 wheel declares `cp310-cp310` tags; its C++ wrapper runs, but `pip check` exits 1 on this Python 3.12 environment. | Replace it with maintained `xgboost-cpu==3.4.1`, verify real early-stopping fit/predict, and pin the production stack to the same CPU environment tested by CI. |

## What the workflow evidence establishes

The complete forecast run [38163671660](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/38163671660) executed the full pipeline successfully. Its log shows **predict-only** LSTM inference, XGBoost training, BNN calibration, governance and attribution. Attribution's external archive request timed out on that run; the separate [backfill run 38166643660](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/38166643660) succeeded afterward. A successful process is not a claim of statistical skill or complete fresh attribution.

Run [38166576431](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/38166576431) was the opposing daylight-saving cron occurrence. Its selector intentionally skipped the pipeline. Its short duration does not demonstrate failed training.

The repository schedules forecasts at **06:35 Pacific**, using UTC cron `35 13 * * *` during PDT and `35 14 * * *` during PST. The matched October 11 run began at approximately **11:24 PDT**, well after that scheduled time. This audit does not establish the cause of GitHub's scheduling delay. The selector checks the triggering cron, so a delayed matching run is not incorrectly discarded. Long elapsed time is not a forecast-quality criterion.

The current published forecast uses **persistence** at all three horizons. The existing paired-skill confidence intervals do not establish learned-model superiority. NWS anchoring also lacks the required paired improvement. Those decisions remain valid protections; this change does not force a gate to pass.

## Method transition and retained evidence

New fitted bundles and first-issued rows carry `purged-frozen-regime-v3`, exact model/preprocessing fingerprints, UTC issue time, and a publication finalization time. Forecast targets remain **feature date + 1, 3, or 5 calendar days**. The feature date for a new production decision must be yesterday's complete Pacific day.

All 95 existing prediction rows, all realized values, and every committed model/artifact remain unchanged in this PR. They remain available for historical and exploratory attribution. They are not retroactively certified as issued under a method introduced by this change. Production learned-model and NWS-anchoring gates select published forecasts of the current method; `HOLD` rows do not qualify. Scheduled parameter refits retain this protocol identity; the count does not restart at each weekly refit.

The thresholds remain unchanged: **30 paired realized forecasts**, at most 90 recent pairs, and a learned-model loss-difference bootstrap upper bound below zero; XGBoost validation improvement of at least **0.20°F**; NWS paired improvement of at least **0.25°F**; and the existing BNN calibration/coverage floors. There is **no new 60-outcome rule**. Until the changed method establishes its own paired evidence, canonical forecasts use the existing baseline fallback while its first-issued model candidates are recorded for evaluation.

## Activation

On the first **new decision day** after merging, the runner detects the retained legacy checkpoint and retrains the regime and forecaster together automatically. `--skip-train` refuses an unqualified or mismatched bundle. An explicit `--retrain` also refits both stages. A rerun of an already published day preserves that day's original forecast and updates outcomes only.

`validate_artifacts.py` reports **18 PASS, 1 WARN, 0 FAIL** on the retained original artifacts. The warning explicitly identifies the old checkpoint as requiring a protocol retrain. It is not v3 approval. Missing or mismatched provenance in a newly marked v3 artifact is a validation failure, and production governance treats invalid model lineage as critical.

The production artifact restore now restores the regime bundle and saved validation table with the neural model and scalers. Hash checks reject mixed bundles rather than silently combining artifacts from different training runs.

## Verification and limits

The regression suite covers calendar gaps, train/validation purging, finite paired counts, method qualification, held forecasts, saved validation roles, stale interval attachment, invalid interval centers, immutable forecasts, preserved baselines and complete-day outcomes. CPU integration actually fits a Gaussian HMM and the KMeans fallback, trains/reloads a small LSTM, advances the data by one day, runs MC-dropout calibration and rejects a replaced scaler. Test data and model artifacts are temporary and synthetic; no forecast is issued to the repository by these tests.

The CI workflow runs the same pinned Python 3.12 CPU stack on Linux and Windows, including model integration, compilation, fatal lint checks, artifact validation and a check that verification did not rewrite committed forecasts. PyTorch CPU installation is pinned separately to `2.14.1`; the remaining 40 packages are pinned in `requirements-ci-lock.txt`.

XGBoost's [official installation documentation](https://github.com/dmlc/xgboost/blob/master/doc/install.rst) provides the CPU package, which imports as `xgboost` without CUDA algorithms. When upgrading an existing virtual environment, uninstall the old `xgboost` distribution before installing `xgboost-cpu`; both supply the same import namespace. The tested CPU package passes `pip check` and the real fitting regression.

These tests establish implementation and artifact consistency, **not improved forecast accuracy**. This audit does not run or publish a production forecast on `main`, modify thresholds to obtain approval, or replace the old artifacts with synthetic results. The first corrected production retrain and subsequent realized outcomes are still required.

Source-vintage limitations also remain: historical weather reanalysis and the latest NOAA ENSO history are not archived point-in-time release histories. A publication lag alone does not prove that revised past values were available on their original dates. The historical test split has already been inspected during development. Its metrics are useful diagnostics, but should not be presented as an untouched confirmatory experiment. The recorded prospective forecasts and unchanged publication gates are the appropriate evidence for this method's live skill.
