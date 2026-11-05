# Forecast dashboard and evidence audit — November 4, 2026

## Scope and frozen source

Audited main commit `5eb52b78b1f5d9d5b464a170a672ecd1531ba8ab`, its
production artifacts, recent Actions logs, and the actual successful Pages
artifact. The user's screenshots show this same November 4 decision, with
November 3 features and 119 retained ledger rows.

This correction changes publication diagnostics and browser behavior, and moves
Part 9's output-directory creation from module import to its explicit writer.
Importing the new read-only audit must not create artifact directories. It does
not change the learned models, model/scaler/regime files, training protocol,
30-pair threshold, paired-loss confidence interval, NWS approval policy,
calibration thresholds, forecasts, or realized outcomes. The published
`purged-frozen-regime-v3` cohort and first-issued values remain intact.

## What the screenshot means

The three 71.8°F values are the recorded persistence fallback, not three
identical neural predictions. Part 2B's reason is
`H1:live_skill_gate_persistence | H3:live_skill_gate_persistence | H5:live_skill_gate_persistence`.
The last observed high is 71.8°F, and the learned candidates have not yet met
the required paired evidence floor.

| Horizon | Actual target | Published forecast | Pairs available at decision | Realized canonical outcomes after backfill |
| --- | --- | --- | --- | --- |
| H=1 | November 4 | 71.8°F | 22 / 30 | 23 |
| H=3 | November 6 | 71.8°F | 19 / 30 | 21 |
| H=5 | November 8 | 71.8°F | 17 / 30 | 19 |

Decision-time model approval and later attribution are separate quantities.
Backfill occurs after the decision; a canonical realized outcome can also lack
a complete paired model candidate. The corrected telemetry preserves both
counts. Thirty samples alone still do not approve a model: the existing
validation and paired-loss confidence-interval requirements also apply.

`CAUTION` is supported by two recorded warnings (9 of 11 checks pass):

* `NWS_SANITY`: H=5 persistence is 71.8°F versus the frozen NWS 82.0°F,
  a 10.2°F difference, exceeding the 8°F strong-warning tier.
* `BNN_CALIBRATION_GATE`: the aggregate interval display flag is false.
  Per-horizon evidence allows H=1 but suppresses H=3 and H=5 for recent live
  undercoverage. The old aggregate message inaccurately implies every horizon
  is suppressed. New diagnostics explain the individual flags while retaining
  the original warning and publication mode.

The failed half-split calibration diagnostic is informational under the
existing policy; it is not reclassified to force normal publication. Display
bands centered on a governed forecast remain display/risk bands, not a new
claim of strict 90% predictive coverage. Current-method canonical live MAEs
are 3.970°F, 7.829°F, and 9.484°F on 23, 21, and 19 outcomes respectively.
The audit does not interpret that small sample as established model skill.

## Reproduced defects and corrections

1. **Incorrect relative target label.** The page called H=1 "Tomorrow" even
   though November 3 features plus one day target November 4, the decision
   day. The selector now uses H=1/H=3/H=5; the summary compares the actual
   target with today's Los Angeles date. It also identifies past targets on
   an old open page and refreshes that label every minute.
2. **Wrong chart calendar and invented visual continuity.** All horizons used
   `decision_date` for their x values, including realized H=3/H=5 temperatures
   from different days. The line also bridged August 26 to September 14
   decision dates: 19 days between records, with 18 missing dates. Each horizon
   now uses its own target date. Explicit null separators break both forecast
   and residual lines across missing dates and method changes. No historical
   forecasts are fabricated. Current-method history is the default; the latest
   70 retained mixed-method rows are available as a separate view.
3. **Unapproved browser substitution.** Fault injection showed that a missing
   canonical value could become an 82.5°F blend value even on a HOLD row.
   With a corrupted/missing prediction log, the old full page inserted five
   demonstration rows and displayed 82.5°F. The corrected page displays only
   explicit published canonical values. Unavailable data leaves the forecast
   blank and the mode UNKNOWN, with an explanation.
4. **Unverified concurrent artifact loading.** A valid build-time hash manifest
   did not guarantee the files loaded independently by an already open browser
   came from one deployment. The loader now checks SHA-256 and byte length for
   every requested artifact, reconciles canonical values/dates/governance with
   generated diagnostics, and rereads the manifest to detect a deployment
   change during loading. It discards a failed or mixed snapshot. Local previews
   use their own bundle instead of fetching moving `main`.
5. **Unexplained boolean/blank status.** `Governance: false` meant two WARN
   checks failed, not that all publication was blocked. The page now shows
   9/11 checks with CAUTION, displays the recorded fallback and interval reasons,
   and labels withheld live skill as Pending with its realized sample counts.

`audit_forecast.py` provides the same read-only diagnostics used by
`build_dashboard.py`. The builder writes `forecast_diagnostics.json` into its
staged publication and hashes it and the new browser contract module in the
manifest. It rejects missing canonical forecasts, contradictory target dates,
duplicate ledger dates, and inconsistent ensemble/governance data. Uncertainty
display additionally requires the same feature date/checkpoint, matching
bounds, an available current BNN sleeve, and explicit per-horizon approval.

## Actions: service conflict versus data/model behavior

* [Production run 41910731819](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/41910731819)
  completed successfully. Its logs show the intended persistence fallback,
  both WARN checks, and 19 PASS / 0 WARN / 0 FAIL artifact checks. The archive
  read timed out during Part 9; missing observations were preserved for retry.
* [Pages run 41911086410](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/41911086410)
  built and uploaded successfully, then exhausted the existing 180-second busy
  retry limit: `Pages remained busy; no other deployment was cancelled`.
* [Subsequent Pages run 41911160089](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/41911160089)
  successfully deployed the later backfill commit `5eb52b7`. Its actual uploaded
  `github-pages` artifact 13363505181 was downloaded and inspected: all 12
  manifest-listed file hashes and lengths match, including `.nojekyll`.

Thus the isolated red Pages run is an exhausted backend-busy retry, followed
by a successful publication. It is separate from the forecast fallback and
chart defects. The bounded deployer, its failure reporting, and its rule
against cancelling another deployment are preserved. This PR validates a
preview; it does not deploy an unreviewed feature branch.

## Verification and reproduction

Baseline: 86 tests passed, all 19 artifact checks passed, isolated publication
build/verification passed. The original page's label, canonical substitution,
demo fallback, and gap behavior were reproduced separately because the existing
tests did not cover them.

Corrected suite: 107 tests passed locally, including execution of the delivered
JavaScript module and inline page with a small DOM and captured Plotly traces.
Tests cover target-day alignment, Pacific midnight, gaps, the current cohort,
individual uncertainty flags, missing manifests, corrupted bytes, publication
changes during loading, frozen decision-time counts, source immutability,
duplicate/missing/conflicting forecast evidence, and the existing model/runtime
and deployment integration tests. The DOM harness does not measure pixel layout.
All 19 artifact checks still pass. The publication-only environment builds and
verifies all 14 listed files without importing the model runtime.

With Python 3.12 activated on a clean checkout, `bash verify_audit.sh` installs
the locked CPU environment, runs the complete suite and artifact validator,
executes the read-only audit, and builds/verifies an isolated Pages preview.
Node.js is needed to execute the JavaScript tests (both CI runner operating
systems provide it); local runs without Node explicitly report skipped tests.
The preview is printed below `.venv/audit-check.*/site`. To view that exact
bundle, serve it with `python -m http.server 8000 --directory PATH_TO_SITE` and
open `http://localhost:8000/`.

Expected: `integrity_status: PASS` alongside `publish_mode: CAUTION`, with the
existing persistence forecasts. Correcting the display does not generate new
prospective evidence or force the scientific gates open. Linux and Windows
GitHub CI run the same delivered verification entry point; PR Pages builds are
verified but their deployment job is skipped.
