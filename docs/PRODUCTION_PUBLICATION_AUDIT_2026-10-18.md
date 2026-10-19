# October 18, 2026 production-publication audit

Baseline main: `84654499312a820d3cdb508df97a9d2bfb375307`.
Baseline tree: `8f3a2c2f518197a1f43a0f50088b2d817f8ca74b`.

## What actually ran

[Forecast run 39146749769](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/39146749769)
completed the full pipeline on October 18. It reused the October 16 LSTM and
frozen preprocessing, fitted the optional XGBoost sleeve, evaluated BNN bands,
published `CAUTION`, and passed all 19 retained artifact checks. The matched
06:35 PDT cron started at 09:49 PDT; this audit does not establish the cause of
GitHub's delay. The selector correctly used the triggering cron expression.

[Run 39149722752](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/39149722752)
was the opposing-DST occurrence and intentionally skipped. Its green status
does not mean a second model fit occurred.

[Backfill 39149730765](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/39149730765)
also succeeded and committed baseline main. This is not an exchange-market
closure problem. Weather production operates daily, including weekends.

The latest forecast is **89.4°F at all three horizons**, with source
`persistence`. That is the stored October 17 completed-day Open-Meteo grid-point
high, not an invented model result. The established learned-model gate remains
closed while its prospective evidence accumulates. Governance warns about
21.4°F and 22.4°F deviations from the H=3/H=5 NWS benchmarks and suppresses BNN
bands that lack their required calibration evidence. This patch does not force
learned forecasts, NWS anchoring, or uncertainty bands to pass.

## Reproduced defects

1. **Pages did not follow bot publication.** The latest successful Pages push
   run was [35802016417](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/35802016417),
   September 23. Direct HTTP reads of the deployed governance JSON and ledger
   on October 18 returned **September 22**. The writers push using
   `GITHUB_TOKEN`, whose pushes do not trigger a Pages workflow. Removing
   `[skip ci]` from a commit message does not change that rule. See
   [GitHub's token documentation](https://docs.github.com/en/actions/concepts/security/github_token).
   The existing page also fetches raw files from moving `main`; therefore stale
   deployed files alone do **not** prove that every browser-visible value was
   stale. They establish a missing deployment handoff and an inconsistent
   publication design.
2. **Attribution mixed methods.** The committed report declares
   `skill_trustworthy=true` using 101/99/97 realized rows. Only seven ledger rows
   belong to the published corrected method, with **6/4/2** realized canonical
   forecasts. All 95 older rows must remain historical diagnostics, not evidence
   for this method. A six-row corrected-method fixture mixed with 40 older rows
   reproduced the erroneous report count of 46 and the wrong skill status.
3. **Report telemetry disagreed with the gate.** Part 9 advertised a 60-sample
   promotion minimum while the actual per-horizon gate uses 30 finite paired
   forecasts plus its other criteria. The reporting value did not control the
   gate. A separate fixture counted two infinite values as realized samples.
   Model-only reporting also preferred post-gate pre-anchor output over an
   available frozen paper candidate; a regression distinguishes that candidate
   from the governed persistence forecast.
4. **Artifact conflicts could be silently resolved.** Both writers selectively
   restored files and later used `git pull --strategy-option=ours ... || true`.
   This can mix code/model snapshots or choose a stale ledger over concurrent
   evidence. No lost update is asserted for the audited October 18 snapshot.
5. **The current baseline test fixture failed.** The 39-test baseline produced
   38 passes and one error. Its synthetic HOLD test patched temporary artifact
   paths but left real training-lineage checks pointed at the now-qualified
   production bundle. The test then attempted to load missing temporary model
   metadata. The fixture now explicitly isolates calendar and lineage checks;
   the separate real model/lineage integration tests remain active.
6. **GitHub's Windows runner had a global dependency conflict.** The first PR
   checks passed Linux and Pages, but Windows failed `pip check`: preinstalled
   `pipx 1.17.12` requires `filelock>=4.0.9`, conflicting with the project's
   tested `filelock==3.32.3`. Every workflow now creates an isolated virtual
   environment before installing its locked stack. A regression executes the
   actual bootstrap and confirms runner/model packages are not inherited.

Six new regressions failed before correction: mixed-cohort reporting, finite
counts, promotion telemetry, paper candidate selection, missing Pages handoff,
and unsafe writer conflict handling.

## Corrections and verification

- Pages follows successful completion of either trusted main-branch writer.
  Failed or non-main completions cannot deploy. A workflow completion's original
  `head_sha` predates its artifact push, so Pages resolves current main once and
  builds all files from that checkout. PRs build and verify without deploying.
- `build_dashboard.py` validates checkpoint/preprocessing hashes, canonical and
  governed ledger agreement, and the latest governance decision/source/mode.
  It recomputes attribution **offline from frozen ledger outcomes**, never from
  newly fetched observations. Only the generated site report is refreshed.
  Source ledger, outcomes, model artifacts and committed report remain unchanged.
- The deployed page reads same-origin snapshot files. `publication.json` records
  the source commit, dirty-checkout status, builder fingerprints, cohort and file
  hashes. Missing, changed or unlisted publication files fail verification.
  The existing appearance and historical forecast table are retained.
- Part 9 reports published current-method metrics and rolling history. All-row
  attribution remains explicitly labeled historical diagnostics in the report;
  all ledger rows remain in the attribution tape. Skill sample counts require
  finite paired values. No statistical superiority follows from sample size
  alone, and retrospective all-history climatology is labeled diagnostic.
- One shared **30-pair** constant removes the false 60-sample telemetry. It does
  not relax the learned-model CI, XGBoost validation, NWS anchoring, calibration,
  or lineage requirements. New report timestamps include their UTC offset.
- Writers fast-forward the entire checkout before dependency installation and
  data use. Before committing, they require the remote tip to still equal their
  base. Any concurrent update fails rather than overwriting it; normal
  non-forced push also rejects a subsequent race.
- Pages and backfill share a pinned lightweight publication environment.
  CI verifies it without torch, sklearn, hmmlearn or XGBoost installed, separately
  from the full CPU model environment.

Local verification: **56 tests passed**. Coverage includes actual production
shell scripts against temporary bare Git remotes, concurrent-update rejection,
full-checkout synchronization, finite counts, method isolation, HOLD exclusion,
unpaired outcomes, idempotent reporting, source-data preservation, snapshot
tamper detection, JavaScript loader execution and syntax, and the existing real
LSTM/HMM/XGBoost/BNN integration. Both environments pass `pip check`; compilation,
fatal-error lint and all workflow YAML parse successfully.

The untouched retained report now produces one explicit validator warning:
**18 PASS, 1 WARN, 0 FAIL**. It is historical mixed-method telemetry awaiting
refresh. The separately generated snapshot report uses the corrected schema,
seven current-method rows and 6/4/2 canonical samples, with
`skill_status=INSUFFICIENT_DATA`. All **102** original forecasts and outcomes
are preserved in this source-only patch.

## Activation and limits

After review and merge, the source change triggers a verified Pages build.
Subsequent successful forecast and backfill runs trigger further snapshot
deployments automatically. A new manual Pages run on main can also build the
current snapshot. No same-day forecast rewrite or forced retrain is needed.

This audit verifies implementation, publication consistency and honest evidence
reporting. It does not establish improved temperature accuracy. The target is
still the Open-Meteo completed-day grid-point maximum near LAX, not a newly
substituted station measurement. Historical reanalysis/ENSO source-vintage
limitations remain. Prospective candidates must continue accumulating their
unchanged required evidence before approval.
