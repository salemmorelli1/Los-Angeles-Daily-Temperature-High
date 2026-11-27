# Weather acquisition and publication audit — November 27, 2026

The missed forecast was an upstream acquisition failure with inadequate retry
handling. It was not a market holiday or a governance rejection. A successful
backfill or Pages deployment did not repair the missing forecast.

## Evidence inspected

The audited main commit is `292891ce5085aef63a2a3d51e1778361196c17cd`.

| Evidence | Observed result |
| --- | --- |
| [Production run 45909398329](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/45909398329), November 26, 23:11 UTC | Part 0's one-day archive request for November 25 returned HTTP 429 on all three attempts. Production exited before generating or committing new forecasts. |
| [Backfill run 45909539723](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/45909539723) | Succeeded and updated outcomes. It did not issue a November 26 forecast. |
| [Pages run 45909592210](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/45909592210) | Build and deployment succeeded. Downloaded artifact 15159706856 verifies all 14 manifest entries and records the November 25 decision. |
| Pages artifact ZIP SHA-256 | `786442893dc5de34492e1b319fd4142e1b51f52335e97ab8835da1d6b4334824` |
| Opposing-DST schedule run 45905448453 | Green, but all forecast steps were skipped by the intended schedule selector. Green is not evidence of a fresh forecast. |
| Clean baseline verification | 107 tests ran: one failure and one error in dashboard tests. They assumed November 4 was the current target and an August gap remained within the last 70 plotted rows. |

The failed archive request was already incremental, not an eight-year download.
The logs establish rate limiting; they do not establish which provider quota
was exceeded or whether a shared runner IP contributed. Open-Meteo documents
[free-service usage limits](https://open-meteo.com/en/terms). Retrying cannot
guarantee recovery before a quota resets.

## Corrections

1. `weather_http.py` supplies the same retry policy to Part 0 and Part 9.
   Defaults permit four requests and at most 180 seconds of cumulative waits;
   individual request timeouts are separate. It retries 408, 429, 500, 502, 503,
   504 and transient connection/timeouts. Rate-limit fallback waits are 30, 60,
   and 60 seconds. Other transient waits start at five seconds. Permanent
   client errors and certificate errors fail immediately.
2. `Retry-After` supports both whole seconds and HTTP dates as specified by
   [RFC 9110 §10.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3).
   If a requested wait exceeds the remaining budget, acquisition fails rather
   than retrying early. Failed responses close before sleeping or raising.
3. Part 9 skips acquisition when all completed-day outcomes are already
   recorded. It first fills blanks from the identified Part 0 daily-high cache
   with the same coordinates, units and Pacific calendar. The known cache
   schema and coordinates must match, and dates must lie within its metadata
   range. Unidentified, foreign or conflicting caches are ignored. The remote
   request spans only remaining missing target dates. On this checkout at the
   November 27 Pacific date, that is November 26 alone, rather than the retained
   ledger's May-to-November range.
4. Finite recorded outcomes, forecasts and issue-time baselines remain frozen.
   Today's incomplete day and future days cannot become realized outcomes.
   An exhausted required Part 0 request still raises; no saved artifact is
   refreshed and production cannot continue on invented inputs. Part 9 retains
   its nonfatal, missing-outcome behavior when its optional fetch is unavailable.
5. The page shows **Verified archived snapshot** and an issuance-date notice
   when the latest decision predates the current Pacific date. This applies to
   all horizon selections, including future H5 targets. It describes the
   historical governance mode separately. Decision-time paired counts now
   appear for mixed persistence/blend sources as well as pure persistence.
6. Dashboard tests derive their viewing clock and expectations from the
   verified snapshot. Explicit fixtures exercise interval permissions and
   calendar gaps independently of changing production data and window length.

## Gates at the audited decision

Artifact integrity is **PASS** (19 checks), while publication is **CAUTION**
(9 of 11 governance checks). These answer different questions.

| Horizon | Canonical forecast | Decision-time learned-model evidence | Interval display |
| --- | --- | --- | --- |
| H1, November 25 | 69.7°F persistence | 43 pairs; skill not established | Allowed |
| H3, November 27 | 69.7°F persistence | 40 pairs; skill not established | Withheld: recent live undercoverage |
| H5, November 29 | 67.70596008300781°F blend | 38 pairs; paired-loss CI below zero | Withheld: recent live undercoverage |

H1/H3 candidate-minus-persistence loss confidence intervals still include zero;
their sample counts exceeding 30 do not prove skill. H5's interval excludes
zero in the required direction. The NWS sanity warning and aggregate BNN
warning explain CAUTION. No validation threshold, model weight, checkpoint,
forecast, outcome or promotion decision is changed by this correction.

The target is the Open-Meteo gridded daily maximum near LAX, not an assertion
that it equals the KLAX station thermometer. See the provider's
[historical-data documentation](https://open-meteo.com/en/docs/historical-weather-api).
Changing the outcome source would require a separately versioned evaluation.

## Validation and activation

The corrected CPU suite has 124 tests, including a real local HTTP 429/200
exchange, persistent outage preservation, request-budget limits, cache identity,
frozen finite outcomes, incomplete-day exclusion and the delivered dashboard's
JavaScript/Plotly trace construction. The delivered `verify_audit.sh` additionally
checks dependency consistency, fatal lint/compilation, 19 artifact contracts,
the read-only diagnostics and a separate publication environment with no model
runtime. The PR records Linux/Windows Actions and preview artifact verification.

The feature branch supplies code and tests without regenerating retained
artifacts. After review and merge, run **Daily LA Temperature Forecast** on
`main`, with `model=lstm` and `retrain=false` (the existing age/schema rules
still decide whether retraining is due). Verify the pipeline step actually ran,
the decision/feature/target dates advanced and artifact validation passed. Its
successful completion triggers Pages automatically. If 429 persists, allow the
provider budget to recover before another run. Do not run Pages alone as a
forecast repair, force a gate open or backdate an unissued forecast.
