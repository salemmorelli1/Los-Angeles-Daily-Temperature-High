# October 27, 2026 Actions deployment audit

Audited main: `50af2ea5d6851dab8f5752e87bb7586dffc80934`.
Audited tree: `e10329d0f6519c0a80a88cf9ef6230b502fdf351`.

## Confirmed failure

[Pages run 40392364965](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/40392364965)
failed on October 26 at 20:05 UTC (4:05 p.m. Eastern). Its **build succeeded**:
the locked publication environment, dashboard generation, verification and
artifact upload all passed. The deploy job failed during creation, before
status polling, with HTTP 400:

> due to in progress deployment

GitHub identified build `2d8f9f2c0be59e5b0ce521017431cc8013c8f7f8` as the
conflicting deployment. [Run 40392203998](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/40392203998)
had reported that build successful at 20:04:00 UTC. The later creation request
was rejected at 20:05:04 UTC. These logs establish a backend deployment conflict;
they do **not** establish whether cancellation, delayed backend cleanup, or
another deployment of that same build caused it.

The latest [forecast run 40387417025](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/40387417025)
and [backfill run 40392276217](https://github.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/actions/runs/40392276217)
succeeded. This observed failure is not a weather-model exception or an exchange
closure. The baseline's 56 regression/integration tests pass; the retained
artifact validator reports **19 PASS, 0 WARN, 0 FAIL**.

## Corrections

1. Production Pages runs serialize without cancelling an active publisher.
   PR validation uses a separate concurrency key, and receives only read
   permission. Pages-write and OIDC-write permissions belong to the main-only
   deploy job. The default pending-run replacement behavior still coalesces
   pending publication requests; the running publication is not interrupted.
2. The build resolves current main once for production events. PRs still use
   their merge commit. The build exports its actual source commit and upload
   artifact ID. Deployment checks out that exact source commit, uses that exact
   artifact ID, and rejects a mismatched code/build version. This also avoids
   using a writer's pre-artifact event SHA as the published build version.
3. `deploy_pages.py` uses the documented GitHub Pages create/status/cancel
   endpoints and the runner's OIDC identity, as `actions/deploy-pages` does.
   Creation retries only HTTP 400/409 messages containing the observed busy
   response, for up to three minutes with ten-second waits. Identity refreshes
   on each attempt. Other creation errors, including permissions, invalid
   artifacts and ambiguous network failures, fail immediately.
4. A created deployment is polled for up to ten minutes. Only `succeed` passes.
   Terminal deployment errors fail immediately. Transient 404/429/server status
   reads have a separate ten-error cap. On interruption, polling timeout or
   another nonterminal failure, cleanup attempts to cancel only the deployment
   this invocation created. A conflicting earlier deployment is never cancelled.
5. HTTP redirects are refused rather than forwarding bearer credentials. API
   errors omit server bodies; OIDC values are registered with Actions masking.
   The CLI refuses local, PR, non-main and untrusted upstream contexts.
6. The Pages uploader is pinned to a reviewed commit. Hidden files are included
   so the generated `.nojekyll` remains present in the verified publication.
7. `verify_audit.sh` is the single verification entry point for Windows Git Bash
   and Linux CI. It uses fresh, isolated CPU-model and publication environments,
   all regression tests, retained-artifact validation, and an offline Pages
   build/verification. It requires a clean source tree and checks it afterwards.

## Verification scope

The additional tests replay the actual busy response, permanent-busy timeout,
permission and artifact failures, pending-to-success status transitions,
backend failures, transient status registration, status error budgets,
interruption cleanup, refreshed OIDC identities, exact artifact/version binding,
and the main/PR privilege boundary. A local HTTP server exercises real urllib
request bodies, authentication headers, HTTP errors, redirect refusal, malformed
responses, size limits and the empty cancellation response. No test publishes
a site or uses real credentials.

The existing real LSTM/HMM/XGBoost/BNN tests and artifact contracts remain in
the full suite. The patch does not change forecasts, outcomes, model artifacts,
training data, cohorts, gates, calibration thresholds or the dashboard builder.

GitHub verification runs the same delivered script on Ubuntu and Windows. PR
Pages validation builds and uploads the site, with deployment skipped. A real
production deployment of this correction requires review and merge; simulated
conflict recovery and successful PR checks are not evidence of a production
deployment. Exact tested head/tree and CI links are recorded in the PR and the
delivery response.

## References

- [GitHub Actions concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
- [GitHub Pages deployment API](https://docs.github.com/en/rest/pages/pages#create-a-github-pages-deployment).
- [Reviewed deploy-pages v5 implementation](https://github.com/actions/deploy-pages/blob/368f82528645a54fb793d4d04e342629a3f51346/src/internal/deployment.js).
- [Reviewed Pages upload action](https://github.com/actions/upload-pages-artifact/blob/fc324d3547104276b827a68afc52ff2a11cc49c9/action.yml).
