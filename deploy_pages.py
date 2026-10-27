#!/usr/bin/env python3
"""Deploy a verified Actions artifact; retry only GitHub's busy-Pages response.

Uses the same create/status/cancel REST endpoints and Actions OIDC identity as
actions/deploy-pages. No model runtime, personal token, or local deployment is
required. All retries and polling are bounded; only `succeed` is success.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API = "https://api.github.com"
LIMIT = 1_048_576
FINAL_ERRORS = {
    "deployment_failed", "deployment_content_failed", "deployment_cancelled",
    "deployment_lost",
}


class APIError(RuntimeError):
    def __init__(self, status: int, message: str = ""):
        super().__init__(f"GitHub request failed (HTTP {status})")
        self.status = status
        # Inspect error text for the known conflict without logging response
        # bodies, request headers, bearer credentials, or the OIDC payload.
        self.busy = status in (400, 409) and "due to in progress deployment" in message.lower()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(url: str, token: str, payload: dict | None = None) -> dict:
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "latemp-pages-deployment",
    })
    try:
        with build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(LIMIT + 1)
    except HTTPError as error:
        with error:
            raw = error.read(LIMIT + 1)
        try:
            body = json.loads(raw) if len(raw) <= LIMIT else {}
        except (ValueError, UnicodeError):
            body = {}
        message = body.get("message", "") if isinstance(body, dict) else ""
        raise APIError(error.code, message if isinstance(message, str) else "") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("GitHub request could not complete; credentials were not logged") from None
    if len(raw) > LIMIT:
        raise RuntimeError("GitHub response exceeds the size limit")
    try:
        value = json.loads(raw) if raw else {}
    except (ValueError, UnicodeError):
        raise RuntimeError("GitHub returned invalid JSON") from None
    if not isinstance(value, dict):
        raise RuntimeError("GitHub returned an invalid response object")
    return value


class PagesAPI:
    def __init__(self, environment: dict):
        self.token = environment["GITHUB_TOKEN"]
        self.identity_token = environment["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
        self.identity_url = environment["ACTIONS_ID_TOKEN_REQUEST_URL"]
        parsed = urlsplit(self.identity_url)
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or not (parsed.hostname or "").endswith(".actions.githubusercontent.com")):
            raise ValueError("Unsupported Actions identity endpoint")
        repository = environment["GITHUB_REPOSITORY"]
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository)
                or repository.split("/")[-1] in {".", ".."}):
            raise ValueError("Invalid GitHub repository")
        self.endpoint = f"{API}/repos/{repository}/pages/deployments"

    def create(self, artifact_id: int, version: str) -> dict:
        # Refresh identity on every create attempt, including after waiting.
        identity = request_json(self.identity_url, self.identity_token).get("value")
        if not isinstance(identity, str) or not identity or any(c in identity for c in "\r\n"):
            raise RuntimeError("Actions identity response lacks a valid token")
        print(f"::add-mask::{identity}", flush=True)
        return request_json(self.endpoint, self.token, {
            "artifact_id": artifact_id, "pages_build_version": version, "oidc_token": identity,
        })

    def status(self, deployment_id: str) -> dict:
        return request_json(f"{self.endpoint}/{quote(deployment_id, safe='')}", self.token)

    def cancel(self, deployment_id: str) -> None:
        request_json(f"{self.endpoint}/{quote(deployment_id, safe='')}/cancel", self.token, {})


def deploy(api, artifact_id: int, version: str, *, busy_timeout=180, poll_timeout=600,
           interval=10, clock=time.monotonic, sleep=time.sleep, emit=print) -> dict:
    if type(artifact_id) is not int or artifact_id <= 0:
        raise ValueError("A positive uploaded artifact ID is required")
    if not re.fullmatch(r"[0-9a-f]{40}", version):
        raise ValueError("Build version must be the source commit SHA")
    if not (0 < busy_timeout <= 180 and 0 < poll_timeout <= 600 and 0 < interval <= 30):
        raise ValueError("Invalid deployment time limits")
    deadline = clock() + busy_timeout
    while True:
        try:
            created = api.create(artifact_id, version)
            break
        except APIError as error:
            if not error.busy:
                raise
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("Pages remained busy; no other deployment was cancelled") from None
            emit("Pages is busy; waiting before retrying the same uploaded artifact.")
            sleep(min(interval, remaining))

    deployment_id = str(created.get("id") or version)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", deployment_id):
        raise RuntimeError("GitHub returned an invalid deployment ID")
    terminal = False
    deadline = clock() + poll_timeout
    read_errors = 0
    try:
        while True:
            # Only status reads can retry transient HTTP server failures.
            # A failed create is never replayed after an ambiguous network error.
            try:
                status = api.status(deployment_id).get("status")
                if not isinstance(status, str) or not status:
                    raise RuntimeError("GitHub returned an invalid deployment status")
            except APIError as error:
                read_errors += 1
                if (error.status not in (404, 429) and error.status < 500) or read_errors >= 10:
                    raise
                status = None
            if status == "succeed":
                terminal = True
                emit("Pages deployment succeeded.")
                return created
            if status in FINAL_ERRORS:
                terminal = True
                raise RuntimeError(f"Pages deployment failed: {status}")
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for Pages deployment success")
            sleep(min(interval, remaining))
    finally:
        if not terminal:
            # Clean up only the deployment this invocation successfully created.
            # A preceding busy deployment is never cancelled or taken over.
            try:
                api.cancel(deployment_id)
            except Exception:
                emit("Could not cancel this run's deployment; check Pages status before retrying.")


def validate_context(environment: dict, version: str) -> None:
    if (environment.get("GITHUB_ACTIONS") != "true"
            or environment.get("GITHUB_REF") != "refs/heads/main"
            or environment.get("GITHUB_EVENT_NAME") not in {"push", "workflow_run", "workflow_dispatch"}):
        raise ValueError("Deployment is restricted to trusted main-branch Actions runs")
    if environment["GITHUB_EVENT_NAME"] == "workflow_run":
        event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        run = event.get("workflow_run", {})
        if (run.get("conclusion") != "success" or run.get("head_branch") != "main"
                or (run.get("head_repository") or {}).get("full_name") != environment.get("GITHUB_REPOSITORY")):
            raise ValueError("Untrusted upstream workflow completion")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    if actual.stdout.strip() != version:
        raise ValueError("Deployment code and built source commit disagree")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--build-version", required=True)
    args = parser.parse_args(argv)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("Deployment interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        validate_context(os.environ, args.build_version)
        result = deploy(PagesAPI(os.environ), args.artifact_id, args.build_version)
        page_url = result.get("page_url", "")
        if isinstance(page_url, str) and page_url and not any(c in page_url for c in "\r\n"):
            if "://" not in page_url:
                page_url = "https://" + page_url
            if urlsplit(page_url).scheme == "https":
                with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                    stream.write(f"page_url={page_url}\n")
        return 0
    except (RuntimeError, ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        # APIError deliberately omits server bodies and request credentials.
        print(f"Pages deployment FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
