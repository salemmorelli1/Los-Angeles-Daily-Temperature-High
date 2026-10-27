"""Replay the observed Pages conflict and exercise the HTTP/Actions boundary."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml

import deploy_pages as pages

ROOT = Path(__file__).resolve().parents[1]
SHA = "50af2ea5d6851dab8f5752e87bb7586dffc80934"
BUSY = ("Deployment request failed for " + SHA + " due to in progress deployment. "
        "Please cancel 2d8f9f2c0be59e5b0ce521017431cc8013c8f7f8 first or wait for it to complete.")


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeAPI:
    def __init__(self, creates=None, statuses=None):
        self.creates = list(creates or [{"id": SHA, "page_url": "https://example.invalid/"}])
        self.statuses = list(statuses or [{"status": "succeed"}])
        self.created = []
        self.polled = []
        self.cancelled = []

    def create(self, artifact, version):
        self.created.append((artifact, version))
        item = self.creates.pop(0) if len(self.creates) > 1 else self.creates[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def status(self, deployment):
        self.polled.append(deployment)
        item = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def cancel(self, deployment):
        self.cancelled.append(deployment)


def run(api, clock=None, **kwargs):
    clock = clock or Clock()
    return pages.deploy(api, 12686210352, SHA, clock=clock, sleep=clock.sleep,
                        emit=lambda message: None, **kwargs)


class DeploymentTests(unittest.TestCase):
    def test_observed_400_conflict_waits_then_deploys_the_same_artifact(self):
        clock = Clock()
        api = FakeAPI([pages.APIError(400, BUSY), {"id": SHA}])
        result = run(api, clock)
        self.assertEqual(result["id"], SHA)
        self.assertEqual(api.created, [(12686210352, SHA)] * 2)
        self.assertEqual(clock.sleeps, [10])
        self.assertEqual(api.cancelled, [])

    def test_409_busy_conflict_also_recovers(self):
        api = FakeAPI([pages.APIError(409, BUSY), {"id": SHA}])
        run(api)
        self.assertEqual(len(api.created), 2)

    def test_persistent_busy_response_fails_without_cancelling_another_run(self):
        clock = Clock()
        api = FakeAPI([pages.APIError(400, BUSY)])
        with self.assertRaisesRegex(TimeoutError, "remained busy"):
            run(api, clock, busy_timeout=25)
        self.assertEqual(clock.now, 25)
        self.assertEqual(api.cancelled, [])
        self.assertEqual(api.polled, [])

    def test_permissions_invalid_artifact_and_unrelated_errors_are_not_retried(self):
        for code, message in [(400, "invalid artifact"), (403, BUSY), (404, "disabled"),
                              (409, "unrelated conflict"), (422, BUSY), (500, BUSY)]:
            api = FakeAPI([pages.APIError(code, message)])
            with self.subTest(code=code), self.assertRaises(pages.APIError):
                run(api)
            self.assertEqual(len(api.created), 1)
            self.assertEqual(api.cancelled, [])

    def test_ambiguous_create_network_failure_is_not_replayed(self):
        api = FakeAPI([RuntimeError("request interrupted")])
        with self.assertRaises(RuntimeError):
            run(api)
        self.assertEqual(len(api.created), 1)

    def test_success_requires_backend_succeed_after_pending_statuses(self):
        clock = Clock()
        api = FakeAPI(statuses=[{"status": "queued"}, {"status": "deployment_attempt_error"},
                                {"status": "succeed"}])
        run(api, clock)
        self.assertEqual(api.polled, [SHA] * 3)
        self.assertEqual(clock.sleeps, [10, 10])

    def test_terminal_backend_failures_cannot_be_reported_as_success(self):
        for status in pages.FINAL_ERRORS:
            api = FakeAPI(statuses=[{"status": status}])
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "failed"):
                run(api)
            self.assertEqual(api.cancelled, [])

    def test_pending_timeout_cancels_only_this_runs_created_deployment(self):
        api = FakeAPI(creates=[{"id": "this-run"}], statuses=[{"status": "queued"}])
        with self.assertRaisesRegex(TimeoutError, "waiting"):
            run(api, poll_timeout=25)
        self.assertEqual(api.cancelled, ["this-run"])

    def test_server_status_errors_can_recover_without_recreating_the_deployment(self):
        api = FakeAPI(statuses=[pages.APIError(503), pages.APIError(429), {"status": "succeed"}])
        run(api)
        self.assertEqual(len(api.created), 1)
        self.assertEqual(len(api.polled), 3)

    def test_forbidden_status_does_not_poll_forever_and_cleans_up_own_deployment(self):
        api = FakeAPI(statuses=[pages.APIError(403)])
        with self.assertRaises(pages.APIError):
            run(api)
        self.assertEqual(len(api.polled), 1)
        self.assertEqual(api.cancelled, [SHA])

    def test_status_can_be_temporarily_unregistered_after_creation(self):
        api = FakeAPI(statuses=[pages.APIError(404), {"status": "succeed"}])
        run(api)
        self.assertEqual(len(api.created), 1)
        self.assertEqual(api.cancelled, [])

    def test_status_read_error_budget_fails_and_cancels_own_deployment(self):
        api = FakeAPI(statuses=[pages.APIError(503)])
        with self.assertRaises(pages.APIError):
            run(api)
        self.assertEqual(len(api.polled), 10)
        self.assertEqual(api.cancelled, [SHA])

    def test_malformed_status_fails_and_cleans_up(self):
        api = FakeAPI(statuses=[{"status": ["succeed"]}])
        with self.assertRaisesRegex(RuntimeError, "invalid deployment status"):
            run(api)
        self.assertEqual(api.cancelled, [SHA])

    def test_interruption_cleans_up_the_deployment_it_created(self):
        api = FakeAPI(statuses=[InterruptedError("cancelled")])
        with self.assertRaises(InterruptedError):
            run(api)
        self.assertEqual(api.cancelled, [SHA])

    def test_invalid_artifact_version_or_budget_cannot_start_a_deployment(self):
        api = FakeAPI()
        for artifact, version, kwargs in [(True, SHA, {}), (0, SHA, {}), (3, "main", {}),
                                           (3, SHA, {"busy_timeout": 1000})]:
            with self.assertRaises(ValueError):
                pages.deploy(api, artifact, version, **kwargs)
        self.assertEqual(api.created, [])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        requests, responses = self.requests, self.responses

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def respond(self):
                requests.append({"path": self.path, "method": self.command,
                                 "authorization": self.headers.get("Authorization"),
                                 "body": self.rfile.read(int(self.headers.get("Content-Length", 0)))})
                code, body, headers = responses.pop(0)
                self.send_response(code)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def test_actual_http_request_preserves_json_and_bearer_header(self):
        self.responses.append((200, b'{"id":"test"}', {}))
        result = pages.request_json(self.url, "fixture-token", {"artifact_id": 123})
        self.assertEqual(result, {"id": "test"})
        self.assertEqual(self.requests[0]["authorization"], "Bearer fixture-token")
        self.assertEqual(json.loads(self.requests[0]["body"]), {"artifact_id": 123})
        self.assertEqual(self.requests[0]["method"], "POST")

    def test_actual_http_busy_error_is_recognized_without_echoing_response(self):
        self.responses.append((400, json.dumps({"message": BUSY + " SECRET"}).encode(), {}))
        with self.assertRaises(pages.APIError) as caught:
            pages.request_json(self.url, "fixture-token", {})
        self.assertTrue(caught.exception.busy)
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertNotIn("fixture-token", str(caught.exception))

    def test_redirect_is_not_followed_with_credentials(self):
        self.responses.append((302, b"", {"Location": self.url + "/redirected"}))
        with self.assertRaises(pages.APIError):
            pages.request_json(self.url, "fixture-token")
        self.assertEqual(len(self.requests), 1)

    def test_malformed_and_oversized_responses_fail(self):
        for body in (b"not JSON", b"[]", b"x" * (pages.LIMIT + 1)):
            self.responses.append((200, body, {}))
            with self.assertRaises(RuntimeError):
                pages.request_json(self.url, "fixture-token")

    def test_empty_cancel_response_is_supported(self):
        self.responses.append((204, b"", {}))
        self.assertEqual(pages.request_json(self.url, "fixture-token", {}), {})


class IdentityAndWorkflowTests(unittest.TestCase):
    def environment(self, **updates):
        return {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main",
                "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_TOKEN": "fixture-api-token", "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "fixture-oidc-token",
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc.actions.githubusercontent.com/request", **updates}

    def test_create_refreshes_oidc_and_uses_the_exact_artifact_and_source(self):
        api = pages.PagesAPI(self.environment())
        output = io.StringIO()
        replies = [{"value": "identity-one"}, {"id": SHA}, {"value": "identity-two"}, {"id": SHA}]
        with patch.object(pages, "request_json", side_effect=replies) as request, contextlib.redirect_stdout(output):
            api.create(123, SHA)
            api.create(123, SHA)
        for i, identity in [(1, "identity-one"), (3, "identity-two")]:
            args = request.call_args_list[i].args
            self.assertEqual(args[0], "https://api.github.com/repos/owner/repo/pages/deployments")
            self.assertEqual(args[2], {"artifact_id": 123, "pages_build_version": SHA, "oidc_token": identity})
        self.assertEqual(output.getvalue().splitlines(), ["::add-mask::identity-one", "::add-mask::identity-two"])

    def test_identity_endpoint_and_repository_cannot_redirect_credentials(self):
        for updates in [{"ACTIONS_ID_TOKEN_REQUEST_URL": "https://example.invalid/"},
                        {"ACTIONS_ID_TOKEN_REQUEST_URL": "http://oidc.actions.githubusercontent.com/request"},
                        {"GITHUB_REPOSITORY": "../repo"}]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                pages.PagesAPI(self.environment(**updates))

    def test_pr_local_and_non_main_contexts_cannot_deploy(self):
        for updates in [{"GITHUB_EVENT_NAME": "pull_request"}, {"GITHUB_REF": "refs/heads/feature"},
                        {"GITHUB_ACTIONS": "false"}]:
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "trusted main"):
                pages.validate_context(self.environment(**updates), SHA)

    def test_deploy_source_must_match_the_build_commit(self):
        with patch.object(pages.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, SHA + "\n")):
            pages.validate_context(self.environment(), SHA)
            with self.assertRaisesRegex(ValueError, "disagree"):
                pages.validate_context(self.environment(), "a" * 40)

    def test_untrusted_workflow_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.json"
            for updates in [{"conclusion": "failure"}, {"head_branch": "feature"},
                            {"head_repository": {"full_name": "other/repo"}}]:
                event = {"workflow_run": {"conclusion": "success", "head_branch": "main",
                                         "head_repository": {"full_name": "owner/repo"}, **updates}}
                path.write_text(json.dumps(event), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Untrusted"):
                    pages.validate_context(self.environment(GITHUB_EVENT_NAME="workflow_run",
                                                            GITHUB_EVENT_PATH=str(path)), SHA)

    def test_cli_reports_failure_and_leaves_outputs_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            api = FakeAPI([pages.APIError(403, "SECRET")])
            with patch.dict(os.environ, self.environment(GITHUB_OUTPUT=str(output))), \
                    patch.object(pages, "validate_context"), patch.object(pages, "PagesAPI", return_value=api), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(pages.main(["--artifact-id", "123", "--build-version", SHA]), 1)
            self.assertFalse(output.exists())
            self.assertIn("HTTP 403", stderr.getvalue())
            self.assertNotIn("SECRET", stderr.getvalue())

    def test_cli_outputs_url_only_after_confirmed_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict(os.environ, self.environment(GITHUB_OUTPUT=str(output))), \
                    patch.object(pages, "validate_context"), patch.object(pages, "PagesAPI", return_value=FakeAPI()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pages.main(["--artifact-id", "123", "--build-version", SHA]), 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "page_url=https://example.invalid/\n")

    def test_workflow_serializes_publication_and_keeps_pr_permissions_read_only(self):
        workflow = yaml.load((ROOT / ".github/workflows/pages.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(workflow["concurrency"]["cancel-in-progress"], "false")
        self.assertIn("github.event.pull_request.number", workflow["concurrency"]["group"])
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        build, deploy = workflow["jobs"]["build"], workflow["jobs"]["deploy"]
        self.assertIn("github.event_name != 'pull_request'", deploy["if"])
        self.assertIn("refs/heads/main", deploy["if"])
        self.assertEqual(deploy["permissions"]["pages"], "write")
        self.assertEqual(deploy["permissions"]["id-token"], "write")
        self.assertIn("steps.upload.outputs.artifact_id", build["outputs"]["artifact_id"])
        upload = next(step for step in build["steps"] if step.get("id") == "upload")
        self.assertEqual(upload["with"]["include-hidden-files"], "true")
        checkout = deploy["steps"][0]
        self.assertEqual(checkout["with"]["ref"], "${{ needs.build.outputs.source_commit }}")
        publication = deploy["steps"][-1]
        self.assertIn('"$PAGES_ARTIFACT_ID"', publication["run"])
        self.assertIn('"$PAGES_SOURCE_COMMIT"', publication["run"])
        self.assertNotIn("continue-on-error", publication)


if __name__ == "__main__":
    unittest.main()
