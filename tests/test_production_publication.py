"""Publication and method-cohort regressions for the October 18 audit."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import build_dashboard as dashboard
import part2b_xgb_ensemble as ensemble
import part9_live_attribution as attribution
from forecast_protocol import TRAINING_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]


def ledger(legacy=40, current=6):
    rows = []
    for i, day in enumerate(pd.date_range("2025-01-01", periods=legacy + current)):
        row = {
            "feature_date": day, "decision_date": day + pd.Timedelta(days=1),
            "training_protocol_version": "legacy" if i < legacy else TRAINING_PROTOCOL_VERSION,
            "publish_mode": "CAUTION", "forecast_finalized_at": "2025-03-01T15:00:00+00:00",
            "forecast_source": "persistence", "model": "LSTM",
        }
        for h in (1, 3, 5):
            row.update({
                f"target_date_h{h}": day + pd.Timedelta(days=h),
                f"forecast_h{h}": 71.0, f"forecast_candidate_h{h}": 70.0,
                f"forecast_pre_anchor_h{h}": 71.0, f"persistence_h{h}": 71.0,
                f"realized_h{h}": 72.0, f"nws_h{h}": 73.0,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def run_attribution(frame, root):
    with contextlib.ExitStack() as stack:
        for name, value in {
            "PROJECT_DIR": root, "PART2_DIR": root / "artifacts_part2",
            "PART3_DIR": root / "artifacts_part3", "ARTIFACTS_DIR": root / "artifacts_part9",
        }.items():
            stack.enter_context(patch.object(attribution, name, value))
        (root / "artifacts_part9").mkdir(exist_ok=True)
        stack.enter_context(patch.object(attribution, "load_prediction_log", return_value=frame.copy()))
        stack.enter_context(patch.object(attribution, "load_historical", return_value=pd.DataFrame()))
        stack.enter_context(patch.object(attribution, "load_nws_forecast", return_value={}))
        stack.enter_context(patch.object(attribution, "fetch_realized_temps", return_value=pd.DataFrame()))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        result = attribution.main()
    report = json.loads((root / "artifacts_part9/live_attribution_report.json").read_text())
    return result, report


class AttributionPublicationTests(unittest.TestCase):
    def test_current_method_does_not_borrow_legacy_sample_size(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report = run_attribution(ledger(), Path(directory))
        self.assertEqual(result, 0)
        self.assertEqual(report["realized_samples_by_horizon"], {"h1": 6, "h3": 6, "h5": 6})
        self.assertFalse(report["summary"]["skill_trustworthy"])
        self.assertEqual(report["summary"]["skill_status"], "INSUFFICIENT_DATA")
        self.assertEqual(report["historical_diagnostics"]["n_prediction_rows"], 46)
        self.assertEqual(report["cohort"]["training_protocol_version"], TRAINING_PROTOCOL_VERSION)

    def test_nonfinite_values_do_not_count_as_realized_pairs(self):
        frame = ledger(legacy=0, current=32)
        frame.loc[0, "realized_h1"] = np.inf
        frame.loc[1, "forecast_h1"] = -np.inf
        metrics = attribution.compute_metrics(frame, pd.DataFrame())["h1"]
        self.assertEqual(metrics["n_samples"], 30)
        self.assertEqual(metrics["n_paired_persistence"], 30)

    def test_report_promotion_minimum_matches_actual_gate(self):
        self.assertEqual(attribution.PROMOTE_MIN_SAMPLES, ensemble.LIVE_SKILL_MIN_REALIZED_SAMPLES)

    def test_model_candidate_is_distinct_from_governed_persistence(self):
        frame = ledger(legacy=0, current=6)
        metrics = attribution.compute_model_only_metrics(frame)["h1"]
        self.assertEqual(metrics["pred_col_used"], "forecast_candidate_h1")
        self.assertEqual(metrics["mae_f"], 2.0)

    def test_hold_rows_do_not_count_as_published_evidence(self):
        frame = ledger()
        frame.loc[frame.index[-1], "publish_mode"] = "HOLD"
        report = attribution.build_report_core(frame, pd.DataFrame(), {})
        self.assertEqual(report["cohort"]["n_rows"], 5)

    def test_many_unpaired_outcomes_do_not_establish_skill(self):
        frame = ledger(legacy=0, current=35)
        frame.loc[:32, "persistence_h1"] = np.nan
        report = attribution.build_report_core(frame, pd.DataFrame(), {})
        self.assertEqual(report["metrics_by_horizon"]["h1"]["n_paired_persistence"], 2)
        self.assertFalse(report["summary"]["skill_trustworthy"])

    def test_report_is_idempotent_and_does_not_relabel_rows(self):
        frame = ledger()
        before = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            _, first = run_attribution(frame, Path(directory))
            _, second = run_attribution(frame, Path(directory))
        self.assertEqual(first, second)
        pd.testing.assert_frame_equal(frame, before)


class DashboardSnapshotTests(unittest.TestCase):
    def test_build_uses_current_cohort_without_fetches_or_source_writes(self):
        sources = [p for p in ROOT.glob("artifacts_part*/*") if p.is_file()]
        before = {p: dashboard.file_sha256(p) for p in sources}
        with tempfile.TemporaryDirectory() as directory, patch("requests.get", side_effect=AssertionError("network forbidden")):
            output = Path(directory) / "site"
            manifest = dashboard.build_dashboard(ROOT, output)
            self.assertEqual(dashboard.verify_dashboard(output), manifest)
            self.assertEqual((output / "artifacts_part2/prediction_log.csv").read_bytes(),
                             (ROOT / "artifacts_part2/prediction_log.csv").read_bytes())
            report = json.loads((output / "artifacts_part9/live_attribution_report.json").read_text())
            frame = pd.read_csv(ROOT / "artifacts_part2/prediction_log.csv")
            current = frame.loc[frame["training_protocol_version"].eq(TRAINING_PROTOCOL_VERSION)]
            self.assertEqual(report["cohort"]["n_rows"], len(current))
            self.assertEqual(report["historical_diagnostics"]["n_prediction_rows"], len(frame))
        self.assertEqual({p: dashboard.file_sha256(p) for p in sources}, before)

    def test_modified_and_unlisted_publication_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            dashboard.build_dashboard(ROOT, output)
            extra = output / "unexpected.txt"
            extra.write_text("unlisted")
            with self.assertRaisesRegex(ValueError, "file set"):
                dashboard.verify_dashboard(output)
            extra.unlink()
            (output / "artifacts_part2/prediction_log.csv").write_text("rewritten")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                dashboard.verify_dashboard(output)

    def test_mixed_governance_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            shutil.copyfile(ROOT / "index.html", source / "index.html")
            for folder in ROOT.glob("artifacts_part*"):
                if folder.is_dir():
                    shutil.copytree(folder, source / folder.name)
            report = source / "artifacts_part3/governance_report.json"
            data = json.loads(report.read_text())
            data["decision_date"] = "2000-01-01"
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "publication disagree"):
                dashboard.build_dashboard(source, Path(directory) / "site")

    def test_output_cannot_overwrite_the_checkout(self):
        with self.assertRaisesRegex(ValueError, "source checkout"):
            dashboard.build_dashboard(ROOT, ROOT)
        with self.assertRaisesRegex(ValueError, "overlaps source"):
            dashboard.build_dashboard(ROOT, ROOT / "artifacts_part2/new-output")

    def test_browser_loader_uses_deployed_snapshot_and_preview_uses_repository(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is required to execute the browser loader contract")
        text = (ROOT / "index.html").read_text()
        function = text[text.index("function artifactBase()") : text.index("const rawBase =")]
        for host, href, expected in [
            ("salemmorelli1.github.io", "https://salemmorelli1.github.io/Los-Angeles-Daily-Temperature-High/index.html", "https://salemmorelli1.github.io/Los-Angeles-Daily-Temperature-High"),
            ("localhost", "http://localhost:8000/", "https://raw.githubusercontent.com/salemmorelli1/Los-Angeles-Daily-Temperature-High/main"),
        ]:
            program = "const CFG={owner:'salemmorelli1',repo:'Los-Angeles-Daily-Temperature-High',branch:'main'};"
            program += f"const location={json.dumps({'hostname':host,'href':href})};" + function
            program += "console.log(artifactBase());"
            result = subprocess.run([node, "-e", program], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), expected)
        script = re.findall(r"<script>([\s\S]*?)</script>", text)[-1]
        subprocess.run([node, "--check"], input=script, text=True, capture_output=True, check=True)


class WorkflowPublicationTests(unittest.TestCase):
    def test_runtime_bootstrap_is_isolated_from_runner_packages(self):
        bootstrap = None
        for name in ("ci.yml", "daily-production.yml", "daily-backfill.yml", "pages.yml"):
            workflow = yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
            job = next(iter(workflow["jobs"].values()))
            steps = job["steps"]
            position = next(i for i, step in enumerate(steps) if step["name"] == "Create isolated Python environment")
            install = next(i for i, step in enumerate(steps) if step["name"].startswith("Install"))
            self.assertLess(position, install)
            bootstrap = steps[position]["run"]
        program = bootstrap.split("python - <<'PYENV'\n", 1)[1].rsplit("PYENV", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_output = root / "github-path.txt"
            env = {**os.environ, "GITHUB_WORKSPACE": str(root), "GITHUB_PATH": str(path_output)}
            subprocess.run([sys.executable, "-c", program], env=env, check=True, capture_output=True, text=True)
            binaries = Path(path_output.read_text(encoding="utf-8").strip())
            interpreter = binaries / ("python.exe" if os.name == "nt" else "python")
            subprocess.run([str(interpreter), "-c", "import sys,site,importlib.util; assert sys.prefix != sys.base_prefix; assert not site.ENABLE_USER_SITE; assert importlib.util.find_spec('torch') is None"], check=True, capture_output=True, text=True)

    def test_pages_follows_successful_main_production_and_backfill(self):
        text = (ROOT / ".github/workflows/pages.yml").read_text()
        self.assertIn("workflow_run:", text)
        self.assertIn("Daily LA Temperature Forecast", text)
        self.assertIn("Daily Realized Temperature Backfill", text)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", text)
        self.assertIn("github.event.workflow_run.head_branch == 'main'", text)

    def test_writers_do_not_silently_resolve_concurrent_artifact_conflicts(self):
        for name in ("daily-production.yml", "daily-backfill.yml"):
            text = (ROOT / ".github/workflows" / name).read_text()
            self.assertNotIn("--strategy-option=ours", text, name)
            self.assertIn("git merge --ff-only", text, name)

    def test_every_workflow_parses(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            self.assertIsInstance(yaml.load(path.read_text(), Loader=yaml.BaseLoader), dict)

    def test_actual_writer_scripts_reject_concurrent_updates(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            candidates = [Path("C:/Program Files/Git/bin/bash.exe")]
            git = shutil.which("git")
            if git:
                candidates.insert(0, Path(git).resolve().parent.parent / "bin/bash.exe")
            bash = next((str(p) for p in candidates if p.is_file()), None)
        if bash is None:
            self.skipTest("Git Bash required to execute publication scripts")
        for name in ("daily-production.yml", "daily-backfill.yml"):
            workflow = yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
            job = next(iter(workflow["jobs"].values()))
            publish = next(step["run"] for step in job["steps"] if step["name"].startswith("Commit and push"))
            sync = next(step["run"] for step in job["steps"] if step["name"].startswith("Synchronize"))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = os.environ.copy()
                env.update({"GITHUB_REF_NAME": "main", "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"})

                def git(*args, cwd=root):
                    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()

                git("init", "--bare", str(root / "origin.git"))
                git("init", "-b", "main", str(root / "seed"))
                seed = root / "seed"
                (seed / "source.txt").write_text("v1")
                git("add", ".", cwd=seed)
                git("commit", "-m", "initial", cwd=seed)
                git("remote", "add", "origin", str(root / "origin.git"), cwd=seed)
                git("push", "-u", "origin", "main", cwd=seed)
                git("clone", "--branch", "main", str(root / "origin.git"), str(root / "writer"))
                writer = root / "writer"
                (seed / "source.txt").write_text("v2")
                git("commit", "-am", "concurrent source change", cwd=seed)
                git("push", "origin", "main", cwd=seed)
                remote = git("rev-parse", "HEAD", cwd=seed)
                failed = subprocess.run([bash, "-e", "-c", publish], cwd=writer, env=env, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0, name)
                self.assertEqual(git("ls-remote", "origin", "refs/heads/main", cwd=writer).split()[0], remote)
                subprocess.run([bash, "-e", "-c", sync], cwd=writer, env=env, capture_output=True, text=True, check=True)
                self.assertEqual(git("rev-parse", "HEAD", cwd=writer), remote)
                self.assertEqual((writer / "source.txt").read_text(), "v2")


if __name__ == "__main__":
    unittest.main()
