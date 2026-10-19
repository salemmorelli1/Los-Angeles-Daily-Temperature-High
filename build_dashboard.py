#!/usr/bin/env python3
"""Build a verified Pages snapshot from one checkout, without fetching outcomes."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from forecast_protocol import file_sha256, forecast_is_published, training_lineage_errors
from part9_live_attribution import build_report_core, _report_with_stable_generated_at

ROOT = Path(__file__).resolve().parent
SCHEMA = "LA_PUBLICATION_V1"
ARTIFACTS = (
    "artifacts_part2/prediction_log.csv", "artifacts_part2/part2_meta.json",
    "artifacts_part2b/part2b_summary.json", "artifacts_part2c/part2c_meta.json",
    "artifacts_part2c/calibration_report.json", "artifacts_part3/governance_report.json",
    "artifacts_part6/regime_meta.json",
)


def verify_dashboard(output: Path) -> dict:
    manifest = json.loads((output / "publication.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported dashboard publication schema")
    files = manifest.get("files", {})
    present = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if not files or set(files) != present - {"publication.json"}:
        raise ValueError("Dashboard publication file set mismatch")
    for relative, record in files.items():
        path = output / relative
        if file_sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Dashboard publication hash mismatch: {relative}")
    return manifest


def build_dashboard(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Dashboard output must not replace the source checkout")
    if output.is_relative_to(root) and output.relative_to(root).parts[0] in {
        "report", "assets", "tests", ".git", *[f"artifacts_part{p}" for p in (0, 1, 2, "2a", "2b", "2c", 3, 6, 9)],
    }:
        raise ValueError("Dashboard output overlaps source inputs")
    if output.exists():
        raise ValueError("Dashboard output already exists; choose a new output directory")
    paths = [root / name for name in ARTIFACTS]
    paths += [root / "index.html", root / "artifacts_part3/prediction_log.csv",
              root / "artifacts_part0/historical_daily.parquet"]
    hashes = {p: file_sha256(p) for p in paths}
    if hashes[root / "artifacts_part2/prediction_log.csv"] != hashes[root / "artifacts_part3/prediction_log.csv"]:
        raise ValueError("Canonical and governed prediction logs disagree")
    errors = training_lineage_errors(root)
    if errors:
        raise ValueError(f"Dashboard training lineage is invalid: {errors}")
    frame = pd.read_csv(root / "artifacts_part2/prediction_log.csv", float_precision="round_trip")
    if frame.empty:
        raise ValueError("Dashboard prediction log is empty")
    frame = frame.sort_values(["feature_date", "decision_date"], kind="stable").reset_index(drop=True)
    latest = frame.iloc[-1]
    governance = json.loads((root / "artifacts_part3/governance_report.json").read_text(encoding="utf-8"))
    if (not forecast_is_published(latest)
            or str(latest["decision_date"]) != governance.get("decision_date")
            or latest.get("publish_mode") != governance.get("publish_mode")
            or latest.get("forecast_source") != governance.get("forecast_source")):
        raise ValueError("Latest forecast and governance publication disagree")
    history = pd.read_parquet(root / "artifacts_part0/historical_daily.parquet")
    # Current-method NWS values are frozen on each ledger row; never borrow a
    # newly fetched forecast as a historical benchmark in the Pages builder.
    core = build_report_core(frame, history, {})
    report = _report_with_stable_generated_at(core, root / "artifacts_part9/live_attribution_report.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False,
                            capture_output=True, text=True)
    source_commit = commit.stdout.strip() if commit.returncode == 0 else "UNVERSIONED"
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                            check=False, capture_output=True, text=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="la-pages-", dir=output.parent) as directory:
        stage = Path(directory) / "site"
        stage.mkdir()
        shutil.copyfile(root / "index.html", stage / "index.html")
        for name in ("report", "assets"):
            if (root / name).is_dir():
                shutil.copytree(root / name, stage / name)
        if (root / "favicon.ico").is_file():
            shutil.copyfile(root / "favicon.ico", stage / "favicon.ico")
        for relative in ARTIFACTS:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        target = stage / "artifacts_part9/live_attribution_report.json"
        target.parent.mkdir(parents=True)
        target.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        (stage / ".nojekyll").touch()
        manifest = {
            "schema_version": SCHEMA, "source_commit": source_commit,
            "source_modified": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "builder_code_sha256": {name: file_sha256(ROOT / name) for name in (
                "build_dashboard.py", "part9_live_attribution.py", "forecast_protocol.py",
            )},
            "decision_date": str(latest["decision_date"]),
            "attribution_cohort": core["cohort"],
            "files": {p.relative_to(stage).as_posix(): {
                "sha256": file_sha256(p), "size_bytes": p.stat().st_size,
            } for p in sorted(stage.rglob("*")) if p.is_file()},
        }
        (stage / "publication.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        verify_dashboard(stage)
        if any(file_sha256(p) != digest for p, digest in hashes.items()):
            raise ValueError("Source publication changed while building the dashboard")
        stage.replace(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("_site"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = verify_dashboard(args.output) if args.verify else build_dashboard(args.root, args.output)
    print(json.dumps({"status": "PASS", "source_commit": manifest["source_commit"],
                      "decision_date": manifest["decision_date"], "files": len(manifest["files"]),
                      "cohort": manifest["attribution_cohort"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
