"""Execute publication loading and chart behavior on real and corrupted bundles."""
from __future__ import annotations

import copy
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

import audit_forecast as audit
import build_dashboard as dashboard
from forecast_protocol import file_sha256
from part9_live_attribution import build_report_core

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "assets/dashboard_contract.js"


def node_json(program, payload=None):
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("Node is required for executable dashboard contract tests")
    result = subprocess.run(
        [node, "-e", program, str(MODULE)], input=json.dumps(payload),
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


class AuditDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "artifacts_part2/prediction_log.csv", float_precision="round_trip")
        cls.values = {name: json.loads((ROOT / path).read_text()) for name, path in {
            "governance": "artifacts_part3/governance_report.json",
            "ensemble": "artifacts_part2b/part2b_summary.json",
            "uncertainty": "artifacts_part2c/part2c_meta.json",
            "calibration": "artifacts_part2c/calibration_report.json",
        }.items()}
        cls.values["attribution"] = build_report_core(
            cls.frame, pd.read_parquet(ROOT / "artifacts_part0/historical_daily.parquet"), {}
        )

    def diagnostic(self, frame=None, values=None):
        return audit.build_diagnostics(
            self.frame.copy(deep=True) if frame is None else frame,
            **copy.deepcopy(self.values if values is None else values),
        )

    def test_read_only_audit_uses_one_cohort_and_preserves_every_artifact(self):
        paths = [path for path in ROOT.glob("artifacts_part*/*") if path.is_file()]
        before = {path: file_sha256(path) for path in paths}
        with patch("requests.get", side_effect=AssertionError("No network in audit")):
            result = audit.inspect_repository(ROOT)
        self.assertEqual(result["integrity_status"], "PASS")
        self.assertEqual(result["cohort"], self.values["attribution"]["cohort"])
        self.assertEqual(before, {path: file_sha256(path) for path in paths})
        json.dumps(result, allow_nan=False)

    def test_importing_the_read_only_audit_does_not_create_artifact_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused-root"
            env = {**os.environ, "LATEMP_ROOT": str(output), "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run([sys.executable, "-c", "import audit_forecast"], cwd=ROOT,
                           env=env, check=True, capture_output=True, text=True)
            self.assertFalse(output.exists())

    def test_decision_time_pair_counts_are_not_replaced_with_later_backfills(self):
        values = copy.deepcopy(self.values)
        values["ensemble"]["live_model_skill_gate"]["h1"]["n_paired"] = 12
        values["attribution"]["realized_samples_by_horizon"]["h1"] = 23
        result = self.diagnostic(values=values)
        self.assertEqual(result["horizons"]["h1"]["decision_time_model_gate"]["n_paired"], 12)
        self.assertEqual(result["horizons"]["h1"]["realized_samples"], 23)
        self.assertEqual(result["horizons"]["h1"]["decision_time_model_gate"]["min_samples"], 30)

    def test_wrong_target_calendar_is_rejected(self):
        frame = self.frame.copy()
        frame.loc[frame.index[-1], "target_date_h1"] = "2099-01-01"
        with self.assertRaisesRegex(ValueError, "target clock"):
            self.diagnostic(frame)

    def test_missing_canonical_cannot_substitute_a_paper_candidate(self):
        frame = self.frame.copy()
        frame.loc[frame.index[-1], "forecast_h1"] = np.nan
        with self.assertRaisesRegex(ValueError, "canonical forecast missing"):
            self.diagnostic(frame)

    def test_duplicate_forecasts_cannot_inflate_history(self):
        frame = pd.concat([self.frame, self.frame.tail(1)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.diagnostic(frame)

    def test_mismatched_ensemble_canonical_is_rejected(self):
        values = copy.deepcopy(self.values)
        values["ensemble"]["canonical_forecast"]["h1"] += 0.5
        with self.assertRaisesRegex(ValueError, "ensemble disagree"):
            self.diagnostic(values=values)

    def test_per_horizon_display_survives_an_aggregate_warning(self):
        result = self.diagnostic()
        h1 = result["horizons"]["h1"]["interval"]
        self.assertTrue(h1["displayable"])
        self.assertFalse(result["horizons"]["h3"]["interval"]["displayable"])
        warning = next(w for w in result["warnings"] if w["name"] == "BNN_CALIBRATION_GATE")
        self.assertFalse(warning["passed"])
        self.assertIn("H=1 available", warning["display_message"])

    def test_stale_or_mixed_checkpoint_uncertainty_is_not_shown(self):
        for key, field, value in [
            ("uncertainty", "checkpoint_sha256", "other"),
            ("calibration", "checkpoint_sha256", "other"),
            ("uncertainty", "feature_date", "2000-01-01"),
        ]:
            values = copy.deepcopy(self.values)
            values[key][field] = value
            with self.subTest(key=key, field=field):
                self.assertFalse(self.diagnostic(values=values)["horizons"]["h1"]["interval"]["displayable"])

    def test_inconsistent_uncertainty_bounds_are_not_shown(self):
        frame = self.frame.copy()
        frame.loc[frame.index[-1], "bnn_lo90_h1"] += 1.0
        interval = self.diagnostic(frame)["horizons"]["h1"]["interval"]
        self.assertFalse(interval["displayable"])
        self.assertIsNone(interval["lo_f"])

    def test_stack_without_bnn_cannot_display_leftover_uncertainty(self):
        values = copy.deepcopy(self.values)
        values["governance"]["bnn_available"] = False
        self.assertFalse(self.diagnostic(values=values)["horizons"]["h1"]["interval"]["displayable"])

    def test_boolean_and_nonfinite_numbers_are_not_forecasts(self):
        for value in (True, np.bool_(True), np.nan, np.inf, "NaN", None):
            self.assertIsNone(audit.finite(value))

    def test_the_observed_nineteen_day_gap_is_reported_without_inventing_rows(self):
        result = self.diagnostic()
        gap = next(g for g in result["history_gaps"] if g["previous_feature_date"] == "2026-08-25")
        self.assertEqual(gap, {"previous_feature_date": "2026-08-25",
                               "next_feature_date": "2026-09-13", "missing_days": 18})


class BrowserPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.site = Path(cls.directory.name) / "site"
        dashboard.build_dashboard(ROOT, cls.site)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def page(self, fault="", retained=False, horizon=1):
        # Run the delivered inline page, external contract, loader and Plotly
        # trace construction. A small DOM records outputs; no browser packages
        # or network requests are needed in the publication-only runtime.
        program = r"""
const fs=require('fs'), vm=require('vm'), path=require('path');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements=new Map(), traces={};
const element=id=>{
  if(!elements.has(id)) elements.set(id,{textContent:'',innerHTML:'',value:'',style:{setProperty(){}},classList:{add(){},remove(){},toggle(){}},addEventListener(){},querySelector(name){return element(id+'/'+name);}});
  return elements.get(id);
};
const document={getElementById:element,querySelectorAll:()=>[],documentElement:element('html')};
let manifestReads=0;
const fetcher=async url=>{
  const file=new URL(url).pathname.replace(/^\/salemmorelli1\/Los-Angeles-Daily-Temperature-High\/main\//,'/').slice(1);
  if(input.fault==='missing' && file==='publication.json') return {ok:false,status:404};
  let bytes=fs.readFileSync(path.join(input.site,file));
  if(input.fault==='corrupt' && file.endsWith('prediction_log.csv')) bytes=Buffer.from('corrupt');
  if(file==='publication.json' && ++manifestReads>1 && input.fault==='race') bytes=Buffer.from(bytes.toString().replace('"source_commit": "','"source_commit": "changed-'));
  return {ok:true,text:async()=>bytes.toString(),arrayBuffer:async()=>Uint8Array.from(bytes).buffer};
};
const RealDate=Date;
class FixedDate extends RealDate {constructor(...args){super(...(args.length?args:['2026-11-04T22:00:00Z']));}static now(){return new RealDate('2026-11-04T22:00:00Z').getTime();}}
const context={document,location:{hostname:'localhost',pathname:'/',href:'http://localhost/'},URL,Intl,Date:FixedDate,
  TextDecoder,Uint8Array,crypto:require('crypto').webcrypto,fetch:fetcher,console,setTimeout(){},clearTimeout(){},setInterval(){},addEventListener(){},
  Plotly:{react(id,series,layout){traces[id]={series,layout};}}};
context.window=context; vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const html=fs.readFileSync(path.join(input.site,'index.html'),'utf8');
let script=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
script=script.replace('      loadAll();','      globalThis.pageTest={state,updateStage,renderCharts}; globalThis.pendingLoad=loadAll();');
vm.runInContext(script,context);
(async()=>{await context.pendingLoad;
 context.pageTest.state.historyScope=input.retained?'retained':'current';
 context.pageTest.state.horizon=input.horizon; context.pageTest.updateStage(); context.pageTest.renderCharts();
 console.log(JSON.stringify({texts:Object.fromEntries([...elements].map(([id,el])=>[id,el.textContent])),
  traces,rows:context.pageTest.state.predLog.length,error:context.pageTest.state.loadError}));
})().catch(error=>{console.error(error);process.exit(1);});
"""
        return node_json(program, {"site": str(self.site), "fault": fault,
                                   "retained": retained, "horizon": horizon})

    def test_actual_page_uses_a_verified_current_method_snapshot(self):
        page = self.page()
        self.assertEqual(page["error"], "")
        self.assertEqual(page["texts"]["connectionLabel"], "Verified artifact snapshot")
        self.assertEqual(page["texts"]["stageSummary"], "Today's daily high")
        self.assertEqual(page["texts"]["metricSkill"], "Pending")
        self.assertIn("Decision-time paired outcomes", page["texts"]["forecastNotice"])
        self.assertIn("Current published method", page["texts"]["historyNote"])
        self.assertNotIn("Not displayable", page["texts"]["stageInterval"])

    def test_suppressed_horizon_has_no_display_interval(self):
        self.assertEqual(self.page(horizon=3)["texts"]["stageInterval"], "Not displayable")

    def test_missing_manifest_shows_no_demo_or_model_substitution(self):
        page = self.page("missing")
        self.assertEqual(page["rows"], 0)
        self.assertEqual(page["texts"]["stageTemp"], "—")
        self.assertEqual(page["texts"]["stageMode"], "UNKNOWN")
        self.assertIn("Publication unavailable", page["texts"]["forecastNotice"])

    def test_corrupt_artifact_and_mid_load_deployment_fail_closed(self):
        for fault in ("corrupt", "race"):
            with self.subTest(fault=fault):
                page = self.page(fault)
                self.assertEqual(page["rows"], 0)
                self.assertEqual(page["texts"]["stageTemp"], "—")
                self.assertTrue(page["error"])

    def test_plotly_receives_distinct_target_dates_for_each_horizon(self):
        series = self.page()["traces"]["forecastHistoryChart"]["series"]
        self.assertEqual(series[0]["x"][0], "2026-10-12")
        self.assertEqual(series[2]["x"][0], "2026-10-14")
        self.assertEqual(series[4]["x"][0], "2026-10-16")
        self.assertEqual(series[0]["x"], series[1]["x"])

    def test_retained_history_and_residual_lines_break_at_the_observed_gap(self):
        page = self.page(retained=True)
        for name in ("forecastHistoryChart", "errorChart"):
            series = page["traces"][name]["series"][0]
            start = series["x"].index("2026-08-26")
            end = series["x"].index("2026-09-14")
            self.assertIn(None, series["y"][start + 1:end])
            self.assertFalse(series["connectgaps"])

    def test_canonical_only_and_pacific_clock_rules_run_in_javascript(self):
        program = r"""
const contract=require(process.argv[1]);
const row={feature_date:'2026-11-03',target_date_h1:'2026-11-04',publish_mode:'CAUTION',governance_run_at:'2026-11-04T14:35Z',blend_h1:'82.5'};
console.log(JSON.stringify({missing:contract.forecast(row,1),hold:contract.forecast({...row,publish_mode:'HOLD',forecast_h1:'72'},1),
 today:contract.targetStatus(row,1,new Date('2026-11-05T07:59:59Z')),
 past:contract.targetStatus(row,1,new Date('2026-11-05T08:00:00Z')),
 upcoming:contract.targetStatus(row,1,new Date('2026-11-04T07:59:59Z')),
 invalid:contract.targetDate({...row,target_date_h1:'2026-11-05'},1)}));
"""
        result = node_json(program)
        self.assertIsNone(result["missing"])
        self.assertIsNone(result["hold"])
        self.assertIsNone(result["invalid"])
        self.assertEqual(result["today"], "Today's daily high")
        self.assertEqual(result["past"], "Past target · archived forecast")
        self.assertEqual(result["upcoming"], "Upcoming daily high")

    def test_inline_javascript_parses(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node required")
        script = re.findall(r"<script>([\s\S]*?)</script>", (ROOT / "index.html").read_text())[-1]
        subprocess.run([node, "--check"], input=script, text=True, capture_output=True, check=True)


if __name__ == "__main__":
    unittest.main()
