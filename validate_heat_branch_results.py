#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heat-branch audit checker.

Run after a retrain / Part 2B rerun:

    python validate_heat_branch_results.py

This checker does not replace validate_artifacts.py. It classifies the heat
experiment and prints the exact reason to keep, reject, or keep monitoring.

Current audit rule
------------------
Reject the asymmetric heat-branch LSTM objective if:
  * the experiment is enabled,
  * all test-set heat-event hit rates are 0, and
  * the average LSTM test MAE is materially high.

Separately, warn if the live canonical forecasts remain more than 3°F colder
than NWS on warm NWS days.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


P2_META = Path("artifacts_part2/part2_meta.json")
P2B_SUMMARY = Path("artifacts_part2b/part2b_summary.json")
P3_GOVERNANCE = Path("artifacts_part3/governance_report.json")

MATERIAL_MAE_F = 4.50
MAX_ALLOWED_WARM_NWS_COLD_GAP_F = 3.0
WARM_EVENT_THRESHOLD_F = 78.0


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        print(f"❌ Missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _heat_diag(metrics: dict, prefix: str, horizon: int) -> dict:
    return metrics.get(f"{prefix}h{horizon}_heat_event_diagnostics", {})


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def main() -> int:
    meta = _load_json(P2_META)
    if meta is None:
        return 1

    heat = meta.get("heat_experiment", {})
    val_metrics = meta.get("val_metrics", {})
    test_metrics = meta.get("test_metrics", {})

    print("\n=== PART 2 HEAT EXPERIMENT ===")
    for key in [
        "enabled", "decision", "warm_event_f", "heat_event_f", "warm_weight",
        "heat_weight", "warm_underpred_penalty", "heat_underpred_penalty",
        "n_warm_or_hot_train_sequences", "n_heat_train_sequences",
    ]:
        print(f"{key}: {heat.get(key)}")

    val_mae = _safe_float(meta.get("val_mae_f"))
    test_mae = _safe_float(meta.get("test_mae_f"))

    print("\n=== GLOBAL LSTM MAE ===")
    print(f"val_mae_f:  {val_mae}")
    print(f"test_mae_f: {test_mae}")

    print("\n=== HEAT EVENT DIAGNOSTICS ===")
    test_hit_rates = []
    for h in [1, 3, 5]:
        v = _heat_diag(val_metrics, "val_", h)
        t = _heat_diag(test_metrics, "test_", h)
        test_hit_rates.append(_safe_float(t.get("hit_rate"), 0.0))

        print(f"H={h}")
        print(
            "  val : "
            f"n={v.get('n_true_heat_days')} "
            f"hits={v.get('predicted_heat_hits')} "
            f"hit_rate={v.get('hit_rate')} "
            f"bias={v.get('heat_bias_f')} "
            f"mae={v.get('heat_mae_f')}"
        )
        print(
            "  test: "
            f"n={t.get('n_true_heat_days')} "
            f"hits={t.get('predicted_heat_hits')} "
            f"hit_rate={t.get('hit_rate')} "
            f"bias={t.get('heat_bias_f')} "
            f"mae={t.get('heat_mae_f')}"
        )

    heat_enabled = bool(heat.get("enabled", meta.get("hyperparameters", {}).get("heat_branch_experiment", False)))
    all_test_heat_hits_zero = all((x or 0.0) <= 0.0 for x in test_hit_rates)

    if heat_enabled and all_test_heat_hits_zero and test_mae is not None and test_mae >= MATERIAL_MAE_F:
        decision = "REJECT_HEAT_BRANCH_LSTM"
        reason = (
            "Asymmetric heat-branch LSTM objective is enabled, all test heat-event "
            f"hit rates are 0, and test_mae_f={test_mae:.3f} >= {MATERIAL_MAE_F:.2f}."
        )
    elif not heat_enabled:
        decision = "HEAT_BRANCH_DISABLED"
        reason = "Production Part 2 is not using the asymmetric heat-branch objective."
    else:
        decision = "MONITOR"
        reason = "Heat branch did not trip the rejection rule; compare against prior baseline before promotion."

    print("\n=== HEAT BRANCH DECISION ===")
    print(f"decision: {decision}")
    print(f"reason: {reason}")

    p2b = _load_json(P2B_SUMMARY)
    if p2b:
        print("\n=== PART 2B CANONICAL FORECAST / NWS GAP ===")
        print(f"forecast_source: {p2b.get('forecast_source')}")
        print(f"canonical_forecast: {p2b.get('canonical_forecast')}")
        print(f"nws_anchor_policy: {p2b.get('nws_anchor_policy')}")

        details = p2b.get("nws_anchor_details", {})
        warm_gap_warnings = []
        for h in ["h1", "h3", "h5"]:
            d = details.get(h, {})
            final_f = _safe_float(d.get("final_f"))
            nws_f = _safe_float(d.get("nws_f"))
            if final_f is None or nws_f is None:
                continue
            gap = nws_f - final_f
            if nws_f >= WARM_EVENT_THRESHOLD_F and gap > MAX_ALLOWED_WARM_NWS_COLD_GAP_F:
                warm_gap_warnings.append(f"{h}: NWS={nws_f:.1f}, final={final_f:.1f}, cold_gap={gap:.1f}")

        if warm_gap_warnings:
            print("⚠️ warm_nws_gap_warning:")
            for w in warm_gap_warnings:
                print(f"  - {w}")
        else:
            print("warm_nws_gap_warning: none")

    gov = _load_json(P3_GOVERNANCE)
    if gov:
        print("\n=== GOVERNANCE ===")
        print(f"publish_mode: {gov.get('publish_mode')}")
        print(f"bnn_intervals_displayable: {gov.get('bnn_intervals_displayable')}")
        print(f"checks_passed: {gov.get('checks_passed')}/{gov.get('checks_total')}")

    print("\nRecommended interpretation:")
    print("  - Use decision=REJECT_HEAT_BRANCH_LSTM to disable the asymmetric Part 2 objective.")
    print("  - Use warm_nws_gap_warning to decide whether Part 2B's NWS anchor is still too weak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

