#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heat-branch result checker.

Run after the heat experiment retrain:

    python validate_heat_branch_results.py

This checker does not replace validate_artifacts.py. It only summarizes the
specific metrics needed for the heat-branch decision.
"""

from __future__ import annotations

import json
from pathlib import Path


P2_META = Path("artifacts_part2/part2_meta.json")
P2B_SUMMARY = Path("artifacts_part2b/part2b_summary.json")


def _load_json(path: Path):
    if not path.exists():
        print(f"❌ Missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _heat_diag(metrics: dict, prefix: str, horizon: int):
    return metrics.get(f"{prefix}h{horizon}_heat_event_diagnostics", {})


def main() -> int:
    meta = _load_json(P2_META)
    if meta is None:
        return 1

    heat = meta.get("heat_experiment", {})
    print("\n=== PART 2 HEAT EXPERIMENT ===")
    print(f"enabled: {heat.get('enabled')}")
    print(f"heat_event_f: {heat.get('heat_event_f')}")
    print(f"heat_weight: {heat.get('heat_weight')}")
    print(f"heat_underpred_penalty: {heat.get('heat_underpred_penalty')}")
    print(f"n_heat_train_sequences: {heat.get('n_heat_train_sequences')}")

    print("\n=== GLOBAL MAE ===")
    print(f"val_mae_f:  {meta.get('val_mae_f')}")
    print(f"test_mae_f: {meta.get('test_mae_f')}")

    val_metrics = meta.get("val_metrics", {})
    test_metrics = meta.get("test_metrics", {})

    print("\n=== HEAT EVENT DIAGNOSTICS ===")
    for h in [1, 3, 5]:
        v = _heat_diag(val_metrics, "val_", h)
        t = _heat_diag(test_metrics, "test_", h)
        print(f"H={h}")
        print(f"  val : n={v.get('n_true_heat_days')} hits={v.get('predicted_heat_hits')} hit_rate={v.get('hit_rate')} bias={v.get('heat_bias_f')}")
        print(f"  test: n={t.get('n_true_heat_days')} hits={t.get('predicted_heat_hits')} hit_rate={t.get('hit_rate')} bias={t.get('heat_bias_f')}")

    p2b = _load_json(P2B_SUMMARY)
    if p2b:
        print("\n=== PART 2B CANONICAL FORECAST ===")
        print(f"forecast_source: {p2b.get('forecast_source')}")
        print(f"nws_anchor_used: {p2b.get('nws_anchor_used')}")
        print(f"canonical_forecast: {p2b.get('canonical_forecast')}")

    print("\nDecision rule:")
    print("  Keep branch only if heat-event bias improves without a material global MAE regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
