#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply the optional Part 2B heat-branch anchor threshold tweak.

This is intentionally a small patcher instead of a full Part 2B replacement,
because your current Part 2B contains several audit-specific production fixes
and blend-tuning logic. This script only changes:

    HEAT_EVENT_MIN_MODEL_F = <old value>

to:

    HEAT_EVENT_MIN_MODEL_F = 81.0

Run from the repository root:

    python apply_heat_branch_part2b_patch.py
"""

from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("part2b_xgb_ensemble.py")
BACKUP = Path("part2b_xgb_ensemble.py.before_heat_branch_patch")


def main() -> int:
    if not TARGET.exists():
        raise FileNotFoundError(f"{TARGET} not found. Run this from the repo root.")

    text = TARGET.read_text(encoding="utf-8")

    pattern = r"(?m)^HEAT_EVENT_MIN_MODEL_F\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*#.*)?$"
    replacement = (
        "HEAT_EVENT_MIN_MODEL_F = 81.0       "
        "# heat-branch experiment: allow NWS heat guard to engage earlier"
    )

    if not re.search(pattern, text):
        raise RuntimeError(
            "Could not find HEAT_EVENT_MIN_MODEL_F assignment in part2b_xgb_ensemble.py. "
            "No changes made."
        )

    new_text = re.sub(pattern, replacement, text, count=1)

    if new_text == text:
        print("No change needed.")
        return 0

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")
        print(f"Backup written: {BACKUP}")

    TARGET.write_text(new_text, encoding="utf-8")
    print(f"Updated {TARGET}: HEAT_EVENT_MIN_MODEL_F = 81.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
