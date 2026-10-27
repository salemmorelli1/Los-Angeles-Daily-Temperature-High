#!/usr/bin/env bash
# The same verification entry point is executed by Linux and Windows CI.
set -Eeuo pipefail
trap 'printf "Verification stopped at line %s (exit %s).\n" "$LINENO" "$?" >&2' ERR
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate Python 3.12 first."'
test -z "$(git status --porcelain)"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LATEMP_ROOT="$(python -c 'from pathlib import Path; print(Path.cwd())')"
mkdir -p .venv
verification_dir="$(mktemp -d .venv/audit-check.XXXXXX)"
python -m venv "$verification_dir/runtime"
if [[ -f "$verification_dir/runtime/Scripts/python.exe" ]]; then
  verification_python="$verification_dir/runtime/Scripts/python.exe"
else
  verification_python="$verification_dir/runtime/bin/python"
fi
"$verification_python" -m pip install --disable-pip-version-check torch==2.14.1 \
  --index-url https://download.pytorch.org/whl/cpu
"$verification_python" -m pip install --disable-pip-version-check -r requirements-ci-lock.txt
"$verification_python" -m pip check
"$verification_python" -c 'import torch, hmmlearn, xgboost; assert torch.version.cuda is None'
"$verification_python" -m compileall -q ./*.py tests
"$verification_python" -m ruff check --select E9,F63,F7,F82 .
"$verification_python" run_daily_forecast.py --help
"$verification_python" part2_deep_learning_forecaster.py --help
"$verification_python" part6_weather_regime_engine.py --help
"$verification_python" deploy_pages.py --help
"$verification_python" -m unittest discover -s tests -v
"$verification_python" validate_artifacts.py

python -m venv "$verification_dir/publication"
if [[ -f "$verification_dir/publication/Scripts/python.exe" ]]; then
  publication_python="$verification_dir/publication/Scripts/python.exe"
else
  publication_python="$verification_dir/publication/bin/python"
fi
"$publication_python" -m pip install --disable-pip-version-check -r requirements-publication-lock.txt
"$publication_python" -m pip check
"$publication_python" -c 'import importlib.util; assert all(importlib.util.find_spec(m) is None for m in ("torch", "sklearn", "hmmlearn", "xgboost"))'
"$publication_python" build_dashboard.py --output "$verification_dir/site"
"$publication_python" build_dashboard.py --output "$verification_dir/site" --verify
git diff --check
git diff --exit-code
test -z "$(git status --porcelain)"
git rev-parse HEAD
printf 'Verification PASS. Preview: %s/site\n' "$verification_dir"
