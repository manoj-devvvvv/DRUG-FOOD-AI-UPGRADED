#!/usr/bin/env bash
# Rebuild every artifact in order. Long running: about 40-60 minutes on a 4-core laptop.
# One deployed model (historical artifact name "S"); the paper-style Mode P is retired.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
PY=${PY:-python}

$PY src/brick1_audit.py
$PY src/brick2_clean_drugs.py
$PY src/brick3_clean_foods.py
$PY src/brick4_labels.py
$PY src/brick5_features.py
PYTHONPATH=src $PY -m pytest tests -q
$PY src/brick6_splits.py

$PY src/brick7_selection.py --mode S
$PY src/brick8_models.py --mode S
$PY src/brick9_evaluate.py --mode S
$PY src/brick10_shap.py --mode S

$PY src/brick11_recommender.py
$PY src/brick12_enhancements.py
$PY src/brick13_build_evidence.py

$PY src/brick14_domain_calibration.py --mode S
$PY src/brick15_generalization.py --mode S --cap-ratio 2

# Single deployed model: class-ratio sweep + cap 2:1 retrain, then stage CSVs.
$PY src/brick17_single_model.py --ratios 1,2,3 --deploy-ratio 2 --apply

# Baselines run after deployment so the ML-only/hybrid paths score the served model.
$PY src/brick16_baselines.py --mode S

$PY scripts/export_stage_csvs.py
$PY scripts/build_deliverables.py

$PY scripts/make_report.py
$PY scripts/build_notebooks.py
$PY scripts/check_gates.py

if command -v jupyter >/dev/null 2>&1; then
  jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
fi

echo "done: reports/report.md and notebooks/*.ipynb"
