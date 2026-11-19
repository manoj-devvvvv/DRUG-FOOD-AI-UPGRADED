# DFI serving and UI hardening — 2026-11-18

## Changes

- Added ascorbic acid (vitamin C) to the grape marker map in the serving config, persisted marker artifact, and label-generation dictionary. The source dataset contains a positive Chromium + Ascorbic acid pair associated with Grape.
- Changed food-level candidate selection so exact negative dataset evidence remains highest priority, followed by exact positive/other dataset evidence, then ML-only estimates. This stops an unverified score for another food constituent from hiding a verified exact-pair annotation.
- When the model's top class is beneficial but falls below the validation acceptance threshold, the API now states that the leading signal is positive while clearly marking it low-confidence and not a recommendation. The system still abstains from a confident recommendation.
- Simplified the prediction UI: the three class probability bars remain visible; technical metadata and SMILES diagnostics are removed from the default result; candidate lists and the top-six SHAP explanation are collapsed until requested.
- Added a Chromium + grape positive-data example and an integration test for its evidence lookup.

## Validation limits

The saved training artifact reports 8 positive examples in the Split G training partition. The full specific-pair label set contains 24 class-1 pairs; the Split G held-out test partition has only 1 positive pair. The existing held-out report records 1/1 recall for that single test item but only 0.125 precision for the positive class (8 positive predictions total). This is far too small to support a 9/10 or 90% reliability claim.

The data package contains no additional raw interaction file from which to curate new positive labels. Do not synthetically invent positive labels or reuse the paper-style labels as beneficial: the project's label audit already found that many source `Positive` annotations are warnings, not benefits. Improving class-1 generalization requires more independently curated beneficial drug–food pairs, including diverse drugs and foods, followed by a new grouped validation/test split and retraining.

## Checks performed in this container

- Python source compilation passed (`compileall`).
- `config.yaml` parsed successfully; grape marker list includes ascorbic acid.
- Frontend JavaScript syntax check passed (`node --check`).
- Full API integration tests could not be executed here because the runtime lacks the project dependencies `PyBioMed` and `pyarrow`; both are declared in `requirements.txt`. Run `pytest -q` after installing the listed requirements in the project environment.

## Model-artifact note

The trained LightGBM weights and calibration artifact were not changed. Retraining with only eight positive training pairs would risk memorizing those few observations and would not justify a 9/10 quality claim. This patch repairs positive evidence retrieval, makes low-confidence positive model signals visible without mislabelling them as accepted predictions, and reduces UI clutter. It is a serving/data/UI correction, not a newly validated higher-accuracy model.
