# Deletion manifest — paper-style mode retirement (2026-11-16)

The paper-style label variant copied one drug-level label onto every sampled
compound. It was leakage-prone by construction and is no longer built, served or
evaluated. Its artifacts were deleted; datasets and all pair-level artifacts were
kept.

## Deleted patterns

| Location | Pattern | Reason |
|---|---|---|
| `models/` | `*_P.joblib`, `*_P.json`, `*_P.pkl` | Model, calibration, domain, metadata, scaler and feature artifacts for the retired variant |
| `reports/` | `*_P.json`, `*_P.md`, `*_P.csv`, `metrics_P.*`, `cv_results_P.csv`, `tuning_*_P.csv`, `generalization_P.*`, `baselines_P.*`, `system_comparison_P.csv` | Per-variant evaluation, tuning and report artifacts |
| `reports/predictions/` | `*_P.*` | Held-out prediction dumps |
| `figures/` | `cm_P_*`, `shap_summary_P_*`, comparison figures for P | Figures for the retired variant |
| `data/processed/` | `splits_P.parquet`, `evidence_P.parquet` | Per-variant processed tables |
| `notebooks/` | `07b_selection_P.ipynb`, `08b_models_P.ipynb` | Walkthroughs for the retired variant |
| logs and caches | P run logs, `__pycache__/`, `.pytest_cache/` | Stale build state |

## Kept

- All raw and cleaned datasets, including `data/interim/pairs_labeled.parquet`,
  which still contains the source label rows for both historical variants.
- All pair-level artifacts: model, calibration, domain, metadata, selected
  features, splits, evidence, reports, figures and notebooks.
- The implementation audit and change log, so the removal is documented.

## Serving cleanup (2026-11-18)

| Location | Removed | Reason |
|---|---|---|
| `reports/` | `predictions_S_G.parquet`, `predictions_S_R.parquet` | Brick 9 dumps from earlier fits; the deployed artifact's Split G rows are shipped as `step_csvs/model_predictions_G.csv`, and the R dump came from a separate R model. |
| `reports/` | `*.log` (`brick7`, `brick8`, `brick9`, `brick10`, `brick11`, `brick12`, `web_server`) | Superseded run logs; every stage result is stored as JSON/MD/CSV. |
| project root | `dfi.zip`, `finalefi.zip` | Stale project snapshots predating the single-model and serving fixes. |
| project root | `.pytest_cache/`, `__pycache__/` | Regenerated caches. |

## Verification

```bash
find . -name "*_P.*" -o -name "*_P_*" | grep -v __pycache__
```

returns no artifact files. `scripts/check_gates.py` includes a “Single served
split” gate that fails if `data/processed/splits_P.parquet` reappears.
