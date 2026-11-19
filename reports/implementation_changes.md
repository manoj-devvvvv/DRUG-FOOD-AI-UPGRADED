# Implementation changes: before → after

Change log for the DFI project. Part 1 records the 2026-11-15 remediation;
part 2 records the 2026-11-16 rollout to one deployed model, Gemini name
resolution, full per-class SHAP and stage CSV exports. Every number below was
measured from the saved artifacts. This is not a claim that the model is
clinically reliable.

The pre-change audit is in [`implementation_audit.md`](implementation_audit.md).

---

## Part 1 — 2026-11-15 remediation (summary)

The earlier pass found and fixed provenance and evaluation defects:

- Rules became a versioned registry with IDs, provenance and validation status.
- Dataset evidence became a full provenance record with a lookup caveat, never
  counted as ML accuracy.
- A unified response schema added `prediction_source`, explicit abstention
  reasons, model/rule/evidence blocks and calibrated-confidence labelling.
- Calibration moved from a hardcoded 0.6 cutoff to temperature scaling fitted on
  Split G validation plus a validation-selected acceptance threshold.
- The applicability domain moved from all drugs and a hardcoded 0.3 cutoff to
  train-only drug and food references with p10 validation thresholds.
- Splits F (unseen compounds) and D (double-cold) were added with leakage checks.
- Baselines and separate rule / evidence / ML / hybrid paths were measured.
- Per-prediction SHAP and a 29-test suite were added.

The defects that prompted that pass (label collapse, test-drug domain leakage,
unserved calibration, inline rules, no prediction source, no per-prediction SHAP)
remain fixed.

---

## Part 2 — 2026-11-16 single-model rollout

### 2.1 One deployed model; the leakage-prone mode was deleted

The project used to build two label variants: pair-level labels and a
“paper-like” variant that copied one drug-level label onto every sampled
compound. The paper-like variant reached 65,100 class-1 pairs and its Split G
accuracy fell to 0.4102, while its random-split scores were high because the
label was copied, not because of chemistry.

Changes:

- `brick4_labels.py` output now feeds only the pair-level pipeline.
- `brick6_splits.py` builds only the pair-level splits.
- Serving loads one engine; `PredictRequest` has no `mode`; the UI toggle was
  removed; `/api/health` and `/api/meta` describe one model.
- Every paper-like model, calibration, domain, metadata, scaler, prediction,
  tuning, CV, baseline, comparison, `splits_P.parquet`, `evidence_P.parquet`,
  report, figure and notebook was deleted. Datasets and all pair-level artifacts
  were kept.
- `scripts/run_all.sh`, `scripts/check_gates.py`, `scripts/make_report.py` and
  `scripts/build_notebooks.py` were reduced to the single-model pipeline.

### 2.2 Class-ratio control and the deployed artifact

The non-significant class dominated training (8,210 of 10,495 rows). New
[`src/brick17_single_model.py`](../src/brick17_single_model.py) caps that class in
the **training partition only**, keeping strong rows first and filling with weak
synthetic rows. Validation and test keep the real distribution.

| Recipe | Train rows | Validation macro-F1 | Validation avoid recall |
|---|---:|---:|---:|
| uncapped | 10,495 | 0.4465 | 0.312 |
| cap 1:1 | 4,562 | 0.3084 | 0.646 |
| **cap 2:1 (deployed)** | **6,839** | **0.4304** | **0.402** |
| cap 3:1 | 9,116 | 0.4387 | 0.331 |

The validation macro-F1 winner was the uncapped recipe. Cap 2:1 is deployed as a
deliberate **safety trade-off**: −0.016 validation macro-F1 for +0.090
avoid-class recall. It is not presented as an improvement. The choice is
recorded in `models/calibration_S.json`, `reports/class_balance_sweep.*` and
`reports/single_model_evaluation.*`.

The deployed model is LightGBM version `single-2026-11-16`
(`models/best_model_S.joblib`), refit on the capped training partition. The test
partition was scored once after the ratio, temperature and threshold were frozen:

| Split | Accuracy | Balanced acc. | Macro-F1 | MCC | Class-1 support |
|---|---:|---:|---:|---:|---:|
| G — unseen drugs (headline) | 0.7269 | 0.7428 | 0.4865 | 0.2408 | 1 |
| R — pair-random reference | 0.7771 | 0.6363 | 0.5633 | 0.5037 | 3 |
| F — unseen compounds | 0.7593 | 0.6446 | 0.6296 | 0.3995 | 2 |
| D — double-cold | 0.6760 | 0.4683 | 0.4601 | −0.0734 | 0 |

Split G per class: avoid precision/recall/F1 0.4375/0.3937/0.4145 (support 640);
beneficial 0.125/1.000/0.2222 (support 1); non-significant
0.8112/0.8346/0.8227 (support 1,977).

### 2.3 Calibration and abstention refreshed for the deployed artifact

`brick17_single_model.py` now owns `reports/calibration_S.json` and
`reports/calibration_S.md` as well as `models/calibration_S.json`. This fixed a
real defect: the old report still described the pre-deployment model
(temperature 1.3203, threshold 0.69). The gate checker caught the mismatch.

| Quantity | Deployed value |
|---|---:|
| Temperature | 1.3405 |
| Validation ECE raw → calibrated | 0.0893 → 0.0321 |
| Test ECE raw → calibrated | 0.0713 → 0.0363 |
| Acceptance threshold | 0.77 |
| Validation coverage | 0.3878 |
| Test coverage | 0.4534 (1,187 pairs) |
| Test accuracy on accepted | 0.8172 |

The threshold is still the lowest validation threshold whose accepted error is
at most 20%; no test data is used to fit or choose it.

Applicability domain is unchanged: train-only references (223 drugs, 2,291
compounds), Morgan radius 2 / 2,048 bits, p10 thresholds (drug 0.1856, food
0.3333).

### 2.4 Gemini-first name resolution

New [`src/resolve.py`](../src/resolve.py) resolves free-text names in this order:

1. **local database** — curated names already carry structures;
2. **Gemini** — `gemini-flash-lite-latest` (override with `GEMINI_MODEL`, with model fallbacks) is asked for
   JSON and told not to invent a structure;
3. **PubChem** — fallback when Gemini has no key, is unreachable or returns no
   valid SMILES;
4. **user SMILES** — validated directly and never overridden.

Safety properties:

- every candidate SMILES must parse with RDKit and is stored canonicalized;
- `GEMINI_API_KEY` is read from `dfi/.env` or the process environment, with shell
  variables taking precedence; the key is never written to the repo and never
  returned in a response. `src/env.py` loads the file at API/CLI startup and
  `.env.example` is the committed template;
- answers are cached in `data/interim/name_resolution_cache.json`;
- `DFI_ONLINE_RESOLVER=0` disables network calls (tests run this way);
- each response has a `resolution` block with source, canonical SMILES and note
  for both drug and food; the UI shows it in the details panel, including the
  abstention view.

A built-in structure fallback (glucose/dextrose) was restored so offline behavior
does not regress.

### 2.5 SHAP for every feature and every class

[`src/explain.py`](../src/explain.py) previously returned the top six
contributions for the predicted class only. It now returns:

- `features` — all 18 signed contributions for the predicted class, sorted by
  absolute value;
- `by_class` — the full 18-feature vector for each of classes 0, 1 and 2, with
  labels;
- `top_features` — kept for compatibility;
- the same caveat that TreeSHAP describes model behavior, not biology.

The predictor renders the predicted class expanded and the other two classes in
collapsible sections. The explainer is still built once and pre-warmed at
startup. Tests assert 18 features × 3 classes on real API responses.

### 2.6 Stage CSV exports

New [`scripts/export_stage_csvs.py`](../scripts/export_stage_csvs.py) writes one
human-readable CSV per pipeline stage plus `manifest.json`:

| File | Rows | Source |
|---|---:|---|
| `drug_cleaned.csv` | 1,359 | `data/interim/drugs_clean.parquet` |
| `food_cleaned.csv` | 5,022 | `data/interim/foods_clean.parquet` |
| `pairs.csv` | 21,042 | `splits_S.parquet` + names |
| `features_generation.csv` | 6,236 × 63 | per-molecule MOE descriptors |
| `feature_extraction.csv` | 20,947 × 18 | selected features only |

The full 3,780-column pair matrix is not exported; the manifest says so.

### 2.7 Backend and frontend

- `web/app.py`: one engine, no mode in the request schema or logging,
  `resolution_status` in `/api/health` and `/api/meta`, SHAP pre-warmed.
- `web/static/index.html`: the “Specific / Paper-like” control is gone; copy
  states that unknown names are resolved by Gemini then PubChem.
- `web/static/app.js`: per-class SHAP rendering; drug/food resolution source and
  SMILES shown for ML, rule, evidence and abstention results.
- `web/static/styles.css`: styles for the per-class SHAP blocks.
- `web/static/about.html`: single model, cap trade-off, resolution order,
  deployed metrics and updated limitations.

### 2.8 Tests and gates

- New `tests/test_resolver.py`: Gemini JSON parsing, invalid-SMILES rejection,
  cache reuse without network, missing-key behavior, PubChem fallback, offline
  mode and overlong names.
- `tests/test_serving.py`: single-model payloads, resolution block, 18 × 3 SHAP,
  new threshold, calibration consistency.
- `tests/test_provenance_units.py` and `tests/conftest.py`: pair-level only,
  offline resolution forced.
- **38 tests pass**; **29/29 gate checks pass** after the calibration report was
  regenerated.

---

## 3. Before → after (current system)

| Area | Before | After |
|---|---|---|
| Deployed models | Two label variants served | One pair-level model, one meaning per class |
| Class balance | Non-significant class dominated training | Train-only cap 2:1; validation/test untouched |
| Name resolution | Curated tables only | Local DB → Gemini → PubChem → user SMILES, all RDKit-validated |
| Confidence | Hardcoded 0.6 cutoff | Temperature 1.3405 + validation threshold 0.77 |
| SHAP | Top six features, predicted class | All 18 features × all 3 classes |
| Stage data | Parquet intermediates | Named CSV per stage + manifest |
| Reports | Mixed pre/post-deployment calibration | Deployment script owns the served calibration report |
| Tests | 29 | 38 |
| Gates | 31 legacy checks | 29 checks, single-model pipeline |

---

## 4. Honest limitations

- Class 1 has 24 real pairs from 14 drugs. Split G test has one, so its F1
  (0.2222) is not evidence of learning.
- The majority baseline has higher raw Split G accuracy (0.7552 vs 0.7269).
  Balanced accuracy (0.7428 vs 0.3333) and MCC (0.2408 vs 0) show real signal,
  but most avoid pairs are still missed.
- Double-cold generalization is weak (MCC −0.0734); broad chemical
  generalization is not claimed.
- Rules cover 0% of the Split G test partition. Evidence lookup covers 46.8% but
  its 100% accuracy is a label echo, excluded from model comparison.
- The applicability domain is a validated heuristic, not proof of chemical
  validity.
- Gemini may resolve an ambiguous name to the wrong structure. The response
  exposes the SMILES and source so the user can check it; it is not silently
  trusted.
- The external validation set from the source paper was not obtained; no
  comparison with its reported 0.9756 accuracy is claimed.
- SHAP shows model behavior, not biological mechanism.
- This is a research tool, not a medical device.

---

## 5. Reproduction

```bash
# from dfi/
export PYTHONPATH=src PYTHONIOENCODING=utf-8
bash scripts/run_all.sh

# single-model stages
python src/brick17_single_model.py --ratios 1,2,3 --deploy-ratio 2 --apply
python scripts/export_stage_csvs.py
python scripts/make_report.py

python -m pytest -q tests
python scripts/check_gates.py
python -m uvicorn web.app:app --host 127.0.0.1 --port 8010
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` there before starting the app
to enable Gemini name resolution. The app loads `dfi/.env` at startup and shell
variables override it; `.env` is git-ignored and never read from the repository.

---

## 6. Files added or changed

Added in this rollout:
`src/resolve.py`, `src/env.py`, `src/brick17_single_model.py`, `scripts/export_stage_csvs.py`,
`tests/test_resolver.py`, `tests/test_env.py`, `.env.example`, `.gitignore`,
`data/processed/csv/` exports and manifest.

Changed in this rollout:
`src/recommend.py`, `src/explain.py`, `src/brick6_splits.py`, `web/app.py`,
`web/static/index.html`, `web/static/app.js`, `web/static/styles.css`,
`web/static/about.html`, `scripts/run_all.sh`, `scripts/check_gates.py`,
`scripts/make_report.py`, `scripts/build_notebooks.py`, `ARCHITECTURE.md`,
`reports/report.md`, `reports/calibration_S.{json,md}`.

Deleted: all paper-like model, calibration, domain, metadata, scaler, prediction,
tuning, CV, baseline, comparison, split, evidence, figure, report and notebook
artifacts. Datasets and pair-level artifacts were not deleted.
