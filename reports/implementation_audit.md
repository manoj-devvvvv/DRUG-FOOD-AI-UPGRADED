# Implementation audit — Drug–Food Interaction system

Date: 2026-11-15
Scope: `dfi/` as it exists before the remediation described in the master prompt.
This report records what is implemented, what is weak, and the order in which the
weaknesses will be fixed. It does not claim any result that has not been measured.

---

## 1. What the system actually does

### 1.1 Data

| Stage | Count |
|---|---|
| Raw drugs | 1,516 |
| Raw interactions | 23,950 |
| Raw food-compound links | 838,492 (177 foods, 13,282 compounds) |
| Cleaned drugs | 1,359 |
| Cleaned compounds | 5,022 (4,877 organic, 145 inorganic) |
| Retained links | 152,320 |

`pairs_labeled.parquet` has exactly one row per `(mode, drug_id, compound_id)`:
no duplicate keys and no conflicting labels were found.

| Mode | Pairs | Class 0 | Class 1 | Class 2 | Weak class-2 | Drugs | Compounds |
|---|---|---|---|---|---|---|---|
| S (specific, default) | 21,042 | 4,898 | **24** | 16,120 | 11,517 | 444 | 4,572 |
| P (paper-like) | 407,699 | 229,499 | 65,100 | 113,100 | 0 | 1,359 | 4,877 |

### 1.2 Labels

- Raw `Effect` column: Possible 17,827, No Effect 2,374, Positive 1,915, Negative 1,437,
  Harmful 393.
- The 100-sentence audit found that `Effect=Positive` is mostly a warning
  ("avoid grapefruit", "increases plasma concentration"). Mode S therefore treats
  `Positive` as harm unless the sentence states a benefit, and `Possible` as
  non-significant unless the sentence states otherwise.
- Food-level rows (empty `Component`) map to curated marker compounds
  (grapefruit → bergamottin/naringin/bergapten, etc.) instead of every compound
  of the food.
- `negative_precedence` collapses repeated annotations per pair (0 < 1 < 2).

### 1.3 Features

- 60 MOE descriptors per molecule (PyBioMed), 3,780 pair features:
  3,600 products, 60 sums, 60 absolute differences, 60 ratios.
- Four-step selection on the Mode S Split G **training** partition:
  variance > 0.8 → 3,521; Pearson |r| < 0.75 → 1,126; LASSO α=0.01 → 174;
  RFE → 18. Saved in `models/selected_features_S.json` (and `_P`).
- Training/inference feature parity is enforced by the same `pair_matrix` call,
  but there is no automated parity test.

### 1.4 Splits and models

- Split G: whole drugs assigned to train/val/test. Mode S Split G test:
  2,633 pairs (651 / 1 / 1,981 labels); 55 test drugs.
- Split R: stratified random pairs; the same drugs appear everywhere.
- Split G training is used for feature selection, so its test partition is
  uncontaminated by selection **for a newly refit model**, but the deployed model
  was selected by validation macro-F1 across five algorithms.
- Deployed: LightGBM in both modes. Mode S Split G test: accuracy 0.7525,
  macro-F1 0.5066; Split R: 0.8036 / 0.5049.
- `models/best_model_S.joblib` is a `BalancedClassifier` wrapping
  `LGBMClassifier`. `models/train_drugs_S.json` stores training drug IDs.
- No model metadata file exists: the serving layer cannot report a version,
  training date, feature list, or selection provenance.

### 1.5 Inference paths (`src/recommend.py`)

Order of execution in `DFIRecommender.recommend()`:

1. Resolve the drug by name (normalized) or SMILES.
2. **Inline hardcoded rule layer** (`_rule_layer`, lines 255–306).
3. Resolve the food to a compound (name, marker, food-specific compounds, SMILES).
4. For each candidate: **exact-pair evidence lookup** (`self.known`, lines 129–136)
   → **applicability domain** (drug only, hardcoded Tanimoto 0.3) → **LightGBM**.
5. Aggregation across candidates prefers class-0 evidence, then confident class-0,
   then any evidence, then any supported result.

Returned `status` values: `ok`, `rule`, `evidence`, `uncertain`, `not_supported`.
The API only adds `label`/`severity` and rewrites probabilities; it does not
state where the answer came from.

---

## 2. Defects found

### 2.1 Scientific correctness

| # | Defect | Evidence | Impact |
|---|---|---|---|
| S1 | Positive class is almost non-existent | 1,822 `Positive` rows → 1,084 relabelled 0, 646 relabelled 1; only **24 unique pairs** survive; 8 of them are one vitamin-C compound against 8 drugs | Class 1 cannot be learned or evaluated. Test set has 1 positive example. |
| S2 | Applicability domain is not train-only | `recommend.py:161` builds fingerprints from **all 1,359 drugs**, including test drugs | Test-drug similarity is inflated; reported domain status is not a valid uncertainty signal |
| S3 | Applicability domain ignores food compounds | `_domain()` only takes drug SMILES | Novel food compounds are never flagged |
| S4 | AD threshold is a hardcoded 0.3 with no validation | `recommend.py:346` and the `0.3` literal in `_score_pair` | No coverage/error analysis behind it |
| S5 | Calibration is not served | `brick12_enhancements.py` fits sigmoid + split conformal but only writes a report | UI confidence is raw `predict_proba`, not a calibrated probability |
| S6 | Rule layer is unversioned and has no provenance | `_rule_layer` returns `(class, note)` only | Cannot distinguish curated pharmacology from model output; no rule ID, no validation status |
| S7 | Dataset evidence is a bare label map | `self.known[(drug_id, compound_id)] = label` | No record ID, source, annotation quality, PMID, or warning that this is a **lookup**, not a prediction |
| S8 | No prediction provenance in response | no `prediction_source` field | API/UI cannot tell ML from rule from lookup |
| S9 | No local SHAP at inference | `brick10_shap.py` writes global plots only | Per-prediction feature attribution is unavailable |
| S10 | No unseen-food or double-cold evaluation | only Split R and Split G | Generalization claim is limited to unseen drugs |
| S11 | No majority-class / logistic baselines in the deployed evaluation | `brick8_models.py` compares five tree/NN models but not a trivial or linear baseline | Cannot show how much signal the complex models add |
| S12 | Rule/evidence paths are not evaluated separately | no system-level ablation | Unknown how often each path answers and how accurate it is |
| S13 | Sentence-rule audit is only partially validated | 100 rows; 63% rule coverage; 60% agreement on the 15 fired confirmed rows; `manual_label` column empty | The corrected rules are plausible but not gold-standard validated |

### 2.2 Engineering

| # | Defect | Evidence |
|---|---|---|
| E1 | Tests cover only feature arithmetic | `tests/test_features.py` (3 tests) |
| E2 | No API schema or path tests | no `TestClient` tests |
| E3 | Evaluation scripts bypass rules/evidence by construction, but there is no regression test enforcing that | `brick9_evaluate.py` / `brick8b_compare.py` use the model directly |
| E4 | No model version in artifacts | no metadata JSON |
| E5 | Errors are returned as free-text `reason` | no error codes, no structured abstention reason |
| E6 | Frontend renders one confidence component for every path | `app.js` prints `confidence` whenever present |
| E7 | No server logging of prediction source | `web/app.py` has no request logging |

### 2.3 Class-imbalance root cause

The funnel is:

1. 2,172 rows with `Negative`/`Harmful` become class 0.
2. 2,324 `No Effect` rows become class 2.
3. 1,822 `Positive` rows: corrected semantics send 1,084 to class 0 (warnings),
   646 to class 1, 92 to class 2.
4. 17,363 `Possible` rows: text rules send 8,271 to 0, 278 to 1, 8,814 to 2.
5. Only 6,974 interaction rows resolve to a compound target; 1,319 of the 1,822
   `Positive` rows name a `Component`, but many point at the same compounds
   (vitamin C, lentinan polysaccharides, apigenin).
6. Negative precedence collapses repeated mentions.
7. The 23 held-out clinical pairs are removed.

Result: **24 unique positive pairs, 14 drugs**. At least 8 of the 24 come from a
single sentence ("Take with foods containing vitamin C"). This is a dataset
limitation, not a bug that can be fixed by sampling. The honest options are:
(a) report class 1 as effectively unlearnable in Mode S, (b) supplement with an
external interaction dataset, or (c) move benefit-oriented statements to a
separate, clearly-labelled rule/evidence path rather than pretending the model
predicts them.

---

## 3. Remediation plan (incremental, testable)

1. **Provenance layer** — versioned rule module with IDs, patterns, provenance,
   validation status; evidence index with record IDs and annotation metadata;
   `prediction_source`, `predicted_label`, model metadata in every response.
2. **Dataset audit and imbalance report** — complete funnel, conflicts, missing
   structures, external dataset candidates with licensing and label definitions.
3. **Generalization** — add unseen-food (Split F) and double-cold (Split D)
   partitions, verify no drug/compound overlap, retrain LightGBM per partition,
   report per-class metrics and limitations.
4. **Baselines and imbalance experiments** — majority class, balanced logistic
   regression, LightGBM with class weights, random oversampling, and weak-label
   weighting, all fitted on training folds only.
5. **Calibration and abstention** — temperature scaling on validation, ECE/Brier/
   reliability table, validation-selected confidence threshold, coverage–risk curve.
6. **Applicability domain** — train-only drug and compound reference sets,
   validation-derived thresholds, separate drug/food similarity in the response.
7. **Local SHAP** — TreeSHAP top-k contributions for the predicted class, with a
   caveat that attribution is not causation.
8. **Serving and UI** — source-specific display (rule ID/provenance, evidence
   record, ML probabilities/AD/SHAP), structured abstention reasons, logging.
9. **Tests** — every path, schema, source consistency, abstention, invalid input,
   AD, descriptor failure, missing artifact, and feature parity.
10. **Reports** — dataset, model, evaluation, prediction architecture, SHAP,
    and a before/after change log. Old metrics stay labelled as old.

Constraints applied throughout: no fabricated predictions, metrics, records,
references or confidence; no evaluation on training data; no rule/evidence
override in ML-only metrics; no clinical-reliability claims; no deletion of
working artifacts without justification.
