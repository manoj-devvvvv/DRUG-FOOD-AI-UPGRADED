# Food–Drug Interaction ML with SHAP

**Team:** Team BuildQ
**Project:** Food–Drug ML model with SHAP
**Purpose:** predict whether a drug and a food compound should be avoided together,
taken together, or are non-significant — with calibrated probabilities, dataset
evidence, and SHAP explanations.

This document explains how the system was built, how it is served, and which
technologies it uses. It complements the per-brick reports in `reports/` and the
notebooks in `notebooks/`.

---

## 1. What the system does

Input: a drug and a food compound, supplied as a name, database ID or SMILES.
Output:

- one of three classes — `0` avoid together, `1` beneficial / take with food,
  `2` non-significant;
- an explicit prediction source — `ml_model`, `rule_based`, `dataset_evidence` or
  `none` — so a dataset lookup or curated rule is never presented as ML output;
- for ML answers, calibrated class probabilities, applicability-domain status and a
  per-prediction TreeSHAP explanation covering every feature and every class;
- for every answer, the resolved canonical SMILES for the drug and the food plus where
  it came from (local database, Gemini, PubChem or user SMILES);
- for rule answers, the rule ID, version, provenance and validation status;
- for dataset-evidence answers, the source record, annotation, quality, PMID/DOI and a
  lookup caveat;
- abstention (`uncertain` / `not_supported`) with a machine-readable reason when the
  calibrated probability is below the validation-selected threshold, the pair is
  outside the applicability domain, the compound is a mineral with no dataset record
  and no food-level fallback, or input is invalid.

Every response carries the disclaimer *“Research tool; confirm with a physician or
pharmacist.”*

---

## 2. Repository at a glance

```
dfi/
├── config.yaml                  paths, cutoffs, model and label settings
├── requirements.txt             pinned Python environment
├── ARCHITECTURE.md              this document
├── src/                         pipeline (bricks) and inference engine
│   ├── common.py                config and JSON/Parquet helpers
│   ├── brick1_audit.py          raw data audit
│   ├── brick2_clean_drugs.py    drug cleaning and deduplication
│   ├── brick3_clean_foods.py    compound cleaning, Tanimoto dedup
│   ├── brick4_labels.py         label construction (single pair-level policy)
│   ├── features.py              60 MOE descriptors and 3,780 pair features
│   ├── brick5_features.py       descriptor computation
│   ├── brick6_splits.py         Split G / R / F / D
│   ├── brick7_selection.py      four-step selection to 18 features
│   ├── brick8_models.py         five classifiers, tuning, frozen winner
│   ├── brick8b_compare.py       all five models on held-out test data
│   ├── brick9_evaluate.py       test evaluation, threshold sweep, per-drug spread
│   ├── brick10_shap.py          TreeSHAP per class
│   ├── brick11_recommender.py   clinical case suite
│   ├── brick12_enhancements.py  ablations, calibration, conformal, bootstrap CIs
│   ├── brick13_build_evidence.py  dataset evidence and rule registry
│   ├── brick14_domain_calibration.py  temperature scaling and applicability domain
│   ├── brick15_generalization.py  R / G / F / D partitions and leakage checks
│   ├── brick16_baselines.py     baselines, imbalance experiments, system paths
│   ├── brick17_single_model.py  class-ratio sweep and deployed model
│   ├── resolve.py               Gemini → PubChem → SMILES name resolution
│   ├── calibration.py           temperature scaling and abstention helpers
│   ├── domain.py                Morgan/Tanimoto applicability domain
│   ├── explain.py               TreeSHAP for all features and classes
│   ├── rules.py                 curated rule registry
│   └── recommend.py             production inference engine
├── models/                      selected features, frozen model, calibration, metadata
├── data/interim|processed       cleaned tables, features, splits, stage CSV exports
├── reports/                     JSON, CSV, Markdown artifacts per brick
├── figures/                     confusion matrices, SHAP plots, comparison charts
├── notebooks/                   walkthrough notebooks 00–13 + 08c comparison
├── scripts/                     run_all.sh, check_gates.py, make_report.py, export_stage_csvs.py
└── web/
    ├── app.py                   FastAPI service
    └── static/
        ├── landing.html         landing page
        ├── index.html           predictor page
        ├── about.html           project documentation page
        ├── styles.css           shared design system and predictor styles
        ├── pages.css            landing/about layout
        └── app.js               predictor behaviour
```

---

## 3. End-to-end development flow

| Stage | Script | Main artifacts | Key result |
|---|---|---|---|
| Environment | — | `reports/brick0_environment.md` | Python, RDKit, XGBoost, LightGBM, SHAP verified |
| Data audit | `brick1_audit.py` | `brick1_audit.json/md` | 1,516 drugs, 23,950 interactions, 838,492 food links |
| Clean drugs | `brick2_clean_drugs.py` | `brick2_funnel.*`, `drugs_clean.parquet` | 1,359 unique drugs with valid structures |
| Clean foods | `brick3_clean_foods.py` | `brick3_funnel.*`, `foods_clean.parquet` | 4,877 organic compounds after Tanimoto dedup |
| Labels | `brick4_labels.py` | `pairs_labeled.parquet`, `brick4_labels.md` | 21,042 labelled pairs |
| Features | `brick5_features.py` | `drug_moe.parquet`, `food_moe.parquet` | 60 MOE descriptors per molecule, all finite |
| Splits | `brick6_splits.py` | `splits_S.parquet` | Drug-grouped Split G, random Split R, compound Split F, double-cold Split D |
| Selection | `brick7_selection.py` | `selected_features_S.json` | 3,780 → 18 features |
| Models | `brick8_models.py` | tuning CSVs, `brick8_models_S.json` | LightGBM wins the five-model comparison |
| Comparison | `brick8b_compare.py` | `model_comparison.*` | All five models on held-out test |
| Evaluation | `brick9_evaluate.py` | `metrics_S.json`, predictions | Split G headline metrics |
| Explainability | `brick10_shap.py` | `shap_summary_S_*.png` figures | Per-class TreeSHAP attributions |
| Case suite | `brick11_recommender.py` | `brick11_case_study.*` | 20 cases, 17/17 scored pass |
| Enhancements | `brick12_enhancements.py` | `brick12_enhancements.*` | Ablations, conformal, bootstrap CIs |
| Provenance | `brick13_build_evidence.py` | `evidence_S.parquet`, `dataset_audit.*`, `rules_v1.json` | 9,525 non-weak evidence rows, audit of the 24 positive pairs |
| Calibration + domain | `brick14_domain_calibration.py` | `calibration_S.json`, `domain_S.json` | Temperature scaling, acceptance threshold, train-only AD references |
| Generalization | `brick15_generalization.py` | `generalization_S.*` | R / G / F / D partitions with leakage checks |
| Baselines | `brick16_baselines.py` | `baselines_S.*`, `system_comparison_S.csv` | Majority, logistic, imbalance experiments, rule/evidence/ML/hybrid paths |
| Class ratio + deploy | `brick17_single_model.py` | `class_balance_sweep.*`, deployed `best_model_S.joblib`, `single_model_evaluation.*` | Train-only class-2 cap 2:1; refit, recalibrate, score once |
| Stage exports | `scripts/export_stage_csvs.py` | `data/processed/csv/*.csv` + `manifest.json` | One human-readable CSV per pipeline stage |
| Report | `make_report.py` | `reports/report.md` | Consolidated narrative |
| Web app | `web/app.py` | running FastAPI service | Landing, predictor, about, API |

The full pipeline can be regenerated with `scripts/run_all.sh`; `scripts/check_gates.py`
verifies the artifact and report gates.

---

## 4. Data and labels

### 4.1 Raw and cleaned data

| Stage | Count |
|---|---:|
| Raw drugs | 1,516 |
| Raw interaction records | 23,950 |
| Raw food–compound links | 838,492 |
| Unique raw compounds | 13,282 |
| Clean drugs | 1,359 |
| Clean organic compounds | 4,877 |
| Inorganic compounds (excluded) | 145 |
| Retained food–compound links | 152,320 |

Drugs are cleaned to valid, deduplicated structures (InChIKey). Compounds are
deduplicated with canonical SMILES and a Morgan/Tanimoto near-duplicate filter,
while protected marker compounds (ethanol, tyramine, caffeine, bergamottin, naringin,
hyperforin, phylloquinone, vitamin K1, dimethyl disulfide) are always retained.

### 4.2 Label policy

The source data is record-level free text, so turning it into pair-level training
labels requires a policy. The project now has **one** label policy and **one** served
model.

| | Specific pair labels |
|---|---|
| Label granularity | one per drug–compound pair |
| Compound selection | named in the text, curated food markers, or food-specific compounds |
| Semantics | warning-style “Positive” → avoid (class 0) |
| Drugs covered | 444 |
| Pairs | 21,042 |
| Class counts | `{0: 4,898, 1: 24, 2: 16,120}` |
| Synthetic weak pairs | 11,517 (weight 0.5) |
| Intended use | prediction under unseen-drug validation |

The earlier “paper-like” mode copied one drug-level label onto every sampled
compound, which inflated class 1 to 65,100 pairs and made the task look easier than
it is. It was removed from the pipeline, serving path and artifacts because its
scores came from label leakage rather than chemistry.

---

## 5. Feature engineering

1. **Molecule → descriptors.** RDKit parses canonical SMILES; PyBioMed computes 60
   MOE descriptors per molecule (VSA, EState, Labute ASA, slogP and related terms).
2. **Pair → feature vector.** For each pair the 60 descriptor values of the drug and
   the compound are combined into:
   - 3,600 products `drug_i * food_j`
   - 60 sums `drug_i + food_i`
   - 60 absolute differences `|drug_i - food_i|`
   - 60 ratios `drug_i / food_i`
   - total: **3,780 candidate features**
3. **Selection.** Four steps on a stratified 10,495-pair training sample:
   - variance filter (threshold 0.8): 3,780 → 3,521
   - Pearson correlation pruning (|r| ≥ 0.75): → 1,126
   - LASSO (α = 0.01) non-zero coefficients: → 174
   - RFECV / RFE to a fixed 18 features: → **18**
   Stability selection runs the selection three times; the final set is persisted in
   `models/selected_features_S.json`.

Example features include `PEOEVSA5+PEOEVSA5`, `VSAEstate6+VSAEstate6`,
`slogPVSA1/slogPVSA1`, `MRVSA9/MRVSA9`, `abs(MRVSA5-MRVSA5)`. The exact list is
loaded at inference time, so the serving code cannot drift from training.

---

## 6. Splits and validation protocol

- **Split G (headline).** Whole drugs are assigned to train/validation/test, so test
  drugs never appear during training. This measures generalization to unseen drugs.
  10,537 train / 7,872 validation / 2,633 test pairs (2,618 after feature
  filtering).
- **Split R (reference).** Random pair split with the paper's class proportions; used
  for comparability and to expose how much easier random-split evaluation is.
- **Split F (unseen compounds).** Whole food compounds are assigned to partitions, so a
  test compound never appears during training.
- **Split D (double-cold).** Drugs and compounds are assigned independently and only
  pairs whose drug and compound land in the same partition are kept. This is the
  hardest split and costs rows (5,237 train / 2,920 validation / 321 test).
- **Grouped CV.** During tuning, `GroupKFold` groups by drug on Split G so folds do not
  share drugs. Tuning itself uses `PredefinedSplit` (train → validation).
- **Held-out cases.** 23 interaction rows plus 6 clinical probes are removed before
  splitting and reused in the Brick 11 case study.

---

## 7. Models and tuning

Five classifiers share the same 18 features and training protocol:

| Model | Library | Class handling |
|---|---|---|
| XGBoost | `xgboost` | balanced sample weights, weak-pair weight 0.5 |
| LightGBM | `lightgbm` | balanced sample weights, weak-pair weight 0.5 |
| RandomForest | `scikit-learn` | balanced sample weights, weak-pair weight 0.5 |
| ExtraTrees | `scikit-learn` | balanced sample weights, weak-pair weight 0.5 |
| MLP | `scikit-learn` | standardized inputs, early stopping; no sample weights |

Hyperparameters are tuned with `RandomizedSearchCV` on the validation partition,
scoring macro-F1. The search grids cover depth, learning rate, number of trees,
sampling, regularization and, for MLP, hidden sizes and alpha.

### 7.1 Model-comparison run (uncapped training, Split G — unseen drugs)

| Model | Accuracy | Macro-F1 | Avoid precision | Avoid recall | Non-sig. recall |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **0.7525** | **0.5066** | **0.4964** | 0.3203 | 0.8923 |
| MLP | 0.7380 | 0.3879 | 0.4403 | 0.2594 | 0.8933 |
| RandomForest | 0.7002 | 0.4092 | 0.4024 | 0.4641 | 0.7769 |
| XGBoost | 0.5157 | 0.3642 | 0.3042 | 0.7438 | 0.4416 |
| ExtraTrees | 0.4927 | 0.3402 | 0.3089 | 0.8297 | 0.3834 |

Split R: LightGBM 0.8036 accuracy / 0.5049 macro-F1, again the best. LightGBM is the
model family kept for deployment.

### 7.2 Class-ratio control and the deployed single model

Class 2 (non-significant) dominates the training partition. `brick17_single_model.py`
caps class 2 **in the training partition only**, using strong rows first and weak
synthetic rows to fill, and compares caps of 1:1, 2:1 and 3:1 with the uncapped recipe
on validation. Validation and test always keep the real class distribution.

| Recipe | Train rows | Validation macro-F1 | Validation avoid recall |
|---|---:|---:|---:|
| uncapped | 10,495 | 0.4465 | 0.312 |
| cap 1:1 | 4,562 | 0.3084 | 0.646 |
| **cap 2:1 (deployed)** | **6,839** | **0.4304** | **0.402** |
| cap 3:1 | 9,116 | 0.4387 | 0.331 |

The validation macro-F1 winner was the uncapped recipe. Cap 2:1 is deployed as a
**safety trade-off**, not an improvement: it costs 0.016 validation macro-F1 and buys
+0.090 validation avoid-class recall. This choice was made deliberately and is recorded
in `reports/class_balance_sweep.*` and `models/calibration_S.json`.

The deployed artifact is a single LightGBM model (`models/best_model_S.joblib`,
version `single-2026-11-16`) refit on the capped training partition, then scored once
on the untouched Split G test partition:

| Split | Accuracy | Balanced accuracy | Macro-F1 | MCC | Avoid recall | Non-sig. recall |
|---|---:|---:|---:|---:|---:|---:|
| G (unseen drugs, headline) | 0.7269 | 0.7428 | 0.4865 | 0.2408 | 0.3937 | 0.8346 |
| R (pair-random reference) | 0.7771 | 0.6363 | 0.5633 | 0.5037 | — | — |
| F (unseen compounds) | 0.7593 | 0.6446 | 0.6296 | 0.3995 | — | — |
| D (double-cold) | 0.6760 | 0.4683 | 0.4601 | −0.0734 | — | — |

Class 1 contains only 24 real pairs (one in the Split G test partition), so its F1 is
noise and is not used to select the model. Every report prints support beside it.

---

## 8. Explainability with SHAP

- **Global.** TreeSHAP is applied per class on the test partition. Artifacts:
  `figures/shap_summary_S_{negative,positive,non_significant}.png` and
  `reports/brick10_shap_S.json/md` (three plots).
- **Per prediction.** The serving engine builds a `shap.TreeExplainer` over the
  deployed LightGBM model once and returns contributions for **all 18 features and
  all three classes** in every answered prediction (`explainability` block, method
  `tree_shap`). `features` is the full signed vector for the predicted class;
  `by_class` carries the full vector for each of classes 0, 1 and 2 with labels. The
  UI expands the predicted class and keeps the other two collapsible. The explainer
  is pre-warmed at API startup so the first request does not pay the construction cost.
- **Answer paths that carry SHAP.** Model answers (`ok`, `uncertain`) attach
  `explainability` as the decision rationale. Curated-rule and dataset-evidence
  answers keep `status`/`prediction_source` unchanged but attach the same full
  `explainability` block plus a `model_support` block, so the model score is shown as
  corroborating context rather than as the decision. Outside-domain and inorganic
  abstentions carry the same block with `model_support.role = "diagnostic"`. SHAP is
  therefore absent only when no resolvable structure exists on either side, which
  means there is nothing to explain.
- Every explanation carries the caveat: TreeSHAP describes how features move the model
  score for this prediction; it is not evidence of a biological mechanism.

---

## 9. Calibration, abstention and applicability domain

These artifacts are produced by `src/brick14_domain_calibration.py` and refreshed by
`src/brick17_single_model.py` for the deployed capped model. Calibration is fitted on
the Split G validation partition; the held-out test partition is scored only after the
temperature and threshold are frozen.

### 9.1 Temperature scaling

| Temperature | Validation ECE (raw → calibrated) | Test ECE (raw → calibrated) |
|---:|---:|---:|
| 1.3405 | 0.0893 → 0.0321 | 0.0713 → 0.0363 |

`models/calibration_S.json` stores the temperature, the acceptance threshold, the
threshold policy, reliability bins, the threshold sweep and the test selective scores.
Serving applies the temperature to raw probabilities via `src/calibration.py`; if the
artifact is missing, it serves raw probabilities and labels them uncalibrated.

### 9.2 Abstention

The acceptance threshold is the lowest threshold on the validation curve that keeps
accepted-prediction error at or below 20%.

| Acceptance threshold | Validation coverage | Test coverage | Test accuracy on accepted |
|---:|---:|---:|---:|
| 0.77 | 0.3878 | 0.4534 (1,187 pairs) | 0.8172 |

When the calibrated top probability is below the threshold the response is
`status="uncertain"`, `abstained=true`, `abstention_reason="insufficient_confidence"`.
The probability rows remain visible so the user can see how close the call was.

### 9.3 Applicability domain

Two independent Morgan (radius 2, 2,048 bits) Tanimoto checks are applied:

- **drug:** maximum similarity to a **Split G training** drug;
- **food:** maximum similarity to a **Split F training** compound.

Thresholds are the 10th percentile of validation similarities for unseen drugs and
unseen compounds. This keeps roughly 90% of the validation partitions inside while
flagging the most dissimilar structures.

| Drug threshold | Food threshold | Drug references | Food references |
|---:|---:|---:|---:|
| 0.1856 | 0.3333 | 223 | 2,291 |

If a similarity is below its threshold the pair is refused with
`status="not_supported"` and `abstention_reason="outside_applicability_domain"`.
The response reports both similarities, both thresholds and whether the thresholds are
validation-derived. The earlier version used all drugs (including test drugs) as
references and a hardcoded 0.3 boundary; both defects are removed.

### 9.4 Earlier selective-prediction analysis

Brick 12's split-conformal experiment (90% coverage: 0.9213 empirical coverage, 0.4267
abstention, 0.8628 selective accuracy) remains in `reports/brick12_enhancements.*` as a
research comparison. It is **not** used by the serving path; serving uses the
validation-selected temperature and threshold above.

---

## 10. Inference and serving path

```
POST /api/predict
        │
        ▼
DFIRecommender.recommend(drug, food)
        │
        ├─ 1. name resolution: local database → Gemini → PubChem → user SMILES
        ├─ 2. expand food markers plus the food's linked organic compounds
        ├─ 3. versioned clinical rule layer ──► prediction_source "rule_based"
        ├─ 4. exact-pair dataset evidence  ──► prediction_source "dataset_evidence"
        ├─ 5. descriptor + inorganic guards; a mineral without evidence falls
        │     back to the organic compounds of foods that contain it
        ├─ 6. applicability-domain check   ──► abstain "outside_applicability_domain"
        ├─ 7. LightGBM.predict_proba → temperature scaling
        │      ├─ calibrated top ≥ acceptance threshold ──► "ok" / "ml_model"
        │      └─ below threshold ──► "uncertain" / abstain "insufficient_confidence"
        └─ 8. local TreeSHAP over the deployed model (all features × all classes)
```

This ordering is deliberate. Pair descriptors are symmetric and cannot identify a
named drug, so established clinical pairs are handled by rules or by their exact
dataset annotation. The ML model is only used for chemically plausible pairs that are
not covered by those layers, and the response says which mechanism answered.

Everything the prediction path needs at serving time is configuration, not code: the
compound-frequency cutoff, the marker/food/mineral candidate caps, food-name aliases,
marker-compound fallbacks, built-in structures and the stripped food suffixes all live
under `serving:` in `config.yaml`. Class labels, severities, response sentences, the
disclaimer and the demo pairs live under `labels:` and `web:`. Thresholds, AD cutoffs
and calibration come from saved artifacts. There are no food-specific branches in the
recommender.

### 10.1 Name resolution

`src/resolve.py` converts a free-text drug or food name into a canonical SMILES:

1. **local database** — curated drugs and compounds already carry structures;
2. **Gemini** — every uncached name is sent to `gemini-flash-lite-latest` (or
   `GEMINI_MODEL`), asked for JSON and explicitly told not to invent a structure;
3. **PubChem** — used when Gemini is unavailable, has no key or returns no valid SMILES;
4. **user SMILES** — a SMILES typed by the user is validated directly and never
   overridden.

Every returned SMILES is canonicalized with RDKit before use; anything that does not
parse is rejected. Answers are cached in `data/interim/name_resolution_cache.json`
(`DFI_ONLINE_RESOLVER=0` disables network calls; `GEMINI_API_KEY` is read from the
environment or `dfi/.env`, with real shell variables taking precedence, and is never
written to the repository or returned in a response). The
`resolution` block in each response reports the source, the canonical SMILES and any
note, for both the drug and the food.

### 10.2 Unified response schema

Every non-error response carries:

| Field | Meaning |
|---|---|
| `status` | `ok`, `uncertain`, `rule`, `evidence` or `not_supported` |
| `prediction_source` | `ml_model`, `rule_based`, `dataset_evidence` or `none` |
| `abstained`, `abstention_reason` | explicit abstention flag and machine-readable reason |
| `predicted_class`, `predicted_label`, `sentence` | class and user-facing sentence |
| `confidence`, `confidence_type` | `calibrated_model_probability` only for ML answers; `not_applicable` otherwise |
| `probabilities` / `probability_rows` | class probabilities, ML answers only |
| `model` | name, version, feature count, training split; `null` for rule/evidence |
| `rule` | rule ID, version, provenance, validation status; `null` otherwise |
| `evidence` | record ID, annotation, conclusion, source, PMID/DOI, quality, lookup caveat |
| `resolution` | drug/food source (`local_database`, `gemini`, `pubchem`, `user_smiles`), canonical SMILES, note |
| `applicability_domain` | status, drug/food similarities and thresholds, validation flag |
| `calibration` | method, temperature, fitted-on partition |
| `explainability` | TreeSHAP vectors for all 18 features and all three classes, plus caveat; present on `ok`, `uncertain`, `rule`, `evidence`, outside-domain and inorganic abstentions, `null` only when no drug/food structure can be featurized |
| `shap_available`, `shap_reason` | explicit SHAP status on every response; `shap_available` is false only when no structure pair can be scored, and `shap_reason` carries the machine-readable abstention reason |
| `candidates` | food-level candidates, each with its own class label, severity, `shap_available` and full `explainability` block |
| `model_support` | rule/evidence/diagnostic support block: availability, role, explained compound, predicted class and probabilities, AD and calibration context |
| `explanation`, `disclaimer` | plain-language provenance and safety notice |

Rule and dataset-evidence answers never carry a model name, probability vector or
confidence value at the top level; their `confidence` field is `null` and
`confidence_type` is `not_applicable`. The model's view of such a pair is exposed
only inside `model_support`, labelled as supporting context so it cannot be mistaken
for the decision source.

Response statuses:

| Status | Meaning |
|---|---|
| `ok` | model scored the pair and the calibrated top probability clears the threshold |
| `uncertain` | ML probability below the validation-selected threshold; the most likely class is still shown with its confidence and abstention reason |
| `rule` | a versioned curated rule fired |
| `evidence` | exact labelled drug–compound pair in the dataset |
| `not_supported` | invalid input, descriptor failure, out of domain, or a mineral with no dataset record and no food-level fallback |

---

## 11. Backend architecture (FastAPI)

`web/app.py`:

- Adds `src/` to `sys.path`, imports `DFIRecommender`, rule metadata and name-normalization helpers.
- Keeps a single lazily constructed engine in a thread-safe cache; the model is
  loaded once and reused across requests.
- Pre-warms the engine and its SHAP explainer in a startup thread, so the first
  prediction does not pay the explainer construction cost.
- Uses Pydantic request validation (drug/food required, 1–300 characters); there is no
  mode field.
- Logs prediction source, status, abstention flag, resolution sources, inputs and
  latency for every request.
- Appends label, severity, latency, probability rows, resolution and model metadata to
  responses.

### Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | landing page (`landing.html`) |
| `/predict` | GET | predictor page (`index.html`) |
| `/about` | GET | project documentation page (`about.html`) |
| `/figures/{name}` | GET | PNG figures used by the about page |
| `/static/*` | GET | CSS, JS and page assets |
| `/api/health` | GET | readiness, counts, deployed model version, acceptance threshold, rule version, resolver status |
| `/api/meta` | GET | model metadata, feature list, full rule registry, AD thresholds/references, abstention policy, resolution order |
| `/api/examples` | GET | demo pairs for the example chips |
| `/api/suggest` | GET | debounced autocomplete for drug/food inputs |
| `/api/predict` | POST | run the hybrid inference path |

Example request:

```json
POST /api/predict
{"drug": "warfarin", "food": "grape juice"}
```

Example response (rule layer, abridged):

```json
{
  "status": "rule",
  "prediction_source": "rule_based",
  "predicted_class": 0,
  "sentence": "Warfarin should not be taken with food containing grape juice.",
  "confidence": null,
  "confidence_type": "not_applicable",
  "model": null,
  "rule": {
    "rule_id": "WF-GRAPE",
    "version": "2026.11.1",
    "provenance": "Curated from standard clinical pharmacology cautions...",
    "validation_status": "unvalidated_curated"
  },
  "latency_ms": 0.1
}
```

When the ML path answers, the response sets `prediction_source="ml_model"` and adds
`confidence` (calibrated), `probability_rows`, `model`, `resolution`,
`applicability_domain`, `calibration` and the TreeSHAP `explainability` block. When it
abstains, the response keeps the probabilities but sets `abstained=true` and a reason.

---

## 12. Frontend architecture

Three static pages, no framework and no build step:

1. **Landing (`/`)** — team name, project title “Food–Drug ML model with SHAP”, two
   buttons (About model → `/about`, Predict → `/predict`), key dataset statistics.
2. **Predictor (`/predict`)** — drug/food inputs with autocomplete, example chips,
   verdict card, probability bars, source badge (`ML model` / `Curated rule` /
   `Dataset evidence` / `No prediction`), abstention reason, matched compounds,
   candidates and disclaimer. ML answers also show the drug/food resolution source and
   SMILES, the applicability domain, calibration temperature and per-prediction
   TreeSHAP contributions for every feature and class (the predicted class expanded,
   the other two collapsible); rule answers show rule ID, version, provenance and
   validation status; evidence answers show record ID, annotation, source, quality,
   PMID/DOI and a lookup caveat.
3. **About (`/about`)** — abstract, prediction flow, backend/frontend architecture,
   techniques (MOE descriptors, pairwise algebra, Tanimoto/Morgan, feature selection,
   grouped validation, SHAP, conformal), the single label policy, real metrics table,
   tech stack, limitations and safety, plus embedded SHAP and model-comparison figures.

Implementation notes:

- `app.js` uses `fetch` with `AbortController` so stale autocomplete or prediction
  requests cannot overwrite newer results.
- Rendering is XSS-safe: all dynamic values go through `textContent`, never
  `innerHTML`, except one static inline SVG.
- Accessibility: labelled inputs, `aria-live` result region, `aria-busy` during
  loading, visible focus styles, reduced-motion support.
- The UI is responsive down to phone width; probability bars animate on entry.

---

## 13. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | pandas 2.3, NumPy 2.3, PyArrow 25 |
| Chemistry | RDKit 2026.9, PyBioMed 1.0 (MOE descriptors) |
| ML | scikit-learn 1.7, XGBoost 3.4, LightGBM 4.7 |
| Explainability | SHAP 0.54 (TreeSHAP) |
| Evaluation | GroupKFold, PredefinedSplit, RandomizedSearchCV, split conformal, bootstrap, matplotlib |
| Backend | FastAPI 0.111, Uvicorn 0.29, Pydantic 2.12, joblib |
| Name resolution | Gemini `generateContent` REST, PubChem PUG REST, `requests`, RDKit validation, JSON cache |
| Frontend | HTML5, CSS3, vanilla JavaScript (Fetch API), Google Fonts |
| Notebooks | Jupyter, nbformat |
| QA | pytest, `scripts/check_gates.py` |

No database is required. Models, selected features and cleaned tables are local
artifacts; the API reads them at startup.

---

## 14. Artifacts

- Models: `models/best_model_S.joblib`, `models/xgb_final_S.json`,
  `models/selected_features_S.json`, `models/train_drugs_S.json`,
  `models/food_markers.json`.
- Provenance: `data/processed/evidence_S.parquet`, `models/rules_v1.json`,
  `models/model_metadata_S.json`, `reports/dataset_audit.{json,md}`.
- Calibration and domain: `models/calibration_S.json`, `models/domain_S.json`,
  `reports/calibration_S.{json,md}`.
- Generalization, baselines and deployment: `reports/generalization_S.*`,
  `reports/baselines_S.*`, `reports/system_comparison_S.csv`,
  `reports/class_balance_sweep.{csv,md}`, `reports/single_model_evaluation.{json,md}`.
- Metrics: `reports/metrics_S.json`, `reports/model_comparison.*`,
  `reports/cv_results_S.csv`, `reports/tuning_*_S.csv`.
- Explainability: `figures/shap_summary_S_*`, `reports/brick10_shap_S.*`.
- Cases: `reports/brick11_case_study.json/md`.
- Enhancements: `reports/brick12_enhancements.json/md`.
- Stage CSV exports: `data/processed/csv/` contains `drug_cleaned.csv` (1,359 rows),
  `food_cleaned.csv` (5,022), `pairs.csv` (21,042), `features_generation.csv` (6,236 × 63)
  and `feature_extraction.csv` (20,947 × 18), indexed by `manifest.json`.
- Narrative: `reports/report.md`, `reports/retune_audit.md`,
  `reports/implementation_audit.md` (pre-change audit),
  `reports/implementation_changes.md` (before/after change log) and
  `reports/deletion_manifest.md` (what the mode retirement removed).

---

## 15. Run and reproduce

```bash
# from dfi/
pip install -r requirements.txt

# full pipeline (long)
bash scripts/run_all.sh

# individual stages
PYTHONPATH=src python src/brick8_models.py --mode S
PYTHONPATH=src python src/brick8b_compare.py --mode S
PYTHONPATH=src python src/brick17_single_model.py --ratios 1,2,3 --deploy-ratio 2 --apply
PYTHONPATH=src python scripts/export_stage_csvs.py

# web app
python -m uvicorn web.app:app --host 127.0.0.1 --port 8010
# open http://127.0.0.1:8010
```

Name resolution calls Gemini and PubChem only when a name is not already in the local
database or cache. Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and start the
app; the loader reads `dfi/.env` at startup and shell variables win over file values.
`.env` is git-ignored and never read from the repository. `DFI_ONLINE_RESOLVER=0`
forces fully offline behavior.

On Windows, if port 8010 is busy:

```bash
netstat -ano | findstr :8010
taskkill /PID <pid> /F
```

---

## 16. Testing and gates

- `scripts/check_gates.py` checks the artifact and report conditions: data funnels,
  label counts, feature verification, split leakage, model reload equality, the single
  served split, the class-ratio sweep and deployed calibration, stage CSV exports,
  evaluation artifacts, SHAP figures, case study, enhancements, report and notebooks.
  It currently reports **29/29 gates passed**.
- `tests/` contains 38 tests:
  - feature contract and 3,780-feature arithmetic;
  - rule registry versioning, precedence and original matching semantics;
  - temperature scaling, 1-D/2-D handling and missing-artifact fallback;
  - applicability-domain thresholds, train-only references and an outside-domain case;
  - double-cold partition disjointness;
  - Gemini/PubChem name resolution offline: response parsing, invalid-SMILES rejection,
    cache reuse without network, no-key behavior and overlong names;
  - serving integration over the real app: rule, dataset-evidence, accepted ML,
    uncertain ML, outside-AD, inorganic, unknown drug/food and invalid requests;
  - schema and source-consistency checks across all paths, including `resolution`;
  - single-model calibration, SHAP shape (18 features × 3 classes) and feature parity
    between the selected-feature artifact and the served model.
- Run with `python -m pytest -q tests` (the new `tests/conftest.py` sets the import paths
  and forces offline resolution).
- Manual smoke tests: warfarin + grapefruit juice → rule; simvastatin + Cyanidin →
  dataset evidence with no confidence; acetaminophen + glucose → calibrated ML answer
  with per-class SHAP; warfarin + perfluorooctane SMILES → outside-domain abstention.

---

## 17. Known limitations and safety

### 17.1 Labels and class 1

- The positive funnel is extremely thin: 1,822 source rows annotated `Positive` collapse
  to 24 unique drug–compound pairs across 14 drugs (646 rows → class 1, 1,084 →
  class 0, 92 → class 2). Resampling cannot create new biology.
- Class 1 support is 0–3 pairs in every test partition (1 in Split G, 2 in Split F, 0 in
  Split D), so class-1 precision/recall/F1 are not meaningful. The reports print support
  beside every metric so this is visible.
- Two thirds of the 24 positive pairs come from `positive`-annotated sentences and one
  third from `possible` sentences; the audit is in `reports/dataset_audit.md`.

### 17.2 What the deployed model actually achieves

- Split G (unseen drugs, scored once after the class ratio and threshold were frozen):
  accuracy 0.7269, balanced accuracy 0.7428, macro-F1 0.4865, MCC 0.2408, avoid-class
  recall 0.3937. The majority-class baseline has higher raw accuracy (0.7552) but
  balanced accuracy 0.3333 and macro-F1 0.2868, so the model carries real signal while
  still missing most avoid-class cases.
- The deployed cap 2:1 lowered validation macro-F1 from 0.4465 (uncapped) to 0.4304 and
  raised validation avoid recall from 0.312 to 0.402. That is a deliberate safety
  trade-off; it is not presented as a metric improvement.
- Split R (pair-random, same drugs on both sides) is much easier: accuracy 0.7771,
  macro-F1 0.5633. It is a diagnostic, not the headline.
- Unseen compounds (F): accuracy 0.7593, macro-F1 0.6296 (class-1 support 2).
- Double-cold (D): accuracy 0.6760 on 321 rows, MCC −0.0734, class-1 support 0. This is
  a genuine warning that combined drug+compound novelty is not solved.
- The external validation set used by the source paper was not obtained; numbers are
  not directly comparable to its 0.9756 accuracy.

### 17.3 Mechanism coverage and evaluation integrity

- The curated rule layer covers **0%** of the Split G test partition, so the hybrid path
  equals ML-only there. Rules are safety scaffolding, not a source of measured accuracy.
- Exact-pair evidence lookup covers 46.8% of test rows, but its 100% accuracy is a
  label echo. It is excluded from model comparison and reported only for coverage.
- The applicability domain is a similarity heuristic: a test drug with 0.20 similarity
  sits just above the 0.1856 threshold and is still scored. It is not a proof of
  chemical validity.
- Inorganic compounds and descriptor failures are refused rather than scored.
- The model cannot identify a named drug from symmetric pair descriptors, so clinical
  correctness for known pairs depends on the rule and evidence layers.
- Name resolution can fail for mixtures or ambiguous names. Gemini is instructed not to
  invent structures and every answer is RDKit-validated, but a wrong structure would
  silently change the prediction; the response therefore exposes the resolved SMILES
  and its source.

### 17.4 Safety

- Confidence is a calibrated model probability, not a guarantee of correctness.
- TreeSHAP explains model behaviour, not biological causation.
- This is a research tool, not a medical device. Confirm every result with a physician
  or pharmacist.
