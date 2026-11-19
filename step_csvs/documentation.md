# Documentation

This project is a research drug-food interaction predictor. It is not clinical
advice. The deployed artifact is **LightGBM single-2026-11-16**
(historical artifact name `S`); there is one served model, not a paper-style Mode P model.

## 1. ML

### 1.1 Exact pipeline and file names

| # | Stage | File | What it does |
|---:|---|---|---|
| 1 | Raw audit | `src/brick1_audit.py` | Reads the three raw CSVs, records row/column counts and effect counts. |
| 2 | Drug cleaning | `src/brick2_clean_drugs.py` | Parses structures with RDKit, removes empty/failed structures and duplicate InChIKey14 values; writes `drugs_clean.parquet`. |
| 3 | Food cleaning | `src/brick3_clean_foods.py` | Corrects the swapped food columns, parses compounds, removes InChIKey duplicates, removes Morgan/Tanimoto near-duplicates at 0.75, protects marker compounds, flags organic/inorganic; writes `foods_clean.parquet`. |
| 4 | Labels | `src/brick4_labels.py` | Maps the raw `Effect` values to classes 0/1/2, applies negative precedence, builds Mode S pairs, creates weak labels, holds out 23 case-study pairs; writes `splits_S.parquet` inputs. |
| 5 | MOE descriptors | `src/brick5_features.py` | Computes the 60 PyBioMed MOE descriptors for 1,359 drugs and 4,877 organic foods; writes `drug_moe.parquet` and `food_moe.parquet`. |
| 6 | Pair features | `src/features.py` | Builds 3,780 pair features: 3,600 cross-products, 60 sums, 60 absolute differences, 60 ratios. |
| 7 | Splits | `src/brick6_splits.py` | Creates Split R (pair-random) and Split G (whole-drug groups), plus the F and D cold partitions used by brick 15. |
| 8 | Feature selection | `src/brick7_selection.py --mode S` | Variance threshold -> Pearson correlation -> Lasso -> RFECV -> RFE to 18 features on Split G training rows. |
| 9 | Model tuning | `src/brick8_models.py --mode S` | Tunes five model families on Split G training data and records the LightGBM parameter set later used for deployment. |
| 10 | Generalization evaluation | `src/brick15_generalization.py --mode S --cap-ratio 2` | Trains the same fixed LightGBM recipe on R/G/F/D partitions and scores every test partition once. |
| 11 | Domain + calibration | `src/brick14_domain_calibration.py --mode S` | Fits the Morgan/Tanimoto applicability domain and temperature scaling on validation data. |
| 12 | Single deployed model | `src/brick17_single_model.py --ratios 1,2,3 --deploy-ratio 2 --apply` | Sweeps class-2 caps, refits the tuned LightGBM on the 2:1 capped training rows, rewrites calibration for the deployed artifact, and saves `best_model_S.joblib`. |
| 13 | Evidence and rules | `src/brick13_build_evidence.py` | Builds the 9,525-row evidence lookup, serving metadata, and the versioned rule document. |
| 14 | Baselines | `src/brick16_baselines.py --mode S` | Scores majority, logistic, and LightGBM imbalance baselines on the same Split G test rows. |
| 15 | Serving explanations | `src/explain.py` | Computes TreeSHAP values for all 18 features and all 3 classes per request. |
| 16 | Deliverables | `scripts/export_stage_csvs.py`, `scripts/build_deliverables.py` | Writes the stage CSVs, `step_csvs/`, and `final_metrics/`. |

The orchestrator is `scripts/run_all.sh`. It runs the stages in the order above (with the
report-only experiments) and ends with `scripts/check_gates.py`.

### 1.2 Techniques actually used

- RDKit parsing, canonical SMILES, InChIKey14 duplicate removal.
- Morgan fingerprints (radius 2, 2048 bits) and Tanimoto similarity for food near-duplicate removal and for both sides of the applicability domain.
- PyBioMed MOE descriptors (60 per molecule).
- Pairwise descriptor arithmetic: cross-products, sums, absolute differences, ratios.
- `VarianceThreshold(0.8)`, Pearson correlation pruning at |r| < 0.75, Lasso (alpha 0.01), RFECV, and RFE to 18 features.
- LightGBM gradient-boosted decision trees (GBDT).
- Balanced class weights via `compute_sample_weight('balanced', y)`.
- Weak-label downweighting (weak rows multiplied by 0.5).
- Training-only class-2 capping at 2:1.
- Group-aware and cold-start splits (pair-random, whole-drug, whole-compound, double-cold).
- Temperature scaling for probability calibration.
- Selective prediction with a validation-selected acceptance threshold (0.77).
- Applicability domain from maximum Tanimoto similarity to training-only references.
- TreeSHAP for per-request feature contributions.
- Comparison baselines: majority class, logistic regression, Random Forest, Extra Trees, XGBoost, and MLP. Only LightGBM is served.

### 1.3 Techniques not used

- No SMOTE, ADASYN, or random oversampling in the deployed model. The oversampled baseline exists only in `reports/baselines_S.csv`.
- No XGBoost, Random Forest, Extra Trees, or MLP in serving. They are comparison rows only.
- No deep learning, graph neural network, or pretrained molecular embedding.
- No external validation dataset. `reports/brick6_splits.json` records that the external Ryu et al. set was not obtained; Split G/F/D and the held-out case studies stand in.
- No calibration method other than temperature scaling.
- No conformal prediction in serving; the E2 conformal experiment is reported for completeness in `reports/brick12_enhancements.json` and is not used by the app.
- No uncertainty decomposition.

### 1.4 Final deployed model and configuration

| Setting | Value |
|---|---|
| Version | `single-2026-11-16` |
| Artifact | `models/best_model_S.joblib` |
| SHA-256 prefix | `c8d4f9dadf2d63b2` |
| Estimator | `lightgbm.sklearn.LGBMClassifier` inside `brick8_models.BalancedClassifier` |
| Features | 18 selected pair features |
| Labels | 0 = Avoid together, 1 = Beneficial / take with food, 2 = No significant interaction |
| Training partition | Split G training rows (unseen-drug split), class-2 capped 2:1 |
| Training rows after cap | 6,839 (2,277 / 8 / 4,554) |
| Sample weights | balanced class weights x 0.5 for weak rows |
| Calibration | temperature scaling, T = 1.3405 |
| Accept threshold | 0.77 |
| Drug AD threshold | 0.1856 (223 references) |
| Food AD threshold | 0.3333 (2,291 references) |

Exact `LGBMClassifier` parameters loaded from the deployed artifact:

| Parameter | Value |
|---|---|
| `boosting_type` | `gbdt` |
| `n_estimators` | 200 |
| `learning_rate` | 0.02 |
| `num_leaves` | 127 |
| `max_depth` | -1 |
| `min_child_samples` | 20 |
| `min_child_weight` | 0.001 |
| `min_split_gain` | 0.0 |
| `subsample` | 1.0 |
| `subsample_freq` | 0 |
| `colsample_bytree` | 1.0 |
| `reg_alpha` | 0.0 |
| `reg_lambda` | 0.0 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| `verbose` | -1 |
| `objective` | `None` (inferred multiclass) |

### 1.5 Held-out scores

Split G test (unseen drugs, headline):

| Metric | Value |
|---|---:|
| Accuracy | 0.7269 |
| Balanced accuracy | 0.7428 |
| Macro F1 | 0.4865 |
| Weighted F1 | 0.7227 |
| MCC | 0.2408 |
| Cohen kappa | 0.2403 |
| ROC AUC (OvR macro) | 0.7670 |
| PR AUC (macro) | 0.4965 |
| Confusion matrix (rows=true) | `[[252, 4, 384], [0, 1, 0], [324, 3, 1650]]` |

Split R test (pair-random): accuracy 0.7771, balanced accuracy
0.6363, macro F1 0.5633, MCC 0.5037,
ROC AUC 0.9043.

Split F test (unseen compounds): accuracy 0.7593, balanced accuracy
0.6446, macro F1 0.6296, MCC 0.3995,
ROC AUC 0.8533.

Split D test (double-cold): accuracy 0.6760, balanced accuracy
0.4683, macro F1 0.4601, MCC -0.0734,
ROC AUC NA (no class-1 test pairs). Split D is a stress test with only 321 pairs
and must not be presented as the headline.

Split G is the deployed artifact `models/best_model_S.joblib`, and its confusion matrix is
reproduced exactly by `step_csvs/model_predictions_G.csv`. Splits R, F, and D are scored by
refitting the same fixed LightGBM recipe on each partition (see
`src/brick15_generalization.py`); they measure recipe generalization and are not additional
predictions of the deployed artifact.

Selective prediction at threshold 0.77 on Split G test: coverage
45.34% (1,187
of 2,618 pairs accepted), accuracy on accepted rows
81.72%.

Class 1 has only 24 labelled pairs in the entire dataset and
8 in the capped training set. Every class-1 score is unstable.

## 2. Backend

### 2.1 Framework and runtime

| Component | Choice |
|---|---|
| Web framework | FastAPI 0.111.0 |
| ASGI server | Uvicorn 0.29.0 |
| Validation/serialization | Pydantic 2.12.3 |
| Model persistence | joblib |
| Chemistry | RDKit |
| Name resolution | local database, Google Gemini `generateContent`, PubChem PUG REST |
| Entry point | `web/app.py` (`app = FastAPI(...)`) |
| Serving logic | `src/recommend.py` (`DFIRecommender`) |
| Name resolution | `src/resolve.py` (`NameResolver`) |
| Curated rules | `src/rules.py` (version `2026.11.1`) |
| Applicability domain | `src/domain.py` |
| Per-request SHAP | `src/explain.py` |
| Environment loader | `src/env.py` (reads `dfi/.env`, shell values win) |

Run from `dfi/`:

```bash
uvicorn web.app:app --port 8000
```

`GEMINI_API_KEY` is read from the environment or `dfi/.env`. The key is never written to the
repository or returned by an endpoint.

### 2.2 API endpoints

| Method | Path | Purpose | Input / output |
|---|---|---|---|
| GET | `/` | Landing page | `landing.html` |
| GET | `/predict` | Predictor page | `index.html` |
| GET | `/about` | Method and provenance page | `about.html` |
| GET | `/static/*` | CSS, JS, images | static files |
| GET | `/figures/{name}` | Serve a report PNG by bare filename | `.png` filenames only; other names return 404 |
| GET | `/api/health` | Readiness and served-model status | drug/compound counts, thresholds, model metadata, resolution status, rules version |
| GET | `/api/meta` | Full method metadata | 18 feature names, rule document, AD thresholds/reference counts, abstention info |
| GET | `/api/examples` | Eight example pairs for the UI chips | list of `{drug, food, note}` |
| GET | `/api/suggest?kind=&q=&limit=` | Autocomplete; `kind=drug` or `kind=food` | prefix matches first, then substring matches |
| POST | `/api/predict` | Score one drug-food pair | `{"drug": "...", "food": "..."}`; returns prediction/provenance/explanation plus `inputs` and `latency_ms` |

`POST /api/predict` requires both fields (1-300 characters). Validation errors return 422; missing
model artifacts return 503. A successful response is one of:

| Status | Source | Meaning |
|---|---|---|
| `ok` | `ml_model` | Model probability is at or above the 0.77 acceptance threshold. |
| `uncertain` | `ml_model` | Model confidence is below 0.77. The most likely class is still shown; calibrated confidence and the abstention reason stay in the details. |
| `evidence` | `dataset_evidence` | Exact pair has a saved dataset annotation; this is a lookup, not an ML prediction. A supporting TreeSHAP block is included for the same pair. |
| `rule` | `rule_based` | A curated rule matched before the model was called. A supporting TreeSHAP block is included for the matched compound. |
| `not_supported` | `none` | Unknown structure, descriptor failure, outside applicability domain, or an inorganic compound with no dataset record and no food-level fallback. Whenever a drug/food structure pair exists (outside-domain, inorganic), a diagnostic TreeSHAP block is still returned. |

### 2.3 What `POST /api/predict` does, step by step

1. Pydantic validates the JSON body and trims both strings.
2. `engine()` returns the singleton `DFIRecommender`, which loads the model, selected features, scaler, domain, rules, evidence, and name resolver once. A background thread warms the SHAP explainer.
3. Drug resolution: exact normalized local name first; otherwise `NameResolver` (Gemini, then PubChem), then a raw SMILES input. A local hit makes no network call.
4. The curated rule layer is evaluated on the drug name and canonical food key. A match returns `status=rule` and `prediction_source=rule_based`; local food resolution still runs (`resolve_remote=False`) so the matched compound can carry a supporting TreeSHAP block.
5. Food resolution: exact compound, built-in structure, marker compounds, or food-level links; unknown names go to `NameResolver` (Gemini, then PubChem). A user SMILES is used directly. Food names combine the configured marker compounds with the specific organic compounds linked to that food (for example `milk` returns calcium plus the compounds of Milk (Cow), Soy Milk, 558 unique compounds, capped at 12 candidates).
6. Exact-pair evidence is checked first. A saved non-weak annotation returns `status=evidence` and is explicitly labelled a dataset lookup, even when the compound is an inorganic mineral (53 calcium rows are reachable this way). The served model still computes TreeSHAP for the same pair and returns it as `model_support`.
7. For each remaining candidate, `_score_pair` computes the 60 MOE descriptors. An inorganic compound with no dataset record and no food-linked organic fallback is returned as `status=not_supported` (no class is claimed) but still carries a diagnostic TreeSHAP block. A mineral with no dataset record falls back to the organic compounds of the foods that contain it, labelled as a food-level match.
8. Applicability domain: Morgan radius 2 / 2048-bit maximum Tanimoto to training-only drugs and foods. Outside -> `status=not_supported` with reason `outside_applicability_domain`.
9. The 18 selected pair features are built in the same order as training, the calibrated LightGBM `predict_proba` is called, and temperature scaling is applied.
10. If the top calibrated probability is below 0.77, the result is `status=uncertain` with `abstained=true`; the UI still shows the most likely class, with the confidence and abstention reason in the details. Otherwise the result is `status=ok`.
11. TreeSHAP computes signed contributions for all 18 features and all 3 classes on every answered path. ML answers use it as the decision explanation; rule and evidence answers expose it through `model_support`; outside-domain and inorganic abstentions keep it as a diagnostic. Every response reports `shap_available` and, when false, `shap_reason`; only inputs with no featurizable structure lack SHAP.
12. If food-level matching produced several candidate compounds, the most cautious confident result wins (evidence-avoid first, then confident avoid, then evidence, then highest confidence); the candidate list is attached, and each candidate carries its own label, severity and full TreeSHAP block.
13. The response includes resolution provenance, AD values, calibration info, model version, probabilities, `model_support` when the answer is a rule/evidence/diagnostic, and a research disclaimer. `web/app.py` adds `probability_rows`, `label`, `severity`, `inputs`, and `latency_ms`.

### 2.4 Backend files

- `web/app.py` - routes, request model, singleton engine, response augmentation.
- `src/recommend.py` - resolution order, rule layer, evidence lookup, AD, scoring, abstention, SHAP.
- `src/resolve.py` - Gemini/PubChem resolution, canonicalization, JSON cache.
- `src/domain.py` - Morgan/Tanimoto applicability domain.
- `src/explain.py` - TreeSHAP explainer.
- `src/rules.py` - versioned curated rules.
- `src/env.py` - `.env` loader with shell precedence.
- `models/best_model_S.joblib`, `models/calibration_S.json`, `models/domain_S.json`, `models/selected_features_S.json`, `models/model_metadata_S.json` - serving artifacts.

## 3. Frontend

### 3.1 Tech stack

- HTML5 pages: `web/static/landing.html`, `web/static/index.html`, `web/static/about.html`.
- Hand-written CSS in `web/static/styles.css` (no framework, no build step).
- Vanilla JavaScript (ES2020) in `web/static/app.js`; Fetch API and `AbortController`.
- Native `<datalist>` autocomplete; inline SVG icons; no external runtime dependency.
- The frontend never computes a prediction. Every score, probability, evidence record, and SHAP value comes from `POST /api/predict`.

### 3.2 Pages

| Page | Route | Role |
|---|---|---|
| Landing | `/` | Explains the project and links to the predictor and method page. |
| Predictor | `/predict` | Two inputs (drug and food, name or SMILES), example chips, result panel. |
| About | `/about` | Method, provenance, limitations, and served-model explanation. |

### 3.3 End-to-end flow after clicking "Predict interaction"

1. The form `submit` handler prevents the browser's default navigation and calls `runPrediction()` (`web/static/app.js`).
2. `runPrediction()` trims both inputs. If either is empty, it writes "Enter both a drug and a food compound." and focuses the missing field; no request is sent.
3. Any earlier in-flight request is aborted with `AbortController`. A new controller is stored, the button is disabled, its label becomes "Scoring...", and a skeleton loader replaces the result panel.
4. The browser sends `POST /api/predict` with `Content-Type: application/json` and `{"drug": "...", "food": "..."}`.
5. FastAPI validates the body. The singleton recommender resolves the drug, checks curated rules, resolves the food (exact compound, markers plus food-linked compounds, then Gemini/PubChem), checks exact-pair evidence, checks the applicability domain, computes descriptors and the 18 pair features, runs the calibrated LightGBM model, applies the 0.77 threshold, and computes TreeSHAP for all features and classes. Mineral queries such as `calcium` and food-level queries such as `milk` therefore return a result instead of an empty state.
6. The server returns JSON with `status`, `prediction_source`, `predicted_class`/`predicted_label`, `sentence`, `reason`, `probability_rows` (if probabilities exist), `confidence`, `applicability_domain`, `calibration`, `explainability`, `resolution`, `inputs`, `latency_ms`, and `disclaimer`.
7. If `res.ok` is false, `renderError()` shows the `detail` message. A network failure shows the "API is unreachable" message with the Uvicorn command.
8. If the request succeeds, `renderResult()` chooses the badge and severity:
   - `ok` + `ml_model` -> class badge and "ML model" source tag.
   - `rule` -> class badge with "Curated rule" source tag.
   - `evidence` -> class badge with "Dataset evidence" source tag.
   - `uncertain` -> class badge with the "ML model" source tag; calibrated confidence and the abstention reason appear in the details.
   - `not_supported` -> `renderUnsupported()` with the abstention reason, AD values, and resolution details.
9. The verdict shows the human sentence with the matched drug and food highlighted, then the reason. When probabilities exist, three bars animate to class 0/1/2 percentages.
10. The details block shows matched names, prediction source, model name/version, training split, AD status and thresholds, calibration temperature, rule ID/provenance, evidence record/annotation/PMID/DOI, resolution source and SMILES, abstention reason, original inputs, and any food-level candidate list.
11. SHAP rendering shows every feature for the predicted class and collapsible blocks for the other two classes, for `ok`, `uncertain`, `rule`, `evidence`, and outside-domain abstentions. Rule and evidence results also print the `model_support` note and the explained compound. Each row is a signed contribution bar with the numeric value; the caveat explains that these are model contributions, not causal effects.
12. The disclaimer strip is always appended ("Research tool; confirm with a physician or pharmacist.").
13. The `finally` block re-enables the button, restores "Predict interaction", clears the loading state, and removes `aria-busy`.
14. On page load, `loadHealth()` calls `/api/health` for the ready pill and `loadExamples()` calls `/api/examples` for the chips. Typing in either field debounces 150 ms, aborts the previous suggestion request, and calls `/api/suggest` for up to 8 options.

### 3.4 Safety and presentation details

- User-visible text is inserted with `textContent`; only fixed SVG markup uses `innerHTML`.
- Result severity is driven by `status` and class, not by colour alone; badges carry text labels.
- The UI always shows the prediction source, so a curated rule or dataset lookup is never presented as an ML prediction.
- Abstention is a first-class result, not an error.
