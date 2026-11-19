# Pipeline context: rows at the start and after every step

All numbers in this file are read from the saved artifacts in `data/`, `models/`, and
`reports/`; none are estimated. The deployed model is **LightGBM
single-2026-11-16** (historical artifact name `S`). There is no Mode P model.

## 1. Initial datasets (raw inputs)

| Dataset | File | Rows | Columns |
|---|---|---:|---:|
| Drug table | `Druginfo.csv` | 1,516 | 17 |
| Drug-food interactions | `InteractionInformation.csv` | 23,950 | 17 |
| Food-compound links | `food_fhdi_clean.csv` | 838,492 | 11 |

The raw drug table has **1,437** drugs with a Canonical SMILES.
The raw interactions table covers **1,507** drugs,
**1,336** food/herb identifiers, and
**253** named components. The raw food-link table covers
**177** foods and **13,282**
unique compounds.

Raw `Effect` values (row counts):

| Effect | Rows |
|---|---:|
| Possible | 17,827 |
| No Effect | 2,374 |
| Positive | 1,915 |
| Negative | 1,437 |
| Harmful | 393 |
| possible (lower-case) | 4 |
| **Total** | **23,950** |

## 2. Drug cleaning (`src/brick2_clean_drugs.py`)

| Step | Drugs |
|---|---:|
| Loaded from `Druginfo.csv` | 1,516 |
| Parse failures or empty structure | -79 |
| Parsed with RDKit | 1,437 |
| Duplicate InChIKey14 removed | -78 |
| **Final cleaned drugs** | **1,359** |

## 3. Food cleaning (`src/brick3_clean_foods.py`)

| Step | Compounds | Links |
|---|---:|---:|
| Raw food-compound rows | 13,282 unique | 838,492 |
| No SMILES | -1 | |
| RDKit parse failures | -8 | |
| Parsed unique compounds | 13,273 | |
| Duplicate InChIKey removed | -309 | |
| Morgan/Tanimoto near-duplicates removed at 0.75 | -7,942 | |
| **Final cleaned compounds** | **5,022** | **152,320** |

The final compound table is **4,877 organic** and
**145 inorganic**. Training uses only the organic compounds, so
the model never learns from minerals. Serving still answers for them: the exact-pair dataset
annotation is checked first, a mineral query with no annotation falls back to the foods that
contain it, and a mineral with no fallback is returned as `not_supported` with a diagnostic
TreeSHAP block when its descriptors compute (see section 15).
Marker names configured in `config.yaml` (ethanol, tyramine, caffeine, bergamottin, naringin,
hyperforin, phylloquinone, vitamin K1, dimethyl disulfide) are protected from near-duplicate
removal; 7 protected compounds are present in the final set.

## 4. Mode S labels (`src/brick4_labels.py`)

| Quantity | Value |
|---|---:|
| Labelled Mode S pairs | 21,042 |
| Unique drugs | 444 |
| Unique compounds | 4,572 |
| Class 0 - Avoid | 4,898 |
| Class 1 - Beneficial | 24 |
| Class 2 - No significant interaction | 16,120 |
| Weak class-2 pairs | 11,517 |
| Non-weak class-2 pairs | 4,603 |
| Case-study pairs held out of training | 23 |

Class 1 is tiny (**24** pairs). This is a property of the source data, not a
modelling choice; every class-1 metric in this project must be read as unstable.

## 5. Descriptors (`src/brick5_features.py`)

| Quantity | Value |
|---|---:|
| Drug molecules with MOE vectors | 1,359 |
| Food molecules with MOE vectors | 4,877 |
| Total descriptor rows | 6,236 |
| MOE descriptors per molecule | 60 |
| Pair features generated | 3,780 |
| Mode S pairs featurized | 20,947 |
| Pairs dropped (inorganic food has no MOE vector) | 95 |

The 3,780 pair features are 3,600 cross-products, 60 same-descriptor
sums, 60 absolute differences, and 60 ratios.

## 6. Feature selection (`src/brick7_selection.py --mode S`)

| Step | Remaining features | Setting |
|---|---:|---|
| Start | 3,780 | 60 descriptors x pairwise arithmetic |
| Variance threshold | 3,521 | threshold 0.8 |
| Pearson correlation filter | 1,126 | absolute Pearson r >= 0.75 removed |
| Lasso | 174 | alpha 0.01 |
| RFECV | 89 | 5-fold CV |
| Final RFE | 18 | target 18 features |

Selection used only Split G training rows; the fitted scaler used by Lasso was fit inside the
selection step. The 18 features were not reselected after the class-cap retrain.

## 7. Splits (`src/brick6_splits.py`)

Split assignments cover all **21,042** pairs. The model is trained and evaluated only on
the **20,947** pairs that have both descriptor vectors.

| Split | Meaning | Train assigned | Val assigned | Test assigned | Train featurized | Val featurized | Test featurized |
|---|---|---:|---:|---:|---:|---:|---:|
| R | pair-random | 10,537 | 7,891 | 2,614 | 10,488 | 7,862 | 2,597 |
| G | unseen drugs | 10,537 | 7,872 | 2,633 | 10,495 | 7,834 | 2,618 |
| F | unseen compounds | 10,537 | 7,891 | 2,614 | - | 7,837 | 2,613 |
| D | double-cold | 5,259 | 2,937 | 321 | - | 2,920 | 321 |

The headline evaluation is **Split G test** (unseen drugs). Split R is the optimistic
pair-random reference. Split F tests unseen food compounds. Split D is the strictest
double-cold check and has only 321 test pairs, including no class-1 pairs.

## 8. Training and deployed model (`src/brick17_single_model.py`)

| Quantity | Value |
|---|---:|
| Model | LightGBM single-2026-11-16 |
| Algorithm | LightGBM gradient-boosted trees (`boosting_type=gbdt`) |
| Feature count | 18 |
| Split G training rows before capping | 10,495 |
| Split G training rows after class-2 cap 2:1 | **6,839** |
| Capped training counts (0 / 1 / 2) | 2,277 / 8 / 4,554 |
| Class weights | `compute_sample_weight('balanced', y)` |
| Weak-row weight | 0.5 |
| Validation rows (uncapped distribution) | 7,834 |
| Test rows (uncapped distribution) | 2,618 |
| Artifact | `models/best_model_S.joblib` |
| SHA-256 prefix | `c8d4f9dadf2d63b2` |

The class-2 cap is applied to **training rows only**. Validation and test keep the real
distribution. The deployed cap is 2:1 (2,277 class-0, 8 class-1, 4,554 class-2 rows), a
safety-oriented choice; it is not the best validation-macro-F1 recipe (uncapped 0.4465 vs
cap 2:1 0.4304).

## 9. Calibration and abstention (`src/brick14_domain_calibration.py`, rewritten by Brick 17)

| Quantity | Value |
|---|---:|
| Calibration method | temperature scaling |
| Temperature | 1.3405 |
| Fitted on | Split G validation (unseen drugs) |
| Acceptance threshold | 0.77 |
| Validation coverage at threshold | 38.78% |
| Test coverage at threshold | 45.34% |
| Test accepted rows | 1,187 |
| Test accuracy on accepted rows | 81.72% |
| Validation ECE raw -> calibrated | 0.0893 -> 0.0321 |
| Test ECE raw -> calibrated | 0.0713 -> 0.0363 |

## 10. Applicability domain (`src/brick14_domain_calibration.py`)

Morgan radius 2, 2048 bits, maximum Tanimoto similarity to training-only references.

| Side | Threshold | References | Rule |
|---|---:|---:|---|
| Drug | 0.1856 | 223 | 10th percentile of validation similarities for unseen drugs (Split G validation) |
| Food | 0.3333 | 2,291 | 10th percentile of validation similarities for unseen compounds (Split F validation) |

## 11. Tanimoto similarity on the food dataset

Tanimoto similarity is used twice on foods: near-duplicate removal in cleaning and the food
side of the applicability domain. This build recomputes it from all **5,022** cleaned compounds
with the same Morgan radius 2 / 2048-bit settings.

| Quantity | Value |
|---|---:|
| Pairwise comparisons | 12,607,731 |
| Mean pairwise similarity | 0.0821 |
| Max pairwise similarity in final cleaned set | 0.9024 |
| Pairs at or above the 0.75 cleaning cutoff | 3 |
| ... of which involve a protected marker compound | 3 |
| ... of which involve no protected compound | 0 |
| Pairs at or above 0.90 | 1 |
| Nearest-neighbour median | 0.6154 |
| Nearest-neighbour max | 0.9024 |
| Training food references | 2,291 |
| Unseen organic compounds | 2,592 |
| Unseen organic inside food domain | 2,361 |
| Unseen organic outside food domain | 231 |

The 3 pairs at or above 0.75 all involve compounds explicitly
protected in `config.yaml` (phylloquinone, naringin, bergamottin); the non-protected pair count
above the cutoff is **0**. This is the intended protected-marker
bypass, not an unnoticed duplicate: the cleaning removes near-duplicates at 0.75 but keeps the
marker compounds needed for known interaction cases. Detailed neighbours are in
`food_tanimoto_neighbors.csv`, and the per-compound domain result is in
`food_tanimoto_applicability.csv`.

## 12. Evidence, predictions, and evaluation rows

| Artifact | Rows |
|---|---:|
| Dataset evidence lookup (`evidence_S.csv`) | 9,525 |
| Split G test predictions from the deployed artifact (`model_predictions_G.csv`) | 2,618 |
| Split G test pairs | 2,618 |
| Split R test pairs | 2,597 |
| Split F test pairs | 2,613 |
| Split D test pairs | 321 |

## 13. Headline test scores (exact)

| Split | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | MCC | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| R pair-random | 0.7771 | 0.6363 | 0.5633 | 0.7917 | 0.5037 | 0.9043 | 0.5963 |
| G unseen drugs | 0.7269 | 0.7428 | 0.4865 | 0.7227 | 0.2408 | 0.7670 | 0.4965 |
| F unseen compounds | 0.7593 | 0.6446 | 0.6296 | 0.7691 | 0.3995 | 0.8533 | 0.6595 |
| D double-cold | 0.6760 | 0.4683 | 0.4601 | 0.6526 | -0.0734 | NA | 0.4961 |

Split G is the deployed artifact `models/best_model_S.joblib` scored once on unseen
drugs. Split R, F, and D scores come from refitting the same fixed LightGBM recipe on each
partition (the evaluation-stage protocol); they measure the recipe's generalization and are
not additional predictions from the deployed artifact. `model_predictions_G.csv` therefore
contains only Split G rows, and no Split R prediction file is shipped because the Brick 9
R parquet comes from a separately retrained R model.

## 14. File index

| File | What it contains |
|---|---|
| `drug_cleaned.csv` | 1,359 cleaned drugs |
| `food_cleaned.csv` | 5,022 cleaned food compounds |
| `pairs.csv` | 21,042 labelled Mode S pairs with names and splits |
| `features_generation.csv` | 6,236 molecules x 60 MOE descriptors |
| `feature_extraction.csv` | 20,947 pairs x 18 selected pair features |
| `feature_selection_funnel.csv` | 3,780 -> 18 feature-selection funnel |
| `feature_selection.csv` | 18 selected features with meanings |
| `split_assignments.csv` | Every pair with Split R and Split G assignment |
| `split_counts.csv` | Row/class counts per split and partition |
| `model_predictions_G.csv` | 2,618 Split G test rows scored by the deployed artifact (calibrated and raw probabilities) |
| `evidence_S.csv` | 9,525 dataset evidence records |
| `calibration_S.csv` | Raw/calibrated NLL, Brier, ECE |
| `applicability_domain_S.csv` | Drug/food AD thresholds and references |
| `food_tanimoto_neighbors.csv` | Nearest Tanimoto neighbour for every food compound |
| `food_tanimoto_applicability.csv` | Food similarity to training references and AD status |
| `food_tanimoto_summary.csv` | Food Tanimoto summary statistics |
| `class_balance_sweep.csv` | Four class-ratio recipes on validation |
| `model_comparison_S.csv` | Five model families on Split G and R |
| `baselines_S.csv` | Majority/logistic/LightGBM imbalance baselines |
| `raw_dataset_counts.csv` | Initial raw row/column/effect counts |
| `pipeline_counts.csv` | Input/output rows for every pipeline stage |

## 15. Serving-time resolution (`src/recommend.py`)

| Behaviour | Detail |
|---|---|
| Evidence order | The exact-pair dataset annotation is checked before descriptor and inorganic checks. Calcium pairs with a saved record therefore return `status=evidence` instead of an inorganic abstention. |
| Calcium evidence | 53 non-weak dataset rows for compound 3514 (`Calcium`, `[Ca+2]`). |
| Milk resolution | `milk` returns the marker compound calcium plus the organic compounds linked to Milk (Cow), Soy Milk (612 link rows, 558 unique compounds); candidates are capped at 12. |
| Mineral fallback | An inorganic compound with no dataset record falls back to the organic compounds those foods contain (frequency <= 20 foods, capped at 10). The result is labelled as a food-level match. |
| Candidate limits | `config.yaml` `serving` sets the frequency cutoff (20), marker cap (12), food cap (10), mineral cap (10), marker-name match cap (3), food-name aliases (9), marker-compound fallbacks (18 foods), built-in structures (2), and the stripped food suffixes (7); the prediction path contains no food-specific code branches. |
| Low-confidence display | A result below the 0.77 threshold keeps its predicted class label in the UI; calibrated confidence and the abstention reason remain in the response and the details block. |
| SHAP coverage | Every answered result carries a TreeSHAP block with all 18 features and all 3 classes. Rule and dataset-evidence answers compute it for the matched compound and expose it through `model_support` (the answer itself still comes from the rule or annotation). Outside-domain and inorganic abstentions carry a diagnostic SHAP block. Food-level candidates each carry their own SHAP block. `shap_available`/`shap_reason` state the outcome on every response; only inputs with no resolvable structure on either side lack SHAP. |
| Config-driven display | Class labels, severities, response sentences, the disclaimer and the demo pairs come from `config.yaml` (`labels:`, `web:`). The browser renders the API's `predicted_label`; it keeps no class-name map of its own. |
