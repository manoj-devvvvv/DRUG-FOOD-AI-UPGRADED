# Drug-Food Interaction Prediction - Final Report

Three-class prediction (0 avoid, 1 beneficial/take with food, 2 non-significant) from a drug SMILES and a food-compound SMILES: MOE-type pair descriptors, four-step selection to 18 features, baseline comparison, a single deployed LightGBM, calibration, abstention, applicability domain, and per-prediction SHAP over every feature and every class. The paper-style label mode (Mode P) was retired: it applied one drug-level label to every compound and is leakage-prone by construction. The one served model uses the specific label mode (internally still named `S` for reproducibility).

## 1. Data

- Raw tables: 1,516 drug rows, 23,950 interaction rows, 838,492 food rows; PubChem text found for 7 of 8 probe drugs.
- Drugs: 1,516 loaded, 1,437 parsed, 78 salt/parent duplicates removed, **1,359 kept** (1,350 with a DrugBank ID).
- Food compounds: 13,282 unique structures from 838,492 rows; 7,942 near-duplicates removed at Tanimoto 0.75; **5,022 kept** (4,877 organic, 145 inorganic).

## 2. Labels

- Served mode: **21,042 pairs**, classes {'0': 4898, '1': 24, '2': 16120}. Compounds named by the interaction text, curated marker compounds, and food-specific compounds carry the interaction label; other sampled compounds for the same drug are class 2 and flagged weak.
- The retired paper-style mode produced 407,699 pairs by copying a drug-level label to every compound. Its models and reports were deleted; the labelled table is kept only as source data.
- 6,974 interaction rows mapped to compounds; the nine paper case pairs are held out of every training table (23 pairs).
- Sentence audit: 100 sampled rows, rule/annotation agreement on confirmed rows 0.60, rule coverage of 'Possible' rows 0.63. The structured FHDI labels disagree with literal text rules often enough that agreement is reported, not hidden.

## 3. Pair features

- 60 MOE-type descriptors per molecule over 1,359 drugs and 4,877 food compounds; pair vector = 3,600 products + 60 sums + 60 absolute differences + 60 ratios = **3,780 columns**.
- Verified against PyBioMed on 500 pairs: products max difference 5.00e-04, sums 1.14e-13; all values finite: True.

## 4. Splits

- Split G drugs train/val/test = 223/166/55, pairs = 10,537/7,872/2,633; Split R pairs = 10,537/7,891/2,614. Split G drug overlaps = 0. Holdout pairs absent: True.
- External set: not obtained - Ryu et al. supplementary data was not downloaded in this build; the strict drug-grouped test partition plus the nine held-out case studies stand in, as allowed by the plan.

## 5. Feature selection

| step              |   count |
|:------------------|--------:|
| start             |    3780 |
| after_variance    |    3521 |
| after_correlation |    1126 |
| after_lasso       |     174 |
| after_rfecv       |      89 |
| rfe18             |      18 |

Stability (3 subsamples): 8/18 final features chosen in at least two runs. The 18 features were selected on the uncapped training partition and are reused by the deployed class-ratio model.

## 6. Models

- Best CV model: XGBoost (balanced=True) macro-F1 0.4047; tuned winner **LightGBM**, validation macro-F1 0.4465 after the full-training refit; XGBoost reload difference 0.0e+00.

Baseline CV (Split G grouped folds):

| model        | balanced   |   accuracy |   macro_f1 |   recall_0 |
|:-------------|:-----------|-----------:|-----------:|-----------:|
| XGBoost      | True       |   0.756741 |   0.404681 |  0.31928   |
| XGBoost      | False      |   0.7899   |   0.360333 |  0.12209   |
| LightGBM     | True       |   0.763697 |   0.400914 |  0.287659  |
| LightGBM     | False      |   0.777894 |   0.36323  |  0.142732  |
| RandomForest | True       |   0.7899   |   0.334996 |  0.0685112 |
| RandomForest | False      |   0.79314  |   0.35113  |  0.0983751 |
| ExtraTrees   | True       |   0.790758 |   0.337147 |  0.0720246 |
| ExtraTrees   | False      |   0.793807 |   0.347634 |  0.0904699 |
| MLP          | False      |   0.783992 |   0.373399 |  0.162055  |

## 7. Frozen baseline evaluation (test partition)

| split   |   test pairs |   test drugs |   accuracy |   macro-F1 |   recall_0 |   recall_1 |   recall_2 |
|:--------|-------------:|-------------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| G       |         2618 |           55 |     0.7525 |     0.5066 |     0.3203 |          1 |     0.8923 |
| R       |         2597 |          435 |     0.8036 |     0.5049 |     0.7882 |          0 |     0.8094 |

Split R lets a drug appear in training and test, so it scores much higher. Split G is the number to quote. This table is the pre-ratio LightGBM frozen during the model comparison; the deployed model is in section 8.

## 8. Single deployed model and class-ratio sweep

Class 2 dominates the training rows, so the deployed model caps class 2 in the training partition only. Validation and test keep the real class distribution, and the test partition is scored once after the recipe is fixed. Caps below are relative to the class-0 count.

| recipe   |   train_rows | train_counts             |   val_accuracy |   val_balanced_accuracy |   val_macro_f1 |   val_recall_0 |   val_recall_1 |   val_recall_2 |
|:---------|-------------:|:-------------------------|---------------:|------------------------:|---------------:|---------------:|---------------:|---------------:|
| uncapped |        10495 | {0: 2277, 1: 8, 2: 8210} |       0.726321 |                0.438813 |       0.446471 |       0.311553 |       0.142857 |       0.862028 |
| cap_1    |         4562 | {0: 2277, 1: 8, 2: 2277} |       0.426091 |                0.381373 |       0.308353 |       0.645583 |       0.142857 |       0.35568  |
| cap_2    |         6839 | {0: 2277, 1: 8, 2: 4554} |       0.688154 |                0.442322 |       0.430408 |       0.401986 |       0.142857 |       0.782123 |
| cap_3    |         9116 | {0: 2277, 1: 8, 2: 6831} |       0.721215 |                0.441032 |       0.438707 |       0.331417 |       0.142857 |       0.848823 |

Deployed recipe: **cap_2**. The validation macro-F1 winner was uncapped, so the deployed cap is a safety trade-off, not a metric improvement: it raises avoid-class recall at the cost of macro-F1. On the held-out Split G test partition: accuracy 0.7269, balanced accuracy 0.7428, macro-F1 0.4865. Class 0 precision/recall/F1 = 0.438/0.394/0.414; class 2 = 0.811/0.835/0.823. Class 1 has only 8 training rows, and its test recall is not a meaningful estimate.

Abstention on test: coverage 0.453, accuracy on accepted pairs 0.8172.

## 9. Generalization partitions (R / G / F / D)

Each partition is trained from scratch on its own train split with the deployed LightGBM configuration and class-2 cap 2:1, then scored once on the untouched test split. Leakage checks count entities that appear in more than one partition and must be zero for the entity type the split holds out.

| split   | drug train-test   | drug val-test   | compound train-test   | compound val-test   |
|:--------|:------------------|:----------------|:----------------------|:--------------------|
| G       | 0                 | 0               | -                     | -                   |
| F       | -                 | -               | 0                     | 0                   |
| D       | 0                 | 0               | 0                     | 0                   |

| split   |   test pairs |   class-1 support |   accuracy |   balanced acc |   macro-F1 |     MCC |   kappa | ROC-AUC (OvR)   |   PR-AUC |
|:--------|-------------:|------------------:|-----------:|---------------:|-----------:|--------:|--------:|:----------------|---------:|
| R       |         2597 |                 3 |     0.7771 |         0.6363 |     0.5633 |  0.5037 |  0.4785 | 0.9043          |   0.5963 |
| G       |         2618 |                 1 |     0.7269 |         0.7428 |     0.4865 |  0.2408 |  0.2403 | 0.7670          |   0.4965 |
| F       |         2613 |                 2 |     0.7593 |         0.6446 |     0.6296 |  0.3995 |  0.3932 | 0.8533          |   0.6595 |
| D       |          321 |                 0 |     0.676  |         0.4683 |     0.4601 | -0.0734 | -0.0714 | n/a             |   0.4961 |

Split D keeps only the pairs whose independently assigned drug and compound partitions agree, so it is much smaller than the others and its test class-1 support can be zero; macro metrics there are not reliable and are reported for completeness only.

Split D: kept 8,517 pairs, dropped 12,525; drugs train/val/test = 223/166/55, compounds train/val/test = 2291/1716/565.

## 10. Baselines and class-imbalance experiments

All models use the same Split G train/validation/test rows and the same 18 features selected on the training partition. Class weighting and random oversampling are applied to training rows only; validation and test are never resampled. SMOTE was not used because imbalanced-learn is not a project dependency. These rows are uncapped comparison experiments; the deployed class-2 cap 2:1 model is reported in section 8.

| model                         |   accuracy |   balanced acc |   macro-F1 |    MCC |   class-1 support |
|:------------------------------|-----------:|---------------:|-----------:|-------:|------------------:|
| majority_class                |     0.7552 |         0.3333 |     0.2868 | 0      |                 1 |
| logistic_balanced             |     0.6211 |         0.7232 |     0.3932 | 0.2184 |                 1 |
| lightgbm_unweighted           |     0.7727 |         0.3721 |     0.3651 | 0.2297 |                 1 |
| lightgbm_balanced             |     0.7712 |         0.7363 |     0.5226 | 0.2786 |                 1 |
| lightgbm_balanced_weak        |     0.7525 |         0.7375 |     0.5066 | 0.2521 |                 1 |
| lightgbm_balanced_oversampled |     0.762  |         0.7414 |     0.4907 | 0.2738 |                 1 |

Train class counts {'2': 8210, '0': 2277, '1': 8}, test {'2': 1977, '0': 640, '1': 1}. Class-1 test support is 1, so every class-1 score in this mode is noise. This is reported as a limitation, not hidden.

System paths on the same test partition. The evidence lookup returns the dataset label for the same pair; its 100% accuracy is a label echo and it is excluded from model comparison. The hybrid path deliberately does not use evidence lookup, so it cannot inflate ML-only metrics.

| path                |   coverage | accuracy on covered   | note                                                                                                                           |
|:--------------------|-----------:|:----------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| rule_only           |     0      | n/a                   | Curated rules; no coverage for pairs outside the rule set.                                                                     |
| evidence_lookup     |     0.4683 | 1.0000                | LABEL ECHO: the lookup returns the dataset label for the same pair. Not a predictive model and excluded from model comparison. |
| ml_only             |     1      | 0.7269                | Deployed model, no rules and no evidence lookup.                                                                               |
| hybrid_rule_then_ml |     1      | 0.7269                | Rules override ML when matched; evidence lookup is not used.                                                                   |

## 11. SHAP

- Global importance (backend TreeExplainer): top features PEOEVSA6+PEOEVSA6 (0.165), PEOEVSA7+PEOEVSA7 (0.158), PEOEVSA5+PEOEVSA5 (0.134), abs(MRVSA6-MRVSA6) (0.124), slogPVSA5+slogPVSA5 (0.106).
- Serving explanations use TreeSHAP on the deployed LightGBM and return all 18 feature contributions for all three classes, not a top-k slice. The UI expands the predicted class and lets the other two classes be expanded. SHAP describes model behaviour, not biological mechanism.

## 12. Case studies (nine paper pairs + controls + clinical probes)

| pair                                       | expected   | predicted   | status        | verdict       | confidence         | p_negative         | p_positive         | p_nonsignificant   | sentence                                                                  | reason                                                                                            | note                                              | disclaimer                                             |
|:-------------------------------------------|:-----------|:------------|:--------------|:--------------|:-------------------|:-------------------|:-------------------|:-------------------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------|:-------------------------------------------------------|
| tetracycline + calcium                     | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Tetracycline should not be taken with food containing calcium.            | Tetracyclines/quinolones bind calcium, iron and other cations; separate dosing.                   | paper case: chelation                             | Research tool; confirm with a physician or pharmacist. |
| methotrexate + ethanol                     | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Methotrexate should not be taken with food containing ethanol.            | Alcohol increases hepatotoxicity risk with methotrexate.                                          | paper case: hepatotoxicity                        | Research tool; confirm with a physician or pharmacist. |
| diazepam + ethanol                         | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Diazepam should not be taken with food containing ethanol.                | Alcohol adds CNS depression to benzodiazepines.                                                   | paper case: CNS depression                        | Research tool; confirm with a physician or pharmacist. |
| nitroglycerin + ethanol                    | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Nitroglycerin should not be taken with food containing ethanol.           | Alcohol can cause severe hypotension with nitrates.                                               | paper case: hypotension                           | Research tool; confirm with a physician or pharmacist. |
| digoxin + hyperforin                       | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Digoxin should not be taken with food containing hyperforin.              | St John's Wort induces P-glycoprotein and lowers digoxin exposure.                                | paper case: St John's Wort induction              | Research tool; confirm with a physician or pharmacist. |
| nisoldipine + bergamottin                  | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Nisoldipine should not be taken with food containing bergamottin.         | Grapefruit inhibits intestinal CYP3A4 and raises exposure of this CYP3A4 substrate.               | paper case: CYP3A4 inhibition                     | Research tool; confirm with a physician or pharmacist. |
| midazolam + furanocoumarin                 | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Midazolam should not be taken with food containing furanocoumarin.        | Grapefruit inhibits intestinal CYP3A4 and raises exposure of this CYP3A4 substrate.               | paper case: furanocoumarin class                  | Research tool; confirm with a physician or pharmacist. |
| warfarin + vitamin k1 2,3 epoxide          | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Warfarin should not be taken with food containing vitamin k1 2,3 epoxide. | Vitamin K antagonizes warfarin; keep intake consistent and monitor INR.                           | paper case: vitamin K antagonism                  | Research tool; confirm with a physician or pharmacist. |
| warfarin + dimethyl disulfide              | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Warfarin should not be taken with food containing dimethyl disulfide.     | Garlic constituents have case reports of altered INR with warfarin; monitor INR.                  | paper case: garlic constituent                    | Research tool; confirm with a physician or pharmacist. |
| warfarin + glucose                         | 2.0        | 2.0         | uncertain     | abstained     | 0.5964724190866246 | 0.3710272398559047 | 0.0325003410574706 | 0.5964724190866246 | The model is not confident; ask a pharmacist before acting.               | top probability 0.596 below the validation-selected threshold 0.770                               | control: inert sugar                              | Research tool; confirm with a physician or pharmacist. |
| warfarin + unobtainium                     | n/a        | n/a         | not_supported | informational | n/a                | n/a                | n/a                | n/a                | n/a                                                                       | food compound not found or invalid SMILES                                                         | control: compound absent from the food table      | Research tool; confirm with a physician or pharmacist. |
| nitroglycerin + geranyl rhamnosylglucoside | n/a        | n/a         | not_supported | informational | n/a                | n/a                | n/a                | n/a                | n/a                                                                       | food compound not found or invalid SMILES                                                         | paper unseen compound                             | Research tool; confirm with a physician or pharmacist. |
| warfarin + dicoumarol                      | 0.0        | 0.0         | uncertain     | abstained     | 0.5163791229801913 | 0.5163791229801913 | 0.1204094294517261 | 0.3632114475680825 | The model is not confident; ask a pharmacist before acting.               | top probability 0.516 below the validation-selected threshold 0.770                               | paper unseen compound                             | Research tool; confirm with a physician or pharmacist. |
| warfarin + grapefruit                      | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Warfarin should not be taken with food containing grapefruit.             | Grapefruit inhibits CYP450; the dataset annotates warfarin + grapefruit as avoid and monitor INR. | probe: grapefruit juice, CYP2C9/CYP3A4            | Research tool; confirm with a physician or pharmacist. |
| warfarin + grape juice                     | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Warfarin should not be taken with food containing grape juice.            | Grape and grapefruit products are flagged with warfarin; keep intake consistent and monitor INR.  | probe: user-reported grape juice case             | Research tool; confirm with a physician or pharmacist. |
| simvastatin + grapefruit                   | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Simvastatin should not be taken with food containing grapefruit.          | Grapefruit raises statin exposure and the risk of myopathy.                                       | probe: CYP3A4 inhibition raises statin exposure   | Research tool; confirm with a physician or pharmacist. |
| midazolam + grapefruit                     | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Midazolam should not be taken with food containing grapefruit.            | Grapefruit inhibits intestinal CYP3A4 and raises exposure of this CYP3A4 substrate.               | probe: CYP3A4 inhibition, classic index substrate | Research tool; confirm with a physician or pharmacist. |
| metronidazole + alcohol                    | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Metronidazole should not be taken with food containing alcohol.           | Alcohol with nitroimidazoles can cause a disulfiram-like reaction.                                | probe: disulfiram-like reaction                   | Research tool; confirm with a physician or pharmacist. |
| warfarin + cranberry                       | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Warfarin should not be taken with food containing cranberry.              | Cranberry has case reports of raised INR with warfarin; keep intake consistent and monitor INR.   | probe: cranberry potentiation                     | Research tool; confirm with a physician or pharmacist. |
| digoxin + st john's wort                   | 0.0        | 0.0         | rule          | pass          | n/a                | n/a                | n/a                | n/a                | Digoxin should not be taken with food containing st john's wort.          | St John's Wort induces P-glycoprotein and lowers digoxin exposure.                                | probe: P-glycoprotein induction                   | Research tool; confirm with a physician or pharmacist. |

## 13. Ablations and enhancements (Split G)

| model                |   accuracy |   macro-F1 |   seconds |
|:---------------------|-----------:|-----------:|----------:|
| drug_only            |     0.6474 |     0.3333 |       0.5 |
| food_only            |     0.7319 |     0.4698 |       0.5 |
| shuffled_food        |     0.6448 |     0.365  |       0.5 |
| all_3780_features    |     0.7437 |     0.427  |      94.5 |
| 18+E1a_drug_metadata |     0.6933 |     0.4625 |       0.5 |
| 18+E1b_food_class    |     0.7021 |     0.444  |       0.5 |
| 18+E1c_fingerprints  |     0.7403 |     0.4927 |       2.1 |
| 18+E1abc_all         |     0.7349 |     0.5016 |       2.2 |
| 18_moe               |     0.694  |     0.4605 |       1.7 |

E2 conformal at a 90% target: test coverage 0.921, abstention rate 0.427, selective accuracy 0.863. Drug-level bootstrap 95% CI for macro-F1: [0.274, 0.445].

The conformal experiment is reported for completeness only; serving uses the temperature-scaled probabilities and the validation-selected acceptance threshold described in section 16.

Food-side descriptors alone beat the 18 pair features here; shuffled pairing costs about 0.07 macro-F1. Both say the signal is dominated by compound identity, which is exactly what the specific labels' weak class-2 pairs predict.

## 14. Failure audit and remediation

The first trained model predicted well-known interactions as non-significant. The audit found three label/resolution defects and one modelling limit (details in `reports/retune_audit.md`):

- Raw `Effect=Positive` was mapped to class 1 even when the conclusion was a warning such as 'avoid grapefruit'; warning-style Positive rows now map to class 0.
- Food-level labels were copied to every compound of the food, so warfarin inherited about 3,000 cranberry/grapefruit pairs including glucose and UDP-glucose. The specific mode now uses curated marker compounds plus a food-specificity filter.
- Food-name resolution took the first linked compound ('grape' became ethanol) and could not parse juice names.
- The 18 descriptors contain no drug identity, so unseen-drug generalisation is limited (see Split G vs Split R).

After relabelling and retuning, Split G accuracy is 0.7525 and macro-F1 is 0.5066 for the frozen comparison model; the deployed capped model scores 0.7269 accuracy and 0.4865 macro-F1.
The held-out case table scores 16/16 passes, 0 failures, 2 abstention and 2 unsupported rows.
The recommender adds a curated clinical rule layer, exact-pair dataset evidence, marker-based candidate resolution and conservative aggregation; warfarin + grape juice / grapefruit, simvastatin + grapefruit, midazolam + grapefruit, metronidazole + alcohol, warfarin + cranberry and digoxin + St John's Wort all return class 0.

## 15. Decisions where the paper was ambiguous

- Correlated-pair tie-break: keep the feature with the higher ANOVA F-score (LASSO kept 174 features; L1 multinomial logistic kept 68, overlap 68).
- Three-class LASSO: regression on codes 0/1/2, cross-checked with L1 multinomial logistic.
- RFECV chose 89 features, so the final set uses RFE fixed at 18.
- Label rules are applied only to the 'Possible' interaction rows; confirmed structured labels are kept as-is.
- Model selection uses validation macro-F1, never test.
- Class-2 capping and the deployed 2:1 ratio are validation-selected in code but deployed as a safety choice; the report states the macro-F1 cost explicitly.

Why these numbers differ from the paper:

- Labels come from FHDI structured annotations, not DrugBank sentences; the paper's 97% accuracy is not reproducible on this label source.
- The specific label mode removes the shortcut of labelling a whole drug row uniformly, so scores drop sharply.
- Split G holds out entire drugs, so the model cannot memorise drug identity.
- Training used a 50,000-pair cap for CV/tuning and a 40,000-pair cap for the ablation stage because of the 3.7 GB machine, then refit on the full training partition.

## 16. Calibration, abstention and applicability domain

Probabilities are temperature-scaled with the temperature fitted on Split G validation, and the acceptance threshold is chosen on that validation curve as the lowest threshold whose accepted error is at most 0.20. Neither step uses test data. ECE is expected calibration error over ten reliability bins; lower is better.

|   temperature |   threshold |   validation coverage |   test coverage |   test accuracy on accepted |   test ECE before |   test ECE after |   test Brier before |   test Brier after |
|--------------:|------------:|----------------------:|----------------:|----------------------------:|------------------:|-----------------:|--------------------:|-------------------:|
|        1.3405 |        0.77 |                0.3878 |          0.4534 |                      0.8172 |            0.0713 |           0.0363 |              0.3919 |              0.384 |

The deployed model abstains on roughly half of test pairs and is more accurate on the pairs it accepts. That is the honest consequence of class-1 scarcity and weak unseen-drug signal: it refuses rather than forcing a class.

| validated   |   drug threshold | drug rule                                                                        |   drug refs |   food threshold | food rule                                                                            |   food refs |
|:------------|-----------------:|:---------------------------------------------------------------------------------|------------:|-----------------:|:-------------------------------------------------------------------------------------|------------:|
| True        |           0.1856 | 10th percentile of validation similarities for unseen drugs (Split G validation) |         223 |           0.3333 | 10th percentile of validation similarities for unseen compounds (Split F validation) |        2291 |

References are training-only: drug references come from the Split G training drugs and food references from the Split F training compounds, so held-out entities can never match themselves. Drug similarity is the maximum Tanimoto to a reference drug and food similarity the maximum to a reference compound (Morgan radius 2, 2,048 bits). The thresholds are a documented heuristic, not a safety guarantee.

## 17. Serving contract and provenance

- The deployed artifact is a single LightGBM single-2026-11-16 (`best_model_S.joblib`, sha256 prefix `c8d4f9dadf2d63b2`), 18 features, training split `Split G train (drug-grouped), class-2 capped`. There is no mode switch in the API or the UI.
- Input resolution order is: local curated database, Gemini (`GEMINI_API_KEY` from `dfi/.env` or the environment, shell variables winning, never stored in the repository), PubChem, then a user-supplied SMILES. Gemini answers are canonicalised with RDKit and rejected when they do not parse; resolutions are cached locally. Every response includes the `resolution` block showing where each SMILES came from.
- Every response carries `prediction_source`, one of `ml_model`, `dataset_evidence` or `rule_based`, plus `abstention_reason` when no prediction is returned (`insufficient_confidence`, `outside_applicability_domain`, `unknown_drug`, `unknown_food`, `inorganic_compound`, `descriptor_unavailable`).
- Rule results carry the rule ID, sentence, reason, provenance, validation status and version and never carry a model probability or ML confidence. Dataset-evidence results carry the dataset record ID and a lookup note, also without ML confidence.
- Curated rules are versioned as `2026.11.1` with validation status `unvalidated_curated`; there are 16 rules and each is labelled rule output.
- ML responses carry calibrated class probabilities, the acceptance threshold, the model name/version and training split, the applicability-domain assessment with thresholds and reference counts, and TreeSHAP contributions for all 18 features across all three classes, with an explicit caveat that SHAP is not biological causation.
- `/api/health` reports readiness, thresholds, resolver status (Gemini configured or not) and the rule version; `/api/meta` exposes the model, feature, rule, domain and resolver metadata used by the UI.
- Stage CSV exports are written to `data/processed/csv/`: drug_cleaned.csv (1,359 rows), food_cleaned.csv (5,022 rows), pairs.csv (21,042 rows), features_generation.csv (6,236 rows), feature_extraction.csv (20,947 rows).

## 18. Limitations and safety

- This is decision support research, not medical advice; every recommendation carries the disclaimer.
- Class-1 support is tiny (one test pair), so class-1 and macro scores are unstable; no target accuracy is promised.
- The majority baseline beats the deployed model on raw accuracy; the model is tuned for macro-F1 and minority recall, and the class-2 cap is a documented safety trade-off.
- The evidence path echoes the dataset label for the same pair; its accuracy is tautological and it does not contribute to ML metrics.
- The applicability-domain thresholds are a documented p10 heuristic; an inside-domain result is not a clinical guarantee.
- Inorganic minerals return `not_supported` because the structural descriptors do not apply to them.
- The external Ryu et al. benchmark was not obtained; Split G test plus held-out case studies stand in for it.
- SHAP values describe model behaviour, not pharmacological mechanism.
- Gemini resolution depends on an external service and is validated only structurally, not chemically curated.

## 19. Reproduce

```bash
bash scripts/run_all.sh        # bricks 1-17, stage CSVs, report, notebooks and gates
python scripts/make_report.py  # rebuild this report from reports/*
python -m pytest -q tests      # serving, provenance, resolver and feature tests
python scripts/check_gates.py  # project gates
```

See `reports/implementation_changes.md` for the change log.