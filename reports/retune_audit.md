# Why the first model failed, and what changed

This audit was triggered by the report that **warfarin + grape juice was predicted non-significant** and that the model behaved "dumb" on well-known interactions.

## 1. Root causes found

### 1.1 The labels encoded warnings as benefits

The raw `Effect` column maps `Positive` to class 1 (beneficial / take with food). Roughly half of the `Positive` conclusions are actually warnings such as "Avoid grapefruit products" or "increases plasma concentrations". Warfarin + bergamottin and warfarin + naringin were therefore labelled **beneficial** in Mode S.

The corrected mapping is:

| raw annotation | class | rule |
|---|---|---|
| Harmful, Negative | 0 avoid | always |
| Positive | 0 avoid | unless the text says take with food/meals/milk, improves efficacy, or improves absorption |
| No Effect | 2 non-significant | always |
| Possible | 0 or 2 | text rules; default 2 |

Raw interaction rows previously counted every `Positive` annotation as a benefit; the corrected rules move the warning-style `Positive` rows into class 0. Mode S ends with **21,042 pairs `{0: 4,898, 1: 24, 2: 16,120}`**.

Verified fixes: warfarin + grapefruit 0, warfarin + cranberry 0, simvastatin + grapefruit 0, midazolam + grapefruit 0, metronidazole + alcohol 0, tetracycline + calcium 0.

### 1.2 Food labels were copied to every compound of the food

The old Mode S mapping applied a food-level label to *all* compounds listed for that food. Warfarin received about **1,055 American-cranberry, 997 European-cranberry and 1,040 grapefruit pairs**, including ubiquitous metabolites such as UDP-glucose and glucose-6-phosphate. This is why **warfarin + glucose was predicted avoid at 85.5% confidence**.

The new mapping uses:

- curated marker compounds per food (grapefruit -> bergamottin, naringin, bergapten; cranberry -> resveratrol, catechin, proanthocyanidins, cyanidin; garlic -> allicin, dimethyl disulfide; etc.);
- a food-specificity filter (compound must appear in <= 20 foods, at most 25 compounds per interaction row);
- no propagation to the full compound list of a food.

Warfarin now has **315 Mode S pairs (84 avoid + 231 non-significant)** and **zero glucose pairs**.

### 1.3 The model is descriptor-only

The 18 selected features are MOE descriptors of the drug and the food compound (products, sums, differences, ratios). There is no drug identity feature. On unseen drugs the model can only generalise through chemistry:

| split | baseline accuracy | baseline macro-F1 |
|---|---|---|
| Split G (drugs disjoint) | 0.4477 | 0.4446 |
| Split R (random pairs) | 0.7974 | 0.7950 |

That gap is the signature of drug-level prior learning. Hyperparameter tuning cannot remove it. It is why the recommender now uses a curated rule layer and an exact-pair evidence lookup in addition to the model.

### 1.4 Food-name resolution bugs

The old resolver took the first linked compound, so "grape" became ethanol, "grapefruit" became bergamottin only, and "grapefruit juice" was unsupported. The new resolver canonicalises juice/raw/fresh/dried/extract/powder/tea suffixes, uses marker compounds, and only then falls back to specific linked compounds or a SMILES string.

## 2. Changes made

- Corrected label rules and marker/specificity mapping (`brick4_labels.py`).
- Relabelled and rebuilt splits, selection and models.
- Expanded XGBoost search to 80 configurations, weighted weak synthetic negatives (0.5), and stratified capping (`brick8_models.py`).
- Added inference layers in `recommend.py`:
  1. curated clinical rules (warfarin/grapefruit/grape/cranberry/vitamin K/alcohol/garlic, statins and CYP3A4 substrates + grapefruit, nitroimidazoles + alcohol, MAOIs + tyramine, digoxin + St John's Wort, tetracyclines/quinolones + cations, levothyroxine + calcium/iron, nitrates/methotrexate/benzodiazepines + alcohol);
  2. exact-pair dataset evidence for non-weak labels;
  3. marker-based food resolution and conservative aggregation over candidates;
  4. abstention below 0.6 confidence; refusal for invalid, inorganic or out-of-domain inputs; always appends the disclaimer.
- Added seven clinical probe cases to Brick 11 and six to the training holdout.

## 3. Validation on the available data

### 3.1 Mode S, Split G test (55 unseen drugs, 2,618 pairs)

| metric | before | after |
|---|---|---|
| accuracy | 0.4477 | **0.7525** |
| macro-F1 | 0.4446 | **0.5066** |
| class-0 precision | 0.3717 | **0.4964** |
| class-0 recall | 0.4119 | 0.3203 |

Confidence sweep (answer only above threshold):

| threshold | coverage | accuracy | class-0 recall |
|---|---|---|---|
| 0.5 | 99.1% | 0.7560 | 0.319 |
| 0.6 | 88.5% | 0.7811 | 0.292 |
| 0.7 | 77.3% | 0.8005 | 0.256 |
| 0.8 | 62.3% | 0.8223 | 0.201 |

Per-drug accuracy: median 0.857, 10th percentile 0.495; 12 of 55 drugs perfect, none at zero.

### 3.2 Mode S, Split R (435 drugs, 2,597 pairs)

Accuracy **0.8036**, macro-F1 **0.5049**, class-0 recall **0.7882**, class-0 precision **0.5537**. Macro-F1 is dominated by the single class-1 test pair; accuracy and class-0 recall are the meaningful numbers here.

### 3.3 Held-out case studies (Brick 11, 20 rows)

**17 of 17 scored cases pass; 0 fail; 1 abstains; 2 are informational (not supported).**

| pair | predicted | status |
|---|---|---|
| warfarin + grapefruit | 0 | rule |
| warfarin + grape juice | 0 | rule |
| simvastatin + grapefruit | 0 | rule |
| midazolam + grapefruit | 0 | rule |
| metronidazole + alcohol | 0 | rule |
| warfarin + cranberry | 0 | rule |
| digoxin + St John's Wort | 0 | rule |
| tetracycline + calcium | 0 | rule |
| warfarin + glucose | 2 at 0.52 | abstained |

### 3.4 Ablations and enhancements (Brick 12, Mode S Split G)

| configuration | macro-F1 | accuracy |
|---|---|---|
| 18 MOE pair features | 0.4605 | 0.6940 |
| drug-only | 0.3333 | 0.6474 |
| food-only | 0.4698 | 0.7319 |
| shuffled food pairing | 0.3650 | 0.6448 |
| 18 + drug/food fingerprints + metadata | **0.5016** | 0.7349 |

The shuffled-food drop confirms the model uses the pairing, but food-only performing as well as the pair model shows the interaction signal in the 18 descriptors is modest. Fingerprints add the largest gain.

Split-conformal at 90% target: **coverage 0.921**, abstention 42.7%, selective accuracy **86.3%** on answered pairs.

### 3.5 API/UI smoke tests

`POST /api/predict` returns class 0 ("Avoid together") for warfarin + grape juice / grapefruit, simvastatin + grapefruit, metronidazole + alcohol, tetracycline + calcium; class 0 evidence for warfarin + pomegranate; abstains on warfarin + glucose; rejects unknown drugs and foods with a clear reason; always includes the disclaimer.

## 4. Remaining limitations

- Class 1 has only 24 labelled pairs; it is not learnable. Treat class 1 as illustrative only.
- Mode S covers 444 of 1,359 drugs. Drugs outside it are marked as extrapolation.
- The model's class-0 recall on unseen drugs is 0.32; rule and evidence layers, not the model, are responsible for catching the well-known interactions.
- Rules are curated and finite; evidence exists only for pairs present in the dataset.
- This is a research tool, not clinical validation. Confirm with a physician or pharmacist.
