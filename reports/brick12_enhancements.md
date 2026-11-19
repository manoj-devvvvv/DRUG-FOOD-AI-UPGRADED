# Brick 12 - ablations and section 15 enhancements

Mode S, Split G, XGBoost with the tuned Brick 8 parameters. Training rows 10,495 (capped), test rows 2,618.

## Ablations and feature sets

| config               |   macro_f1 |   accuracy |   n_features |
|:---------------------|-----------:|-----------:|-------------:|
| 18_moe               |   0.460534 |   0.694041 |           18 |
| drug_only            |   0.333273 |   0.647441 |           18 |
| food_only            |   0.469777 |   0.731856 |           18 |
| shuffled_food        |   0.365041 |   0.644767 |           18 |
| all_3780_features    |   0.427047 |   0.743697 |           18 |
| 18+E1a_drug_metadata |   0.46246  |   0.693277 |           19 |
| 18+E1b_food_class    |   0.444003 |   0.702063 |           28 |
| 18+E1c_fingerprints  |   0.492651 |   0.74026  |          530 |
| 18+E1abc_all         |   0.501584 |   0.734912 |          541 |

## E2 calibration and split conformal

Variant: 18+E1abc_all; q_hat = 0.7889 at a 90% target.
Test coverage 0.921, average prediction-set size 1.43, abstention rate 0.427.
Argmax accuracy 0.7552; selective accuracy 0.8628 on the 1,501 non-abstained pairs.
Drug-level bootstrap 95% CI for macro-F1: [0.27376850035663336, 0.4454842481525476].

Reading: drug- and food-only models show whether the pair interaction adds signal. A large drop under shuffled food pairing is the direct evidence that the model uses the pairing.