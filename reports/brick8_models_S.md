# Brick 8 - models (Mode S)

Selected features: 18. Baselines use 5-fold grouped CV on a 50,000-pair sample of the Split G training partition.
The winning model and XGBoost are then refit on the full training partition before being saved.

## Baseline CV

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

## Validation after tuning

| model        |   accuracy |   macro_f1 |   seconds |
|:-------------|-----------:|-----------:|----------:|
| XGBoost      |   0.475109 |   0.359492 |     104.2 |
| LightGBM     |   0.726321 |   0.446471 |      59.2 |
| RandomForest |   0.658284 |   0.426137 |      14.6 |
| ExtraTrees   |   0.471024 |   0.321082 |       5.2 |
| MLP          |   0.729385 |   0.370419 |       2.9 |

Best by validation macro-F1: **LightGBM** (0.4465 on the sample).
After the full-training refit: macro-F1 0.4465 (accuracy 0.7263). XGBoost full refit macro-F1 0.3595.
XGBoost reload max probability difference: 0.00e+00.

Tuned XGBoost validation confusion matrix:

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |     1375 |        1 |      537 |
| true 1 |       11 |        3 |        0 |
| true 2 |     3539 |       24 |     2344 |