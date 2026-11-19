# Brick 8b - all five models on held-out test data (Mode S)

Features: 18 selected pair descriptors. Weak pairs weighted 0.5. Non-MLP models use balanced class weights; MLP uses its own early stopping. Each model is refit on the full training partition of the split and scored on that split's test partition, so no tuning-validation rows leak into the test numbers.

Split G tests generalization to unseen drugs (headline). Split R is the paper-like random split for reference. Selective accuracy only answers rows whose top probability clears 0.60.

## Split G (unseen drugs)

| model        |   accuracy |   balanced_accuracy |   macro_f1 |   neg_precision |   neg_recall |   nonsig_recall |   coverage_at_threshold |   selective_accuracy |   seconds |
|:-------------|-----------:|--------------------:|-----------:|----------------:|-------------:|----------------:|------------------------:|---------------------:|----------:|
| XGBoost      |     0.5157 |              0.7284 |     0.3642 |          0.3042 |       0.7438 |          0.4416 |                  0.3705 |               0.5423 |       2   |
| LightGBM     |     0.7525 |              0.7375 |     0.5066 |          0.4964 |       0.3203 |          0.8923 |                  0.8846 |               0.7811 |       3.3 |
| RandomForest |     0.7002 |              0.4137 |     0.4092 |          0.4024 |       0.4641 |          0.7769 |                  0.7059 |               0.7684 |       1.2 |
| ExtraTrees   |     0.4927 |              0.7377 |     0.3402 |          0.3089 |       0.8297 |          0.3834 |                  0.1123 |               0.7075 |       0.6 |
| MLP          |     0.738  |              0.3842 |     0.3879 |          0.4403 |       0.2594 |          0.8933 |                  0.9137 |               0.7592 |       1.1 |

Best on Split G by macro-F1: **LightGBM** (0.5066).

## Split R (random split)

| model        |   accuracy |   balanced_accuracy |   macro_f1 |   neg_precision |   neg_recall |   nonsig_recall |   coverage_at_threshold |   selective_accuracy |   seconds |
|:-------------|-----------:|--------------------:|-----------:|----------------:|-------------:|----------------:|------------------------:|---------------------:|----------:|
| XGBoost      |     0.4594 |              0.6364 |     0.3765 |          0.2898 |       0.921  |          0.3217 |                  0.3323 |               0.5435 |       0.5 |
| LightGBM     |     0.8036 |              0.5325 |     0.5049 |          0.5537 |       0.7882 |          0.8094 |                  0.8398 |               0.8459 |       2.6 |
| RandomForest |     0.7251 |              0.4997 |     0.4562 |          0.4453 |       0.7933 |          0.7059 |                  0.6119 |               0.7999 |       1.2 |
| ExtraTrees   |     0.432  |              0.7379 |     0.3395 |          0.2828 |       0.9311 |          0.2826 |                  0.1352 |               0.735  |       0.7 |
| MLP          |     0.7932 |              0.4316 |     0.4428 |          0.5773 |       0.3765 |          0.9185 |                  0.8872 |               0.8255 |       1   |

Best on Split R by macro-F1: **LightGBM** (0.5049).

Class 1 has very few real pairs, so its F1 is noisy; class 0 (avoid) and class 2 (non-significant) dominate the macro average.

Figure: `figures\model_comparison_S.png`