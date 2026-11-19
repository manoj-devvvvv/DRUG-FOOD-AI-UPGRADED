# Generalization evaluation (Mode S)

Every partition is trained from scratch with the same fixed LightGBM configuration (18 selected features, balanced class weights, weak-pair weight 0.5, class-2 cap 2.0:1 applied to training rows only). Test rows are scored only after fitting.

## Leakage checks (overlap must be zero)

| split   |   drug_train_val |   drug_train_test |   drug_val_test |   compound_train_val |   compound_train_test |   compound_val_test |
|:--------|-----------------:|------------------:|----------------:|---------------------:|----------------------:|--------------------:|
| R       |              nan |               nan |             nan |                  nan |                   nan |                 nan |
| G       |                0 |                 0 |               0 |                  nan |                   nan |                 nan |
| F       |              nan |               nan |             nan |                    0 |                     0 |                   0 |
| D       |                0 |                 0 |               0 |                    0 |                     0 |                   0 |

## Results

| split   | partition   |    n |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |     mcc |   kappa |   class_1_support |   roc_auc_ovr_macro |   pr_auc_macro |
|:--------|:------------|-----:|-----------:|--------------------:|-----------:|--------------:|--------:|--------:|------------------:|--------------------:|---------------:|
| R       | validation  | 7862 |     0.7819 |              0.6051 |     0.5574 |        0.7948 |  0.4997 |  0.48   |                 8 |              0.8883 |         0.5876 |
| R       | test        | 2597 |     0.7771 |              0.6363 |     0.5633 |        0.7917 |  0.5037 |  0.4785 |                 3 |              0.9043 |         0.5963 |
| G       | validation  | 7834 |     0.6882 |              0.4423 |     0.4304 |        0.6919 |  0.1819 |  0.1817 |                14 |              0.6246 |         0.4548 |
| G       | test        | 2618 |     0.7269 |              0.7428 |     0.4865 |        0.7227 |  0.2408 |  0.2403 |                 1 |              0.767  |         0.4965 |
| F       | validation  | 7837 |     0.7611 |              0.5091 |     0.4974 |        0.7731 |  0.4125 |  0.4022 |                15 |              0.7656 |         0.5253 |
| F       | test        | 2613 |     0.7593 |              0.6446 |     0.6296 |        0.7691 |  0.3995 |  0.3932 |                 2 |              0.8533 |         0.6595 |
| D       | validation  | 2920 |     0.6719 |              0.3594 |     0.3597 |        0.6536 |  0.0899 |  0.0879 |                 1 |              0.4101 |         0.3612 |
| D       | test        |  321 |     0.676  |              0.4683 |     0.4601 |        0.6526 | -0.0734 | -0.0714 |                 0 |            nan      |         0.4961 |

## Per-class metrics on the test partition

### Split R — pair-random (same drugs can appear on both sides)

Test rows 2597; class counts {0: 595, 1: 3, 2: 1999}; ROC-AUC OVR macro 0.9042858977043086; PR-AUC macro 0.5962957640843382.

|    |   precision |   recall |       f1 |   support |
|---:|------------:|---------:|---------:|----------:|
|  0 |    0.511182 | 0.806723 | 0.625815 |       595 |
|  1 |    0.166667 | 0.333333 | 0.222222 |         3 |
|  2 |    0.930387 | 0.768884 | 0.841961 |      1999 |

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      480 |        0 |      115 |
| true 1 |        2 |        1 |        0 |
| true 2 |      457 |        5 |     1537 |

**Limitation:** Class 1 has only 3 examples; class-1 metrics are unreliable.

### Split G — unseen drugs (whole-drug groups)

Test rows 2618; class counts {0: 640, 1: 1, 2: 1977}; ROC-AUC OVR macro 0.7669632011579517; PR-AUC macro 0.4964628166416256.

|    |   precision |   recall |       f1 |   support |
|---:|------------:|---------:|---------:|----------:|
|  0 |    0.4375   | 0.39375  | 0.414474 |       640 |
|  1 |    0.125    | 1        | 0.222222 |         1 |
|  2 |    0.811209 | 0.834598 | 0.822737 |      1977 |

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      252 |        4 |      384 |
| true 1 |        0 |        1 |        0 |
| true 2 |      324 |        3 |     1650 |

**Limitation:** Class 1 has only 1 examples; class-1 metrics are unreliable.

### Split F — unseen food compounds (whole-compound groups)

Test rows 2613; class counts {0: 611, 1: 2, 2: 2000}; ROC-AUC OVR macro 0.8532828384106681; PR-AUC macro 0.6595283146445987.

|    |   precision |   recall |       f1 |   support |
|---:|------------:|---------:|---------:|----------:|
|  0 |    0.489924 | 0.636661 | 0.553737 |       611 |
|  1 |    0.5      | 0.5      | 0.5      |         2 |
|  2 |    0.87727  | 0.797    | 0.835211 |      2000 |

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      389 |        0 |      222 |
| true 1 |        0 |        1 |        1 |
| true 2 |      405 |        1 |     1594 |

**Limitation:** Class 1 has only 2 examples; class-1 metrics are unreliable.

### Split D — double-cold (drug and compound groups both unseen)

Test rows 321; class counts {0: 70, 1: 0, 2: 251}; ROC-AUC OVR macro None; PR-AUC macro 0.49610956655992.

|    |   precision |   recall |       f1 |   support |
|---:|------------:|---------:|---------:|----------:|
|  0 |    0.145833 | 0.1      | 0.118644 |        70 |
|  1 |    0        | 0        | 0        |         0 |
|  2 |    0.769231 | 0.836653 | 0.801527 |       251 |

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |        7 |        0 |       63 |
| true 1 |        0 |        0 |        0 |
| true 2 |       41 |        0 |      210 |

**Limitation:** Class 1 has no examples in this evaluation partition; class-1 metrics are not meaningful.

## Limitations

- Split R is an easier diagnostic because drugs repeat across partitions.
- Split F holds out compounds but drugs repeat.
- Split D drops pairs whose drug and compound partitions differ, so its size is limited.
- Class 1 support is tiny in Mode S; class-1 metrics are unreliable wherever support < 10.