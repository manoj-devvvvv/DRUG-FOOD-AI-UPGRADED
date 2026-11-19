# Single model: class-ratio sweep and held-out result

Selected recipe **cap_2** on Split G validation macro-F1. Temperature 1.3405, acceptance threshold 0.77.

## Ratio sweep (validation only)

| recipe   |   train_rows | train_counts             |   val_accuracy |   val_balanced_accuracy |   val_macro_f1 |   val_recall_0 |   val_recall_1 |   val_recall_2 |
|:---------|-------------:|:-------------------------|---------------:|------------------------:|---------------:|---------------:|---------------:|---------------:|
| uncapped |        10495 | {0: 2277, 1: 8, 2: 8210} |       0.726321 |                0.438813 |       0.446471 |       0.311553 |       0.142857 |       0.862028 |
| cap_1    |         4562 | {0: 2277, 1: 8, 2: 2277} |       0.426091 |                0.381373 |       0.308353 |       0.645583 |       0.142857 |       0.35568  |
| cap_2    |         6839 | {0: 2277, 1: 8, 2: 4554} |       0.688154 |                0.442322 |       0.430408 |       0.401986 |       0.142857 |       0.782123 |
| cap_3    |         9116 | {0: 2277, 1: 8, 2: 6831} |       0.721215 |                0.441032 |       0.438707 |       0.331417 |       0.142857 |       0.848823 |

## Held-out Split G test (scored once)

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       0 |      0.4375 |   0.3937 | 0.4145 |       640 |
|       1 |      0.125  |   1      | 0.2222 |         1 |
|       2 |      0.8112 |   0.8346 | 0.8227 |      1977 |

Accuracy 0.7269; balanced accuracy 0.7428; macro-F1 0.4865.

Abstention: coverage 0.453, accuracy on accepted 0.8171861836562764.

## Notes

- Capping is applied to training rows only; validation and test keep the real class distribution.
- Deployed cap 2:1 for avoid-class recall as a product decision. The validation macro-F1 winner was uncapped (0.4465 vs 0.4304); this is a trade-off, not a metric improvement.
- Class 1 has only 8 training rows, so class-1 metrics remain unstable regardless of ratio.
- The 18 features were selected on the uncapped training partition and are reused unchanged.