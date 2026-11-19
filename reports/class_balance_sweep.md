# Class-ratio sweep (Mode S)

Selected recipe: **uncapped** on validation macro-F1.

| recipe   |   train_rows | train_counts             |   val_accuracy |   val_balanced_accuracy |   val_macro_f1 |   val_recall_0 |   val_recall_1 |   val_recall_2 |
|:---------|-------------:|:-------------------------|---------------:|------------------------:|---------------:|---------------:|---------------:|---------------:|
| uncapped |        10495 | {0: 2277, 1: 8, 2: 8210} |       0.726321 |                0.438813 |       0.446471 |       0.311553 |       0.142857 |       0.862028 |
| cap_1    |         4562 | {0: 2277, 1: 8, 2: 2277} |       0.426091 |                0.381373 |       0.308353 |       0.645583 |       0.142857 |       0.35568  |
| cap_2    |         6839 | {0: 2277, 1: 8, 2: 4554} |       0.688154 |                0.442322 |       0.430408 |       0.401986 |       0.142857 |       0.782123 |
| cap_3    |         9116 | {0: 2277, 1: 8, 2: 6831} |       0.721215 |                0.441032 |       0.438707 |       0.331417 |       0.142857 |       0.848823 |

## Notes

- Capping is applied to training rows only; validation and test keep the real class distribution.
- Selection uses validation macro-F1 with balanced accuracy as the tie-break; test is scored once after selection.
- Class 1 has only 9 training rows, so class-1 metrics remain unstable regardless of ratio.
- The 18 features were selected on the uncapped training partition and are reused unchanged.