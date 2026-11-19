# Baselines and system paths (Mode S)

All models use the same Split G train/validation/test rows and the same 18 selected features. Hyperparameters come from the saved LightGBM tuning run.

## Model comparison

| model                         | partition   |    n |   accuracy |   balanced_accuracy |   macro_f1 |    mcc |   class_1_support |
|:------------------------------|:------------|-----:|-----------:|--------------------:|-----------:|-------:|------------------:|
| majority_class                | validation  | 7834 |     0.754  |              0.3333 |     0.2866 | 0      |                14 |
| majority_class                | test        | 2618 |     0.7552 |              0.3333 |     0.2868 | 0      |                 1 |
| logistic_balanced             | validation  | 7834 |     0.6062 |              0.4762 |     0.3817 | 0.167  |                14 |
| logistic_balanced             | test        | 2618 |     0.6211 |              0.7232 |     0.3932 | 0.2184 |                 1 |
| lightgbm_unweighted           | validation  | 7834 |     0.7686 |              0.3682 |     0.3586 | 0.2113 |                14 |
| lightgbm_unweighted           | test        | 2618 |     0.7727 |              0.3721 |     0.3651 | 0.2297 |                 1 |
| lightgbm_balanced             | validation  | 7834 |     0.7531 |              0.435  |     0.4471 | 0.2165 |                14 |
| lightgbm_balanced             | test        | 2618 |     0.7712 |              0.7363 |     0.5226 | 0.2786 |                 1 |
| lightgbm_balanced_weak        | validation  | 7834 |     0.7263 |              0.4388 |     0.4465 | 0.1954 |                14 |
| lightgbm_balanced_weak        | test        | 2618 |     0.7525 |              0.7375 |     0.5066 | 0.2521 |                 1 |
| lightgbm_balanced_oversampled | validation  | 7834 |     0.7365 |              0.4425 |     0.4328 | 0.2152 |                14 |
| lightgbm_balanced_oversampled | test        | 2618 |     0.762  |              0.7414 |     0.4907 | 0.2738 |                 1 |

## Class imbalance experiments

Train class counts: {2: 8210, 0: 2277, 1: 8}. Test class counts: {2: 1977, 0: 640, 1: 1}. Random oversampling is applied to training rows only; validation and test are untouched.

## System paths on the test partition

| path                |   coverage |   accuracy_on_covered |   accuracy_on_all_with_abstention_as_wrong | note                                                                                                                           |
|:--------------------|-----------:|----------------------:|-------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------|
| rule_only           |   0        |            nan        |                                   0        | Curated rules; no coverage for pairs outside the rule set.                                                                     |
| evidence_lookup     |   0.468296 |              1        |                                   0.468296 | LABEL ECHO: the lookup returns the dataset label for the same pair. Not a predictive model and excluded from model comparison. |
| ml_only             |   1        |              0.726891 |                                   0.726891 | Deployed model, no rules and no evidence lookup.                                                                               |
| hybrid_rule_then_ml |   1        |              0.726891 |                                   0.726891 | Rules override ML when matched; evidence lookup is not used.                                                                   |

## Rules fired on test

{}

## Notes and limitations

- Class 1 support is <10 in every partition; its metrics are not reliable.
- SMOTE was not applied because imbalanced-learn is not a project dependency; random oversampling is the tested resampling method.
- Evidence lookup is shown for coverage only. Its 100% accuracy is tautological.
- The hybrid path does not use evidence lookup, to avoid turning evaluation into label retrieval.