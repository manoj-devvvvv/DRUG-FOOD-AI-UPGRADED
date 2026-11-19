# Brick 7 - feature selection (Mode S)

Selection sample: 10,495 training pairs; class counts {0: 2277, 1: 8, 2: 8210}.

| Step | Remaining features |
|---|---|
| start | 3780 |
| variance > 0.8 | 3521 |
| Pearson |r| < 0.75 | 1126 |
| LASSO alpha=0.01 nonzero | 174 |
| RFECV optimum | 89 |
| RFE fixed 18 | 18 |

RFECV macro-F1 0.3506; RFE18 macro-F1 0.3263.

Stability: 8 of 18 final features were chosen in at least 2 of 3 runs.

## Selected features

| name                  | term     |   stability_hits |
|:----------------------|:---------|-----------------:|
| MRVSA6*EstateVSA6     | product  |                2 |
| PEOEVSA5+PEOEVSA5     | sum      |                3 |
| PEOEVSA6+PEOEVSA6     | sum      |                2 |
| PEOEVSA7+PEOEVSA7     | sum      |                2 |
| VSAEstate6+VSAEstate6 | sum      |                3 |
| slogPVSA1/slogPVSA1   | ratio    |                3 |
| MRVSA5/MRVSA5         | ratio    |                3 |
| MRVSA9/MRVSA9         | ratio    |                2 |
| slogPVSA9*PEOEVSA8    | product  |                1 |
| MRVSA3*PEOEVSA8       | product  |                0 |
| MRVSA5*PEOEVSA10      | product  |                0 |
| slogPVSA5+slogPVSA5   | sum      |                1 |
| slogPVSA7+slogPVSA7   | sum      |                0 |
| slogPVSA9+slogPVSA9   | sum      |                0 |
| slogPVSA11+slogPVSA11 | sum      |                1 |
| VSAEstate2+VSAEstate2 | sum      |                0 |
| abs(MRVSA5-MRVSA5)    | abs_diff |                0 |
| abs(MRVSA6-MRVSA6)    | abs_diff |                1 |

LASSO vs L1-logistic: 174 vs 68 features, overlap 68.