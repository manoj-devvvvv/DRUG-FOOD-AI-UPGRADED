# Brick 10 - SHAP (Mode S)

Backend: TreeExplainer; sample: 2,618 test pairs.

## Overall top 10 (mean |SHAP| across classes)

| feature               |   mean_abs_shap |
|:----------------------|----------------:|
| PEOEVSA6+PEOEVSA6     |       0.165421  |
| PEOEVSA7+PEOEVSA7     |       0.158251  |
| PEOEVSA5+PEOEVSA5     |       0.134456  |
| abs(MRVSA6-MRVSA6)    |       0.124469  |
| slogPVSA5+slogPVSA5   |       0.105523  |
| slogPVSA9+slogPVSA9   |       0.090871  |
| slogPVSA11+slogPVSA11 |       0.0876699 |
| slogPVSA1/slogPVSA1   |       0.0771143 |
| VSAEstate6+VSAEstate6 |       0.0755851 |
| VSAEstate2+VSAEstate2 |       0.0736096 |

## Paper's top features

Paper list: MRVSA0, PEOEVSA5, EstateVSA7, PEOEVSA9, EstateVSA2. Present in our top 10: PEOEVSA5+PEOEVSA5.

## Per-class top 5

| negative              | positive            | non_significant     |
|:----------------------|:--------------------|:--------------------|
| PEOEVSA5+PEOEVSA5     | PEOEVSA7+PEOEVSA7   | PEOEVSA6+PEOEVSA6   |
| slogPVSA9+slogPVSA9   | PEOEVSA6+PEOEVSA6   | PEOEVSA7+PEOEVSA7   |
| slogPVSA11+slogPVSA11 | abs(MRVSA6-MRVSA6)  | abs(MRVSA6-MRVSA6)  |
| slogPVSA1/slogPVSA1   | PEOEVSA5+PEOEVSA5   | slogPVSA5+slogPVSA5 |
| slogPVSA5+slogPVSA5   | slogPVSA5+slogPVSA5 | MRVSA9/MRVSA9       |

## Feature definitions

| name                  | term     | drug_descriptor_meaning                                           | food_descriptor_meaning                                           |
|:----------------------|:---------|:------------------------------------------------------------------|:------------------------------------------------------------------|
| MRVSA6*EstateVSA6     | product  | surface-area contribution to molar refractivity                   | surface-area contribution to electrotopological state             |
| PEOEVSA5+PEOEVSA5     | sum      | surface-area contribution to Gasteiger partial charge             | surface-area contribution to Gasteiger partial charge             |
| PEOEVSA6+PEOEVSA6     | sum      | surface-area contribution to Gasteiger partial charge             | surface-area contribution to Gasteiger partial charge             |
| PEOEVSA7+PEOEVSA7     | sum      | surface-area contribution to Gasteiger partial charge             | surface-area contribution to Gasteiger partial charge             |
| VSAEstate6+VSAEstate6 | sum      | VSA-binned electrotopological state contribution                  | VSA-binned electrotopological state contribution                  |
| slogPVSA1/slogPVSA1   | ratio    | van der Waals surface-area contribution to SlogP (hydrophobicity) | van der Waals surface-area contribution to SlogP (hydrophobicity) |
| MRVSA5/MRVSA5         | ratio    | surface-area contribution to molar refractivity                   | surface-area contribution to molar refractivity                   |
| MRVSA9/MRVSA9         | ratio    | surface-area contribution to molar refractivity                   | surface-area contribution to molar refractivity                   |
| slogPVSA9*PEOEVSA8    | product  | van der Waals surface-area contribution to SlogP (hydrophobicity) | surface-area contribution to Gasteiger partial charge             |
| MRVSA3*PEOEVSA8       | product  | surface-area contribution to molar refractivity                   | surface-area contribution to Gasteiger partial charge             |
| MRVSA5*PEOEVSA10      | product  | surface-area contribution to molar refractivity                   | surface-area contribution to Gasteiger partial charge             |
| slogPVSA5+slogPVSA5   | sum      | van der Waals surface-area contribution to SlogP (hydrophobicity) | van der Waals surface-area contribution to SlogP (hydrophobicity) |
| slogPVSA7+slogPVSA7   | sum      | van der Waals surface-area contribution to SlogP (hydrophobicity) | van der Waals surface-area contribution to SlogP (hydrophobicity) |
| slogPVSA9+slogPVSA9   | sum      | van der Waals surface-area contribution to SlogP (hydrophobicity) | van der Waals surface-area contribution to SlogP (hydrophobicity) |
| slogPVSA11+slogPVSA11 | sum      | van der Waals surface-area contribution to SlogP (hydrophobicity) | van der Waals surface-area contribution to SlogP (hydrophobicity) |
| VSAEstate2+VSAEstate2 | sum      | VSA-binned electrotopological state contribution                  | VSA-binned electrotopological state contribution                  |
| abs(MRVSA5-MRVSA5)    | abs_diff | surface-area contribution to molar refractivity                   | surface-area contribution to molar refractivity                   |
| abs(MRVSA6-MRVSA6)    | abs_diff | surface-area contribution to molar refractivity                   | surface-area contribution to molar refractivity                   |

Chemical readings of these features are hypotheses, not confirmed mechanisms.