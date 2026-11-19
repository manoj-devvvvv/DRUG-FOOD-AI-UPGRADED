# Brick 0 - environment

Run on Python 3.13.7 (the paper used 3.9). PyBioMed 1.0 is used for the MOE-type
descriptors and for the pair-function verification; its PyInteraction module needs NumPy < 2 style
integer indexing, so Brick 5 verifies the vectorised builder against it directly.

| Package | Version |
|---|---|
| scikit-learn | 1.7.2 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| shap | 0.54.1 |
| rdkit | 2026.09.1 |
| PyBioMed | 1.0 |
| pandas | 2.3.3 |
| numpy | 2.3.3 |
