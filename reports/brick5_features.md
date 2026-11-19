# Brick 5 - feature extraction

- MOE descriptors per molecule: **60**.
- Pair columns: **3,780** = 3,600 products + 60 sums + 60 abs-diffs + 60 ratios.
- Drug descriptors kept: 1359 (unique vectors); duplicate vectors removed: 0.
- Food descriptors kept: 4877 (unique vectors); duplicate vectors removed: 0.
- All values finite: True.

## Verification on 500 random pairs

  value
    500
   3780
 0.0005
    0.0
    0.0
   True
   True
   True

PyBioMed rounds interaction values to 3 decimals, so parity is judged at 1e-3; the unrounded builder is also checked against direct arithmetic at 1e-9 (float64 verification). Production pair matrices are stored as float32.

Descriptor and pair-feature failures are listed in `brick5_features.json`.