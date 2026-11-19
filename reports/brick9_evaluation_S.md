# Brick 9 - evaluation (Mode S)

Frozen model: **LightGBM**.

## Split G (headline: yes)

- Test pairs: 2,618; drugs: 55; accuracy: **0.7525**; macro-F1: **0.5066**.
- Negative recall: 0.3203; precision: 0.4964.

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      205 |        0 |      435 |
| true 1 |        0 |        1 |        0 |
| true 2 |      208 |        5 |     1764 |

Confidence-threshold sweep (answer only when the top probability clears the threshold):

|   threshold |   coverage |   answered |   accuracy |   negative_recall |
|------------:|-----------:|-----------:|-----------:|------------------:|
|         0.5 |   0.990833 |       2594 |   0.755975 |          0.318612 |
|         0.6 |   0.884645 |       2316 |   0.781088 |          0.291902 |
|         0.7 |   0.773491 |       2025 |   0.800494 |          0.25576  |
|         0.8 |   0.623377 |       1632 |   0.822304 |          0.200647 |
|         0.9 |   0.407181 |       1066 |   0.846154 |          0.175141 |

Per-drug spread: 55 test drugs; median accuracy 0.857; p10 0.495; perfect 12; zero 0.

## Split R (headline: no)

- Test pairs: 2,597; drugs: 435; accuracy: **0.8036**; macro-F1: **0.5049**.
- Negative recall: 0.7882; precision: 0.5537.

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      469 |        0 |      126 |
| true 1 |        2 |        0 |        1 |
| true 2 |      376 |        5 |     1618 |

Confidence-threshold sweep (answer only when the top probability clears the threshold):

|   threshold |   coverage |   answered |   accuracy |   negative_recall |
|------------:|-----------:|-----------:|-----------:|------------------:|
|         0.5 |   0.986138 |       2561 |   0.807107 |          0.794872 |
|         0.6 |   0.839815 |       2181 |   0.845942 |          0.808163 |
|         0.7 |   0.687332 |       1785 |   0.886275 |          0.852243 |
|         0.8 |   0.504043 |       1309 |   0.925898 |          0.876404 |
|         0.9 |   0.257605 |        669 |   0.956652 |          0.87619  |

Per-drug spread: 435 test drugs; median accuracy 1.000; p10 0.500; perfect 220; zero 11.

Paper reference: external accuracy 0.9756, internal negative recall 0.8707 and precision 0.9822. The external set was not obtained; numbers are not directly comparable.