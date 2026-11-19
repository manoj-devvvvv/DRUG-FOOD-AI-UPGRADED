# Brick 9 - evaluation (Mode S)

Frozen model: LightGBM.

## Split G (headline)

- test pairs 2,618; test drugs 55
- accuracy 0.7525; macro-F1 0.5066

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       0 |      0.4964 |   0.3203 | 0.3894 |       640 |
|       1 |      0.1667 |   1      | 0.2857 |         1 |
|       2 |      0.8022 |   0.8923 | 0.8448 |      1977 |

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      205 |        0 |      435 |
| true 1 |        0 |        1 |        0 |
| true 2 |      208 |        5 |     1764 |

## Split R (paper-style, optimistic)

- test pairs 2,597; test drugs 435
- accuracy 0.8036; macro-F1 0.5049

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       0 |      0.5537 |   0.7882 | 0.6505 |       595 |
|       1 |      0      |   0      | 0      |         3 |
|       2 |      0.9272 |   0.8094 | 0.8643 |      1999 |

|        |   pred 0 |   pred 1 |   pred 2 |
|:-------|---------:|---------:|---------:|
| true 0 |      469 |        0 |      126 |
| true 1 |        2 |        0 |        1 |
| true 2 |      376 |        5 |     1618 |
