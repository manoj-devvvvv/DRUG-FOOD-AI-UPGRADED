# Dataset audit

Generated 2026-11-15. Counts come from the cleaned artifacts and the rebuilt interaction frame; no records were added.

## Raw and cleaned data

| stage                       |   count |
|:----------------------------|--------:|
| raw drugs                   |    1516 |
| raw interactions            |   23950 |
| raw food-compound links     |  838492 |
| cleaned drugs               |    1359 |
| cleaned organic compounds   |    4877 |
| cleaned inorganic compounds |     145 |

Missing SMILES after cleaning: drugs 0, compounds 0.

## Pair integrity

Duplicate (mode, drug, compound) keys: **0**. Conflicting labels for the same key: **0**.

## Class distribution

### Mode P

Pairs 407,699 · drugs 1,359 · compounds 4,877 · weak class-2 0.

|   class |   all pairs |   non-weak pairs |
|--------:|------------:|-----------------:|
|       0 |      229499 |           229499 |
|       1 |       65100 |            65100 |
|       2 |      113100 |           113100 |

### Mode S

Pairs 21,042 · drugs 444 · compounds 4,572 · weak class-2 11,517.

|   class |   all pairs |   non-weak pairs |
|--------:|------------:|-----------------:|
|       0 |        4898 |             4898 |
|       1 |          24 |               24 |
|       2 |       16120 |             4603 |

## Why class 1 is almost empty (Mode S)

- 23,307 interaction rows survive the drug filter.
- 1,822 rows have `Effect=Positive`; 1,319 name a component.
- Corrected semantics label 646 of those rows as beneficial and 1,084 as harmful (the sentence says avoid/increase exposure).
- After compound mapping and negative precedence only **24 unique pairs** from **14 drugs** remain.

The surviving positives are dominated by a few repeated sentences (for example 'Take with foods containing vitamin C' applied to eight drugs). This is a dataset limitation: class 1 cannot be learned or evaluated reliably in Mode S.

Source effect of the 24 positive pairs: {'positive': 12, 'possible': 12}.

## Annotation quality of the evidence index

- Mode S: {'structured_effect_annotation': 5605, 'text_rule_derived': 3920}
- Mode P: {'drug_level_annotation_applied_to_compound': 407699}

The 100-sentence audit is a sample, not a validation set. It reports rule coverage and agreement on confirmed rows only; `Possible` rows are labelled by the same rules and are circular. The manual label column in `reports/sentence_audit_100.csv` is still empty.

## External data candidates

| name                                                | origin                                                                                                             | license                                                           | labels                                                                                  | relevance                                                                          | status         |
|:----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------|
| FHDI (current source)                               | Food-Herb-Drug Interaction database distributed as Druginfo.csv / InteractionInformation.csv / food_fhdi_clean.csv | Not redistributed here; verify upstream terms before publication. | Effect column {Positive, Negative, Harmful, No Effect, Possible} + free-text Conclusion | Already used; supplies the 24 positive Mode S pairs.                               | in use         |
| Ryu et al. drug-food interaction supplementary set  | Published drug-food interaction prediction study; supplementary tables                                             | Unknown until verified from the publication                       | Binary or three-class drug-food interaction labels                                      | Could add drug-food pairs, but label semantics must be mapped and audited first.   | not downloaded |
| DrugBank food-interaction fields (academic release) | DrugBank academic download                                                                                         | Academic/non-commercial terms; account required                   | Free-text food interaction descriptions                                                 | Could support rule/evidence provenance, not automatically a balanced training set. | not downloaded |
| FooDB / USDA food composition                       | FooDB compound tables                                                                                              | Verify upstream license; non-commercial terms are common          | None (compound composition only)                                                        | Already represented by the current compound links; no interaction labels.          | not downloaded |

No external records were ingested. Any future source must be normalized, audited for duplicates/conflicts, and kept in a separate processed file; synthetic labels are never treated as verified evidence.