# Brick 4 - labels

- Historical paper-style rows: **407,699**; class counts {0: 229499, 1: 65100, 2: 113100}. Retained only as source data; no split, model or report uses them.
- Pair-level rows served: **21,042**; class counts {0: 4898, 1: 24, 2: 16120} (11,517 weak class-2 pairs).
- Interaction rows resolved to specific pair-level targets: 6,974 of 23,307; 707 rows used curated food markers, 676 used food-specific compounds; mean targets per row 2.7.
- Label source: {'structured': 13226, 'text_rule': 10081}.

## Why labels changed

The structured `Effect` column is ambiguous: 48% of `Positive` annotations say 'avoid' or warn about increased exposure, so `Positive` cannot mean beneficial. The served pair-level label policy labels `Negative`/`Harmful` as 0, `No Effect` as 2, and decides `Positive`/`Possible` from the sentence: no-effect guard first, then harm, then benefit. Food-level rows map to curated marker compounds instead of every compound of the food (which previously labelled ubiquitous molecules such as glucose as warfarin interactions).

## Rule audit (100 sentences)

|   sampled |   possible_rows |   confirmed_rows |   rule_fired_all |   rule_coverage_all |   confirmed_rule_fired |   confirmed_agreement |   possible_rule_fired |   possible_rule_coverage |   raw_agreement_vs_effect_on_fired | note                                                                                                                                                                                                                                                                           |
|----------:|----------------:|-----------------:|-----------------:|--------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|-----------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|       100 |              76 |               24 |               63 |                0.63 |                     15 |                   0.6 |                    48 |                 0.631579 |                           0.142857 | Confirmed rows test the rules against experimental annotations. Possible rows are labelled from the text rules by design, so their agreement is circular; coverage is reported instead. The CSV in reports/ has an empty manual_label column for the human 100-sentence audit. |

## Held-out case-study pairs

| drug          | compound               | drug_ids   | compound_ids   | supported   |
|:--------------|:-----------------------|:-----------|:---------------|:------------|
| tetracycline  | calcium                | ['D01209'] | [3514]         | True        |
| methotrexate  | ethanol                | ['D00688'] | [753]          | True        |
| diazepam      | ethanol                | ['D00115'] | [753]          | True        |
| nitroglycerin | ethanol                | ['D00266'] | [753]          | True        |
| digoxin       | hyperforin             | ['D00828'] | [2332]         | True        |
| nisoldipine   | bergamottin            | ['D00263'] | [11942]        | True        |
| midazolam     | furanocoumarin         | ['D00245'] | [11942]        | True        |
| warfarin      | vitamin k1 2,3 epoxide | ['D01217'] | [125170]       | True        |
| warfarin      | dimethyl disulfide     | ['D01217'] | [3492]         | True        |

## Held-out clinical probes

| drug          | food           | drug_ids   | compound_ids              |
|:--------------|:---------------|:-----------|:--------------------------|
| warfarin      | grapefruit     | ['D01217'] | [2542, 11869, 11942]      |
| simvastatin   | grapefruit     | ['D00572'] | [2542, 11869, 11942]      |
| midazolam     | grapefruit     | ['D00245'] | [2542, 11869, 11942]      |
| metronidazole | alcohol        | ['D00241'] | [753]                     |
| warfarin      | cranberry      | ['D01217'] | [2571, 2602, 4902, 12389] |
| digoxin       | st john's wort | ['D00828'] | [2332, 12606]             |

Pairs removed from every training table: 23.

The pair-level policy labels compounds named by the interaction text, curated marker compounds of the matched food, or food-specific compounds; other sampled compounds for the same drug are class 2 and flagged weak. The old paper-style rows are kept only in this source table and are not used downstream.