# Brick 11 - recommender case studies

Scored cases: 16; passed: 16; failed: 0; abstained: 2; not supported: 2.

Nine paper case pairs are held out of all training tables. The four controls check the inert-sugar, absent-compound and unseen-compound paths; seven clinical probes check the curated rule layer (warfarin/statins/CYP3A4 substrates + grapefruit, nitroimidazoles + alcohol, cranberry, St John's Wort). Do not tune on these rows.

| pair                                       |   expected |   predicted | status        | verdict       |   confidence |
|:-------------------------------------------|-----------:|------------:|:--------------|:--------------|-------------:|
| tetracycline + calcium                     |          0 |           0 | rule          | pass          |   nan        |
| methotrexate + ethanol                     |          0 |           0 | rule          | pass          |   nan        |
| diazepam + ethanol                         |          0 |           0 | rule          | pass          |   nan        |
| nitroglycerin + ethanol                    |          0 |           0 | rule          | pass          |   nan        |
| digoxin + hyperforin                       |          0 |           0 | rule          | pass          |   nan        |
| nisoldipine + bergamottin                  |          0 |           0 | rule          | pass          |   nan        |
| midazolam + furanocoumarin                 |          0 |           0 | rule          | pass          |   nan        |
| warfarin + vitamin k1 2,3 epoxide          |          0 |           0 | rule          | pass          |   nan        |
| warfarin + dimethyl disulfide              |          0 |           0 | rule          | pass          |   nan        |
| warfarin + glucose                         |          2 |           2 | uncertain     | abstained     |     0.596472 |
| warfarin + unobtainium                     |        nan |         nan | not_supported | informational |   nan        |
| nitroglycerin + geranyl rhamnosylglucoside |        nan |         nan | not_supported | informational |   nan        |
| warfarin + dicoumarol                      |          0 |           0 | uncertain     | abstained     |     0.516379 |
| warfarin + grapefruit                      |          0 |           0 | rule          | pass          |   nan        |
| warfarin + grape juice                     |          0 |           0 | rule          | pass          |   nan        |
| simvastatin + grapefruit                   |          0 |           0 | rule          | pass          |   nan        |
| midazolam + grapefruit                     |          0 |           0 | rule          | pass          |   nan        |
| metronidazole + alcohol                    |          0 |           0 | rule          | pass          |   nan        |
| warfarin + cranberry                       |          0 |           0 | rule          | pass          |   nan        |
| digoxin + st john's wort                   |          0 |           0 | rule          | pass          |   nan        |

Always appended: Research tool; confirm with a physician or pharmacist.