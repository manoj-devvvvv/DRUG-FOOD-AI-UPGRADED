# Brick 3 - food cleaning funnel

 count
838492
838492
 13282
     0
     1
     8
 13273
   309
     7
  7942
  5022
  4877
   145
     7
152320

Protected compounds found: Ethanol, Caffeine, Hyperforin, Dimethyl disulfide, Naringin, Bergamottin, Phylloquinone.

Protected names not found in the food table: tyramine, vitamin k1.

Outputs: `C:\Users\MANOJ DOPPA\Desktop\buildq\dfi\data\interim\foods_clean.parquet` (5022 compounds) and `C:\Users\MANOJ DOPPA\Desktop\buildq\dfi\data\interim\food_links.parquet` (152320 links).

Inorganic and metal-containing compounds are kept in the table but flagged `is_organic=False`; they are excluded from model training and handled by a name rule at inference.

Note: the file header is shifted, so `moldb_smiles` contains InChIKeys. The actual structures are in the `SMILES` column, which is what this brick uses.