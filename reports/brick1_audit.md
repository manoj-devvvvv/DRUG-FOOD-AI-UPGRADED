# Brick 1 - data audit

- Drugs: **1516** rows, 1516 unique FHDI IDs.
- Interactions: **23950** rows, 1507 drugs, 1336 foods/herbs, 253 components.
- Food table: **838492** rows, 13282 compounds, 177 foods.

## Column check (first row)

Drug columns: FHDI_Drug_ID, Drug_Name, Synonyms, Drug_Type, Company, Summary, Formula, Canonical_SMILES, InChI, InChIKey, TTD_ID, DrugBank_ID, PubChem_Compound_ID, PubChem_Substance_ID, ChEBI_ID, CAS_Number, ADReCS_Drug_ID

Interaction columns: Drug_ID, Drug_Name, Food_Herb_ID, Food_Herb_Name, Component, Note, Result, Experimental_Species, Experimental_Design, Effect, Potential_Target, Target_ID, Conclusion, PMID, Reference, DOI, Relationship_classification

Food columns: food_id, id, name, moldb_smiles, SMILES, Food_Name, orig_food_common_name, kingdom, superklass, klass, subklass

## Food header misalignment

`moldb_smiles` contains InChIKeys and `SMILES` contains structures; the two columns agree 0 times. All later code uses the `SMILES` column. This is the header/value misalignment the plan warned about.

## Label source

Primary labels come from the structured FHDI interaction table (`Effect`, `Relationship_classification`, `Conclusion`), which carries DrugBank-derived rows. PubChem PUG-View was probed on 5-8 CIDs as a cross-check:

    cid         drug                                                  sections  n_strings                                                                                                                           sample
   3394 Flurbiprofen [/Drug and Medication Information/Drug-Food Interactions]          2                                                                                                                   Avoid alcohol.
  54454  Simvastatin [/Drug and Medication Information/Drug-Food Interactions]          1 Avoid grapefruit products. Co-administration with grapefruit products may increase the risk for adverse effects such as myalgia.
   4192    Midazolam [/Drug and Medication Information/Drug-Food Interactions]          2                                                            Avoid alcohol. Midazolam may make your sleepiness or dizziness worse.
  71353   Dapoxetine                                                        []          0                                                                                                                                 
5284373 Cyclosporine [/Drug and Medication Information/Drug-Food Interactions]          4                                                                                                       Avoid grapefruit products.
  68911   Artemether [/Drug and Medication Information/Drug-Food Interactions]          2                                                                                                       Avoid grapefruit products.
   4091    Metformin [/Drug and Medication Information/Drug-Food Interactions]          2                                                                                                                   Avoid alcohol.
6918837 Panobinostat [/Drug and Medication Information/Drug-Food Interactions]          5    Avoid grapefruit products. Grapefruit inhibits CYP3A4 metabolism, which may increase the serum concentration of panobinostat.

## Effect distribution

   Effect  rows
 Possible 17827
No Effect  2374
 Positive  1915
 Negative  1437
  Harmful   393
 possible     4
