"""Export human-readable CSV copies of each pipeline stage.

Outputs (data/processed/csv/):
  drug_cleaned.csv        cleaned drug table
  food_cleaned.csv        cleaned food-compound table
  pairs.csv               Mode S labelled pairs with split assignments
  features_generation.csv 60 MOE-type descriptors per molecule
  feature_extraction.csv  the 18 selected pair features for Mode S pairs

The CSV files are derived from the saved parquet artifacts; they do not change
training data. The full 3,780-column pair matrix is intentionally not exported
because it is very large; the selected 18-feature matrix is the one the model
uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_json, save_json, setup_stdout
from features import DESC_NAMES, pair_matrix


def main() -> None:
    setup_stdout()
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    models = Path(cfg["paths"]["models"])
    out = processed / "csv"
    out.mkdir(parents=True, exist_ok=True)

    drugs = pd.read_parquet(interim / "drugs_clean.parquet")
    foods = pd.read_parquet(interim / "foods_clean.parquet")
    splits = pd.read_parquet(processed / "splits_S.parquet")
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")

    drugs.to_csv(out / "drug_cleaned.csv", index=False, encoding="utf-8-sig")
    foods.to_csv(out / "food_cleaned.csv", index=False, encoding="utf-8-sig")

    pairs = (
        splits.merge(drugs[["drug_id", "name"]].rename(columns={"name": "drug_name"}),
                     on="drug_id", how="left")
        .merge(foods[["compound_id", "name"]].rename(columns={"name": "compound_name"}),
               on="compound_id", how="left")
    )
    pairs = pairs[["drug_id", "drug_name", "compound_id", "compound_name", "label",
                   "weak", "sentence_id", "split_R", "split_G"]]
    pairs.to_csv(out / "pairs.csv", index=False, encoding="utf-8-sig")

    drug_names = drugs.set_index("drug_id")["name"].to_dict()
    food_names = foods.set_index("compound_id")["name"].to_dict()
    drug_desc = drug_moe.copy()
    drug_desc.insert(0, "molecule_type", "drug")
    drug_desc.insert(1, "name", drug_desc["molecule_id"].map(drug_names))
    food_desc = food_moe.copy()
    food_desc.insert(0, "molecule_type", "food")
    food_desc.insert(1, "name", food_desc["molecule_id"].map(food_names))
    descriptors = pd.concat([drug_desc, food_desc], ignore_index=True)
    descriptors.to_csv(out / "features_generation.csv", index=False, encoding="utf-8-sig")

    sel = load_json(models / "selected_features_S.json")
    names = [f["name"] for f in sel["features"]]
    dvals = drug_moe[DESC_NAMES].to_numpy()
    fvals = food_moe[DESC_NAMES].to_numpy()
    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}
    keep = splits[splits["drug_id"].isin(drow) & splits["compound_id"].isin(frow)].reset_index(drop=True)
    di = np.array([drow[x] for x in keep["drug_id"]], dtype=np.int64)
    fi = np.array([frow[x] for x in keep["compound_id"]], dtype=np.int64)
    X = np.empty((len(keep), len(names)), dtype=np.float32)
    for start, chunk in pair_matrix(dvals, fvals, di, fi, names=names, chunk_size=20000):
        X[start:start + len(chunk)] = chunk
    extracted = pd.DataFrame(X, columns=names)
    extracted.insert(0, "split_G", keep["split_G"].to_numpy())
    extracted.insert(0, "label", keep["label"].to_numpy())
    extracted.insert(0, "compound_id", keep["compound_id"].to_numpy())
    extracted.insert(0, "drug_id", keep["drug_id"].to_numpy())
    extracted.to_csv(out / "feature_extraction.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "directory": str(out.relative_to(processed.parent.parent)),
        "files": {
            "drug_cleaned.csv": {"rows": int(len(drugs)), "source": "data/interim/drugs_clean.parquet"},
            "food_cleaned.csv": {"rows": int(len(foods)), "source": "data/interim/foods_clean.parquet"},
            "pairs.csv": {"rows": int(len(pairs)), "source": "data/processed/splits_S.parquet + names"},
            "features_generation.csv": {
                "rows": int(len(descriptors)), "columns": len(descriptors.columns),
                "source": "data/interim/{drug,food}_moe.parquet",
            },
            "feature_extraction.csv": {
                "rows": int(len(extracted)), "columns": len(names),
                "source": "splits_S.parquet + selected_features_S.json",
                "note": "Selected 18 features only; the full 3,780-column matrix is not exported.",
            },
        },
    }
    save_json(manifest, out / "manifest.json")
    print({k: v["rows"] for k, v in manifest["files"].items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
