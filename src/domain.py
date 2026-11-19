"""Applicability domain for drugs and food compounds.

Reference fingerprints are built from the **training partition only**. Two
similarities are reported:

* drug: maximum Tanimoto to a training drug
* food: maximum Tanimoto to a training compound

Thresholds are derived from validation partitions (unseen drugs for the drug
side, unseen compounds for the food side) by the build script and stored in
`models/domain_{mode}.json`. If that artifact is absent, the class falls back
to an explicitly unvalidated default and says so.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from common import load_json

RDLogger.DisableLog("rdApp.*")

DEFAULT_THRESHOLD = 0.30


def _fallback_reference_ids(splits_path: Path) -> tuple[set, set]:
    if not splits_path.exists():
        return set(), set()
    frame = pd.read_parquet(splits_path)
    train = frame[frame["split_G"] == "train"]
    return set(train["drug_id"].astype(str)), set(train["compound_id"].astype(int))


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


class ApplicabilityDomain:
    def __init__(self, mode: str, models_dir: Path, processed_dir: Path, drugs, foods):
        self.mode = mode
        path = Path(models_dir) / f"domain_{mode}.json"
        self.artifact = load_json(path) if path.exists() else None
        if self.artifact:
            self.thresholds = {
                "drug": float(self.artifact["drug"]["threshold"]),
                "food": float(self.artifact["food"]["threshold"]),
            }
            drug_ids = set(self.artifact.get("reference", {}).get("drug_ids", []))
            compound_ids = set(self.artifact.get("reference", {}).get("compound_ids", []))
            self.validated = bool(self.artifact.get("validated", False))
        else:
            self.thresholds = {"drug": DEFAULT_THRESHOLD, "food": DEFAULT_THRESHOLD}
            drug_ids, compound_ids = _fallback_reference_ids(
                Path(processed_dir) / f"splits_{mode}.parquet"
            )
            self.validated = False

        self.drug_fps = self._build(
            drugs.loc[drugs["drug_id"].astype(str).isin(drug_ids), "smiles"]
        )
        self.food_fps = self._build(
            foods.loc[foods["compound_id"].astype(int).isin(compound_ids), "smiles"]
        )
        self.reference_counts = {"drug": len(self.drug_fps), "food": len(self.food_fps)}

    @staticmethod
    def _build(smiles_series) -> list:
        out = []
        for smi in smiles_series:
            fp = fingerprint(smi)
            if fp is not None:
                out.append(fp)
        return out

    @staticmethod
    def _max_similarity(fp, reference: list) -> float | None:
        if fp is None or not reference:
            return None
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, reference)))

    def assess(self, drug_smiles: str, food_smiles: str | None) -> dict:
        drug_fp = fingerprint(drug_smiles)
        food_fp = fingerprint(food_smiles) if food_smiles else None
        drug_sim = self._max_similarity(drug_fp, self.drug_fps)
        food_sim = self._max_similarity(food_fp, self.food_fps)

        if drug_sim is None:
            status = "unknown"
            note = "drug structure could not be fingerprinted"
        elif food_sim is None:
            status = "unknown"
            note = "food structure could not be fingerprinted"
        elif drug_sim < self.thresholds["drug"]:
            status = "outside"
            note = (
                "drug similarity is below the "
                + ("validation-derived" if self.validated else "unvalidated default")
                + " threshold"
            )
        elif food_sim < self.thresholds["food"]:
            status = "outside"
            note = (
                "food similarity is below the "
                + ("validation-derived" if self.validated else "unvalidated default")
                + " threshold"
            )
        else:
            status = "inside"
            basis = "validation-derived" if self.validated else "unvalidated default"
            note = f"drug and food are above the {basis} similarity thresholds"

        return {
            "status": status,
            "drug_similarity": drug_sim,
            "food_similarity": food_sim,
            "drug_threshold": self.thresholds["drug"],
            "food_threshold": self.thresholds["food"],
            "validated": self.validated,
            "reference_counts": self.reference_counts,
            "note": note,
        }
