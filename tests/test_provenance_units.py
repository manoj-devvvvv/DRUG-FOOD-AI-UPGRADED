"""Unit tests for the provenance, calibration, domain, and split contracts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibration import acceptance_threshold, apply as apply_calibration, temper
from common import load_config, load_json
from rules import RULE_VERSION, RULES, canonical_rule_food_key, evaluate_rules, rules_document
from splits_extra import overlap_check, split_double_cold, split_unseen_compounds
from domain import ApplicabilityDomain, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_rule_registry_is_versioned_and_complete():
    document = rules_document()
    assert document["version"] == RULE_VERSION
    assert len(document["rules"]) == len(RULES) >= 10
    ids = [rule["rule_id"] for rule in document["rules"]]
    assert len(ids) == len(set(ids))
    for rule in document["rules"]:
        assert rule["provenance"]
        assert rule["validation_status"] == "unvalidated_curated"
        assert rule["version"] == RULE_VERSION
        assert rule["label"] in (0, 1, 2)
        assert rule["drug_any"] and rule["food_any"]


def test_rule_matching_preserves_original_semantics():
    assert evaluate_rules("Warfarin", "grapefruit juice").rule_id == "WF-GRAPEFRUIT"
    assert evaluate_rules("warfarin", "bergamottin").rule_id == "WF-GRAPEFRUIT"
    assert evaluate_rules("Simvastatin", "Grapefruit").rule_id == "STATIN-GRAPEFRUIT"
    assert evaluate_rules("metronidazole", "beer").rule_id == "NITRO-ALCOHOL"
    assert evaluate_rules("acetaminophen", "glucose") is None
    assert canonical_rule_food_key("phylloquinone") == "vitamin k"


def test_rule_precedence_is_first_match():
    # Warfarin + garlic matches WF-GARLIC, not the earlier grape rules.
    match = evaluate_rules("warfarin", "garlic")
    assert match.rule_id == "WF-GARLIC"
    # Warfarin + vitamin K matches WF-VITAMIN-K before any generic rules.
    assert evaluate_rules("warfarin", "spinach").rule_id == "WF-VITAMIN-K"


def test_calibration_apply_handles_vector_and_reports_status():
    proba = np.array([0.2, 0.1, 0.7])
    calibrated, info = apply_calibration("S", ROOT / "models", proba)
    assert info["calibrated"] is True
    assert info["method"] == "temperature_scaling"
    assert math.isclose(float(calibrated.sum()), 1.0, abs_tol=1e-9)
    assert calibrated.shape == proba.shape


def test_calibration_falls_back_when_artifact_missing(tmp_path):
    proba = np.array([0.5, 0.25, 0.25])
    calibrated, info = apply_calibration("X", tmp_path, proba)
    assert info["calibrated"] is False
    assert np.allclose(calibrated, proba)
    threshold, metadata = acceptance_threshold("X", tmp_path, 0.6)
    assert threshold == 0.6
    assert metadata["source"] == "config fallback"


def test_temper_preserves_probability_simplex():
    proba = np.array([[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    for temperature in (0.5, 1.0, 2.0):
        out = temper(proba, temperature)
        assert np.allclose(out.sum(axis=1), 1.0)


def test_domain_artifact_thresholds_are_validation_derived():
    artifact = load_json(ROOT / "models" / "domain_S.json")
    assert artifact["validated"] is True
    assert "10th percentile" in artifact["drug"]["rule"]
    assert "10th percentile" in artifact["food"]["rule"]
    assert 0 < artifact["drug"]["threshold"] < 1
    assert 0 < artifact["food"]["threshold"] < 1


def test_domain_references_are_train_only():
    artifact = load_json(ROOT / "models" / "domain_S.json")
    splits = pd.read_parquet(ROOT / "data" / "processed" / "splits_S.parquet")
    g_train = splits[splits["split_G"] == "train"]
    assert set(artifact["reference"]["drug_ids"]) == set(g_train["drug_id"].astype(str))

    f_split = splits.copy()
    f_split["split_F"] = split_unseen_compounds(f_split, load_config())
    f_train = set(f_split.loc[f_split["split_F"] == "train", "compound_id"].astype(int))
    f_held = set(f_split.loc[f_split["split_F"] != "train", "compound_id"].astype(int))
    referenced = set(artifact["reference"]["compound_ids"])
    assert referenced
    assert referenced <= f_train
    assert not (referenced & f_held)


def test_domain_assess_flags_an_unrelated_structure():
    drugs = pd.read_parquet(ROOT / "data" / "interim" / "drugs_clean.parquet")
    foods = pd.read_parquet(ROOT / "data" / "interim" / "foods_clean.parquet")
    domain = ApplicabilityDomain("S", ROOT / "models", ROOT / "data" / "processed", drugs, foods)
    warfarin = drugs.loc[drugs["name"].str.lower() == "warfarin", "smiles"].iloc[0]
    pfo = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
    result = domain.assess(warfarin, pfo)
    assert result["status"] == "outside"
    assert result["food_similarity"] < result["food_threshold"]
    assert fingerprint(pfo) is not None


def test_double_cold_keeps_only_same_partition_pairs():
    cfg = load_config()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "drug_id": [f"D{i}" for i in rng.integers(0, 20, size=300)],
        "compound_id": rng.integers(0, 30, size=300),
    })
    series, info = split_double_cold(frame, cfg)
    kept = frame.assign(split_D=series).dropna(subset=["split_D"])
    assert info["kept"] == len(kept)
    assert info["kept"] + info["dropped"] == len(frame)
    assert overlap_check(kept, "split_D", "drug_id") == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert overlap_check(kept, "split_D", "compound_id") == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_model_metadata_matches_artifact():
    metadata = load_json(ROOT / "models" / "model_metadata_S.json")
    selected = load_json(ROOT / "models" / "selected_features_S.json")
    assert metadata["feature_count"] == selected["n_selected"] == len(selected["features"])
    assert metadata["training_split"]
    assert metadata["model_version"]
