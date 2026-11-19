"""Serving-contract tests: every prediction path, provenance, abstention, schema.

The tests use the real FastAPI app, the saved model artifacts, and the real
dataset/evidence tables. They are intentionally integration-level: the point is
to prove which mechanism produced each response, not to re-score the model.
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from web.app import app

OUTSIDE_FOOD = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


def predict(client, drug, food):
    response = client.post("/api/predict", json={"drug": drug, "food": food})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_exposes_model_and_thresholds(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ready"
    assert data["model"]["name"]
    assert data["model"]["version"]
    assert data["rules_version"]
    assert 0.0 < data["accept_threshold"] <= 1.0
    assert isinstance(data["resolution"]["gemini_configured"], bool)
    assert data["resolution"]["order"][0] == "local database"
    assert "GEMINI_API_KEY" not in str(data)


def test_rule_path_is_labelled_and_has_no_fake_confidence(client):
    data = predict(client, "simvastatin", "grapefruit")
    assert data["status"] == "rule"
    assert data["prediction_source"] == "rule_based"
    assert data["predicted_class"] == 0
    assert data["confidence"] is None
    assert data["confidence_type"] == "not_applicable"
    assert data["probabilities"] is None
    assert data["model"] is None
    assert data["rule"]["rule_id"]
    assert data["rule"]["version"]
    assert data["rule"]["provenance"]


def test_evidence_path_is_a_lookup_not_an_ml_prediction(client):
    data = predict(client, "simvastatin", "Cyanidin")
    assert data["status"] == "evidence"
    assert data["prediction_source"] == "dataset_evidence"
    assert data["confidence"] is None
    assert data["probabilities"] is None
    assert data["model"] is None
    assert data["evidence"]["record_id"] is not None
    assert "lookup" in data["evidence"]["lookup_note"].lower()


def test_positive_dataset_pair_is_retrieved_through_food_marker(client):
    # The source annotation is Chromium + Vitamin C (Ascorbic acid), with
    # Grape listed as the food. The serving marker must preserve that mapping.
    data = predict(client, "Chromium", "grape")
    assert data["status"] == "evidence"
    assert data["prediction_source"] == "dataset_evidence"
    assert data["predicted_class"] == 1
    assert data["predicted_label"] == "Beneficial / take with food"
    assert data["evidence"]["record_id"] == 132


def test_ml_path_has_calibrated_confidence_and_local_shap(client):
    data = predict(client, "acetaminophen", "glucose")
    assert data["status"] == "ok"
    assert data["prediction_source"] == "ml_model"
    assert data["abstained"] is False
    assert data["confidence_type"] == "calibrated_model_probability"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model"]["name"]
    assert data["model"]["version"]
    assert data["applicability_domain"]["status"] == "inside"
    rows = data["probability_rows"]
    assert [row["class"] for row in rows] == [0, 1, 2]
    assert math.isclose(sum(row["probability"] for row in rows), 1.0, abs_tol=1e-6)
    assert data["calibration"]["calibrated"] is True
    shap = data["explainability"]
    assert shap["method"] == "tree_shap"
    assert 0 < len(shap["top_features"]) <= 6
    assert all(math.isfinite(f["shap_value"]) for f in shap["top_features"])
    assert "not evidence of a biological mechanism" in shap["caveat"]
    # Every class and every feature must be explained, not just a top-k slice.
    assert shap["feature_count"] == data["model"]["feature_count"]
    assert {int(k) for k in shap["by_class"]} == {0, 1, 2}
    for cls, block in shap["by_class"].items():
        assert block["feature_count"] == data["model"]["feature_count"]
        assert len(block["contributions"]) == data["model"]["feature_count"]
        assert block["label"]
    assert len(shap["features"]) == data["model"]["feature_count"]


@pytest.mark.parametrize(
    "drug,food,status",
    [
        ("acetaminophen", "glucose", "ok"),
        ("aspirin", "calcium", "uncertain"),
        ("simvastatin", "Cyanidin", "evidence"),
        ("simvastatin", "grapefruit", "rule"),
        ("warfarin", OUTSIDE_FOOD, "not_supported"),
        ("warfarin", "[Na+]", "not_supported"),
    ],
)
def test_every_answered_path_carries_full_shap(client, drug, food, status):
    data = predict(client, drug, food)
    assert data["status"] == status
    shap = data["explainability"]
    assert shap is not None
    assert data["shap_available"] is True
    feature_count = data["model"]["feature_count"] if data["model"] else \
        data["model_support"]["model"]["feature_count"]
    assert shap["feature_count"] == feature_count
    assert len(shap["features"]) == feature_count
    assert {int(k) for k in shap["by_class"]} == {0, 1, 2}
    for block in shap["by_class"].values():
        assert len(block["contributions"]) == feature_count
    if status in {"evidence", "rule"}:
        assert data["model"] is None
        assert data["model_support"]["available"] is True
        assert data["model_support"]["role"] == "supporting_context"
        assert data["model_support"]["compound"]
    if status == "not_supported":
        assert data["model_support"]["role"] == "diagnostic"


def test_outside_applicability_domain_abstains_without_prediction(client):
    data = predict(client, "warfarin", OUTSIDE_FOOD)
    assert data["status"] == "not_supported"
    assert data["prediction_source"] == "none"
    assert data["abstained"] is True
    assert data["abstention_reason"] == "outside_applicability_domain"
    assert data["predicted_class"] is None
    assert data["confidence"] is None
    assert data["applicability_domain"]["status"] == "outside"
    assert data["applicability_domain"]["validated"] is True


def test_mineral_with_dataset_evidence_returns_annotation(client):
    data = predict(client, "warfarin", "calcium")
    assert data["status"] == "evidence"
    assert data["prediction_source"] == "dataset_evidence"
    assert data["predicted_class"] == 0
    assert data["confidence"] is None
    assert data["evidence"]["lookup_note"]


def test_mineral_without_evidence_falls_back_to_foods(client):
    data = predict(client, "metformin", "calcium")
    assert data["status"] != "not_supported"
    assert data["prediction_source"] in {"ml_model", "dataset_evidence", "rule_based"}
    assert data["predicted_class"] is not None


def test_milk_resolves_to_marker_and_organic_compounds(client):
    data = predict(client, "warfarin", "milk")
    assert data["status"] != "not_supported"
    assert data["predicted_class"] is not None
    candidates = data.get("candidates", [])
    assert candidates
    assert all(row["label"] for row in candidates)
    assert all(row["explainability"] is not None for row in candidates)
    assert all(row["shap_available"] is True for row in candidates)


def test_unknown_drug_and_food_are_reported_separately(client):
    unknown_drug = predict(client, "notadrugxyz", "glucose")
    assert unknown_drug["abstention_reason"] == "unknown_drug"
    assert unknown_drug["prediction_source"] == "none"
    assert unknown_drug["shap_available"] is False
    assert unknown_drug["shap_reason"] == "unknown_drug"

    unknown_food = predict(client, "warfarin", "notafoodxyz")
    assert unknown_food["abstention_reason"] == "unknown_food"
    assert unknown_food["shap_available"] is False
    assert unknown_food["shap_reason"] == "unknown_food"


@pytest.mark.parametrize(
    "payload",
    [
        {"drug": "", "food": "glucose"},
        {"drug": "warfarin", "food": "   "},
        {"food": "glucose"},
        {"drug": "a" * 301, "food": "glucose"},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_every_response_declares_a_source_consistent_with_artifacts(client):
    schema_keys = {
        "status", "prediction_source", "drug", "food", "abstained",
        "confidence", "confidence_type", "probabilities", "model", "rule",
        "evidence", "applicability_domain", "calibration", "explainability",
        "explanation", "resolution", "disclaimer",
    }
    cases = [
        ("simvastatin", "grapefruit"),
        ("simvastatin", "Cyanidin"),
        ("acetaminophen", "glucose"),
        ("warfarin", OUTSIDE_FOOD),
        ("warfarin", "calcium"),
    ]
    allowed = {"ml_model", "dataset_evidence", "rule_based", "none"}
    for drug, food in cases:
        data = predict(client, drug, food)
        assert schema_keys <= set(data)
        assert data["prediction_source"] in allowed
        assert isinstance(data["abstained"], bool)
        assert isinstance(data["explanation"], str) and data["explanation"]
        if data["prediction_source"] != "ml_model":
            assert data["model"] is None
            assert data["confidence"] is None
            assert data["probabilities"] is None
        if data["confidence"] is not None:
            assert data["confidence_type"] == "calibrated_model_probability"
        if data["abstained"]:
            assert data["abstention_reason"]
        if data["status"] == "not_supported":
            assert data["predicted_class"] is None
            assert data["abstention_reason"] != "insufficient_confidence"
        if data["status"] == "uncertain":
            assert data["abstention_reason"] == "insufficient_confidence"
            assert data["confidence"] is not None
            assert data["probabilities"] is None  # exposed through probability_rows only


def test_single_model_scores_with_its_calibration(client):
    data = predict(client, "acetaminophen", "glucose")
    assert data["prediction_source"] == "ml_model"
    assert data["calibration"]["calibrated"] is True
    assert data["calibration"]["temperature"] != 1.0


def test_meta_documents_rules_domain_and_abstention(client):
    data = client.get("/api/meta").json()
    assert data["rules_version"]
    assert data["rule_layer"]["version"] == data["rules_version"]
    assert len(data["rule_layer"]["rules"]) >= 10
    assert data["applicability_domain"]["validated"] is True
    assert data["applicability_domain"]["reference_counts"]["drug"] > 0
    assert data["applicability_domain"]["reference_counts"]["food"] > 0
    assert data["abstention"]["accept_threshold"] > 0
    assert len(data["features"]) == data["model"]["feature_count"]


def test_feature_parity_between_selection_and_model(client):
    from common import load_json
    from pathlib import Path

    meta = client.get("/api/meta").json()
    cfg = __import__("common").load_config()
    selected = load_json(Path(cfg["paths"]["models"]) / "selected_features_S.json")
    names = [entry["name"] for entry in selected["features"]]
    assert names == meta["features"]
