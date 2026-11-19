"""Offline unit tests for Gemini-first name resolution.

No test touches the network: Gemini and PubChem are stubbed. The point is to
prove parsing, validation, caching, and fallback behavior are correct.
"""
from __future__ import annotations

from resolve import FALLBACK_MODELS, NameResolver, Resolution, canonicalize, parse_gemini_response


def gemini_payload(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_canonicalize_accepts_valid_and_rejects_invalid_structures():
    assert canonicalize("CCO") == "CCO"
    assert canonicalize("not-a-smiles") is None
    assert canonicalize("") is None
    assert canonicalize(None) is None


def test_parse_gemini_response_returns_validated_smiles():
    resolution = parse_gemini_response(gemini_payload('{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'))
    assert resolution.source == "gemini"
    assert resolution.smiles == "CC(=O)Oc1ccccc1C(=O)O"


def test_parse_gemini_response_rejects_invalid_smiles():
    resolution = parse_gemini_response(gemini_payload('{"smiles": "definitely not a molecule"}'))
    assert resolution.smiles is None
    assert resolution.source == "gemini"
    assert resolution.note


def test_parse_gemini_response_handles_null_and_bad_json():
    assert parse_gemini_response(gemini_payload('{"smiles": null, "reason": "mixture"}')).note == "mixture"
    assert parse_gemini_response(gemini_payload("not json")).smiles is None
    assert parse_gemini_response({}).smiles is None


def test_cached_resolution_does_not_call_network(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("network should not be used for a cached name")

    monkeypatch.setattr(NameResolver, "_gemini", explode)
    monkeypatch.setattr(NameResolver, "_pubchem", staticmethod(explode))
    resolver = NameResolver(tmp_path / "cache.json", online=True)
    resolver._store(
        resolver._cache_key("food", "caffeine"),
        Resolution("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "gemini"),
    )

    result = resolver.resolve("Caffeine", "food")
    assert result.smiles
    assert result.source.endswith("+cache")


def test_missing_api_key_is_reported_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        NameResolver, "_pubchem", staticmethod(lambda *a, **k: Resolution(None, "pubchem", "stub"))
    )
    resolver = NameResolver(tmp_path / "cache.json", online=True)
    result = resolver.resolve("unknownthing", "drug", pubchem_fallback=False)
    assert result.smiles is None
    assert "GEMINI_API_KEY is not set" in result.note


def test_gemini_failure_falls_back_to_pubchem(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(
        NameResolver, "_gemini", lambda self, name, kind: Resolution(None, "gemini", "HTTP 500")
    )
    monkeypatch.setattr(
        NameResolver, "_pubchem", staticmethod(lambda name, timeout: Resolution("CCO", "pubchem"))
    )
    resolver = NameResolver(tmp_path / "cache.json", online=True)
    result = resolver.resolve("ethanolish", "food")
    assert result.source == "pubchem"
    assert result.smiles == "CCO"
    cached = resolver.resolve("ethanolish", "food")
    assert cached.source.endswith("+cache")


def test_offline_mode_disables_network_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("DFI_ONLINE_RESOLVER", "0")
    resolver = NameResolver(tmp_path / "cache.json")
    result = resolver.resolve("already-known-to-nobody", "drug")
    assert result.smiles is None
    assert result.source == "unresolved"
    assert "disabled" in result.note


def test_overlong_names_are_rejected(tmp_path):
    resolver = NameResolver(tmp_path / "cache.json", online=False)
    result = resolver.resolve("x" * 500, "drug")
    assert result.source == "invalid"


def test_retired_model_retries_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    calls = []

    def fake(self, key, model, body):
        calls.append(model)
        if len(calls) == 1:
            return Resolution(None, "gemini", "HTTP 404: model is no longer available")
        return Resolution("CCO", "gemini")

    monkeypatch.setattr(NameResolver, "_gemini_once", fake)
    resolver = NameResolver(tmp_path / "cache.json", model="gemini-2.5-flash", online=True)
    result = resolver.resolve("ethanolish", "food", pubchem_fallback=False)
    assert result.smiles == "CCO"
    assert calls == ["gemini-2.5-flash", FALLBACK_MODELS[0]]


def test_valid_null_answer_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    calls = []

    def fake(self, key, model, body):
        calls.append(model)
        return Resolution(None, "gemini", "mixture")

    monkeypatch.setattr(NameResolver, "_gemini_once", fake)
    resolver = NameResolver(tmp_path / "cache.json", online=True)
    result = resolver.resolve("grapefruit juice", "food", pubchem_fallback=False)
    assert result.smiles is None
    assert calls == [resolver.model]
