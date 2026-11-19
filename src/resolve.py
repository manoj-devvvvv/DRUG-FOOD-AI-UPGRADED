"""Name-to-SMILES resolution with Gemini first, then PubChem.

Gemini is asked for every uncached name. Its answer is never trusted directly:
the SMILES must parse with RDKit and canonicalize before it can reach the
model. A cached resolution is reused to avoid repeated API calls. If Gemini is
unavailable or returns an invalid structure, PubChem is queried as a final
check. No API key is stored in the repository; it is read from the environment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PUBCHEM_ENDPOINT = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/CanonicalSMILES,ConnectivitySMILES/JSON"
)
DEFAULT_MODEL = "gemini-flash-lite-latest"
FALLBACK_MODELS = ("gemini-3.1-flash-lite", "gemini-flash-latest")
MAX_NAME_LENGTH = 120


def _http_note(response) -> str:
    note = f"HTTP {response.status_code}"
    try:
        message = response.json().get("error", {}).get("message", "")
    except ValueError:
        message = ""
    return f"{note}: {str(message)[:160]}" if message else note


def _retryable(resolution: "Resolution") -> bool:
    note = resolution.note or ""
    return resolution.smiles is None and (
        note.startswith(("HTTP 404", "HTTP 429", "HTTP 5"))
        or "Timeout" in note
        or note.startswith("ConnectionError")
    )



@dataclass(frozen=True)
class Resolution:
    smiles: str | None
    source: str
    note: str | None = None

    def as_dict(self) -> dict:
        return {"smiles": self.smiles, "source": self.source, "note": self.note}


def canonicalize(smiles: str | None) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def gemini_prompt(name: str, kind: str) -> str:
    return (
        f"Return the single best-matching chemical structure for the {kind} name below. "
        'Reply only as JSON: {"smiles": "<canonical SMILES>"} when a specific chemical '
        'structure exists, or {"smiles": null, "reason": "<short reason>"} when the name is '
        "a mixture, a category, or not a chemical structure. Do not invent a structure.\n"
        f"Name: {name}"
    )


def parse_gemini_response(payload: dict) -> Resolution:
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return Resolution(None, "gemini", "response was not valid JSON")
    smiles = canonicalize(data.get("smiles"))
    if smiles is None:
        reason = data.get("reason") or "no specific chemical structure returned"
        return Resolution(None, "gemini", str(reason)[:200])
    return Resolution(smiles, "gemini")


class NameResolver:
    def __init__(self, cache_path: Path, model: str | None = None, timeout: float = 10.0,
                 online: bool | None = None):
        self.cache_path = Path(cache_path)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.timeout = float(timeout)
        self.online = (os.environ.get("DFI_ONLINE_RESOLVER", "1") != "0") if online is None else bool(online)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    @property
    def gemini_configured(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def _cache_key(self, kind: str, name: str) -> str:
        return f"{kind}|{name.strip().lower()}"

    def _store(self, key: str, resolution: Resolution) -> None:
        self.cache[key] = resolution.as_dict()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        tmp.replace(self.cache_path)

    def _gemini(self, name: str, kind: str) -> Resolution:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return Resolution(None, "gemini", "GEMINI_API_KEY is not set")
        body = {
            "contents": [{"parts": [{"text": gemini_prompt(name, kind)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "smiles": {"type": "STRING", "nullable": True},
                        "reason": {"type": "STRING", "nullable": True},
                    },
                },
            },
        }
        resolution = self._gemini_once(key, self.model, body)
        if not _retryable(resolution):
            return resolution
        for model in FALLBACK_MODELS:
            if model == self.model:
                continue
            retry = self._gemini_once(key, model, body)
            if not _retryable(retry):
                return retry
            resolution = retry
        return resolution

    def _gemini_once(self, key: str, model: str, body: dict) -> Resolution:
        try:
            response = requests.post(
                GEMINI_ENDPOINT.format(model=model),
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return Resolution(None, "gemini", _http_note(response))
            return parse_gemini_response(response.json())
        except (requests.RequestException, ValueError) as exc:
            return Resolution(None, "gemini", f"{type(exc).__name__}: {exc}"[:200])

    @staticmethod
    def _pubchem(name: str, timeout: float) -> Resolution:
        url = PUBCHEM_ENDPOINT.format(name=quote(name))
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "dfi-research/1.0"})
            if response.status_code == 404:
                return Resolution(None, "pubchem", "name not found in PubChem")
            if response.status_code != 200:
                return Resolution(None, "pubchem", f"HTTP {response.status_code}")
            props = response.json()["PropertyTable"]["Properties"][0]
            smiles = canonicalize(props.get("CanonicalSMILES") or props.get("ConnectivitySMILES"))
            if smiles is None:
                return Resolution(None, "pubchem", "no valid SMILES in the response")
            return Resolution(smiles, "pubchem")
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            return Resolution(None, "pubchem", f"{type(exc).__name__}: {exc}"[:200])

    def resolve(self, name: str, kind: str, use_cache: bool = True,
                pubchem_fallback: bool = True) -> Resolution:
        name = (name or "").strip()
        if not name or len(name) > MAX_NAME_LENGTH:
            return Resolution(None, "invalid", "empty or overlong name")
        key = self._cache_key(kind, name)
        if use_cache and key in self.cache:
            entry = self.cache[key]
            smiles = canonicalize(entry.get("smiles"))
            if smiles:
                return Resolution(smiles, f"{entry.get('source', 'cache')}+cache", entry.get("note"))
        if not self.online:
            return Resolution(None, "unresolved", "online resolution disabled")
        gemini = self._gemini(name, kind)
        if gemini.smiles:
            self._store(key, gemini)
            return gemini
        if not pubchem_fallback:
            return Resolution(None, "unresolved", f"gemini: {gemini.note}")
        pubchem = self._pubchem(name, self.timeout)
        if pubchem.smiles:
            self._store(key, pubchem)
            return pubchem
        return Resolution(None, "unresolved", f"gemini: {gemini.note}; pubchem: {pubchem.note}")
