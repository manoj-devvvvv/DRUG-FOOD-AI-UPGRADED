"""Recommendation engine: drug + food -> class probabilities and a sentence.

Usage:
    python src/recommend.py "warfarin" "ethanol"
"""
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from PyBioMed.PyMolecule.moe import GetMOE

from common import ROOT, load_config, load_json, norm_name
from env import load_project_env
from features import DESC_NAMES, pair_matrix
from rules import RULE_VERSION, evaluate_rules
from domain import ApplicabilityDomain
from calibration import acceptance_threshold, apply as apply_calibration
from explain import LocalExplainer
from resolve import NameResolver

RDLogger.DisableLog("rdApp.*")



def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def moe_vector(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        d = GetMOE(mol)
    except Exception:
        return None, None
    arr = np.array([d[k] for k in DESC_NAMES], dtype=np.float64)
    if not np.isfinite(arr).all():
        return None, None
    return mol, arr


def is_inorganic(mol, metals: set[str]) -> bool:
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    return bool(symbols & metals) or "C" not in symbols


def json_safe(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class DFIRecommender:
    def __init__(self, mode: str = "S", resolver: NameResolver | None = None):
        self.cfg = load_config()
        label_cfg = self.cfg["labels"]
        self.labels = {int(k): str(v) for k, v in label_cfg["display"].items()}
        self.severity = {int(k): str(v) for k, v in label_cfg["severity"].items()}
        self.sentences = {int(k): str(v) for k, v in label_cfg["sentences"].items()}
        self.disclaimer = str(label_cfg["disclaimer"])
        self.mode = mode
        self.models = Path(self.cfg["paths"]["models"])
        self.interim = Path(self.cfg["paths"]["interim"])
        self.resolver = resolver or NameResolver(self.interim / "name_resolution_cache.json")

        sel = load_json(self.models / f"selected_features_{mode}.json")
        self.feature_names = [f["name"] for f in sel["features"]]
        import joblib
        import brick8_models
        sys.modules["__main__"].BalancedClassifier = brick8_models.BalancedClassifier
        self.model = joblib.load(self.models / f"best_model_{mode}.joblib")

        self.drugs = pd.read_parquet(self.interim / "drugs_clean.parquet")
        self.foods = pd.read_parquet(self.interim / "foods_clean.parquet")
        self.links = pd.read_parquet(self.interim / "food_links.parquet")
        self.metals = {m.strip() for group in self.cfg["data"]["metals"] for m in group.split(",")}
        serving = self.cfg["serving"]
        self.compound_frequency_cutoff = int(serving["compound_frequency_cutoff"])
        self.max_marker_candidates = int(serving["max_marker_candidates"])
        self.max_food_candidates = int(serving["max_food_candidates"])
        self.max_mineral_candidates = int(serving["max_mineral_candidates"])
        self.max_marker_name_matches = int(serving["max_marker_name_matches"])
        self.food_suffixes = tuple(serving["food_suffixes"])
        self.aliases = dict(serving["aliases"])
        self.smiles_fallback = dict(serving["smiles_fallback"])

        markers = self.models / "food_markers.json"
        self.food_markers = load_json(markers) if markers.exists() else dict(serving["food_markers"])
        train_drugs = self.models / f"train_drugs_{mode}.json"
        self.train_drugs = set(load_json(train_drugs)) if train_drugs.exists() else set()

        meta_path = self.models / f"model_metadata_{mode}.json"
        self.model_metadata = load_json(meta_path) if meta_path.exists() else {
            "model_name": "LightGBM",
            "model_version": "unknown",
            "feature_count": len(self.feature_names),
            "training_split": "unknown",
        }
        self.model_info = {
            "name": self.model_metadata.get("model_name", "LightGBM"),
            "version": self.model_metadata.get("model_version", "unknown"),
            "mode": mode,
            "feature_count": len(self.feature_names),
            "training_split": self.model_metadata.get("training_split"),
        }

        # Dataset evidence layer: non-weak annotations with their source record.
        self.evidence: dict[tuple[str, int], dict] = {}
        evidence_path = Path(self.cfg["paths"]["processed"]) / f"evidence_{mode}.parquet"
        if evidence_path.exists():
            frame = pd.read_parquet(evidence_path)
            for row in frame.to_dict("records"):
                clean = {k: json_safe(v) for k, v in row.items()}
                self.evidence[(str(clean["drug_id"]), int(clean["compound_id"]))] = clean
            self.evidence_source = "evidence_index"
        else:
            self.evidence_source = "labels_only"
            labeled = self.interim / "pairs_labeled.parquet"
            if labeled.exists():
                pairs = pd.read_parquet(labeled)
                pairs = pairs[(pairs["mode"] == mode) & (~pairs["weak"])]
                for d, c, l in pairs[["drug_id", "compound_id", "label"]].itertuples(index=False):
                    self.evidence[(str(d), int(c))] = {
                        "drug_id": str(d), "compound_id": int(c), "label": int(l),
                        "annotation_quality": "label_only", "record_id": None,
                    }

        self.drug_by_name = {}
        for rid, name in zip(self.drugs["drug_id"], self.drugs["name"]):
            self.drug_by_name.setdefault(norm_name(name), []).append(rid)
        self.drug_name = self.drugs.set_index("drug_id")["name"].to_dict()
        self.drug_smiles = self.drugs.set_index("drug_id")["smiles"].to_dict()

        self.compound_by_name = {}
        for cid, name in zip(self.foods["compound_id"], self.foods["name"]):
            self.compound_by_name.setdefault(norm_name(name), []).append(int(cid))
        self.compound_row = self.foods.set_index("compound_id").to_dict("index")

        self.food_to_compounds = {}
        self.compound_to_foods = {}
        for fid, fname, cid in self.links[["food_id", "food_name", "compound_id"]].itertuples(index=False):
            self.food_to_compounds.setdefault(norm_name(fname), []).append(int(cid))
            self.compound_to_foods.setdefault(int(cid), set()).add(norm_name(fname))
        self.food_names = {norm_name(x): x for x in self.links["food_name"].unique()}
        self.compound_freq = (
            self.links.drop_duplicates(["food_id", "compound_id"])
            .groupby("compound_id")["food_id"].nunique()
            .to_dict()
        )

        # Applicability domain with training-only drug and compound references.
        self.domain = ApplicabilityDomain(
            mode, self.models, Path(self.cfg["paths"]["processed"]), self.drugs, self.foods
        )
        self._explainer = None
        self._explainer_lock = threading.Lock()
        self.accept_threshold, self.threshold_info = acceptance_threshold(
            mode, self.models, float(self.cfg["models"]["confidence_threshold"])
        )

    def _local_explainer(self) -> LocalExplainer:
        if self._explainer is None:
            with self._explainer_lock:
                if self._explainer is None:
                    self._explainer = LocalExplainer(self.model)
        return self._explainer

    def warm(self) -> None:
        """Construct the SHAP explainer ahead of the first request."""
        self._local_explainer()

    def _resolve_drug(self, query: str):
        local = self.drug_by_name.get(norm_name(query))
        if local:
            did = local[0]
            return did, self.drug_name[did], self.drug_smiles[did], {
                "smiles": self.drug_smiles[did],
                "source": "local_database",
                "note": None,
            }
        resolution = self.resolver.resolve(query, "drug")
        if resolution.smiles:
            return None, query, resolution.smiles, resolution.as_dict()
        smi = canonical_smiles(query)
        if smi:
            return None, query, smi, {"smiles": smi, "source": "user_smiles", "note": resolution.note}
        return None, None, None, resolution.as_dict()

    def _canonical_food_key(self, query: str) -> str:
        key = norm_name(query)
        key = self.aliases.get(key, key)
        for suffix in self.food_suffixes:
            if key.endswith(suffix):
                key = key[: -len(suffix)].strip()
                break
        return key

    def _marker_ids(self, key: str) -> list[int]:
        for mkey in sorted(self.food_markers, key=len, reverse=True):
            if mkey and mkey in key:
                ids: list[int] = []
                for marker in self.food_markers[mkey]:
                    hits = self.compound_by_name.get(norm_name(marker), [])
                    if not hits:
                        hits = [
                            cid
                            for cname, cids in self.compound_by_name.items()
                            if norm_name(marker) and norm_name(marker) in cname
                            for cid in cids
                        ][:self.max_marker_name_matches]
                    ids.extend(hits)
                if ids:
                    return sorted(set(ids))
        return []

    def _food_link_ids(self, key: str) -> list[int]:
        cids = list(self.food_to_compounds.get(key, []))
        for fkey, ids in self.food_to_compounds.items():
            if not fkey or fkey == key:
                continue
            if (re.search(rf"\b{re.escape(fkey)}\b", key)
                    or re.search(rf"\b{re.escape(key)}\b", fkey)):
                cids.extend(ids)
        return sorted(set(cids))

    def _organic_link_ids(self, food_key: str) -> list[int]:
        cids = [
            c for c in self._food_link_ids(food_key)
            if self.compound_freq.get(c, float("inf")) <= self.compound_frequency_cutoff
            and self.compound_row.get(c, {}).get("is_organic", False)
        ]
        return sorted(cids, key=lambda c: (self.compound_freq.get(c, float("inf")), c))

    def _compounds_in_foods(self, food_keys, exclude: set[int]) -> list[int]:
        found: set[int] = set()
        for food_key in food_keys:
            for cid in self.food_to_compounds.get(food_key, []):
                if cid in exclude or cid in found:
                    continue
                row = self.compound_row.get(cid)
                if row is None or not row.get("is_organic", False):
                    continue
                if self.compound_freq.get(cid, float("inf")) > self.compound_frequency_cutoff:
                    continue
                found.add(cid)
        return sorted(found, key=lambda c: (self.compound_freq.get(c, float("inf")), c))

    def _compound_candidates(self, query: str, resolve_remote: bool = True) -> tuple[list[tuple], dict]:
        key = self.aliases.get(norm_name(query), norm_name(query))
        exact = key in self.compound_by_name
        user_smiles = canonical_smiles(query)
        local: list[tuple[int | None, str, str, str | None]] = []

        if exact:
            cid = self.compound_by_name[key][0]
            row = self.compound_row[cid]
            local = [(cid, row["name"], row["smiles"], None)]
            if not row.get("is_organic", True):
                foods = sorted(self.compound_to_foods.get(cid, set()))
                fallback_ids = self._compounds_in_foods(foods, exclude={cid})
                note = (
                    f"mineral compound; food-level match through "
                    f"{len(foods)} foods containing {row['name']} (specific compounds)"
                )
                for other in fallback_ids[:self.max_mineral_candidates]:
                    other_row = self.compound_row.get(other)
                    if other_row is not None:
                        local.append((int(other), other_row["name"], other_row["smiles"], note))
        elif user_smiles is None:
            fallback = self.smiles_fallback.get(key)
            if fallback:
                local = [(None, key, fallback, "built-in structure (name lookup)")]
            else:
                food_key = self._canonical_food_key(query)
                marker_ids = self._marker_ids(food_key)
                link_ids = self._organic_link_ids(food_key)
                if marker_ids:
                    cids = list(dict.fromkeys(list(marker_ids) + link_ids))[:self.max_marker_candidates]
                    note = (
                        f"food-level match via {self.food_names.get(food_key, query)} "
                        "(marker compounds + specific compounds)"
                    )
                else:
                    cids = link_ids[:self.max_food_candidates]
                    note = f"food-level match via {self.food_names.get(food_key, query)} (specific compounds)"
                for cid in cids:
                    row = self.compound_row.get(int(cid))
                    if row is not None:
                        local.append((int(cid), row["name"], row["smiles"], note))

        if user_smiles is not None:
            return [(None, query, user_smiles, None)], {
                "smiles": user_smiles, "source": "user_smiles", "note": None,
            }

        if local:
            return local, {"smiles": local[0][2], "source": "local_database", "note": None}

        if not resolve_remote:
            return [], {"smiles": None, "source": "local_only", "note": "no local food candidate"}

        resolution = self.resolver.resolve(query, "food")
        out: list[tuple] = []
        if resolution.smiles:
            out.append((None, query, resolution.smiles, "Gemini-resolved structure"))
        return out, resolution.as_dict()

    def _abstain(self, reason_code: str, reason: str, dname, food, **extra) -> dict:
        return {
            "status": "not_supported",
            "prediction_source": "none",
            "abstained": True,
            "abstention_reason": reason_code,
            "drug": dname,
            "food": food,
            "predicted_class": None,
            "predicted_label": None,
            "confidence": None,
            "confidence_type": "not_applicable",
            "probabilities": None,
            "sentence": None,
            "reason": reason,
            "explanation": "The system did not produce a prediction for this input.",
            "model": None,
            "rule": None,
            "evidence": None,
            "applicability_domain": None,
            "calibration": None,
            "explainability": None,
            "resolution": None,
            "disclaimer": self.disclaimer,
            **extra,
        }

    def _evidence_block(self, rec: dict) -> dict:
        return {
            "record_id": json_safe(rec.get("record_id")),
            "annotation": json_safe(rec.get("source_effect")),
            "conclusion": json_safe(rec.get("source_conclusion")),
            "source_food": json_safe(rec.get("source_food")),
            "source_component": json_safe(rec.get("source_component")),
            "relationship": json_safe(rec.get("source_relationship")),
            "pmid": json_safe(rec.get("pmid")),
            "doi": json_safe(rec.get("doi")),
            "reference": json_safe(rec.get("reference")),
            "label_source": json_safe(rec.get("label_source")),
            "annotation_quality": json_safe(rec.get("annotation_quality")),
            "lookup_note": (
                "Existing dataset annotation for this exact drug-compound pair; "
                "this is a lookup, not an independent ML prediction."
            ),
        }

    def _model_context(self, dname, dsmiles, dv, cname, csmiles, cv=None):
        if cv is None:
            _, cv = moe_vector(csmiles)
        if cv is None:
            return None, "descriptor_unavailable"
        ad = self.domain.assess(dsmiles, csmiles)
        _, X = next(pair_matrix(
            np.vstack([dv]), np.vstack([cv]), np.array([0]), np.array([0]),
            names=self.feature_names, chunk_size=1,
        ))
        raw = np.asarray(self.model.predict_proba(X)[0], dtype=np.float64)
        proba, calib = apply_calibration(self.mode, self.models, raw)
        cls = int(np.argmax(proba))
        conf = float(proba[cls])
        explanation = self._local_explainer().explain(
            X, self.feature_names, cls, labels=self.labels
        )
        return {
            "predicted_class": cls,
            "predicted_label": self.labels[cls],
            "probabilities": {int(i): float(p) for i, p in enumerate(proba)},
            "confidence": conf,
            "confidence_type": (
                "calibrated_model_probability" if calib["calibrated"] else "raw_model_probability"
            ),
            "applicability_domain": ad,
            "domain_similarity": ad["drug_similarity"],
            "model": self.model_info,
            "calibration": calib,
            "explainability": explanation,
        }, None

    def _support_block(self, context, reason, answer_source: str, compound=None) -> dict:
        if context is None:
            return {
                "available": False,
                "role": "unavailable",
                "answer_source": answer_source,
                "compound": compound,
                "reason": reason,
                "note": (
                    "The served model cannot score this compound, so no TreeSHAP "
                    "attribution is available for this answer."
                ),
            }
        role = "diagnostic" if answer_source == "diagnostic" else "supporting_context"
        if answer_source == "dataset_evidence":
            note = (
                "TreeSHAP from the served model for the same pair. The answer above comes "
                "from the dataset annotation, not from this model score."
            )
        elif answer_source == "curated_rule":
            note = (
                "TreeSHAP from the served model for the matched compound. The answer above "
                "comes from the curated rule, not from this model score."
            )
        else:
            note = (
                "Diagnostic TreeSHAP only: this pair is not answered, so no class is "
                "predicted; the attributions show how the model would read the structure."
            )
        if context.get("explainability") is None:
            note = (
                "The model could score this pair, but TreeSHAP did not return feature "
                "attributions for this input."
            )
        return {
            "available": True,
            "shap_available": context.get("explainability") is not None,
            "role": role,
            "answer_source": answer_source,
            "compound": compound,
            "predicted_class": context["predicted_class"],
            "predicted_label": context["predicted_label"],
            "probabilities": context["probabilities"],
            "confidence": context["confidence"],
            "confidence_type": context["confidence_type"],
            "model": context["model"],
            "applicability_domain": context["applicability_domain"],
            "calibration": context["calibration"],
            "explainability": context.get("explainability"),
            "note": note,
        }

    def _score_pair(self, did, dname, dsmiles, dv, cid, cname, csmiles, note):
        record = self.evidence.get((str(did), int(cid))) if did is not None and cid is not None else None
        if record is not None:
            known = int(record["label"])
            support_context, support_reason = self._model_context(
                dname, dsmiles, dv, cname, csmiles
            )
            result = {
                "status": "evidence",
                "prediction_source": "dataset_evidence",
                "abstained": False,
                "drug": dname,
                "food": cname,
                "predicted_class": known,
                "predicted_label": self.labels[known],
                "confidence": None,
                "confidence_type": "not_applicable",
                "probabilities": None,
                "sentence": self.sentences[known].replace("A", dname).replace("B", cname),
                "reason": "Dataset annotation for this exact drug-compound pair (non-weak label).",
                "explanation": (
                    "This result is an existing dataset annotation retrieved for the exact "
                    "drug-compound pair. It was not generated by the ML model."
                ),
                "evidence": self._evidence_block(record),
                "model": None,
                "rule": None,
                "applicability_domain": None,
                "calibration": None,
                "explainability": None,
                "model_support": self._support_block(
                    support_context, support_reason, "dataset_evidence", compound=cname
                ),
                "note": note,
                "disclaimer": self.disclaimer,
            }
            if support_context is not None:
                result["explainability"] = support_context["explainability"]
            return self._finalize(result)

        cmol, cv = moe_vector(csmiles)
        if cv is None:
            return self._abstain(
                "descriptor_unavailable", "descriptor computation failed", dname, cname
            )
        if is_inorganic(cmol, self.metals):
            result = self._abstain(
                "inorganic_compound",
                "inorganic compound: no evidence record and no organic fallback",
                dname, cname,
            )
            context, _ = self._model_context(dname, dsmiles, dv, cname, csmiles, cv=cv)
            if context is not None:
                result["explainability"] = context["explainability"]
                result["model_support"] = self._support_block(
                    context, None, "diagnostic", compound=cname
                )
            return result

        context, reason = self._model_context(
            dname, dsmiles, dv, cname, csmiles, cv=cv
        )
        if context is None:
            return self._abstain(reason, "descriptor computation failed", dname, cname)
        ad = context["applicability_domain"]
        if ad["status"] == "outside":
            result = self._abstain(
                "outside_applicability_domain",
                ("Outside applicability domain: " + ad["note"]),
                dname,
                cname,
                applicability_domain=ad,
                domain_similarity=ad["drug_similarity"],
            )
            result["explainability"] = context["explainability"]
            result["model_support"] = self._support_block(context, None, "diagnostic", compound=cname)
            return result

        cls = context["predicted_class"]
        conf = context["confidence"]
        calib = context["calibration"]

        base = {
            "prediction_source": "ml_model",
            "drug": dname,
            "food": cname,
            "predicted_class": cls,
            "predicted_label": self.labels[cls],
            "probabilities": context["probabilities"],
            "confidence": conf,
            "confidence_type": context["confidence_type"],
            "applicability_domain": ad,
            "domain_similarity": ad["drug_similarity"],
            "model": self.model_info,
            "rule": None,
            "evidence": None,
            "calibration": calib,
            "explainability": context["explainability"],
            "food_level_note": note,
            "disclaimer": self.disclaimer,
        }
        if did is not None and self.train_drugs and did not in self.train_drugs:
            base["extrapolation"] = True
            base["note"] = "drug not in this mode's training labels; treat as extrapolation"

        if conf < self.accept_threshold:
            base.update({
                "status": "uncertain",
                "abstained": True,
                "abstention_reason": "insufficient_confidence",
                "sentence": (
                    "The model's leading class is a potentially beneficial interaction signal, "
                    "but confidence is below the acceptance threshold. Do not change medication "
                    "or diet based on this result; ask a pharmacist."
                    if cls == 1 else
                    "The model is not confident; ask a pharmacist before acting."
                ),
                "reason": (
                    f"top probability {conf:.3f} below the validation-selected "
                    f"threshold {self.accept_threshold:.3f}"
                ),
                "explanation": (
                    "The calibrated model probability is below the acceptance threshold, "
                    "so the system abstains instead of treating the top class as a reliable "
                    "recommendation. The positive class has very limited labelled training data."
                    if cls == 1 else
                    "The calibrated model probability is below the acceptance threshold, "
                    "so the system abstains instead of forcing a class."
                ),
            })
            return base

        base.update({
            "status": "ok",
            "abstained": False,
            "sentence": self.sentences[cls].replace("A", dname).replace("B", cname),
            "reason": (
                "Prediction from the trained model using molecular descriptors and the "
                "saved feature configuration."
            ),
            "explanation": (
                "Prediction generated by the ML model from pairwise MOE descriptors. "
                "Confidence is a calibrated model probability."
            ),
        })
        return base

    def _finalize(self, result: dict) -> dict:
        result.setdefault("explainability", None)
        result["shap_available"] = result["explainability"] is not None
        if not result["shap_available"]:
            result["shap_reason"] = (
                result.get("abstention_reason")
                or "no featurizable drug/food structure"
            )
        return result

    def recommend(self, drug: str, food: str) -> dict:
        did, dname, dsmiles, drug_resolution = self._resolve_drug(drug)
        if dsmiles is None:
            return self._finalize(self._abstain(
                "unknown_drug", "drug not found or invalid SMILES", drug, food,
                resolution={"drug": drug_resolution, "food": None},
            ))

        food_key = self._canonical_food_key(food)
        match = evaluate_rules(dname, food_key)
        if match is not None:
            support_context, support_reason, support_compound = None, "descriptor_unavailable", None
            food_resolution = {
                "smiles": None,
                "source": "rule_match",
                "note": "rule matched on the food name",
            }
            _, dv = moe_vector(dsmiles)
            if dv is not None:
                candidates, local_resolution = self._compound_candidates(
                    food, resolve_remote=False
                )
                if candidates:
                    _, compound_name, compound_smiles, compound_note = candidates[0]
                    support_context, support_reason = self._model_context(
                        dname, dsmiles, dv, compound_name, compound_smiles
                    )
                    support_compound = compound_name
                    food_resolution = {
                        "smiles": compound_smiles,
                        "source": local_resolution.get("source", "local_database"),
                        "note": compound_note or "matched compound used for the support explanation",
                    }
            result = {
                "status": "rule",
                "prediction_source": "rule_based",
                "abstained": False,
                "drug": dname,
                "food": food,
                "predicted_class": int(match.label),
                "predicted_label": self.labels[int(match.label)],
                "sentence": match.sentence.replace("A", dname).replace("B", food),
                "reason": match.reason,
                "confidence": None,
                "confidence_type": "not_applicable",
                "probabilities": None,
                "rule": {
                    "rule_id": match.rule_id,
                    "version": match.version,
                    "provenance": match.provenance,
                    "validation_status": match.validation_status,
                },
                "model": None,
                "evidence": None,
                "applicability_domain": None,
                "calibration": None,
                "explainability": None,
                "model_support": self._support_block(
                    support_context, support_reason, "curated_rule", compound=support_compound
                ),
                "resolution": {"drug": drug_resolution, "food": food_resolution},
                "explanation": (
                    "Matched a curated rule layered in front of the model. This is rule-based "
                    "output, not an ML prediction."
                ),
                "disclaimer": self.disclaimer,
            }
            if support_context is not None:
                result["explainability"] = support_context["explainability"]
            return self._finalize(result)

        candidates, food_resolution = self._compound_candidates(food)
        if not candidates:
            return self._finalize(self._abstain(
                "unknown_food", "food compound not found or invalid SMILES", dname, food,
                resolution={"drug": drug_resolution, "food": food_resolution},
            ))

        dmol, dv = moe_vector(dsmiles)
        if dv is None:
            return self._finalize(self._abstain(
                "descriptor_unavailable", "descriptor computation failed", dname, food
            ))

        results = [
            self._score_pair(did, dname, dsmiles, dv, cid, cname, csmiles, note)
            for cid, cname, csmiles, note in candidates
        ]
        supported = [r for r in results if r["status"] != "not_supported"]
        if not supported:
            results[0]["resolution"] = {"drug": drug_resolution, "food": food_resolution}
            return self._finalize(results[0])

        evidence0 = [
            r for r in supported
            if r["status"] == "evidence" and r.get("predicted_class") == 0
        ]
        evidence1 = [
            r for r in supported
            if r["status"] == "evidence" and r.get("predicted_class") == 1
        ]
        evidence_other = [r for r in supported if r["status"] == "evidence"]
        risky = [
            r for r in supported
            if r.get("predicted_class") == 0 and (r.get("confidence") or 0.0) >= self.accept_threshold
        ]
        # A labelled pair is stronger than an unverified model score for a
        # different constituent of the same food. Negative dataset evidence
        # remains first; positive/neutral exact evidence must not be hidden by
        # an unrelated ML-only candidate.
        pool = evidence0 or evidence1 or evidence_other or risky or supported
        best = max(
            pool,
            key=lambda r: 1.0 if r["status"] == "evidence" else (r.get("confidence") or 0.0),
        )
        if len(supported) > 1:
            if best.get("predicted_class") is not None:
                best["sentence"] = (
                    self.sentences[int(best["predicted_class"])]
                    .replace("A", dname).replace("B", food)
                )
            best["note"] = (
                f"food-level match: scored {len(supported)} candidate compounds; "
                "showing the highest-priority result from exact dataset evidence and model scores"
            )
            best["candidates"] = [
                {
                    "compound": r["food"],
                    "class": r.get("predicted_class"),
                    "label": (
                        self.labels.get(int(r["predicted_class"]))
                        if r.get("predicted_class") is not None else None
                    ),
                    "severity": (
                        self.severity.get(int(r["predicted_class"]))
                        if r.get("predicted_class") is not None else None
                    ),
                    "confidence": r.get("confidence"),
                    "status": r["status"],
                    "prediction_source": r.get("prediction_source"),
                    "shap_available": r.get("explainability") is not None,
                    "explainability": r.get("explainability"),
                }
                for r in supported
            ]
        best["resolution"] = {"drug": drug_resolution, "food": food_resolution}
        return self._finalize(best)


def main():
    load_project_env()
    r = DFIRecommender("S")
    drug = sys.argv[1] if len(sys.argv) > 1 else "warfarin"
    food = sys.argv[2] if len(sys.argv) > 2 else "ethanol"
    print(r.recommend(drug, food))


if __name__ == "__main__":
    main()
