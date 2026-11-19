"""Build provenance artifacts used by the serving layer.

Outputs:
  data/processed/evidence_{mode}.parquet   dataset evidence with record metadata
  reports/dataset_audit.{json,md}          dataset / label / imbalance audit
  models/model_metadata_{mode}.json        serving model metadata
  models/rules_v1.json                     versioned rule document

This script never changes training labels or model artifacts.
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from brick4_labels import classify_text_corrected
from common import dataframe_to_markdown, load_config, load_json, save_json, setup_stdout
from rules import rules_document

SENTENCE = {
    0: "A should not be taken with food containing B.",
    1: "A could be taken with food containing B.",
    2: "A may be taken with food containing B.",
}
LABEL_NAME = {0: "avoid", 1: "beneficial", 2: "non-significant"}


def rebuild_interactions(cfg: dict) -> pd.DataFrame:
    drugs = pd.read_parquet(Path(cfg["paths"]["interim"]) / "drugs_clean.parquet")
    inter = pd.read_csv(cfg["paths"]["interactions_raw"], encoding="utf-8-sig", low_memory=False)
    inter = inter[inter["Drug_ID"].isin(set(drugs["drug_id"]))].copy()
    inter["effect_class"] = inter["Effect"].map(cfg["labels"]["effect_to_class"])
    inter = inter[inter["effect_class"].notna()].copy()
    inter["row_id"] = np.arange(len(inter))
    inter["effect_norm"] = inter["Effect"].astype(str).str.strip().str.lower()
    inter["rule_corrected"] = inter["Conclusion"].map(classify_text_corrected)
    return inter


def build_evidence(cfg: dict, inter: pd.DataFrame) -> dict[str, pd.DataFrame]:
    processed = Path(cfg["paths"]["processed"])
    out: dict[str, pd.DataFrame] = {}
    for mode in ("S", "P"):
        path = processed / f"splits_{mode}.parquet"
        if not path.exists():
            continue
        pairs = pd.read_parquet(path)
        nonweak = pairs[~pairs["weak"]].copy()
        cols = [
            "row_id", "Drug_ID", "Food_Herb_Name", "Component", "Effect", "Conclusion",
            "PMID", "Reference", "DOI", "Relationship_classification", "effect_norm",
            "rule_corrected",
        ]
        joined = nonweak.merge(inter[cols], left_on="sentence_id", right_on="row_id", how="left")
        text_rule = joined["effect_norm"].isin(["positive", "possible"]) & joined["rule_corrected"].notna()
        joined["label_source"] = np.where(text_rule, "text_rule", "structured_effect")
        if mode == "P":
            joined["annotation_quality"] = "drug_level_annotation_applied_to_compound"
        else:
            joined["annotation_quality"] = np.where(
                text_rule, "text_rule_derived", "structured_effect_annotation"
            )
        evidence = joined[
            [
                "drug_id", "compound_id", "label", "sentence_id", "Drug_ID", "Food_Herb_Name",
                "Component", "Effect", "Conclusion", "PMID", "Reference", "DOI",
                "Relationship_classification", "label_source", "annotation_quality",
            ]
        ].rename(columns={
            "sentence_id": "record_id",
            "Drug_ID": "source_drug_id",
            "Food_Herb_Name": "source_food",
            "Component": "source_component",
            "Effect": "source_effect",
            "Conclusion": "source_conclusion",
            "PMID": "pmid",
            "Reference": "reference",
            "DOI": "doi",
            "Relationship_classification": "source_relationship",
        })
        evidence["mode"] = mode
        evidence.to_parquet(processed / f"evidence_{mode}.parquet", index=False)
        out[mode] = evidence
    return out


def dataset_audit(cfg: dict, inter: pd.DataFrame, evidence: dict[str, pd.DataFrame]) -> dict:
    interim = Path(cfg["paths"]["interim"])
    pairs = pd.read_parquet(interim / "pairs_labeled.parquet")
    drugs = pd.read_parquet(interim / "drugs_clean.parquet")
    foods = pd.read_parquet(interim / "foods_clean.parquet")
    b1 = load_json(Path(cfg["paths"]["reports"]) / "brick1_audit.json")
    b2 = load_json(Path(cfg["paths"]["reports"]) / "brick2_funnel.json")
    b3 = load_json(Path(cfg["paths"]["reports"]) / "brick3_funnel.json")
    b4 = load_json(Path(cfg["paths"]["reports"]) / "brick4_label_report.json")

    conflicts = int((pairs.groupby(["mode", "drug_id", "compound_id"])["label"].nunique() > 1).sum())
    duplicates = int(pairs.duplicated(["mode", "drug_id", "compound_id"]).sum())

    modes = {}
    for mode, group in pairs.groupby("mode"):
        nonweak = group[~group["weak"]]
        modes[mode] = {
            "pairs": int(len(group)),
            "unique_drugs": int(group["drug_id"].nunique()),
            "unique_compounds": int(group["compound_id"].nunique()),
            "class_counts": {int(k): int(v) for k, v in group["label"].value_counts().sort_index().items()},
            "nonweak_class_counts": {
                int(k): int(v) for k, v in nonweak["label"].value_counts().sort_index().items()
            },
            "weak_pairs": int(group["weak"].sum()),
        }

    # Positive-class funnel for Mode S.
    s = pairs[pairs["mode"] == "S"]
    s_nonweak = s[~s["weak"]]
    joined = s_nonweak.merge(
        inter[["row_id", "effect_norm", "Component", "Conclusion"]],
        left_on="sentence_id", right_on="row_id", how="left",
    )
    joined = joined.rename(columns={"Component": "component"})
    positive = joined[joined["label"] == 1]
    positive_rows = inter[inter["effect_norm"] == "positive"]
    funnel = {
        "interaction_rows_after_drug_filter": int(len(inter)),
        "effect_row_counts": {str(k): int(v) for k, v in inter["effect_norm"].value_counts().items()},
        "positive_rows": int(len(positive_rows)),
        "positive_rows_with_component": int(positive_rows["Component"].notna().sum()),
        "positive_rows_labelled_1_after_correction": int(
            (inter.loc[inter["effect_norm"] == "positive", "rule_corrected"] == 1).sum()
        ),
        "positive_rows_labelled_0_after_correction": int(
            inter.loc[inter["effect_norm"] == "positive", "effect_norm"].notna().sum()
            - (inter.loc[inter["effect_norm"] == "positive", "rule_corrected"] == 1).sum()
            - (inter.loc[inter["effect_norm"] == "positive", "rule_corrected"] == 2).sum()
        ),
        "unique_positive_pairs_mode_s": int(len(positive)),
        "unique_positive_drugs_mode_s": int(positive["drug_id"].nunique()),
        "positive_pairs_by_source_effect": {
            str(k): int(v) for k, v in positive["effect_norm"].value_counts().items()
        },
        "positive_pairs_sample": positive[
            ["drug_id", "compound_id", "effect_norm", "Conclusion"]
        ].head(30).to_dict("records"),
    }

    audit = {
        "date": str(date.today()),
        "raw": {
            "drug_rows": int(b1["drugs"]["shape"][0]),
            "drugs_with_smiles": int(b1["drugs"]["has_smiles"]),
            "interaction_rows": int(b1["interactions"]["shape"][0]),
            "food_compound_links": int(b1["foods"]["shape"][0]),
            "unique_food_ids": int(b1["foods"]["unique_food_ids"]),
            "unique_compounds": int(b1["foods"]["unique_compounds"]),
            "effect_counts": b1["interactions"]["effect_counts"],
        },
        "cleaned": {
            "drugs": int(len(drugs)),
            "drugs_missing_smiles": int(drugs["smiles"].isna().sum()),
            "compounds": int(len(foods)),
            "organic_compounds": int(foods["is_organic"].sum()),
            "inorganic_compounds": int((~foods["is_organic"]).sum()),
            "compounds_missing_smiles": int(foods["smiles"].isna().sum()),
            "drug_funnel": b2.get("drugs"),
            "compound_funnel": b3.get("compounds"),
        },
        "pair_integrity": {
            "duplicate_pair_keys": duplicates,
            "conflicting_pair_labels": conflicts,
            "note": "One row per (mode, drug, compound); negative precedence resolves repeats before this check.",
        },
        "modes": modes,
        "positive_class_funnel_mode_s": funnel,
        "evidence_quality": {
            mode: {
                str(k): int(v)
                for k, v in ev["annotation_quality"].value_counts().items()
            }
            for mode, ev in evidence.items()
        },
        "sentence_audit": b4.get("sentence_audit"),
        "external_datasets": [
            {
                "name": "FHDI (current source)",
                "origin": "Food-Herb-Drug Interaction database distributed as Druginfo.csv / InteractionInformation.csv / food_fhdi_clean.csv",
                "license": "Not redistributed here; verify upstream terms before publication.",
                "labels": "Effect column {Positive, Negative, Harmful, No Effect, Possible} + free-text Conclusion",
                "relevance": "Already used; supplies the 24 positive Mode S pairs.",
                "status": "in use",
            },
            {
                "name": "Ryu et al. drug-food interaction supplementary set",
                "origin": "Published drug-food interaction prediction study; supplementary tables",
                "license": "Unknown until verified from the publication",
                "labels": "Binary or three-class drug-food interaction labels",
                "relevance": "Could add drug-food pairs, but label semantics must be mapped and audited first.",
                "status": "not downloaded",
            },
            {
                "name": "DrugBank food-interaction fields (academic release)",
                "origin": "DrugBank academic download",
                "license": "Academic/non-commercial terms; account required",
                "labels": "Free-text food interaction descriptions",
                "relevance": "Could support rule/evidence provenance, not automatically a balanced training set.",
                "status": "not downloaded",
            },
            {
                "name": "FooDB / USDA food composition",
                "origin": "FooDB compound tables",
                "license": "Verify upstream license; non-commercial terms are common",
                "labels": "None (compound composition only)",
                "relevance": "Already represented by the current compound links; no interaction labels.",
                "status": "not downloaded",
            },
        ],
        "external_data_policy": (
            "No external records were ingested. Any future source must be normalized, "
            "audited for duplicates/conflicts, and kept in a separate processed file; "
            "synthetic labels are never treated as verified evidence."
        ),
    }
    return audit


def write_dataset_markdown(audit: dict, path: Path) -> None:
    f = audit["positive_class_funnel_mode_s"]
    lines = [
        "# Dataset audit",
        "",
        f"Generated {audit['date']}. Counts come from the cleaned artifacts and the "
        "rebuilt interaction frame; no records were added.",
        "",
        "## Raw and cleaned data",
        "",
        pd.DataFrame([
            {"stage": "raw drugs", "count": audit["raw"]["drug_rows"]},
            {"stage": "raw interactions", "count": audit["raw"]["interaction_rows"]},
            {"stage": "raw food-compound links", "count": audit["raw"]["food_compound_links"]},
            {"stage": "cleaned drugs", "count": audit["cleaned"]["drugs"]},
            {"stage": "cleaned organic compounds", "count": audit["cleaned"]["organic_compounds"]},
            {"stage": "cleaned inorganic compounds", "count": audit["cleaned"]["inorganic_compounds"]},
        ]).to_markdown(index=False),
        "",
        f"Missing SMILES after cleaning: drugs {audit['cleaned']['drugs_missing_smiles']}, "
        f"compounds {audit['cleaned']['compounds_missing_smiles']}.",
        "",
        "## Pair integrity",
        "",
        f"Duplicate (mode, drug, compound) keys: **{audit['pair_integrity']['duplicate_pair_keys']}**. "
        f"Conflicting labels for the same key: **{audit['pair_integrity']['conflicting_pair_labels']}**.",
        "",
        "## Class distribution",
        "",
    ]
    for mode, m in audit["modes"].items():
        def count(source: dict, cls: int) -> int:
            return int(source.get(cls, source.get(str(cls), 0)))

        lines += [
            f"### Mode {mode}",
            "",
            f"Pairs {m['pairs']:,} · drugs {m['unique_drugs']:,} · compounds {m['unique_compounds']:,} "
            f"· weak class-2 {m['weak_pairs']:,}.",
            "",
            pd.DataFrame([
                {"class": k, "all pairs": count(m["class_counts"], k),
                 "non-weak pairs": count(m["nonweak_class_counts"], k)}
                for k in (0, 1, 2)
            ]).to_markdown(index=False),
            "",
        ]
    lines += [
        "## Why class 1 is almost empty (Mode S)",
        "",
        f"- {f['interaction_rows_after_drug_filter']:,} interaction rows survive the drug filter.",
        f"- {f['positive_rows']:,} rows have `Effect=Positive`; {f['positive_rows_with_component']:,} name a component.",
        f"- Corrected semantics label {f['positive_rows_labelled_1_after_correction']:,} of those rows as beneficial "
        f"and {f['positive_rows_labelled_0_after_correction']:,} as harmful (the sentence says avoid/increase exposure).",
        f"- After compound mapping and negative precedence only **{f['unique_positive_pairs_mode_s']} unique pairs** "
        f"from **{f['unique_positive_drugs_mode_s']} drugs** remain.",
        "",
        "The surviving positives are dominated by a few repeated sentences (for example "
        "'Take with foods containing vitamin C' applied to eight drugs). This is a dataset "
        "limitation: class 1 cannot be learned or evaluated reliably in Mode S.",
        "",
        "Source effect of the 24 positive pairs: "
        f"{f['positive_pairs_by_source_effect']}.",
        "",
        "## Annotation quality of the evidence index",
        "",
    ]
    for mode, q in audit["evidence_quality"].items():
        lines.append(f"- Mode {mode}: {q}")
    lines += [
        "",
        "The 100-sentence audit is a sample, not a validation set. It reports rule coverage and "
        "agreement on confirmed rows only; `Possible` rows are labelled by the same rules and are "
        "circular. The manual label column in `reports/sentence_audit_100.csv` is still empty.",
        "",
        "## External data candidates",
        "",
        pd.DataFrame(audit["external_datasets"]).to_markdown(index=False),
        "",
        audit["external_data_policy"],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def model_metadata(cfg: dict) -> dict:
    models = Path(cfg["paths"]["models"])
    out = {}
    for mode in ("S", "P"):
        sel_path = models / f"selected_features_{mode}.json"
        report_path = Path(cfg["paths"]["reports"]) / f"brick8_models_{mode}.json"
        if not sel_path.exists():
            continue
        sel = load_json(sel_path)
        report = load_json(report_path) if report_path.exists() else {}
        artifact = models / f"best_model_{mode}.joblib"
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()[:16] if artifact.exists() else None
        meta = {
            "mode": mode,
            "model_name": report.get("best_model", "LightGBM"),
            "model_version": "brick8-2026-11-15",
            "artifact": artifact.name,
            "artifact_sha256_prefix": digest,
            "feature_count": len(sel.get("features", [])),
            "features": [x["name"] for x in sel.get("features", [])],
            "training_split": "Split G train (drug-grouped)",
            "selection_partition": "Split G train only",
            "confidence_threshold_uncalibrated": float(cfg["models"]["confidence_threshold"]),
            "validation_macro_f1": report.get("best_val_macro_f1"),
            "source_report": str(report_path.relative_to(Path(cfg["paths"]["reports"]).parent)),
            "created": str(date.today()),
        }
        save_json(meta, models / f"model_metadata_{mode}.json")
        out[mode] = meta
    return out


def main() -> None:
    setup_stdout()
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    reports = Path(cfg["paths"]["reports"])
    models = Path(cfg["paths"]["models"])

    inter = rebuild_interactions(cfg)
    evidence = build_evidence(cfg, inter)
    print("evidence:", {k: len(v) for k, v in evidence.items()})

    audit = dataset_audit(cfg, inter, evidence)
    save_json(audit, reports / "dataset_audit.json")
    write_dataset_markdown(audit, reports / "dataset_audit.md")
    print("audit modes:", {m: v["class_counts"] for m, v in audit["modes"].items()})
    print("positive pairs:", audit["positive_class_funnel_mode_s"]["unique_positive_pairs_mode_s"])

    meta = model_metadata(cfg)
    print("model metadata:", {m: (v["model_name"], v["feature_count"]) for m, v in meta.items()})

    save_json(rules_document(), models / "rules_v1.json")
    print("rules:", len(rules_document()["rules"]))


if __name__ == "__main__":
    main()
