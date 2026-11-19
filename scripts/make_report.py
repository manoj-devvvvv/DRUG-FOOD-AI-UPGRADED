"""Build report.md from the saved result files.

Usage: python scripts/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def js(name):
    path = R / name
    if not path.exists():
        path = ROOT / "models" / name
    return json.loads(path.read_text(encoding="utf-8"))


def table(rows, cols=None):
    df = pd.DataFrame(rows)
    return df[cols].to_markdown(index=False) if cols else df.to_markdown(index=False)


b1 = js("brick1_audit.json")
b2 = js("brick2_funnel.json")
b3 = js("brick3_funnel.json")
b4 = js("brick4_label_report.json")
b5 = js("brick5_features.json")
b6 = js("brick6_splits.json")
sel_log = js("brick7_selection_log_S.json")
sel = js("selected_features_S.json")
b8 = js("brick8_models_S.json")
m9 = js("metrics_S.json")
shap = js("brick10_shap_S.json")
cases = pd.read_csv(R / "case_study_table.csv")
b11 = js("brick11_case_study.json")
b12 = js("brick12_enhancements.json")
cv = pd.read_csv(R / "cv_results_S.csv")
cal = js("calibration_S.json")
gen = js("generalization_S.json")
base = js("baselines_S.json")
dom = js("domain_S.json")
meta = js("model_metadata_S.json")
single = js("single_model_evaluation.json")
sweep = pd.read_csv(R / "class_balance_sweep.csv")
rules_doc = js("rules_v1.json")
csv_manifest_path = ROOT / "data/processed/csv/manifest.json"
csv_manifest = json.loads(csv_manifest_path.read_text(encoding="utf-8")) if csv_manifest_path.exists() else None

lines = []
add = lines.append

add("# Drug-Food Interaction Prediction - Final Report")
add("")
add("Three-class prediction (0 avoid, 1 beneficial/take with food, 2 non-significant) from a drug SMILES "
    "and a food-compound SMILES: MOE-type pair descriptors, four-step selection to 18 features, baseline "
    "comparison, a single deployed LightGBM, calibration, abstention, applicability domain, and per-prediction "
    "SHAP over every feature and every class. The paper-style label mode (Mode P) was retired: it applied one "
    "drug-level label to every compound and is leakage-prone by construction. The one served model uses the "
    "specific label mode (internally still named `S` for reproducibility).")
add("")
add("## 1. Data")
add("")
add(f"- Raw tables: {b1['drugs']['shape'][0]:,} drug rows, {b1['interactions']['shape'][0]:,} interaction rows, "
    f"{b1['foods']['shape'][0]:,} food rows; PubChem text found for 7 of 8 probe drugs.")
add(f"- Drugs: {b2['loaded']:,} loaded, {b2['parsed']:,} parsed, {b2['duplicate_inchikey14_removed']:,} salt/parent duplicates removed, "
    f"**{b2['final_drugs']:,} kept** ({b2['final_with_db_id']:,} with a DrugBank ID).")
add(f"- Food compounds: {b3['unique_compounds']:,} unique structures from {b3['rows_loaded']:,} rows; "
    f"{b3['near_duplicate_tanimoto_removed']:,} near-duplicates removed at Tanimoto 0.75; "
    f"**{b3['kept_after_tanimoto']:,} kept** ({b3['organic_compounds']:,} organic, {b3['inorganic_compounds']:,} inorganic).")
add("")
add("## 2. Labels")
add("")
add(f"- Served mode: **{b4['mode_s_pairs']:,} pairs**, classes {b4['mode_s_class_counts']}. Compounds named by the "
    "interaction text, curated marker compounds, and food-specific compounds carry the interaction label; other "
    "sampled compounds for the same drug are class 2 and flagged weak.")
add(f"- The retired paper-style mode produced {b4['mode_p_pairs']:,} pairs by copying a drug-level label to every "
    "compound. Its models and reports were deleted; the labelled table is kept only as source data.")
add(f"- {b4['interaction_rows_mapped_mode_s']:,} interaction rows mapped to compounds; "
    f"the nine paper case pairs are held out of every training table ({len(b4['holdout_pairs'])} pairs).")
add(f"- Sentence audit: {b4['sentence_audit']['sampled']} sampled rows, "
    f"rule/annotation agreement on confirmed rows {b4['sentence_audit']['confirmed_agreement']:.2f}, "
    f"rule coverage of 'Possible' rows {b4['sentence_audit']['possible_rule_coverage']:.2f}. "
    "The structured FHDI labels disagree with literal text rules often enough that agreement is reported, not hidden.")
add("")
add("## 3. Pair features")
add("")
add(f"- 60 MOE-type descriptors per molecule over {b5['drug_molecules']:,} drugs and {b5['food_molecules']:,} food compounds; "
    "pair vector = 3,600 products + 60 sums + 60 absolute differences + 60 ratios = **3,780 columns**.")
add(f"- Verified against PyBioMed on {b5['verification']['pairs_checked']} pairs: products max difference "
    f"{b5['verification']['max_abs_diff_vs_pybio_products']:.2e}, sums "
    f"{b5['verification']['max_abs_diff_vs_pybio_sums']:.2e}; all values finite: {b5['all_values_finite']}.")
add("")
add("## 4. Splits")
add("")
s = b6["modes"]["S"]
g, r = s["split_G"], s["split_R"]
sp = pd.read_parquet(ROOT / "data" / "processed" / "splits_S.parquet")
dset = {k: set(sp.loc[sp["split_G"] == k, "drug_id"]) for k in ("train", "val", "test")}
overlaps = (len(dset["train"] & dset["val"]) + len(dset["train"] & dset["test"])
            + len(dset["val"] & dset["test"]))
add(f"- Split G drugs train/val/test = {g['train']['drugs']}/{g['val']['drugs']}/{g['test']['drugs']}, "
    f"pairs = {g['train']['rows']:,}/{g['val']['rows']:,}/{g['test']['rows']:,}; "
    f"Split R pairs = {r['train']['rows']:,}/{r['val']['rows']:,}/{r['test']['rows']:,}. "
    f"Split G drug overlaps = {overlaps}. Holdout pairs absent: {b6['heldout_pairs_absent']}.")
add(f"- External set: {b6['external_set']['status']} - {b6['external_set']['decision'].rstrip('.')}.")
add("")
add("## 5. Feature selection")
add("")
add(table([{"step": k, "count": v} for k, v in sel_log["counts"].items()]))
add("")
add(f"Stability (3 subsamples): {sel['stability']['stable_kept_in_final']}/18 final features chosen in at least two runs. "
    "The 18 features were selected on the uncapped training partition and are reused by the deployed class-ratio model.")
add("")
add("## 6. Models")
add("")
best = cv.sort_values("macro_f1", ascending=False).iloc[0]
add(f"- Best CV model: {best['model']} (balanced={bool(best['balanced'])}) macro-F1 {best['macro_f1']:.4f}; "
    f"tuned winner **{b8['best_model']}**, validation macro-F1 {b8['best_val_full_refit']['macro_f1']:.4f} "
    f"after the full-training refit; XGBoost reload difference {b8['xgb_reload_max_prob_diff']:.1e}.")
add("")
add("Baseline CV (Split G grouped folds):")
add("")
add(cv[["model", "balanced", "accuracy", "macro_f1", "recall_0"]].to_markdown(index=False))
add("")
add("## 7. Frozen baseline evaluation (test partition)")
add("")
rows = []
for split in ("G", "R"):
    s = m9["splits"][split]
    rows.append({"split": split, "test pairs": s["test_rows"], "test drugs": s["test_drugs"],
                 "accuracy": round(s["accuracy"], 4), "macro-F1": round(s["macro_f1"], 4),
                 "recall_0": round(s["per_class"]["0"]["recall"], 4),
                 "recall_1": round(s["per_class"]["1"]["recall"], 4),
                 "recall_2": round(s["per_class"]["2"]["recall"], 4)})
add(table(rows))
add("")
add("Split R lets a drug appear in training and test, so it scores much higher. Split G is the number to quote. "
    "This table is the pre-ratio LightGBM frozen during the model comparison; the deployed model is in section 8.")
add("")
add("## 8. Single deployed model and class-ratio sweep")
add("")
add("Class 2 dominates the training rows, so the deployed model caps class 2 in the training partition only. "
    "Validation and test keep the real class distribution, and the test partition is scored once after the "
    "recipe is fixed. Caps below are relative to the class-0 count.")
add("")
add(sweep.to_markdown(index=False))
add("")
add(f"Deployed recipe: **{single['recipe']}**. The validation macro-F1 winner was uncapped, so the deployed cap is a "
    "safety trade-off, not a metric improvement: it raises avoid-class recall at the cost of macro-F1. "
    f"On the held-out Split G test partition: accuracy {single['test']['accuracy']:.4f}, balanced accuracy "
    f"{single['test']['balanced_accuracy']:.4f}, macro-F1 {single['test']['macro_f1']:.4f}. "
    f"Class 0 precision/recall/F1 = {single['test']['per_class']['0']['precision']:.3f}/"
    f"{single['test']['per_class']['0']['recall']:.3f}/{single['test']['per_class']['0']['f1']:.3f}; "
    f"class 2 = {single['test']['per_class']['2']['precision']:.3f}/"
    f"{single['test']['per_class']['2']['recall']:.3f}/{single['test']['per_class']['2']['f1']:.3f}. "
    "Class 1 has only 8 training rows, and its test recall is not a meaningful estimate.")
add("")
add(f"Abstention on test: coverage {single['test_selective']['coverage']:.3f}, accuracy on accepted pairs "
    f"{single['test_selective']['accuracy_on_accepted']:.4f}.")
add("")
add("## 9. Generalization partitions (R / G / F / D)")
add("")
add("Each partition is trained from scratch on its own train split with the deployed LightGBM configuration and "
    "class-2 cap 2:1, then scored once on the untouched test split. Leakage checks count entities that appear in "
    "more than one partition and must be zero for the entity type the split holds out.")
add("")
leak_rows = []
for split in ("G", "F", "D"):
    checks = gen["leakage_checks"][split]
    row = {"split": split, "drug train-test": "-", "drug val-test": "-",
           "compound train-test": "-", "compound val-test": "-"}
    for key, values in checks.items():
        row[f"{key} train-test"] = values["train_test"]
        row[f"{key} val-test"] = values["val_test"]
    leak_rows.append(row)
add(table(leak_rows))
add("")
gen_rows = []
for split in ("R", "G", "F", "D"):
    t = gen["partitions"][split]["metrics"]["test"]

    def num(key):
        value = t.get(key)
        return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

    gen_rows.append({
        "split": split, "test pairs": t["support"],
        "class-1 support": t["class_counts"].get("1", 0),
        "accuracy": num("accuracy"), "balanced acc": num("balanced_accuracy"),
        "macro-F1": num("macro_f1"), "MCC": num("mcc"), "kappa": num("cohen_kappa"),
        "ROC-AUC (OvR)": num("roc_auc_ovr_macro"), "PR-AUC": num("pr_auc_macro"),
    })
add(table(gen_rows))
add("")
add("Split D keeps only the pairs whose independently assigned drug and compound partitions agree, so it is much "
    "smaller than the others and its test class-1 support can be zero; macro metrics there are not reliable and "
    "are reported for completeness only.")
add("")
dc = gen["double_cold_construction"]
add(f"Split D: kept {dc['kept']:,} pairs, dropped {dc['dropped']:,}; "
    f"drugs train/val/test = {dc['drugs']['train']}/{dc['drugs']['val']}/{dc['drugs']['test']}, "
    f"compounds train/val/test = {dc['compounds']['train']}/{dc['compounds']['val']}/{dc['compounds']['test']}.")
add("")
add("## 10. Baselines and class-imbalance experiments")
add("")
add("All models use the same Split G train/validation/test rows and the same 18 features selected on the training "
    "partition. Class weighting and random oversampling are applied to training rows only; validation and test are "
    "never resampled. SMOTE was not used because imbalanced-learn is not a project dependency. These rows are "
    "uncapped comparison experiments; the deployed class-2 cap 2:1 model is reported in section 8.")
add("")
base_rows = []
for name, values in base["metrics"].items():
    t = values["test"]
    base_rows.append({
        "model": name,
        "accuracy": f"{t['accuracy']:.4f}", "balanced acc": f"{t['balanced_accuracy']:.4f}",
        "macro-F1": f"{t['macro_f1']:.4f}", "MCC": f"{t['mcc']:.4f}",
        "class-1 support": t["class_counts"].get("1", 0),
    })
add(table(base_rows))
add("")
counts = base["class_counts"]
add(f"Train class counts {counts['train']}, test {counts['test']}. Class-1 test support is 1, so every class-1 score "
    "in this mode is noise. This is reported as a limitation, not hidden.")
add("")
add("System paths on the same test partition. The evidence lookup returns the dataset label for the same pair; its "
    "100% accuracy is a label echo and it is excluded from model comparison. The hybrid path deliberately does not "
    "use evidence lookup, so it cannot inflate ML-only metrics.")
add("")
sys_rows = []
for path in base["system_paths"]:
    sys_rows.append({
        "path": path["path"], "coverage": f"{path['coverage']:.4f}",
        "accuracy on covered": "n/a" if path["accuracy_on_covered"] is None else f"{path['accuracy_on_covered']:.4f}",
        "note": path["note"],
    })
add(table(sys_rows))
add("")
add("## 11. SHAP")
add("")
top = ", ".join(f"{k} ({v:.3f})" for k, v in list(shap["top_10_features"].items())[:5])
add(f"- Global importance (backend {shap['backend']}): top features {top}.")
add("- Serving explanations use TreeSHAP on the deployed LightGBM and return all 18 feature contributions for all "
    "three classes, not a top-k slice. The UI expands the predicted class and lets the other two classes be "
    "expanded. SHAP describes model behaviour, not biological mechanism.")
add("")
add("## 12. Case studies (nine paper pairs + controls + clinical probes)")
add("")
add(cases.fillna("n/a").to_markdown(index=False))
add("")
add("## 13. Ablations and enhancements (Split G)")
add("")
ablations = {**b12["ablations"], **b12["enhancements"], "18_moe": b12["features"]["18_moe"]}
add(table([{"model": name, "accuracy": round(v["accuracy"], 4), "macro-F1": round(v["macro_f1"], 4),
            "seconds": v.get("seconds")} for name, v in ablations.items()]))
add("")
e2 = b12["e2_calibration_conformal"]
add(f"E2 conformal at a 90% target: test coverage {e2['coverage']:.3f}, abstention rate {e2['abstain_rate']:.3f}, "
    f"selective accuracy {e2['selective_accuracy']:.3f}. "
    f"Drug-level bootstrap 95% CI for macro-F1: [{e2['bootstrap_drug_level']['macro_f1_ci95'][0]:.3f}, "
    f"{e2['bootstrap_drug_level']['macro_f1_ci95'][1]:.3f}].")
add("")
add("The conformal experiment is reported for completeness only; serving uses the temperature-scaled probabilities "
    "and the validation-selected acceptance threshold described in section 16.")
add("")
add("Food-side descriptors alone beat the 18 pair features here; shuffled pairing costs about 0.07 macro-F1. "
    "Both say the signal is dominated by compound identity, which is exactly what the specific labels' weak "
    "class-2 pairs predict.")
add("")
add("## 14. Failure audit and remediation")
add("")
add("The first trained model predicted well-known interactions as non-significant. The audit found three label/resolution "
    "defects and one modelling limit (details in `reports/retune_audit.md`):")
add("")
add("- Raw `Effect=Positive` was mapped to class 1 even when the conclusion was a warning such as 'avoid grapefruit'; "
    "warning-style Positive rows now map to class 0.")
add("- Food-level labels were copied to every compound of the food, so warfarin inherited about 3,000 cranberry/grapefruit "
    "pairs including glucose and UDP-glucose. The specific mode now uses curated marker compounds plus a food-specificity filter.")
add("- Food-name resolution took the first linked compound ('grape' became ethanol) and could not parse juice names.")
add("- The 18 descriptors contain no drug identity, so unseen-drug generalisation is limited (see Split G vs Split R).")
add("")
add(f"After relabelling and retuning, Split G accuracy is {m9['splits']['G']['accuracy']:.4f} "
    f"and macro-F1 is {m9['splits']['G']['macro_f1']:.4f} for the frozen comparison model; the deployed capped model "
    f"scores {single['test']['accuracy']:.4f} accuracy and {single['test']['macro_f1']:.4f} macro-F1.")
add(f"The held-out case table scores {b11['passed']}/{b11['scored_cases']} passes, {b11['failed']} failures, "
    f"{b11['abstained']} abstention and {b11['not_supported']} unsupported rows.")
add("The recommender adds a curated clinical rule layer, exact-pair dataset evidence, marker-based candidate resolution "
    "and conservative aggregation; warfarin + grape juice / grapefruit, simvastatin + grapefruit, midazolam + grapefruit, "
    "metronidazole + alcohol, warfarin + cranberry and digoxin + St John's Wort all return class 0.")
add("")
add("## 15. Decisions where the paper was ambiguous")
add("")
add(f"- Correlated-pair tie-break: keep the feature with the higher ANOVA F-score (LASSO kept "
    f"{sel['l1_logistic_comparison']['lasso_nonzero']} features; L1 multinomial logistic kept "
    f"{sel['l1_logistic_comparison']['l1_logistic_nonzero']}, overlap {sel['l1_logistic_comparison']['overlap']}).")
add("- Three-class LASSO: regression on codes 0/1/2, cross-checked with L1 multinomial logistic.")
add(f"- RFECV chose {sel['rfe_comparison']['rfecv_n_features']} features, so the final set uses RFE fixed at 18.")
add("- Label rules are applied only to the 'Possible' interaction rows; confirmed structured labels are kept as-is.")
add("- Model selection uses validation macro-F1, never test.")
add("- Class-2 capping and the deployed 2:1 ratio are validation-selected in code but deployed as a safety choice; "
    "the report states the macro-F1 cost explicitly.")
add("")
add("Why these numbers differ from the paper:")
add("")
add("- Labels come from FHDI structured annotations, not DrugBank sentences; the paper's 97% accuracy is not "
    "reproducible on this label source.")
add("- The specific label mode removes the shortcut of labelling a whole drug row uniformly, so scores drop sharply.")
add("- Split G holds out entire drugs, so the model cannot memorise drug identity.")
add("- Training used a 50,000-pair cap for CV/tuning and a 40,000-pair cap for the ablation stage because of the "
    "3.7 GB machine, then refit on the full training partition.")
add("")
add("## 16. Calibration, abstention and applicability domain")
add("")
add("Probabilities are temperature-scaled with the temperature fitted on Split G validation, and the acceptance "
    "threshold is chosen on that validation curve as the lowest threshold whose accepted error is at most 0.20. "
    "Neither step uses test data. ECE is expected calibration error over ten reliability bins; lower is better.")
add("")
cal_rows = [{
    "temperature": f"{cal['temperature']:.4f}",
    "threshold": f"{cal['accept_threshold']:.4f}",
    "validation coverage": f"{cal['threshold_validation_coverage']:.4f}",
    "test coverage": f"{cal['test_selective']['coverage']:.4f}",
    "test accuracy on accepted": f"{cal['test_selective']['accuracy_on_accepted']:.4f}",
    "test ECE before": f"{cal['test_before']['ece']:.4f}",
    "test ECE after": f"{cal['test_after']['ece']:.4f}",
    "test Brier before": f"{cal['test_before']['brier']:.4f}",
    "test Brier after": f"{cal['test_after']['brier']:.4f}",
}]
add(table(cal_rows))
add("")
add("The deployed model abstains on roughly half of test pairs and is more accurate on the pairs it accepts. "
    "That is the honest consequence of class-1 scarcity and weak unseen-drug signal: it refuses rather than forcing "
    "a class.")
add("")
dom_rows = [{
    "validated": bool(dom["validated"]),
    "drug threshold": f"{dom['drug']['threshold']:.4f}",
    "drug rule": dom["drug"]["rule"],
    "drug refs": dom["drug"]["reference_count"],
    "food threshold": f"{dom['food']['threshold']:.4f}",
    "food rule": dom["food"]["rule"],
    "food refs": dom["food"]["reference_count"],
}]
add(table(dom_rows))
add("")
add("References are training-only: drug references come from the Split G training drugs and food references from the "
    "Split F training compounds, so held-out entities can never match themselves. Drug similarity is the maximum "
    "Tanimoto to a reference drug and food similarity the maximum to a reference compound (Morgan radius 2, 2,048 "
    "bits). The thresholds are a documented heuristic, not a safety guarantee.")
add("")
add("## 17. Serving contract and provenance")
add("")
add(f"- The deployed artifact is a single {meta['model_name']} {meta['model_version']} "
    f"(`{meta['artifact']}`, sha256 prefix `{meta['artifact_sha256_prefix']}`), 18 features, training split "
    f"`{meta['training_split']}`. There is no mode switch in the API or the UI.")
add("- Input resolution order is: local curated database, Gemini (`GEMINI_API_KEY` from `dfi/.env` or the environment, "
    "shell variables winning, never stored in the repository), PubChem, then a user-supplied SMILES. Gemini answers are canonicalised with RDKit and "
    "rejected when they do not parse; resolutions are cached locally. Every response includes the `resolution` "
    "block showing where each SMILES came from.")
add("- Every response carries `prediction_source`, one of `ml_model`, `dataset_evidence` or `rule_based`, plus "
    "`abstention_reason` when no prediction is returned (`insufficient_confidence`, `outside_applicability_domain`, "
    "`unknown_drug`, `unknown_food`, `inorganic_compound`, `descriptor_unavailable`).")
add("- Rule results carry the rule ID, sentence, reason, provenance, validation status and version and never carry a "
    "model probability or ML confidence. Dataset-evidence results carry the dataset record ID and a lookup note, "
    "also without ML confidence.")
add(f"- Curated rules are versioned as `{rules_doc['version']}` with validation status "
    f"`{rules_doc['validation_status']}`; there are {len(rules_doc['rules'])} rules and each is labelled rule output.")
add("- ML responses carry calibrated class probabilities, the acceptance threshold, the model name/version and "
    "training split, the applicability-domain assessment with thresholds and reference counts, and TreeSHAP "
    "contributions for all 18 features across all three classes, with an explicit caveat that SHAP is not biological causation.")
add("- `/api/health` reports readiness, thresholds, resolver status (Gemini configured or not) and the rule version; "
    "`/api/meta` exposes the model, feature, rule, domain and resolver metadata used by the UI.")
if csv_manifest:
    sizes = ", ".join(f"{name} ({meta['rows']:,} rows)" for name, meta in csv_manifest["files"].items())
    add(f"- Stage CSV exports are written to `data/processed/csv/`: {sizes}.")
add("")
add("## 18. Limitations and safety")
add("")
add("- This is decision support research, not medical advice; every recommendation carries the disclaimer.")
add("- Class-1 support is tiny (one test pair), so class-1 and macro scores are unstable; no target accuracy is promised.")
add("- The majority baseline beats the deployed model on raw accuracy; the model is tuned for macro-F1 and minority "
    "recall, and the class-2 cap is a documented safety trade-off.")
add("- The evidence path echoes the dataset label for the same pair; its accuracy is tautological and it does not "
    "contribute to ML metrics.")
add("- The applicability-domain thresholds are a documented p10 heuristic; an inside-domain result is not a clinical guarantee.")
add("- Inorganic minerals return `not_supported` because the structural descriptors do not apply to them.")
add("- The external Ryu et al. benchmark was not obtained; Split G test plus held-out case studies stand in for it.")
add("- SHAP values describe model behaviour, not pharmacological mechanism.")
add("- Gemini resolution depends on an external service and is validated only structurally, not chemically curated.")
add("")
add("## 19. Reproduce")
add("")
add("```bash")
add("bash scripts/run_all.sh        # bricks 1-17, stage CSVs, report, notebooks and gates")
add("python scripts/make_report.py  # rebuild this report from reports/*")
add("python -m pytest -q tests      # serving, provenance, resolver and feature tests")
add("python scripts/check_gates.py  # project gates")
add("```")
add("")
add("See `reports/implementation_changes.md` for the change log.")

(R / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", R / "report.md")
