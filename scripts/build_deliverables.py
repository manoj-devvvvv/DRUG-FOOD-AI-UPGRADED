"""Build step_csvs/ and final_metrics/ from saved pipeline artifacts.

This script is read-only with respect to data/ and models/. It never retrains,
never rewrites a model artifact, and never changes a label. Every number written
comes from a saved parquet/JSON/CSV artifact or is recomputed from the saved
molecules with the same Morgan/Tanimoto settings used in cleaning and serving.

Usage:  PYTHONPATH=src python scripts/build_deliverables.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SERVING = CONFIG["serving"]
LABELS = {int(k): str(v) for k, v in CONFIG["labels"]["display"].items()}

from brick8_models import BalancedClassifier  # noqa: F401  (needed to unpickle)
from brick9_evaluate import build_matrix
from calibration import apply as apply_calibration
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
CSV_STAGE = PROCESSED / "csv"
STEP = ROOT / "step_csvs"
METRICS = ROOT / "final_metrics"

SHORT = {0: "Avoid", 1: "Beneficial", 2: "Non-significant"}
SPLIT_DESC = {
    "R": "pair-random (same drugs may appear on both sides)",
    "G": "unseen drugs (whole-drug groups)",
    "F": "unseen food compounds (whole-compound groups)",
    "D": "double-cold (unseen drugs and unseen compounds)",
}
SPLIT_ORDER = ["R", "G", "F", "D"]
METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "cohen_kappa",
    "roc_auc_ovr_macro",
    "pr_auc_macro",
]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
})


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, len(header)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
print("loading artifacts ...")
drugs = pd.read_parquet(INTERIM / "drugs_clean.parquet")
foods = pd.read_parquet(INTERIM / "foods_clean.parquet")
drug_moe = pd.read_parquet(INTERIM / "drug_moe.parquet")
food_moe = pd.read_parquet(INTERIM / "food_moe.parquet")
splits = pd.read_parquet(PROCESSED / "splits_S.parquet")
evidence = pd.read_parquet(PROCESSED / "evidence_S.parquet")
links = pd.read_parquet(INTERIM / "food_links.parquet")
calcium_id = int(foods.loc[foods["name"].str.lower() == "calcium", "compound_id"].iloc[0])
calcium_evidence_rows = int((evidence["compound_id"] == calcium_id).sum())
milk_mask = links["food_name"].str.lower().str.contains("milk", na=False)
milk_foods = sorted(links.loc[milk_mask, "food_name"].unique())
milk_rows = int(milk_mask.sum())
milk_compounds = int(links.loc[milk_mask, "compound_id"].nunique())

gen = read_json(REPORTS / "generalization_S.json")
single = read_json(REPORTS / "single_model_evaluation.json")
calibration = read_json(REPORTS / "calibration_S.json")
calibration_model = read_json(MODELS / "calibration_S.json")
domain = read_json(MODELS / "domain_S.json")
metadata = read_json(MODELS / "model_metadata_S.json")
selected = read_json(MODELS / "selected_features_S.json")
funnel2 = read_json(REPORTS / "brick2_funnel.json")
funnel3 = read_json(REPORTS / "brick3_funnel.json")
label_report = read_json(REPORTS / "brick4_label_report.json")
feature_report = read_json(REPORTS / "brick5_features.json")
split_report = read_json(REPORTS / "brick6_splits.json")
selection_log = read_json(REPORTS / "brick7_selection_log_S.json")
dataset_audit = read_json(REPORTS / "dataset_audit.json")
audit1 = read_json(REPORTS / "brick1_audit.json")

comparison = pd.read_csv(REPORTS / "model_comparison_S.csv")
sweep = pd.read_csv(REPORTS / "class_balance_sweep.csv")
baselines = pd.read_csv(REPORTS / "baselines_S.csv")
feature_table = pd.read_csv(REPORTS / "feature_table_S.csv")

model = joblib.load(MODELS / "best_model_S.joblib")
base = model.base
estimator_params = {k: json_safe(v) for k, v in base.get_params().items()}

# Per-row Split G test predictions come from the deployed artifact, not from the
# pre-deployment evaluation parquets. The parquet files in reports/ are written
# by Brick 9 and use different models (Split R is refit on its own train rows),
# so they cannot be labelled as deployed-model predictions.
FEATURE_NAMES = [f["name"] for f in selected["features"]]
g_test_pairs = splits[splits["split_G"] == "test"]
X_g_test, y_g_test, _, used_g_test = build_matrix(
    g_test_pairs, FEATURE_NAMES, drug_moe, food_moe
)
raw_g_proba = np.asarray(model.predict_proba(X_g_test), dtype=np.float64)
calibrated_g_proba, calibration_info = apply_calibration("S", MODELS, raw_g_proba)
pred_g = pd.DataFrame({
    "drug_id": used_g_test["drug_id"].to_numpy(),
    "compound_id": used_g_test["compound_id"].to_numpy(),
    "label": y_g_test,
    "pred": calibrated_g_proba.argmax(axis=1),
    "p0": calibrated_g_proba[:, 0],
    "p1": calibrated_g_proba[:, 1],
    "p2": calibrated_g_proba[:, 2],
    "p0_raw": raw_g_proba[:, 0],
    "p1_raw": raw_g_proba[:, 1],
    "p2_raw": raw_g_proba[:, 2],
})
g_test_metrics = gen["partitions"]["G"]["metrics"]["test"]
if len(pred_g) != int(g_test_metrics["support"]):
    raise RuntimeError(
        f"deployed Split G test rows {len(pred_g)} do not match saved support "
        f"{g_test_metrics['support']}"
    )
if confusion_matrix(pred_g["label"], pred_g["pred"], labels=[0, 1, 2]).tolist() != \
        g_test_metrics["confusion_matrix"]:
    raise RuntimeError("deployed-model predictions do not reproduce the saved Split G confusion matrix")

STEP.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)
for stale in (STEP / "model_predictions_R.csv", METRICS / "roc_pr_curves_R.png"):
    stale.unlink(missing_ok=True)

step_manifest: dict[str, dict] = {}
metrics_manifest: dict[str, dict] = {}


def register_step(path: Path, description: str, sources: list[str]) -> None:
    rows, cols = csv_shape(path)
    step_manifest[path.name] = {
        "rows": rows,
        "columns": cols,
        "bytes": path.stat().st_size,
        "description": description,
        "sources": sources,
    }


def register_metrics(path: Path, description: str, sources: list[str]) -> None:
    rows, cols = csv_shape(path) if path.suffix.lower() == ".csv" else (None, None)
    metrics_manifest[path.name] = {
        "rows": rows,
        "columns": cols,
        "bytes": path.stat().st_size,
        "description": description,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Canonical stage CSVs
# ---------------------------------------------------------------------------
print("copying canonical stage CSVs ...")
STAGE_FILES = {
    "drug_cleaned.csv": ("Cleaned drug table (one row per drug with a parsed structure).",
                         ["data/interim/drugs_clean.parquet"]),
    "food_cleaned.csv": ("Cleaned food-compound table after InChIKey and Tanimoto near-duplicate removal.",
                         ["data/interim/foods_clean.parquet"]),
    "pairs.csv": ("Mode S labelled drug-compound pairs with Split R and Split G assignments.",
                  ["data/processed/splits_S.parquet", "data/interim/drugs_clean.parquet",
                   "data/interim/foods_clean.parquet"]),
    "features_generation.csv": ("60 PyBioMed MOE descriptors per cleaned molecule (drugs and organic foods).",
                                ["data/interim/drug_moe.parquet", "data/interim/food_moe.parquet"]),
    "feature_extraction.csv": ("The 18 selected pair features for every featurized Mode S pair.",
                               ["data/processed/splits_S.parquet", "models/selected_features_S.json"]),
}
EXPECTED_STAGE_ROWS = {
    "drug_cleaned.csv": len(drugs),
    "food_cleaned.csv": len(foods),
    "pairs.csv": len(splits),
    "features_generation.csv": len(drug_moe) + len(food_moe),
    "feature_extraction.csv": 20947,
}
for name, (description, sources) in STAGE_FILES.items():
    source = CSV_STAGE / name
    target = STEP / name
    if not source.is_file():
        raise FileNotFoundError(f"missing canonical stage CSV: {source}")
    shutil.copyfile(source, target)
    rows, _ = csv_shape(target)
    if rows != EXPECTED_STAGE_ROWS[name]:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_STAGE_ROWS[name]} rows, found {rows}. "
            "Regenerate stage CSVs before building deliverables."
        )
    register_step(target, description, sources)

# ---------------------------------------------------------------------------
# Split assignments, predictions, evidence, comparison tables
# ---------------------------------------------------------------------------
print("building split, prediction, and evidence CSVs ...")
drug_names = drugs.set_index("drug_id")["name"].to_dict()
food_names = foods.set_index("compound_id")["name"].to_dict()

split_assign = splits.copy()
split_assign.insert(1, "drug_name", split_assign["drug_id"].map(drug_names))
split_assign.insert(3, "compound_name", split_assign["compound_id"].map(food_names))
split_assign.insert(4, "label_name", split_assign["label"].map(SHORT))
split_assign = split_assign[[
    "drug_id", "drug_name", "compound_id", "compound_name",
    "label", "label_name", "weak", "sentence_id", "split_R", "split_G",
]]
path = STEP / "split_assignments.csv"
write_csv(split_assign, path)
register_step(path, "Every Mode S pair with its label, weak flag, and Split R/G assignment.",
              ["data/processed/splits_S.parquet", "data/interim/drugs_clean.parquet",
               "data/interim/foods_clean.parquet"])


def prediction_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "split", code)
    out.insert(1, "drug_name", out["drug_id"].map(drug_names))
    out.insert(3, "compound_name", out["compound_id"].map(food_names))
    out.insert(5, "actual_label", out["label"].map(SHORT))
    out.insert(6, "predicted_label", out["pred"].map(SHORT))
    out["confidence"] = out[["p0", "p1", "p2"]].max(axis=1)
    out["correct"] = (out["label"] == out["pred"]).astype(int)
    out["probability_type"] = "temperature_scaled"
    out["temperature"] = calibration_model["temperature"]
    return out[[
        "split", "drug_id", "drug_name", "compound_id", "compound_name",
        "label", "actual_label", "pred", "predicted_label",
        "p0", "p1", "p2", "p0_raw", "p1_raw", "p2_raw",
        "confidence", "correct", "probability_type", "temperature",
    ]]


path = STEP / "model_predictions_G.csv"
write_csv(prediction_frame(pred_g, "G"), path)
register_step(
    path,
    "Deployed artifact best_model_S.joblib scored once on the Split G test pairs; "
    "p0/p1/p2 are temperature-scaled, p*_raw are the estimator probabilities.",
    ["models/best_model_S.joblib", "models/selected_features_S.json",
     "data/processed/splits_S.parquet", "models/calibration_S.json"],
)

evidence_out = evidence.copy()
evidence_out.insert(1, "drug_name", evidence_out["drug_id"].map(drug_names))
evidence_out.insert(3, "compound_name", evidence_out["compound_id"].map(food_names))
path = STEP / "evidence_S.csv"
write_csv(evidence_out, path)
register_step(path, "Dataset evidence lookup table for exact drug-compound pairs (not an ML prediction).",
              ["data/processed/evidence_S.parquet"])

for name, description, sources in [
    ("class_balance_sweep.csv", "Validation scores for the four training class-ratio recipes.",
     ["reports/class_balance_sweep.csv"]),
    ("model_comparison_S.csv", "Five model families compared on Split G and Split R test partitions.",
     ["reports/model_comparison_S.csv"]),
    ("baselines_S.csv", "Majority, logistic, and LightGBM imbalance baselines on Split G.",
     ["reports/baselines_S.csv"]),
]:
    shutil.copyfile(REPORTS / name, STEP / name)
    register_step(STEP / name, description, sources)

# Feature selection funnel and selected features
funnel_rows = [
    {"step": "all pair features", "count": int(selection_log["counts"]["start"]),
     "method": "60 MOE descriptors -> 3,600 cross products + 60 sums + 60 abs diffs + 60 ratios",
     "parameter": ""},
    {"step": "variance filter", "count": int(selection_log["counts"]["after_variance"]),
     "method": "VarianceThreshold", "parameter": "threshold = 0.8"},
    {"step": "correlation filter", "count": int(selection_log["counts"]["after_correlation"]),
     "method": "Pearson |r| pruning", "parameter": "|r| < 0.75"},
    {"step": "Lasso", "count": int(selection_log["counts"]["after_lasso"]),
     "method": "L1 Lasso on standardized features", "parameter": "alpha = 0.01"},
    {"step": "RFECV", "count": int(selection_log["counts"]["after_rfecv"]),
     "method": "Recursive feature elimination with cross-validation", "parameter": "5-fold"},
    {"step": "final RFE", "count": int(selection_log["counts"]["rfe18"]),
     "method": "Recursive feature elimination to the target size", "parameter": "n_features = 18"},
]
funnel_frame = pd.DataFrame(funnel_rows)
path = STEP / "feature_selection_funnel.csv"
write_csv(funnel_frame, path)
register_step(path, "Feature-selection funnel from 3,780 pair features to 18.",
              ["reports/brick7_selection_log_S.json"])

selected_features = feature_table.copy()
selected_features.insert(0, "selection_rank", np.arange(1, len(selected_features) + 1))
path = STEP / "feature_selection.csv"
write_csv(selected_features, path)
register_step(path, "The 18 selected pair features with descriptor meanings and stability hits.",
              ["reports/feature_table_S.csv", "models/selected_features_S.json"])

# Calibration and applicability-domain stage tables
cal_rows = []
for stage in ("validation_before", "validation_after", "test_before", "test_after"):
    block = calibration[stage]
    cal_rows.append({
        "partition": stage.split("_")[0],
        "stage": "raw" if stage.endswith("before") else "calibrated",
        "accuracy": block["accuracy"],
        "nll": block["nll"],
        "brier": block["brier"],
        "ece": block["ece"],
        "mean_confidence": block["mean_confidence"],
    })
calibration_frame = pd.DataFrame(cal_rows)
path = STEP / "calibration_S.csv"
write_csv(calibration_frame, path)
register_step(path, "Temperature-scaling calibration scores before and after calibration.",
              ["reports/calibration_S.json"])

ad_rows = [
    {"side": "drug", "threshold": domain["drug"]["threshold"],
     "reference_count": domain["drug"]["reference_count"],
     "rule": domain["drug"]["rule"],
     "validation_p10": domain["drug"]["validation_distribution"]["p10"]},
    {"side": "food", "threshold": domain["food"]["threshold"],
     "reference_count": domain["food"]["reference_count"],
     "rule": domain["food"]["rule"],
     "validation_p10": domain["food"]["validation_distribution"]["p10"]},
]
ad_frame = pd.DataFrame(ad_rows)
path = STEP / "applicability_domain_S.csv"
write_csv(ad_frame, path)
register_step(path, "Applicability-domain thresholds and reference counts.",
              ["models/domain_S.json"])

# ---------------------------------------------------------------------------
# Tanimoto similarity on the food dataset
# ---------------------------------------------------------------------------
print("computing Morgan/Tanimoto similarity for the food dataset ...")
radius = 2
bits = 2048
cutoff = 0.75
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)

fps = []
valid_rows = []
for row in foods.itertuples(index=False):
    mol = Chem.MolFromSmiles(str(row.smiles))
    if mol is None:
        raise RuntimeError(f"cleaned food compound {row.compound_id} has an unparsable SMILES")
    fps.append(fpgen.GetFingerprint(mol))
    valid_rows.append(row)

n_compounds = len(fps)
neighbor_rows = []
threshold_counts = {0.90: 0, 0.75: 0, 0.50: 0}
similarity_sum = 0.0
similarity_pairs = 0
max_pair_similarity = 0.0
protected_pairs_above_cutoff = 0
nonprotected_pairs_above_cutoff = 0

for i, fp in enumerate(fps):
    raw_sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps), dtype=np.float64)
    sims = raw_sims.copy()
    sims[i] = -1.0
    j = int(np.argmax(sims))
    max_sim = float(sims[j])
    others = sims[sims >= 0.0]
    mean_sim = float(others.mean())
    similarity_sum += float(others.sum())
    similarity_pairs += len(others)
    max_pair_similarity = max(max_pair_similarity, max_sim)
    for value in threshold_counts:
        threshold_counts[value] += int((others >= value).sum())

    row = valid_rows[i]
    for hit in np.where(raw_sims >= cutoff)[0]:
        if hit > i:
            if bool(row.is_protected) or bool(valid_rows[hit].is_protected):
                protected_pairs_above_cutoff += 1
            else:
                nonprotected_pairs_above_cutoff += 1
    nearest = valid_rows[j]
    neighbor_rows.append({
        "compound_id": int(row.compound_id),
        "name": row.name,
        "is_organic": bool(row.is_organic),
        "is_protected": bool(row.is_protected),
        "nearest_compound_id": int(nearest.compound_id),
        "nearest_name": nearest.name,
        "max_tanimoto": max_sim,
        "mean_tanimoto_to_other_compounds": mean_sim,
    })

neighbors = pd.DataFrame(neighbor_rows)
path = STEP / "food_tanimoto_neighbors.csv"
write_csv(neighbors, path)
register_step(path, "Nearest Morgan/Tanimoto neighbour for every cleaned food compound.",
              ["data/interim/foods_clean.parquet"])

reference_ids = {int(x) for x in domain["reference"]["compound_ids"]}
ref_fps = [fps[i] for i in range(n_compounds) if int(valid_rows[i].compound_id) in reference_ids]
if not ref_fps:
    raise RuntimeError("applicability-domain food reference set is empty")

applicability_rows = []
for i, fp in enumerate(fps):
    row = valid_rows[i]
    cid = int(row.compound_id)
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=np.float64)
    max_ref = float(sims.max())
    is_reference = cid in reference_ids
    is_organic = bool(row.is_organic)
    if not is_organic:
        status = "inorganic_abstains"
    elif is_reference:
        status = "training_reference"
    elif max_ref >= float(domain["food"]["threshold"]):
        status = "inside"
    else:
        status = "outside"
    applicability_rows.append({
        "compound_id": cid,
        "name": row.name,
        "is_organic": is_organic,
        "in_training_reference": is_reference,
        "max_tanimoto_to_train_reference": max_ref,
        "food_threshold": float(domain["food"]["threshold"]),
        "domain_status": status,
    })

applicability = pd.DataFrame(applicability_rows)
path = STEP / "food_tanimoto_applicability.csv"
write_csv(applicability, path)
register_step(path, "Food-side applicability-domain similarity to the 2,291 training reference compounds.",
              ["data/interim/foods_clean.parquet", "models/domain_S.json"])

unseen_organic = applicability[
    (~applicability["in_training_reference"]) & (applicability["is_organic"])
]
unseen_below = int((unseen_organic["domain_status"] == "outside").sum())
unseen_inside = int((unseen_organic["domain_status"] == "inside").sum())
unseen_outside = unseen_below
nn_values = neighbors["max_tanimoto"].to_numpy(dtype=float)
summary_rows = [
    {"metric": "cleaned_compounds", "value": n_compounds},
    {"metric": "organic_compounds", "value": int(foods["is_organic"].sum())},
    {"metric": "inorganic_compounds", "value": int((~foods["is_organic"]).sum())},
    {"metric": "fingerprint", "value": f"Morgan radius={radius}, {bits} bits"},
    {"metric": "tanimoto_dedupe_cutoff", "value": cutoff},
    {"metric": "near_duplicates_removed_in_cleaning", "value": int(funnel3["near_duplicate_tanimoto_removed"])},
    {"metric": "pairwise_comparisons", "value": n_compounds * (n_compounds - 1) // 2},
    {"metric": "mean_pairwise_tanimoto", "value": similarity_sum / similarity_pairs},
    {"metric": "max_pairwise_tanimoto", "value": max_pair_similarity},
    {"metric": "pairs_at_or_above_0.75", "value": threshold_counts[0.75] // 2},
    {"metric": "pairs_at_or_above_0.75_involving_protected", "value": protected_pairs_above_cutoff},
    {"metric": "pairs_at_or_above_0.75_non_protected", "value": nonprotected_pairs_above_cutoff},
    {"metric": "pairs_at_or_above_0.90", "value": threshold_counts[0.90] // 2},
    {"metric": "pairs_at_or_above_0.50", "value": threshold_counts[0.50] // 2},
    {"metric": "nearest_neighbor_tanimoto_min", "value": float(nn_values.min())},
    {"metric": "nearest_neighbor_tanimoto_p10", "value": float(np.quantile(nn_values, 0.10))},
    {"metric": "nearest_neighbor_tanimoto_median", "value": float(np.median(nn_values))},
    {"metric": "nearest_neighbor_tanimoto_p90", "value": float(np.quantile(nn_values, 0.90))},
    {"metric": "nearest_neighbor_tanimoto_max", "value": float(nn_values.max())},
    {"metric": "training_reference_compounds", "value": len(ref_fps)},
    {"metric": "food_applicability_threshold", "value": float(domain["food"]["threshold"])},
    {"metric": "unseen_organic_compounds", "value": len(unseen_organic)},
    {"metric": "unseen_organic_inside_domain", "value": unseen_inside},
    {"metric": "unseen_organic_outside_domain", "value": unseen_below},
]
tanimoto_summary = pd.DataFrame(summary_rows)
path = STEP / "food_tanimoto_summary.csv"
write_csv(tanimoto_summary, path)
register_step(path, "Summary of Morgan/Tanimoto similarity on the cleaned food dataset.",
              ["data/interim/foods_clean.parquet", "models/domain_S.json",
               "reports/brick3_funnel.json"])

if nonprotected_pairs_above_cutoff != 0:
    raise RuntimeError("a non-protected food pair is at or above the 0.75 cleaning cutoff")
if protected_pairs_above_cutoff + nonprotected_pairs_above_cutoff != threshold_counts[0.75] // 2:
    raise RuntimeError("protected/non-protected pair counts do not match the total above cutoff")

# ---------------------------------------------------------------------------
# Raw and pipeline count tables
# ---------------------------------------------------------------------------
raw_rows = [
    {"dataset": "Druginfo.csv", "metric": "rows", "value": audit1["drugs"]["shape"][0],
     "notes": "raw drug table"},
    {"dataset": "Druginfo.csv", "metric": "columns", "value": audit1["drugs"]["shape"][1],
     "notes": ""},
    {"dataset": "Druginfo.csv", "metric": "drugs_with_smiles", "value": audit1["drugs"]["has_smiles"],
     "notes": "Canonical_SMILES present"},
    {"dataset": "InteractionInformation.csv", "metric": "rows", "value": audit1["interactions"]["shape"][0],
     "notes": "raw drug-food interaction table"},
    {"dataset": "InteractionInformation.csv", "metric": "columns", "value": audit1["interactions"]["shape"][1],
     "notes": ""},
    {"dataset": "InteractionInformation.csv", "metric": "unique_drugs", "value": audit1["interactions"]["unique_drugs"],
     "notes": ""},
    {"dataset": "InteractionInformation.csv", "metric": "unique_components", "value": audit1["interactions"]["unique_components"],
     "notes": ""},
    {"dataset": "InteractionInformation.csv", "metric": "unique_food_herb_ids", "value": audit1["interactions"]["unique_food_herb_ids"],
     "notes": ""},
    {"dataset": "food_fhdi_clean.csv", "metric": "rows", "value": audit1["foods"]["shape"][0],
     "notes": "raw food-compound links"},
    {"dataset": "food_fhdi_clean.csv", "metric": "columns", "value": audit1["foods"]["shape"][1],
     "notes": ""},
    {"dataset": "food_fhdi_clean.csv", "metric": "unique_food_ids", "value": audit1["foods"]["unique_food_ids"],
     "notes": ""},
    {"dataset": "food_fhdi_clean.csv", "metric": "unique_compounds", "value": audit1["foods"]["unique_compounds"],
     "notes": ""},
]
for effect, count in audit1["interactions"]["effect_counts"].items():
    raw_rows.append({
        "dataset": "InteractionInformation.csv",
        "metric": f"effect_rows__{effect.strip().lower().replace(' ', '_')}",
        "value": int(count),
        "notes": "raw Effect value",
    })
raw_counts = pd.DataFrame(raw_rows)
path = STEP / "raw_dataset_counts.csv"
write_csv(raw_counts, path)
register_step(path, "Initial row and column counts of the three raw datasets plus raw effect counts.",
              ["reports/brick1_audit.json"])

# Featurized split counts (pairs whose drug and compound both have MOE vectors)
drug_ids = set(drug_moe["molecule_id"])
food_ids = set(food_moe["molecule_id"])
featurized = splits[splits["drug_id"].isin(drug_ids) & splits["compound_id"].isin(food_ids)]
split_rows = []
for code, column in (("R", "split_R"), ("G", "split_G")):
    for part in ("train", "val", "test"):
        sub = featurized[featurized[column] == part]
        split_rows.append({
            "split": code, "partition": part, "assigned_rows": int((splits[column] == part).sum()),
            "featurized_rows": len(sub),
            "class_0": int((sub["label"] == 0).sum()),
            "class_1": int((sub["label"] == 1).sum()),
            "class_2": int((sub["label"] == 2).sum()),
            "weak_pairs": int(sub["weak"].sum()),
        })
for code in ("F", "D"):
    for part in ("train", "val", "test"):
        counts = gen["partitions"][code]["counts"][part]
        metric = gen["partitions"][code]["metrics"].get("validation" if part == "val" else part)
        split_rows.append({
            "split": code, "partition": part, "assigned_rows": int(counts["rows"]),
            "featurized_rows": int(metric["support"]) if metric else None,
            "class_0": int(counts["class_counts"]["0"]),
            "class_1": int(counts["class_counts"]["1"]),
            "class_2": int(counts["class_counts"]["2"]),
            "weak_pairs": None,
        })
split_counts = pd.DataFrame(split_rows)
for column in ("featurized_rows", "weak_pairs"):
    split_counts[column] = split_counts[column].astype("Int64")
path = STEP / "split_counts.csv"
write_csv(split_counts, path)
register_step(path, "Row and class counts for every split and partition.",
              ["data/processed/splits_S.parquet", "reports/generalization_S.json"])

pipeline_rows = [
    {"step": 1, "stage": "raw drug audit", "script": "src/brick1_audit.py",
     "input_rows": audit1["drugs"]["shape"][0], "output_rows": audit1["drugs"]["shape"][0],
     "note": "read-only audit of the three raw CSVs"},
    {"step": 2, "stage": "drug cleaning", "script": "src/brick2_clean_drugs.py",
     "input_rows": funnel2["loaded"], "output_rows": funnel2["final_drugs"],
     "note": f"{funnel2['parse_failures_or_empty']} parse failures/empty + "
             f"{funnel2['duplicate_inchikey14_removed']} duplicate InChIKey14 removed"},
    {"step": 3, "stage": "food cleaning", "script": "src/brick3_clean_foods.py",
     "input_rows": funnel3["unique_compounds"], "output_rows": funnel3["kept_after_tanimoto"],
     "note": f"{funnel3['duplicate_inchikey_removed']} InChIKey duplicates + "
             f"{funnel3['near_duplicate_tanimoto_removed']} Tanimoto near-duplicates at 0.75"},
    {"step": 4, "stage": "Mode S labeling", "script": "src/brick4_labels.py",
     "input_rows": len(splits), "output_rows": len(splits),
     "note": f"{label_report['mode_s_weak_pairs']} weak pairs; "
             f"{len(label_report['holdout_pairs'])} case-study pairs held out"},
    {"step": 5, "stage": "MOE descriptor generation", "script": "src/brick5_features.py",
     "input_rows": len(drugs) + len(foods), "output_rows": len(drug_moe) + len(food_moe),
     "note": "60 descriptors per molecule; non-organic foods are excluded from training"},
    {"step": 6, "stage": "pair feature extraction", "script": "src/features.py (used by bricks 7-17)",
     "input_rows": len(splits), "output_rows": len(featurized),
     "note": "3,780 pair features; 95 pairs dropped because the food is inorganic"},
    {"step": 7, "stage": "feature selection", "script": "src/brick7_selection.py --mode S",
     "input_rows": selection_log["counts"]["start"], "output_rows": selection_log["counts"]["rfe18"],
     "note": "variance -> Pearson -> Lasso -> RFECV -> RFE18"},
    {"step": 8, "stage": "split assignment", "script": "src/brick6_splits.py",
     "input_rows": len(splits), "output_rows": len(splits),
     "note": "Split R and Split G assigned for all 21,042 pairs"},
    {"step": 9, "stage": "class-2 cap (training only)", "script": "src/brick17_single_model.py",
     "input_rows": int((featurized["split_G"] == "train").sum()),
     "output_rows": int(single["ratio_sweep"][2]["train_rows"]),
     "note": "class 2 capped at 2 x class 0; validation and test keep the real distribution"},
    {"step": 10, "stage": "final model training", "script": "src/brick17_single_model.py --apply",
     "input_rows": int(single["ratio_sweep"][2]["train_rows"]),
     "output_rows": int(single["ratio_sweep"][2]["train_rows"]),
     "note": "one LightGBM model with 18 features; writes best_model_S.joblib"},
    {"step": 11, "stage": "evidence lookup", "script": "src/brick13_build_evidence.py",
     "input_rows": len(evidence), "output_rows": len(evidence),
     "note": "dataset evidence for exact pairs; not used as an ML feature"},
    {"step": 12, "stage": "test predictions", "script": "scripts/build_deliverables.py (deployed artifact)",
     "input_rows": int(gen["partitions"]["G"]["metrics"]["test"]["support"]),
     "output_rows": len(pred_g),
     "note": "deployed best_model_S.joblib scored once on the Split G test pairs"},
]
pipeline_counts = pd.DataFrame(pipeline_rows)
path = STEP / "pipeline_counts.csv"
write_csv(pipeline_counts, path)
register_step(path, "Stage-by-stage input/output row counts for the deployed pipeline.",
              ["reports/brick2_funnel.json", "reports/brick3_funnel.json",
               "reports/brick4_label_report.json", "reports/brick7_selection_log_S.json",
               "reports/single_model_evaluation.json", "reports/generalization_S.json"])

# ---------------------------------------------------------------------------
# Metrics tables
# ---------------------------------------------------------------------------
print("building metric tables ...")
metric_rows = []
per_class_rows = []
cm_rows = []
for code in SPLIT_ORDER:
    partition_block = gen["partitions"][code]
    for part in ("validation", "test"):
        metrics = partition_block["metrics"][part]
        row = {
            "split": code,
            "split_description": SPLIT_DESC[code],
            "partition": part,
            "n": int(metrics["support"]),
            "class_0_support": int(metrics["class_counts"]["0"]),
            "class_1_support": int(metrics["class_counts"]["1"]),
            "class_2_support": int(metrics["class_counts"]["2"]),
        }
        for key in METRIC_COLS:
            row[key] = metrics.get(key)
        metric_rows.append(row)
        for cls in (0, 1, 2):
            pc = metrics["per_class"][str(cls)]
            per_class_rows.append({
                "split": code, "partition": part, "class": cls, "label": SHORT[cls],
                "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"],
                "support": int(pc["support"]),
            })
        cm = metrics["confusion_matrix"]
        cm_rows.append({
            "split": code, "partition": part, "n": int(metrics["support"]),
            "true_avoid_pred_avoid": int(cm[0][0]), "true_avoid_pred_beneficial": int(cm[0][1]),
            "true_avoid_pred_nonsignificant": int(cm[0][2]),
            "true_beneficial_pred_avoid": int(cm[1][0]), "true_beneficial_pred_beneficial": int(cm[1][1]),
            "true_beneficial_pred_nonsignificant": int(cm[1][2]),
            "true_nonsignificant_pred_avoid": int(cm[2][0]), "true_nonsignificant_pred_beneficial": int(cm[2][1]),
            "true_nonsignificant_pred_nonsignificant": int(cm[2][2]),
        })

metrics_frame = pd.DataFrame(metric_rows)
path = METRICS / "metrics_by_split.csv"
write_csv(metrics_frame, path)
register_metrics(path, "Every aggregate metric for R/G/F/D on validation and test.",
                 ["reports/generalization_S.json"])

per_class_frame = pd.DataFrame(per_class_rows)
path = METRICS / "per_class_metrics.csv"
write_csv(per_class_frame, path)
register_metrics(path, "Per-class precision, recall, F1, and support.",
                 ["reports/generalization_S.json"])

cm_frame = pd.DataFrame(cm_rows)
path = METRICS / "confusion_matrices.csv"
write_csv(cm_frame, path)
register_metrics(path, "Exact 3x3 confusion matrices (rows=true, columns=predicted).",
                 ["reports/generalization_S.json"])

label_rows = []
for code in SPLIT_ORDER:
    for part in ("train", "val", "test"):
        counts = gen["partitions"][code]["counts"][part]
        label_rows.append({
            "split": code, "partition": part, "assigned_rows": int(counts["rows"]),
            "class_0": int(counts["class_counts"]["0"]),
            "class_1": int(counts["class_counts"]["1"]),
            "class_2": int(counts["class_counts"]["2"]),
            "class_1_note": "unstable" if int(counts["class_counts"]["1"]) < 30 else "",
        })
label_frame = pd.DataFrame(label_rows)
path = METRICS / "label_distribution.csv"
write_csv(label_frame, path)
register_metrics(path, "Label distribution for every split and partition.",
                 ["reports/generalization_S.json"])

# Calibration and selective tables
calibration_summary = pd.DataFrame(cal_rows)
path = METRICS / "calibration_summary.csv"
write_csv(calibration_summary, path)
register_metrics(path, "Raw and calibrated NLL, Brier, ECE, and mean confidence.",
                 ["reports/calibration_S.json"])

bin_rows = []
for split in ("validation", "test"):
    for stage in ("before", "after"):
        block = calibration[f"{split}_{stage}"]
        for item in block["reliability_bins"]:
            bin_rows.append({
                "split": split, "stage": "raw" if stage == "before" else "calibrated",
                "bin_low": item["bin_low"], "bin_high": item["bin_high"],
                "count": item["count"], "mean_confidence": item["mean_confidence"],
                "accuracy": item["accuracy"],
            })
bins_frame = pd.DataFrame(bin_rows)
path = METRICS / "reliability_bins.csv"
write_csv(bins_frame, path)
register_metrics(path, "Reliability-diagram points before and after temperature scaling.",
                 ["reports/calibration_S.json"])

threshold_frame = pd.DataFrame(calibration["threshold_curve"])
path = METRICS / "threshold_curve.csv"
write_csv(threshold_frame, path)
register_metrics(path, "Validation coverage and accepted error for every acceptance threshold.",
                 ["reports/calibration_S.json"])

selective_frame = pd.DataFrame([{
    "split": "G",
    "partition": "test",
    "temperature": calibration["temperature"],
    "accept_threshold": calibration["accept_threshold"],
    "coverage": calibration["test_selective"]["coverage"],
    "accepted": calibration["test_selective"]["accepted"],
    "accuracy_on_accepted": calibration["test_selective"]["accuracy_on_accepted"],
    "validation_coverage": calibration["threshold_validation_coverage"],
}])
path = METRICS / "selective_prediction.csv"
write_csv(selective_frame, path)
register_metrics(path, f"Selective-prediction coverage and accuracy at the {calibration['accept_threshold']} acceptance threshold.",
                 ["reports/calibration_S.json", "reports/single_model_evaluation.json"])

# ROC/PR per class from the deployed model's Split G test probabilities.
# The saved generalization report computed OvR ROC/PR on raw estimator
# probabilities; temperature scaling changes the numbers at the 4th decimal.
roc_rows = []
g_probs_raw = pred_g[["p0_raw", "p1_raw", "p2_raw"]].to_numpy(dtype=float)
g_labels = pred_g["label"].to_numpy()
for cls in (0, 1, 2):
    binary = (g_labels == cls).astype(int)
    support = int(binary.sum())
    if support == 0 or support == len(binary):
        roc_auc = None
        ap = None
    else:
        fpr, tpr, _ = roc_curve(binary, g_probs_raw[:, cls])
        roc_auc = float(auc(fpr, tpr))
        ap = float(average_precision_score(binary, g_probs_raw[:, cls]))
    roc_rows.append({
        "split": "G", "partition": "test", "class": cls, "label": SHORT[cls],
        "support": support, "roc_auc_ovr": roc_auc, "pr_auc": ap,
    })
roc_frame = pd.DataFrame(roc_rows)
roc_mean = float(roc_frame["roc_auc_ovr"].mean())
pr_mean = float(roc_frame["pr_auc"].mean())
if abs(roc_mean - g_test_metrics["roc_auc_ovr_macro"]) > 1e-12 or \
        abs(pr_mean - g_test_metrics["pr_auc_macro"]) > 1e-12:
    raise RuntimeError(
        "per-class ROC/PR means do not reproduce the saved Split G macro scores: "
        f"{roc_mean} vs {g_test_metrics['roc_auc_ovr_macro']}, "
        f"{pr_mean} vs {g_test_metrics['pr_auc_macro']}"
    )
path = METRICS / "roc_pr_auc.csv"
write_csv(path=path, frame=roc_frame)
register_metrics(path, "One-vs-rest ROC AUC and PR AUC per class on deployed-model raw test probabilities.",
                 ["models/best_model_S.joblib", "data/processed/splits_S.parquet",
                  "reports/generalization_S.json"])

# Text classification reports
report_lines = [
    "Exact classification reports for the single LightGBM recipe 2026-11-16.",
    "Split G scores are the deployed artifact. Split R, F, and D scores come from",
    "refitting the same fixed recipe on each partition, as defined by the evaluation stage.",
    "Rows are true labels; columns are predicted labels in confusion_matrices.csv.",
    "Class 1 has very small support; its scores are unstable and must not be read as a reliable estimate.",
    "",
]
for code in SPLIT_ORDER:
    for part in ("validation", "test"):
        block = gen["partitions"][code]["metrics"][part]
        n = block["support"]
        report_lines.append(f"Split {code} - {SPLIT_DESC[code]} - {part} (n={n})")
        report_lines.append("class              precision   recall       f1  support")
        for cls in (0, 1, 2):
            pc = block["per_class"][str(cls)]
            report_lines.append(
                f"{SHORT[cls]:<15} {pc['precision']:10.6f} {pc['recall']:8.6f} "
                f"{pc['f1']:8.6f} {int(pc['support']):8d}"
            )
        weighted_precision = sum(
            block["per_class"][str(c)]["precision"] * block["per_class"][str(c)]["support"]
            for c in (0, 1, 2)
        ) / n
        weighted_recall = sum(
            block["per_class"][str(c)]["recall"] * block["per_class"][str(c)]["support"]
            for c in (0, 1, 2)
        ) / n
        report_lines.append(
            f"{'macro avg':<15} {block['macro_precision']:10.6f} {block['macro_recall']:8.6f} "
            f"{block['macro_f1']:8.6f} {n:8d}"
        )
        report_lines.append(
            f"{'weighted avg':<15} {weighted_precision:10.6f} {weighted_recall:8.6f} "
            f"{block['weighted_f1']:8.6f} {n:8d}"
        )
        report_lines.append(
            f"accuracy {block['accuracy']:.6f}; balanced accuracy {block['balanced_accuracy']:.6f}; "
            f"MCC {block['mcc']:.6f}; kappa {block['cohen_kappa']:.6f}; "
            f"ROC-AUC {block['roc_auc_ovr_macro'] if block['roc_auc_ovr_macro'] is not None else 'NA'}; "
            f"PR-AUC {block['pr_auc_macro']:.6f}"
        )
        report_lines.append("")
path = METRICS / "classification_reports.txt"
path.write_text("\n".join(report_lines), encoding="utf-8")
register_metrics(path, "Human-readable exact classification reports for every split and partition.",
                 ["reports/generalization_S.json"])

# Copy comparison/sweep/baseline tables to final_metrics too
for name, description in [
    ("model_comparison_S.csv", "Five model families on Split G and Split R test partitions."),
    ("class_balance_sweep.csv", "Validation scores for uncapped, 1:1, 2:1, and 3:1 class-2 caps."),
    ("baselines_S.csv", "Majority, logistic, and LightGBM imbalance baselines."),
]:
    shutil.copyfile(REPORTS / name, METRICS / name)
    register_metrics(METRICS / name, description, [f"reports/{name}"])

# Model configuration and checksums
print("writing model configuration and checksums ...")
model_config = {
    "model_name": metadata["model_name"],
    "model_version": metadata["model_version"],
    "artifact": metadata["artifact"],
    "artifact_sha256_prefix": metadata["artifact_sha256_prefix"],
    "estimator_wrapper": "brick8_models.BalancedClassifier",
    "estimator_class": f"{type(base).__module__}.{type(base).__name__}",
    "estimator_params": estimator_params,
    "sample_weight_recipe": (
        "sklearn compute_sample_weight('balanced', y) multiplied by 0.5 for weak class-2 rows"
    ),
    "class_balance": metadata["class_balance"],
    "training_split": metadata["training_split"],
    "selection_partition": metadata["selection_partition"],
    "feature_count": int(metadata["feature_count"]),
    "features": metadata["features"],
    "feature_selection_funnel": selection_log["counts"],
    "label_space": {str(k): v for k, v in LABELS.items()},
    "calibration": {
        "method": calibration["method"],
        "temperature": calibration["temperature"],
        "fitted_on": calibration["fitted_on"],
        "accept_threshold": calibration["accept_threshold"],
        "threshold_policy": calibration["threshold_policy"],
        "threshold_target_error": calibration["threshold_target_error"],
    },
    "applicability_domain": {
        "fingerprint": domain["fingerprint"],
        "validated": domain["validated"],
        "drug": domain["drug"],
        "food": domain["food"],
        "reference_counts": {
            "drugs": len(domain["reference"]["drug_ids"]),
            "compounds": len(domain["reference"]["compound_ids"]),
        },
    },
    "serving_abstention": single["test_selective"],
}
path = METRICS / "model_config.json"
write_json(model_config, path)
register_metrics(path, "Complete deployed-model configuration and provenance.", [
    "models/model_metadata_S.json", "models/best_model_S.joblib",
    "models/calibration_S.json", "models/domain_S.json", "models/selected_features_S.json",
])

checksum_targets = [
    "models/best_model_S.joblib",
    "models/model_metadata_S.json",
    "models/calibration_S.json",
    "models/domain_S.json",
    "models/selected_features_S.json",
    "models/train_drugs_S.json",
    "data/processed/splits_S.parquet",
    "data/processed/evidence_S.parquet",
    "data/interim/drugs_clean.parquet",
    "data/interim/foods_clean.parquet",
    "data/interim/drug_moe.parquet",
    "data/interim/food_moe.parquet",
    "reports/generalization_S.json",
    "reports/single_model_evaluation.json",
    "reports/calibration_S.json",
]
checksum_rows = []
for rel in checksum_targets:
    target = ROOT / rel
    checksum_rows.append({
        "file": rel,
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
    })
checksum_frame = pd.DataFrame(checksum_rows)
path = METRICS / "artifact_checksums.csv"
write_csv(checksum_frame, path)
register_metrics(path, "SHA-256 checksums of the model, data, and report artifacts used here.",
                 [rel for rel in checksum_targets])

actual_model_sha = sha256(MODELS / "best_model_S.joblib")
if not actual_model_sha.startswith(metadata["artifact_sha256_prefix"]):
    raise RuntimeError("deployed model SHA does not match model_metadata_S.json")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("drawing figures ...")
COLORS = {0: "#b23b3b", 1: "#2f7d4f", 2: "#4a6fa5"}


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, code: str, part: str, path: Path) -> None:
    cm = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    image = ax.imshow(cm, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    labels = [SHORT[c] for c in (0, 1, 2)]
    ax.set_xticks(range(3), labels, rotation=20, ha="right")
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix - Split {code} {part}\n{SPLIT_DESC[code]}")
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(3):
        for j in range(3):
            share = 0.0 if row_totals[i, 0] == 0 else cm[i, j] / row_totals[i, 0]
            ax.text(j, i, f"{cm[i, j]}\n({share * 100:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black", fontsize=9)
    save_figure(fig, path)
    register_metrics(path, f"Confusion matrix image for Split {code} {part}.",
                     ["reports/generalization_S.json"])


for code in SPLIT_ORDER:
    for part in ("validation", "test"):
        block = gen["partitions"][code]["metrics"][part]
        plot_confusion(block["confusion_matrix"], code, part,
                       METRICS / f"confusion_matrix_{code}_{part}.png")

# 2x2 overview of the four test confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
for ax, code in zip(axes.ravel(), SPLIT_ORDER):
    cm = np.asarray(gen["partitions"][code]["metrics"]["test"]["confusion_matrix"], dtype=int)
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(3), [SHORT[c] for c in (0, 1, 2)], rotation=20, ha="right")
    ax.set_yticks(range(3), [SHORT[c] for c in (0, 1, 2)])
    ax.set_title(f"Split {code} test (n={int(cm.sum())})")
    threshold = cm.max() * 0.55
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
fig.suptitle("Single LightGBM 2026-11-16 - test confusion matrices (Split G = deployed artifact)", fontsize=12)
fig.tight_layout()
path = METRICS / "confusion_matrices_S_test.png"
save_figure(fig, path)
register_metrics(path, "2x2 overview of the four test confusion matrices.",
                 ["reports/generalization_S.json"])


def metrics_for(code: str, part: str) -> dict:
    return gen["partitions"][code]["metrics"][part]


def plot_metrics_overview(part: str, path: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("mcc", "MCC"),
    ]
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, code in enumerate(SPLIT_ORDER):
        values = [metrics_for(code, part)[key] for key, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=f"Split {code}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.018 if value >= 0 else -0.05),
                    f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(min(-0.15, min(metrics_for(c, part)["mcc"] for c in SPLIT_ORDER) - 0.08), 1.05)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Score")
    ax.set_title(f"Single LightGBM recipe - {part} metrics (Split G = deployed artifact; R/F/D refits)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    save_figure(fig, path)
    register_metrics(path, f"Metric overview image for all splits ({part}).",
                     ["reports/generalization_S.json"])


plot_metrics_overview("validation", METRICS / "metrics_overview_validation.png")
plot_metrics_overview("test", METRICS / "metrics_overview_test.png")


def plot_metrics_table(path: Path) -> None:
    columns = ["Split", "Partition", "n", "Accuracy", "Bal. acc.", "Macro F1",
               "Weighted F1", "MCC", "Kappa", "ROC AUC", "PR AUC"]
    rows = []
    for code in SPLIT_ORDER:
        for part in ("validation", "test"):
            m = metrics_for(code, part)
            rows.append([
                code, part, str(m["support"]), f"{m['accuracy']:.4f}",
                f"{m['balanced_accuracy']:.4f}", f"{m['macro_f1']:.4f}",
                f"{m['weighted_f1']:.4f}", f"{m['mcc']:.4f}", f"{m['cohen_kappa']:.4f}",
                "NA" if m["roc_auc_ovr_macro"] is None else f"{m['roc_auc_ovr_macro']:.4f}",
                f"{m['pr_auc_macro']:.4f}",
            ])
    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#dce6f1")
        table[(0, j)].set_text_props(weight="bold")
    for i in range(1, len(rows) + 1):
        if rows[i - 1][1] == "test":
            for j in range(len(columns)):
                table[(i, j)].set_facecolor("#f5f7fa")
    ax.set_title("Single LightGBM 2026-11-16 - exact metric scores (Split G deployed; R/F/D refits)", pad=14)
    save_figure(fig, path)
    register_metrics(path, "Rendered table of every aggregate metric score.",
                     ["reports/generalization_S.json"])


plot_metrics_table(METRICS / "metrics_table_S.png")


def plot_per_class_f1(path: Path) -> None:
    x = np.arange(3)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, code in enumerate(SPLIT_ORDER):
        values = [metrics_for(code, "test")["per_class"][str(c)]["f1"] for c in (0, 1, 2)]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=f"Split {code}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [SHORT[c] for c in (0, 1, 2)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 on the test partition (class 1 support is 0-3)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_figure(fig, path)
    register_metrics(path, "Per-class F1 image for the four test partitions.",
                     ["reports/generalization_S.json"])


plot_per_class_f1(METRICS / "per_class_f1_S.png")


def plot_roc_pr(code: str, frame: pd.DataFrame, path: Path) -> None:
    y = frame["label"].to_numpy()
    probs = frame[["p0_raw", "p1_raw", "p2_raw"]].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for cls in (0, 1, 2):
        binary = (y == cls).astype(int)
        support = int(binary.sum())
        if support == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, cls])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(binary, probs[:, cls])
        ap = average_precision_score(binary, probs[:, cls])
        style = "--" if cls == 1 else "-"
        axes[0].plot(fpr, tpr, style, color=COLORS[cls],
                     label=f"{SHORT[cls]} (n={support}, AUC={roc_auc:.3f})")
        axes[1].plot(recall, precision, style, color=COLORS[cls],
                     label=f"{SHORT[cls]} (n={support}, AP={ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", linewidth=0.7)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"Split {code} test - one-vs-rest ROC")
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Split {code} test - one-vs-rest precision-recall")
    axes[1].legend(loc="upper right")
    fig.suptitle(f"{SPLIT_DESC[code]}; raw estimator probabilities; class 1 is sparse and unstable",
                 fontsize=9)
    fig.tight_layout()
    save_figure(fig, path)
    register_metrics(path, f"ROC and precision-recall curves for Split {code} test.",
                     ["models/best_model_S.joblib", "data/processed/splits_S.parquet"])


plot_roc_pr("G", pred_g, METRICS / "roc_pr_curves_G.png")


def plot_calibration(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, split in zip(axes, ("validation", "test")):
        for stage, color, marker in (("before", "#b23b3b", "o"), ("after", "#2f6fb2", "s")):
            bins = calibration[f"{split}_{stage}"]["reliability_bins"]
            xs = [b["mean_confidence"] for b in bins]
            ys = [b["accuracy"] for b in bins]
            ax.plot(xs, ys, marker=marker, markersize=4, linewidth=1.5, color=color,
                    label=f"{'raw' if stage == 'before' else 'calibrated'} "
                          f"(ECE={calibration[f'{split}_{stage}']['ece']:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", linewidth=0.7, linestyle=":")
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{split.capitalize()} reliability")
        ax.legend(loc="upper left")
    fig.suptitle(f"Temperature scaling T={calibration['temperature']:.4f} fitted on Split G validation")
    fig.tight_layout()
    save_figure(fig, path)
    register_metrics(path, "Reliability diagrams before and after temperature scaling.",
                     ["reports/calibration_S.json"])


plot_calibration(METRICS / "calibration_reliability_S.png")


def plot_threshold_curve(path: Path) -> None:
    curve = pd.DataFrame(calibration["threshold_curve"])
    selected = curve.iloc[(curve["threshold"] - calibration["accept_threshold"]).abs().argsort()[:1]]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(curve["threshold"], curve["error"], color="#b23b3b", label="Accepted error")
    ax.set_xlabel("Acceptance threshold")
    ax.set_ylabel("Accepted error", color="#b23b3b")
    ax.tick_params(axis="y", labelcolor="#b23b3b")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], curve["coverage"], color="#2f6fb2", label="Coverage")
    ax2.set_ylabel("Coverage", color="#2f6fb2")
    ax2.tick_params(axis="y", labelcolor="#2f6fb2")
    ax.axvline(calibration["accept_threshold"], color="black", linestyle="--", linewidth=0.9)
    ax.text(calibration["accept_threshold"] + 0.01,
            ax.get_ylim()[1] * 0.92,
            f"selected {calibration['accept_threshold']:.2f}\n"
            f"coverage {float(selected['coverage'].iloc[0]):.3f}\n"
            f"error {float(selected['error'].iloc[0]):.3f}",
            fontsize=8)
    ax.set_title("Validation coverage/error trade-off for abstention")
    save_figure(fig, path)
    register_metrics(path, "Validation threshold curve for the abstention decision.",
                     ["reports/calibration_S.json"])


plot_threshold_curve(METRICS / "threshold_coverage_S.png")


def plot_model_comparison(path: Path) -> None:
    models = ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    x = np.arange(len(models))
    width = 0.36
    for ax, metric, title in ((axes[0], "macro_f1", "Macro F1"),
                              (axes[1], "accuracy", "Accuracy")):
        for i, split in enumerate(("G", "R")):
            sub = comparison[comparison["split"] == f"Split {split}"].set_index("model")
            values = [float(sub.loc[m, metric]) for m in models]
            bars = ax.bar(x + (i - 0.5) * width, values, width, label=f"Split {split}")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, models, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by model family and split")
        ax.legend()
    fig.suptitle("Model comparison (same 18 features; each split has its own training partition)")
    fig.tight_layout()
    save_figure(fig, path)
    register_metrics(path, "Model-family comparison image for Split G and Split R.",
                     ["reports/model_comparison_S.csv"])


plot_model_comparison(METRICS / "model_comparison_S.png")


def plot_class_balance(path: Path) -> None:
    recipes = ["uncapped", "cap_1", "cap_2", "cap_3"]
    labels = [f"{r}\n({int(sweep.set_index('recipe').loc[r, 'train_rows']):,} rows)" for r in recipes]
    sub = sweep.set_index("recipe")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    x = np.arange(len(recipes))
    width = 0.36
    for ax, col, title in ((axes[0], "val_macro_f1", "Validation macro F1"),
                           (axes[1], "val_recall_0", "Validation recall - Avoid class")):
        values = [float(sub.loc[r, col]) for r in recipes]
        bars = ax.bar(x, values, width, color=["#8a8a8a", "#c78a3b", "#2f6fb2", "#5a9e6f"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.4f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_ylim(0, 0.9)
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].text(2, float(sub.loc["cap_2", "val_macro_f1"]) + 0.07, "deployed",
                 ha="center", color="#2f6fb2", fontsize=8, fontweight="bold")
    fig.suptitle("Class-2 cap sweep on Split G validation (training rows only are capped)")
    fig.tight_layout()
    save_figure(fig, path)
    register_metrics(path, "Class-ratio sweep image with the deployed 2:1 recipe marked.",
                     ["reports/class_balance_sweep.csv"])


plot_class_balance(METRICS / "class_balance_sweep_S.png")


def plot_food_tanimoto(path: Path) -> None:
    nn = neighbors["max_tanimoto"].to_numpy(dtype=float)
    unseen = applicability[(~applicability["in_training_reference"]) &
                           (applicability["is_organic"])]
    ref = unseen["max_tanimoto_to_train_reference"].to_numpy(dtype=float)
    threshold = float(domain["food"]["threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    axes[0].hist(nn, bins=40, color="#4a6fa5", edgecolor="white")
    axes[0].axvline(cutoff, color="#b23b3b", linestyle="--", linewidth=1.2,
                    label=f"cleaning cutoff {cutoff}")
    axes[0].set_xlabel("Nearest-neighbour Tanimoto similarity")
    axes[0].set_ylabel("Food compounds")
    axes[0].set_title(f"All {n_compounds:,} cleaned compounds; max pair = {max_pair_similarity:.4f}")
    axes[0].legend(loc="upper left")
    axes[1].hist(ref, bins=40, color="#5a9e6f", edgecolor="white")
    axes[1].axvline(threshold, color="#b23b3b", linestyle="--", linewidth=1.2,
                    label=f"AD threshold {threshold:.4f}")
    axes[1].set_xlabel("Max Tanimoto to training reference")
    axes[1].set_ylabel("Unseen organic compounds")
    axes[1].set_title(f"{len(unseen):,} unseen compounds: {unseen_outside:,} outside, "
                      f"{len(unseen) - unseen_outside:,} inside")
    axes[1].legend(loc="upper left")
    fig.suptitle("Morgan radius 2 / 2048-bit Tanimoto similarity on the food dataset")
    fig.tight_layout()
    save_figure(fig, path)
    register_metrics(path, "Food Tanimoto distributions for cleaning and applicability domain.",
                     ["step_csvs/food_tanimoto_neighbors.csv",
                      "step_csvs/food_tanimoto_applicability.csv"])


plot_food_tanimoto(METRICS / "food_tanimoto_distribution_S.png")

# ---------------------------------------------------------------------------
# context.md
# ---------------------------------------------------------------------------
print("writing context.md and documentation.md ...")
raw_interactions = audit1["interactions"]
split_assigned = {
    code: {part: int((splits[column] == part).sum()) for part in ("train", "val", "test")}
    for code, column in (("R", "split_R"), ("G", "split_G"))
}
feat_split = {}
for code, column in (("R", "split_R"), ("G", "split_G")):
    feat_split[code] = {
        part: int((featurized[column] == part).sum()) for part in ("train", "val", "test")
    }
class_counts = {int(k): int(v) for k, v in dataset_audit["modes"]["S"]["class_counts"].items()}
nonweak_counts = {int(k): int(v) for k, v in dataset_audit["modes"]["S"]["nonweak_class_counts"].items()}
train_counts = {int(k): int(v) for k, v in metadata["class_balance"]["train_counts"].items()}
test_g = gen["partitions"]["G"]["metrics"]["test"]
test_r = gen["partitions"]["R"]["metrics"]["test"]
test_f = gen["partitions"]["F"]["metrics"]["test"]
test_d = gen["partitions"]["D"]["metrics"]["test"]

context = f"""# Pipeline context: rows at the start and after every step

All numbers in this file are read from the saved artifacts in `data/`, `models/`, and
`reports/`; none are estimated. The deployed model is **{metadata['model_name']}
{metadata['model_version']}** (historical artifact name `S`). There is no Mode P model.

## 1. Initial datasets (raw inputs)

| Dataset | File | Rows | Columns |
|---|---|---:|---:|
| Drug table | `Druginfo.csv` | {audit1['drugs']['shape'][0]:,} | {audit1['drugs']['shape'][1]} |
| Drug-food interactions | `InteractionInformation.csv` | {audit1['interactions']['shape'][0]:,} | {audit1['interactions']['shape'][1]} |
| Food-compound links | `food_fhdi_clean.csv` | {audit1['foods']['shape'][0]:,} | {audit1['foods']['shape'][1]} |

The raw drug table has **{audit1['drugs']['has_smiles']:,}** drugs with a Canonical SMILES.
The raw interactions table covers **{raw_interactions['unique_drugs']:,}** drugs,
**{raw_interactions['unique_food_herb_ids']:,}** food/herb identifiers, and
**{raw_interactions['unique_components']:,}** named components. The raw food-link table covers
**{audit1['foods']['unique_food_ids']:,}** foods and **{audit1['foods']['unique_compounds']:,}**
unique compounds.

Raw `Effect` values (row counts):

| Effect | Rows |
|---|---:|
| Possible | {raw_interactions['effect_counts']['Possible']:,} |
| No Effect | {raw_interactions['effect_counts']['No Effect']:,} |
| Positive | {raw_interactions['effect_counts']['Positive']:,} |
| Negative | {raw_interactions['effect_counts']['Negative']:,} |
| Harmful | {raw_interactions['effect_counts']['Harmful']:,} |
| possible (lower-case) | {raw_interactions['effect_counts']['possible']:,} |
| **Total** | **{sum(raw_interactions['effect_counts'].values()):,}** |

## 2. Drug cleaning (`src/brick2_clean_drugs.py`)

| Step | Drugs |
|---|---:|
| Loaded from `Druginfo.csv` | {funnel2['loaded']:,} |
| Parse failures or empty structure | -{funnel2['parse_failures_or_empty']:,} |
| Parsed with RDKit | {funnel2['parsed']:,} |
| Duplicate InChIKey14 removed | -{funnel2['duplicate_inchikey14_removed']:,} |
| **Final cleaned drugs** | **{funnel2['final_drugs']:,}** |

## 3. Food cleaning (`src/brick3_clean_foods.py`)

| Step | Compounds | Links |
|---|---:|---:|
| Raw food-compound rows | {funnel3['unique_compounds']:,} unique | {funnel3['raw_links']:,} |
| No SMILES | -{funnel3['no_smiles']:,} | |
| RDKit parse failures | -{funnel3['parse_failures']:,} | |
| Parsed unique compounds | {funnel3['parsed_unique_compounds']:,} | |
| Duplicate InChIKey removed | -{funnel3['duplicate_inchikey_removed']:,} | |
| Morgan/Tanimoto near-duplicates removed at 0.75 | -{funnel3['near_duplicate_tanimoto_removed']:,} | |
| **Final cleaned compounds** | **{funnel3['kept_after_tanimoto']:,}** | **{funnel3['final_links']:,}** |

The final compound table is **{funnel3['organic_compounds']:,} organic** and
**{funnel3['inorganic_compounds']:,} inorganic**. Training uses only the organic compounds, so
the model never learns from minerals. Serving still answers for them: the exact-pair dataset
annotation is checked first, a mineral query with no annotation falls back to the foods that
contain it, and a mineral with no fallback is returned as `not_supported` with a diagnostic
TreeSHAP block when its descriptors compute (see section 15).
Marker names configured in `config.yaml` (ethanol, tyramine, caffeine, bergamottin, naringin,
hyperforin, phylloquinone, vitamin K1, dimethyl disulfide) are protected from near-duplicate
removal; {funnel3['protected_present_final']} protected compounds are present in the final set.

## 4. Mode S labels (`src/brick4_labels.py`)

| Quantity | Value |
|---|---:|
| Labelled Mode S pairs | {dataset_audit['modes']['S']['pairs']:,} |
| Unique drugs | {dataset_audit['modes']['S']['unique_drugs']:,} |
| Unique compounds | {dataset_audit['modes']['S']['unique_compounds']:,} |
| Class 0 - Avoid | {class_counts[0]:,} |
| Class 1 - Beneficial | {class_counts[1]:,} |
| Class 2 - No significant interaction | {class_counts[2]:,} |
| Weak class-2 pairs | {dataset_audit['modes']['S']['weak_pairs']:,} |
| Non-weak class-2 pairs | {nonweak_counts[2]:,} |
| Case-study pairs held out of training | {len(label_report['holdout_pairs']):,} |

Class 1 is tiny (**{class_counts[1]}** pairs). This is a property of the source data, not a
modelling choice; every class-1 metric in this project must be read as unstable.

## 5. Descriptors (`src/brick5_features.py`)

| Quantity | Value |
|---|---:|
| Drug molecules with MOE vectors | {feature_report['drug_molecules']:,} |
| Food molecules with MOE vectors | {feature_report['food_molecules']:,} |
| Total descriptor rows | {len(drug_moe) + len(food_moe):,} |
| MOE descriptors per molecule | {len(drug_moe.columns) - 1} |
| Pair features generated | {selection_log['counts']['start']:,} |
| Mode S pairs featurized | {len(featurized):,} |
| Pairs dropped (inorganic food has no MOE vector) | {len(splits) - len(featurized):,} |

The {selection_log['counts']['start']:,} pair features are 3,600 cross-products, 60 same-descriptor
sums, 60 absolute differences, and 60 ratios.

## 6. Feature selection (`src/brick7_selection.py --mode S`)

| Step | Remaining features | Setting |
|---|---:|---|
| Start | {selection_log['counts']['start']:,} | 60 descriptors x pairwise arithmetic |
| Variance threshold | {selection_log['counts']['after_variance']:,} | threshold 0.8 |
| Pearson correlation filter | {selection_log['counts']['after_correlation']:,} | absolute Pearson r >= 0.75 removed |
| Lasso | {selection_log['counts']['after_lasso']:,} | alpha 0.01 |
| RFECV | {selection_log['counts']['after_rfecv']:,} | 5-fold CV |
| Final RFE | {selection_log['counts']['rfe18']:,} | target 18 features |

Selection used only Split G training rows; the fitted scaler used by Lasso was fit inside the
selection step. The 18 features were not reselected after the class-cap retrain.

## 7. Splits (`src/brick6_splits.py`)

Split assignments cover all **{len(splits):,}** pairs. The model is trained and evaluated only on
the **{len(featurized):,}** pairs that have both descriptor vectors.

| Split | Meaning | Train assigned | Val assigned | Test assigned | Train featurized | Val featurized | Test featurized |
|---|---|---:|---:|---:|---:|---:|---:|
| R | pair-random | {split_assigned['R']['train']:,} | {split_assigned['R']['val']:,} | {split_assigned['R']['test']:,} | {feat_split['R']['train']:,} | {feat_split['R']['val']:,} | {feat_split['R']['test']:,} |
| G | unseen drugs | {split_assigned['G']['train']:,} | {split_assigned['G']['val']:,} | {split_assigned['G']['test']:,} | {feat_split['G']['train']:,} | {feat_split['G']['val']:,} | {feat_split['G']['test']:,} |
| F | unseen compounds | {gen['partitions']['F']['counts']['train']['rows']:,} | {gen['partitions']['F']['counts']['val']['rows']:,} | {gen['partitions']['F']['counts']['test']['rows']:,} | - | {gen['partitions']['F']['metrics']['validation']['support']:,} | {gen['partitions']['F']['metrics']['test']['support']:,} |
| D | double-cold | {gen['partitions']['D']['counts']['train']['rows']:,} | {gen['partitions']['D']['counts']['val']['rows']:,} | {gen['partitions']['D']['counts']['test']['rows']:,} | - | {gen['partitions']['D']['metrics']['validation']['support']:,} | {gen['partitions']['D']['metrics']['test']['support']:,} |

The headline evaluation is **Split G test** (unseen drugs). Split R is the optimistic
pair-random reference. Split F tests unseen food compounds. Split D is the strictest
double-cold check and has only {test_d['support']:,} test pairs, including no class-1 pairs.

## 8. Training and deployed model (`src/brick17_single_model.py`)

| Quantity | Value |
|---|---:|
| Model | {metadata['model_name']} {metadata['model_version']} |
| Algorithm | LightGBM gradient-boosted trees (`boosting_type=gbdt`) |
| Feature count | {metadata['feature_count']} |
| Split G training rows before capping | {feat_split['G']['train']:,} |
| Split G training rows after class-2 cap 2:1 | **{train_counts[0] + train_counts[1] + train_counts[2]:,}** |
| Capped training counts (0 / 1 / 2) | {train_counts[0]:,} / {train_counts[1]:,} / {train_counts[2]:,} |
| Class weights | `compute_sample_weight('balanced', y)` |
| Weak-row weight | 0.5 |
| Validation rows (uncapped distribution) | {gen['partitions']['G']['metrics']['validation']['support']:,} |
| Test rows (uncapped distribution) | {test_g['support']:,} |
| Artifact | `models/{metadata['artifact']}` |
| SHA-256 prefix | `{metadata['artifact_sha256_prefix']}` |

The class-2 cap is applied to **training rows only**. Validation and test keep the real
distribution. The deployed cap is 2:1 (2,277 class-0, 8 class-1, 4,554 class-2 rows), a
safety-oriented choice; it is not the best validation-macro-F1 recipe (uncapped 0.4465 vs
cap 2:1 0.4304).

## 9. Calibration and abstention (`src/brick14_domain_calibration.py`, rewritten by Brick 17)

| Quantity | Value |
|---|---:|
| Calibration method | temperature scaling |
| Temperature | {calibration['temperature']} |
| Fitted on | {calibration['fitted_on']} |
| Acceptance threshold | {calibration['accept_threshold']} |
| Validation coverage at threshold | {pct(calibration['threshold_validation_coverage'], 2)} |
| Test coverage at threshold | {pct(calibration['test_selective']['coverage'], 2)} |
| Test accepted rows | {calibration['test_selective']['accepted']:,} |
| Test accuracy on accepted rows | {pct(calibration['test_selective']['accuracy_on_accepted'], 2)} |
| Validation ECE raw -> calibrated | {num(calibration['validation_before']['ece'])} -> {num(calibration['validation_after']['ece'])} |
| Test ECE raw -> calibrated | {num(calibration['test_before']['ece'])} -> {num(calibration['test_after']['ece'])} |

## 10. Applicability domain (`src/brick14_domain_calibration.py`)

Morgan radius 2, 2048 bits, maximum Tanimoto similarity to training-only references.

| Side | Threshold | References | Rule |
|---|---:|---:|---|
| Drug | {domain['drug']['threshold']} | {domain['drug']['reference_count']:,} | {domain['drug']['rule']} |
| Food | {domain['food']['threshold']} | {domain['food']['reference_count']:,} | {domain['food']['rule']} |

## 11. Tanimoto similarity on the food dataset

Tanimoto similarity is used twice on foods: near-duplicate removal in cleaning and the food
side of the applicability domain. This build recomputes it from all **{n_compounds:,}** cleaned compounds
with the same Morgan radius 2 / 2048-bit settings.

| Quantity | Value |
|---|---:|
| Pairwise comparisons | {n_compounds * (n_compounds - 1) // 2:,} |
| Mean pairwise similarity | {num(similarity_sum / similarity_pairs)} |
| Max pairwise similarity in final cleaned set | {num(max_pair_similarity)} |
| Pairs at or above the 0.75 cleaning cutoff | {threshold_counts[0.75] // 2:,} |
| ... of which involve a protected marker compound | {protected_pairs_above_cutoff:,} |
| ... of which involve no protected compound | {nonprotected_pairs_above_cutoff:,} |
| Pairs at or above 0.90 | {threshold_counts[0.90] // 2:,} |
| Nearest-neighbour median | {num(float(np.median(nn_values)))} |
| Nearest-neighbour max | {num(float(nn_values.max()))} |
| Training food references | {len(ref_fps):,} |
| Unseen organic compounds | {len(unseen_organic):,} |
| Unseen organic inside food domain | {unseen_inside:,} |
| Unseen organic outside food domain | {unseen_outside:,} |

The {threshold_counts[0.75] // 2} pairs at or above 0.75 all involve compounds explicitly
protected in `config.yaml` (phylloquinone, naringin, bergamottin); the non-protected pair count
above the cutoff is **{nonprotected_pairs_above_cutoff}**. This is the intended protected-marker
bypass, not an unnoticed duplicate: the cleaning removes near-duplicates at 0.75 but keeps the
marker compounds needed for known interaction cases. Detailed neighbours are in
`food_tanimoto_neighbors.csv`, and the per-compound domain result is in
`food_tanimoto_applicability.csv`.

## 12. Evidence, predictions, and evaluation rows

| Artifact | Rows |
|---|---:|
| Dataset evidence lookup (`evidence_S.csv`) | {len(evidence):,} |
| Split G test predictions from the deployed artifact (`model_predictions_G.csv`) | {len(pred_g):,} |
| Split G test pairs | {test_g['support']:,} |
| Split R test pairs | {test_r['support']:,} |
| Split F test pairs | {test_f['support']:,} |
| Split D test pairs | {test_d['support']:,} |

## 13. Headline test scores (exact)

| Split | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | MCC | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| R pair-random | {num(test_r['accuracy'])} | {num(test_r['balanced_accuracy'])} | {num(test_r['macro_f1'])} | {num(test_r['weighted_f1'])} | {num(test_r['mcc'])} | {num(test_r['roc_auc_ovr_macro'])} | {num(test_r['pr_auc_macro'])} |
| G unseen drugs | {num(test_g['accuracy'])} | {num(test_g['balanced_accuracy'])} | {num(test_g['macro_f1'])} | {num(test_g['weighted_f1'])} | {num(test_g['mcc'])} | {num(test_g['roc_auc_ovr_macro'])} | {num(test_g['pr_auc_macro'])} |
| F unseen compounds | {num(test_f['accuracy'])} | {num(test_f['balanced_accuracy'])} | {num(test_f['macro_f1'])} | {num(test_f['weighted_f1'])} | {num(test_f['mcc'])} | {num(test_f['roc_auc_ovr_macro'])} | {num(test_f['pr_auc_macro'])} |
| D double-cold | {num(test_d['accuracy'])} | {num(test_d['balanced_accuracy'])} | {num(test_d['macro_f1'])} | {num(test_d['weighted_f1'])} | {num(test_d['mcc'])} | NA | {num(test_d['pr_auc_macro'])} |

Split G is the deployed artifact `models/{metadata['artifact']}` scored once on unseen
drugs. Split R, F, and D scores come from refitting the same fixed LightGBM recipe on each
partition (the evaluation-stage protocol); they measure the recipe's generalization and are
not additional predictions from the deployed artifact. `model_predictions_G.csv` therefore
contains only Split G rows, and no Split R prediction file is shipped because the Brick 9
R parquet comes from a separately retrained R model.

## 14. File index

| File | What it contains |
|---|---|
| `drug_cleaned.csv` | {len(drugs):,} cleaned drugs |
| `food_cleaned.csv` | {len(foods):,} cleaned food compounds |
| `pairs.csv` | {len(splits):,} labelled Mode S pairs with names and splits |
| `features_generation.csv` | {len(drug_moe) + len(food_moe):,} molecules x 60 MOE descriptors |
| `feature_extraction.csv` | {len(featurized):,} pairs x 18 selected pair features |
| `feature_selection_funnel.csv` | 3,780 -> 18 feature-selection funnel |
| `feature_selection.csv` | 18 selected features with meanings |
| `split_assignments.csv` | Every pair with Split R and Split G assignment |
| `split_counts.csv` | Row/class counts per split and partition |
| `model_predictions_G.csv` | {len(pred_g):,} Split G test rows scored by the deployed artifact (calibrated and raw probabilities) |
| `evidence_S.csv` | {len(evidence):,} dataset evidence records |
| `calibration_S.csv` | Raw/calibrated NLL, Brier, ECE |
| `applicability_domain_S.csv` | Drug/food AD thresholds and references |
| `food_tanimoto_neighbors.csv` | Nearest Tanimoto neighbour for every food compound |
| `food_tanimoto_applicability.csv` | Food similarity to training references and AD status |
| `food_tanimoto_summary.csv` | Food Tanimoto summary statistics |
| `class_balance_sweep.csv` | Four class-ratio recipes on validation |
| `model_comparison_S.csv` | Five model families on Split G and R |
| `baselines_S.csv` | Majority/logistic/LightGBM imbalance baselines |
| `raw_dataset_counts.csv` | Initial raw row/column/effect counts |
| `pipeline_counts.csv` | Input/output rows for every pipeline stage |

## 15. Serving-time resolution (`src/recommend.py`)

| Behaviour | Detail |
|---|---|
| Evidence order | The exact-pair dataset annotation is checked before descriptor and inorganic checks. Calcium pairs with a saved record therefore return `status=evidence` instead of an inorganic abstention. |
| Calcium evidence | {calcium_evidence_rows:,} non-weak dataset rows for compound 3514 (`Calcium`, `[Ca+2]`). |
| Milk resolution | `milk` returns the marker compound calcium plus the organic compounds linked to {", ".join(milk_foods)} ({milk_rows:,} link rows, {milk_compounds:,} unique compounds); candidates are capped at {SERVING['max_marker_candidates']}. |
| Mineral fallback | An inorganic compound with no dataset record falls back to the organic compounds those foods contain (frequency <= {SERVING['compound_frequency_cutoff']} foods, capped at {SERVING['max_mineral_candidates']}). The result is labelled as a food-level match. |
| Candidate limits | `config.yaml` `serving` sets the frequency cutoff ({SERVING['compound_frequency_cutoff']}), marker cap ({SERVING['max_marker_candidates']}), food cap ({SERVING['max_food_candidates']}), mineral cap ({SERVING['max_mineral_candidates']}), marker-name match cap ({SERVING['max_marker_name_matches']}), food-name aliases ({len(SERVING['aliases'])}), marker-compound fallbacks ({len(SERVING['food_markers'])} foods), built-in structures ({len(SERVING['smiles_fallback'])}), and the stripped food suffixes ({len(SERVING['food_suffixes'])}); the prediction path contains no food-specific code branches. |
| Low-confidence display | A result below the {calibration['accept_threshold']} threshold keeps its predicted class label in the UI; calibrated confidence and the abstention reason remain in the response and the details block. |
| SHAP coverage | Every answered result carries a TreeSHAP block with all 18 features and all 3 classes. Rule and dataset-evidence answers compute it for the matched compound and expose it through `model_support` (the answer itself still comes from the rule or annotation). Outside-domain and inorganic abstentions carry a diagnostic SHAP block. Food-level candidates each carry their own SHAP block. `shap_available`/`shap_reason` state the outcome on every response; only inputs with no resolvable structure on either side lack SHAP. |
| Config-driven display | Class labels, severities, response sentences, the disclaimer and the demo pairs come from `config.yaml` (`labels:`, `web:`). The browser renders the API's `predicted_label`; it keeps no class-name map of its own. |
"""
(STEP / "context.md").write_text(context, encoding="utf-8")

# ---------------------------------------------------------------------------
# documentation.md
# ---------------------------------------------------------------------------
documentation = f"""# Documentation

This project is a research drug-food interaction predictor. It is not clinical
advice. The deployed artifact is **{metadata['model_name']} {metadata['model_version']}**
(historical artifact name `S`); there is one served model, not a paper-style Mode P model.

## 1. ML

### 1.1 Exact pipeline and file names

| # | Stage | File | What it does |
|---:|---|---|---|
| 1 | Raw audit | `src/brick1_audit.py` | Reads the three raw CSVs, records row/column counts and effect counts. |
| 2 | Drug cleaning | `src/brick2_clean_drugs.py` | Parses structures with RDKit, removes empty/failed structures and duplicate InChIKey14 values; writes `drugs_clean.parquet`. |
| 3 | Food cleaning | `src/brick3_clean_foods.py` | Corrects the swapped food columns, parses compounds, removes InChIKey duplicates, removes Morgan/Tanimoto near-duplicates at 0.75, protects marker compounds, flags organic/inorganic; writes `foods_clean.parquet`. |
| 4 | Labels | `src/brick4_labels.py` | Maps the raw `Effect` values to classes 0/1/2, applies negative precedence, builds Mode S pairs, creates weak labels, holds out 23 case-study pairs; writes `splits_S.parquet` inputs. |
| 5 | MOE descriptors | `src/brick5_features.py` | Computes the 60 PyBioMed MOE descriptors for {feature_report['drug_molecules']:,} drugs and {feature_report['food_molecules']:,} organic foods; writes `drug_moe.parquet` and `food_moe.parquet`. |
| 6 | Pair features | `src/features.py` | Builds {selection_log['counts']['start']:,} pair features: 3,600 cross-products, 60 sums, 60 absolute differences, 60 ratios. |
| 7 | Splits | `src/brick6_splits.py` | Creates Split R (pair-random) and Split G (whole-drug groups), plus the F and D cold partitions used by brick 15. |
| 8 | Feature selection | `src/brick7_selection.py --mode S` | Variance threshold -> Pearson correlation -> Lasso -> RFECV -> RFE to 18 features on Split G training rows. |
| 9 | Model tuning | `src/brick8_models.py --mode S` | Tunes five model families on Split G training data and records the LightGBM parameter set later used for deployment. |
| 10 | Generalization evaluation | `src/brick15_generalization.py --mode S --cap-ratio 2` | Trains the same fixed LightGBM recipe on R/G/F/D partitions and scores every test partition once. |
| 11 | Domain + calibration | `src/brick14_domain_calibration.py --mode S` | Fits the Morgan/Tanimoto applicability domain and temperature scaling on validation data. |
| 12 | Single deployed model | `src/brick17_single_model.py --ratios 1,2,3 --deploy-ratio 2 --apply` | Sweeps class-2 caps, refits the tuned LightGBM on the 2:1 capped training rows, rewrites calibration for the deployed artifact, and saves `best_model_S.joblib`. |
| 13 | Evidence and rules | `src/brick13_build_evidence.py` | Builds the {len(evidence):,}-row evidence lookup, serving metadata, and the versioned rule document. |
| 14 | Baselines | `src/brick16_baselines.py --mode S` | Scores majority, logistic, and LightGBM imbalance baselines on the same Split G test rows. |
| 15 | Serving explanations | `src/explain.py` | Computes TreeSHAP values for all 18 features and all 3 classes per request. |
| 16 | Deliverables | `scripts/export_stage_csvs.py`, `scripts/build_deliverables.py` | Writes the stage CSVs, `step_csvs/`, and `final_metrics/`. |

The orchestrator is `scripts/run_all.sh`. It runs the stages in the order above (with the
report-only experiments) and ends with `scripts/check_gates.py`.

### 1.2 Techniques actually used

- RDKit parsing, canonical SMILES, InChIKey14 duplicate removal.
- Morgan fingerprints (radius 2, 2048 bits) and Tanimoto similarity for food near-duplicate removal and for both sides of the applicability domain.
- PyBioMed MOE descriptors (60 per molecule).
- Pairwise descriptor arithmetic: cross-products, sums, absolute differences, ratios.
- `VarianceThreshold(0.8)`, Pearson correlation pruning at |r| < 0.75, Lasso (alpha 0.01), RFECV, and RFE to 18 features.
- LightGBM gradient-boosted decision trees (GBDT).
- Balanced class weights via `compute_sample_weight('balanced', y)`.
- Weak-label downweighting (weak rows multiplied by 0.5).
- Training-only class-2 capping at 2:1.
- Group-aware and cold-start splits (pair-random, whole-drug, whole-compound, double-cold).
- Temperature scaling for probability calibration.
- Selective prediction with a validation-selected acceptance threshold ({calibration['accept_threshold']}).
- Applicability domain from maximum Tanimoto similarity to training-only references.
- TreeSHAP for per-request feature contributions.
- Comparison baselines: majority class, logistic regression, Random Forest, Extra Trees, XGBoost, and MLP. Only LightGBM is served.

### 1.3 Techniques not used

- No SMOTE, ADASYN, or random oversampling in the deployed model. The oversampled baseline exists only in `reports/baselines_S.csv`.
- No XGBoost, Random Forest, Extra Trees, or MLP in serving. They are comparison rows only.
- No deep learning, graph neural network, or pretrained molecular embedding.
- No external validation dataset. `reports/brick6_splits.json` records that the external Ryu et al. set was not obtained; Split G/F/D and the held-out case studies stand in.
- No calibration method other than temperature scaling.
- No conformal prediction in serving; the E2 conformal experiment is reported for completeness in `reports/brick12_enhancements.json` and is not used by the app.
- No uncertainty decomposition.

### 1.4 Final deployed model and configuration

| Setting | Value |
|---|---|
| Version | `{metadata['model_version']}` |
| Artifact | `models/{metadata['artifact']}` |
| SHA-256 prefix | `{metadata['artifact_sha256_prefix']}` |
| Estimator | `{type(base).__module__}.{type(base).__name__}` inside `brick8_models.BalancedClassifier` |
| Features | {metadata['feature_count']} selected pair features |
| Labels | 0 = Avoid together, 1 = Beneficial / take with food, 2 = No significant interaction |
| Training partition | Split G training rows (unseen-drug split), class-2 capped 2:1 |
| Training rows after cap | {train_counts[0] + train_counts[1] + train_counts[2]:,} ({train_counts[0]:,} / {train_counts[1]:,} / {train_counts[2]:,}) |
| Sample weights | balanced class weights x 0.5 for weak rows |
| Calibration | temperature scaling, T = {calibration['temperature']} |
| Accept threshold | {calibration['accept_threshold']} |
| Drug AD threshold | {domain['drug']['threshold']} (223 references) |
| Food AD threshold | {domain['food']['threshold']} (2,291 references) |

Exact `LGBMClassifier` parameters loaded from the deployed artifact:

| Parameter | Value |
|---|---|
| `boosting_type` | `{estimator_params['boosting_type']}` |
| `n_estimators` | {estimator_params['n_estimators']} |
| `learning_rate` | {estimator_params['learning_rate']} |
| `num_leaves` | {estimator_params['num_leaves']} |
| `max_depth` | {estimator_params['max_depth']} |
| `min_child_samples` | {estimator_params['min_child_samples']} |
| `min_child_weight` | {estimator_params['min_child_weight']} |
| `min_split_gain` | {estimator_params['min_split_gain']} |
| `subsample` | {estimator_params['subsample']} |
| `subsample_freq` | {estimator_params['subsample_freq']} |
| `colsample_bytree` | {estimator_params['colsample_bytree']} |
| `reg_alpha` | {estimator_params['reg_alpha']} |
| `reg_lambda` | {estimator_params['reg_lambda']} |
| `random_state` | {estimator_params['random_state']} |
| `n_jobs` | {estimator_params['n_jobs']} |
| `verbose` | {estimator_params['verbose']} |
| `objective` | `{estimator_params['objective']}` (inferred multiclass) |

### 1.5 Held-out scores

Split G test (unseen drugs, headline):

| Metric | Value |
|---|---:|
| Accuracy | {num(test_g['accuracy'])} |
| Balanced accuracy | {num(test_g['balanced_accuracy'])} |
| Macro F1 | {num(test_g['macro_f1'])} |
| Weighted F1 | {num(test_g['weighted_f1'])} |
| MCC | {num(test_g['mcc'])} |
| Cohen kappa | {num(test_g['cohen_kappa'])} |
| ROC AUC (OvR macro) | {num(test_g['roc_auc_ovr_macro'])} |
| PR AUC (macro) | {num(test_g['pr_auc_macro'])} |
| Confusion matrix (rows=true) | `{test_g['confusion_matrix']}` |

Split R test (pair-random): accuracy {num(test_r['accuracy'])}, balanced accuracy
{num(test_r['balanced_accuracy'])}, macro F1 {num(test_r['macro_f1'])}, MCC {num(test_r['mcc'])},
ROC AUC {num(test_r['roc_auc_ovr_macro'])}.

Split F test (unseen compounds): accuracy {num(test_f['accuracy'])}, balanced accuracy
{num(test_f['balanced_accuracy'])}, macro F1 {num(test_f['macro_f1'])}, MCC {num(test_f['mcc'])},
ROC AUC {num(test_f['roc_auc_ovr_macro'])}.

Split D test (double-cold): accuracy {num(test_d['accuracy'])}, balanced accuracy
{num(test_d['balanced_accuracy'])}, macro F1 {num(test_d['macro_f1'])}, MCC {num(test_d['mcc'])},
ROC AUC NA (no class-1 test pairs). Split D is a stress test with only {test_d['support']} pairs
and must not be presented as the headline.

Split G is the deployed artifact `models/{metadata['artifact']}`, and its confusion matrix is
reproduced exactly by `step_csvs/model_predictions_G.csv`. Splits R, F, and D are scored by
refitting the same fixed LightGBM recipe on each partition (see
`src/brick15_generalization.py`); they measure recipe generalization and are not additional
predictions of the deployed artifact.

Selective prediction at threshold {calibration['accept_threshold']} on Split G test: coverage
{pct(calibration['test_selective']['coverage'], 2)} ({calibration['test_selective']['accepted']:,}
of {test_g['support']:,} pairs accepted), accuracy on accepted rows
{pct(calibration['test_selective']['accuracy_on_accepted'], 2)}.

Class 1 has only {class_counts[1]} labelled pairs in the entire dataset and
{train_counts[1]} in the capped training set. Every class-1 score is unstable.

## 2. Backend

### 2.1 Framework and runtime

| Component | Choice |
|---|---|
| Web framework | FastAPI {__import__('fastapi').__version__} |
| ASGI server | Uvicorn {__import__('uvicorn').__version__} |
| Validation/serialization | Pydantic {__import__('pydantic').__version__} |
| Model persistence | joblib |
| Chemistry | RDKit |
| Name resolution | local database, Google Gemini `generateContent`, PubChem PUG REST |
| Entry point | `web/app.py` (`app = FastAPI(...)`) |
| Serving logic | `src/recommend.py` (`DFIRecommender`) |
| Name resolution | `src/resolve.py` (`NameResolver`) |
| Curated rules | `src/rules.py` (version `{__import__('rules').RULE_VERSION}`) |
| Applicability domain | `src/domain.py` |
| Per-request SHAP | `src/explain.py` |
| Environment loader | `src/env.py` (reads `dfi/.env`, shell values win) |

Run from `dfi/`:

```bash
uvicorn web.app:app --port 8000
```

`GEMINI_API_KEY` is read from the environment or `dfi/.env`. The key is never written to the
repository or returned by an endpoint.

### 2.2 API endpoints

| Method | Path | Purpose | Input / output |
|---|---|---|---|
| GET | `/` | Landing page | `landing.html` |
| GET | `/predict` | Predictor page | `index.html` |
| GET | `/about` | Method and provenance page | `about.html` |
| GET | `/static/*` | CSS, JS, images | static files |
| GET | `/figures/{{name}}` | Serve a report PNG by bare filename | `.png` filenames only; other names return 404 |
| GET | `/api/health` | Readiness and served-model status | drug/compound counts, thresholds, model metadata, resolution status, rules version |
| GET | `/api/meta` | Full method metadata | 18 feature names, rule document, AD thresholds/reference counts, abstention info |
| GET | `/api/examples` | Eight example pairs for the UI chips | list of `{{drug, food, note}}` |
| GET | `/api/suggest?kind=&q=&limit=` | Autocomplete; `kind=drug` or `kind=food` | prefix matches first, then substring matches |
| POST | `/api/predict` | Score one drug-food pair | `{{"drug": "...", "food": "..."}}`; returns prediction/provenance/explanation plus `inputs` and `latency_ms` |

`POST /api/predict` requires both fields (1-300 characters). Validation errors return 422; missing
model artifacts return 503. A successful response is one of:

| Status | Source | Meaning |
|---|---|---|
| `ok` | `ml_model` | Model probability is at or above the {calibration['accept_threshold']} acceptance threshold. |
| `uncertain` | `ml_model` | Model confidence is below {calibration['accept_threshold']}. The most likely class is still shown; calibrated confidence and the abstention reason stay in the details. |
| `evidence` | `dataset_evidence` | Exact pair has a saved dataset annotation; this is a lookup, not an ML prediction. A supporting TreeSHAP block is included for the same pair. |
| `rule` | `rule_based` | A curated rule matched before the model was called. A supporting TreeSHAP block is included for the matched compound. |
| `not_supported` | `none` | Unknown structure, descriptor failure, outside applicability domain, or an inorganic compound with no dataset record and no food-level fallback. Whenever a drug/food structure pair exists (outside-domain, inorganic), a diagnostic TreeSHAP block is still returned. |

### 2.3 What `POST /api/predict` does, step by step

1. Pydantic validates the JSON body and trims both strings.
2. `engine()` returns the singleton `DFIRecommender`, which loads the model, selected features, scaler, domain, rules, evidence, and name resolver once. A background thread warms the SHAP explainer.
3. Drug resolution: exact normalized local name first; otherwise `NameResolver` (Gemini, then PubChem), then a raw SMILES input. A local hit makes no network call.
4. The curated rule layer is evaluated on the drug name and canonical food key. A match returns `status=rule` and `prediction_source=rule_based`; local food resolution still runs (`resolve_remote=False`) so the matched compound can carry a supporting TreeSHAP block.
5. Food resolution: exact compound, built-in structure, marker compounds, or food-level links; unknown names go to `NameResolver` (Gemini, then PubChem). A user SMILES is used directly. Food names combine the configured marker compounds with the specific organic compounds linked to that food (for example `milk` returns calcium plus the compounds of {", ".join(milk_foods)}, {milk_compounds:,} unique compounds, capped at {SERVING['max_marker_candidates']} candidates).
6. Exact-pair evidence is checked first. A saved non-weak annotation returns `status=evidence` and is explicitly labelled a dataset lookup, even when the compound is an inorganic mineral ({calcium_evidence_rows:,} calcium rows are reachable this way). The served model still computes TreeSHAP for the same pair and returns it as `model_support`.
7. For each remaining candidate, `_score_pair` computes the 60 MOE descriptors. An inorganic compound with no dataset record and no food-linked organic fallback is returned as `status=not_supported` (no class is claimed) but still carries a diagnostic TreeSHAP block. A mineral with no dataset record falls back to the organic compounds of the foods that contain it, labelled as a food-level match.
8. Applicability domain: Morgan radius 2 / 2048-bit maximum Tanimoto to training-only drugs and foods. Outside -> `status=not_supported` with reason `outside_applicability_domain`.
9. The 18 selected pair features are built in the same order as training, the calibrated LightGBM `predict_proba` is called, and temperature scaling is applied.
10. If the top calibrated probability is below {calibration['accept_threshold']}, the result is `status=uncertain` with `abstained=true`; the UI still shows the most likely class, with the confidence and abstention reason in the details. Otherwise the result is `status=ok`.
11. TreeSHAP computes signed contributions for all 18 features and all 3 classes on every answered path. ML answers use it as the decision explanation; rule and evidence answers expose it through `model_support`; outside-domain and inorganic abstentions keep it as a diagnostic. Every response reports `shap_available` and, when false, `shap_reason`; only inputs with no featurizable structure lack SHAP.
12. If food-level matching produced several candidate compounds, the most cautious confident result wins (evidence-avoid first, then confident avoid, then evidence, then highest confidence); the candidate list is attached, and each candidate carries its own label, severity and full TreeSHAP block.
13. The response includes resolution provenance, AD values, calibration info, model version, probabilities, `model_support` when the answer is a rule/evidence/diagnostic, and a research disclaimer. `web/app.py` adds `probability_rows`, `label`, `severity`, `inputs`, and `latency_ms`.

### 2.4 Backend files

- `web/app.py` - routes, request model, singleton engine, response augmentation.
- `src/recommend.py` - resolution order, rule layer, evidence lookup, AD, scoring, abstention, SHAP.
- `src/resolve.py` - Gemini/PubChem resolution, canonicalization, JSON cache.
- `src/domain.py` - Morgan/Tanimoto applicability domain.
- `src/explain.py` - TreeSHAP explainer.
- `src/rules.py` - versioned curated rules.
- `src/env.py` - `.env` loader with shell precedence.
- `models/best_model_S.joblib`, `models/calibration_S.json`, `models/domain_S.json`, `models/selected_features_S.json`, `models/model_metadata_S.json` - serving artifacts.

## 3. Frontend

### 3.1 Tech stack

- HTML5 pages: `web/static/landing.html`, `web/static/index.html`, `web/static/about.html`.
- Hand-written CSS in `web/static/styles.css` (no framework, no build step).
- Vanilla JavaScript (ES2020) in `web/static/app.js`; Fetch API and `AbortController`.
- Native `<datalist>` autocomplete; inline SVG icons; no external runtime dependency.
- The frontend never computes a prediction. Every score, probability, evidence record, and SHAP value comes from `POST /api/predict`.

### 3.2 Pages

| Page | Route | Role |
|---|---|---|
| Landing | `/` | Explains the project and links to the predictor and method page. |
| Predictor | `/predict` | Two inputs (drug and food, name or SMILES), example chips, result panel. |
| About | `/about` | Method, provenance, limitations, and served-model explanation. |

### 3.3 End-to-end flow after clicking "Predict interaction"

1. The form `submit` handler prevents the browser's default navigation and calls `runPrediction()` (`web/static/app.js`).
2. `runPrediction()` trims both inputs. If either is empty, it writes "Enter both a drug and a food compound." and focuses the missing field; no request is sent.
3. Any earlier in-flight request is aborted with `AbortController`. A new controller is stored, the button is disabled, its label becomes "Scoring...", and a skeleton loader replaces the result panel.
4. The browser sends `POST /api/predict` with `Content-Type: application/json` and `{{"drug": "...", "food": "..."}}`.
5. FastAPI validates the body. The singleton recommender resolves the drug, checks curated rules, resolves the food (exact compound, markers plus food-linked compounds, then Gemini/PubChem), checks exact-pair evidence, checks the applicability domain, computes descriptors and the 18 pair features, runs the calibrated LightGBM model, applies the {calibration['accept_threshold']} threshold, and computes TreeSHAP for all features and classes. Mineral queries such as `calcium` and food-level queries such as `milk` therefore return a result instead of an empty state.
6. The server returns JSON with `status`, `prediction_source`, `predicted_class`/`predicted_label`, `sentence`, `reason`, `probability_rows` (if probabilities exist), `confidence`, `applicability_domain`, `calibration`, `explainability`, `resolution`, `inputs`, `latency_ms`, and `disclaimer`.
7. If `res.ok` is false, `renderError()` shows the `detail` message. A network failure shows the "API is unreachable" message with the Uvicorn command.
8. If the request succeeds, `renderResult()` chooses the badge and severity:
   - `ok` + `ml_model` -> class badge and "ML model" source tag.
   - `rule` -> class badge with "Curated rule" source tag.
   - `evidence` -> class badge with "Dataset evidence" source tag.
   - `uncertain` -> class badge with the "ML model" source tag; calibrated confidence and the abstention reason appear in the details.
   - `not_supported` -> `renderUnsupported()` with the abstention reason, AD values, and resolution details.
9. The verdict shows the human sentence with the matched drug and food highlighted, then the reason. When probabilities exist, three bars animate to class 0/1/2 percentages.
10. The details block shows matched names, prediction source, model name/version, training split, AD status and thresholds, calibration temperature, rule ID/provenance, evidence record/annotation/PMID/DOI, resolution source and SMILES, abstention reason, original inputs, and any food-level candidate list.
11. SHAP rendering shows every feature for the predicted class and collapsible blocks for the other two classes, for `ok`, `uncertain`, `rule`, `evidence`, and outside-domain abstentions. Rule and evidence results also print the `model_support` note and the explained compound. Each row is a signed contribution bar with the numeric value; the caveat explains that these are model contributions, not causal effects.
12. The disclaimer strip is always appended ("Research tool; confirm with a physician or pharmacist.").
13. The `finally` block re-enables the button, restores "Predict interaction", clears the loading state, and removes `aria-busy`.
14. On page load, `loadHealth()` calls `/api/health` for the ready pill and `loadExamples()` calls `/api/examples` for the chips. Typing in either field debounces 150 ms, aborts the previous suggestion request, and calls `/api/suggest` for up to 8 options.

### 3.4 Safety and presentation details

- User-visible text is inserted with `textContent`; only fixed SVG markup uses `innerHTML`.
- Result severity is driven by `status` and class, not by colour alone; badges carry text labels.
- The UI always shows the prediction source, so a curated rule or dataset lookup is never presented as an ML prediction.
- Abstention is a first-class result, not an error.
"""
(STEP / "documentation.md").write_text(documentation, encoding="utf-8")

# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
step_manifest["context.md"] = {
    "bytes": (STEP / "context.md").stat().st_size,
    "description": "Initial dataset rows and the row count after every pipeline step.",
    "sources": ["all saved artifacts listed inside the file"],
}
step_manifest["documentation.md"] = {
    "bytes": (STEP / "documentation.md").stat().st_size,
    "description": "Three-section documentation: ML, Backend, Frontend.",
    "sources": ["all saved artifacts and source files referenced inside the file"],
}
write_json({
    "generated": datetime.now().isoformat(timespec="seconds"),
    "model_version": metadata["model_version"],
    "model_sha256_prefix": metadata["artifact_sha256_prefix"],
    "files": step_manifest,
}, STEP / "manifest.json")

write_json({
    "generated": datetime.now().isoformat(timespec="seconds"),
    "model_version": metadata["model_version"],
    "model_sha256_prefix": metadata["artifact_sha256_prefix"],
    "files": metrics_manifest,
}, METRICS / "manifest.json")

print(f"wrote {len(step_manifest)} entries to {STEP}")
print(f"wrote {len(metrics_manifest)} entries to {METRICS}")
print("model sha256 prefix:", metadata["artifact_sha256_prefix"])
print("done")
