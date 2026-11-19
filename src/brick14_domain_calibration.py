"""Fit applicability-domain thresholds and probability calibration on validation data.

Nothing here touches the final test partition until all choices are frozen.

Outputs:
  models/domain_{mode}.json
  models/calibration_{mode}.json
  reports/calibration_{mode}.{json,md}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score

from brick8_models import BalancedClassifier, build_features
from common import dataframe_to_markdown, load_config, load_json, save_json, setup_stdout
from domain import fingerprint
from splits_extra import split_unseen_compounds
from rdkit import DataStructs

sys.modules["__main__"].BalancedClassifier = BalancedClassifier

N_CLASSES = 3


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier(proba: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list]:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = float(accuracy_score(y[mask], pred[mask]))
        avg = float(conf[mask].mean())
        value += mask.sum() / total * abs(acc - avg)
        rows.append({
            "bin_low": round(float(lo), 3), "bin_high": round(float(hi), 3),
            "count": int(mask.sum()), "mean_confidence": round(avg, 4),
            "accuracy": round(acc, 4),
        })
    return float(value), rows


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    logp = np.log(np.clip(proba, 1e-12, 1.0))

    def objective(t: float) -> float:
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return nll(p, y)

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def temper(proba: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / t
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def threshold_sweep(proba: np.ndarray, y: np.ndarray, target_error: float = 0.20) -> dict:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    grid = np.round(np.arange(0.40, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        mask = conf >= t
        coverage = float(mask.mean())
        error = float(1.0 - accuracy_score(y[mask], pred[mask])) if mask.any() else 1.0
        rows.append({"threshold": float(t), "coverage": coverage, "error": error})
    eligible = [r for r in rows if r["error"] <= target_error]
    if eligible:
        chosen = max(eligible, key=lambda r: (r["coverage"], -r["threshold"]))
        policy = f"lowest threshold on validation with accepted error <= {target_error:.2f}"
    else:
        chosen = min(rows, key=lambda r: (r["error"], -r["coverage"]))
        policy = "no threshold met the error target; chose minimum validation error"
    accepted = conf >= chosen["threshold"]
    return {
        "target_error": target_error,
        "accept_threshold": chosen["threshold"],
        "policy": policy,
        "curve": rows,
        "validation_coverage": chosen["coverage"],
        "validation_error": chosen["error"],
        "validation_accepted": int(accepted.sum()),
    }


def domain_thresholds(cfg: dict, mode: str, drugs: pd.DataFrame, foods: pd.DataFrame) -> dict:
    processed = Path(cfg["paths"]["processed"])
    pairs = pd.read_parquet(processed / f"splits_{mode}.parquet")
    g_train = pairs[pairs["split_G"] == "train"]
    g_val = pairs[pairs["split_G"] == "val"]

    pairs_f = pairs.copy()
    pairs_f["split_F"] = split_unseen_compounds(pairs_f, cfg)
    parts = {
        s: set(pairs_f.loc[pairs_f["split_F"] == s, "compound_id"])
        for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if parts[a] & parts[b]:
            raise SystemExit(f"compound overlap in Split F: {a}/{b}")
    f_train = pairs_f[pairs_f["split_F"] == "train"]
    f_val = pairs_f[pairs_f["split_F"] == "val"]

    drug_smiles = drugs.set_index(drugs["drug_id"].astype(str))["smiles"].to_dict()
    food_smiles = foods.set_index(foods["compound_id"].astype(int))["smiles"].to_dict()

    def ref_fps(ids):
        out = []
        for i in ids:
            fp = fingerprint(drug_smiles.get(str(i)) if isinstance(i, str) else food_smiles.get(int(i)))
            if fp is not None:
                out.append(fp)
        return out

    def sims(ids, refs, kind):
        values = {}
        for i in set(ids):
            smi = drug_smiles.get(str(i)) if kind == "drug" else food_smiles.get(int(i))
            fp = fingerprint(smi)
            if fp is not None and refs:
                values[i] = float(max(DataStructs.BulkTanimotoSimilarity(fp, refs)))
        return np.array(list(values.values()))

    drug_refs = ref_fps(set(g_train["drug_id"].astype(str)))
    food_refs = ref_fps(set(f_train["compound_id"].astype(int)))
    drug_val = sims(g_val["drug_id"].astype(str), drug_refs, "drug")
    food_val = sims(f_val["compound_id"].astype(int), food_refs, "food")

    def summarize(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"count": 0}
        return {
            "count": int(values.size),
            "p05": round(float(np.percentile(values, 5)), 4),
            "p10": round(float(np.percentile(values, 10)), 4),
            "median": round(float(np.median(values)), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }

    drug_stats, food_stats = summarize(drug_val), summarize(food_val)
    # The 10th percentile keeps roughly 90% of the validation partitions inside
    # while flagging the most dissimilar 10% as outside.
    drug_threshold = float(drug_stats.get("p10", 0.3))
    food_threshold = float(food_stats.get("p10", 0.3))
    artifact = {
        "mode": mode,
        "validated": True,
        "fingerprint": {"radius": 2, "bits": 2048, "type": "morgan"},
        "drug": {
            "threshold": round(drug_threshold, 4),
            "rule": "10th percentile of validation similarities for unseen drugs (Split G validation)",
            "validation_distribution": drug_stats,
            "reference_count": len(drug_refs),
        },
        "food": {
            "threshold": round(food_threshold, 4),
            "rule": "10th percentile of validation similarities for unseen compounds (Split F validation)",
            "validation_distribution": food_stats,
            "reference_count": len(food_refs),
        },
        "reference": {
            "drug_ids": sorted(str(x) for x in set(g_train["drug_id"].astype(str))),
            "compound_ids": sorted(int(x) for x in set(f_train["compound_id"].astype(int))),
        },
        "note": (
            "Similarity is a chemical-space check, not a biological guarantee. "
            "Thresholds are derived from validation partitions only."
        ),
    }
    return artifact


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["S", "P"], default="S")
    args = ap.parse_args()
    cfg = load_config()
    models = Path(cfg["paths"]["models"])
    reports = Path(cfg["paths"]["reports"])
    interim = Path(cfg["paths"]["interim"])

    drugs = pd.read_parquet(interim / "drugs_clean.parquet")
    foods = pd.read_parquet(interim / "foods_clean.parquet")

    print("domain thresholds...")
    domain_artifact = domain_thresholds(cfg, args.mode, drugs, foods)
    save_json(domain_artifact, models / f"domain_{args.mode}.json")
    print("  drug threshold", domain_artifact["drug"]["threshold"], domain_artifact["drug"]["validation_distribution"])
    print("  food threshold", domain_artifact["food"]["threshold"], domain_artifact["food"]["validation_distribution"])

    print("building features...")
    sel = load_json(models / f"selected_features_{args.mode}.json")
    names = [f["name"] for f in sel["features"]]
    data, _, _ = build_features(cfg, args.mode, names)
    model = joblib.load(models / f"best_model_{args.mode}.joblib")

    y_va, y_te = data["val"]["y"], data["test"]["y"]
    raw_va = model.predict_proba(data["val"]["X"])
    raw_te = model.predict_proba(data["test"]["X"])
    temperature = fit_temperature(raw_va, y_va)
    cal_va = temper(raw_va, temperature)
    cal_te = temper(raw_te, temperature)
    sweep = threshold_sweep(cal_va, y_va)

    def score(proba, y):
        e, bins = ece(proba, y)
        return {
            "accuracy": float(accuracy_score(y, proba.argmax(axis=1))),
            "nll": nll(proba, y),
            "brier": brier(proba, y),
            "ece": e,
            "mean_confidence": float(proba.max(axis=1).mean()),
            "reliability_bins": bins,
        }

    accepted_te = cal_te.max(axis=1) >= sweep["accept_threshold"]
    test_selective = {
        "coverage": float(accepted_te.mean()),
        "accepted": int(accepted_te.sum()),
        "accuracy_on_accepted": float(
            accuracy_score(y_te[accepted_te], cal_te.argmax(axis=1)[accepted_te])
        ) if accepted_te.any() else None,
    }
    artifact = {
        "mode": args.mode,
        "method": "temperature_scaling",
        "temperature": round(temperature, 4),
        "fitted_on": "Split G validation (unseen drugs)",
        "accept_threshold": sweep["accept_threshold"],
        "threshold_policy": sweep["policy"],
        "threshold_target_error": sweep["target_error"],
        "threshold_validation_coverage": sweep["validation_coverage"],
        "validation_before": score(raw_va, y_va),
        "validation_after": score(cal_va, y_va),
        "test_before": score(raw_te, y_te),
        "test_after": score(cal_te, y_te),
        "test_selective": test_selective,
        "threshold_curve": sweep["curve"],
        "note": (
            "The test partition was scored only after the temperature and threshold "
            "were frozen on validation. Class 1 support is too small for a reliable "
            "class-1 calibration curve."
        ),
    }
    save_json(artifact, models / f"calibration_{args.mode}.json")
    save_json(artifact, reports / f"calibration_{args.mode}.json")

    lines = [
        f"# Calibration and abstention (Mode {args.mode})",
        "",
        f"Temperature **{temperature:.3f}** fitted on Split G validation. "
        f"Acceptance threshold **{sweep['accept_threshold']:.2f}**: {sweep['policy']}.",
        "",
        "## Probability quality",
        "",
        pd.DataFrame([
            {"partition": "validation raw", **{k: v for k, v in artifact["validation_before"].items() if k != "reliability_bins"}},
            {"partition": "validation calibrated", **{k: v for k, v in artifact["validation_after"].items() if k != "reliability_bins"}},
            {"partition": "test raw", **{k: v for k, v in artifact["test_before"].items() if k != "reliability_bins"}},
            {"partition": "test calibrated", **{k: v for k, v in artifact["test_after"].items() if k != "reliability_bins"}},
        ]).to_markdown(index=False),
        "",
        "## Abstention on the held-out test partition",
        "",
        f"Coverage {test_selective['coverage']:.3f} ({test_selective['accepted']} pairs); "
        f"accuracy on accepted predictions {test_selective['accuracy_on_accepted']}.",
        "",
        "## Validation threshold curve",
        "",
        pd.DataFrame(sweep["curve"]).to_markdown(index=False),
        "",
        artifact["note"],
        "",
        "## Applicability domain",
        "",
        f"Drug threshold {domain_artifact['drug']['threshold']} from {domain_artifact['drug']['validation_distribution']}; "
        f"food threshold {domain_artifact['food']['threshold']} from {domain_artifact['food']['validation_distribution']}.",
        "",
        domain_artifact["note"],
    ]
    (reports / f"calibration_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
