"""Model baselines, imbalance experiments, and system-level path comparison.

All models see the same Split G training rows and selected features. The test
partition is scored once per model. Rule-only and evidence-only paths are
reported separately; evidence is a dataset lookup and cannot be counted as a
predictive model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score

from brick8_models import BalancedClassifier, build_features
from brick15_generalization import metrics
from common import dataframe_to_markdown, load_config, load_json, save_json, setup_stdout
from rules import evaluate_rules

sys.modules["__main__"].BalancedClassifier = BalancedClassifier
LABELS = [0, 1, 2]


def fit_lgbm(train, params, balanced: bool, weak_weight: float):
    base = LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
    model = BalancedClassifier(base=base) if balanced else base
    sw = np.where(train["weak"], weak_weight, 1.0)
    model.fit(train["X"], train["y"], sample_weight=sw)
    return model


def oversample(X, y, weak, seed: int):
    rng = np.random.default_rng(seed)
    counts = {c: int((y == c).sum()) for c in LABELS}
    target = max(counts.values())
    picks = [np.arange(len(y))]
    for c in LABELS:
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        extra = target - idx.size
        if extra > 0:
            picks.append(rng.choice(idx, size=extra, replace=True))
    chosen = np.concatenate(picks)
    rng.shuffle(chosen)
    return X[chosen], y[chosen], weak[chosen]


def evaluate_model(model, part) -> dict:
    proba = model.predict_proba(part["X"])
    pred = proba.argmax(axis=1)
    return metrics(part["y"], pred, proba)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["S", "P"], default="S")
    args = ap.parse_args()
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models"])
    reports = Path(cfg["paths"]["reports"])
    processed = Path(cfg["paths"]["processed"])
    interim = Path(cfg["paths"]["interim"])

    sel = load_json(models_dir / f"selected_features_{args.mode}.json")
    names = [f["name"] for f in sel["features"]]
    data, _, _ = build_features(cfg, args.mode, names)
    model_report = load_json(reports / f"brick8_models_{args.mode}.json")
    params = model_report["tuning"]["LightGBM"]["best_params"]
    weak_weight = float(cfg["models"].get("weak_sample_weight", 0.5))

    experiments = {}
    print("majority class...")
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(data["train"]["X"], data["train"]["y"])
    experiments["majority_class"] = majority

    print("balanced logistic regression...")
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ])
    logistic.fit(data["train"]["X"], data["train"]["y"])
    experiments["logistic_balanced"] = logistic

    print("LightGBM unweighted...")
    experiments["lightgbm_unweighted"] = fit_lgbm(data["train"], params, balanced=False, weak_weight=1.0)
    print("LightGBM balanced...")
    experiments["lightgbm_balanced"] = fit_lgbm(data["train"], params, balanced=True, weak_weight=1.0)
    print("LightGBM balanced + weak weight...")
    experiments["lightgbm_balanced_weak"] = fit_lgbm(
        data["train"], params, balanced=True, weak_weight=weak_weight
    )

    print("LightGBM balanced + random oversampling...")
    Xo, yo, wo = oversample(data["train"]["X"], data["train"]["y"], data["train"]["weak"], cfg["seed"])
    oversampled = {"X": Xo, "y": yo, "weak": wo}
    experiments["lightgbm_balanced_oversampled"] = fit_lgbm(
        oversampled, params, balanced=True, weak_weight=weak_weight
    )

    rows = []
    details = {}
    for name, model in experiments.items():
        val = evaluate_model(model, data["val"])
        test = evaluate_model(model, data["test"])
        details[name] = {"validation": val, "test": test}
        rows.append({
            "model": name, "partition": "validation", "n": val["support"],
            "accuracy": round(val["accuracy"], 4),
            "balanced_accuracy": round(val["balanced_accuracy"], 4),
            "macro_f1": round(val["macro_f1"], 4),
            "mcc": round(val["mcc"], 4),
            "class_1_support": val["class_counts"][1],
        })
        rows.append({
            "model": name, "partition": "test", "n": test["support"],
            "accuracy": round(test["accuracy"], 4),
            "balanced_accuracy": round(test["balanced_accuracy"], 4),
            "macro_f1": round(test["macro_f1"], 4),
            "mcc": round(test["mcc"], 4),
            "class_1_support": test["class_counts"][1],
        })
        print(name, "test macroF1", round(test["macro_f1"], 4), "bal acc", round(test["balanced_accuracy"], 4))

    table = pd.DataFrame(rows)
    table.to_csv(reports / f"baselines_{args.mode}.csv", index=False)

    # ---- System-level path comparison on the same test partition.
    drugs = pd.read_parquet(interim / "drugs_clean.parquet").set_index("drug_id")["name"].to_dict()
    foods = pd.read_parquet(interim / "foods_clean.parquet").set_index("compound_id")["name"].to_dict()
    test_df = data["test"]["df"].copy()
    test_df["drug_name"] = test_df["drug_id"].map(drugs)
    test_df["compound_name"] = test_df["compound_id"].map(foods)
    y = test_df["label"].to_numpy()

    rule_pred = np.full(len(test_df), -1, dtype=int)
    rule_ids = []
    for i, row in enumerate(test_df.itertuples(index=False)):
        match = evaluate_rules(str(row.drug_name), str(row.compound_name))
        if match is not None:
            rule_pred[i] = int(match.label)
            rule_ids.append(match.rule_id)
    rule_cov = rule_pred >= 0

    evidence = pd.read_parquet(processed / f"evidence_{args.mode}.parquet")
    evidence_keys = set(zip(evidence["drug_id"].astype(str), evidence["compound_id"].astype(int)))
    ev_cov = np.array([
        (str(d), int(c)) in evidence_keys
        for d, c in zip(test_df["drug_id"], test_df["compound_id"])
    ])
    ev_label = test_df["label"].to_numpy()

    deployed = joblib.load(models_dir / f"best_model_{args.mode}.joblib")
    ml_pred = deployed.predict(data["test"]["X"])
    ml_correct = ml_pred == y

    hybrid = ml_pred.copy()
    hybrid[rule_cov] = rule_pred[rule_cov]

    system_rows = [
        {
            "path": "rule_only", "coverage": float(rule_cov.mean()),
            "accuracy_on_covered": float((rule_pred[rule_cov] == y[rule_cov]).mean()) if rule_cov.any() else None,
            "accuracy_on_all_with_abstention_as_wrong": float((rule_pred[rule_cov] == y[rule_cov]).sum() / len(y)),
            "note": "Curated rules; no coverage for pairs outside the rule set.",
        },
        {
            "path": "evidence_lookup", "coverage": float(ev_cov.mean()),
            "accuracy_on_covered": 1.0 if ev_cov.any() else None,
            "accuracy_on_all_with_abstention_as_wrong": float(ev_cov.mean()),
            "note": "LABEL ECHO: the lookup returns the dataset label for the same pair. Not a predictive model and excluded from model comparison.",
        },
        {
            "path": "ml_only", "coverage": 1.0,
            "accuracy_on_covered": float(ml_correct.mean()),
            "accuracy_on_all_with_abstention_as_wrong": float(ml_correct.mean()),
            "note": "Deployed model, no rules and no evidence lookup.",
        },
        {
            "path": "hybrid_rule_then_ml", "coverage": 1.0,
            "accuracy_on_covered": float((hybrid == y).mean()),
            "accuracy_on_all_with_abstention_as_wrong": float((hybrid == y).mean()),
            "note": "Rules override ML when matched; evidence lookup is not used.",
        },
    ]
    system = pd.DataFrame(system_rows)
    system.to_csv(reports / f"system_comparison_{args.mode}.csv", index=False)

    report = {
        "mode": args.mode,
        "experiment": "fixed LightGBM hyperparameters from brick8; balanced weights and weak weight "
                      f"{weak_weight}; no resampling outside the training partition",
        "class_counts": {
            "train": {int(k): int(v) for k, v in pd.Series(data["train"]["y"]).value_counts().items()},
            "validation": {int(k): int(v) for k, v in pd.Series(data["val"]["y"]).value_counts().items()},
            "test": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        },
        "metrics": details,
        "system_paths": system_rows,
        "rule_ids_fired_on_test": {rid: int(rule_ids.count(rid)) for rid in sorted(set(rule_ids))},
        "notes": [
            "Class 1 support is <10 in every partition; its metrics are not reliable.",
            "SMOTE was not applied because imbalanced-learn is not a project dependency; random oversampling is the tested resampling method.",
            "Evidence lookup is shown for coverage only. Its 100% accuracy is tautological.",
            "The hybrid path does not use evidence lookup, to avoid turning evaluation into label retrieval.",
        ],
    }
    save_json(report, reports / f"baselines_{args.mode}.json")

    lines = [
        f"# Baselines and system paths (Mode {args.mode})",
        "",
        "All models use the same Split G train/validation/test rows and the same "
        f"{len(names)} selected features. Hyperparameters come from the saved LightGBM tuning run.",
        "",
        "## Model comparison",
        "",
        table.to_markdown(index=False),
        "",
        "## Class imbalance experiments",
        "",
        f"Train class counts: {report['class_counts']['train']}. Test class counts: {report['class_counts']['test']}. "
        "Random oversampling is applied to training rows only; validation and test are untouched.",
        "",
        "## System paths on the test partition",
        "",
        system.to_markdown(index=False),
        "",
        "## Rules fired on test",
        "",
        str(report["rule_ids_fired_on_test"]),
        "",
        "## Notes and limitations",
        "",
    ] + [f"- {n}" for n in report["notes"]]
    (reports / f"baselines_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n", table.to_string(index=False))
    print("\n", system.to_string(index=False))


if __name__ == "__main__":
    main()
