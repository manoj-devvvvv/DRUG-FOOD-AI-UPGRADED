"""Class-ratio sweep and the single deployed model.

Non-significant pairs dominate Mode S. This script caps that class in the
training partition only, compares caps of 1:1, 2:1 and 3:1 against the
uncapped recipe on validation, and refits the winner. Validation and test keep
the real class distribution. Test is scored once, after the ratio is chosen.

The chosen model overwrites the serving artifacts (best_model_S.joblib,
calibration_S.json, model_metadata_S.json). Feature selection is not rerun, so
the 18 features remain those selected on the uncapped Split G training rows.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

from brick8_models import BalancedClassifier, build_features
from brick14_domain_calibration import brier, ece, fit_temperature, nll, temper, threshold_sweep
from common import load_config, load_json, save_json, setup_stdout

LABELS = [0, 1, 2]


def cap_positions(frame: pd.DataFrame, ratio: float | None, seed: int) -> np.ndarray:
    """Return positional indices after capping class 2 at ratio * class 0.

    Strong class-2 rows are kept first; weak synthetic rows fill the remainder.
    If the cap is below the number of strong rows, strong rows are sampled.
    """
    rng = np.random.default_rng(seed)
    y = frame["label"].to_numpy()
    weak = frame["weak"].to_numpy(dtype=bool)
    pos = np.arange(len(frame))
    p0 = pos[y == 0]
    p1 = pos[y == 1]
    p2 = pos[y == 2]
    if ratio is None:
        return np.sort(np.r_[p0, p1, p2])

    strong = p2[~weak[y == 2]]
    weak_rows = p2[weak[y == 2]]
    target = min(len(p2), int(round(float(ratio) * len(p0))))
    if target <= len(strong):
        keep2 = rng.choice(strong, size=target, replace=False)
    else:
        extra = target - len(strong)
        keep2 = np.r_[strong, rng.choice(weak_rows, size=extra, replace=False)]
    return np.sort(np.r_[p0, p1, keep2])


def fit_lightgbm(X, y, weak, params, weak_weight, seed):
    base = LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbose=-1)
    sw = compute_sample_weight("balanced", y) * np.where(weak, weak_weight, 1.0)
    base.fit(X, y, sample_weight=sw)
    return base


def score(proba, y) -> dict:
    pred = proba.argmax(axis=1)
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support,
    )

    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class": {
            int(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def calibration_block(proba, y) -> dict:
    value, bins = ece(proba, y)
    return {
        "accuracy": float((proba.argmax(axis=1) == y).mean()),
        "nll": float(nll(proba, y)),
        "brier": float(brier(proba, y)),
        "ece": float(value),
        "mean_confidence": float(proba.max(axis=1).mean()),
        "reliability_bins": bins,
    }


def write_calibration_report(calibration: dict, reports: Path) -> None:
    """Write reports/calibration_S.{json,md} for the deployed model.

    Brick 14 produces the same filenames before deployment; this overwrites them
    so every report describes the artifact that is actually served.
    """
    save_json(calibration, reports / "calibration_S.json")
    rows = [
        {"partition": "validation raw", **calibration["validation_before"]},
        {"partition": "validation calibrated", **calibration["validation_after"]},
        {"partition": "test raw", **calibration["test_before"]},
        {"partition": "test calibrated", **calibration["test_after"]},
    ]
    frame = pd.DataFrame(rows)[["partition", "accuracy", "nll", "brier", "ece", "mean_confidence"]]
    selective = calibration["test_selective"]
    lines = [
        "# Calibration and abstention (deployed model)",
        "",
        f"Temperature **{calibration['temperature']:.4f}** fitted on Split G validation. "
        f"Acceptance threshold **{calibration['accept_threshold']:.2f}**: lowest threshold on "
        "validation with accepted error <= 0.20.",
        "",
        f"Training recipe: **{calibration['class_balance']['recipe']}** "
        f"(class 2 capped in the training partition only).",
        "",
        "## Probability quality",
        "",
        frame.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Abstention on the held-out test partition",
        "",
        f"Coverage {selective['coverage']:.4f} ({selective['accepted']} pairs); accuracy on "
        f"accepted predictions {selective['accuracy_on_accepted']:.4f}.",
        "",
        "## Validation threshold curve",
        "",
        pd.DataFrame(calibration["threshold_curve"]).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (reports / "calibration_S.md").write_text("\n".join(lines), encoding="utf-8")


def sha_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="1,2,3")
    ap.add_argument("--deploy-ratio", type=float, default=None,
                    help="deploy this cap instead of the validation macro-F1 winner")
    ap.add_argument("--apply", action="store_true", help="overwrite the serving artifacts")
    args = ap.parse_args()

    cfg = load_config()
    models = Path(cfg["paths"]["models"])
    reports = Path(cfg["paths"]["reports"])
    seed = int(cfg["seed"])
    weak_weight = float(cfg["models"].get("weak_sample_weight", 0.5))

    sel = load_json(models / "selected_features_S.json")
    names = [f["name"] for f in sel["features"]]
    data, _, _ = build_features(cfg, "S", names)
    params = load_json(reports / "brick8_models_S.json")["tuning"]["LightGBM"]["best_params"]

    train = data["train"]
    val = data["val"]
    ratios: list[float | None] = [None] + [float(x) for x in args.ratios.split(",") if x.strip()]

    rows = []
    fits: dict[float | None, object] = {}
    for ratio in ratios:
        positions = cap_positions(train["df"], ratio, seed)
        X = train["X"][positions]
        y = train["y"][positions]
        weak = train["weak"][positions]
        model = fit_lightgbm(X, y, weak, params, weak_weight, seed)
        val_proba = model.predict_proba(val["X"])
        m = score(val_proba, val["y"])
        counts = {int(c): int((y == c).sum()) for c in LABELS}
        rows.append({
            "recipe": "uncapped" if ratio is None else f"cap_{ratio:g}",
            "train_rows": int(len(y)),
            "train_counts": counts,
            "val_accuracy": m["accuracy"],
            "val_balanced_accuracy": m["balanced_accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_recall_0": m["per_class"][0]["recall"],
            "val_recall_1": m["per_class"][1]["recall"],
            "val_recall_2": m["per_class"][2]["recall"],
        })
        fits[ratio] = model
        print(f"ratio={ratio} train={counts} val macro-F1={m['macro_f1']:.4f} "
              f"balanced-acc={m['balanced_accuracy']:.4f} recall0={m['per_class'][0]['recall']:.3f} "
              f"recall1={m['per_class'][1]['recall']:.3f}")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(reports / "class_balance_sweep.csv", index=False, encoding="utf-8-sig")
    winner = max(rows, key=lambda r: (r["val_macro_f1"], r["val_balanced_accuracy"], -r["train_rows"]))
    winner_ratio = None if winner["recipe"] == "uncapped" else float(winner["recipe"].split("_")[1])
    if args.deploy_ratio is not None:
        best_ratio = float(args.deploy_ratio)
        matches = [r for r in rows if r["recipe"] == f"cap_{best_ratio:g}"]
        if not matches:
            raise SystemExit(f"deploy ratio {best_ratio} was not evaluated; include it in --ratios")
        best_row = matches[0]
        selection_note = (
            f"Deployed cap {best_ratio:g}:1 for avoid-class recall as a product decision. "
            f"The validation macro-F1 winner was {winner['recipe']} ({winner['val_macro_f1']:.4f} vs "
            f"{best_row['val_macro_f1']:.4f}); this is a trade-off, not a metric improvement."
        )
    else:
        best_ratio = winner_ratio
        best_row = winner
        selection_note = f"Selected {winner['recipe']} on validation macro-F1."
    print(f"\n{selection_note}")

    notes = [
        "Capping is applied to training rows only; validation and test keep the real class distribution.",
        selection_note,
        "Class 1 has only 8 training rows, so class-1 metrics remain unstable regardless of ratio.",
        "The 18 features were selected on the uncapped training partition and are reused unchanged.",
    ]

    if not args.apply:
        lines = [
            "# Class-ratio sweep (Mode S)",
            "",
            f"Selected recipe: **{best_row['recipe']}** on validation macro-F1.",
            "",
            sweep.to_markdown(index=False),
            "",
            "## Notes",
            "",
            *[f"- {n}" for n in notes],
        ]
        (reports / "class_balance_sweep.md").write_text("\n".join(lines), encoding="utf-8")
        print("wrote sweep report; rerun with --apply to deploy the winner")
        return

    positions = cap_positions(train["df"], best_ratio, seed)
    Xtr, ytr, wtr = train["X"][positions], train["y"][positions], train["weak"][positions]
    base = fit_lightgbm(Xtr, ytr, wtr, params, weak_weight, seed)
    deployed = BalancedClassifier(base)

    model_path = models / "best_model_S.joblib"
    joblib.dump(deployed, model_path)

    raw_va = base.predict_proba(val["X"])
    raw_te = base.predict_proba(data["test"]["X"])
    temperature = fit_temperature(raw_va, val["y"])
    cal_va = temper(raw_va, temperature)
    cal_te = temper(raw_te, temperature)
    sweep_threshold = threshold_sweep(cal_va, val["y"])
    accepted = cal_te.max(axis=1) >= sweep_threshold["accept_threshold"]
    test_selective = {
        "coverage": float(accepted.mean()),
        "accepted": int(accepted.sum()),
        "accuracy_on_accepted": float((cal_te.argmax(axis=1)[accepted] == data["test"]["y"][accepted]).mean())
        if accepted.any() else None,
    }

    calibration = {
        "mode": "S",
        "method": "temperature_scaling",
        "temperature": round(temperature, 4),
        "fitted_on": "Split G validation (unseen drugs)",
        "accept_threshold": sweep_threshold["accept_threshold"],
        "threshold_policy": sweep_threshold["policy"],
        "threshold_target_error": sweep_threshold["target_error"],
        "threshold_validation_coverage": sweep_threshold["validation_coverage"],
        "validation_before": calibration_block(raw_va, val["y"]),
        "validation_after": calibration_block(cal_va, val["y"]),
        "test_before": calibration_block(raw_te, data["test"]["y"]),
        "test_after": calibration_block(cal_te, data["test"]["y"]),
        "test_selective": test_selective,
        "threshold_curve": sweep_threshold["curve"],
        "class_balance": {
            "recipe": best_row["recipe"],
            "train_counts": best_row["train_counts"],
            "selection_split": "Split G validation",
            "capped_partition": "train only",
        },
        "note": (
            "Temperature is fitted on Split G validation only. The acceptance threshold is the lowest "
            "validation threshold whose accepted error is at most 0.20. Test was scored once after the "
            "ratio and threshold were fixed."
        ),
    }
    save_json(calibration, models / "calibration_S.json")
    write_calibration_report(calibration, reports)

    val_metrics = score(cal_va, val["y"])
    test_metrics = score(cal_te, data["test"]["y"])
    metadata = load_json(models / "model_metadata_S.json")
    metadata.update({
        "model_name": "LightGBM",
        "model_version": f"single-{date.today().isoformat()}",
        "artifact": str(model_path.name),
        "artifact_sha256_prefix": sha_prefix(model_path),
        "training_split": "Split G train (drug-grouped), class-2 capped",
        "class_balance": calibration["class_balance"],
        "accept_threshold": calibration["accept_threshold"],
        "validation_macro_f1": round(val_metrics["macro_f1"], 4),
        "created": str(date.today()),
    })
    save_json(metadata, models / "model_metadata_S.json")

    evaluation = {
        "model": metadata["model_version"],
        "recipe": best_row["recipe"],
        "feature_count": len(names),
        "validation": val_metrics,
        "test": test_metrics,
        "test_selective": test_selective,
        "calibration": {k: calibration[k] for k in
                        ("temperature", "accept_threshold", "threshold_validation_coverage")},
        "ratio_sweep": rows,
        "notes": notes,
    }
    save_json(evaluation, reports / "single_model_evaluation.json")

    lines = [
        "# Single model: class-ratio sweep and held-out result",
        "",
        f"Selected recipe **{best_row['recipe']}** on Split G validation macro-F1. "
        f"Temperature {temperature:.4f}, acceptance threshold {calibration['accept_threshold']:.2f}.",
        "",
        "## Ratio sweep (validation only)",
        "",
        sweep.to_markdown(index=False),
        "",
        "## Held-out Split G test (scored once)",
        "",
        pd.DataFrame([
            {"class": c, **{k: round(v, 4) for k, v in test_metrics["per_class"][c].items()}}
            for c in LABELS
        ]).to_markdown(index=False),
        "",
        f"Accuracy {test_metrics['accuracy']:.4f}; balanced accuracy {test_metrics['balanced_accuracy']:.4f}; "
        f"macro-F1 {test_metrics['macro_f1']:.4f}.",
        "",
        f"Abstention: coverage {test_selective['coverage']:.3f}, accuracy on accepted "
        f"{test_selective['accuracy_on_accepted']}.",
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in notes],
    ]
    (reports / "single_model_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"deployed {model_path.name}; test macro-F1 {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
