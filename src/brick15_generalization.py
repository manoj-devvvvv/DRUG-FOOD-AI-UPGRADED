"""Generalization evaluation: pair-random, unseen-drug, unseen-food, double-cold.

Each partition trains the same fixed LightGBM configuration from scratch on its
own training rows. The final test partition is only scored after fitting.
Class 1 has very few examples; metrics that need positive support are reported
as null instead of being invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from brick17_single_model import cap_positions
from common import dataframe_to_markdown, load_config, load_json, save_json, setup_stdout
from features import DESC_NAMES, pair_matrix
from splits_extra import overlap_check, split_double_cold, split_unseen_compounds

LABELS = [0, 1, 2]


def load_moe(cfg: dict):
    interim = Path(cfg["paths"]["interim"])
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")
    return drug_moe, food_moe


def build_partition(df: pd.DataFrame, split_col: str, names: list[str], drug_moe, food_moe):
    dvals = drug_moe[DESC_NAMES].to_numpy()
    fvals = food_moe[DESC_NAMES].to_numpy()
    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}
    out = {}
    for split in ("train", "val", "test"):
        sub = df[df[split_col] == split].reset_index(drop=True)
        sub = sub[sub["drug_id"].isin(drow) & sub["compound_id"].isin(frow)].reset_index(drop=True)
        di = np.array([drow[x] for x in sub["drug_id"]], dtype=np.int64)
        fi = np.array([frow[x] for x in sub["compound_id"]], dtype=np.int64)
        X = np.empty((len(sub), len(names)), dtype=np.float32)
        for start, chunk in pair_matrix(dvals, fvals, di, fi, names=names, chunk_size=20000):
            X[start:start + len(chunk)] = chunk
        out[split] = {
            "X": X,
            "y": sub["label"].to_numpy(),
            "weak": sub["weak"].to_numpy(dtype=bool),
            "df": sub,
        }
    return out


def metrics(y, pred, proba) -> dict:
    present = sorted(set(int(v) for v in np.unique(y)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, pred, labels=LABELS)
    out = {
        "support": int(len(y)),
        "class_counts": {int(c): int((y == c).sum()) for c in LABELS},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "per_class": {
            int(c): {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i]),
            }
            for i, c in enumerate(LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "roc_auc_ovr_macro": None,
        "pr_auc_macro": None,
        "pr_auc_per_class": None,
        "note": None,
    }
    if len(present) >= 2:
        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=present)
            )
        except ValueError:
            out["roc_auc_ovr_macro"] = None
        per_class = {}
        for c in present:
            per_class[int(c)] = float(average_precision_score((y == c).astype(int), proba[:, c]))
        out["pr_auc_per_class"] = per_class
        out["pr_auc_macro"] = float(np.mean(list(per_class.values())))
    if 1 not in present:
        out["note"] = "Class 1 has no examples in this evaluation partition; class-1 metrics are not meaningful."
    elif int((y == 1).sum()) < 10:
        out["note"] = f"Class 1 has only {int((y == 1).sum())} examples; class-1 metrics are unreliable."
    return out


def fit_eval(train, val, test, params, weak_weight, cap_ratio=None, seed=42):
    positions = cap_positions(train["df"], cap_ratio, seed)
    Xtr, ytr, wtr = train["X"][positions], train["y"][positions], train["weak"][positions]
    base = LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbose=-1)
    sw = compute_sample_weight("balanced", ytr) * np.where(wtr, weak_weight, 1.0)
    base.fit(Xtr, ytr, sample_weight=sw)
    result = {}
    for name, part in (("validation", val), ("test", test)):
        proba = base.predict_proba(part["X"])
        pred = proba.argmax(axis=1)
        result[name] = metrics(part["y"], pred, proba)
    return result, base


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["S", "P"], default="S")
    ap.add_argument("--cap-ratio", type=float, default=None,
                    help="cap class 2 in each training partition at ratio * class 0")
    args = ap.parse_args()
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    reports = Path(cfg["paths"]["reports"])
    models = Path(cfg["paths"]["models"])

    splits = pd.read_parquet(processed / f"splits_{args.mode}.parquet")

    splits["split_F"] = split_unseen_compounds(splits, cfg)
    splits["split_D"], d_info = split_double_cold(splits, cfg)
    splits_d = splits[splits["split_D"].notna()].copy()

    split_specs = {
        "R": (splits, "split_R", "pair-random (same drugs can appear on both sides)"),
        "G": (splits, "split_G", "unseen drugs (whole-drug groups)"),
        "F": (splits, "split_F", "unseen food compounds (whole-compound groups)"),
        "D": (splits_d, "split_D", "double-cold (drug and compound groups both unseen)"),
    }

    sel = load_json(models / f"selected_features_{args.mode}.json")
    names = [f["name"] for f in sel["features"]]
    drug_moe, food_moe = load_moe(cfg)

    try:
        model_report = load_json(reports / f"brick8_models_{args.mode}.json")
        params = model_report["tuning"]["LightGBM"]["best_params"]
    except (FileNotFoundError, KeyError):
        params = {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05}
    weak_weight = float(cfg["models"].get("weak_sample_weight", 0.5))

    report = {
        "mode": args.mode,
        "model": "LightGBM (fixed tuned configuration, balanced class weights, weak weight "
                 f"{weak_weight}, class-2 cap {args.cap_ratio})" if args.cap_ratio else
                 "LightGBM (fixed tuned configuration, balanced class weights, weak weight "
                 f"{weak_weight})",
        "class_2_cap": args.cap_ratio,
        "features": names,
        "params": params,
        "partitions": {},
        "leakage_checks": {},
        "double_cold_construction": d_info,
        "limitations": [
            "Split R is an easier diagnostic because drugs repeat across partitions.",
            "Split F holds out compounds but drugs repeat.",
            "Split D drops pairs whose drug and compound partitions differ, so its size is limited.",
            "Class 1 support is tiny in Mode S; class-1 metrics are unreliable wherever support < 10.",
        ],
    }

    rows = []
    for tag, (frame, col, description) in split_specs.items():
        if frame[col].nunique() < 3:
            print(f"skip {tag}: not enough partitions")
            continue
        print(f"\n=== Split {tag}: {description} ===")
        data = build_partition(frame, col, names, drug_moe, food_moe)
        print({k: (len(v["y"]), {int(c): int((v["y"] == c).sum()) for c in LABELS}) for k, v in data.items()})
        result, _ = fit_eval(data["train"], data["val"], data["test"], params, weak_weight,
                             cap_ratio=args.cap_ratio, seed=int(cfg["seed"]))
        checks = {}
        if tag in ("G", "D"):
            checks["drug"] = overlap_check(frame, col, "drug_id")
        if tag in ("F", "D"):
            checks["compound"] = overlap_check(frame, col, "compound_id")
        report["partitions"][tag] = {
            "description": description,
            "counts": {
                s: {
                    "rows": int((frame[col] == s).sum()),
                    "class_counts": {
                        int(c): int(((frame[col] == s) & (frame["label"] == c)).sum())
                        for c in LABELS
                    },
                    "drugs": int(frame.loc[frame[col] == s, "drug_id"].nunique()),
                    "compounds": int(frame.loc[frame[col] == s, "compound_id"].nunique()),
                }
                for s in ("train", "val", "test")
            },
            "metrics": result,
        }
        report["leakage_checks"][tag] = checks
        for part in ("validation", "test"):
            m = result[part]
            rows.append({
                "split": tag, "partition": part, "n": m["support"],
                "accuracy": round(m["accuracy"], 4),
                "balanced_accuracy": round(m["balanced_accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "weighted_f1": round(m["weighted_f1"], 4),
                "mcc": round(m["mcc"], 4),
                "kappa": round(m["cohen_kappa"], 4),
                "class_1_support": m["class_counts"][1],
                "roc_auc_ovr_macro": None if m["roc_auc_ovr_macro"] is None else round(m["roc_auc_ovr_macro"], 4),
                "pr_auc_macro": None if m["pr_auc_macro"] is None else round(m["pr_auc_macro"], 4),
            })
        print("test", {k: result["test"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")})

    table = pd.DataFrame(rows)
    table.to_csv(reports / f"generalization_{args.mode}.csv", index=False)
    save_json(report, reports / f"generalization_{args.mode}.json")

    lines = [
        f"# Generalization evaluation (Mode {args.mode})",
        "",
        "Every partition is trained from scratch with the same fixed LightGBM configuration "
        f"({len(names)} selected features, balanced class weights, weak-pair weight {weak_weight}"
        + (f", class-2 cap {args.cap_ratio}:1 applied to training rows only" if args.cap_ratio else "")
        + "). Test rows are scored only after fitting.",
        "",
        "## Leakage checks (overlap must be zero)",
        "",
        pd.DataFrame([
            {"split": tag, **{f"{kind}_{pair}": n for kind, pairs in checks.items()
                              for pair, n in pairs.items()}}
            for tag, checks in report["leakage_checks"].items()
        ]).to_markdown(index=False),
        "",
        "## Results",
        "",
        table.to_markdown(index=False),
        "",
        "## Per-class metrics on the test partition",
        "",
    ]
    for tag, part in report["partitions"].items():
        m = part["metrics"]["test"]
        lines += [
            f"### Split {tag} — {part['description']}",
            "",
            f"Test rows {m['support']}; class counts {m['class_counts']}; "
            f"ROC-AUC OVR macro {m['roc_auc_ovr_macro']}; PR-AUC macro {m['pr_auc_macro']}.",
            "",
            pd.DataFrame(m["per_class"]).T.to_markdown(),
            "",
            pd.DataFrame(m["confusion_matrix"], index=["true 0", "true 1", "true 2"],
                         columns=["pred 0", "pred 1", "pred 2"]).to_markdown(),
            "",
        ]
        if m.get("note"):
            lines += [f"**Limitation:** {m['note']}", ""]
    lines += ["## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    (reports / f"generalization_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n", table.to_string(index=False))


if __name__ == "__main__":
    main()
