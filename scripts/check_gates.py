"""Check every plan gate against the artifacts on disk.

Usage: python scripts/check_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, M = ROOT / "reports", ROOT / "models"
checks: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def js(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


drugs = pd.read_parquet(ROOT / "data/interim/drugs_clean.parquet")
foods = pd.read_parquet(ROOT / "data/interim/foods_clean.parquet")
links = pd.read_parquet(ROOT / "data/interim/food_links.parquet")
pairs = pd.read_parquet(ROOT / "data/interim/pairs_labeled.parquet")

check("Brick 0 environment", (ROOT / "requirements.txt").exists(), "requirements.txt")
check("Brick 1 audit", (R / "brick1_audit.json").exists(), "audit json")
check("Brick 2 drugs parsed", drugs["smiles"].notna().all() and len(drugs) > 1000, f"{len(drugs)} drugs")
check("Brick 3 foods flagged", {"is_organic", "is_protected"} <= set(foods.columns),
      f"{len(foods)} compounds, {len(links):,} links")
protected = set(foods.loc[foods["is_protected"], "name"].str.lower())
check("Brick 3 protected survive", len(protected) >= 7, f"{len(protected)} present")
check("Brick 4 rules and lexicon", (R / "label_rules.json").exists() and (R / "label_lexicon.json").exists(), "")
check("Brick 4 audit csv", len(pd.read_csv(R / "sentence_audit_100.csv")) == 100, "100 sentences")
holdout = js("brick4_label_report.json")["holdout_pairs"]
hold_keys = {(str(a), int(b)) for a, b in holdout}
present = pairs[["drug_id", "compound_id"]].apply(lambda r: (str(r[0]), int(r[1])) in hold_keys, axis=1)
check("Brick 4 holdout absent", not present.any(), f"{int(present.sum())} leaked")

b5 = js("brick5_features.json")
check("Brick 5 verification", b5["verification"]["pass_library_tolerance_1e-3"] and b5["all_values_finite"],
      "library tolerance 1e-3, finite")
check("Brick 5 pair matrix", (ROOT / "data/interim/drug_moe.parquet").exists()
      and (ROOT / "data/interim/food_moe.parquet").exists(), "60 descriptors per molecule")

sp = pd.read_parquet(ROOT / "data/processed/splits_S.parquet")
sets = {k: set(sp.loc[sp["split_G"] == k, "drug_id"]) for k in ("train", "val", "test")}
overlap = len(sets["train"] & sets["val"]) + len(sets["train"] & sets["test"]) + len(sets["val"] & sets["test"])
check("Brick 6 Split G disjoint", overlap == 0, f"overlap {overlap}")
check("Brick 6 holdout absent", js("brick6_splits.json")["heldout_pairs_absent"], "all partitions")
check("Single served split", not (ROOT / "data/processed/splits_P.parquet").exists(), "Mode P retired")

sel = json.loads((M / "selected_features_S.json").read_text(encoding="utf-8"))
counts = sel["counts"]
mono = counts["after_variance"] >= counts["after_correlation"] >= counts["after_lasso"] >= 18
check("Brick 7 selection", sel["n_selected"] == 18 and mono,
      f"{counts}, stability {sel['stability']['stable_kept_in_final']}/18")
check("Brick 7 scaler", (M / "scaler_S.pkl").exists(), "")

b8 = js("brick8_models_S.json")
check("Brick 8 model", (M / "best_model_S.joblib").exists()
      and (M / "xgb_final_S.json").exists(), b8["best_model"])
check("Brick 8 reload", b8["xgb_reload_max_prob_diff"] < 1e-9,
      f"diff {b8['xgb_reload_max_prob_diff']:.1e}")
m = js("metrics_S.json")
check("Brick 9 evaluation", {"G", "R"} <= set(m["splits"]),
      f"G acc {m['splits']['G']['accuracy']:.3f}, R acc {m['splits']['R']['accuracy']:.3f}")
check("Brick 9 figures", (ROOT / "figures/cm_S_G.png").exists()
      and (ROOT / "figures/cm_S_R.png").exists(), "")

check("Brick 10 SHAP plots", len(list((ROOT / "figures").glob("shap_*_S_*.png"))) == 3, "per class")
check("Brick 17 single model", (R / "single_model_evaluation.json").exists()
      and (R / "class_balance_sweep.csv").exists(), "cap sweep + held-out result")
cal = js("calibration_S.json")
check("Brick 17 calibration", cal.get("class_balance", {}).get("recipe") == "cap_2",
      f"recipe {cal.get('class_balance', {}).get('recipe')}")
csv_dir = ROOT / "data/processed/csv"
check("Stage CSV exports", (csv_dir / "manifest.json").exists()
      and all((csv_dir / f).exists() for f in
              ("drug_cleaned.csv", "food_cleaned.csv", "pairs.csv",
               "features_generation.csv", "feature_extraction.csv")), "5 stage files")
cases = pd.read_csv(R / "case_study_table.csv")
check("Brick 11 cases", len(cases) == 20, f"{len(cases)} rows")
check("Brick 11 disclaimer", (R / "brick11_case_study.json").exists()
      and "physician or pharmacist" in (R / "case_study_table.md").read_text(encoding="utf-8"), "")

b12 = js("brick12_enhancements.json")
cov = b12["e2_calibration_conformal"]["coverage"]
check("Brick 12 enhancements", {"drug_only", "food_only", "shuffled_food", "all_3780_features"} <= set(b12["ablations"]),
      f"conformal coverage {cov:.3f}")
check("Report", (R / "report.md").exists(), "reports/report.md")
check("Notebooks", len(list((ROOT / "notebooks").glob("*.ipynb"))) >= 12, "")
check("Regeneration script", (ROOT / "scripts/run_all.sh").exists(), "scripts/run_all.sh")

width = max(len(n) for n, _, _ in checks)
failed = 0
for name, ok, detail in checks:
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
print(f"\n{len(checks) - failed}/{len(checks)} gates passed")
sys.exit(1 if failed else 0)
