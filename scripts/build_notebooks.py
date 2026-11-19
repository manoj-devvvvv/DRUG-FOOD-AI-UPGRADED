"""Generate one executed Jupyter notebook per brick from the saved artifacts.

Usage: python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

VIEWER = """\
import json
from pathlib import Path
from IPython.display import Image, display, Markdown
import pandas as pd

ROOT = Path.cwd().parent
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

def md(name):
    p = REPORTS / name
    display(Markdown(p.read_text(encoding="utf-8"))) if p.exists() else print("missing", name)

def js(name, keys=None, indent=2):
    p = REPORTS / name
    if not p.exists():
        print("missing", name); return None
    d = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({k: d[k] for k in keys} if keys else d, indent=indent)[:6000])
    return d
"""

REPRO = {
    "01_data_audit.ipynb": "PYTHONPATH=src python src/brick1_audit.py",
    "02_clean_drugs.ipynb": "PYTHONPATH=src python src/brick2_clean_drugs.py",
    "03_clean_foods.ipynb": "PYTHONPATH=src python src/brick3_clean_foods.py",
    "04_labels.ipynb": "PYTHONPATH=src python src/brick4_labels.py",
    "05_features.ipynb": "PYTHONPATH=src python src/brick5_features.py",
    "06_splits.ipynb": "PYTHONPATH=src python src/brick6_splits.py",
    "07_selection_S.ipynb": "PYTHONPATH=src python src/brick7_selection.py --mode S",
    "08_models_S.ipynb": "PYTHONPATH=src python src/brick8_models.py --mode S",
    "09_evaluation.ipynb": "PYTHONPATH=src python src/brick9_evaluate.py --mode S",
    "10_shap.ipynb": "PYTHONPATH=src python src/brick10_shap.py --mode S",
    "11_recommender.ipynb": "PYTHONPATH=src python src/brick11_recommender.py",
    "12_enhancements.ipynb": "PYTHONPATH=src python src/brick12_enhancements.py",
    "13_final_report.ipynb": "python scripts/make_report.py",
}

SPECS = {
    "00_environment.ipynb": ("Brick 0 - environment and layout", "brick0_environment.md", """
import sklearn, xgboost, lightgbm, shap, rdkit
from rdkit import rdBase
print("python ok; scikit-learn", sklearn.__version__, "| xgboost", xgboost.__version__,
      "| lightgbm", lightgbm.__version__, "| shap", shap.__version__, "| rdkit", rdBase.rdkitVersion)
print(open(ROOT / "requirements.txt").read()[:1200])
"""),
    "01_data_audit.ipynb": ("Brick 1 - load and audit the raw tables", "brick1_audit.md", "js('brick1_audit.json')"),
    "02_clean_drugs.ipynb": ("Brick 2 - clean the drugs", "brick2_funnel.md", "js('brick2_funnel.json')"),
    "03_clean_foods.ipynb": ("Brick 3 - clean the food compounds", "brick3_funnel.md", "js('brick3_funnel.json')"),
    "04_labels.ipynb": ("Brick 4 - label the ground truth", "brick4_labels.md",
                        "md('brick4_labels.md')\nprint(pd.read_csv(REPORTS / 'sentence_audit_100.csv').head(10).to_string())"),
    "05_features.ipynb": ("Brick 5 - MOE-type pair features", "brick5_features.md", "js('brick5_features.json')"),
    "06_splits.ipynb": ("Brick 6 - Split R and Split G", "brick6_splits.md", "js('brick6_splits.json')"),
    "07_selection_S.ipynb": ("Brick 7 - feature selection", "brick7_selection_S.md", "js('brick7_selection_log_S.json')"),
    "08_models_S.ipynb": ("Brick 8 - models", "brick8_models_S.md",
                          "print(pd.read_csv(REPORTS / 'cv_results_S.csv').to_string(index=False))"),
    "09_evaluation.ipynb": ("Brick 9 - evaluation", "metrics_S.md",
                            "md('metrics_S.md')\nfor f in sorted(FIGURES.glob('cm_*.png')): display(Image(filename=str(f)))"),
    "10_shap.ipynb": ("Brick 10 - SHAP explanations", "brick10_shap_S.md",
                      "md('brick10_shap_S.md')\nfor f in sorted(FIGURES.glob('shap_*.png')): display(Image(filename=str(f)))"),
    "11_recommender.ipynb": ("Brick 11 - recommender and case studies", "case_study_table.md",
                             "print(pd.read_csv(REPORTS / 'case_study_table.csv').to_string(index=False))"),
    "12_enhancements.ipynb": ("Brick 12 - ablations, E1 features, E2 calibration", "brick12_enhancements.md",
                              "md('brick12_enhancements.md')\njs('brick12_enhancements.json', ['e2_calibration_conformal'])"),
    "13_final_report.ipynb": ("Final report", "report.md", "md('report.md')"),
}

MD_GATE = {
    "00": "Importing rdkit, PyBioMed, xgboost, lightgbm and shap works; the smoke test builds a 3,780-column pair vector.",
    "01": "Column names are read by name, not position; the food header misalignment is documented.",
    "02": "A funnel shows every removal step and every kept SMILES re-parses.",
    "03": "Protected compounds survive and inorganic entries are flagged.",
    "04": "Rules and lexicon are saved; the 100-sentence audit is recorded; the nine case pairs are held out.",
    "05": "The pair builder matches PyBioMed on 500 random pairs; no NaN or infinite values remain.",
    "06": "Both splits are saved, drug sets are disjoint in Split G, and holdout pairs are absent everywhere.",
    "07": "Counts after every step, 18 selected features, stability overlap across three subsamples.",
    "08": "Baseline table filled, model chosen on validation, saved model reloads with identical scores.",
    "09": "Per-class metrics and confusion matrices for Split G and Split R, side by side.",
    "10": "Per-class SHAP summary plots and the 18-feature meaning table.",
    "11": "Case study table with passes, failures and abstentions, plus the safety disclaimer.",
    "12": "Ablations, E1 variants and the E2 conformal coverage/abstention numbers.",
    "13": "Every number in the report traces to a JSON or CSV in reports/.",
}


def build(name: str, title: str, report: str, code: str) -> None:
    tag = name[:2]
    cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            f"**Gate:** {MD_GATE.get(tag, '')}\n\n"
            f"The narrative report for this brick is `reports/{report}`. "
            "The cells below load the saved artifacts; they do not retrain anything."
        ),
        nbf.v4.new_code_cell(VIEWER),
        nbf.v4.new_code_cell(code),
        nbf.v4.new_code_cell(
            "# Reproduce this brick from scratch (long running):\n"
            f"# !cd .. && PYTHONPATH=src python src/{'brick' + str(int(tag)) if tag.isdigit() else 'brick7_selection'}_*.py"
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbf.write(nb, str(NB / name))


if __name__ == "__main__":
    for name, (title, report, code) in SPECS.items():
        build(name, title, report, code)
        print("wrote", name)
