"""Extra generalization partitions: unseen food compound (F) and double-cold (D).

Split F groups whole compounds, so a test compound never appears in training.
Split D assigns drugs and compounds to partitions independently and keeps only
pairs whose drug and compound land in the same partition; the others are
dropped. That makes both sides unseen in test but costs rows, which is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def assign_groups(sizes: pd.Series, cfg: dict, seed: int) -> dict:
    target = np.array(
        [cfg["splits"]["train_frac"], cfg["splits"]["val_frac"], cfg["splits"]["test_frac"]]
    ) * float(sizes.sum())
    target = np.maximum(target, 1.0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    order = sorted(order, key=lambda g: -sizes[g])
    current = np.zeros(3)
    assignment: dict = {}
    for group in order:
        room = (target - current) / target
        if np.all(current > 0):
            part = int(np.argmax(room))
        else:
            empty = [i for i in range(3) if current[i] == 0]
            part = empty[0] if len(empty) < 3 else int(np.argmax(room))
        assignment[group] = SPLITS[part]
        current[part] += sizes[group]
    return assignment


def split_unseen_compounds(df: pd.DataFrame, cfg: dict) -> pd.Series:
    sizes = df.groupby("compound_id").size()
    assignment = assign_groups(sizes, cfg, int(cfg["seed"]) + 101)
    return df["compound_id"].map(assignment)


def split_double_cold(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, dict]:
    drug_sizes = df.groupby("drug_id").size()
    comp_sizes = df.groupby("compound_id").size()
    drug_map = assign_groups(drug_sizes, cfg, int(cfg["seed"]) + 202)
    comp_map = assign_groups(comp_sizes, cfg, int(cfg["seed"]) + 303)
    drug_split = df["drug_id"].map(drug_map)
    comp_split = df["compound_id"].map(comp_map)
    out = drug_split.where(drug_split == comp_split)
    info = {
        "kept": int(out.notna().sum()),
        "dropped": int(out.isna().sum()),
        "drugs": {s: int(sum(v == s for v in drug_map.values())) for s in SPLITS},
        "compounds": {s: int(sum(v == s for v in comp_map.values())) for s in SPLITS},
    }
    return out, info


def overlap_check(df: pd.DataFrame, split_col: str, key: str) -> dict:
    parts = {s: set(df.loc[df[split_col] == s, key]) for s in SPLITS}
    out = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out[f"{a}_{b}"] = len(parts[a] & parts[b])
    return out
