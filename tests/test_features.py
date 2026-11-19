"""Unit tests for the pair-feature contract (Brick 5 gate)."""
import numpy as np
import pandas as pd

from features import ALL_PAIR_NAMES, DESC_NAMES, compute_moe_table, pair_matrix


def test_pair_names():
    assert len(DESC_NAMES) == 60
    assert len(ALL_PAIR_NAMES) == 3780
    assert len(set(ALL_PAIR_NAMES)) == 3780


def test_pair_arithmetic():
    drug = (np.arange(60, dtype=np.float64) + 1.0)[None, :]
    food = (np.arange(60, dtype=np.float64) + 1.0)[None, :] * 2.0
    names = ["EstateVSA0*EstateVSA1", "MTPSA+MTPSA", "abs(MTPSA-MTPSA)", "slogPVSA0/slogPVSA0"]
    (_, chunk), = list(pair_matrix(drug, food, np.array([0]), np.array([0]), names=names, dtype=np.float64))
    i0, i1 = DESC_NAMES.index("EstateVSA0"), DESC_NAMES.index("EstateVSA1")
    im = DESC_NAMES.index("MTPSA")
    expected = [drug[0, i0] * food[0, i1], drug[0, im] + food[0, im],
                abs(drug[0, im] - food[0, im]), drug[0, im] / food[0, im]]
    assert np.allclose(chunk[0], expected, atol=1e-12)


def test_smoke_3780_finite():
    table, failures = compute_moe_table(["drug", "food"], ["CCO", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"])
    assert not failures and len(table) == 2
    values = table[DESC_NAMES].to_numpy()
    x = np.vstack([chunk for _, chunk in pair_matrix(
        values, values, np.array([0]), np.array([1]), names=None, dtype=np.float64)])
    assert x.shape == (1, 3780)
    assert np.isfinite(x).all()
