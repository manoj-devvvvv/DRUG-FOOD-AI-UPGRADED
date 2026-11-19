"""Temperature-scaling calibration and abstention metadata.

The calibration artifact is fitted on validation data by
`src/brick14_domain_calibration.py`. Serving applies the stored temperature to
raw model probabilities and uses the stored validation-selected acceptance
threshold. Missing artifact means raw probabilities and the config threshold,
flagged as uncalibrated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import load_json


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def temper(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    return softmax(np.log(p) / float(temperature))


def status(mode: str, models_dir: Path) -> dict | None:
    path = Path(models_dir) / f"calibration_{mode}.json"
    return load_json(path) if path.exists() else None


def apply(mode: str, models_dir: Path, proba: np.ndarray) -> tuple[np.ndarray, dict]:
    artifact = status(mode, models_dir)
    arr = np.asarray(proba, dtype=np.float64)
    flat = arr.ndim == 1
    if flat:
        arr = arr.reshape(1, -1)
    if artifact is None:
        calibrated = arr
        info = {
            "method": "none",
            "calibrated": False,
            "note": "No calibration artifact; probabilities are raw model outputs.",
        }
    else:
        calibrated = temper(arr, artifact["temperature"])
        info = {
            "method": "temperature_scaling",
            "calibrated": True,
            "temperature": float(artifact["temperature"]),
            "fitted_on": artifact.get("fitted_on"),
            "note": "Confidence is a calibrated model probability, not a guarantee of correctness.",
        }
    return (calibrated[0] if flat else calibrated), info


def acceptance_threshold(mode: str, models_dir: Path, fallback: float) -> tuple[float, dict]:
    artifact = status(mode, models_dir)
    if artifact is None:
        return float(fallback), {"source": "config fallback", "calibrated": False}
    return float(artifact["accept_threshold"]), {
        "source": artifact.get("threshold_policy", "validation-selected"),
        "calibrated": True,
        "target_error": artifact.get("threshold_target_error"),
        "validation_coverage": artifact.get("threshold_validation_coverage"),
    }
