"""Per-prediction TreeSHAP explanations for the LightGBM serving model.

SHAP values describe how features move the model score for each class. All
features are returned for every reaction class so the explanation is complete,
not a top-k summary. They are not evidence of a biological mechanism.
"""
from __future__ import annotations

import numpy as np

CAVEAT = (
    "TreeSHAP attribution shows how features change the model score for this "
    "prediction; it is not evidence of a biological mechanism."
)


class LocalExplainer:
    def __init__(self, model):
        # The serving artifact is a BalancedClassifier wrapping a LightGBM model.
        self.booster = getattr(model, "base", model)
        self.explainer = None
        self.error = None
        try:
            import shap

            self.explainer = shap.TreeExplainer(self.booster)
        except Exception as exc:  # pragma: no cover - depends on artifact backend
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.explainer is not None

    def _class_values(self, X: np.ndarray) -> list[np.ndarray]:
        """Return one SHAP vector per class for the single-row input."""
        values = self.explainer.shap_values(X)
        if isinstance(values, list):
            return [np.asarray(v)[0] for v in values]
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                return [arr[0, :, c] for c in range(arr.shape[2])]
            return [arr[c][0] for c in range(arr.shape[0])]
        return [arr[0]]

    @staticmethod
    def _contributions(feature_names, feature_values, vector) -> list[dict]:
        rows = [
            {
                "feature": name,
                "value": float(val),
                "shap_value": float(contrib),
                "direction": "increases" if contrib >= 0 else "decreases",
            }
            for name, val, contrib in zip(feature_names, feature_values, np.asarray(vector))
        ]
        rows.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return rows

    def explain(
        self,
        X: np.ndarray,
        feature_names: list[str],
        predicted_class: int,
        top_k: int = 6,
        labels: dict[int, str] | None = None,
    ) -> dict | None:
        if not self.available:
            return None
        try:
            vectors = self._class_values(X)
            feature_values = np.asarray(X[0], dtype=float)
            by_class = {}
            for cls, vector in enumerate(vectors):
                rows = self._contributions(feature_names, feature_values, vector)
                by_class[int(cls)] = {
                    "label": (labels or {}).get(int(cls), f"class {cls}"),
                    "feature_count": len(rows),
                    "contributions": rows,
                }
            predicted = int(predicted_class)
            if predicted not in by_class:
                return None
            ordered = by_class[predicted]["contributions"]
            return {
                "method": "tree_shap",
                "predicted_class": predicted,
                "feature_count": len(ordered),
                "features": ordered,
                "top_features": ordered[:top_k],
                "by_class": by_class,
                "caveat": CAVEAT,
            }
        except Exception:
            return None
