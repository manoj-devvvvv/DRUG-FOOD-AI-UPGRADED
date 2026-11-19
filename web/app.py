"""FastAPI service for the drug-food interaction recommender.

Run from dfi/:  uvicorn web.app:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(ROOT / "src"))

from common import norm_name  # noqa: E402
from env import load_project_env  # noqa: E402
from recommend import DFIRecommender  # noqa: E402
from rules import RULE_VERSION, rules_document  # noqa: E402

load_project_env()

log = logging.getLogger("dfi.web")

_engine: DFIRecommender | None = None
_names: dict[str, list[str]] = {}
_lock = threading.Lock()


def engine() -> DFIRecommender:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = DFIRecommender()
    return _engine


def names(kind: str) -> list[str]:
    if kind not in _names:
        e = engine()
        if kind == "drug":
            values = {n for n in e.drug_by_name if n}
            display = {norm_name(v): v for v in e.drug_name.values()}
            _names["drug"] = sorted((display.get(k, k) for k in values), key=str.lower)
        else:
            compounds = {norm_name(v): v for v in e.foods["name"]}
            foods = {norm_name(v): v for v in e.links["food_name"].unique()}
            merged = {**compounds, **foods}
            _names["food"] = sorted(merged.values(), key=str.lower)
    return _names[kind]


def warm_engine() -> None:
    try:
        engine().warm()
        log.info("engine ready (SHAP explainer loaded)")
    except Exception:
        log.exception("engine warm-up failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    engine()
    threading.Thread(target=warm_engine, name="dfi-warmup", daemon=True).start()
    yield
    _engine = None
    _names.clear()


app = FastAPI(title="Drug-Food Interaction Predictor", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


class PredictRequest(BaseModel):
    drug: str = Field(min_length=1, max_length=300, description="Drug name or SMILES")
    food: str = Field(min_length=1, max_length=300, description="Food or compound name, or SMILES")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "static" / "landing.html")


@app.get("/predict")
def predict_page() -> FileResponse:
    return FileResponse(WEB / "static" / "index.html")


@app.get("/about")
def about_page() -> FileResponse:
    return FileResponse(WEB / "static" / "about.html")


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    safe = Path(name).name
    if safe != name or not safe.endswith(".png"):
        raise HTTPException(status_code=404, detail="Figure not found.")
    path = ROOT / "figures" / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path)


def resolution_status(e: DFIRecommender) -> dict:
    resolver = e.resolver
    return {
        "gemini_configured": resolver.gemini_configured,
        "gemini_model": resolver.model,
        "online_resolution": resolver.online,
        "cached_entries": len(resolver.cache),
        "order": ["local database", "Gemini", "PubChem", "user SMILES"],
    }


@app.get("/api/health")
def health() -> dict:
    e = engine()
    return {
        "status": "ready",
        "drugs": int(len(e.drugs)),
        "compounds": int(len(e.foods)),
        "confidence_threshold": float(e.cfg["models"]["confidence_threshold"]),
        "accept_threshold": float(e.accept_threshold),
        "calibrated": bool(e.threshold_info.get("calibrated")),
        "model": e.model_info,
        "resolution": resolution_status(e),
        "rules_version": RULE_VERSION,
    }


@app.get("/api/meta")
def meta() -> dict:
    e = engine()
    return {
        "model": e.model_info,
        "features": e.feature_names,
        "resolution": resolution_status(e),
        "rules_version": RULE_VERSION,
        "rule_layer": rules_document(),
        "applicability_domain": {
            "validated": e.domain.validated,
            "thresholds": e.domain.thresholds,
            "reference_counts": e.domain.reference_counts,
        },
        "abstention": {
            "accept_threshold": float(e.accept_threshold),
            **e.threshold_info,
        },
    }


@app.get("/api/examples")
def examples() -> dict:
    return {"pairs": engine().cfg["web"]["examples"]}


@app.get("/api/suggest")
def suggest(
    kind: Literal["drug", "food"] = "drug",
    q: str = Query("", max_length=100),
    limit: int = Query(8, ge=1, le=20),
) -> dict:
    key = norm_name(q)
    pool = names(kind)
    if not key:
        picks = pool[:limit]
    else:
        starts = [n for n in pool if norm_name(n).startswith(key)]
        contains = [n for n in pool if key in norm_name(n) and n not in starts]
        picks = (starts + contains)[:limit]
    return {"suggestions": picks}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    drug, food = req.drug.strip(), req.food.strip()
    if not drug or not food:
        raise HTTPException(status_code=422, detail="Both fields are required.")
    t0 = time.perf_counter()
    try:
        result = dict(engine().recommend(drug, food))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are missing; run scripts/run_all.sh.") from exc
    result["inputs"] = {"drug": drug, "food": food}
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "predict source=%s status=%s abstained=%s drug=%r food=%r latency_ms=%.1f",
        result.get("prediction_source"),
        result.get("status"),
        result.get("abstained"),
        drug,
        food,
        result["latency_ms"],
    )
    e = engine()
    cls = result.get("predicted_class")
    if cls is not None:
        cls = int(cls)
        result["label"] = e.labels[cls]
        result["severity"] = e.severity[cls]
    probs = result.get("probabilities")
    if probs:

        def getp(c: int) -> float:
            return float(probs[str(c)] if str(c) in probs else probs[c])

        result["probability_rows"] = [
            {"class": c, "label": e.labels[c], "probability": getp(c)} for c in (0, 1, 2)
        ]
        result["probabilities"] = None
    if "confidence" in result and result["confidence"] is not None:
        result["confidence"] = float(result["confidence"])
    if "domain_similarity" in result and result["domain_similarity"] is not None:
        result["domain_similarity"] = float(result["domain_similarity"])
    return result
