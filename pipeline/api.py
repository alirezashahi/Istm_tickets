"""FastAPI prediction service for the ITSM ticket routing pipeline.

Usage
-----
    pip install fastapi uvicorn
    cd pipeline
    uvicorn api:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import Pipeline

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)


# ── Lifespan: load models once at startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading pipeline models...")
    try:
        app.state.pipeline = Pipeline.load()
        log.info("Pipeline ready.")
    except Exception as exc:
        log.error("Failed to load pipeline: %s", exc)
        app.state.pipeline = None
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="ITSM Ticket Routing API",
    description="Predicts Service → Category → Subcategory for Ivanti helpdesk tickets.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    incident_number: Optional[str] = None
    subject: str = ""
    symptom: str = ""
    sender: str = ""
    category_hint: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "incident_number": "INC0012345",
                    "subject": "Workflow bloccato",
                    "symptom": "Due workflow sembrano bloccati nella coda approvazioni",
                    "sender": "CARLO VANNUCCI",
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    incident_number: Optional[str]
    service: str
    category: str
    subcategory: str
    confidences: dict[str, float]
    is_flat: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health(request: Request):
    if request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    pipeline: Pipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        result = pipeline.predict(
            subject=body.subject,
            symptom=body.symptom,
            sender=body.sender,
            category_hint=body.category_hint,
        )
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        incident_number=body.incident_number,
        service=result.service,
        category=result.category,
        subcategory=result.subcategory,
        confidences=result.confidences,
        is_flat=result.is_flat,
    )
