"""Gemma feedback microservice (FastAPI).

Decouples the heavy LLM from the Streamlit UI. The app POSTs the extracted JD +
resume entities here; this service runs Gemma and returns the feedback email.

Key design points (the teaching bits):
  * The model is loaded ONCE at startup via `lifespan`, never per request.
  * Uvicorn does not accept traffic until that startup finishes — so a slow
    (~40s) model load naturally maps to a Kubernetes *startupProbe*. A liveness
    probe alone would kill the pod mid-load (the classic CrashLoopBackOff).
  * `/health` is a cheap readiness/liveness endpoint; `/feedback` is the work.
  * `/feedback` runs the same `quality.extraction_problem` gate as every other
    entry point. This is a deployed HTTP endpoint, not an internal helper: it
    is published on port 8000 and is reachable by anything that can route to
    it, so "the caller already gated" is an assumption about clients, not a
    property of this service. Ungated, it answered `{"jd_entities": {},
    "resume_entities": {}}` with HTTP 200 and a finished rejection email —
    the exact input the rest of the pipeline refuses.

Run locally:
    .venv/bin/uvicorn gemma_api:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from genfeedback import config
from genfeedback.feedback.generate import generate_feedback_from_entities
from genfeedback.feedback.model_io import load_base_gemma
from genfeedback.quality import extraction_problem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Holds the loaded model between requests; populated during startup.
_state = {}


@asynccontextmanager
async def lifespan(_app):
    logger.info("Loading model %s ...", config.GEMMA_BASE_MODEL)
    _state["bundle"] = load_base_gemma(config.GEMMA_BASE_MODEL)
    logger.info("Model loaded; service ready.")
    yield
    _state.clear()


app = FastAPI(title="GenFeedback Gemma API", version="1.0", lifespan=lifespan)


class FeedbackRequest(BaseModel):
    jd_entities: dict[str, list[str]]
    resume_entities: dict[str, list[str]]
    role: str | None = None
    company: str | None = None
    recruiter: str | None = None


class FeedbackResponse(BaseModel):
    feedback: str
    model: str


def _require_bundle():
    """Return the loaded model, or refuse with 503 when there is none.

    `lifespan` normally puts the bundle in place before uvicorn accepts any
    traffic, so this covers every other way this app object gets served: an
    ASGI mount or test client that never runs the lifespan, a startup that
    failed, and the shutdown window after `_state` is cleared. Without it the
    first request died on a bare `KeyError`, which reaches the caller as a 500
    — telling them the request was at fault when the truth is "not ready, try
    again", the one thing a 503 says and a 500 does not.
    """
    bundle = _state.get("bundle")
    if bundle is None:
        logger.warning("Request received before the model was loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded; the service is not ready yet.",
        )
    return bundle


@app.get("/health")
def health():
    """Readiness/liveness probe target: 200 only once the model is loaded.

    Reporting "ok" from a process holding no model is how an orchestrator is
    told to send traffic to a pod that can only fail, and how `client.py`'s
    `api_healthy()` comes to promise the app a working service.
    """
    _require_bundle()
    return {"status": "ok", "model": config.GEMMA_BASE_MODEL}


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Compare both entity sets with Gemma and return a feedback email.

    Refuses instead of generating when `extraction_problem` reports that the
    entity sets are too thin or too garbled to reason about. Gemma will write a
    fluent, confident rejection email from an empty extraction — that email
    describes the extraction failure, not the candidate, and nothing about the
    HTTP 200 it used to arrive in says so.

    422 rather than 400: the request is well-formed and matches the schema, so
    nothing about it is *bad* — it is understood and cannot be acted on, which
    is precisely RFC 9110's unprocessable content. Nor 503: retrying identical
    entities will never succeed, so it must not look like a transient outage.
    The detail is an object, which is also what distinguishes this refusal from
    FastAPI's own 422 for schema violations (whose detail is a list).

    This endpoint receives ENTITY DICTS ONLY, so the gate's text-based signals
    (empty source, mixed-script/non-Latin document) cannot run here — see the
    module note in quality.py. The caller that holds the text is the one that
    can catch those, and app.py does; this gate is the backstop for everything
    that reaches the port some other way.
    """
    bundle = _require_bundle()

    problem = extraction_problem(req.jd_entities, req.resume_entities)
    if problem:
        raise HTTPException(
            status_code=422,
            detail={"error": "extraction_refused", "reason": problem},
        )

    text = generate_feedback_from_entities(
        req.jd_entities, req.resume_entities, bundle,
        company=req.company, recruiter=req.recruiter, role=req.role,
    )
    return FeedbackResponse(feedback=text, model=config.GEMMA_BASE_MODEL)
