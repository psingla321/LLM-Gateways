"""
LLM Gateway — FastAPI backend
==============================
Start:  uvicorn api:app --reload --port 8000
Open:   http://localhost:8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from gateway import LLMGateway

# ── Singleton gateway ─────────────────────────────────────────────────────────
_gw: LLMGateway | None = None

def get_gw() -> LLMGateway:
    global _gw
    if _gw is None:
        env = os.getenv("MOCK_MODE", "").lower()
        mock = True if env == "true" else (False if env == "false" else None)
        _gw = LLMGateway(mock_mode=mock)
    return _gw


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="LLM Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class ClaimRequest(BaseModel):
    claim_text:     str  = Field(..., min_length=10, max_length=4000)
    team:           str  = "claims-auto"
    user_id:        str  = "demo_user"
    priority:       str  = "normal"
    force_fallback: bool = False


# ── API routes ────────────────────────────────────────────────────────────────
@app.post("/api/claim")
async def process_claim(req: ClaimRequest):
    try:
        result = get_gw().process_claim(
            claim_text           = req.claim_text,
            team                 = req.team,
            user_id              = req.user_id,
            priority             = req.priority,
            _demo_force_fallback = req.force_fallback,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stats")
async def stats():
    return get_gw().logger.summary_stats()


@app.get("/api/teams")
async def teams():
    return get_gw().logger.team_budgets()


@app.get("/api/models")
async def models():
    return get_gw().logger.model_stats()


@app.get("/api/recent")
async def recent():
    return get_gw().logger.recent_requests(limit=25)


@app.delete("/api/recent")
async def clear_recent():
    cleared = get_gw().logger.clear_requests()
    return {"cleared": cleared}


@app.get("/api/mode")
async def mode():
    return {"mock_mode": get_gw().mock_mode}


# ── Serve single-page frontend ────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
