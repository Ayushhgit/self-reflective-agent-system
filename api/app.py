"""
Exposes a REST API for the AREA multi-agent system.

Endpoints
---------
POST /query          Run the full AREA pipeline for a query.
GET  /health         Health check.
GET  /history        Return conversation memory.
POST /memory/clear   Clear conversation memory.
POST /documents      Add documents to the RAG knowledge base.

Run with:
    uvicorn api.app:app --reload --port 8000
"""

import logging
import sys
import os

# Add parent to path so imports resolve correctly when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from graph.workflow import run_query
from memory.conversation_memory import (
    add_record,
    get_history,
    clear as clear_memory,
)
from rag.vector_store import add_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AREA – Autonomous Research & Execution Agent",
    description=(
        "A multi-agent AI system using LangGraph that plans, retrieves, "
        "executes tools, synthesises, evaluates, and reflects to produce "
        "high-quality answers."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared in-session memory ────────────────────────────────────────────────
_session_memory: List[dict] = []


# ─── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's query or task")
    max_iters: Optional[int] = Field(3, ge=1, le=5, description="Max reflection iterations")


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    score: float
    iterations: int
    route: str
    plan: List[str]
    critique: str
    tool_output: str


class DocumentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of document texts to add")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "AREA API v1.0"}


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def query_endpoint(req: QueryRequest):
    """
    Run the AREA multi-agent pipeline for the given query.

    Returns the final answer along with metadata: score, iterations, route, plan.
    """
    global _session_memory
    logger.info("POST /query: %s", req.query[:80])

    try:
        result = run_query(
            query=req.query,
            memory=_session_memory,
            max_iters=req.max_iters,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist to conversation memory
    add_record(
        query=req.query,
        answer=result.get("final_answer", ""),
        score=result.get("score", 0.0),
        route=result.get("route", "unknown"),
    )
    _session_memory = result.get("memory", _session_memory)

    return QueryResponse(
        query=req.query,
        final_answer=result.get("final_answer", ""),
        score=round(result.get("score", 0.0), 4),
        iterations=result.get("iterations", 0),
        route=result.get("route", ""),
        plan=result.get("plan", []),
        critique=result.get("critique", ""),
        tool_output=result.get("tool_output", ""),
    )


@app.get("/history", tags=["Memory"])
def history_endpoint():
    """Return the current conversation history."""
    return {"history": get_history()}


@app.post("/memory/clear", tags=["Memory"])
def clear_memory_endpoint():
    """Clear all conversation history."""
    global _session_memory
    clear_memory()
    _session_memory = []
    return {"status": "cleared"}


@app.post("/documents", tags=["RAG"])
def add_documents_endpoint(req: DocumentRequest):
    """Add documents to the RAG vector store."""
    try:
        add_documents(req.texts)
        return {"status": "added", "count": len(req.texts)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api.app:app", host=API_HOST, port=API_PORT, reload=True)
