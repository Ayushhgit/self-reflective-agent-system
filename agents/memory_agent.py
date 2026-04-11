"""
Finalises the answer and persists the Q&A pair to the in-session conversation
memory so future queries can benefit from prior context.
"""

import logging
from datetime import datetime

from state import AgentState

logger = logging.getLogger(__name__)


def memory_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Memory

    Responsibilities
    ----------------
    - Promote state.draft_answer to state.final_answer.
    - Append the Q&A record to state.memory.
    - Log quality metadata (score, iterations used).
    """
    draft = state.get("draft_answer", "")
    query = state.get("query", "")
    score = state.get("score", 0.0)
    iterations = state.get("iterations", 0)
    memory = list(state.get("memory", []))

    final_answer = draft or "I was unable to generate a satisfactory answer."
    logger.info(
        "[Memory] Finalising answer. Score=%.2f, Iterations=%d",
        score,
        iterations,
    )

    # ── Persist to conversation memory ───────────────────────────────────────
    memory_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": final_answer,
        "score": round(score, 4),
        "iterations": iterations,
        "route": state.get("route", "unknown"),
    }
    memory.append(memory_record)

    logger.info("[Memory] Memory now contains %d records.", len(memory))

    return {**state, "final_answer": final_answer, "memory": memory}
