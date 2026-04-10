"""
AREA – Conversation Memory
─────────────────────────────────────────────────────────────────────────────
A simple in-process conversation store that:
  - Accumulates Q&A records across multiple invocations.
  - Provides a formatted history string for injecting into prompts.
  - Can be cleared between sessions.

For multi-process or persistent deployments, swap this out with a
Redis-backed or SQLite-backed implementation.
"""

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_history: List[Dict[str, Any]] = []


def add_record(query: str, answer: str, score: float, route: str):
    """Append one Q&A record to the in-memory history."""
    _history.append(
        {"query": query, "answer": answer, "score": round(score, 4), "route": route}
    )
    logger.debug("[Memory] Record added. Total records: %d", len(_history))


def get_history() -> List[Dict[str, Any]]:
    """Return a copy of the full conversation history."""
    return list(_history)


def format_history_for_prompt(max_records: int = 5) -> str:
    """
    Return the last N Q&A pairs as a formatted string suitable for
    inclusion in an LLM prompt.
    """
    recent = _history[-max_records:]
    if not recent:
        return "No previous conversation."
    lines = []
    for i, r in enumerate(recent, 1):
        lines.append(f"Q{i}: {r['query']}\nA{i}: {r['answer'][:300]}...")
    return "\n\n".join(lines)


def clear():
    """Clear all conversation history (e.g., at session start)."""
    _history.clear()
    logger.info("[Memory] Conversation history cleared.")


def export_json() -> str:
    """Export history as a JSON string (for persistence)."""
    return json.dumps(_history, indent=2)
