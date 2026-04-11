"""
Critically evaluates the draft answer across three dimensions:

    1. Correctness   – Is the information accurate?
    2. Completeness  – Does it fully address the query?
    3. Hallucination – Are there unsupported claims?

Outputs:
    - state.score    : float in [0, 1]
    - state.critique : Actionable feedback for the Reflector.
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an extremely strict AI answer evaluator. Your job is
to find EVERY flaw, gap, and weakness in the answer. Be harsh and demanding.

Score each dimension 0–10 using this strict rubric:

1. Correctness (0–10)
   - 9–10: Zero factual errors, all claims verifiable
   - 7–8 : Minor imprecision but no wrong facts
   - 5–6 : One or two factual errors present
   - 0–4 : Multiple wrong facts or misleading statements

2. Completeness (0–10)
   - 9–10: Every sub-question answered, examples given, edge cases covered
   - 7–8 : Most parts answered but missing examples or depth
   - 5–6 : Significant parts of the query unanswered
   - 0–4 : Superficial, skips major aspects of the query

3. Hallucination (0–10) — 10 = none, 0 = many
   - 9–10: Every claim is grounded in known facts or provided context
   - 7–8 : One unsupported claim
   - 5–6 : Multiple unsupported or invented details
   - 0–4 : Fabricated references, incorrect APIs, wrong syntax

Scoring rules (BE STRICT):
- A score of 9+ requires perfection in that dimension. Reserve it for truly
  exceptional answers.
- If code is present: deduct points unless it is complete, correct, and
  runnable without modification.
- If the answer is generic or could apply to any topic: deduct completeness.
- overall_score = (correctness + completeness + hallucination) / 30.0
  (do NOT inflate this — compute it exactly from your three scores)
- critique MUST list at least 2 specific, actionable improvements even for
  good answers. Never write "Answer meets quality standards."

Respond in this EXACT JSON format (no extra text, no markdown):
{
  "correctness": <int 0-10>,
  "completeness": <int 0-10>,
  "hallucination": <int 0-10>,
  "overall_score": <float 0.0-1.0>,
  "critique": "<at least 2 specific improvements separated by semicolons>"
}
"""


def evaluator_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Evaluator

    Responsibilities
    ----------------
    - Evaluate state.draft_answer against state.query.
    - Produce state.score and state.critique.
    """
    query = state["query"]
    draft = state.get("draft_answer", "")
    retrieved = state.get("retrieved_docs", [])

    logger.info("[Evaluator] Evaluating draft answer (%d chars).", len(draft))

    if not draft:
        return {**state, "score": 0.0, "critique": "No draft answer was generated."}

    context_hint = ""
    if retrieved:
        context_hint = (
            "\n\nAvailable source context (first 600 chars):\n"
            + retrieved[0][:600]
        )

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Answer to evaluate:\n{draft}"
                f"{context_hint}"
            )
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    score, critique = _parse_evaluation(raw)

    logger.info("[Evaluator] Score=%.2f | Critique: %s", score, critique[:80])

    return {**state, "score": score, "critique": critique}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_evaluation(raw: str):
    """
    Extract score and critique from the LLM response.
    Returns (score: float, critique: str).
    """
    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
        # If model computed overall_score correctly, use it directly
        if "overall_score" in data:
            score = float(data["overall_score"])
        else:
            # Recompute from individual scores to prevent inflation
            c = float(data.get("correctness", 7))
            comp = float(data.get("completeness", 7))
            h = float(data.get("hallucination", 7))
            score = (c + comp + h) / 30.0
        score = max(0.0, min(1.0, score))
        critique = str(data.get("critique", "No critique provided."))
        return score, critique
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Regex fallback for score
    m = re.search(r'"?overall_score"?\s*:\s*([0-9.]+)', raw)
    score = float(m.group(1)) if m else 0.5
    score = max(0.0, min(1.0, score))

    m_c = re.search(r'"critique"\s*:\s*"([^"]+)"', raw)
    critique = m_c.group(1) if m_c else "Unable to parse critique."

    return score, critique
