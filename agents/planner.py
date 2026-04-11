"""
Receives the raw user query and produces an ordered, numbered plan that
the downstream nodes will execute step by step.
"""

import json
import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strategic AI planner. Your job is to decompose a
complex user query into a clear, ordered sequence of 3–6 concrete sub-steps.

Rules:
- Each step must be specific and actionable.
- Steps should build logically on each other.
- Output ONLY a JSON array of strings. No extra text.

Example output:
["Understand the key concepts behind X", "Identify relevant formulas/code",
 "Explain step by step", "Provide a working example", "Summarise findings"]
"""


def planner_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Planner

    Responsibilities
    ----------------
    - Receive the user query from state.
    - Call the LLM to generate a step-by-step execution plan.
    - Store the plan back in state.plan.
    - Reset runtime counters (current_step, iterations).
    """
    query = state["query"]
    logger.info("[Planner] Generating plan for: %s", query[:80])

    llm = get_llm(temperature=0.2)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # ── Parse the JSON list ──────────────────────────────────────────────────
    plan: List[str] = _parse_plan(raw, query)

    logger.info("[Planner] Plan (%d steps): %s", len(plan), plan)

    return {
        **state,
        "plan": plan,
        "current_step": 0,
        "iterations": 0,
        "retrieved_docs": [],
        "tool_output": "",
        "draft_answer": "",
        "final_answer": "",
        "critique": "",
        "score": 0.0,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_plan(raw: str, fallback_query: str) -> List[str]:
    """
    Extract a JSON array of strings from the LLM response.
    Falls back to a sensible single-step plan on parse failure.
    """
    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        plan = json.loads(raw)
        if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
            return plan
    except json.JSONDecodeError:
        pass

    # Fallback: extract numbered lines
    lines = [
        line.lstrip("0123456789.-) ").strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    if lines:
        return lines

    # Last resort
    return [f"Research and answer: {fallback_query}"]
