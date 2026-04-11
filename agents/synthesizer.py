"""
Combines the query, plan, retrieved documents, and/or tool output into a
coherent, well-structured draft answer.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI assistant and technical writer.
Your task is to produce a comprehensive, accurate, and well-structured answer.

Guidelines:
- Use markdown formatting (headings, code blocks, bullet lists).
- Cite retrieved context where relevant.
- If code is required, provide clean, commented, working code.
- Be thorough but concise; avoid unnecessary filler.
- If this is a reflection pass, incorporate the provided critique to improve.
"""


def synthesis_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Synthesizer

    Responsibilities
    ----------------
    - Combine query + plan + retrieved_docs + tool_output into a draft answer.
    - On reflection passes (iterations > 0), incorporate the critique.
    """
    query = state["query"]
    plan = state.get("plan", [])
    retrieved_docs = state.get("retrieved_docs", [])
    tool_output = state.get("tool_output", "")
    critique = state.get("critique", "")
    iterations = state.get("iterations", 0)
    route = state.get("route", "direct")

    logger.info("[Synthesizer] Route=%s, Iteration=%d", route, iterations)

    # ── Build context block ──────────────────────────────────────────────────
    context_parts = []

    if retrieved_docs:
        docs_text = "\n\n---\n\n".join(retrieved_docs)
        context_parts.append(f"## Retrieved Context\n{docs_text}")

    if tool_output:
        context_parts.append(f"## Tool Output\n```\n{tool_output}\n```")

    if critique and iterations > 0:
        context_parts.append(
            f"## Previous Answer Critique (please fix these issues)\n{critique}"
        )

    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
    context_parts.append(f"## Execution Plan\n{plan_text}")

    full_context = "\n\n".join(context_parts)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    llm = get_llm(temperature=0.3)
    user_msg = (
        f"**User Query:** {query}\n\n"
        f"{full_context}\n\n"
        "Please provide a comprehensive answer to the query above."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    draft = response.content.strip()
    logger.info("[Synthesizer] Draft generated (%d chars).", len(draft))

    return {**state, "draft_answer": draft}
