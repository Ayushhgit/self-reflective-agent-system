"""
Analyses the query and plan, then decides which execution path to take:

    "rag"    → Retrieve relevant documents from the vector store.
    "tool"   → Execute an external tool (Python runner, calculator, etc.).
    "direct" → Answer directly from LLM knowledge (no retrieval needed).
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a routing agent. Given a user query and an execution
plan, decide which strategy to use:

- "rag"    : The query needs factual information, documents, or research only.
- "tool"   : The query requires executing code or calculations only.
- "hybrid" : The query needs BOTH research/explanation AND working code/calculation.
- "direct" : The query can be answered from general knowledge without
             retrieval or tool use.

Output ONLY one word: rag, tool, hybrid, or direct.
"""

# Keywords that strongly suggest tool usage
TOOL_KEYWORDS = [
    "calculate", "compute", "sort", "leetcode",
]

# Keywords that strongly suggest RAG usage
RAG_KEYWORDS = [
    "what is", "how does", "history", "theory", "paper", "study", "survey",
]

# Keywords that suggest BOTH research and code are needed
HYBRID_KEYWORDS = [
    ("explain", "implement"), ("explain", "code"), ("explain", "write"),
    ("research", "implement"), ("research", "code"), ("research", "python"),
    ("understand", "implement"), ("concept", "code"), ("concept", "python"),
    ("overview", "implement"), ("explain", "program"), ("theory", "code"),
    ("research", "write"), ("describe", "implement"),
]


def router_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Router

    Responsibilities
    ----------------
    - Inspect query and plan to determine the execution route.
    - Set state.route to "rag" | "tool" | "hybrid" | "direct".

    "hybrid" runs retriever → tool_executor → synthesis, showing all features.
    """
    query = state["query"]
    plan = state.get("plan", [])
    logger.info("[Router] Deciding route for query: %s", query[:80])

    ql = query.lower()

    # ── Hybrid detection (research + code together) ───────────────────────
    if any(a in ql and b in ql for a, b in HYBRID_KEYWORDS):
        route = "hybrid"
        logger.info("[Router] Heuristic → hybrid")

    # ── Pure tool ────────────────────────────────────────────────────────
    elif any(kw in ql for kw in TOOL_KEYWORDS):
        route = "tool"
        logger.info("[Router] Heuristic → tool")

    # ── Pure RAG ─────────────────────────────────────────────────────────
    elif any(kw in ql for kw in RAG_KEYWORDS):
        route = "rag"
        logger.info("[Router] Heuristic → rag")

    else:
        # ── LLM-based routing ────────────────────────────────────────────
        llm = get_llm(temperature=0.0)
        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nPlan:\n{plan_text}"),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip().lower()
        route = _extract_route(raw)
        logger.info("[Router] LLM → %s (raw: %s)", route, raw[:30])

    return {**state, "route": route}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_route(text: str) -> str:
    if "hybrid" in text:
        return "hybrid"
    if "tool" in text:
        return "tool"
    if "rag" in text:
        return "rag"
    if "direct" in text:
        return "direct"
    return "rag"
