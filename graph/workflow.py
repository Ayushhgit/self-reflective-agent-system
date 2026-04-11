"""
Defines and compiles the full multi-agent graph:

    User Query
        ↓
    [planner_node]   – Break query into steps
        ↓
    [router_node]    – Decide: rag | tool | direct
        ↓
    ┌───┴──────────┐
  [retriever]   [tool]   (direct → skip both)
    └───┬──────────┘
        ↓
    [synthesis_node]  – Draft answer
        ↓
    [evaluator_node]  – Score + Critique
        ↓
    ┌───┴───────────────────────────────────┐
 score ≥ 0.85                         score < 0.85
 or max_iters reached                 and iters < max_iters
    ↓                                      ↓
 [memory_node]                      [reflection_node]
    ↓                                      ↓
  END                              [synthesis_node] (loop)
"""

import logging
from functools import lru_cache

from langgraph.graph import StateGraph, END

from state import AgentState
from config import SCORE_THRESHOLD, MAX_ITERATIONS
from agents.planner import planner_node
from agents.router import router_node
from agents.retriever import retriever_node
from agents.tool_executor import tool_node
from agents.synthesizer import synthesis_node
from agents.evaluator import evaluator_node
from agents.reflector import reflection_node
from agents.memory_agent import memory_node

logger = logging.getLogger(__name__)


# ─── Conditional edge functions ───────────────────────────────────────────────

def route_decision(state: AgentState) -> str:
    """
    After router_node: branch to retriever, tool_executor, synthesis, or
    retriever-then-tool (hybrid).

    "hybrid" first goes to retriever; after retrieval a second conditional
    (after_retriever_decision) sends it to tool instead of synthesis.
    """
    route = state.get("route", "direct")
    logger.info("[Graph] Route decision: %s", route)
    # "hybrid" starts at retriever just like "rag"
    return "retriever" if route in ("rag", "hybrid") else route


def after_retriever_decision(state: AgentState) -> str:
    """
    After retriever_node: hybrid route continues to tool; rag goes to synthesis.
    """
    route = state.get("route", "rag")
    if route == "hybrid":
        logger.info("[Graph] Hybrid: retriever done → tool")
        return "tool"
    return "synthesis"


def evaluation_decision(state: AgentState) -> str:
    """
    After evaluator_node: accept the answer or enter the reflection loop.

    Accept conditions:
      - score >= SCORE_THRESHOLD, OR
      - iterations >= MAX_ITERATIONS (give up improving)
    """
    score = state.get("score", 0.0)
    iters = state.get("iterations", 0)
    max_i = state.get("max_iters", MAX_ITERATIONS)

    if score >= SCORE_THRESHOLD or iters >= max_i:
        logger.info(
            "[Graph] Evaluation → accept (score=%.2f, iters=%d/%d)", score, iters, max_i
        )
        return "accept"
    else:
        logger.info(
            "[Graph] Evaluation → reflect (score=%.2f, iters=%d/%d)", score, iters, max_i
        )
        return "reflect"


# ─── Build graph ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_graph():
    """
    Construct and compile the AREA LangGraph.
    Cached so multiple calls reuse the same compiled graph.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────────────────
    builder.add_node("planner",    planner_node)
    builder.add_node("router",     router_node)
    builder.add_node("retriever",  retriever_node)
    builder.add_node("tool",       tool_node)
    builder.add_node("synthesis",  synthesis_node)
    builder.add_node("evaluator",  evaluator_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("memory",     memory_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.set_entry_point("planner")

    # ── Fixed edges ──────────────────────────────────────────────────────────
    builder.add_edge("planner",    "router")
    builder.add_edge("tool",       "synthesis")
    builder.add_edge("synthesis",  "evaluator")
    builder.add_edge("reflection", "synthesis")
    builder.add_edge("memory",     END)

    # ── Conditional: routing (after router) ──────────────────────────────────
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",   # rag + hybrid both start at retriever
            "tool":      "tool",
            "direct":    "synthesis",
        },
    )

    # ── Conditional: after retriever (hybrid → tool, rag → synthesis) ────────
    builder.add_conditional_edges(
        "retriever",
        after_retriever_decision,
        {
            "tool":      "tool",
            "synthesis": "synthesis",
        },
    )

    # ── Conditional: evaluate → accept or reflect ────────────────────────────
    builder.add_conditional_edges(
        "evaluator",
        evaluation_decision,
        {
            "accept":  "memory",
            "reflect": "reflection",
        },
    )

    graph = builder.compile()
    logger.info("[Graph] AREA LangGraph compiled successfully.")
    return graph


# ─── Public API ───────────────────────────────────────────────────────────────

def run_query(query: str, memory: list = None, max_iters: int = None) -> dict:
    """
    Execute the AREA graph for a single user query.

    Parameters
    ----------
    query     : The user's question / task.
    memory    : Existing conversation memory list (optional).
    max_iters : Override the default reflection iteration cap.

    Returns
    -------
    dict with keys: final_answer, score, iterations, route, plan, critique
    """
    graph = build_graph()

    initial_state: AgentState = {
        "query":          query,
        "plan":           [],
        "current_step":   0,
        "route":          "direct",
        "retrieved_docs": [],
        "tool_output":    "",
        "draft_answer":   "",
        "final_answer":   "",
        "critique":       "",
        "score":          0.0,
        "iterations":     0,
        "max_iters":      max_iters if max_iters is not None else MAX_ITERATIONS,
        "memory":         memory or [],
    }

    logger.info("[Graph] Starting AREA run for query: %s", query[:80])
    result = graph.invoke(initial_state)
    logger.info(
        "[Graph] Run complete. Score=%.2f, Route=%s, Iters=%d",
        result.get("score", 0),
        result.get("route", "?"),
        result.get("iterations", 0),
    )
    return result
