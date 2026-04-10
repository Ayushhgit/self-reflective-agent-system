"""
Shared AgentState TypedDict used across all LangGraph nodes.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Central state object that flows through every node in the LangGraph.

    Fields
    ------
    query        : Original user query.
    plan         : Ordered list of sub-steps produced by the Planner.
    current_step : Index of the step currently being executed.
    route        : Routing decision – "rag" | "tool" | "direct".
    retrieved_docs: List of text chunks returned by the RAG retriever.
    tool_output  : Raw output produced by an external tool.
    draft_answer : Working answer produced by the Synthesizer.
    final_answer : Accepted answer after evaluation / reflection.
    critique     : Textual critique from the Evaluator.
    score        : Quality score in [0, 1] from the Evaluator.
    iterations   : Number of reflection–synthesis cycles completed.
    max_iters    : Hard cap on reflection cycles (default 3).
    memory       : Accumulated conversation history (list of dicts).
    """

    query: str
    plan: List[str]
    current_step: int
    route: str                   # "rag" | "tool" | "direct"
    retrieved_docs: List[str]
    tool_output: str
    draft_answer: str
    final_answer: str
    critique: str
    score: float
    iterations: int
    max_iters: int
    memory: List[dict]
