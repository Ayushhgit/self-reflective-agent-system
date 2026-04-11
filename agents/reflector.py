"""
Receives the critique from the Evaluator and produces targeted improvement
instructions that are passed to the Synthesizer in the next iteration.

The Reflector does NOT rewrite the answer itself – it reasons about *how*
to improve it, then updates state so the Synthesizer can produce a better
draft.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a self-reflection engine for an AI research agent.

You will receive:
1. The original query.
2. The draft answer (which has issues).
3. A critique from the evaluator.

Your job is to produce a concise, structured improvement plan (3–5 bullet
points) that the answer synthesizer should follow in the next attempt.

Focus on:
- Fixing factual errors (if any).
- Addressing missing parts.
- Removing unsupported claims.
- Improving code quality (if code is involved).
- Improving clarity and completeness.

Output only the improvement bullet list (no extra text).
"""


def reflection_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Reflector

    Responsibilities
    ----------------
    - Analyse critique and draft answer.
    - Produce a structured improvement plan.
    - Append that plan to state.critique (so Synthesizer can see it).
    - Increment state.iterations.
    """
    query = state["query"]
    draft = state.get("draft_answer", "")
    critique = state.get("critique", "")
    iterations = state.get("iterations", 0)

    logger.info("[Reflector] Reflection pass %d.", iterations + 1)

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Draft Answer:\n{draft[:2000]}\n\n"
                f"Evaluator Critique:\n{critique}"
            )
        ),
    ]

    response = llm.invoke(messages)
    improvement_plan = response.content.strip()
    logger.info("[Reflector] Improvement plan: %s", improvement_plan[:120])

    # Append the improvement plan to the critique so the Synthesizer can use it
    enhanced_critique = f"{critique}\n\n**Improvement Plan:**\n{improvement_plan}"

    return {
        **state,
        "critique": enhanced_critique,
        "iterations": iterations + 1,
    }
