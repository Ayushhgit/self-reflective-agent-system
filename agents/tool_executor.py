"""
Decides which tool is most appropriate for the query, executes it, and stores
the result in state.tool_output.

Registered tools
----------------
- python_executor : Runs Python code snippets safely.
- calculator      : Evaluates mathematical expressions.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_factory import get_llm
from tools.python_executor import execute_python
from tools.calculator import calculate

logger = logging.getLogger(__name__)

TOOL_SELECTION_PROMPT = """You are a tool dispatcher. Choose the correct tool:

- "python"     : The task requires writing or executing Python code.
- "calculator" : The task requires evaluating a mathematical expression.

Output JSON: {"tool": "<tool_name>", "input": "<tool_input>"}

For "python" tool, "input" is the complete Python code to execute.
For "calculator" tool, "input" is the math expression (e.g. "2 ** 32 + 17").

Do not include any other text.
"""


def tool_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Tool Executor

    Responsibilities
    ----------------
    - Use the LLM to select and prepare the appropriate tool call.
    - Execute the chosen tool.
    - Store raw output in state.tool_output.
    """
    query = state["query"]
    plan = state.get("plan", [])
    logger.info("[Tool] Selecting tool for: %s", query[:80])

    llm = get_llm(temperature=0.1)
    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))

    messages = [
        SystemMessage(content=TOOL_SELECTION_PROMPT),
        HumanMessage(content=f"Query: {query}\nPlan:\n{plan_text}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    tool_name, tool_input = _parse_tool_call(raw, query)
    logger.info("[Tool] Selected: %s", tool_name)

    output = _run_tool(tool_name, tool_input)
    logger.info("[Tool] Output (first 200 chars): %s", str(output)[:200])

    return {**state, "tool_output": output}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_tool_call(raw: str, fallback_query: str):
    """Parse LLM JSON output into (tool_name, tool_input)."""
    import json, re

    # Strip code fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
        return data.get("tool", "python"), data.get("input", fallback_query)
    except json.JSONDecodeError:
        pass

    # Minimal regex fallback
    m = re.search(r'"tool"\s*:\s*"(\w+)"', raw)
    tool = m.group(1) if m else "python"
    return tool, fallback_query


def _run_tool(tool_name: str, tool_input: str) -> str:
    """Dispatch to the appropriate tool function."""
    if tool_name == "calculator":
        return calculate(tool_input)
    # Default → python executor
    return execute_python(tool_input)
