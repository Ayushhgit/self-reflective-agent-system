"""
Safely evaluates mathematical expressions using Python's ast module to
prevent arbitrary code execution.
"""

import ast
import operator
import math
import logging

logger = logging.getLogger(__name__)

# Allowed operators and functions
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
}


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Parameters
    ----------
    expression : str
        A mathematical expression string, e.g. "2 ** 32 + sqrt(144)".

    Returns
    -------
    str
        The numeric result as a string, or an error message.
    """
    expression = expression.strip()
    logger.info("[Calculator] Evaluating: %s", expression)
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


def _eval_node(node):
    """Recursively evaluate an AST node with allowlist checking."""
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed.")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed.")
        operand = _eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' not allowed.")
        args = [_eval_node(a) for a in node.args]
        return SAFE_FUNCTIONS[func_name](*args)

    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Name '{node.id}' not allowed.")

    raise ValueError(f"Unsupported AST node type: {type(node)}")
