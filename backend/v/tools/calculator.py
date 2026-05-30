"""Safe arithmetic evaluator exposed as the ``calculator`` agent tool.

The customer-service LLM frequently needs deterministic arithmetic for
discount / shipping / point-conversion math. Letting it do the math in
its head invites off-by-one errors and silent hallucination of numbers
the customer will then quote back. This tool gives it a precise, audited
escape hatch.

Implementation note — we evaluate via ``ast`` with a strict whitelist
rather than ``eval()`` or ``simpleeval``: the surface is small enough
that thirty lines of AST handling is cheaper than a third-party
dependency, and we keep zero attack surface (no names, no attribute
access, no calls, no comprehensions).
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

from backend.v.utils.logging import get_logger

_log = get_logger("tools.calculator")

_MAX_EXPRESSION_LEN = 200
_MAX_RESULT_ABS = 1e18

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported literal: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
        # Cap exponent before evaluating — `2 ** 10000` would otherwise
        # spend seconds materializing a result we'd reject anyway.
        if isinstance(node.op, ast.Pow):
            right = _eval_node(node.right)
            if abs(right) > 64:
                raise ValueError("exponent out of range")
            return op(_eval_node(node.left), right)
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _evaluate(expression: str) -> float:
    if not expression or not expression.strip():
        raise ValueError("empty expression")
    if len(expression) > _MAX_EXPRESSION_LEN:
        raise ValueError(f"expression too long (>{_MAX_EXPRESSION_LEN} chars)")
    tree = ast.parse(expression, mode="eval")
    value = _eval_node(tree.body)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric result")
    if abs(value) > _MAX_RESULT_ABS:
        raise ValueError("result magnitude exceeds safe range")
    return value


def _format(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    # Trim trailing zeros without surprising the customer with sci notation.
    return f"{value:.10f}".rstrip("0").rstrip(".")


@tool("calculator", parse_docstring=True)
async def calculator(expression: str) -> str:
    """Evaluate a numeric expression and return the result as a string.

    Supports + - * / // % ** unary +/- and parentheses. Numbers can be
    integers or decimals. No variables, no function calls, no
    comparisons. Use this for any arithmetic the customer needs (discounts,
    shipping fees, point conversions, totals) instead of computing it
    yourself.

    Args:
        expression: A pure-arithmetic Python expression. Examples:
            "1280 * 0.85", "(99 + 12) * 2", "5000 / 100".
    """
    try:
        value = _evaluate(expression)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        _log.warning("tools.calculator.failed", error=type(exc).__name__, detail=str(exc))
        return f"[calculator_error] {exc}"

    text = _format(value)
    _log.info("tools.calculator.evaluated", expression=expression, result=text)
    return text
