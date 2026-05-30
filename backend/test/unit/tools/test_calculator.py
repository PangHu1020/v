"""Unit tests for the ``calculator`` agent tool.

Covers the safe-AST evaluator end to end: well-formed expressions,
operator precedence, malformed input rejection, division by zero,
exponent capping, and adversarial input that would crash a naive
``eval()`` (variable refs, attribute access, function calls).
"""

from __future__ import annotations

import pytest

from backend.v.tools.calculator import calculator


class TestHappyPath:
    async def test_simple_addition(self) -> None:
        assert (await calculator.ainvoke({"expression": "1 + 2"})) == "3"

    async def test_decimal_multiplication(self) -> None:
        # 1280 * 0.85 = 1088 — a typical discount calc.
        assert (await calculator.ainvoke({"expression": "1280 * 0.85"})) == "1088"

    async def test_precedence_and_parens(self) -> None:
        assert (await calculator.ainvoke({"expression": "(99 + 12) * 2"})) == "222"

    async def test_unary_minus(self) -> None:
        assert (await calculator.ainvoke({"expression": "-5 + 8"})) == "3"

    async def test_floor_div_and_mod(self) -> None:
        assert (await calculator.ainvoke({"expression": "17 // 5"})) == "3"
        assert (await calculator.ainvoke({"expression": "17 % 5"})) == "2"

    async def test_power(self) -> None:
        assert (await calculator.ainvoke({"expression": "2 ** 10"})) == "1024"

    async def test_decimal_result_trims_trailing_zeros(self) -> None:
        assert (await calculator.ainvoke({"expression": "1 / 4"})) == "0.25"


class TestErrors:
    @pytest.mark.parametrize("expr", ["", "   "])
    async def test_empty_returns_error(self, expr: str) -> None:
        result = await calculator.ainvoke({"expression": expr})
        assert result.startswith("[calculator_error]")

    async def test_division_by_zero(self) -> None:
        result = await calculator.ainvoke({"expression": "1 / 0"})
        assert result.startswith("[calculator_error]")

    async def test_syntax_error(self) -> None:
        result = await calculator.ainvoke({"expression": "1 + "})
        assert result.startswith("[calculator_error]")

    async def test_huge_exponent_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "2 ** 10000"})
        assert result.startswith("[calculator_error]")

    async def test_too_long_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "1+" * 200})
        assert result.startswith("[calculator_error]")


class TestSandboxing:
    """Inputs that would execute on a naive ``eval()`` must be refused."""

    async def test_variable_reference_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "x + 1"})
        assert result.startswith("[calculator_error]")

    async def test_attribute_access_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "(1).bit_length"})
        assert result.startswith("[calculator_error]")

    async def test_function_call_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "abs(-1)"})
        assert result.startswith("[calculator_error]")

    async def test_comparison_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "1 < 2"})
        assert result.startswith("[calculator_error]")

    async def test_string_literal_rejected(self) -> None:
        result = await calculator.ainvoke({"expression": "'a' + 'b'"})
        assert result.startswith("[calculator_error]")
