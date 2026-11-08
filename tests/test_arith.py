"""Тесты безопасного арифметического backend и его интеграции."""

import math

import pytest

from src.llm_tools import evaluate_expression, execute_tool
from src.mathcore.arith import ArithError, safe_eval, validate_ast


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", 4.0),
        ("(12+7)*3", 57.0),
        ("2**10", 1024.0),
        ("sqrt(16)", 4.0),
        ("pi", math.pi),
        ("-5+3", -2.0),
        ("10/4", 2.5),
        ("abs(-7)", 7.0),
        ("round(3.14159, 2)", 3.14),
    ],
)
def test_safe_eval_allows_documented_arithmetic(expression: str, expected: float) -> None:
    assert math.isclose(safe_eval(expression), expected, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "(1).__class__",
        "lambda: 1",
        "a.b",
        "x[0]",
        "[i for i in range(3)]",
        "2 if True else 3",
        "1/0",
        "1e400",
        "nan",
        "inf",
        "'abc'",
        "b'x'",
        "print(1)",
        "eval('1')",
        "True",
        "False",
        "unknown",
        "sqrt()",
        "sqrt(1, 2)",
        "sqrt(value=1)",
    ],
)
def test_safe_eval_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(ArithError):
        safe_eval(expression)


@pytest.mark.parametrize("expression", ["sqrt", "abs", "round", "pi + sqrt"])
def test_bare_function_name_is_rejected(expression: str) -> None:
    with pytest.raises(ArithError):
        safe_eval(expression)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sin(pi/2)", 1.0),
        ("cos(0)", 1.0),
        ("cos(pi/4)", math.cos(math.pi / 4)),
        ("sin(0)", 0.0),
        ("tan(0)", 0.0),
        ("sin(tau)", 0.0),
        ("exp(0)", 1.0),
        ("log(e)", 1.0),
    ],
)
def test_safe_eval_trig_and_transcendental(expression: str, expected: float) -> None:
    assert math.isclose(safe_eval(expression), expected, rel_tol=1e-9, abs_tol=1e-12)


def test_safe_eval_rejects_limits_and_invalid_limit() -> None:
    with pytest.raises(ArithError):
        safe_eval("1+2", max_nodes=0)
    with pytest.raises(ArithError):
        safe_eval("1+2", max_nodes=True)
    with pytest.raises(ArithError):
        safe_eval("1+2", max_nodes=1)
    with pytest.raises(ArithError):
        safe_eval("2**1001")
    with pytest.raises(ArithError):
        safe_eval("10**18 * 10")
    with pytest.raises(ArithError):
        safe_eval("1" * 4097)


def test_validate_ast_rejects_manually_created_tree() -> None:
    import ast

    tree = ast.Expression(body=ast.Name(id="danger", ctx=ast.Load()))
    with pytest.raises(ArithError):
        validate_ast(tree)


@pytest.mark.parametrize("expression", ["", "   ", "open('/tmp/should-not-exist')", "1/0"])
def test_legacy_adapter_and_tool_keep_none_contract(expression: str) -> None:
    assert evaluate_expression(expression) is None
    assert execute_tool("calculate", {"expression": expression}) is None


def test_calculate_tool_keeps_expression_schema_and_answer_format() -> None:
    assert execute_tool("calculate", {"expression": "2+2"}) == ("4", None)


def test_calculate_tool_sin_pi_over_2() -> None:
    assert execute_tool("calculate", {"expression": "sin(pi/2)"}) == ("1", None)