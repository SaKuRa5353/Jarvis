"""Unit-тесты математического модуля: безопасный evaluate + tool calculate."""

import math

from src.llm_tools import _format_math_answer, evaluate_expression, execute_tool


def test_calculate_tool_returns_answer():
    result = execute_tool("calculate", {"expression": "2+2"})
    assert result is not None
    answer, phrase_key = result
    assert answer == "4"
    assert phrase_key is None


def test_simple_arithmetic():
    assert evaluate_expression("2+2") == 4.0
    assert evaluate_expression("(2+3)*4") == 20.0
    assert evaluate_expression("10 - 4") == 6.0
    assert evaluate_expression("2^3") is None  # ^ — BitOp, не поддерживается


def test_trigonometry():
    assert math.isclose(evaluate_expression("sin(pi/2)"), 1.0, abs_tol=1e-6)
    assert math.isclose(evaluate_expression("cos(0)"), 1.0, abs_tol=1e-6)
    assert math.isclose(evaluate_expression("tg(pi/4)"), 1.0, abs_tol=1e-6)


def test_constants_and_sqrt():
    assert math.isclose(evaluate_expression("sqrt(16)"), 4.0, abs_tol=1e-6)
    assert math.isclose(evaluate_expression("pi * 2"), math.pi * 2, abs_tol=1e-6)


def test_invalid_or_dangerous():
    assert evaluate_expression("") is None
    assert evaluate_expression("(2+2") is None  # синтаксическая ошибка
    assert evaluate_expression("import os") is None  # код-инъекция не пройдёт
    assert evaluate_expression("1/0") is None  # деление на ноль


def test_format_math_answer_whole_number():
    assert _format_math_answer(4.0) == "4"
    assert _format_math_answer(20.0) == "20"
    assert _format_math_answer(1.0 / 3) == "0.333333"
