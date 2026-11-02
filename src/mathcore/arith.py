"""Безопасное вычисление ограниченной арифметической грамматики.

Разрешённая грамматика состоит из числовых констант, имён ``pi``, ``e`` и
``tau``, унарных ``+``/``-``, бинарных ``+``, ``-``, ``*``, ``/`` и ``**``, а
также прямых вызовов белого списка функций с позиционными аргументами.
Результат каждой операции проверяется: он должен быть конечным числом и не
превышать ``1e18`` по модулю. При ``N`` узлах AST вычисление занимает O(N)
времени и O(N) памяти; вход ограничен длиной и глубиной до разбора и числом
узлов после разбора.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

MAX_ABS_VALUE = 1e18
MAX_EXPR_LENGTH = 4096
MAX_DEPTH = 64
MAX_POWER_EXPONENT = 1000.0


def _safe_round(value: float, ndigits: float = 0.0) -> float:
    """Округляет число; ``ndigits`` должен быть конечным целым в разумном диапазоне."""
    if not math.isfinite(ndigits) or not ndigits.is_integer() or abs(ndigits) > 1000:
        raise ValueError("недопустимый ndigits")
    return float(round(value, int(ndigits)))

ALLOWED_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
ALLOWED_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tg": math.tan,
    "log": math.log,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": _safe_round,
}
ALLOWED_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


class ArithError(ValueError):
    """Ошибка синтаксиса, политики или вычисления арифметики."""


def _check_max_nodes(max_nodes: int) -> None:
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ArithError("лимит узлов должен быть положительным целым числом")


def _depth(node: ast.AST, current: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return current
    return max(_depth(child, current + 1) for child in children)


def validate_ast(tree: ast.AST, max_nodes: int = 200) -> None:
    """Проверяет AST по белому списку узлов и лимитам времени/памяти."""
    _check_max_nodes(max_nodes)
    nodes = list(ast.walk(tree))
    if len(nodes) > max_nodes:
        raise ArithError("превышен лимит узлов выражения")
    if _depth(tree) > MAX_DEPTH:
        raise ArithError("превышена глубина выражения")
    function_nodes = {
        id(node.func)
        for node in nodes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp,
               ast.UnaryOp, ast.Call, ast.operator, ast.unaryop)
    for node in nodes:
        if not isinstance(node, allowed):
            raise ArithError(f"узел {type(node).__name__} запрещён")
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ArithError("разрешены только числовые константы")
            if not math.isfinite(float(node.value)) or abs(float(node.value)) > MAX_ABS_VALUE:
                raise ArithError("числовая константа вне допустимого диапазона")
        elif isinstance(node, ast.Name):
            if node.id not in ALLOWED_CONSTS and id(node) not in function_nodes:
                raise ArithError("неизвестное имя")
        elif isinstance(node, ast.BinOp) and type(node.op) not in ALLOWED_BINOPS:
            raise ArithError("оператор запрещён")
        elif isinstance(node, ast.UnaryOp) and type(node.op) not in ALLOWED_UNARYOPS:
            raise ArithError("унарный оператор запрещён")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ArithError("функция запрещена")
            if node.keywords:
                raise ArithError("именованные аргументы запрещены")


def _checked(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArithError("результат не является вещественным числом")
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_ABS_VALUE:
        raise ArithError("результат вне допустимого диапазона")
    return result


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return _checked(node.value)
    if isinstance(node, ast.Name):
        return _checked(ALLOWED_CONSTS[node.id])
    if isinstance(node, ast.UnaryOp):
        return _checked(ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand)))
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise ArithError("слишком большой показатель степени")
        try:
            return _checked(ALLOWED_BINOPS[type(node.op)](left, right))
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ArithError("ошибка арифметической операции") from exc
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ArithError("функция запрещена")
        function_name = node.func.id
        args = [_eval(arg) for arg in node.args]
        try:
            return _checked(ALLOWED_FUNCS[function_name](*args))
        except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
            raise ArithError("неверные аргументы функции") from exc
    raise ArithError("узел не поддерживается")


def safe_eval(expr: str, max_nodes: int = 200) -> float:
    """Безопасно вычисляет выражение и возвращает конечный ``float``."""
    _check_max_nodes(max_nodes)
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPR_LENGTH:
        raise ArithError("пустое или слишком длинное выражение")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ArithError("некорректный синтаксис выражения") from exc
    validate_ast(tree, max_nodes=max_nodes)
    try:
        return _eval(tree.body)
    except (KeyError, IndexError, RecursionError) as exc:
        raise ArithError("ошибка вычисления") from exc