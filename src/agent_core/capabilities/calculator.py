"""Safe calculator capability."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import ast
import operator
from typing import Any


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    # 重点逻辑：只允许数字常量参与计算，避免执行任意 Python 代码。
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    # 重点逻辑：只允许 OPS 白名单里的运算符，例如加减乘除和乘方。
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    # 重点逻辑：任何函数调用、变量访问、属性访问都会被拒绝。
    raise ValueError("unsupported expression")


def run(arguments: dict[str, Any]) -> dict[str, float]:
    """执行安全算术表达式计算，只解释 AST 白名单节点。"""
    expression = str(arguments.get("expression", ""))
    # 重点逻辑：先把表达式解析成 AST，再由 _eval 解释执行白名单节点。
    tree = ast.parse(expression, mode="eval")
    return {"result": _eval(tree.body)}
