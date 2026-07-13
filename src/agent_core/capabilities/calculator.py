"""Safe calculator capability."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import ast
import operator
from typing import Any


# 运算符白名单把 AST 节点类型映射为纯算术函数，确保解释器不会执行任意 Python 代码。
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    """递归解释算术白名单中的单个 AST 节点。"""

    # 重点逻辑：只允许数字常量参与计算，避免执行任意 Python 代码。
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        # 统一转成 float，使所有受支持表达式都返回稳定的数值类型。
        return float(node.value)
    # 重点逻辑：只允许 OPS 白名单里的运算符，例如加减乘除和乘方。
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        # 递归计算左右操作数后调用白名单函数，不使用 eval/exec。
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    # 一元负号同样必须命中白名单，避免放行其它一元 AST 节点。
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        # 递归求值操作数后执行白名单中的一元运算函数。
        return OPS[type(node.op)](_eval(node.operand))
    # 重点逻辑：任何函数调用、变量访问、属性访问都会被拒绝。
    raise ValueError("unsupported expression")


def run(arguments: dict[str, Any]) -> dict[str, float]:
    """执行安全算术表达式计算，只解释 AST 白名单节点。"""
    # 从工具参数读取表达式并标准化为字符串，缺省值为空字符串以便解析器明确报错。
    expression = str(arguments.get("expression", ""))
    # 重点逻辑：先把表达式解析成 AST，再由 _eval 解释执行白名单节点。
    tree = ast.parse(expression, mode="eval")
    # 仅把表达式根节点交给白名单解释器，并以工具统一的对象结构返回结果。
    return {"result": _eval(tree.body)}
