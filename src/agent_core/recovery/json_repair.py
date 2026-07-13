"""Small JSON repair helper."""

# 文件说明：
# - 本文件属于 Retry / Recovery 层，负责重试、降级、JSON repair 或恢复计划。
# - 失败时应清楚记录原因，不能无依据编造答案。
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """从可能混有说明文字的模型输出中提取并解析第一个 JSON 对象。"""
    # 使用跨行模式定位最外层花括号片段，兼容模型输出中的换行。
    match = re.search(r"\{.*\}", text, re.S)
    # 找不到对象时显式失败，让上层恢复策略决定重试或降级，不能返回伪造数据。
    if not match:
        # 抛出稳定错误类型，便于调用方归类为 json_parse_failed。
        raise ValueError("no JSON object found")
    # 将已匹配片段交给标准库严格解析，返回结构化字典或透传 JSONDecodeError。
    return json.loads(match.group(0))
