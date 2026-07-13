"""JSON helpers used by contracts and recovery."""

# 文件说明：
# - 本文件属于工具函数层，提供 ID、时间、JSON 等通用辅助能力。
# - 工具函数应保持简单、可测试、无业务耦合。
from __future__ import annotations

import json
from typing import Any


def compact_json(value: Any) -> str:
    """Serialize a value as compact UTF-8 JSON."""

    # 保留中文并移除非必要空格，生成适合日志、缓存键或模型上下文的紧凑 JSON。
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning default when parsing fails."""

    # 将解析隔离在异常边界内，使恢复链路可以对损坏缓存使用调用方给定缺省值。
    try:
        # 严格使用标准 JSON 解析器，不接受 Python 字面量等非标准格式。
        return json.loads(text)
    # 任意解析或输入类型异常都由安全缺省值吸收，恢复链路无需感知解析器细节。
    except Exception:
        # 解析或输入类型异常时返回显式 default，保证辅助函数不会中断恢复流程。
        return default
