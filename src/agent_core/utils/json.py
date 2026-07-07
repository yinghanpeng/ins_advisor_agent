"""JSON helpers used by contracts and recovery."""

# 文件说明：
# - 本文件属于工具函数层，提供 ID、时间、JSON 等通用辅助能力。
# - 工具函数应保持简单、可测试、无业务耦合。
from __future__ import annotations

import json
from typing import Any


def compact_json(value: Any) -> str:
    """Serialize a value as compact UTF-8 JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning default when parsing fails."""
    try:
        return json.loads(text)
    except Exception:
        return default

