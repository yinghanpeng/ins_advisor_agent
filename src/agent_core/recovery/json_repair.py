"""Small JSON repair helper."""

# 文件说明：
# - 本文件属于 Retry / Recovery 层，负责重试、降级、JSON repair 或恢复计划。
# - 失败时应清楚记录原因，不能无依据编造答案。
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))

