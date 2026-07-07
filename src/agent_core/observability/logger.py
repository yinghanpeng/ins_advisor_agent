"""Local structured logging.

LangSmith is useful for tracing, but local logs are the always-on source of truth.
"""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_core.utils.time import utc_now_iso


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a compact plain formatter."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


@dataclass
class StructuredLogger:
    """Tiny JSON logger that is easy to replace with a production sink."""

    name: str = "agent_core"
    default_fields: dict[str, Any] = field(default_factory=dict)

    def event(self, event: str, **fields: Any) -> None:
        """输出一条 JSON 结构化事件日志。"""
        payload = {
            "ts": utc_now_iso(),
            "logger": self.name,
            "event": event,
            **self.default_fields,
            **fields,
        }
        logging.getLogger(self.name).info(json.dumps(payload, ensure_ascii=False, default=str))

    def warning(self, event: str, **fields: Any) -> None:
        """输出一条 JSON 结构化 warning 日志。"""
        payload = {
            "ts": utc_now_iso(),
            "logger": self.name,
            "event": event,
            "level": "warning",
            **self.default_fields,
            **fields,
        }
        logging.getLogger(self.name).warning(json.dumps(payload, ensure_ascii=False, default=str))


logger = StructuredLogger()
