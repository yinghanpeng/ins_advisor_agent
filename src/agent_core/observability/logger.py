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
    # 将字符串日志级别转换成 logging 常量；传错时回退到 INFO。
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


@dataclass
class StructuredLogger:
    """Tiny JSON logger that is easy to replace with a production sink."""

    # name 是 logger 名称，生产环境可映射到服务名或模块名。
    name: str = "agent_core"
    # default_fields 会注入每条日志，例如环境、版本、tenant 等全局上下文。
    default_fields: dict[str, Any] = field(default_factory=dict)

    def event(self, event: str, **fields: Any) -> None:
        """输出一条 JSON 结构化事件日志。"""
        # payload 是最终日志对象，统一包含时间、logger、event 和调用方传入字段。
        payload = {
            # ts 使用 UTC ISO 时间，便于跨服务排序。
            "ts": utc_now_iso(),
            # logger 表示日志来源模块。
            "logger": self.name,
            # event 是业务事件名，例如 agent_run_started、trace_event。
            "event": event,
            # default_fields 放在前面，fields 可以覆盖默认字段。
            **self.default_fields,
            # fields 是调用点传入的结构化上下文。
            **fields,
        }
        # ensure_ascii=False 保留中文日志，default=str 兼容 datetime/Enum 等对象。
        logging.getLogger(self.name).info(json.dumps(payload, ensure_ascii=False, default=str))

    def warning(self, event: str, **fields: Any) -> None:
        """输出一条 JSON 结构化 warning 日志。"""
        # warning payload 比普通 event 多一个 level 字段，便于日志平台过滤。
        payload = {
            "ts": utc_now_iso(),
            "logger": self.name,
            "event": event,
            "level": "warning",
            **self.default_fields,
            **fields,
        }
        # warning 走 logging.warning，生产 sink 可以设置不同告警策略。
        logging.getLogger(self.name).warning(json.dumps(payload, ensure_ascii=False, default=str))


# 模块级默认 logger，供简单模块直接 import 使用。
logger = StructuredLogger()
