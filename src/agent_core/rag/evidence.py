"""Evidence compression."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def compress_evidence(items: list[dict], max_chars: int = 1800) -> str:
    lines = []
    for item in items:
        text = item.get("text") or item.get("effective_strategy") or item.get("usable_script") or ""
        source = item.get("source_id") or item.get("source") or "unknown"
        if text:
            lines.append(f"[{source}] {text}")
    digest = "\n".join(lines)
    return digest[:max_chars]

