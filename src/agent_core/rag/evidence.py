"""Evidence compression."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def compress_evidence(items: list[dict], max_chars: int = 1800) -> str:
    """把异构检索结果压缩成带来源标记的有限长度证据文本。"""
    # 按输入顺序累计有效证据行，使排序结果在压缩后保持稳定。
    lines = []
    # 逐条兼容通用 RAG、策略库和话术库三种结果结构。
    for item in items:
        # 按优先级选择正文、有效策略或可用话术；全部缺失时视为空证据。
        text = item.get("text") or item.get("effective_strategy") or item.get("usable_script") or ""
        # 优先使用可追踪的 source_id，缺失时回退通用来源字段与 unknown。
        source = item.get("source_id") or item.get("source") or "unknown"
        # 只拼接非空证据，避免生成只有来源标签的无意义行。
        if text:
            # 将来源与正文绑定，供生成和 grounding 阶段追踪证据出处。
            lines.append(f"[{source}] {text}")
    # 用换行保留各证据边界，减少模型混淆不同来源的风险。
    digest = "\n".join(lines)
    # 在字符上限处截断，控制上下文预算并返回可直接注入提示词的文本。
    return digest[:max_chars]
