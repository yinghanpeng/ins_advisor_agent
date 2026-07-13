"""Local summarizer adapter."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations


def run(arguments: dict) -> dict:
    """用本地截断方式生成摘要，保证无模型环境下也能测试工具链路。"""
    # 把输入正文标准化为字符串，避免切片时因非字符串参数导致类型错误。
    text = str(arguments.get("text", ""))
    # 最大字符数由调用方控制，缺省限制为 300 个字符。
    max_chars = int(arguments.get("max_chars", 300))
    # 本地适配器只做确定性前缀截断，不声称执行了语义摘要。
    summary = text[:max_chars]
    # 同时返回是否发生截断，让调用方能够准确描述结果完整性。
    return {"summary": summary, "truncated": len(text) > max_chars}
