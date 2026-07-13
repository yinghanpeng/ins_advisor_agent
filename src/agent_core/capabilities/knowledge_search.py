"""内部知识库搜索 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的知识库检索服务。"""
    # 服务地址由部署环境注入，代码本身不绑定具体知识库供应商。
    provider_url = os.getenv("KNOWLEDGE_SEARCH_API_URL")
    # 知识检索不能用虚构数据静默降级，因此缺少地址时立即终止本次工具调用。
    if not provider_url:
        # 抛出可诊断的配置错误，供 Agent 工具失败策略处理。
        raise RuntimeError("KNOWLEDGE_SEARCH_API_URL 未配置，知识库搜索工具不能执行")
    # 透传结构化检索参数，并设置有限超时保护主链路。
    response = httpx.post(provider_url, json=arguments, timeout=15)
    # 非成功 HTTP 状态直接转为异常，不把错误响应误当检索证据。
    response.raise_for_status()
    # 将成功响应解析为 Python 对象以便校验工具契约。
    data = response.json()
    # 统一工具结果只接受对象类型，防止下游读取字段时出现隐式错误。
    if not isinstance(data, dict):
        # 返回结构不合约时显式失败，避免未经验证的数据进入知识融合。
        raise RuntimeError("知识库搜索服务返回的 JSON 顶层不是对象")
    # 返回已经过顶层结构校验的知识检索结果。
    return data
