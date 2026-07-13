"""LangSmith trace exporter."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations

import os

import httpx


def export_trace(trace: dict) -> dict:
    """把 trace 发送到配置的 LangSmith endpoint。"""

    # endpoint 由环境注入，以支持官方服务、自建代理或测试环境。
    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    # API key 只从环境读取，避免跟随代码或静态配置进入版本库。
    api_key = os.getenv("LANGSMITH_API_KEY")
    # 导出需要地址和密钥同时存在，否则不应发起匿名或错误目标请求。
    if not endpoint or not api_key:
        # 显式报告配置缺失，让调用方决定是否执行可观测性降级。
        raise RuntimeError("LANGSMITH_ENDPOINT 或 LANGSMITH_API_KEY 未配置，无法导出 trace")
    # 将结构化 trace 发送到 runs endpoint，并限制等待时间避免阻塞业务请求。
    response = httpx.post(
        f"{endpoint.rstrip('/')}/runs",
        headers={"Authorization": f"Bearer {api_key}"},
        json=trace,
        timeout=15,
    )
    # 非成功状态转换为异常，确保调用方能够记录导出失败。
    response.raise_for_status()
    # 解析 LangSmith 成功响应，提取 run 标识等确认信息。
    data = response.json()
    # 导出适配器要求响应顶层对象，拒绝不可解释的列表或标量。
    if not isinstance(data, dict):
        # 对协议不匹配明确失败，防止错误地把 trace 标记为已导出。
        raise RuntimeError("LangSmith 返回的 JSON 顶层不是对象")
    # 返回已校验的远程确认对象。
    return data
