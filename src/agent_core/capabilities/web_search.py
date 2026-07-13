"""公开网页搜索 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的网页搜索服务。"""
    # 搜索服务地址通过环境变量注入，便于在不同环境使用不同供应商。
    provider_url = os.getenv("WEB_SEARCH_API_URL")
    # 未配置数据源时不允许生成伪搜索结果。
    if not provider_url:
        # 抛出可诊断的配置异常，让 Agent 决定向用户说明或降级。
        raise RuntimeError("WEB_SEARCH_API_URL 未配置，网页搜索工具不能执行")
    # 把查询、数量与过滤条件发送给搜索服务，并设置有限超时。
    response = httpx.post(provider_url, json=arguments, timeout=15)
    # 非成功 HTTP 状态不能被当作搜索结果，统一转为异常。
    response.raise_for_status()
    # 解析供应商返回的 JSON 结果。
    data = response.json()
    # 工具契约要求顶层为对象，避免知识融合层面对不稳定结构。
    if not isinstance(data, dict):
        # 结果结构不合约时明确失败，阻止未经验证的证据进入生成流程。
        raise RuntimeError("网页搜索服务返回的 JSON 顶层不是对象")
    # 返回已通过顶层类型校验的搜索结果。
    return data
