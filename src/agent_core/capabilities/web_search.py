"""公开网页搜索 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的网页搜索服务。"""
    provider_url = os.getenv("WEB_SEARCH_API_URL")
    if not provider_url:
        raise RuntimeError("WEB_SEARCH_API_URL 未配置，网页搜索工具不能执行")
    response = httpx.post(provider_url, json=arguments, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("网页搜索服务返回的 JSON 顶层不是对象")
    return data
