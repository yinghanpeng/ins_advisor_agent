"""天气工具 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的天气服务。"""
    provider_url = os.getenv("WEATHER_API_URL")
    if not provider_url:
        raise RuntimeError("WEATHER_API_URL 未配置，天气工具不能执行")
    response = httpx.get(provider_url, params=arguments, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("天气服务返回的 JSON 顶层不是对象")
    return data
