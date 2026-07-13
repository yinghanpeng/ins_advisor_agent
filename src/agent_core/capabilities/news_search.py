"""新闻搜索 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的新闻搜索服务。"""
    # 新闻服务优先使用专用地址，未配置时允许复用通用网页搜索供应商。
    provider_url = os.getenv("NEWS_SEARCH_API_URL") or os.getenv("WEB_SEARCH_API_URL")
    # 两个候选地址都缺失时没有可信数据源，应停止而不是生成假新闻。
    if not provider_url:
        # 抛出配置错误，让工具层按失败恢复策略向用户说明或降级。
        raise RuntimeError("NEWS_SEARCH_API_URL 未配置，新闻搜索工具不能执行")
    # 将新闻检索条件发送给外部服务，并限制等待时间。
    response = httpx.post(provider_url, json=arguments, timeout=15)
    # 非 2xx 响应不能作为新闻结果，统一转换为异常。
    response.raise_for_status()
    # 解析供应商返回的 JSON 数据。
    data = response.json()
    # 结果顶层必须是字段可寻址的对象，才能进入统一工具协议。
    if not isinstance(data, dict):
        # 契约不匹配时明确失败，避免错误数据进入事实性回答。
        raise RuntimeError("新闻搜索服务返回的 JSON 顶层不是对象")
    # 返回已验证的新闻检索结果对象。
    return data
