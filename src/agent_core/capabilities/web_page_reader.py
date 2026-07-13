"""网页正文读取 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的网页正文抽取服务。"""
    # 网页正文抽取地址从环境读取，以便按部署环境切换 provider。
    provider_url = os.getenv("WEB_PAGE_READER_API_URL")
    # 没有 provider 时不能可靠读取远端页面，禁止返回未经获取的内容。
    if not provider_url:
        # 抛出明确配置错误，交由上游工具失败恢复策略处理。
        raise RuntimeError("WEB_PAGE_READER_API_URL 未配置，网页读取工具不能执行")
    # 把 URL 与抽取选项发送给正文服务，并限制最长等待时间。
    response = httpx.post(provider_url, json=arguments, timeout=15)
    # 将非 2xx 状态统一转换为异常，避免解析错误响应。
    response.raise_for_status()
    # 解析成功响应中的 JSON 正文与元数据。
    data = response.json()
    # 工具协议要求顶层对象，以便知识融合层读取正文和来源字段。
    if not isinstance(data, dict):
        # 对返回结构不合约显式报错，防止错误页面混入知识证据。
        raise RuntimeError("网页读取服务返回的 JSON 顶层不是对象")
    # 返回已校验顶层类型的网页读取结果。
    return data
