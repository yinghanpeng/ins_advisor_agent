"""翻译工具 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的翻译服务。"""
    # 翻译供应商地址通过环境变量注入，便于不同部署环境替换实现。
    provider_url = os.getenv("TRANSLATION_API_URL")
    # 没有服务地址就无法提供真实翻译结果，禁止静默返回原文冒充翻译。
    if not provider_url:
        # 抛出明确配置错误，交由工具失败处理逻辑决定用户侧降级文案。
        raise RuntimeError("TRANSLATION_API_URL 未配置，翻译工具不能执行")
    # 将源文本和目标语言等参数发送给外部翻译服务。
    response = httpx.post(provider_url, json=arguments, timeout=15)
    # 对非成功状态显式报错，避免把供应商错误体作为翻译内容。
    response.raise_for_status()
    # 解析成功响应的 JSON 数据。
    data = response.json()
    # 工具协议要求响应顶层为对象，便于稳定读取译文与元数据。
    if not isinstance(data, dict):
        # 供应商违反契约时终止调用，避免下游产生不可解释结果。
        raise RuntimeError("翻译服务返回的 JSON 顶层不是对象")
    # 返回已通过顶层类型校验的翻译结果。
    return data
