"""受控文件解析工具。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """解析调用方已授权上传的文件内容。"""
    if "content" in arguments:
        return {"text": str(arguments["content"]), "metadata": {"source": "provided_content"}}
    provider_url = os.getenv("FILE_PARSER_API_URL")
    if not provider_url:
        raise RuntimeError("FILE_PARSER_API_URL 未配置，且未提供 content")
    response = httpx.post(provider_url, json=arguments, timeout=20)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("文件解析服务返回的 JSON 顶层不是对象")
    return data
