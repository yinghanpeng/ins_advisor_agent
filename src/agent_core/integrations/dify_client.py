"""Dify Workflow HTTP client."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations

import os

import httpx


class DifyClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """初始化 Dify API 配置。"""
        self.base_url = base_url or os.getenv("DIFY_BASE_URL")
        self.api_key = api_key or os.getenv("DIFY_API_KEY")

    def call_workflow(self, payload: dict) -> dict:
        """调用 Dify workflow。"""
        if not self.base_url or not self.api_key:
            raise RuntimeError("DIFY_BASE_URL 或 DIFY_API_KEY 未配置，无法调用 Dify workflow")
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/workflows/run",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Dify workflow 返回的 JSON 顶层不是对象")
        return data
