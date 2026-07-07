"""Dify client adapter placeholder."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations


class DifyClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """初始化 Dify API 配置；本地未配置时保持 mock 模式。"""
        self.base_url = base_url
        self.api_key = api_key

    def call_workflow(self, payload: dict) -> dict:
        """调用 Dify workflow 的预留接口；当前返回 mock 结果方便离线测试。"""
        return {"mode": "mock", "payload": payload}
