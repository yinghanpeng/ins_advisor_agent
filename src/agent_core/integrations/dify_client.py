"""Dify Workflow HTTP client."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations

import os

import httpx


class DifyClient:
    """封装遗留 Dify Workflow HTTP 协议，隔离外部系统调用细节。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """初始化 Dify API 配置。"""

        # 显式参数优先，便于测试注入；未传时再从部署环境读取服务地址。
        self.base_url = base_url or os.getenv("DIFY_BASE_URL")
        # API key 同样允许显式注入，并避免在代码或配置模板中固化密钥。
        self.api_key = api_key or os.getenv("DIFY_API_KEY")

    def call_workflow(self, payload: dict) -> dict:
        """调用 Dify workflow。"""

        # 地址和密钥必须同时存在，否则无法构造可鉴权的真实 workflow 请求。
        if not self.base_url or not self.api_key:
            # 明确抛出配置错误，不使用本地伪结果掩盖外部工作流不可用。
            raise RuntimeError("DIFY_BASE_URL 或 DIFY_API_KEY 未配置，无法调用 Dify workflow")
        # 使用 Bearer 鉴权发送 workflow 输入，并设置三十秒的有限等待时间。
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/workflows/run",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        # 非 2xx 状态直接转换为异常，防止把 Dify 错误体当成业务输出。
        response.raise_for_status()
        # 仅在 HTTP 调用成功后解析 JSON 响应。
        data = response.json()
        # 集成契约要求顶层对象，便于上游稳定读取 outputs 等字段。
        if not isinstance(data, dict):
            # 结构不合约时显式失败，让调用方执行外部依赖降级策略。
            raise RuntimeError("Dify workflow 返回的 JSON 顶层不是对象")
        # 返回已经过顶层类型校验的 workflow 结果。
        return data
