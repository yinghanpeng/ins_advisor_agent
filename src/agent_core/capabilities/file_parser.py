"""受控文件解析工具。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """解析调用方已授权上传的文件内容。"""
    # 调用方已直接提供文件正文时无需访问外部解析服务，可立即走确定性的本地分支。
    if "content" in arguments:
        # 显式标记数据来自请求正文，便于后续引用与审计判断来源。
        return {"text": str(arguments["content"]), "metadata": {"source": "provided_content"}}
    # 未提供正文时从环境变量读取外部文件解析服务地址。
    provider_url = os.getenv("FILE_PARSER_API_URL")
    # 没有正文也没有服务地址时无法可靠解析，因此直接暴露配置错误。
    if not provider_url:
        # 抛错而不伪造解析结果，防止下游模型把空内容当成真实文件。
        raise RuntimeError("FILE_PARSER_API_URL 未配置，且未提供 content")
    # 将已授权的文件定位信息发送给解析服务，并用有限超时避免阻塞 Agent 主链路。
    response = httpx.post(provider_url, json=arguments, timeout=20)
    # 将非 2xx 状态转换为异常，避免继续消费错误页或供应商错误体。
    response.raise_for_status()
    # 仅在 HTTP 状态正常后解析 JSON 响应。
    data = response.json()
    # 工具契约要求顶层为对象，列表或标量无法合并进统一工具结果。
    if not isinstance(data, dict):
        # 对契约不匹配显式失败，让上游工具恢复逻辑决定是否重试或降级。
        raise RuntimeError("文件解析服务返回的 JSON 顶层不是对象")
    # 返回已通过顶层类型校验的供应商结果。
    return data
