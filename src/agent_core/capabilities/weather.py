"""天气工具 provider wrapper。"""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

import os

import httpx


def run(arguments: dict) -> dict:
    """调用配置的天气服务。"""
    # 天气供应商地址由部署环境提供，使业务代码不绑定具体服务。
    provider_url = os.getenv("WEATHER_API_URL")
    # 缺少供应商地址时无法取得实时天气，必须阻止虚构或过期结果进入回答。
    if not provider_url:
        # 以配置异常结束工具调用，让上游选择解释失败或其它安全降级。
        raise RuntimeError("WEATHER_API_URL 未配置，天气工具不能执行")
    # 天气查询参数适合放在 URL 查询串中，并设置十秒上限保护响应延迟。
    response = httpx.get(provider_url, params=arguments, timeout=10)
    # 对非成功 HTTP 状态显式失败，不继续解析供应商错误页。
    response.raise_for_status()
    # 将成功响应解析成 JSON 数据供工具契约校验。
    data = response.json()
    # 统一天气结果要求顶层对象，才能稳定读取温度、城市和时间等字段。
    if not isinstance(data, dict):
        # 供应商结构异常时停止调用，避免不受控数据进入用户答案。
        raise RuntimeError("天气服务返回的 JSON 顶层不是对象")
    # 返回已通过顶层结构校验的实时天气数据。
    return data
