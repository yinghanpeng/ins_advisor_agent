"""Unit converter capability."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations


RATES = {
    ("km", "m"): 1000.0,
    ("m", "km"): 0.001,
    ("kg", "g"): 1000.0,
    ("g", "kg"): 0.001,
}


def run(arguments: dict) -> dict:
    """执行本地单位换算，当前只支持 RATES 中声明的单位对。"""
    value = float(arguments["value"])
    source = arguments["from"]
    target = arguments["to"]
    if source == target:
        return {"value": value, "unit": target}
    rate = RATES.get((source, target))
    if rate is None:
        raise ValueError(f"unsupported conversion: {source}->{target}")
    return {"value": value * rate, "unit": target}
