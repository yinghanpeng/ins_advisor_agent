"""Unit converter capability."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations


# 换算表显式列出支持的有向单位对，避免引入不可控的动态表达式计算。
RATES = {
    ("km", "m"): 1000.0,
    ("m", "km"): 0.001,
    ("kg", "g"): 1000.0,
    ("g", "kg"): 0.001,
}


def run(arguments: dict) -> dict:
    """执行本地单位换算，当前只支持 RATES 中声明的单位对。"""
    # 将调用方数值标准化为浮点数，使字符串形式的合法数字也可处理。
    value = float(arguments["value"])
    # 读取源单位，缺失时保留 KeyError 以暴露不完整工具参数。
    source = arguments["from"]
    # 读取目标单位，后续与源单位共同定位白名单换算率。
    target = arguments["to"]
    # 相同单位无需查询换算表，可原值返回并保持目标单位标签。
    if source == target:
        # 返回标准工具对象，避免没有必要的浮点乘法。
        return {"value": value, "unit": target}
    # 以有向单位对查询显式维护的换算系数。
    rate = RATES.get((source, target))
    # 未登记的单位组合不允许推测换算规则。
    if rate is None:
        # 抛出明确错误，要求调用方改用受支持的单位对。
        raise ValueError(f"unsupported conversion: {source}->{target}")
    # 将输入值乘以白名单系数，并用目标单位封装结果。
    return {"value": value * rate, "unit": target}
