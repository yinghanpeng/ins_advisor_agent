"""意图识别的模型分类器 + 关键词兜底。

# 文件说明：
# - 意图识别是整条主链路的"路由总闸"：分错方向后面全错。
# - 因此这里采用"模型优先、规则兜底"的策略：
#     1. 若配置了可用的 intent_classifier 模型，则调用真实模型做结构化分类；
#     2. 模型未配置 / 调用失败 / 输出不合法 / 置信度过低 时，返回 None，由调用方回退到关键词规则；
# - 任何异常都不会向上抛出，保证本地 demo、测试和生产在模型不可用时仍能稳定运行。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field

from agent_core.config.runtime import RuntimeSettings, load_runtime_settings
from agent_core.models.client import OpenAICompatibleChatClient


# 允许的意图集合：必须与关键词兜底 _classify_intent_by_rules 保持一致，避免模型返回未知意图。
ALLOWED_INTENTS: set[str] = {
    "weather_query",
    "calculator_query",
    "web_or_news_search",
    "insurance_advisor_help",
    "general_chat",
}

# 允许的能力路由：general 走通用能力/工具，domain 走业务 Skill。
ALLOWED_ROUTES: set[str] = {"general", "domain"}


class ModelIntentDecision(BaseModel):
    """模型意图分类的结构化输出。"""

    # 意图标签，必须落在 ALLOWED_INTENTS 内。
    intent: str = Field(..., description="用户意图标签。")
    # 能力路由：general 或 domain。
    capability_route: str = Field(..., description="能力路由：general 或 domain。")
    # 命中的业务 Skill（domain 时给出，例如 insurance_advisor）。
    domain_skill: str | None = Field(default=None, description="命中的业务 Skill 名称。")
    # 模型自评置信度，低于阈值时视为不可信并回退规则。
    confidence: float = Field(default=0.0, description="意图分类置信度，取值 0~1。")


# 系统提示词：约束模型只输出规定的意图与路由，且必须是 JSON。
_SYSTEM_PROMPT = (
    "你是保险顾问 Agent 的意图路由器。请判断用户输入的意图并输出 JSON。\n"
    "intent 只能是以下之一：\n"
    "- weather_query：查询天气；\n"
    "- calculator_query：数学计算；\n"
    "- web_or_news_search：联网搜索、查新闻、查融资/最新动态；\n"
    "- insurance_advisor_help：保险客户沟通、破冰、KYC、异议处理、计划书、成交等；\n"
    "- general_chat：其它普通对话。\n"
    "capability_route：insurance_advisor_help 用 domain，其余用 general。\n"
    "domain_skill：当 capability_route=domain 时填 insurance_advisor，否则为 null。\n"
    "confidence：0~1 之间的置信度。\n"
    '只输出 JSON，例如 {"intent":"weather_query","capability_route":"general",'
    '"domain_skill":null,"confidence":0.9}'
)


@lru_cache(maxsize=4)
def _load_settings_cached(config_dir: str) -> RuntimeSettings:
    """带缓存地加载运行配置，避免每次请求都读盘。"""
    return load_runtime_settings(config_dir)


def _resolve_chat_client(settings: RuntimeSettings) -> OpenAICompatibleChatClient:
    """解析意图分类使用的模型客户端；优先 intent_classifier，缺失则退回 fast_reasoning。

    require_model 在配置不完整（如本地未设置 API Key）时会抛错，
    由外层 classify_intent_via_model 统一捕获并回退到关键词规则。
    """
    try:
        # 优先使用专用的意图分类模型端点。
        config = settings.require_model("intent_classifier")
    except Exception:
        # 没有专用端点时退回到轻量快速推理模型。
        config = settings.require_model("fast_reasoning")
    return OpenAICompatibleChatClient(config)


def classify_intent_via_model(
    text: str,
    *,
    config_dir: str = "configs",
    min_confidence: float = 0.55,
) -> ModelIntentDecision | None:
    """尝试用模型对用户输入做意图分类；不可用时返回 None 让调用方回退规则。

    Args:
        text: 用户原始输入。
        config_dir: 配置目录，测试可覆盖。
        min_confidence: 置信度阈值，低于此值视为不可信。

    Returns:
        合法且置信度达标的 ModelIntentDecision；否则返回 None。
    """
    # 空输入没有分类价值，直接回退规则。
    if not text or not text.strip():
        return None
    try:
        # 读取配置并解析模型客户端；未配置真实模型时会在这里抛错并被捕获。
        settings = _load_settings_cached(config_dir)
        client = _resolve_chat_client(settings)
        # 调用模型并用 Pydantic 校验结构化 JSON 输出。
        decision, _result = client.complete_json(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            schema_model=ModelIntentDecision,
        )
        # 意图或路由不在白名单内，说明模型跑偏，回退规则更安全。
        if decision.intent not in ALLOWED_INTENTS:
            return None
        if decision.capability_route not in ALLOWED_ROUTES:
            return None
        # 置信度不足时不信任模型结果，交回关键词规则兜底。
        if decision.confidence < min_confidence:
            return None
        # domain 路由但没给 Skill 时补默认，保证下游领域链路可用。
        if decision.capability_route == "domain" and not decision.domain_skill:
            decision.domain_skill = "insurance_advisor"
        return decision
    except Exception:
        # 任何异常（配置缺失、网络失败、JSON 非法等）都不外抛，统一回退规则。
        return None
