"""真实模型客户端。

本文件只实现真实 HTTP 调用，不提供本地答案、内置样例或规则替代输出。
这样做的原因很简单：生产级 Agent 的关键节点必须可观测、可追踪、可计费，
如果模型节点在配置缺失时悄悄返回本地结果，后续 trace、eval 和合规审计都会失真。
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any, TypeVar

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from agent_core.config.runtime import ModelEndpointConfig


# 泛型 T 约束为 Pydantic 模型，使 complete_json 能保留调用方传入 schema 的返回类型。
T = TypeVar("T", bound=BaseModel)

# MODEL_TRACE_SINK 保存当前请求的模型观测回调；ContextVar 可隔离 FastAPI 并发请求。
MODEL_TRACE_SINK: ContextVar[Callable[[str, dict[str, Any]], None] | None] = ContextVar(
    "model_trace_sink",
    default=None,
)


@contextmanager
def bind_model_trace_sink(
    sink: Callable[[str, dict[str, Any]], None] | None,
) -> Iterator[None]:
    """在当前请求上下文绑定模型调用 Trace Sink，并在退出时恢复旧值。"""

    # set 返回 Token，确保嵌套调用或并发请求不会覆盖其它请求的 Sink。
    token = MODEL_TRACE_SINK.set(sink)
    # 调用方的完整 Agent 图在该上下文内执行，所有模型客户端都能读取同一 Sink。
    try:
        # yield 把执行权交回 WorkflowEngine，同时保持 ContextVar 绑定有效。
        yield
    # 无论业务成功还是异常，都恢复进入上下文前的值，避免请求间串 Trace。
    finally:
        # reset 使用精确 Token 恢复旧上下文，而不是粗暴写成 None。
        MODEL_TRACE_SINK.reset(token)


def _emit_model_trace(event: str, payload: dict[str, Any]) -> None:
    """把模型请求或响应发送给当前请求 Sink；未绑定时安全跳过。"""

    # 读取当前上下文的回调；普通离线调用模型客户端时可能没有 WorkflowEngine。
    sink = MODEL_TRACE_SINK.get()
    # 只有请求绑定了 Sink 才生成事件，避免模型客户端反向依赖可观测层。
    if sink is not None:
        # Sink 最终进入 AgentState Trace，再按 LangSmith 数据策略决定是否远程上传。
        sink(event, payload)


def _estimate_token_count(value: Any) -> int:
    """在供应商未返回 usage 时，用中英文字符规则估算非零 Token 数。"""

    # 结构化消息使用稳定 JSON 序列化，确保角色名、字段名和正文都进入估算。
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # CJK 字符在常见 GPT tokenizer 中通常接近每字一个或多个 Token，按每字一个估算。
    cjk_count = sum(1 for character in serialized if "\u4e00" <= character <= "\u9fff")
    # 非 CJK 非空白字符按约四字符一个 Token 估算，覆盖英文、数字和 JSON 标点。
    other_count = sum(
        1
        for character in serialized
        if not character.isspace() and not "\u4e00" <= character <= "\u9fff"
    )
    # 至少返回一个 Token，避免有真实模型调用时 LangSmith 顶部仍显示空值。
    return max(1, cjk_count + (other_count + 3) // 4)


class ChatCompletionResult(BaseModel):
    """一次 Chat Completion 调用的结构化结果。"""

    # 允许字段名称带 model_ 前缀，避免 Pydantic 将 model_name 误判为受保护命名空间。
    model_config = ConfigDict(protected_namespaces=())

    # content 是供应商响应中最终供业务节点消费的文本。
    content: str = Field(..., description="模型返回的主文本内容。")
    # model_name 记录供应商实际返回的模型，便于路由与成本审计。
    model_name: str = Field(..., description="实际调用的模型名称，用于 trace 和成本统计。")
    # latency_ms 保存完整 HTTP 调用耗时，用于性能监控与超时调优。
    latency_ms: int = Field(..., description="模型端到端调用耗时，单位毫秒。")
    # token_input 优先使用供应商输入 Token，缺失时使用带来源标记的本地估算。
    token_input: int = Field(default=0, description="输入 Token 数，优先来自供应商 usage。")
    # token_output 优先使用供应商输出 Token，缺失时使用带来源标记的本地估算。
    token_output: int = Field(default=0, description="输出 Token 数，优先来自供应商 usage。")
    # token_usage_source 区分供应商精确 usage 与缺失时的本地估算。
    token_usage_source: str = Field(
        default="provider",
        description="Token 来源：provider 为供应商 usage，estimated 为本地字符估算。",
    )
    # raw_response 只保留排障所需摘要，避免业务层直接依赖供应商完整响应格式。
    raw_response: dict[str, Any] = Field(default_factory=dict, description="原始响应摘要，用于排障。")


class RerankResult(BaseModel):
    """Reranker 对单个候选文档的排序结果。"""

    # index 指回输入文档列表的位置，供调用方恢复原文及其元数据。
    index: int = Field(..., description="候选文档在输入列表中的下标。")
    # score 是内部统一相关度字段；兼容标准 score 与 AIVue 返回的 relevance_score。
    score: float = Field(
        ...,
        validation_alias=AliasChoices("score", "relevance_score"),
        description="Reranker 返回的相关性分数。",
    )


class OpenAICompatibleChatClient:
    """OpenAI-compatible Chat Completion 客户端。

    每个模型节点都从 `configs/models.yaml` 读取自己的配置。比如 guardrail 节点使用
    `models.guardrail`，长期记忆召回决策使用 `models.memory_recall_decision`。
    """

    def __init__(self, config: ModelEndpointConfig) -> None:
        """保存单节点模型配置，并在首次使用前验证必要连接参数。"""

        # 每个客户端实例绑定一个节点级 endpoint 配置，防止运行中交叉使用模型。
        self.config = config
        # 构造阶段即校验配置，让错误在发出网络请求前暴露。
        self._validate_config()

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        """调用真实 Chat Completion 接口并返回文本结果。"""

        # 在构造请求前记录单调时钟，用于计算不受系统时间回拨影响的端到端耗时。
        started_at = time.perf_counter()
        # 墙上时钟用于 LangSmith LLM Run 的绝对开始时间和首 Token 时间展示。
        started_wall_time = datetime.now(timezone.utc).isoformat()
        # 请求体显式包含模型、消息和采样温度，保持 OpenAI-compatible 协议一致。
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        # 只有调用方要求结构化输出时才写入 response_format，兼容不支持该字段的普通调用。
        if response_format is not None:
            # 原样透传调用方声明的响应格式，例如 json_object。
            payload["response_format"] = response_format

        # 在 HTTP 调用前记录实际模型、完整 messages、温度和 response_format，供远程 Trace 回放。
        _emit_model_trace(
            "model_call_started",
            {
                "provider": self.config.provider,
                "endpoint_path": "/chat/completions",
                "model_request": payload,
                "started_at": started_wall_time,
            },
        )
        # 模型网络错误也需要形成成对的失败事件，便于定位超时、鉴权或供应商协议问题。
        try:
            # 通过统一重试函数调用 chat completions endpoint。
            response = self._post_json("/chat/completions", payload)
        # 失败事件保留异常类型和正文；LangSmith 凭据保护层会再次递归脱敏。
        except Exception as exc:
            # 发送失败详情后继续抛出原异常，不能让 Trace 改变业务恢复语义。
            _emit_model_trace(
                "model_call_failed",
                {
                    "provider": self.config.provider,
                    "model_request": payload,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            # 保持原始异常链，由上层节点决定重试、降级或终止。
            raise
        # choices 是 OpenAI-compatible 协议承载候选回答的顶层数组。
        choices = response.get("choices")
        # 空数组或错误类型都不能产出有效回答，因此在进入业务节点前阻断。
        if not isinstance(choices, list) or not choices:
            # 协议失败也记录完整供应商响应，避免只有网络异常才可观测。
            _emit_model_trace(
                "model_call_failed",
                {
                    "provider": self.config.provider,
                    "model_request": payload,
                    "model_response": response,
                    "exception_type": "ModelProtocolError",
                    "exception_message": "模型响应缺少 choices，无法继续生产链路",
                },
            )
            # 抛出协议错误而不是返回空文本，防止下游把供应商故障当正常回答。
            raise RuntimeError("模型响应缺少 choices，无法继续生产链路")
        # 仅消费第一个候选，并先检查候选是否为对象以防属性访问异常。
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        # message 合法时提取 content；其它结构统一视为缺失。
        content = message.get("content") if isinstance(message, dict) else None
        # 当前业务契约只接受字符串正文，工具调用等其它协议需由专用节点处理。
        if not isinstance(content, str):
            # content 协议错误保留原始响应，便于确认供应商返回了 tool_calls 还是其它结构。
            _emit_model_trace(
                "model_call_failed",
                {
                    "provider": self.config.provider,
                    "model_request": payload,
                    "model_response": response,
                    "exception_type": "ModelProtocolError",
                    "exception_message": "模型响应缺少 message.content，无法解析节点输出",
                },
            )
            # 阻止非字符串内容进入 prompt、记忆或最终回答。
            raise RuntimeError("模型响应缺少 message.content，无法解析节点输出")

        # usage 可选；供应商未返回对象时使用空字典并进入带来源标记的本地估算。
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        # 兼容 OpenAI prompt_tokens 和新式 input_tokens 两套字段。
        reported_input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        # 兼容 OpenAI completion_tokens 和新式 output_tokens 两套字段。
        reported_output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        # 输入侧有供应商数字时精确采用，否则基于完整 messages 做非零估算。
        token_input = (
            int(reported_input_tokens)
            if reported_input_tokens is not None
            else _estimate_token_count(messages)
        )
        # 输出侧有供应商数字时精确采用，否则基于最终 content 做非零估算。
        token_output = (
            int(reported_output_tokens)
            if reported_output_tokens is not None
            else _estimate_token_count(content)
        )
        # 两侧都由供应商提供时标记 provider，任一缺失则明确标记为 estimated。
        token_usage_source = (
            "provider"
            if reported_input_tokens is not None and reported_output_tokens is not None
            else "estimated"
        )
        # 将供应商差异收敛为稳定的领域结果，并保留必要观测字段。
        result = ChatCompletionResult(
            content=content,
            model_name=str(response.get("model") or self.config.model),
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            token_input=token_input,
            token_output=token_output,
            token_usage_source=token_usage_source,
            raw_response={
                "id": response.get("id"),
                "created": response.get("created"),
                "usage": usage,
            },
        )
        # 非流式 Chat Completion 只能在完整响应到达时观测第一个 Token，因此该时间是可测得的 TTFT 上界。
        first_token_time = datetime.now(timezone.utc).isoformat()
        # 成功事件保留实际请求、供应商原始响应和规范化结果，支持 LangSmith 中逐次比对。
        _emit_model_trace(
            "model_call_finished",
            {
                "provider": self.config.provider,
                "model_request": payload,
                "model_response": response,
                "normalized_result": result.model_dump(mode="json"),
                "started_at": started_wall_time,
                "first_token_time": first_token_time,
                "completed_at": first_token_time,
            },
        )
        # 返回已观测的规范化结果，业务节点无需感知 Trace 实现。
        return result

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        schema_model: type[T],
        temperature: float = 0.0,
    ) -> tuple[T, ChatCompletionResult]:
        """调用模型并用 Pydantic 校验结构化 JSON 输出。

        模型输出不合法时抛出异常，由工作流节点记录 schema_validation_error。
        这比静默降级更安全，因为长期记忆、工具规划、合规判断都不能吃不可信 JSON。
        """
        # 请求模型使用 JSON 对象格式，并保留原始调用指标供 trace 使用。
        result = self.complete(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        # JSON 解析和 schema 校验必须作为一个失败域处理，任一步失败都拒绝输出。
        try:
            # 先把模型文本解析为标准 JSON，不接受 Python 字面量等宽松格式。
            data = json.loads(result.content)
            # 再用调用方指定的 Pydantic schema 校验类型、必填项和字段约束。
            parsed = schema_model.model_validate(data)
        # JSON 语法错误或 Schema 不合约都属于同一结构化输出失败边界。
        except (json.JSONDecodeError, ValidationError) as exc:
            # 结构化校验失败事件保留模型正文、目标 Schema 与具体校验错误，支持线上回放修复 Prompt。
            _emit_model_trace(
                "model_output_validation_failed",
                {
                    "schema_name": schema_model.__name__,
                    "model_content": result.content,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            # 统一包装为节点可识别的运行时错误，同时保留原始异常链便于排障。
            raise RuntimeError(f"模型结构化输出校验失败：{exc}") from exc
        # Schema 校验成功后记录完整解析对象，区分供应商原文与业务真正消费的数据。
        _emit_model_trace(
            "model_output_validated",
            {
                "schema_name": schema_model.__name__,
                "parsed_output": parsed.model_dump(mode="json"),
            },
        )
        # 同时返回已校验业务对象与模型调用元数据，避免调用方重复解析。
        return parsed, result

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向指定 Chat endpoint 发送 JSON，并在配置上限内执行同步重试。"""

        # 保存最后一次异常，重试耗尽后用它构造可诊断的最终错误。
        last_error: Exception | None = None
        # max_retries 表示额外重试次数，因此总尝试次数需要加一。
        for attempt in range(self.config.max_retries + 1):
            # 单次请求的连接、状态码与解析错误都进入同一重试策略。
            try:
                # 每次尝试创建并关闭客户端，确保连接资源在异常路径也能释放。
                with httpx.Client(timeout=self.config.timeout_ms / 1000) as client:
                    # 使用节点配置的鉴权信息和 JSON 请求体调用完整 endpoint。
                    response = client.post(
                        self._url(path),
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                # 非 2xx 状态转换为 HTTP 异常，参与统一重试。
                response.raise_for_status()
                # 仅对成功响应解析 JSON，解析异常同样可以重试。
                data = response.json()
                # 客户端契约要求顶层对象，防止业务代码依赖不可预测的数组或标量。
                if not isinstance(data, dict):
                    # 返回结构错误视为供应商协议故障，允许在额度内再次尝试。
                    raise RuntimeError("模型服务返回的 JSON 顶层不是对象")
                # 首次成功且结构合法时立即结束重试循环。
                return data
            # 捕获本次网络、状态码、解析或协议错误，并按配置决定是否继续重试。
            except Exception as exc:
                # 记录当前失败，下一次失败会覆盖为更接近最终状态的原因。
                last_error = exc
                # 已执行完最后一次允许的尝试时停止循环。
                if attempt >= self.config.max_retries:
                    # 跳出循环后由统一出口包装最终错误并保留异常链。
                    break
        # 所有尝试失败后向节点暴露调用失败，不提供虚构或缓存答案。
        raise RuntimeError(f"模型服务调用失败：{last_error}") from last_error

    def _url(self, path: str) -> str:
        """将去除尾斜杠的 base_url 与以斜杠开头的接口路径拼接。"""

        # 统一去掉 base_url 末尾斜杠，避免生成双斜杠 endpoint。
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _validate_config(self) -> None:
        """校验 Chat 客户端支持的供应商类型与必填连接参数。"""

        # 只接受已实现 OpenAI-compatible 协议的 provider 标识。
        if self.config.provider not in {"openai_compatible", "openai_compatible_or_http"}:
            # 未知 provider 可能使用不同协议，必须拒绝而不是盲目发请求。
            raise RuntimeError(f"不支持的模型供应商：{self.config.provider}")
        # base_url、api_key 和 model 任一缺失都无法构造可审计的真实模型请求。
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            # 在构造阶段暴露完整配置要求，避免直到 HTTP 层才出现模糊错误。
            raise RuntimeError("模型配置不完整：base_url、api_key、model 均不能为空")


class OpenAICompatibleEmbeddingClient:
    """OpenAI-compatible Embedding 客户端。"""

    def __init__(self, config: ModelEndpointConfig) -> None:
        """保存 Embedding endpoint 配置并立即校验必填连接参数。"""

        # 实例绑定单一 embedding 模型配置，避免向量维度和供应商在运行中漂移。
        self.config = config
        # 请求发出前验证配置，尽早暴露部署错误。
        self._validate_config()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """调用真实 Embedding 服务，把文本批量转成向量。"""

        # 空批次没有可计算内容，而且部分供应商会返回含糊错误，因此在客户端先阻断。
        if not texts:
            # 要求调用方先完成 query 或 chunk 生成，避免掩盖上游逻辑缺陷。
            raise ValueError("embedding 输入不能为空，调用方需要先完成 query/chunk 生成")
        # 按 OpenAI-compatible 协议组装模型名与批量文本输入。
        payload = {"model": self.config.model, "input": texts}
        # 完整 Trace 记录原始文本批次和模型，便于复盘向量检索召回差异。
        _emit_model_trace(
            "embedding_call_started",
            {
                "provider": self.config.provider,
                "endpoint_path": "/embeddings",
                "embedding_request": payload,
            },
        )
        # Embedding 网络或供应商错误形成独立失败事件后继续抛给检索层。
        try:
            # 通过统一 HTTP 重试入口调用 embeddings endpoint。
            response = self._post_json("/embeddings", payload)
        # 捕获完整调用失败，事件正文最终仍经过 LangSmith 凭据清理。
        except Exception as exc:
            # 记录请求、异常类型和异常正文，便于区分连接失败与供应商错误。
            _emit_model_trace(
                "embedding_call_failed",
                {
                    "provider": self.config.provider,
                    "embedding_request": payload,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            # 不改变原有失败语义，由意图/RAG 层决定是否重试或终止。
            raise
        # data 数组按输入顺序承载每条文本的向量结果。
        rows = response.get("data")
        # 响应必须一一对应输入文本，否则下游无法可靠关联文档与向量。
        if not isinstance(rows, list) or len(rows) != len(texts):
            # 条数不一致时拒绝整批结果，防止向量错位污染检索索引。
            raise RuntimeError("Embedding 响应条数与输入文本数不一致")
        # 使用独立结果列表收集完成类型和维度校验的浮点向量。
        embeddings: list[list[float]] = []
        # 逐条验证供应商结果，任一非法向量都会使整批调用失败。
        for row in rows:
            # 只有对象行才读取 embedding 字段，其它结构统一视为非法。
            vector = row.get("embedding") if isinstance(row, dict) else None
            # 向量必须是纯数值列表，字符串、空对象等都不能参与相似度计算。
            if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
                # 显式拒绝非法向量，避免数值运算阶段才以难定位的类型错误失败。
                raise RuntimeError("Embedding 响应中存在非法向量")
            # 配置声明维度时严格校验，防止模型切换后与既有向量库不兼容。
            if self.config.dimensions is not None and len(vector) != self.config.dimensions:
                # 抛出期望值与实际值，帮助部署人员快速定位模型配置错误。
                raise RuntimeError(
                    f"Embedding 维度不匹配：期望 {self.config.dimensions}，实际 {len(vector)}"
                )
            # 将 int/float 统一转成 float，保证后续相似度实现获得稳定元素类型。
            embeddings.append([float(value) for value in vector])
        # 成功事件保留供应商原始向量响应和规范化向量，完整模式可直接比较模型版本差异。
        _emit_model_trace(
            "embedding_call_finished",
            {
                "provider": self.config.provider,
                "embedding_request": payload,
                "embedding_response": response,
                "normalized_embeddings": embeddings,
            },
        )
        # 保持供应商返回顺序，向调用方返回与输入文本一一对应的向量列表。
        return embeddings

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向 Embedding endpoint 发送 JSON，并按配置执行有限次数重试。"""

        # 保存最终一次失败原因，供重试耗尽后的异常链使用。
        last_error: Exception | None = None
        # 总尝试次数等于首次请求加 max_retries 次额外重试。
        for attempt in range(self.config.max_retries + 1):
            # 网络、状态码、JSON 与响应契约错误都纳入相同的重试边界。
            try:
                # 使用上下文管理器确保单次尝试完成后关闭 HTTP 客户端资源。
                with httpx.Client(timeout=self.config.timeout_ms / 1000) as client:
                    # 携带 Bearer 鉴权与 JSON 内容类型调用配置的 embedding 服务。
                    response = client.post(
                        f"{self.config.base_url.rstrip('/')}{path}",
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                # 非 2xx 响应转换为异常，以便重试或统一失败。
                response.raise_for_status()
                # 仅在 HTTP 成功后解析响应 JSON。
                data = response.json()
                # Embedding 协议要求顶层为对象，业务层不处理数组或标量响应。
                if not isinstance(data, dict):
                    # 顶层结构异常视为供应商协议故障，可在剩余次数内重试。
                    raise RuntimeError("Embedding 服务返回的 JSON 顶层不是对象")
                # 请求成功且结构合法时立即返回，停止后续尝试。
                return data
            # 捕获单次 Embedding 调用失败，保留根因并进入有限重试路径。
            except Exception as exc:
                # 记录当前异常，确保最终错误包含最近一次失败上下文。
                last_error = exc
                # 到达配置的最后一次尝试后退出重试循环。
                if attempt >= self.config.max_retries:
                    # 由循环后的统一出口包装错误，避免多处重复 raise 文案。
                    break
        # 重试全部耗尽后明确通知调用方，且保留底层异常链用于排障。
        raise RuntimeError(f"Embedding 服务调用失败：{last_error}") from last_error

    def _validate_config(self) -> None:
        """校验 Embedding 客户端发起真实请求所需的配置字段。"""

        # 缺少地址、密钥或模型名时无法得到可审计的真实向量。
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            # 构造阶段立即失败，防止运行到检索链路才出现模糊网络错误。
            raise RuntimeError("Embedding 配置不完整：base_url、api_key、model 均不能为空")


class RerankerClient:
    """HTTP Reranker 客户端。

    Rerank 服务的接口差异较大，本客户端约定请求体为
    `{model, query, documents, top_k}`，响应体为 `{results: [{index, score}]}`；
    同时兼容 AIVue 使用的 `{results: [{index, relevance_score}]}` 响应字段。
    生产接入其它供应商时，只需要新增适配器，不改业务检索代码。
    """

    def __init__(self, config: ModelEndpointConfig) -> None:
        """保存 Reranker endpoint 配置并立即验证连接所需字段。"""

        # 每个实例绑定一个 reranker 模型，保证排序分数来源可追踪。
        self.config = config
        # 在首次调用前校验配置，避免把部署错误延迟到知识检索中途。
        self._validate_config()

    def rerank(self, *, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        """调用真实 reranker 服务，返回候选文档排序结果。"""

        # 去除空白后仍为空的 query 没有排序语义，应由调用方先完成查询理解。
        if not query.strip():
            # 直接拒绝无效 query，避免消耗外部服务额度并得到不可解释排名。
            raise ValueError("reranker query 不能为空")
        # 没有候选文档时无需访问外部服务，可确定性返回空排名。
        if not documents:
            # 显式声明元素类型，保持空分支与正常分支的返回契约一致。
            ranked: list[RerankResult] = []
            # 立即返回空列表，避免供应商对空 documents 的非标准行为。
            return ranked
        # 按约定协议组装模型、查询、候选文档及所需返回数量。
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_k": top_k,
        }
        # 完整 Trace 记录 query、候选文档和 top_k，便于解释最终知识排序。
        _emit_model_trace(
            "reranker_call_started",
            {
                "provider": self.config.provider,
                "endpoint_path": "/rerank",
                "reranker_request": payload,
            },
        )
        # Reranker 调用失败时记录请求和异常，再维持原有上抛语义。
        try:
            # 通过统一重试入口调用 rerank endpoint。
            response = self._post_json("/rerank", payload)
        # 网络、状态码、解析和供应商协议失败统一进入事件。
        except Exception as exc:
            # 异常正文可能包含供应商响应，远程投影器会递归清除认证凭据。
            _emit_model_trace(
                "reranker_call_failed",
                {
                    "provider": self.config.provider,
                    "reranker_request": payload,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            # 保留原始异常链供上层知识检索恢复策略使用。
            raise
        # results 应包含文档下标与相关度分数的对象列表。
        raw_results = response.get("results")
        # 缺失或非列表 results 无法建立可靠排序，必须阻断。
        if not isinstance(raw_results, list):
            # 抛出协议错误而不是将无结果误判为零相关度。
            raise RuntimeError("Reranker 响应缺少 results")
        # 用 Pydantic 逐项验证下标和分数类型，拒绝不合约的供应商结果。
        ranked = [RerankResult.model_validate(item) for item in raw_results]
        # 成功事件同时保留供应商响应和规范化排序，便于定位字段适配或截断差异。
        _emit_model_trace(
            "reranker_call_finished",
            {
                "provider": self.config.provider,
                "reranker_request": payload,
                "reranker_response": response,
                "normalized_results": [item.model_dump(mode="json") for item in ranked[:top_k]],
            },
        )
        # 即使供应商返回更多结果，也只暴露调用方请求的 top_k 条。
        return ranked[:top_k]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向 Reranker endpoint 发送 JSON，并按节点配置执行同步重试。"""

        # 保存最近一次异常，作为所有尝试失败后的根因。
        last_error: Exception | None = None
        # 首次请求与配置的额外重试次数共同构成完整尝试范围。
        for attempt in range(self.config.max_retries + 1):
            # 将连接、状态、解析和契约错误统一纳入单次尝试。
            try:
                # 上下文管理器保证每次尝试结束后释放 HTTP 连接资源。
                with httpx.Client(timeout=self.config.timeout_ms / 1000) as client:
                    # 使用配置的 Bearer 密钥向 reranker 服务发送 JSON 请求。
                    response = client.post(
                        f"{self.config.base_url.rstrip('/')}{path}",
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                # 对非 2xx 响应抛错，使其进入剩余重试或最终失败路径。
                response.raise_for_status()
                # HTTP 成功后解析供应商 JSON 响应。
                data = response.json()
                # 顶层必须为对象，才能可靠读取 results 字段。
                if not isinstance(data, dict):
                    # 结构不合约时视为供应商故障，不向业务层泄漏异常格式。
                    raise RuntimeError("Reranker 服务返回的 JSON 顶层不是对象")
                # 请求及契约校验成功后立即返回结果对象。
                return data
            # 捕获单次 Reranker 调用失败，统一记录后按重试预算继续或退出。
            except Exception as exc:
                # 捕获并保存本次失败原因，允许按配置继续尝试。
                last_error = exc
                # 已到最后一次尝试时退出循环，避免超过成本与延迟预算。
                if attempt >= self.config.max_retries:
                    # 交给统一出口抛出带根因链的错误。
                    break
        # 所有尝试失败后明确中止 rerank，不使用未排序结果冒充成功。
        raise RuntimeError(f"Reranker 服务调用失败：{last_error}") from last_error

    def _validate_config(self) -> None:
        """校验 Reranker 客户端连接真实服务所需的必填配置。"""

        # 地址、密钥或模型名任一为空都无法构造有效请求。
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            # 构造阶段立即暴露配置缺失，保持检索链路失败原因清晰。
            raise RuntimeError("Reranker 配置不完整：base_url、api_key、model 均不能为空")
