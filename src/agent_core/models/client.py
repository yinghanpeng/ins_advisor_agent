"""真实模型客户端。

本文件只实现真实 HTTP 调用，不提供本地答案、内置样例或规则替代输出。
这样做的原因很简单：生产级 Agent 的关键节点必须可观测、可追踪、可计费，
如果模型节点在配置缺失时悄悄返回本地结果，后续 trace、eval 和合规审计都会失真。
"""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_core.config.runtime import ModelEndpointConfig


T = TypeVar("T", bound=BaseModel)


class ChatCompletionResult(BaseModel):
    """一次 Chat Completion 调用的结构化结果。"""

    model_config = ConfigDict(protected_namespaces=())

    content: str = Field(..., description="模型返回的主文本内容。")
    model_name: str = Field(..., description="实际调用的模型名称，用于 trace 和成本统计。")
    latency_ms: int = Field(..., description="模型端到端调用耗时，单位毫秒。")
    token_input: int = Field(default=0, description="模型供应商返回的输入 token 数，缺失时为 0。")
    token_output: int = Field(default=0, description="模型供应商返回的输出 token 数，缺失时为 0。")
    raw_response: dict[str, Any] = Field(default_factory=dict, description="原始响应摘要，用于排障。")


class RerankResult(BaseModel):
    """Reranker 对单个候选文档的排序结果。"""

    index: int = Field(..., description="候选文档在输入列表中的下标。")
    score: float = Field(..., description="Reranker 返回的相关性分数。")


class OpenAICompatibleChatClient:
    """OpenAI-compatible Chat Completion 客户端。

    每个模型节点都从 `configs/models.yaml` 读取自己的配置。比如 guardrail 节点使用
    `models.guardrail`，长期记忆召回决策使用 `models.memory_recall_decision`。
    """

    def __init__(self, config: ModelEndpointConfig) -> None:
        self.config = config
        self._validate_config()

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        """调用真实 Chat Completion 接口并返回文本结果。"""
        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = self._post_json("/chat/completions", payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("模型响应缺少 choices，无法继续生产链路")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("模型响应缺少 message.content，无法解析节点输出")

        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        return ChatCompletionResult(
            content=content,
            model_name=str(response.get("model") or self.config.model),
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            token_input=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            token_output=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            raw_response={
                "id": response.get("id"),
                "created": response.get("created"),
                "usage": usage,
            },
        )

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
        result = self.complete(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(result.content)
            parsed = schema_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeError(f"模型结构化输出校验失败：{exc}") from exc
        return parsed, result

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout_ms / 1000) as client:
                    response = client.post(
                        self._url(path),
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("模型服务返回的 JSON 顶层不是对象")
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
        raise RuntimeError(f"模型服务调用失败：{last_error}") from last_error

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _validate_config(self) -> None:
        if self.config.provider not in {"openai_compatible", "openai_compatible_or_http"}:
            raise RuntimeError(f"不支持的模型供应商：{self.config.provider}")
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            raise RuntimeError("模型配置不完整：base_url、api_key、model 均不能为空")


class OpenAICompatibleEmbeddingClient:
    """OpenAI-compatible Embedding 客户端。"""

    def __init__(self, config: ModelEndpointConfig) -> None:
        self.config = config
        self._validate_config()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """调用真实 Embedding 服务，把文本批量转成向量。"""
        if not texts:
            raise ValueError("embedding 输入不能为空，调用方需要先完成 query/chunk 生成")
        payload = {"model": self.config.model, "input": texts}
        response = self._post_json("/embeddings", payload)
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise RuntimeError("Embedding 响应条数与输入文本数不一致")
        embeddings: list[list[float]] = []
        for row in rows:
            vector = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
                raise RuntimeError("Embedding 响应中存在非法向量")
            if self.config.dimensions is not None and len(vector) != self.config.dimensions:
                raise RuntimeError(
                    f"Embedding 维度不匹配：期望 {self.config.dimensions}，实际 {len(vector)}"
                )
            embeddings.append([float(value) for value in vector])
        return embeddings

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout_ms / 1000) as client:
                    response = client.post(
                        f"{self.config.base_url.rstrip('/')}{path}",
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Embedding 服务返回的 JSON 顶层不是对象")
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
        raise RuntimeError(f"Embedding 服务调用失败：{last_error}") from last_error

    def _validate_config(self) -> None:
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            raise RuntimeError("Embedding 配置不完整：base_url、api_key、model 均不能为空")


class RerankerClient:
    """HTTP Reranker 客户端。

    Rerank 服务的接口差异较大，本客户端约定请求体为
    `{model, query, documents, top_k}`，响应体为 `{results: [{index, score}]}`。
    生产接入其它供应商时，只需要新增适配器，不改业务检索代码。
    """

    def __init__(self, config: ModelEndpointConfig) -> None:
        self.config = config
        self._validate_config()

    def rerank(self, *, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        """调用真实 reranker 服务，返回候选文档排序结果。"""
        if not query.strip():
            raise ValueError("reranker query 不能为空")
        if not documents:
            ranked: list[RerankResult] = []
            return ranked
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_k": top_k,
        }
        response = self._post_json("/rerank", payload)
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("Reranker 响应缺少 results")
        ranked = [RerankResult.model_validate(item) for item in raw_results]
        return ranked[:top_k]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout_ms / 1000) as client:
                    response = client.post(
                        f"{self.config.base_url.rstrip('/')}{path}",
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Reranker 服务返回的 JSON 顶层不是对象")
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
        raise RuntimeError(f"Reranker 服务调用失败：{last_error}") from last_error

    def _validate_config(self) -> None:
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            raise RuntimeError("Reranker 配置不完整：base_url、api_key、model 均不能为空")
