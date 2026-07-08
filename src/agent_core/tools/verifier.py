"""工具结果校验。

ToolResultVerifier 只检查结构和安全边界，不替代 Grounding。Grounding 负责判断最终回答是否
被证据支持；Verifier 负责判断工具返回能否被 workflow 消费。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_core.tools.schemas import ToolSpec


class ToolVerificationResult(BaseModel):
    """工具结果校验结论。"""

    ok: bool = Field(..., description="工具结果是否通过结构校验。")
    errors: list[str] = Field(default_factory=list, description="结构校验错误列表。")


class ToolResultVerifier:
    """基于 ToolSpec.output_schema 做轻量结构校验。"""

    def verify(self, spec: ToolSpec, output: dict[str, Any]) -> ToolVerificationResult:
        """校验工具输出是否满足最基本的 JSON Schema 约束。"""
        schema = spec.output_schema or {}
        errors: list[str] = []
        if schema.get("type") == "object" and not isinstance(output, dict):
            errors.append("工具输出不是 object")
        required = schema.get("required") or []
        for field_name in required:
            if field_name not in output:
                errors.append(f"工具输出缺少必需字段：{field_name}")
        if "_source_boundary" not in output:
            errors.append("工具结果缺少 source boundary，不能进入上下文")
        return ToolVerificationResult(ok=not errors, errors=errors)
