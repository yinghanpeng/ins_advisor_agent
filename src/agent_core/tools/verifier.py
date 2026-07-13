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

    # ok 是执行器是否允许该工具结果进入上下文的总开关。
    ok: bool = Field(..., description="工具结果是否通过结构校验。")
    # errors 保存全部结构与来源边界问题，便于一次定位 Provider 契约漂移。
    errors: list[str] = Field(default_factory=list, description="结构校验错误列表。")


class ToolInputValidationResult(BaseModel):
    """工具入参校验结论，缺失字段可直接驱动主链路澄清。"""

    # ok 只有在必填字段与局部约束全部通过时才为 True。
    ok: bool = Field(..., description="工具入参是否满足 ToolSpec.input_schema。")
    # missing_fields 与普通错误分开保存，使主链路可针对缺参向用户澄清。
    missing_fields: list[str] = Field(
        default_factory=list,
        description="缺失的必填参数名。",
    )
    # errors 保存类型、长度、枚举和额外字段等不可通过猜测修复的问题。
    errors: list[str] = Field(
        default_factory=list,
        description="类型、长度或枚举等参数错误。",
    )


class ToolInputValidator:
    """执行工具前基于 ToolSpec.input_schema 校验参数。

    当前工具 Schema 只使用 JSON Schema 的稳定子集：object、properties、required、type、
    minLength、enum 和 additionalProperties。保持轻量实现可以避免为几个固定工具引入完整
    JSON Schema 运行时，同时让参数契约真正成为唯一校验来源。
    """

    # JSON Schema 类型映射到受支持的 Python 运行时类型，保持校验器轻量可控。
    _PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    def validate(self, spec: ToolSpec, arguments: dict[str, Any]) -> ToolInputValidationResult:
        """返回结构化错误；调用方决定澄清用户还是以工具错误终止。"""
        # 空 Schema 视为无额外约束；非空 Schema 只解析受支持的稳定子集。
        schema = spec.input_schema or {}
        # errors 保存类型/枚举/长度错误，missing_fields 单独驱动澄清文案。
        errors: list[str] = []
        # 单独累计必填字段缺口，调用方可据此生成确定性的补充问题。
        missing_fields: list[str] = []

        # Schema 声明 object 时拒绝非字典参数，不进行隐式转换。
        if schema.get("type") == "object" and not isinstance(arguments, dict):
            # 顶层类型错误无法继续字段校验，立即返回唯一且明确的错误。
            return ToolInputValidationResult(ok=False, errors=["工具入参必须是 object"])

        # required 缺失时使用空列表，逐字段判断不存在、None 或空字符串。
        required = schema.get("required") or []
        # 每个必填字段独立记录，前端可以一次提示全部缺口。
        for field_name in required:
            # value 只在字段存在时有意义，但 get 便于统一判断 None。
            value = arguments.get(field_name)
            # 不存在、None 和纯空白字符串都属于缺失；数值 0 和布尔 False 仍是有效值。
            if (
                field_name not in arguments
                or value is None
                or (isinstance(value, str) and not value.strip())
            ):
                # 记录字段名而非拼入普通错误，让上层能够结构化处理缺参。
                missing_fields.append(str(field_name))

        # properties 定义允许字段及其局部约束；缺失时为空映射。
        properties = schema.get("properties") or {}
        # 逐个校验调用方实际提交的字段，避免只检查 required 而漏掉错误额外值。
        for field_name, value in arguments.items():
            # 当前字段没有声明时根据 additionalProperties 决定拒绝或忽略。
            field_schema = properties.get(field_name)
            # 未声明字段只有在 Schema 明确 false 时才报错。
            if field_schema is None:
                # additionalProperties=false 形成严格 Tool 参数边界。
                if schema.get("additionalProperties") is False:
                    # 严格 Schema 拒绝模型臆造的额外参数，防止参数走私。
                    errors.append(f"工具入参包含未声明字段：{field_name}")
                # 未声明字段无需再执行类型、长度或枚举检查，继续下一个参数。
                continue
            # 可选字段显式为 None 时跳过其它约束；必填 None 已在上面记录缺失。
            if value is None:
                # 可选空值没有可验证内容；必填空值已由前一阶段记录。
                continue
            # 读取字段声明的 JSON Schema 类型，缺失时跳过类型约束。
            expected_type = field_schema.get("type")
            # 将声明类型映射为 Python 类型；未知类型保持 None 并由未来扩展处理。
            python_type = self._PYTHON_TYPES.get(expected_type)
            # 只对支持的 JSON Schema type 执行 Python 类型映射。
            if python_type is not None:
                # bool 是 int 的子类，number/integer 参数不能因此接受布尔值。
                type_matches = isinstance(value, python_type) and not (
                    expected_type in {"number", "integer"} and isinstance(value, bool)
                )
                # 类型不符时记录错误并跳过该字段后续长度/枚举判断。
                if not type_matches:
                    # 记录期望类型但不回显敏感值，避免错误消息泄露入参内容。
                    errors.append(f"工具入参 {field_name} 类型应为 {expected_type}")
                    # 类型已错误时跳过依赖正确类型的字符串长度和枚举判断。
                    continue
            # 字符串按去空白后的长度检查 minLength，避免仅空格绕过。
            if expected_type == "string" and len(value.strip()) < int(
                field_schema.get("minLength", 0)
            ):
                # 字符串短于最小长度时记录稳定错误，空白已在 strip 后计算。
                errors.append(f"工具入参 {field_name} 长度不足")
            # 读取枚举白名单；未声明 enum 时不限制具体取值。
            allowed_values = field_schema.get("enum")
            # enum 存在时要求精确命中，不能自动大小写转换或模糊匹配。
            if allowed_values is not None and value not in allowed_values:
                # 精确枚举未命中时拒绝，避免执行器自行猜测或纠正业务含义。
                errors.append(f"工具入参 {field_name} 不在允许范围内")

        # 汇总缺参和普通错误；两者都为空时才允许 Runner 执行。
        return ToolInputValidationResult(
            ok=not missing_fields and not errors,
            missing_fields=missing_fields,
            errors=errors,
        )


class ToolResultVerifier:
    """基于 ToolSpec.output_schema 做轻量结构校验。"""

    def verify(self, spec: ToolSpec, output: dict[str, Any]) -> ToolVerificationResult:
        """校验工具输出是否满足最基本的 JSON Schema 约束。"""
        # 空 output_schema 使用空约束，但 source boundary 仍始终必需。
        schema = spec.output_schema or {}
        # 累积全部结构错误，一次返回给执行器。
        errors: list[str] = []
        # Schema 声明 object 时拒绝非字典结果。
        if schema.get("type") == "object" and not isinstance(output, dict):
            # 记录结构错误但继续汇总其它问题，便于一次性排查。
            errors.append("工具输出不是 object")
        # 逐项检查输出必需字段，避免下游读取不存在键。
        required = schema.get("required") or []
        # 每个缺失字段产生独立错误，便于定位 Provider 契约漂移。
        for field_name in required:
            # 必需字段不存在即失败；空值是否允许由各工具 Schema 继续扩展。
            if field_name not in output:
                # 记录具体缺失字段，帮助 Provider 与 ToolSpec 对齐。
                errors.append(f"工具输出缺少必需字段：{field_name}")
        # 所有可进入上下文的工具结果必须携带来源边界，即使 output_schema 未声明。
        if "_source_boundary" not in output:
            # 缺少来源边界意味着外部文本无法安全进入 Prompt，因此强制失败。
            errors.append("工具结果缺少 source boundary，不能进入上下文")
        # 以错误列表是否为空计算总结果，并完整返回错误供执行器包装。
        return ToolVerificationResult(ok=not errors, errors=errors)
