"""Model routing policy."""

# 文件说明：
# - 本文件属于成本控制层，负责 token budget、预算决策或模型路由。
# - 预算压力下应压缩上下文、减少工具调用或降级输出。
from __future__ import annotations


def choose_model(task_complexity: str, budget_pressure: bool = False) -> str:
    """根据预算压力与任务复杂度选择模型档位。"""

    # 预算压力具有最高优先级：即使任务复杂也先使用低成本快速模型。
    if budget_pressure:
        # 返回逻辑模型别名，具体供应商型号由配置层映射。
        return "small-fast-model"
    # 无预算压力时，高复杂度任务需要更强的推理模型。
    if task_complexity == "high":
        # 返回推理模型档位以换取更高的复杂任务正确率。
        return "reasoning-model"
    # 普通任务默认使用通用对话模型，在质量与成本之间保持平衡。
    return "default-chat-model"
