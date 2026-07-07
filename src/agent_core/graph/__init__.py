"""Graph runtime adapters."""

# 文件说明：
# - 本文件属于显式状态机层，负责状态对象、节点函数、边或 checkpoint。
# - 所有复杂任务都应通过状态流转表达，避免把流程藏在大 Prompt 中。
