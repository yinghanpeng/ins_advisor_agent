"""Memory policy."""

# 文件说明：
# - 本文件属于 Memory 层，负责 session/task/preference 分层记忆和策略。
# - 生产环境需要替换为带租户隔离的持久化存储。
MEMORY_POLICY = {
    "store_sensitive_personal_data": False,
    "store_sales_profiles": True,
    "require_tenant_boundary": True,
}

