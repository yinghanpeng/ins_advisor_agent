"""清理本地开发数据库表。

该脚本只用于本地重置，不应在生产环境执行。生产降级应使用 Alembic 版本化脚本。
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text


# 按外键依赖从业务事件表向运行主表逆序删除，仅用于明确授权的本地数据库重置。
DROP_SQL = """
DROP TABLE IF EXISTS feedback_events CASCADE;
DROP TABLE IF EXISTS generated_outputs CASCADE;
DROP TABLE IF EXISTS human_approval_requests CASCADE;
DROP TABLE IF EXISTS tool_results CASCADE;
DROP TABLE IF EXISTS tool_calls CASCADE;
DROP TABLE IF EXISTS rag_chunk_embeddings CASCADE;
DROP TABLE IF EXISTS rag_chunks CASCADE;
DROP TABLE IF EXISTS rag_documents CASCADE;
DROP TABLE IF EXISTS memory_recall_results CASCADE;
DROP TABLE IF EXISTS memory_recall_decisions CASCADE;
DROP TABLE IF EXISTS long_term_memory_items CASCADE;
DROP TABLE IF EXISTS task_memory CASCADE;
DROP TABLE IF EXISTS short_term_messages CASCADE;
DROP TABLE IF EXISTS state_transitions CASCADE;
DROP TABLE IF EXISTS agent_trace_events CASCADE;
DROP TABLE IF EXISTS agent_runs CASCADE;
"""


def main() -> None:
    """读取数据库连接并在单个事务中清理本地开发表。"""
    # 连接串只从环境变量读取，避免把凭据固化在一次性维护脚本中。
    database_url = os.getenv("DATABASE_URL")
    # 缺少连接配置时立即停止，防止误连默认数据库或产生不确定行为。
    if not database_url:
        # 用明确异常提示操作者先补齐必需环境变量。
        raise RuntimeError("请先设置 DATABASE_URL")
    # 创建短生命周期 Engine；future=True 使用 SQLAlchemy 2.x 事务语义。
    engine = create_engine(database_url, future=True)
    # 在一个 begin 事务中执行整段 DDL，异常时由 SQLAlchemy 回滚。
    with engine.begin() as connection:
        # 将固定 SQL 包装为 text 执行，不拼接任何用户输入。
        connection.execute(text(DROP_SQL))
    # 事务成功提交后输出完成提示，避免失败时误报清理成功。
    print("local database tables dropped")


# 仅直接运行脚本时执行 main，作为模块导入不会触发破坏性 DDL。
if __name__ == "__main__":
    # 调用唯一脚本入口，保证配置校验与事务逻辑不会被绕过。
    main()
