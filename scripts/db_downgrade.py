"""清理本地开发数据库表。

该脚本只用于本地重置，不应在生产环境执行。生产降级应使用 Alembic 版本化脚本。
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text


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
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("请先设置 DATABASE_URL")
    engine = create_engine(database_url, future=True)
    with engine.begin() as connection:
        connection.execute(text(DROP_SQL))
    print("local database tables dropped")


if __name__ == "__main__":
    main()
