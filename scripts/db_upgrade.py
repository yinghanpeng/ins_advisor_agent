"""执行带版本台账和校验和保护的 SQL migration。

该脚本读取 DATABASE_URL，并顺序执行 migrations/*.sql。每个 migration 只执行一次；
已执行文件如果被修改会立即失败，避免不同实例持有不一致的数据库结构。
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from sqlalchemy import create_engine, text


def main() -> None:
    """在 PostgreSQL advisory lock 内按顺序执行尚未应用的迁移。"""
    # DATABASE_URL 只从环境变量读取，避免把生产凭据写进脚本或命令参数。
    database_url = os.getenv("DATABASE_URL")
    # 没有明确连接配置时停止，禁止回退到任何隐式默认数据库。
    if not database_url:
        # 抛出可操作的配置错误，让发布流程在执行 DDL 前失败。
        raise RuntimeError("请先设置 DATABASE_URL")
    # 创建短生命周期 Engine；migration 结束后会显式释放连接池。
    engine = create_engine(database_url, future=True)
    # 文件名排序即迁移顺序，因此 migration 文件必须使用递增数字前缀。
    migrations = sorted(Path("migrations").glob("*.sql"))
    # 没有迁移文件通常意味着工作目录错误或发布包不完整，应直接终止。
    if not migrations:
        # 抛出明确错误而不是成功退出，避免发布系统误认为数据库已升级。
        raise RuntimeError("未找到 migrations/*.sql")
    # 外层异常边界确保无论连接或 DDL 哪一步失败都最终释放 Engine。
    try:
        # 使用显式 Connection 管理 advisory lock 和多次独立事务。
        with engine.connect() as connection:
            # 全局 advisory lock 防止多个发布实例同时执行同一批 DDL。
            connection.execute(text("SELECT pg_advisory_lock(hashtext('ins_advisor_agent_migrations'))"))
            # 内层异常边界确保拿到 advisory lock 后必定执行解锁逻辑。
            try:
                # 旧消息加密迁移从 Secret 环境变量读取密钥，只设置到当前数据库 Session。
                migration_key = os.getenv("MEMORY_ENCRYPTION_KEY", "")
                # 把迁移密钥设为当前 Session 参数，避免插入 SQL 字符串或持久化到表中。
                connection.execute(
                    text("SELECT set_config('app.migration_encryption_key', :key, false)"),
                    {"key": migration_key},
                )
                # migration 台账不启用业务 RLS，只保存版本、校验和和应用时间。
                connection.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version text PRIMARY KEY,
                            checksum text NOT NULL,
                            applied_at timestamptz NOT NULL DEFAULT now()
                        )
                        """
                    )
                )
                # 提交台账初始化事务，避免后续 connection.begin 与隐式事务冲突。
                connection.commit()
                # 按文件名顺序逐个核验并应用迁移，保证部署顺序可复现。
                for migration in migrations:
                    # 统一按 UTF-8 读取并计算内容校验和，文件改动可被可靠识别。
                    sql = migration.read_text(encoding="utf-8")
                    # SHA-256 校验和用于检测已应用迁移文件被原地篡改。
                    checksum = sha256(sql.encode("utf-8")).hexdigest()
                    # 以完整文件名作为稳定版本键，与 schema_migrations 主键一致。
                    version = migration.name
                    # 查询该版本既有校验和；不存在表示迁移尚未应用。
                    row = connection.execute(
                        text("SELECT checksum FROM schema_migrations WHERE version=:version"),
                        {"version": version},
                    ).mappings().one_or_none()
                    # SQLAlchemy 2.x 的 SELECT 也会开启事务；先结束只读事务再进入 DDL 事务。
                    connection.commit()
                    # 已存在台账记录时只校验文件一致性，不重复执行 DDL。
                    if row:
                        # 已执行 migration 绝不能原地修改；应新增下一号 migration。
                        if row["checksum"] != checksum:
                            # 校验和不一致立即终止，阻止不同实例形成不一致结构历史。
                            raise RuntimeError(f"migration checksum mismatch: {version}")
                        # 输出可观测跳过信息，便于发布日志确认幂等行为。
                        print(f"skipped {migration} (already applied)")
                        # 当前版本已应用且一致，继续检查下一份迁移文件。
                        continue
                    # 每个 migration 与台账写入在同一事务提交，失败不会留下半套结构。
                    with connection.begin():
                        # 执行仓库内固定迁移 SQL；来源不是运行时用户输入。
                        connection.execute(text(sql))
                        # 在同一事务写入版本和校验和，保证 DDL 与台账原子一致。
                        connection.execute(
                            text(
                                "INSERT INTO schema_migrations (version, checksum) "
                                "VALUES (:version, :checksum)"
                            ),
                            {"version": version, "checksum": checksum},
                        )
                    # 事务成功后记录已应用版本，失败则不会执行此提示。
                    print(f"applied {migration}")
            # 无论前序逻辑成功或失败都执行资源清理，避免连接或执行器泄漏。
            finally:
                # 无论迁移成功或失败都释放 advisory lock，避免阻塞后续发布。
                connection.execute(
                    text("SELECT pg_advisory_unlock(hashtext('ins_advisor_agent_migrations'))")
                )
                # 提交解锁语句所在事务，立即释放数据库级发布互斥锁。
                connection.commit()
    # 无论前序逻辑成功或失败都执行资源清理，避免连接或执行器泄漏。
    finally:
        # 迁移进程退出前关闭连接池，不在一次性脚本中残留数据库连接。
        # 关闭连接池中的全部连接，让一次性迁移进程干净退出。
        engine.dispose()


# 作为命令行脚本直接运行时才开始数据库升级，导入模块不会产生 DDL。
if __name__ == "__main__":
    # 进入统一升级入口，依次执行配置校验、加锁、迁移和资源释放。
    main()
