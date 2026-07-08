"""执行本地 SQL migration。

该脚本读取 DATABASE_URL，并顺序执行 migrations/*.sql。生产环境可以换 Alembic，
但这个入口足够支持本地 `make db-upgrade` 和 CI 初始化数据库。
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("请先设置 DATABASE_URL")
    engine = create_engine(database_url, future=True)
    migrations = sorted(Path("migrations").glob("*.sql"))
    if not migrations:
        raise RuntimeError("未找到 migrations/*.sql")
    with engine.begin() as connection:
        for migration in migrations:
            connection.execute(text(migration.read_text(encoding="utf-8")))
            print(f"applied {migration}")


if __name__ == "__main__":
    main()
