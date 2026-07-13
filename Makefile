.PHONY: db-upgrade db-downgrade db-reset-local memory-retention api-dev api-dev-reload reranker-dev test

# API_HOST/API_PORT 可由 make 调用方覆盖，避免本地已有服务占用默认 8000 端口。
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
# RERANKER_HOST/RERANKER_PORT 独立于 Agent API，避免模型加载占用主 Gateway 进程。
RERANKER_HOST ?= 127.0.0.1
RERANKER_PORT ?= 8002

db-upgrade:
	python3 scripts/db_upgrade.py

db-downgrade:
	python3 scripts/db_downgrade.py

db-reset-local: db-downgrade db-upgrade

memory-retention:
	PYTHONPATH=src python3 scripts/memory_retention.py

# 加载本地 .env 后以稳定模式启动 FastAPI 与根路径的 Agent Playground；仅用于开发联调。
api-dev:
	set -a; . ./.env; set +a; PYTHONPATH=src python3 -m uvicorn agent_core.api.server:app --host $(API_HOST) --port $(API_PORT)

# 需要自动重载时显式使用此命令；受限容器或部分企业终端可能不允许文件监听器。
api-dev-reload:
	set -a; . ./.env; set +a; PYTHONPATH=src python3 -m uvicorn agent_core.api.server:app --host $(API_HOST) --port $(API_PORT) --reload

# 启动本地 CrossEncoder Rerank 服务；首次启动会下载 LOCAL_RERANKER_MODEL 对应权重。
reranker-dev:
	set -a; . ./.env; set +a; PYTHONPATH=src python3 -m uvicorn agent_core.reranker_server:app --host $(RERANKER_HOST) --port $(RERANKER_PORT)

test:
	python3 -m pytest
