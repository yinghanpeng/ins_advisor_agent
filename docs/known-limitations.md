# 当前限制

本项目已经具备生产级结构和本地可运行骨架，但仍有一些能力需要接真实服务。

## 1. FastAPI 未安装

当前环境没有安装 FastAPI。代码已经提供 `src/agent_core/api/server.py`，安装可选依赖后即可启动：

```bash
pip install -e ".[api]"
uvicorn agent_core.api.server:app --reload
```

## 2. 模型 provider 未接入

当前生成节点使用本地 deterministic 文案。后续应接入真实 LLM provider，并保持 Pydantic schema 校验。

## 3. 网络工具未接真实 provider

`web_search`、`news_search`、`weather`、`file_parser` 当前是 adapter/mock。后续应接真实 API，并补 timeout、retry、error schema。

## 4. LangSmith 远程 trace 需要配置

当前 LangSmith 是 adapter。需要 API Key、网络和 project 配置后才能写远程 trace。

## 5. Sales Insight 抽取是本地实现

生产应替换为 LLM + JSON Schema + repair + compliance review + human review。

## 6. 持久化是文件/内存

生产应接入数据库、对象存储、向量库，并实现租户隔离。

