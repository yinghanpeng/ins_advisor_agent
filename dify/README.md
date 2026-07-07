# Dify Control Plane 说明

Dify 在本项目中负责可视化配置、Prompt 管理和内部调试，不负责生产高并发运行。

## 推荐集成流程

1. Dify 接收用户输入和运营配置；
2. Dify 通过 HTTP 节点调用 `POST /agent/run`；
3. FastAPI Agent Gateway 把请求交给 Agent Core；
4. Agent Core 完成状态机、工具、RAG、Sales Intelligence、Guardrails、日志和评估；
5. Dify 展示最终回答和 trace id。

## 为什么这样拆

这样可以保留 Dify 的可视化优势，同时把生产治理能力放到更适合的 Data Plane。

节点说明见：`dify/nodes/*.md`。

