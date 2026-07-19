# Memory 记忆层

Memory 不是简单把所有聊天历史塞进 Prompt，而是按用途分层。

## 三层 Memory

- Session Memory：最近消息、实体和不含槽位值的保险 active-intent 控制信封；
- Task Memory：当前任务状态，例如状态机节点、工具结果和恢复信息；
- Preference Memory：长期偏好，例如用户喜欢低压话术、输出格式偏好。

## 当前实现

- `src/agent_core/memory/session.py`
- `src/agent_core/memory/task.py`
- `src/agent_core/memory/preference.py`
- `src/agent_core/memory/manager.py`
- `src/agent_core/memory/policy.py`

`MemoryManager` 提供统一入口，并带有 audit log。

FastAPI 生产路径额外使用：

- `redis_store.py`：Session/Task Hash、消息 List、租户 LRU ZSet、TTL、版本 CAS 和消息裁剪；
- `production_manager.py`：组合 Redis 与 PostgreSQL；Session/Task 在线快照留在 Redis，PostgreSQL 只追加加密消息审计和长期偏好；
- `postgres_business_store.py`：KYC 事实、Case、Event、Analysis 和 Output 的事务化 Store；
- `privacy.py`：Consent、导出、删除与跨存储清理；
- `migrations/001..005`：通用表、业务表、RLS、旧数据升级和历史运行时审批表删除。

保险客户画像、孩子、决策权和资产类型属于 Business Memory，不写进通用 Preference，也不直接写进
Redis active-intent 信封。详细设计见 [memory-system.md](memory-system.md)。

## 生产要求

- 默认不保存敏感个人信息；
- 必须有租户隔离；
- 需要数据保留周期；
- 需要删除和导出机制。

以上四项已经在 FastAPI 生产路径实现基础版本。企业部署仍应把原文/证据加密密钥放入 Secret Manager，
为数据库应用角色配置最小权限，并通过定时任务执行 `make memory-retention`。当前规范化 KYC 事实和
Session/Analysis 上下文是 RLS + Consent 保护的 JSONB，不是应用层密文；详细边界与后续迁移要求见
[memory-system.md](memory-system.md#加密rls-和隐私治理)。
