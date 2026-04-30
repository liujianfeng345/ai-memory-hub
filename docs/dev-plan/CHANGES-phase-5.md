# 阶段 5 变更记录

## 修改的文件

### 新增文件

| 文件 | 说明 |
|------|------|
| `memory_agent/core/manager.py` | MemoryManager 总调度器，实现依赖装配、写入路由、跨类型检索聚合、记忆整合、删除和会话清理 |
| `examples/__init__.py` | examples 包标识文件 |
| `examples/basic_usage.py` | 完整的使用示例：配置→初始化→写入三种记忆→检索→整合→清理 |
| `tests/test_manager.py` | 集成测试：26 个测试用例覆盖所有公开 API 和错误路径 |
| `README.md` | 项目文档，包含简介、安装、快速开始、配置、架构图 |

### 修改文件

| 文件 | 变更说明 |
|------|----------|
| `memory_agent/__init__.py` | 新增公开 API 导出：MemoryManager, MemoryConfig, 所有异常类, MemoryItem, MemoryType, Episode, Entity, ConsolidateResult；定义 `__all__` 列表；实现延迟初始化（导入时不加载模型） |
| `memory_agent/core/__init__.py` | 新增 MemoryManager 导出 |
| `tests/conftest.py` | 新增 `manager` fixture（使用临时 ChromaDB 目录和 mock LLM 客户端） |

## 核心逻辑

### 1. MemoryManager 构造函数

承担"依赖装配器"角色，按以下顺序创建所有内部组件：

```
InMemoryStore ───┬─► WorkingMemory
ChromaStore("episodic_memory") ─┬─► EpisodicMemory
ChromaStore("semantic_memory") ─┤
LocalEmbedder ──────────────────┤
DeepSeekClient ─────────────────┴─► SemanticMemory
```

使用 `time.perf_counter()` 记录初始化耗时并输出 INFO 日志。

### 2. remember 写入路由

- **内容长度校验**：content 超过 `config.max_content_length`（默认 50000）抛出 `ValueError`。
- **session_id 校验**：memory_type="working" 时 session_id 不能为空，否则抛出 `ValueError`。
- **类型路由**：
  - `"working"` → `WorkingMemory.add()`
  - `"episodic"` → `EpisodicMemory.add_episode()`
  - `"semantic"` → `LLM Client.extract_entities()` → 遍历实体 → `SemanticMemory.add_entity()`（含偏好和关系）
- **实体类型映射**：LLM 返回的 entity_type（如 "object", "location", "event", "other"）通过 `_normalize_entity_type()` 自动映射到 Entity 模型支持的合法类型（person, organization, topic, preference, fact）。
- **异常处理**：底层 `StorageError` 和 `ValueError` 直接透传，其他异常包装为 `StorageError`。

### 3. recall 跨类型检索

- **单类型检索**：直接调用对应记忆模块的 search 方法。
- **跨类型检索**（memory_type=None）：
  - 使用 `asyncio.gather(return_exceptions=True)` 并行检索三种类型。
  - 仅当 session_id 不为空时检索工作记忆。
  - 单个模块检索失败不会影响其他模块的结果。
  - **排序策略**：工作记忆关键词匹配结果排在前面（更高优先级），然后依次是情节记忆和语义记忆的结果。
  - 截取 top_k 条返回。

### 4. consolidate 记忆整合

- 从 `EpisodicMemory.get_recent()` 获取近期情节记忆。
- 将多条情节记忆的 content 拼接为整合上下文。
- 使用 **few-shot 提示词**（包含 2 个示例）调用 LLM 提取实体、偏好和关系。
- 对每个实体：先按名称在语义记忆中查找，存在则更新合并，不存在则新建。
- dry_run=True 时仅生成 `ConsolidateResult` 预览，不实际写入。
- 处理过程中的错误收集到 `ConsolidateResult.errors` 列表中。

### 5. forget 删除

按 memory_type 路由删除：
- working → `WorkingMemory.remove()`
- episodic → `EpisodicMemory.remove()`
- semantic → `SemanticMemory.remove_entity()`（含级联清理关系）

### 6. clear_session 会话清理

调用 `WorkingMemory.expire_session()` 清空指定会话的所有工作记忆，返回清除条数。

## 测试建议

### 环境要求

- 集成测试使用 **mock DeepSeekClient**，不依赖真实 API key。
- 需要 ChromaDB 和 sentence-transformers 已安装。
- 嵌入模型 `models/bge-small-zh-v1.5` 需要在本地可用或能通过网络下载。

### 测试覆盖

| 测试类 | 测试用例数 | 覆盖场景 |
|--------|-----------|----------|
| TestRememberRouting | 3 | working/episodic/semantic 路由正确性 |
| TestRememberValidation | 3 | session_id 缺失、内容超长、无效 memory_type |
| TestRecall | 6 | 单类型检索、跨类型聚合、工作记忆优先级、空查询、Entity 转 MemoryItem |
| TestConsolidate | 4 | 空情节、正常整合、dry_run、LLM 错误捕获 |
| TestForget | 5 | working/episodic/semantic 删除、不存在条目、无效类型 |
| TestClearSession | 2 | 清空会话、空会话 |
| TestErrorPropagation | 1 | StorageError 传播 |
| TestManagerConstruction | 2 | 组件装配验证、Entity 转 MemoryItem |

### 运行测试

```bash
# 仅运行阶段 5 集成测试
pytest tests/test_manager.py -v

# 运行全部测试
pytest tests/ -v

# 含覆盖率
pytest tests/ -v --cov=memory_agent --cov-report=term
```

## 已知限制

1. **ChromaDB 并发限制**：`PersistentClient` 不支持多进程并发写入同一持久化目录。MemoryManager 文档字符串中已注明此限制。

2. **语义记忆写入的原子性**：`remember("semantic")` 操作不是原子的——如果部分实体写入成功部分失败，上游只能通过异常获知，已写入的实体不会回滚。

3. **consolidate 的关系匹配**：关系中的 source/target 按名称精确匹配（区分大小写），若 LLM 返回的名称与已有实体名称不一致，关系将无法建立。

4. **工作记忆搜索精度**：`WorkingMemory.search()` 基于 Jaccard 关键词匹配，不支持语义理解。对于纯同义词查询（如搜索"编程"查找"coding"内容），可能无法召回。

5. **嵌入模型冷启动**：首次实例化 MemoryManager 时，LocalEmbedder 需要加载 BGE 模型文件（懒加载），耗时约 2-5 秒（视硬件而定）。后续实例化可复用已加载的模型。

6. **类图依赖简洁性**：当前版本构造函数直接装配所有组件，后续可考虑抽取工厂函数或 Builder 模式以提高可测试性和可配置性。
