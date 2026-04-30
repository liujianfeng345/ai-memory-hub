# 阶段 5：总调度器、记忆整合与公开 API

## 阶段目标
实现 `MemoryManager` 总调度器（路由写入、检索聚合、记忆整合），完成 `__init__.py` 的公开 API 导出，编写端到端集成测试与使用示例，交付可发布的最小可用版本。

## 前置条件
- 完成阶段 1-4，所有基础设施、存储层、嵌入层、LLM 客户端、三种记忆模块均可正常工作
- 环境变量 `DEEPSEEK_API_KEY` 已配置（集成测试和示例依赖真实 API 调用）

## 开发任务

### MemoryManager（core/manager.py）
1. 实现 `MemoryManager` 类：
   - 构造函数：`__init__(self, config: Optional[MemoryConfig] = None)`。
     - 若 `config` 为 `None`，从环境变量加载默认 `MemoryConfig()`。
     - 校验 `config` 的有效性（如 `deepseek_api_key` 在需要 LLM 的功能使用前可延迟校验）。
     - 初始化 `logging.getLogger(__name__)` 日志。
     - 创建内部组件（做依赖注入的"装配器"角色）：
       - `self._working_memory`: 使用 `InMemoryStore()` + `config.default_ttl_seconds` 构造 `WorkingMemory`。
       - `self._episodic_store`: `ChromaStore(persist_directory=config.chroma_persist_dir, collection_name="episodic_memory", embedding_dimension=512)`。
       - `self._semantic_store`: `ChromaStore(persist_directory=config.chroma_persist_dir, collection_name="semantic_memory", embedding_dimension=512)`。
       - `self._embedder`: `LocalEmbedder(model_name=config.embedding_model_name, device=config.embedding_device)`。
       - `self._llm_client`: `DeepSeekClient(api_key=config.deepseek_api_key, model=config.deepseek_model, base_url=config.deepseek_base_url, timeout=config.deepseek_timeout, max_retries=config.deepseek_max_retries)`。
       - `self._episodic_memory`: `EpisodicMemory(self._episodic_store, self._embedder, self._llm_client)`。
       - `self._semantic_memory`: `SemanticMemory(self._semantic_store, self._embedder, self._llm_client)`。
     - 使用 `time.perf_counter()` 记录初始化耗时并输出 INFO 日志。

2. 实现 `async remember(self, content: str, memory_type: Literal["working", "episodic", "semantic"] = "episodic", metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> MemoryItem`：
   - 校验 `content` 长度不超过 `config.max_content_length`（默认 50000），超限抛出 `ValueError`。
   - 校验 `memory_type` 为 `"working"` 时 `session_id` 不为空，否则抛出 `ValueError`。
   - 根据 `memory_type` 路由：
     - `"working"` → `self._working_memory.add(content, session_id, metadata)`。
     - `"episodic"` → `self._episodic_memory.add_episode(content, metadata, session_id)`。
     - `"semantic"` → 调用 `self._llm_client.extract_entities(content)` 提取实体，逐条调用 `self._semantic_memory.add_entity(...)`；返回最后一个创建的实体（包装为 MemoryItem）。
   - 使用 `time.perf_counter()` 记录操作耗时，DEBUG 日志输出。
   - 捕获底层异常并包装为 `StorageError`（如底层存储失败）或透传已知异常。
   - 返回 `MemoryItem`。

3. 实现 `async recall(self, query: str, memory_type: Optional[Literal["working", "episodic", "semantic"]] = None, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None, min_similarity: float = 0.5) -> List[MemoryItem]`：
   - 若 `memory_type` 为 `None`（检索所有类型）：
     - 并行检索 `working`（仅当 `session_id` 不为空）、`episodic`、`semantic`（使用 `asyncio.gather`）。
     - 聚合结果：将三种来源的结果合并为统一的 `List[MemoryItem]` 列表。
     - 按相似度（若有）或相关性降序排列。工作记忆的关键词匹配结果排在向量检索结果之前（给予工作记忆更高优先级）。
     - 截取 `top_k` 条。
   - 若指定 `memory_type`：仅检索对应类型。
     - `"working"` → `self._working_memory.search(query, session_id, top_k)`。
     - `"episodic"` → `self._episodic_memory.search(query, top_k, filters, min_similarity)`。
     - `"semantic"` → `self._semantic_memory.search_entities(query, top_k, entity_type=None, min_similarity)`，将 `Entity` 转为 `MemoryItem`。
   - 返回结果列表。

4. 实现 `async consolidate(self, session_id: Optional[str] = None, time_window_hours: int = 24, dry_run: bool = False) -> ConsolidateResult`：
   - 从 `self._episodic_memory.get_recent(hours=time_window_hours, session_id=session_id)` 获取近期情景记忆。
   - 若列表为空，返回 `ConsolidateResult(episodes_processed=0, dry_run=dry_run)`。
   - 构建整合提示词：将多条情景记忆的 `content`（或 `summary`）拼接为上下文，调用 `self._llm_client.extract_entities(combined_text)`。
   - 解析返回的实体、偏好、关系列表。
   - 对于每个实体/偏好：
     - 检查 `semantic_memory` 中是否存在同名实体（需要按 name 精确查询，可遍历 `attributes` 元数据或用 ChromaDB 查询）。
     - 若存在同名 → 调用 `update_entity()` 合并，`updated_entities` 计数 +1。
     - 若不存在 → 调用 `add_entity()` 创建新实体，`new_entities` 计数 +1。
     - 对偏好类型同理：`new_preferences` / `updated_preferences`。
   - 对于每个关系：调用 `add_relation()`，`new_relations` 计数 +1。
   - 若 `dry_run=True`，只生成 `ConsolidateResult` 预览信息，不实际写入。
   - 捕获并收集处理过程中的错误，存入 `ConsolidateResult.errors`。
   - 返回 `ConsolidateResult`。

5. 实现 `async forget(self, memory_id: str, memory_type: Literal["working", "episodic", "semantic"]) -> bool`：
   - 路由到对应记忆模块的 remove 方法。
   - 返回删除结果。

6. 实现 `async clear_session(self, session_id: str) -> int`：
   - 调用 `self._working_memory.expire_session(session_id)`。
   - 返回清除的记忆条数。

### 公开 API 导出（memory_agent/__init__.py）
7. 在 `memory_agent/__init__.py` 中导出以下公共符号：
   ```python
   from memory_agent.core.manager import MemoryManager
   from memory_agent.utils.config import MemoryConfig
   from memory_agent.utils.errors import (
       MemoryAgentError, ConfigError, StorageError, ModelLoadError,
       EmbeddingError, LLMServiceError, LLMResponseParseError,
       DimensionMismatchError,
   )
   from memory_agent.models.memory_item import MemoryItem, MemoryType
   from memory_agent.models.episode import Episode
   from memory_agent.models.entity import Entity
   from memory_agent.models.consolidate_result import ConsolidateResult
   ```
8. 定义 `__all__` 列表，显式列出所有公开导出符号。
9. 在 `__init__.py` 中实现延迟初始化逻辑：导入时不自动加载模型和连接数据库，仅在用户实例化 `MemoryManager` 时才触发初始化。

### 使用示例（examples/basic_usage.py）
10. 编写 `examples/basic_usage.py`，演示完整的记忆管理流程：
    - 加载配置：`config = MemoryConfig()`。
    - 初始化管理器：`manager = MemoryManager(config)`。
    - 写入工作记忆：`await manager.remember("用户正在学习 Python 异步编程", memory_type="working", session_id="demo-session")`。
    - 写入情景记忆：`await manager.remember("今天用户问了关于 asyncio 的问题，表现出对 Python 并发编程的浓厚兴趣", memory_type="episodic", session_id="demo-session")`。
    - 写入语义记忆/实体：`await manager.remember("用户偏好：Python 编程语言，日常使用 VSCode 编辑器", memory_type="semantic")`。
    - 检索工作记忆：`results = await manager.recall("异步编程", memory_type="working", session_id="demo-session")`。
    - 跨类型检索：`results = await manager.recall("编程相关", top_k=5, session_id="demo-session")`。
    - 触发记忆整合：`result = await manager.consolidate(time_window_hours=24)`。
    - 清理会话：`await manager.clear_session("demo-session")`。
    - 使用 `asyncio.run(main())` 包裹（文件末尾提供 `if __name__ == "__main__"` 入口）。

### 集成测试（tests/test_manager.py）
11. 在 `tests/test_manager.py` 中编写集成测试：
    - 使用 `@pytest.fixture` 提供 `manager` fixture（使用临时 ChromaDB 目录和 mock LLM 客户端）。
    - `remember` 路由正确：`memory_type="working"` 写入工作记忆。
    - `remember` 缺少 `session_id` 时（working 类型）抛出 `ValueError`。
    - `remember` 内容超长时抛出 `ValueError`。
    - `recall` 跨类型聚合：写入 working 和 episodic 记忆后，`recall(memory_type=None)` 同时返回两者。
    - `recall` 排序：工作记忆结果排在情景记忆之前。
    - `consolidate` 从情景记忆提取实体并更新语义记忆。
    - `consolidate dry_run` 不实际写入数据。
    - `forget` 删除后 `recall` 不再返回该记忆。
    - `clear_session` 清空工作记忆。
    - 错误传播：底层 `StorageError` 被正确传递到调用方。

12. 在 `tests/conftest.py` 中添加 `manager` fixture（如有必要增加 `mock_llm` 等）。

### 项目 README 更新
13. 确认项目根目录的 `README.md` 包含：
    - 项目简介（1-2 段）。
    - 安装说明：`pip install memory-agent` 或 `pip install -e .`。
    - 快速开始：引用 `examples/basic_usage.py` 的核心代码。
    - 配置说明：列出关键环境变量及其默认值。
    - 模块架构图（ASCII art，引用技术文档中的架构图）。
    - 许可证信息。

## 验收标准
1. **端到端：写入与检索工作记忆**
   - 场景：通过 `MemoryManager` 操作
   - 操作：`await mgr.remember("测试内容", memory_type="working", session_id="sess-1")` 后 `await mgr.recall("测试", memory_type="working", session_id="sess-1")`
   - 预期：返回包含 "测试内容" 的 MemoryItem

2. **端到端：写入与检索情景记忆**
   - 场景：通过 `MemoryManager` 操作
   - 操作：`await mgr.remember("用户喜欢喝咖啡", memory_type="episodic")` 后 `await mgr.recall("饮品偏好", memory_type="episodic")`
   - 预期：返回包含 "用户喜欢喝咖啡" 的 Episode，similarity > 0.5

3. **跨类型聚合检索**
   - 场景：已写入 working 和 episodic 两种记忆（同 session）
   - 操作：`await mgr.recall("咖啡", memory_type=None, session_id="sess-1")`
   - 预期：返回结果包含两种类型的记忆，工作记忆排在前面

4. **记忆整合流程**
   - 场景：EpisodicMemory 中有近期对话，包含用户姓名和偏好信息
   - 操作：`await mgr.consolidate(time_window_hours=24)`
   - 预期：返回的 `ConsolidateResult` 中 `new_entities > 0` 或 `new_preferences > 0`，随后可通过 `mgr.recall("用户偏好", memory_type="semantic")` 检索到新创建的实体

5. **dry_run 模式**
   - 场景：同上
   - 操作：`await mgr.consolidate(time_window_hours=24, dry_run=True)`
   - 预期：返回 `ConsolidateResult(dry_run=True)` 且实体计数非零，但语义记忆中无实际新增

6. **错误处理：session_id 缺失**
   - 场景：使用 MemoryManager
   - 操作：`await mgr.remember("content", memory_type="working")`（不传 session_id）
   - 预期：抛出 `ValueError`

7. **公开 API 导出完整**
   - 场景：安装 memory-agent 包后
   - 操作：`from memory_agent import MemoryManager, MemoryConfig, MemoryItem, MemoryAgentError`
   - 预期：所有符号导入成功

8. **示例脚本可运行**
   - 场景：`DEEPSEEK_API_KEY` 已配置，阶段 1-5 全部代码就绪
   - 操作：`python examples/basic_usage.py`
   - 预期：脚本顺畅执行完毕，无异常崩溃，输出清晰的流程日志

9. **集成测试通过**
   - 场景：阶段 5 全部代码和测试就绪
   - 操作：`pytest tests/test_manager.py -v`
   - 预期：所有集成测试通过（依赖真实 API 的测试可 skip）

10. **完整测试套件通过**
    - 场景：阶段 1-5 全部代码和测试就绪
    - 操作：`pytest tests/ -v --cov=memory_agent --cov-report=term`
    - 预期：所有测试通过，代码覆盖率 > 75%

## 注意事项
- `MemoryManager` 是用户唯一需要直接实例化的类。构造函数承担了"依赖装配器"角色，后续可考虑抽取为工厂函数，但当前版本保持简单。
- `consolidate` 的提示词需要包含 few-shot 示例，帮助 LLM 稳定输出期望的 JSON 结构。
- `recall(memory_type=None)` 的并行检索使用 `asyncio.gather`，需注意 `return_exceptions=True` 以防止一个模块失败导致整个检索中断。
- ChromaDB 的 `PersistentClient` 不支持多进程并发写入，`MemoryManager` 文档字符串中应注明此限制。
- `examples/basic_usage.py` 应包含足够的注释，让读代码的开发者能直接理解 API 用法。
- 阶段 5 完成后，`memory-agent` 应达到最小可用版本（MVP），可被外部 Agent 项目通过 `pip install` 集成。
