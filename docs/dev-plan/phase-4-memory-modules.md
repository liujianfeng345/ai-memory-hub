# 阶段 4：核心记忆模块

## 阶段目标
实现三种记忆类型的核心逻辑 —— 工作记忆（WorkingMemory）、情景记忆（EpisodicMemory）、语义记忆（SemanticMemory），每种记忆均可独立测试其写入、检索和删除能力。

## 前置条件
- 完成阶段 1（项目搭建与基础设施），异常体系、配置和数据模型就位
- 完成阶段 2（存储层），`InMemoryStore` 和 `ChromaStore` 均可正常使用
- 完成阶段 3（嵌入层与大模型客户端），`LocalEmbedder` 和 `DeepSeekClient` 可正常工作

## 开发任务

### WorkingMemory（core/working_memory.py）
1. 实现 `WorkingMemory` 类：
   - 构造函数：`__init__(self, store: InMemoryStore, default_ttl_seconds: int = 3600)`。
   - 依赖注入模式，通过构造函数接收 `InMemoryStore` 实例。

2. 实现 `async add(self, content: str, session_id: str, metadata: Optional[Dict[str, Any]] = None, ttl_seconds: Optional[int] = None) -> MemoryItem`：
   - 若 `session_id` 为空字符串，抛出 `ValueError`。
   - 生成 UUID 作为记忆 ID。
   - 构造 `MemoryItem`，`memory_type="working"`，`created_at` 和 `updated_at` 设为当前 UTC 时间。
   - 将完整的 `MemoryItem`（序列化为 dict）存入 `InMemoryStore`，key 为 `f"wm:{memory_id}"`。TTL 取 `ttl_seconds` 或 `default_ttl_seconds`。
   - 同时在 `InMemoryStore` 中维护一个会话索引：key 为 `f"wm:session:{session_id}"`，值为该会话所有记忆 ID 的 list（追加记忆时将新 ID append 进去）。
   - 返回 `MemoryItem`。

3. 实现 `async get_by_session(self, session_id: str, include_expired: bool = False) -> List[MemoryItem]`：
   - 从 `store.get(f"wm:session:{session_id}")` 获取该会话的记忆 ID 列表。
   - 遍历 ID 列表，从 store 获取对应 `MemoryItem`（key `f"wm:{memory_id}"`）。
   - 若 `include_expired=False`（默认），过滤掉 TTL 已过期的条目。
   - 按 `created_at` 降序排列返回。

4. 实现 `async search(self, query: str, session_id: Optional[str] = None, top_k: int = 5) -> List[MemoryItem]`：
   - 使用简单的关键词匹配（不依赖向量）：
     - 将 `query` 分词（按中文/英文标点和空格切分）。
     - 对每条记忆的 `content` 进行分词。
     - 计算 query 和 content 的 Jaccard 相似度（两词集合交集/并集）。
   - 按相似度降序排序，返回 top_k 条。
   - 若指定 `session_id`，仅在该会话范围内搜索。

5. 实现 `async remove(self, memory_id: str) -> bool`：
   - 调用 `store.delete(f"wm:{memory_id}")`。
   - 若删除成功，从对应的会话索引中移除该 ID（需通过 metadata 中的 `session_id` 定位索引 key）。
   - 返回是否实际删除了条目。

6. 实现 `async expire_session(self, session_id: str) -> int`：
   - 获取该会话所有记忆 ID。
   - 逐一调用 `store.expire_now(f"wm:{memory_id}")`。
   - 删除会话索引 key。
   - 返回过期的记忆条数。

### EpisodicMemory（core/episodic_memory.py）
7. 实现 `EpisodicMemory` 类：
   - 构造函数：`__init__(self, chroma_store: ChromaStore, embedder: LocalEmbedder, llm_client: DeepSeekClient)`。
   - 构造函数中获取或确保 ChromaDB Collection 名称为 `"episodic_memory"`（若已有 ChromaStore 实例绑定了其他 collection，需新建专用实例）。

8. 实现 `async add_episode(self, content: str, metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> Episode`：
   - 若 `content` 超过 `SUMMARY_THRESHOLD`（默认 2000 字符，从配置读取），调用 `llm_client.chat()` 生成摘要，存入 `episode.summary`。
   - 调用 `embedder.embed(content)` 生成向量（取 `[0]`）。
   - 生成 UUID 作为 `episode.id`。
   - 构造完整的 `Episode` 对象，`memory_type="episodic"`。
   - 调用 `chroma_store.add(ids=[ep_id], documents=[content], embeddings=[vec], metadatas=[meta_dict])`。
   - 其中 `meta_dict` 包含：`session_id`, `created_at`（ISO 格式字符串）, `importance`, `tags`（JSON 序列化）, `summary`。
   - 返回 `Episode`。

9. 实现 `async search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, min_similarity: float = 0.5) -> List[Episode]`：
   - 调用 `embedder.embed_query(query)` 生成查询向量。
   - 将 `filters` 转换为 ChromaDB `where` 子句：
     - `start_time` / `end_time`（ISO 格式）→ ChromaDB 的 `"$gte"` / `"$lte"` 作用于 `created_at` 元数据。
     - `session_id` → ChromaDB `where` 中的等值过滤。
     - `tags` → 若为列表，使用 `"$contains"` 运算符（ChromaDB 仅支持列表字段的 `$contains`，tags 需存储为列表）。
   - 调用 `chroma_store.query(query_embedding=vec, top_k=top_k, where=where_clause)`。
   - 过滤 `distances` 对应 `(1 - min_similarity)` 的结果（ChromaDB 返回距离，余弦距离需换算）。
   - 将返回结果组装为 `List[Episode]`，并按相似度降序排列。
   - 排除 `embedding` 字段（不在返回给调用方的对象中包含向量数据）。

10. 实现 `async get_recent(self, hours: int = 24, session_id: Optional[str] = None) -> List[Episode]`：
    - 计算 `start_time = datetime.utcnow() - timedelta(hours=hours)`。
    - 构建 `where` 过滤条件，`created_at >= start_time.isoformat()`。
    - 调用 `chroma_store.get()` 并使用元数据过滤（若 ChromaDB 不支持直接时间过滤则回退到 Python 端过滤）。
    - 返回结果列表。

11. 实现 `async get_by_id(self, episode_id: str) -> Optional[Episode]`：
    - 调用 `chroma_store.get(ids=[episode_id])`。
    - 若返回空，返回 `None`。
    - 否则构造 `Episode` 返回。

12. 实现 `async remove(self, episode_id: str) -> bool`：
    - 先调用 `get_by_id(episode_id)` 确认存在。
    - 调用 `chroma_store.delete(ids=[episode_id])`。
    - 返回是否存在并删除成功。

### SemanticMemory（core/semantic_memory.py）
13. 实现 `SemanticMemory` 类：
    - 构造函数：`__init__(self, chroma_store: ChromaStore, embedder: LocalEmbedder, llm_client: DeepSeekClient)`。
    - 使用专用 ChromaDB Collection `"semantic_memory"`（创建新的 ChromaStore 实例）。

14. 实现 `async add_entity(self, name: str, entity_type: str, description: str, attributes: Optional[Dict[str, Any]] = None, related_entities: Optional[List[str]] = None) -> Entity`：
    - 校验 `entity_type` 在合法值集合 `{"person", "organization", "topic", "preference", "fact"}` 中。
    - 调用 `embedder.embed(description)` 生成向量。
    - 检查是否有同名实体（在 ChromaDB 中按 `name` 元数据查询）：若有则合并更新（Merge 策略：覆盖同名字段，合并 `attributes` 字典，追加 `related_entities` 并去重）。
    - 若无同名实体，生成 UUID 创建新 Entity。
    - 调用 `chroma_store.add()` 写入（文档内容为 `description`，元数据包含所有非向量字段）。
    - 返回 `Entity`。

15. 实现 `async search_entities(self, query: str, top_k: int = 10, entity_type: Optional[str] = None, min_similarity: float = 0.5) -> List[Entity]`：
    - 与 `EpisodicMemory.search()` 流程一致（embed query → ChromaDB query → 过滤距离 → 组装 Entity 列表）。
    - 支持 `entity_type` 过滤（`where` 条件）。
    - 返回时排除 `embedding` 字段。

16. 实现 `async get_entity(self, entity_id: str) -> Optional[Entity]`：
    - 与 `EpisodicMemory.get_by_id()` 类似。

17. 实现 `async get_preferences(self, user_id: Optional[str] = None) -> List[Entity]`：
    - 构建 `where = {"entity_type": "preference"}`，若有 `user_id` 则在 `attributes` 中过滤（需在 Python 端做二次过滤，因为 ChromaDB 无法嵌套查询 attributes）。
    - 返回 Entity 列表。

18. 实现 `async update_entity(self, entity_id: str, updates: Dict[str, Any]) -> Entity`：
    - 获取现有实体。
    - 应用更新（Pydantic 的 `model_copy(update=updates)` 或手动合并）。
    - 若 `description` 变化，重新计算嵌入向量。
    - 调用 `chroma_store.delete(ids=[entity_id])` 后 `chroma_store.add()`（ChromaDB 不支持原地更新）。
    - 返回更新后的 Entity。

19. 实现 `async add_relation(self, source_id: str, target_id: str, relation_type: str) -> None`：
    - 获取 source 实体。
    - 在 `related_entities` 中追加 `target_id`（去重）。
    - 更新 source 实体。
    - 同步更新 target 实体的 `related_entities`（双向关系）。

20. 实现 `async get_related_entities(self, entity_id: str, relation_type: Optional[str] = None) -> List[Entity]`：
    - 获取 entity_id 实体。
    - 从 `related_entities` 获取关联实体 ID 列表。
    - 批量 `get` 关联实体，若有 `relation_type` 则在返回前过滤（关系类型存储在 metadata 中）。
    - 返回实体列表。

21. 实现 `async remove_entity(self, entity_id: str) -> bool`：
    - 获取实体。
    - 从所有关联实体的 `related_entities` 中移除该 ID。
    - 删除 ChromaDB 中的记录。
    - 返回是否成功删除。

### 测试（tests/ 目录）
22. 在 `tests/conftest.py` 中添加 fixtures：
    - `working_memory` fixture：使用 `in_memory_store` 构造 `WorkingMemory`。
    - `episodic_memory` fixture：使用临时 ChromaStore（collection="test_episodic"）、`embedder`、`mock_deepseek_client`。
    - `semantic_memory` fixture：使用临时 ChromaStore（collection="test_semantic"）、`embedder`、`mock_deepseek_client`。

23. 在 `tests/test_working_memory.py` 中编写测试：
    - 添加记忆后，`get_by_session` 返回该记忆。
    - 添加多条记忆后，`get_by_session` 返回按时间降序排列的列表。
    - TTL 到期后，`get_by_session(include_expired=False)` 不返回过期记忆。
    - `search` 关键词匹配（"咖啡" 匹配到含 "咖啡" 的记忆）。
    - `search` 限定 `session_id` 的正确过滤。
    - `remove` 成功删除后，`get_by_session` 不再返回该记忆。
    - `expire_session` 清空整个会话。
    - 无 `session_id` 时 `add` 抛出 `ValueError`。
    - 并发 `add` 后会话索引正确（无丢失）。

24. 在 `tests/test_episodic_memory.py` 中编写测试：
    - `add_episode` 写入后，`search` 可检索到。
    - `add_episode` 长内容（>2000 字符）自动生成摘要。
    - `search` 语义检索：写入 "用户喜欢喝咖啡" 和 "Python 是一门编程语言"，搜索 "饮品偏好" 返回前者且相似度更高。
    - `search` 时间过滤：写入不同时间的记忆，`filters={"start_time": "...", "end_time": "..."}` 正确过滤。
    - `search` session_id 过滤。
    - `search` min_similarity 过滤低分结果。
    - `get_recent` 返回指定小时内的记忆。
    - `get_by_id` 和 `remove` 基本功能。
    - 返回的 Episode 对象不包含 `embedding` 字段。

25. 在 `tests/test_semantic_memory.py` 中编写测试：
    - `add_entity` 写入后，`search_entities` 可检索到。
    - 同名实体合并更新（第二次 `add_entity` 同名更新 attributes）。
    - `search_entities` entity_type 过滤。
    - `get_preferences` 返回类型为 "preference" 的实体。
    - `update_entity` 更新 description 后嵌入向量重新计算。
    - `add_relation` 建立双向关系后 `get_related_entities` 返回关联实体。
    - `remove_entity` 级联清理关联关系。
    - 返回的 Entity 对象不包含 `embedding` 字段。

## 验收标准
1. **工作记忆基本写入与读取**
   - 场景：空 WorkingMemory，指定 `session_id="sess-1"`
   - 操作：`await wm.add("用户在学习 Python", session_id="sess-1")` 后 `await wm.get_by_session("sess-1")`
   - 预期：返回 1 条 MemoryItem，`content` 为 "用户在学习 Python"，`memory_type` 为 `"working"`

2. **工作记忆 TTL 自动过期**
   - 场景：写入 TTL=1 秒的记忆
   - 操作：`await wm.add("临时数据", session_id="sess-1", ttl_seconds=1)`，等待 1.5 秒后 `await wm.get_by_session("sess-1")`
   - 预期：返回空列表

3. **工作记忆关键词检索**
   - 场景：写入 3 条记忆，内容分别为 "Python 编程"、"机器学习"、"数据分析"
   - 操作：`await wm.search("编程语言", top_k=2)`
   - 预期：返回 "Python 编程" 排第一，"数据分析" 或 "机器学习" 排第二

4. **情景记忆语义检索**
   - 场景：已写入 "用户喜欢喝蓝山咖啡" 和 "今天天气不错" 两条情景记忆
   - 操作：`await em.search("饮品偏好", top_k=1)`
   - 预期：返回 "用户喜欢喝蓝山咖啡"，且相似度 > 0.6

5. **情景记忆自动摘要**
   - 场景：写入一篇超过 2000 字符的长文
   - 操作：`await em.add_episode(long_text_content)`
   - 预期：返回的 Episode 对象中 `summary` 字段不为 None，且摘要长度 < 原文长度

6. **情景记忆时间过滤**
   - 场景：分别写入一条 30 天前的记忆和一条 1 小时前的记忆
   - 操作：`await em.search("query", filters={"start_time": (now - 2h).isoformat()})`
   - 预期：仅返回 1 小时前的记忆，30 天前的被过滤掉

7. **语义记忆实体 upsert**
   - 场景：空 SemanticMemory
   - 操作：`await sm.add_entity("张三", "person", "一个喜欢咖啡的用户", attributes={"年龄": 30})`，再次调用 `await sm.add_entity("张三", "person", "一个喜欢咖啡和茶的用户", attributes={"城市": "北京"})`
   - 预期：实体合并，`description` 更新，`attributes` 合并为 `{"年龄": 30, "城市": "北京"}`

8. **语义记忆偏好查询**
   - 场景：已写入 2 条 preference 类型实体和 3 条 person 类型实体
   - 操作：`await sm.get_preferences()`
   - 预期：返回 2 条 preference 实体

9. **语义记忆关系管理**
   - 场景：Person 实体 A 和 Preference 实体 B 已存在
   - 操作：`await sm.add_relation(A.id, B.id, "has_preference")`，然后 `await sm.get_related_entities(A.id)`
   - 预期：返回包含实体 B 的列表

10. **记忆模块不泄露向量数据**
    - 场景：执行任意 `search` 或 `get_by_id` 操作
    - 操作：检查返回的 Episode / Entity 对象的 `embedding` 字段
    - 预期：`embedding` 字段为 `None`（向量仅内部使用，不对调用方暴露）

11. **所有记忆模块的测试独立运行**
    - 场景：阶段 4 全部测试文件就绪
    - 操作：`pytest tests/test_working_memory.py tests/test_episodic_memory.py tests/test_semantic_memory.py -v`
    - 预期：所有测试通过

## 注意事项
- WorkingMemory 的会话索引（`wm:session:{session_id}`）需要原子操作。Python 的 GIL 保证了单进程内的原子性，但在 asyncio 协程并发场景下，`InMemoryStore` 已实现 `threading.Lock`，需额外考虑 `asyncio.Lock` 或直接复用现有锁（因为 `asyncio` 在单线程中运行）。
- EpisodicMemory 中 ChromaDB 返回的是距离（distance）而非相似度，`min_similarity` 参数需换算为距离阈值。对于余弦距离（cosine distance）：`similarity = 1 - distance`（当向量已归一化时）。
- SemanticMemory 的合并更新逻辑需处理 `attributes` 字典的深度合并，而非简单覆盖。
- 当前阶段不考虑 `MemoryManager` 调度层，仅编写各记忆模块的单元测试。模块间无互相依赖。
