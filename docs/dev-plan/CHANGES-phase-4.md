# 阶段 4 变更记录 (CHANGES-phase-4.md)

## 概述
实现三种核心记忆类型的完整逻辑：WorkingMemory（工作记忆）、EpisodicMemory（情节记忆）、SemanticMemory（语义记忆），以及对应的单元测试。

测试结果：**315 个测试全部通过**（新增 68 个）。

---

## 修改的文件

### 新增文件

| 文件 | 说明 |
|------|------|
| `memory_agent/core/working_memory.py` | 工作记忆模块，任务 1-6 |
| `memory_agent/core/episodic_memory.py` | 情节记忆模块，任务 7-12 |
| `memory_agent/core/semantic_memory.py` | 语义记忆模块，任务 13-21 |
| `tests/test_working_memory.py` | WorkingMemory 单元测试（21 个测试用例），任务 23 |
| `tests/test_episodic_memory.py` | EpisodicMemory 单元测试（20 个测试用例），任务 24 |
| `tests/test_semantic_memory.py` | SemanticMemory 单元测试（27 个测试用例），任务 25 |

### 修改文件

| 文件 | 变更内容 |
|------|---------|
| `memory_agent/storage/chroma_store.py` | 新增余弦距离（`hnsw:space=cosine`）支持；维度/距离度量不匹配自动重建；reset() 保持一致 |
| `tests/conftest.py` | 新增 `working_memory`、`episodic_memory`、`semantic_memory` 三个 fixture（任务 22） |
| `pyproject.toml` | 更新 pytest-asyncio 依赖版本约束（`>=0.24,<1.0`）；修改 `asyncio_mode` 配置 |

---

## 核心逻辑

### 1. WorkingMemory
- **设计模式**: 依赖注入，通过构造函数接收 `InMemoryStore` 实例
- **会话索引**: `wm:session:{session_id}` key 存储该会话的记忆 ID 列表
- **中文分词**: 混合策略 —— CJK 字符使用「单字 + bigram」，ASCII 使用空格分词
- **Jaccard 相似度**: 基于词集合的交集/并集计算，用于关键词检索
- **并发安全**: 除 `InMemoryStore` 的 `threading.Lock` 外，额外使用 `asyncio.Lock` 保护会话索引的"读取-修改-写入"操作
- **核心方法**:
  - `add()` - 生成 UUID，构造 MemoryItem，写入 store + 更新会话索引
  - `get_by_session()` - 从索引获取 ID 列表，批量获取 MemoryItem，按 created_at 降序
  - `search()` - 分词 → Jaccard 相似度 → 排序 → top_k
  - `remove()` - 删除记忆条目，同时从会话索引清理
  - `expire_session()` - 逐条 expire_now + 删除会话索引

### 2. EpisodicMemory
- **设计模式**: 组合模式，接收 `ChromaStore` + `LocalEmbedder` + `DeepSeekClient`
- **自动摘要**: 当 content 超过 `summary_threshold`（默认 2000 字符）时，调用 LLM 生成摘要。摘要失败不阻止写入
- **距离换算**: ChromaDB 使用余弦距离（`hnsw:space=cosine`），`similarity = 1 - distance`
- **时间过滤**: 存储 `timestamp`（epoch float）到元数据中，支持 ChromaDB `$gte`/`$lte` 查询。通过 `_iso_to_epoch()` 将用户提供的 ISO 时间字符串转为 epoch
- **多条件过滤**: 使用 ChromaDB `$and` 操作符组合多个过滤条件（timestamp + session_id）
- **向量安全**: 所有公开方法的返回值设置 `embedding=None`
- **核心方法**:
  - `add_episode()` - 生成 UUID，调用 embedder，写入 ChromaDB，含摘要自动生成
  - `search()` - embed_query → ChromaDB query → 距离过滤 → 组装 Episode 列表
  - `get_recent()` - 时间范围查询（使用 timestamp epoch）
  - `get_by_id()` / `remove()` - 基本 CRUD

### 3. SemanticMemory
- **设计模式**: 组合模式，接收 `ChromaStore` + `LocalEmbedder` + `DeepSeekClient`（LLM 预留）
- **专用 Collection**: `"semantic_memory"`，与 `"episodic_memory"` 分离
- **实体合并 (upsert)**: `add_entity()` 按 `name` 查找同名实体，存在则合并 —— description 用新值，attributes 深度合并，related_entities 追加去重
- **深度合并**: `_deep_merge_dicts()` 函数递归合并嵌套字典。如 `{"地址": {"城市": "北京"}}` + `{"地址": {"区域": "海淀"}}` → `{"地址": {"城市": "北京", "区域": "海淀"}}`
- **双向关系**: `add_relation()` 同时在 source 和 target 实体的 `related_entities` 中添加对方 ID
- **级联删除**: `remove_entity()` 删除实体前，遍历其 `related_entities` 清理所有关联实体的引用
- **ChromaDB 更新策略**: 由于 ChromaDB 不支持原地更新，`update_entity()` 和 `_merge_entity()` 均采用"删除旧记录 → 写入新记录"的方式
- **核心方法**:
  - `add_entity()` - 校验类型 → 查找同名 → 合并或新建 → 写入 ChromaDB
  - `search_entities()` - embed_query → ChromaDB query → 距离过滤 → 组装 Entity 列表
  - `get_entity()` / `get_preferences()` / `update_entity()`
  - `add_relation()` / `get_related_entities()` / `remove_entity()`

---

## 注意事项

### 给 dev-tester 的建议

1. **WorkingMemory TTL 测试**: 使用 `asyncio.sleep()` 模拟时间流逝，确保测试环境的系统钟表不走偏
2. **语义检索相似度**: BGE-small-zh-v1.5 模型对中文相似度在 0.3-0.7 之间，`min_similarity=0.5` 适合生产环境，测试中需降低（如 0.3）以覆盖更广的匹配
3. **ChromaDB 距离**: 确认 Collection 使用了 `hnsw:space=cosine` metadata。旧 Collection 可能因 L2 距离导致 `1 - distance` 换算错误（相似度可能为负）
4. **并发测试**: `TestConcurrency` 需要 `pytest.mark.asyncio` 标记，多协程通过 `asyncio.gather` 并发执行
5. **模型依赖**: 测试需要 BGE 模型文件在 `models/bge-small-zh-v1.5/`，或从 HuggingFace 下载
6. **Mock DeepSeek**: 测试中 LLM 调用通过 `mock_deepseek_client` 模拟，无需真实 API key

### 边界条件和待验证场景

1. **空 Collection 查询**: `search()` 和 `search_entities()` 在空 Collection 上返回空列表
2. **极低 min_similarity**: 设置 `min_similarity=0.0` 时返回所有命中结果
3. **不存在的实体/记忆**: 所有 `remove()` / `get_by_id()` 对不存在 ID 返回 `False` / `None`
4. **同名实体合并**: 多次 `add_entity` 同名实体触发 upsert，attributes 深度合并
5. **并发会话索引**: 20 个协程并发 `add` 后，会话索引总数正确

---

## 已知限制

1. **WorkingMemory 的 `include_expired=True`**: 由于 `InMemoryStore` 的懒删除策略（`get()` 触发删除），过期数据在第一次访问后即被移除，`include_expired=True` 仅在数据未被访问前有效
2. **SemanticMemory 的标签过滤**: `EpisodicMemory` 中的 tags 当前以 JSON 字符串存储，ChromaDB 的 `$contains` 操作符对字符串不生效，标签过滤暂未在存储层实现，需在 Python 端二次过滤
3. **ChromaDB 更新代价**: `update_entity()` 需要"删除 + 重新写入"，对于频繁更新的场景性能较差。未来可考虑使用支持原地更新的向量数据库或批量更新策略
4. **关系类型**: `add_relation()` 接受 `relation_type` 参数但当前版本不存储关系类型信息。未来可在 metadata 中添加关系类型字段以支持更丰富的查询
5. **中文分词**: `_tokenize()` 使用基于 CJK 字符的简单分词（单字 + bigram），无词典支持。分词精度可能不如 jieba 等专业中文分词库
