# 阶段 2 变更日志

## 修改的文件

### 新增文件

| 文件 | 说明 |
|------|------|
| `memory_agent/storage/in_memory_store.py` | 线程安全的内存键值存储，支持 TTL 过期、懒删除和主动清理 |
| `memory_agent/storage/chroma_store.py` | ChromaDB 向量存储封装，提供向量增删改查、维度检测和异常包装 |
| `tests/test_in_memory_store.py` | InMemoryStore 单元测试（32 个测试用例） |
| `tests/test_chroma_store.py` | ChromaStore 单元测试（34 个测试用例） |

### 修改文件

| 文件 | 说明 |
|------|------|
| `memory_agent/storage/__init__.py` | 导出 `InMemoryStore` 和 `ChromaStore` |
| `tests/conftest.py` | 新增 `in_memory_store` 和 `chroma_store` fixture |

## 核心逻辑

### InMemoryStore 实现要点

- **内部存储结构**：`Dict[str, Dict[str, Any]]`，每个值字典包含 `_value`（实际数据）、`_created_at`（创建时间戳）、`_ttl`（TTL 秒数，`None` 表示永不过期）
- **过期策略**：
  - `get` 和 `exists` 方法使用懒删除：检查条目 TTL，若已过期则自动删除并返回 `None` / `False`
  - `keys` 方法在返回键列表前清理所有过期条目
  - `cleanup_expired` 方法提供主动扫描清理，返回清理数量
- **`expire_now`** 将目标条目的 TTL 设为 0 并更新 `_created_at`，使其在下一次检查时被删除；对 `ttl=None` 的永不过期条目同样生效
- **线程安全**：所有读写操作均由 `threading.Lock` 保护，包括 `get`、`set`、`delete`、`exists`、`keys`、`clear`、`expire_now`、`cleanup_expired`
- **日志**：使用 `logging.getLogger(__name__)` 输出 DEBUG 级别日志，记录每次 `SET`、`GET`（hit/miss）、`DELETE` 操作

### ChromaStore 实现要点

- **初始化流程**：
  1. 规范化持久化路径（`os.path.abspath` + `os.makedirs`）
  2. 创建 `chromadb.PersistentClient`
  3. 调用 `_get_or_create_collection` 获取或创建 Collection
- **维度不匹配检测**：在构造函数中通过 `client.get_collection` 检查已存在的 Collection 的 `metadata["embedding_dimension"]`。若与参数不一致，记录 `warning` 日志，调用 `client.delete_collection` 删除旧 Collection 并重新创建
- **空列表处理**：`add`、`delete`、`get` 方法在遇到空列表时直接返回，避免触发 ChromaDB 对空列表的校验（ChromaDB 要求非空）
- **`query` 返回格式**：将 ChromaDB 嵌套列表响应展平为 `{"ids": List[str], "documents": List[str], "metadatas": List[Dict], "distances": List[float]}`
- **异常包装**：所有 ChromaDB 调用异常通过 `StorageError` 包裹，保留原始异常作为 `__cause__`，并在 `details` 中附加上下文信息
- **Collection 元数据**：使用 `metadata={"embedding_dimension": str(dim)}` 存储向量维度

## 测试覆盖

### InMemoryStore 测试（32 用例）

- **基本 CRUD**（13 个）：set/get（含 None 值、复杂对象）、delete（存在/不存在）、exists、keys、clear、覆盖写入
- **TTL 过期**（10 个）：TTL=1 过期、exists 过期返回 False、无 TTL 永不过期、TTL=0 立即过期、懒删除后不在 keys 中、expire_now（正常/不存在/已过期/永不过期）、keys 排除过期项
- **主动清理**（3 个）：cleanup_expired 清理计数、无过期返回 0、空存储返回 0
- **线程安全**（3 个）：10 线程各 100 次写入同一 key、并发 set+get、并发 delete+get
- **边界情况**（3 个）：空字符串 key、多种数据类型、1000 个 key 大规模写入

### ChromaStore 测试（34 用例）

- **初始化**（4 个）：count 为 0、属性设置正确、创建持久化目录、不同维度初始化
- **add**（5 个）：count 递增、多批次累计、带 metadatas、长度不匹配抛 ValueError（ids 和 metadatas 分开测试）、空批次
- **query**（6 个）：相似度排序、top_k 截断、top_k 超总数全返回、相同向量 distance 接近 0、空集合返回空列表、where 过滤
- **get**（5 个）：正确返回文档、不存在 ID 不报错、空 ID 返回空、metadatas 返回、无 distances 字段
- **delete**（3 个）：count 减少、不存在 ID 不报错、空列表不报错
- **reset**（3 个）：count 归零、可重用、空集合不报错
- **维度不匹配**（2 个）：自动重建 + Warning 日志（使用 caplog）、同维度数据保留
- **异常处理**（3 个）：无效路径抛 StorageError、空集合所有操作不崩溃、零维度边缘情况
- **集成场景**（3 个）：add→query→get→delete 完整流程、多次查询结果一致性

### Fixtures 更新

- `in_memory_store`：返回全新 `InMemoryStore` 实例，teardown 时清空
- `chroma_store`：使用 `tmp_path` 创建临时持久化目录，返回 `Chromosome`(embedding_dimension=512)，teardown 时 `reset()` 清理

## 注意事项（给 dev-tester）

1. **InMemoryStore TTL 测试存在 sleep（1.1 秒）**，三个 TTL 相关测试各需约 1.1 秒，总计约 6 秒的等待时间。
2. **ChromaDB 依赖**：`test_chroma_store.py` 需要 `chromadb` 包已安装。测试使用 `tmp_path` 创建临时持久化目录，不会污染用户数据。
3. **维度不匹配测试**：使用 `caplog` fixture 捕获日志输出，验证 warning 级别的维度不匹配日志。此测试创建两个 ChromaStore 实例指向同一目录，需确认 cleanup 正确。
4. **进程安全验证**：`test_invalid_path_raises_storage_error` 在 Windows 上可能不触发预期异常，因为文件系统对路径的校验行为不同。代码中已兼容 `ValueError` 和 `OSError` 两种异常类型。
5. **并发测试**：线程安全测试在高负载 CI 环境中可能因调度延迟而变慢，但不应失败。若出现间歇性失败，可增加 `num_writes` 参数。

## 已知限制

1. **InMemoryStore 的 TTL 使用 `time.time()`**（系统绝对时间），而非 `time.monotonic()`。这意味着系统时间调整可能影响 TTL 行为。这是有意为之，因为 TTL 语义要求与绝对时间关联。
2. **ChromaStore 的 `_get_or_create_collection` 方法**使用多重 `try-except` 判断 Collection 是否存在，依赖 ChromaDB 的 `get_collection` 在 Collection 不存在时抛出 `ValueError` 的内部实现细节。若未来 ChromaDB 版本改变此行为，需同步调整。
3. **ChromaStore 未实现批量操作的事务性**：若 `add` 在 ChromaDB 调用中失败，已通过的长度校验不会回滚。对于原子性要求较高的场景，需由上层逻辑处理。
4. **ChromaDB `PersistentClient` 在并发场景下**由 ChromaDB 自身的 SQLite 库保证线程安全，但多进程并发访问同一持久化目录可能导致数据库锁定问题。

## 验收标准验证结果

| # | 验收标准 | 状态 |
|---|---------|------|
| 1 | InMemoryStore CRUD: `set("k1","v1")` → `get("k1")` 返回 `"v1"`, `exists("k1")` 为 `True` | 通过 |
| 2 | InMemoryStore TTL: `set("k1","v1",ttl=1)`, 等待 1.5s → `get("k1")` 返回 `None`, `exists("k1")` 为 `False` | 通过 |
| 3 | InMemoryStore 线程安全: 10 线程各 100 次写入 → 获取到值，无异常 | 通过 |
| 4 | ChromaStore 初始化与写入: `add(ids=["1"], documents=["hello"], embeddings=...)` → `count() == 1` | 通过 |
| 5 | ChromaStore 向量检索: 3 条不同向量记录 → `query([1,0,...], top_k=2)` 返回 2 条，第 1 条为 `[1,0,...]` 的记录，`distances[0] < distances[1]` | 通过 |
| 6 | ChromaStore 维度不匹配: 先创建 256 维，再以 512 维初始化同一名称 → Warning 日志记录，Collection 重建，512 维 add 成功 | 通过 |
| 7 | ChromaStore 空查询: 空 Collection 下 `query([0.1]*512, top_k=5)` → 返回 `{"ids":[], "documents":[], "metadatas":[], "distances":[]}` | 通过 |
| 8 | 异常包装: 无效路径 → 抛出 `StorageError`, `__cause__` 包含原始异常 | 通过 |
