# 阶段 2：存储层

## 阶段目标
建立数据持久化基础设施：内存级键值存储（供工作记忆使用）和 ChromaDB 向量存储封装（供情景/语义记忆使用），两者可独立测试验证。

## 前置条件
- 完成阶段 1（项目搭建与基础设施），`memory_agent` 包及其子模块可正常导入
- 阶段 1 中的 `MemoryConfig`、异常类和日志系统已就位

## 开发任务

### InMemoryStore（storage/in_memory_store.py）
1. 实现 `InMemoryStore` 类，管理一个内部 `Dict[str, Dict[str, Any]]` 存储。每个值字典中必须包含 `_value`（实际数据）、`_created_at`（创建时间戳 `time.time()`）、`_ttl`（TTL 秒数，`None` 表示永不过期）。
2. 实现 `set(key: str, value: Any, ttl: Optional[int] = None) -> None` —— 存入键值对，可选 TTL（秒）。
3. 实现 `get(key: str) -> Optional[Any]` —— 获取值。若 key 不存在或已过期，返回 `None`（过期条目应自动删除）。
4. 实现 `delete(key: str) -> bool` —— 删除键，返回是否成功删除。
5. 实现 `exists(key: str) -> bool` —— 检查键是否存在且未过期。
6. 实现 `keys() -> List[str]` —— 返回所有有效键列表（自动排除已过期的键）。
7. 实现 `clear() -> int` —— 清空全部数据，返回清除的条目数。
8. 实现 `expire_now(key: str) -> bool` —— 立即过期指定键（设置 TTL 为 0），返回是否操作成功。
9. 实现 `cleanup_expired() -> int` —— 主动扫描并清理所有过期条目，返回清理数量（供后续定时任务调用）。
10. 使用 `threading.Lock` 保证所有写操作线程安全。
11. 使用 `logging.getLogger(__name__)` 输出 DEBUG 级别日志，记录每次 `set`、`get`（命中/未命中）、`delete` 操作。

### ChromaStore（storage/chroma_store.py）
12. 实现 `ChromaStore` 类，封装 ChromaDB `PersistentClient`：
    - 构造函数接收 `persist_directory: str`、`collection_name: str`、`embedding_dimension: int = 512`。
    - 构造函数内创建 `chromadb.PersistentClient(path=persist_directory)`。
    - 使用 `client.get_or_create_collection(...)` 获取或创建 Collection。Collection 的 `metadata` 中存储 `{"embedding_dimension": dimension}`。
13. 维度不匹配检测：在构造函数中检查已存在的 Collection 的 `metadata["embedding_dimension"]` 是否与 `embedding_dimension` 参数一致。若不一致，记录 `warning` 日志，调用 `client.delete_collection(name)` 删除旧 Collection，然后重新创建。
14. 实现 `add(ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None` —— 批量添加向量记录。需校验 `ids`、`documents`、`embeddings` 三个列表长度一致，不一致时抛出 `ValueError`。
15. 实现 `query(query_embedding: List[float], top_k: int = 10, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]` —— 向量相似度检索。返回字典格式：`{"ids": List[str], "documents": List[str], "metadatas": List[Dict], "distances": List[float]}`。若 Collection 为空，返回所有列表为空列表。
16. 实现 `get(ids: List[str]) -> Dict[str, Any]` —— 按 ID 批量获取记录，返回格式同 `query`（无 distances 字段）。
17. 实现 `delete(ids: List[str]) -> None` —— 按 ID 批量删除。若某些 ID 不存在，不报错。
18. 实现 `count() -> int` —— 返回 Collection 中的记录总数。
19. 实现 `reset() -> None` —— 删除整个 Collection 并重建同名的空 Collection（用于修复损坏数据或测试清理）。
20. 所有 ChromaDB 调用异常应捕获并重新抛出为 `StorageError`，附带原始异常信息。

### 测试（tests/ 目录）
21. 在 `tests/test_in_memory_store.py` 中编写测试：
    - 基本 CRUD：set / get / delete / exists。
    - TTL 过期：写入 TTL=1 秒的数据，`time.sleep(1.1)` 后 get 返回 None。
    - TTL 默认永不过期：不传 TTL 的数据在长时间内仍可获取。
    - cleanup_expired 清理数量正确。
    - 线程安全：多线程并发写入同一 key，验证最终一致性。
    - delete 不存在的 key 返回 False。

22. 在 `tests/test_chroma_store.py` 中编写测试：
    - 初始化后 `count()` 为 0。
    - `add` 后 `count()` 等于添加条数。
    - `query` 返回相似度排序结果（用相同的向量查询应返回该记录且 distance 接近 0）。
    - `query` 的 `top_k` 参数正确截断结果。
    - `get` 按 ID 获取正确返回对应文档。
    - `delete` 后 `count()` 减少。
    - `reset` 后 `count()` 归零。
    - 维度不匹配自动重建 Collection（创建 Collection 写入 metadata dim=256，然后用 dim=512 初始化 ChromaStore，验证 Warning 被记录且后续 add 使用 512 维正常）。
    - 所有操作在空 Collection 的场景下不崩溃。

23. 在 `tests/conftest.py` 中添加 fixtures：
    - `in_memory_store` fixture：返回全新的 `InMemoryStore` 实例。
    - `chroma_store` fixture：使用 `tmp_path` 创建临时持久化目录，返回 `ChromaStore` 实例，teardown 时 `reset()` 清理。

## 验收标准
1. **InMemoryStore 基本 CRUD**
   - 场景：空 InMemoryStore
   - 操作：`store.set("k1", "v1")` 后 `store.get("k1")`
   - 预期：返回 `"v1"`，`store.exists("k1")` 为 `True`

2. **InMemoryStore TTL 自动过期**
   - 场景：写入带 TTL=1 秒的数据
   - 操作：`store.set("k1", "v1", ttl=1)`，等待 1.5 秒后 `store.get("k1")`
   - 预期：返回 `None`，`store.exists("k1")` 为 `False`

3. **InMemoryStore 线程安全**
   - 场景：10 个线程各执行 100 次对同一 key 的 set 操作
   - 操作：并发执行后 `store.get("shared_key")`
   - 预期：能获取到值（未被损坏），且内部 dict 无数据竞争导致的异常

4. **ChromaStore 初始化与写入**
   - 场景：空的持久化目录
   - 操作：`store = ChromaStore(persist_dir, "test_coll")`，调用 `store.add(ids=["1"], documents=["hello"], embeddings=[[0.1]*512])`
   - 预期：`store.count() == 1`

5. **ChromaStore 向量检索**
   - 场景：已写入 3 条记录，向量分别为 `[1,0,...]`, `[0,1,...]`, `[-1,0,...]`（各 512 维）
   - 操作：`store.query(query_embedding=[1.0]+[0.0]*511, top_k=2)`
   - 预期：返回 2 条结果，第一条 ids 对应 `[1,0,...]` 的记录，且 `distances[0] < distances[1]`

6. **ChromaStore 维度不匹配自动修复**
   - 场景：已存在一个使用 256 维向量的 Collection
   - 操作：用 `embedding_dimension=512` 初始化一个新的 ChromaStore（指向同一 Collection 名称）
   - 预期：Warning 日志被记录，Collection 被重建，后续 `add` 使用 512 维向量成功

7. **ChromaStore 空查询**
   - 场景：空 Collection
   - 操作：`store.query(query_embedding=[0.1]*512, top_k=5)`
   - 预期：返回 `{"ids": [], "documents": [], "metadatas": [], "distances": []}`，不抛出异常

8. **异常包装**
   - 场景：使用无效路径（如系统禁止写入的目录）初始化 ChromaStore
   - 操作：`ChromaStore("/root/forbidden", "test")`
   - 预期：抛出 `StorageError`，其 `__cause__` 包含 ChromaDB 原始异常

## 注意事项
- InMemoryStore 使用 `time.time()` 而非 `time.monotonic()`，因为 TTL 语义要求绝对时间（与系统时间关联）。
- TTL 检查采用懒删除策略（get 时检查），配合定期的 `cleanup_expired()` 调用防止内存泄漏。
- ChromaDB PersistentClient 在 Windows 环境下持久化路径需使用 `os.path.abspath()` 规范化，避免路径分隔符问题。
- ChromaStore 不应在构造函数中调用 `reset()`（除非维度不匹配），以避免意外丢失用户数据。
- 测试 ChromaDB 时需确保 `chromadb` 包已被安装（在阶段 1 的 `requirements.txt` 中已声明）。
