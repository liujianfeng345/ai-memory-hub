# 阶段 3：嵌入层与大模型客户端

## 阶段目标
完成本地嵌入模型封装（LocalEmbedder）和 DeepSeek API 客户端（DeepSeekClient），为上层记忆模块提供文本向量化和大模型推理能力。两个模块相互独立，可分别测试。

## 前置条件
- 完成阶段 1（项目搭建与基础设施），异常体系和配置管理就位
- 完成阶段 2（存储层）不做硬性依赖，但本阶段的 DeepSeekClient 会被阶段 4 的情景/语义记忆使用
- 环境变量 `DEEPSEEK_API_KEY` 已配置（用于 LLM 客户端测试）
- `models/bge-small-zh-v1.5/` 目录存在且包含完整的模型文件（已预置）

## 开发任务

### LocalEmbedder（embedding/local_embedder.py）
1. 实现 `LocalEmbedder` 类：
   - 构造函数参数：`model_name: str = "models/bge-small-zh-v1.5"`, `device: str = "cpu"`, `normalize: bool = True`, `cache_dir: Optional[str] = None`。
   - 使用**懒加载**模式：构造函数中不立即加载模型，仅保存配置参数。将 `self._model` 初始化为 `None`。
   - 内部属性 `self._model` 用于持有 `SentenceTransformer` 实例。

2. 实现内部方法 `_ensure_model_loaded()` —— 若 `self._model` 为 `None`，则：
   - 调用 `SentenceTransformer(model_name_or_path=self.model_name, device=self.device, cache_folder=self.cache_dir)` 加载模型。
   - 若加载失败（文件不存在、OOM、网络错误等），抛出 `ModelLoadError`，附带原始异常。
   - 执行一次空文本预热推理（`self._model.encode([""])`），触发 JIT 优化。
   - 使用 `logging.getLogger(__name__)` 输出 INFO 日志记录模型加载耗时。

3. 实现属性 `dimension`（`@property`） —— 调用 `_ensure_model_loaded()` 后返回 `self._model.get_sentence_embedding_dimension()`。

4. 实现 `embed(texts: Union[str, List[str]]) -> List[List[float]]`：
   - 若输入为 `str`，包裹为 `[texts]`。
   - 调用 `_ensure_model_loaded()`。
   - 使用 `self._model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)` 生成向量。
   - 输入校验：`texts` 为空列表时返回 `[]`。
   - 若推理过程抛出异常，捕获并重新抛出为 `EmbeddingError`。
   - 单个文本输入返回 `[[...]]`（长度为 1 的二维列表）。

5. 实现 `embed_query(query: str) -> List[float]`：
   - 调用 `_ensure_model_loaded()`。
   - 使用 `self._model.encode(query, normalize_embeddings=self.normalize, show_progress_bar=False)`。
   - 返回一维向量 `List[float]`（即 `embed()` 返回的 `[0]` 元素）。
   - 若 `query` 为空字符串，抛出 `ValueError`。

6. 实现 `warmup() -> None` —— 公开预热方法（等价于调用 `_ensure_model_loaded()`），供用户提前加载模型。

### DeepSeekClient（llm/deepseek_client.py）
7. 实现 `DeepSeekClient` 类：
   - 构造函数参数：`api_key: Optional[str] = None`, `model: str = "deepseek-chat"`, `base_url: str = "https://api.deepseek.com/v1"`, `timeout: float = 30.0`, `max_retries: int = 3`。
   - 若 `api_key` 为 `None`，从 `os.environ["DEEPSEEK_API_KEY"]` 读取。若仍为 `None` 或空字符串，抛出 `ConfigError("DEEPSEEK_API_KEY 未配置")`。
   - 使用 `openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)` 创建异步客户端（`max_retries=0` 是因为我们要自行实现重试逻辑，支持更精细的控制）。

8. 实现内部方法 `_retry_with_backoff(func, *args, **kwargs)` —— 通用重试逻辑：
   - 重试次数上限为 `max_retries`。
   - 指数退避：第 i 次重试等待 `2^(i-1)` 秒（首次重试 1s，第二次 2s，第三次 4s）。
   - 使用 `asyncio.sleep()` 实现等待。
   - 全部重试失败后，抛出 `LLMServiceError`，附带上一次异常和重试次数。
   - 使用 `logging.getLogger(__name__)` 在每次重试时输出 WARNING 日志。

9. 实现 `async chat(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, response_format: Optional[Dict[str, Any]] = None) -> str`：
   - 使用 `_retry_with_backoff` 包装 OpenAI API 调用。
   - 构造 `self._client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format)` 调用。
   - 返回 `response.choices[0].message.content`。
   - 若 `response_format` 为 `{"type": "json_object"}` 但响应内容不是合法 JSON，抛出 `LLMResponseParseError`，附带原始响应文本。
   - 记录 INFO 日志（含请求 token 数和耗时）。

10. 实现 `async extract_entities(text: str) -> Dict[str, Any]`：
    - 构建结构化的 system prompt，要求 LLM 从文本中提取实体、偏好和关系，返回 JSON。
    - 调用 `self.chat(messages=[{"role": "system", "content": extract_prompt}, {"role": "user", "content": text}], response_format={"type": "json_object"}, temperature=0.3)`。
    - 使用 Pydantic 校验返回的 JSON 结构（定义内部 `_EntityExtractionResult` 模型），确保包含 `entities`、`preferences`、`relations` 三个键。
    - 返回校验通过后的字典。
    - 若 JSON 解析或 Pydantic 校验失败，抛出 `LLMResponseParseError`。

### 测试（tests/ 目录）
11. 在 `tests/conftest.py` 中添加 fixtures：
    - `embedder` fixture：返回 `LocalEmbedder()` 实例（`scope="module"`，避免重复加载模型）。
    - `deepseek_client` fixture：当 `DEEPSEEK_API_KEY` 未配置时 `skip`；返回 `DeepSeekClient()`。
    - `mock_deepseek_client` fixture：使用 `unittest.mock.AsyncMock` 创建假 DeepSeekClient，供不需要真实 API 的测试使用。

12. 在 `tests/test_embedder.py` 中编写测试：
    - 懒加载：实例化后 `_model` 为 `None`。
    - 首次 `embed()` 触发加载，第二次 `embed()` 不重新加载（通过 mock `SentenceTransformer` 的调用次数验证）。
    - `embed` 单条文本返回 `[[0.1, 0.2, ...]]` 格式（512 维）。
    - `embed` 批量文本返回对应数量的向量列表。
    - `embed_query` 返回一维 `List[float]`。
    - 归一化检查：`embed_query("test")` 返回的向量各分量平方和约等于 1.0（容差 1e-4）。
    - `embed([])` 返回 `[]`。
    - `embed_query("")` 抛出 `ValueError`。
    - `dimension` 属性返回 512（bge-small-zh-v1.5 的维度）。
    - 模型加载失败时抛出 `ModelLoadError`（使用不存在的模型路径测试）。

13. 在 `tests/test_deepseek_client.py` 中编写测试：
    - API key 未配置时构造函数抛出 `ConfigError`。
    - `chat` 方法发送简单消息并获得非空响应（需真实 API key）。
    - 指数退避重试：mock `AsyncOpenAI` 前 2 次抛出 `APIConnectionError`，第 3 次成功，验证总共调用了 3 次。
    - 全部重试耗尽后抛出 `LLMServiceError`。
    - JSON mode：`chat(messages=[...], response_format={"type": "json_object"})` 返回合法 JSON 字符串。
    - JSON mode 解析失败：mock 返回非 JSON 内容，验证抛出 `LLMResponseParseError`。
    - `extract_entities` 返回的结构包含 `entities`、`preferences`、`relations` 三个键。
    - `extract_entities` 在 LLM 返回非法 JSON 时抛出 `LLMResponseParseError`。

## 验收标准
1. **LocalEmbedder 懒加载**
   - 场景：全新 LocalEmbedder 实例
   - 操作：`e = LocalEmbedder(); assert e._model is None; result = e.embed("测试文本")`
   - 预期：第一次调用触发模型加载（日志可见加载耗时），返回 512 维向量；第二次调用不再加载

2. **embed 与 embed_query 输出格式**
   - 场景：模型已加载
   - 操作：`vec2d = e.embed("hello"); vec1d = e.embed_query("hello")`
   - 预期：`len(vec2d) == 1`，`len(vec2d[0]) == 512`；`len(vec1d) == 512`；`vec2d[0] == vec1d`

3. **向量归一化**
   - 场景：归一化模式开启（默认）
   - 操作：`v = e.embed_query("测试文本"); norm = sum(x**2 for x in v) ** 0.5`
   - 预期：`abs(norm - 1.0) < 1e-4`

4. **DeepSeekClient 基本对话**
   - 场景：`DEEPSEEK_API_KEY` 已正确配置
   - 操作：`client = DeepSeekClient(); resp = await client.chat([{"role": "user", "content": "回复'你好'"}])`
   - 预期：返回非空字符串，包含"你好"或相关回复

5. **DeepSeekClient 重试机制**
   - 场景：Mock API 连接失败 2 次后成功
   - 操作：调用 `client.chat(...)`
   - 预期：最终返回正确响应，日志中出现 2 条 WARNING 重试记录，总调用次数为 3

6. **extract_entities 结构化提取**
   - 场景：输入文本"张三喜欢喝蓝山咖啡，他今年30岁"
   - 操作：`result = await client.extract_entities(text)`
   - 预期：`result["entities"]` 至少包含 1 个实体（如 `{"name": "张三", "type": "person", ...}`）；`result["preferences"]` 至少包含 1 条偏好（如咖啡相关）；`result["relations"]` 为列表

7. **LLM 响应解析失败处理**
   - 场景：Mock API 返回 `"这不是合法的 JSON"`
   - 操作：调用 `client.chat(messages=[...], response_format={"type": "json_object"})`
   - 预期：抛出 `LLMResponseParseError`，其 `details` 属性包含原始响应文本

8. **边界情况：空输入和无效输入**
   - 场景：LocalEmbedder 和 DeepSeekClient 均已就绪
   - 操作：`e.embed([])`、`e.embed_query("")`、构造无 API key 的 `DeepSeekClient()`
   - 预期：`embed([])` 返回 `[]`；`embed_query("")` 抛出 `ValueError`；无 key 的构造抛出 `ConfigError`

9. **性能指标：嵌入索引**
   - 场景：模型已预热
   - 操作：`e.embed(["测试文本"] * 50)`，记录耗时
   - 预期：50 条文本的嵌入耗时 < 1 秒（基于 BGE-small CPU 性能）

## 注意事项
- LocalEmbedder 首次加载取决于模型大小和磁盘速度，BGE-small 约 100MB 在 SSD 上通常 < 5 秒。
- DeepSeekClient 使用 `openai` 库（官方 Python SDK），因为 DeepSeek API 兼容 OpenAI 格式，`base_url` 指向 DeepSeek 即可。
- 重试逻辑仅对网络层面错误（`APIConnectionError`, `APITimeoutError`, `APIStatusError` 中 5xx）重试，4xx 错误（认证失败等）应立即抛出，不重试。
- 测试文件 `test_deepseek_client.py` 中所有依赖真实 API 的测试需标记 `@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")`。
- BGE 模型指令前缀：对于 `embed_query`，BGE 模型推荐在查询文本前添加 `"为这个句子生成表示以用于检索相关文章："` 前缀以获得更好的检索效果。`embed_query` 应自动添加此前缀。
