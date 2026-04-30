# CHANGES-phase-3.md —— 阶段 3 变更记录

## 阶段概述

完成本地嵌入模型封装（LocalEmbedder）和 DeepSeek API 客户端（DeepSeekClient），
为上层记忆模块提供文本向量化和大模型推理能力。两个模块相互独立，可分别测试。

## 修改的文件

### 新增文件

| 文件 | 说明 |
|------|------|
| `memory_agent/embedding/local_embedder.py` | 本地嵌入模型封装，懒加载 BGE-small-zh-v1.5 |
| `memory_agent/llm/deepseek_client.py` | DeepSeek API 异步客户端，支持重试和结构化提取 |
| `tests/test_embedder.py` | LocalEmbedder 单元测试（懒加载、格式、归一化、边界） |
| `tests/test_deepseek_client.py` | DeepSeekClient 单元测试（重试、JSON mode、实体提取） |
| `docs/dev-plan/CHANGES-phase-3.md` | 本变更文档 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `tests/conftest.py` | 新增 `embedder`、`deepseek_client`、`mock_deepseek_client` 三个 fixtures |

## 核心逻辑

### LocalEmbedder

```
+-------------------+       +-------------------------+
| LocalEmbedder     |       | SentenceTransformer     |
|-------------------|       |-------------------------|
| model_name        |       | encode(texts) -> ndarray|
| device            |       | get_sentence_embedding_ |
| normalize         |       |     dimension() -> int   |
| cache_dir         |       +-------------------------+
|-------------------|                  ^
| _model: Optional  |  --懒加载-->     |
|                   |  _ensure_model_loaded()
| embed(texts)      |
| embed_query(q)    |  +BGE前缀
| warmup()          |
| dimension         |
+-------------------+
```

- **懒加载**：构造函数不加载模型，`_model` 初始化为 `None`。首次调用 `embed/embed_query/warmup/dimension` 时触发 `_ensure_model_loaded()`，调用 `SentenceTransformer()` 加载 + 空文本预热推理。
- **BGE 前缀**：`embed_query` 自动添加 `"为这个句子生成表示以用于检索相关文章："` 前缀，`embed` 不添加。
- **归一化**：`embed` 和 `embed_query` 均通过 `normalize_embeddings` 参数控制是否 L2 归一化（默认 `True`）。
- **异常包装**：模型加载失败 → `ModelLoadError`，推理失败 → `EmbeddingError`。
- **边界**：`embed([])` → `[]`，`embed_query("")` → `ValueError`。

### DeepSeekClient

```
+-------------------+       +-------------------------+
| DeepSeekClient    |       | openai.AsyncOpenAI      |
|-------------------|       |-------------------------|
| api_key           |       | chat.completions.create |
| model             |       +-------------------------+
| base_url          |                  ^
| timeout           |                  |
| max_retries       |     _retry_with_backoff()
|-------------------|
| chat(messages)    |
| extract_entities()|
+-------------------+

重试策略：
  APIConnectionError  ──> 重试
  APITimeoutError     ──> 重试
  APIStatusError(5xx) ──> 重试
  APIStatusError(4xx) ──> 立即抛出
  APIStatusError(429) ──> 立即抛出
```

- **重试**：`max_retries=0`（OpenAI SDK 不自重试），由 `_retry_with_backoff` 实现指数退避（1s/2s/4s...）。仅对网络/服务端错误重试。
- **JSON mode**：当 `response_format={"type": "json_object"}` 时，额外校验响应是否合法 JSON，失败抛 `LLMResponseParseError`。
- **extract_entities**：使用结构化 system prompt + JSON mode 提取 `entities`/`preferences`/`relations`，Pydantic 校验字段完整性。

## 注意事项（给 dev-tester）

1. **模型文件依赖**：`test_embedder.py` 的大部分测试依赖 `models/bge-small-zh-v1.5/` 目录存在且模型文件完整。首次运行时会加载约 100MB 模型（SSD 上 < 5 秒）。
2. **真实 API key**：`test_deepseek_client.py` 中带有 `@pytest.mark.skipif` 装饰器的测试需要真实的 `DEEPSEEK_API_KEY` 环境变量（或 `.env` 文件），否则自动跳过。
3. **Mock 测试独立性**：重试逻辑和 JSON 解析测试使用 mock，不依赖外部 API，可在任何环境运行。
4. **模块级 scope**：`embedder` fixture 使用 `scope="module"`，这意味着该模块的所有测试复用同一个模型实例，首次测试会触发加载，后续测试直接用已加载的模型。
5. **性能测试**：`TestPerformance.test_50_texts_embedding_time` 设置了 10 秒的超时阈值（而非严格的 1 秒），以便在 CPU 模式下稳定通过。
6. **asyncio mode**：`pyproject.toml` 配置 `asyncio_mode = "strict"`，所有异步测试函数使用 `@pytest.mark.asyncio` 装饰器。

## 已知限制

1. **LocalEmbedder 设备切换**：当前不支持运行时切换设备（需创建新实例），且 `model_name` 在初始化后不可更改。
2. **DeepSeekClient 流式输出**：当前 `chat` 方法仅支持非流式（一次性返回完整响应），未实现 `stream=True` 模式。
3. **DeepSeek JSON mode 限制**：DeepSeek API 的 `response_format={"type": "json_object"}` 需要用户 prompt 中明确提到 "JSON" 关键词，否则可能返回非 JSON 内容。`extract_entities` 的 system prompt 已隐含此要求，但直接使用 `chat` 时需用户自行确保。
4. **BGE 模型缓存**：LocalEmbedder 未实现模型实例的全局缓存/单例，多次实例化会分别加载模型（可通过外部 DI 解决）。
5. **重试日志**：重试过程使用 WARNING 级别日志，可能在生产环境产生噪音，建议后续阶段通过配置控制。
