# 阶段 3 测试报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 阶段 | 阶段 3：嵌入层与大模型客户端 |
| 阶段文档 | `docs/dev-plan/phase-3-embedding-llm.md` |
| 变更报告 | `docs/dev-plan/CHANGES-phase-3.md` |
| 测试日期 | 2026-04-30 |
| 测试分支 | main |
| 环境 | Python 3.12.7, Windows 10, CPU 模式 |

## 测试依据

依据阶段文档中定义的 9 项验收标准，逐条设计测试用例并验证。

## 静态检查结果

| 检查项 | 工具 | 结果 |
|--------|------|------|
| Linter | ruff | 全部通过（0 errors） |
| 类型检查 | mypy | 阶段 3 核心文件（`local_embedder.py`, `deepseek_client.py`）无类型错误。`conftest.py:174-176` 的 AsyncMock 赋值和 `chroma_store.py`（阶段 2）的第三方库类型问题是预期行为。 |

## 单元/集成测试结果

### Embedder 测试 (`tests/test_embedder.py`)
```
TestLazyLoading::test_model_is_none_after_init           PASSED
TestLazyLoading::test_first_embed_triggers_loading       PASSED
TestLazyLoading::test_warmup_loads_model                 PASSED
TestEmbedMethod::test_embed_single_string_returns_2d_list PASSED
TestEmbedMethod::test_embed_output_dimension             PASSED
TestEmbedMethod::test_embed_batch_returns_correct_count  PASSED
TestEmbedMethod::test_embed_empty_list_returns_empty_list PASSED
TestEmbedQuery::test_embed_query_returns_1d_list         PASSED
TestEmbedQuery::test_embed_query_empty_string_raises_valueerror PASSED
TestEmbedQuery::test_embed_and_embed_query_consistency   PASSED
TestNormalization::test_embed_query_normalized           PASSED
TestNormalization::test_embed_normalized                 PASSED
TestNormalization::test_no_normalize_mode                PASSED
TestDimensionProperty::test_dimension_returns_512        PASSED
TestDimensionProperty::test_dimension_triggers_lazy_loading PASSED
TestErrorHandling::test_model_load_error_on_invalid_path PASSED
TestErrorHandling::test_embedding_error_during_inference PASSED
TestErrorHandling::test_model_load_error_preserves_original PASSED
TestWarmup::test_warmup_with_real_model                  PASSED
TestPerformance::test_50_texts_embedding_time            PASSED
```
**结果：20/20 通过**（耗时 19.75s，含首次模型加载）

### DeepSeek 客户端测试 (`tests/test_deepseek_client.py`)
```
TestIsRetryableError::test_connection_error_is_retryable PASSED
TestIsRetryableError::test_timeout_error_is_retryable    PASSED
TestIsRetryableError::test_5xx_error_is_retryable        PASSED
TestIsRetryableError::test_4xx_error_is_not_retryable    PASSED
TestIsRetryableError::test_429_error_is_not_retryable    PASSED
TestConfig::test_constructor_uses_env_api_key            PASSED
TestConfig::test_constructor_raises_config_error_when_no_key PASSED
TestConfig::test_constructor_accepts_explicit_key        PASSED
TestConfig::test_default_values                          PASSED
TestRetryWithBackoff::test_retry_succeeds_after_two_failures PASSED
TestRetryWithBackoff::test_all_retries_exhausted         PASSED
TestRetryWithBackoff::test_4xx_error_does_not_retry      PASSED
TestRetryWithBackoff::test_retry_with_timeout_errors     PASSED
TestChatBasic::test_chat_returns_nonempty_response       PASSED (真实 API)
TestJsonMode::test_chat_json_mode_returns_valid_json     PASSED (真实 API)
TestJsonMode::test_chat_json_mode_parse_failure          PASSED
TestJsonMode::test_chat_json_mode_valid_response_no_error PASSED
TestExtractEntities::test_extract_entities_returns_required_keys PASSED (真实 API)
TestExtractEntities::test_extract_entities_invalid_json  PASSED
TestExtractEntities::test_extract_entities_missing_keys  PASSED
TestExtractEntities::test_extract_entities_valid_structure PASSED
```
**结果：21/21 通过**（耗时 18.67s，含真实 API 调用 3 项）

## 验收标准逐条验证

### 验收标准 1：LocalEmbedder 懒加载

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 1.1 | 全新实例 | `e = LocalEmbedder(); assert e._model is None` | `_model` 为 `None` | `_model is None: True` | **通过** |
| 1.2 | 首次调用 | `e.embed("测试文本")` | 触发加载，返回 512 维向量 | 返回 `[1]x[512]` 向量，日志可见加载耗时 | **通过** |
| 1.3 | 第二次调用 | `e.embed("再次测试")` | 不重新加载模型 | `e._model` 与首次加载后为同一对象 | **通过** |

### 验收标准 2：embed 与 embed_query 输出格式

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 2.1 | embed 单条文本 | `vec2d = e.embed("hello")` | `len(vec2d) == 1`, `len(vec2d[0]) == 512` | `1 x 512` | **通过** |
| 2.2 | embed_query | `vec1d = e.embed_query("hello")` | `len(vec1d) == 512` | `512` 维一维列表 | **通过** |
| 2.3 | 维度一致性 | 比较两者维度 | 维度一致（均为 512） | 相同维度，且因 BGE 前缀存在差异 | **通过** |

> 注：`vec2d[0] == vec1d` 不严格成立，因为 `embed_query` 会自动添加 BGE 检索前缀（`"为这个句子生成表示以用于检索相关文章："`），这是正确的设计行为。

### 验收标准 3：向量归一化

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 3.1 | embed_query 归一化 | `v = e.embed_query("测试文本"); norm = sqrt(sum(x**2))` | `abs(norm - 1.0) < 1e-4` | 模长 = 1.0000000176（误差 1.76e-8） | **通过** |
| 3.2 | embed 归一化 | `v2 = e.embed(["测试嵌入"])[0]; norm = sqrt(sum(x**2))` | `abs(norm - 1.0) < 1e-4` | 模长 = 1.0000000138（误差 1.38e-8） | **通过** |

### 验收标准 4：DeepSeekClient 基本对话

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 4.1 | 简单对话 | `await client.chat([{"role": "user", "content": "请只回复'你好'两个字"}])` | 返回非空字符串，包含"你好" | 返回 `"你好"` | **通过** |

### 验收标准 5：DeepSeekClient 重试机制

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 5.1 | Mock 前 2 次失败 | Mock API 前 2 次抛出 `APIConnectionError`，第 3 次返回成功 | 最终返回正确响应，总调用 3 次 | 返回 `"最终响应"`，`call_count = 3`，日志可见 WARNING 重试记录 | **通过** |

### 验收标准 6：extract_entities 结构化提取

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 6.1 | 实体提取 | `await client.extract_entities("张三喜欢喝蓝山咖啡，他今年30岁")` | `entities` 至少 1 个实体 | 2 个实体（张三、蓝山咖啡） | **通过** |
| 6.2 | 偏好提取 | 同上 | `preferences` 至少 1 条偏好 | 1 条偏好（咖啡相关） | **通过** |
| 6.3 | 关系提取 | 同上 | `relations` 为列表 | 列表，长度 = 1 | **通过** |

### 验收标准 7：LLM 响应解析失败处理

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 7.1 | JSON mode 非法响应 | 调用 `chat(response_format={"type": "json_object"})`，Mock 返回 `"这不是合法的 JSON"` | 抛出 `LLMResponseParseError`，`details` 包含原始响应 | `LLMResponseParseError` 抛出，`details["original_response"] = "这不是合法的 JSON"` | **通过** |

### 验收标准 8：边界情况：空输入和无效输入

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 8.1 | 空列表嵌入 | `e.embed([])` | 返回 `[]` | `[]` | **通过** |
| 8.2 | 空字符串查询 | `e.embed_query("")` | 抛出 `ValueError` | `ValueError("embed_query 的 query 参数不能为空字符串")` | **通过** |
| 8.3 | 无 API key 构造 | `DeepSeekClient(api_key=None)` 且无环境变量 | 抛出 `ConfigError` | `ConfigError("[CONFIG_ERROR] DEEPSEEK_API_KEY 未配置")` | **通过** |

### 验收标准 9：性能指标：嵌入索引（50条耗时 < 1秒）

| 用例 | 场景 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|------|
| 9.1 | 50 条文本嵌入 | `e.embed(["测试文本"] * 50)` 记录耗时 | < 1 秒 | **0.1221s** | **通过** |
| 9.2 | 输出正确性 | 验证返回 50 条 512 维向量 | `len(result) == 50`, `len(result[0]) == 512` | 通过 | **通过** |

## 验收结论

```
【验收通过】所有 9 项验收标准均通过。
```

- 单元测试：Embedder 20/20，DeepSeek 21/21，全部通过
- 集成测试：真实 DeepSeek API 调用 3 项全部通过
- 静态检查：ruff lint 0 errors，mypy 阶段 3 核心文件无类型错误
- 性能：50 条文本嵌入耗时 0.1221s，远低于 1 秒阈值
- 边界情况：空输入、无效输入、异常路径均按预期处理

## 测试环境备注

- `DEEPSEEK_API_KEY` 已配置真实密钥，依赖真实 API 的测试（标准 4、标准 6 中的部分验证）均已执行并确认通过
- 模型文件 `models/bge-small-zh-v1.5/` 完整可用（包含 `model.safetensors`, `tokenizer.json` 等全部文件）
- 验证脚本：`tests/verify_phase3.py`（临时文件，可用于复现）
