# 阶段 1：项目搭建与基础设施

## 阶段目标
完成项目骨架搭建，建立异常体系、配置管理、日志系统和数据模型，使得后续任何模块的开发都有统一的基础设施可依赖。

## 前置条件
- 无（本阶段为项目起点）

## 开发任务

### 项目工程配置
1. 在项目根目录创建 `pyproject.toml`，声明 `[project]` 元数据（name=`memory-agent`, version=`0.1.0`, python=`>=3.10`）、`[project.optional-dependencies]`（dev 依赖组）、`[build-system]`（setuptools）。
2. 创建 `requirements.txt`，列出生产依赖：`chromadb>=0.5.0`, `sentence-transformers`, `openai`（兼容 DeepSeek API）, `httpx`, `pydantic>=2.0`, `pydantic-settings`, `python-dotenv`。
3. 创建 `requirements-dev.txt`，列出开发依赖：`pytest>=8.0`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, `pip-audit`。
4. 创建 `.env.example`，包含所有可配置环境变量的说明和默认值（`DEEPSEEK_API_KEY`, `LOG_LEVEL`, `EMBEDDING_MODEL_NAME`, `CHROMA_PERSIST_DIR`, `DEFAULT_TTL_SECONDS`, `MAX_CONTENT_LENGTH` 等）。
5. 创建 `.gitignore`，忽略 `__pycache__/`, `*.egg-info/`, `.env`, `data/`, `models/`（若模型已预置则单独处理）等。

### 包目录结构
6. 创建 `memory_agent/` 包根目录及所有子包目录结构：
   ```
   memory_agent/
   ├── __init__.py       # 暂为空，阶段5填充公开API导出
   ├── py.typed          # PEP 561 类型标记（空文件）
   ├── core/
   │   └── __init__.py
   ├── storage/
   │   └── __init__.py
   ├── embedding/
   │   └── __init__.py
   ├── llm/
   │   └── __init__.py
   ├── models/
   │   └── __init__.py
   └── utils/
       └── __init__.py
   ```
7. 在 `tests/` 目录下创建 `__init__.py` 和 `conftest.py`（暂为空框架，后续阶段逐步填充 fixtures）。

### 自定义异常体系（utils/errors.py）
8. 定义异常基类 `MemoryAgentError`，包含 `error_code: str`、`message: str`、`details: Optional[Dict[str, Any]]`（默认为 None）。构造函数接收 `message` 和可选 `details`。
9. 实现以下具体异常子类（每个子类设定对应的 `error_code` 类属性）：
   - `ConfigError` — `"CONFIG_ERROR"`
   - `StorageError` — `"STORAGE_ERROR"`
   - `ModelLoadError` — `"MODEL_LOAD_ERROR"`
   - `EmbeddingError` — `"EMBEDDING_ERROR"`
   - `LLMServiceError` — `"LLM_SERVICE_ERROR"`
   - `LLMResponseParseError` — `"LLM_RESPONSE_PARSE_ERROR"`
   - `DimensionMismatchError` — `"DIMENSION_MISMATCH_ERROR"`
10. 确保所有异常类可被 pickle 序列化，且 `__str__` 方法返回包含 `error_code` 和 `message` 的格式化字符串。

### 配置管理（utils/config.py）
11. 使用 `pydantic-settings` 的 `BaseSettings` 定义 `MemoryConfig` 类，包含以下配置项（带默认值和环境变量映射）：

    | 属性 | 类型 | 默认值 | 环境变量 | 说明 |
    |------|------|--------|---------|------|
    | deepseek_api_key | str | "" | DEEPSEEK_API_KEY | DeepSeek API 密钥 |
    | deepseek_model | str | "deepseek-chat" | DEEPSEEK_MODEL | 使用的模型名称 |
    | deepseek_base_url | str | "https://api.deepseek.com/v1" | DEEPSEEK_BASE_URL | API 基础地址 |
    | deepseek_timeout | float | 30.0 | DEEPSEEK_TIMEOUT | API 超时秒数 |
    | deepseek_max_retries | int | 3 | DEEPSEEK_MAX_RETRIES | 最大重试次数 |
    | embedding_model_name | str | "models/bge-small-zh-v1.5" | EMBEDDING_MODEL_NAME | 嵌入模型路径 |
    | embedding_device | str | "cpu" | EMBEDDING_DEVICE | 推理设备 |
    | chroma_persist_dir | str | "./data/chroma" | CHROMA_PERSIST_DIR | ChromaDB 持久化目录 |
    | default_ttl_seconds | int | 3600 | DEFAULT_TTL_SECONDS | 工作记忆默认 TTL |
    | log_level | str | "INFO" | LOG_LEVEL | 日志级别 |
    | max_content_length | int | 50000 | MAX_CONTENT_LENGTH | 内容最大字符数 |
    | summary_threshold | int | 2000 | SUMMARY_THRESHOLD | 触发自动摘要的字符数阈值 |

12. `MemoryConfig` 需实现：
    - `model_config` 设置为 `env_file = ".env"`, `env_file_encoding = "utf-8"`, `case_sensitive = False`。
    - 重写 `model_dump()` 方法（或通过 `field_serializer`），对 `deepseek_api_key` 进行脱敏处理：仅保留前4位和后4位，中间用 `****` 替代。

### 日志配置（utils/logger.py）
13. 实现 `setup_logging(config: MemoryConfig) -> None` 函数，配置 Python `logging` 标准库：
    - 日志级别从 `config.log_level` 读取。
    - 默认输出到 `stderr`。
    - 支持两种格式（通过 `LOGGING_FORMAT` 环境变量切换，默认 `text`）：
      - `text`：可读格式 `[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s`
      - `json`：结构化 JSON 格式，每行一条，包含 `timestamp`, `level`, `module`, `message` 字段。
    - 禁止 `chromadb` 的 DEBUG 日志输出，在函数中显式设置 `logging.getLogger("chromadb").setLevel(logging.WARNING)`。

### 数据模型（models/ 目录）
14. 在 `models/memory_item.py` 中定义：
    - `MemoryType` 枚举（`Literal` 或 `StrEnum`）：`working`, `episodic`, `semantic`。
    - `MemoryItem` Pydantic BaseModel，字段：`id: str`（UUID）、`content: str`、`memory_type: MemoryType`、`created_at: datetime`（默认工厂为 UTC now）、`updated_at: datetime`（同）、`metadata: Dict[str, Any] = {}`、`session_id: Optional[str] = None`。

15. 在 `models/episode.py` 中定义 `Episode(MemoryItem)`，追加字段：`summary: Optional[str] = None`、`embedding: Optional[List[float]] = None`、`importance: float = 1.0`、`tags: List[str] = []`。

16. 在 `models/entity.py` 中定义 `Entity(BaseModel)`，字段：`id: str`、`name: str`、`entity_type: str`（值为 `person`/`organization`/`topic`/`preference`/`fact`）、`description: str`、`attributes: Dict[str, Any] = {}`、`embedding: Optional[List[float]] = None`、`related_entities: List[str] = []`、`created_at: datetime`、`updated_at: datetime`、`confidence: float = 1.0`。

17. 在 `models/consolidate_result.py` 中定义 `ConsolidateResult(BaseModel)`，字段：`new_entities: int = 0`、`updated_entities: int = 0`、`new_preferences: int = 0`、`updated_preferences: int = 0`、`new_relations: int = 0`、`episodes_processed: int = 0`、`errors: List[str] = []`、`dry_run: bool = False`。

### 测试基础设施
18. 在 `tests/conftest.py` 中定义（或预留接口）：
    - `tmp_dir` fixture：使用 `tmp_path` 创建临时目录，供后续阶段测试使用。
    - `sample_config` fixture：返回一个使用临时路径和假 API key 的 `MemoryConfig` 实例。
19. 在 `tests/` 目录下创建 `test_errors.py`，测试：
    - 所有异常类型可被正确实例化并携带 `error_code`。
    - `MemoryAgentError` 的 `__str__` 输出包含 `error_code` 和 `message`。
    - 异常继承链正确（子类 is-a `MemoryAgentError`）。

20. 在 `tests/` 目录下创建 `test_config.py`，测试：
    - 默认值正确性。
    - 环境变量覆盖默认值。
    - `model_dump()` 对 API key 脱敏。
    - 非法值（如 `deepseek_timeout = 0`）应通过 Pydantic validator 报错（添加 `field_validator` 校验）。

21. 在 `tests/` 目录下创建 `test_models.py`，测试：
    - 所有数据模型可正常实例化。
    - `MemoryItem` 自动生成 UUID 和时间戳。
    - `Episode` 继承自 `MemoryItem` 且包含子类字段。
    - `Entity` 的 `entity_type` 非法值时校验失败。

## 验收标准
1. **项目可安装（开发模式）**
   - 场景：Python 3.10+ 环境已就绪
   - 操作：在项目根目录执行 `pip install -e ".[dev]"`
   - 预期：安装成功，`pip list` 中显示 `memory-agent` 版本 `0.1.0`

2. **所有 `__init__.py` 和目录结构就位**
   - 场景：项目已安装
   - 操作：执行 `python -c "import memory_agent; from memory_agent import utils, models, core, storage, embedding, llm"`
   - 预期：所有子包导入成功，无 `ModuleNotFoundError`

3. **自定义异常可正常使用**
   - 场景：Python 交互环境
   - 操作：`from memory_agent.utils.errors import ConfigError; raise ConfigError("missing key", {"key": "DEEPSEEK_API_KEY"})`
   - 预期：抛出 `ConfigError`，`str(e)` 包含 `"CONFIG_ERROR"` 和 `"missing key"`，`e.details` 为 `{"key": "DEEPSEEK_API_KEY"}`

4. **配置对象从环境变量加载**
   - 场景：设置 `DEEPSEEK_API_KEY=sk-test12345678` 和 `LOG_LEVEL=DEBUG` 环境变量
   - 操作：实例化 `MemoryConfig()`
   - 预期：`config.deepseek_api_key == "sk-test12345678"`，`config.log_level == "DEBUG"`，其他项保持默认值

5. **配置脱敏不泄露密钥**
   - 场景：配置对象持有 `deepseek_api_key = "sk-abcdefgh12345678"`
   - 操作：调用 `config.model_dump()` 并输出 `deepseek_api_key` 字段
   - 预期：返回值为 `"sk-a****5678"` 或类似脱敏形式

6. **日志 JSON 格式输出**
   - 场景：设置 `LOGGING_FORMAT=json`
   - 操作：调用 `setup_logging(config)` 后使用 `logging.getLogger(__name__).info("test")`
   - 预期：stderr 输出一行 JSON，至少包含 `{"timestamp": ..., "level": "INFO", "module": "...", "message": "test"}`

7. **数据模型序列化与反序列化**
   - 场景：创建一个 `MemoryItem` 实例
   - 操作：`item = MemoryItem(content="test", memory_type="episodic"); json_str = item.model_dump_json(); restored = MemoryItem.model_validate_json(json_str)`
   - 预期：`restored.content == "test"`, `restored.memory_type.value == "episodic"`, `restored.id` 为有效 UUID 格式

8. **测试套件可运行**
   - 场景：阶段1 所有代码和测试文件就绪
   - 操作：在项目根目录执行 `pytest tests/test_errors.py tests/test_config.py tests/test_models.py -v`
   - 预期：所有测试通过，无失败或错误

## 注意事项
- `py.typed` 必须是空文件存在，否则 PEP 561 类型检查器无法识别包的类型注解。
- `MemoryConfig` 使用 `pydantic-settings` 而非手动读取 `.env`，以保持与其他 Pydantic 模型的一致性。
- 异常类的 `error_code` 设计为类属性，方便调用方通过 `type(exc).error_code` 做程序化判断。
- 日志配置中必须抑制 `chromadb` 的 DEBUG 日志，否则后续阶段测试输出会过于冗长。
