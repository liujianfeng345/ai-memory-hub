# CHANGES.md - 阶段 1：项目搭建与基础设施

## 概述

完成 `memory-agent` 项目的骨架搭建，建立了异常体系、配置管理、日志系统和数据模型。所有代码通过 ruff lint、ruff format、mypy 类型检查及 140 个单元测试。

## 修改的文件

### 新增文件 (20 个)

#### 项目工程配置 (4 个)
| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 项目元数据 (name=memory-agent, version=0.1.0, python>=3.10)、setuptools 构建系统、可选依赖组 (dev)、ruff/mypy/pytest 工具配置 |
| `requirements.txt` | 生产依赖：chromadb>=0.5.0, sentence-transformers, openai, httpx, pydantic>=2.0, pydantic-settings, python-dotenv |
| `requirements-dev.txt` | 开发依赖：pytest>=8.0, pytest-asyncio, pytest-cov, ruff, mypy, pip-audit |
| `.env.example` | 环境变量模板，包含所有可配置项 (DEEPSEEK_API_KEY, LOG_LEVEL, EMBEDDING_MODEL_NAME 等) 的说明和默认值 |

#### 包目录结构 (8 个)
| 文件 | 说明 |
|------|------|
| `memory_agent/__init__.py` | 包根 (暂空，阶段5填充 API 导出) |
| `memory_agent/py.typed` | PEP 561 类型标记 (空文件) |
| `memory_agent/core/__init__.py` | 核心业务逻辑模块 |
| `memory_agent/storage/__init__.py` | 持久化存储模块 |
| `memory_agent/embedding/__init__.py` | 嵌入模型模块 |
| `memory_agent/llm/__init__.py` | LLM 服务模块 |
| `memory_agent/models/__init__.py` | 数据模型模块 |
| `memory_agent/utils/__init__.py` | 工具模块 |
| `tests/__init__.py` | 测试套件 |

#### 源代码 (6 个)
| 文件 | 说明 |
|------|------|
| `memory_agent/utils/errors.py` | 自定义异常体系：基类 `MemoryAgentError` + 7 个具体子类 (ConfigError, StorageError, ModelLoadError, EmbeddingError, LLMServiceError, LLMResponseParseError, DimensionMismatchError) |
| `memory_agent/utils/config.py` | 配置管理：`MemoryConfig(BaseSettings)` 类，12 个配置项，支持 .env / 环境变量加载，API key 脱敏，字段校验 |
| `memory_agent/utils/logger.py` | 日志配置：`setup_logging()` 函数，支持 text/json 双格式输出，chromadb DEBUG 日志抑制 |
| `memory_agent/models/memory_item.py` | `MemoryType` 枚举 (working/episodic/semantic) 和 `MemoryItem` 基类 (含 UUID、时间戳、元数据) |
| `memory_agent/models/episode.py` | `Episode(MemoryItem)` 子类 (含 summary、embedding、importance、tags) |
| `memory_agent/models/entity.py` | `Entity` 模型 (含 entity_type 校验、confidence 范围校验、关联实体) |
| `memory_agent/models/consolidate_result.py` | `ConsolidateResult` 模型 (含各计数器、错误列表、dry_run 标志) |

#### 测试代码 (4 个)
| 文件 | 说明 |
|------|------|
| `tests/conftest.py` | `tmp_dir` fixture (临时目录)，`sample_config` fixture (假 API key + 临时路径的 MemoryConfig) |
| `tests/test_errors.py` | 异常体系测试 (52 项)：实例化、error_code、__str__ 格式、继承链、pickle 序列化、可用性验证 |
| `tests/test_config.py` | 配置管理测试 (48 项)：默认值、环境变量覆盖、脱敏、校验 (非法值拒绝)、log_level_int |
| `tests/test_models.py` | 数据模型测试 (40 项)：实例化、UUID 自生成、时间戳、继承关系、序列化往返、类型校验、confidence 范围 |

### 修改文件 (1 个)
| 文件 | 说明 |
|------|------|
| `memory_agent/utils/config.py` | 在 model_config 中添加 `extra="ignore"` 以兼容现有 .env 文件中的旧版环境变量名 |

## 核心逻辑

### 异常体系
- 采用统一基类 `MemoryAgentError(Exception)`，所有异常通过 `error_code` 类属性标识类型
- 支持 `details: dict` 传递结构化上下文信息
- 实现了 `__reduce__` 方法保证 pickle 序列化兼容
- `__str__` 返回 `[ERROR_CODE] message` 格式

### 配置管理
- 基于 `pydantic-settings.BaseSettings`，自动从 `.env` 和环境变量加载
- 优先级：环境变量 > .env 文件 > 默认值
- API key 通过 `field_serializer` 脱敏：保留前4位和后4位，中间用 `****` 替代
- 所有数值型和枚举型字段均有 `field_validator` 校验
- 通过 `extra="ignore"` 容忍未定义的环境变量

### 日志系统
- `setup_logging()` 配置标准 Python logging
- text 格式：`[时间戳] [级别] [模块名] 消息`
- JSON 格式：`{"timestamp": "...", "level": "...", "module": "...", "message": "..."}`
- 强制抑制 chromadb 和 httpx 的 DEBUG 日志

### 数据模型
| 模型 | 继承 | 关键特性 |
|------|------|---------|
| MemoryItem | BaseModel | UUID4 自生成、UTC 时间戳、MemoryType 枚举 |
| Episode | MemoryItem | 默认 memory_type=episodic、embedding、importance、tags |
| Entity | BaseModel | entity_type 校验 (5种)、confidence [0,1]、关联实体 |
| ConsolidateResult | BaseModel | 各计数器非负校验、errors 列表、dry_run 标志 |

## 验证结果

```
140 passed in 0.45s    (pytest)
All checks passed!     (ruff)
19 files already formatted (ruff format)
Success: no issues found (mypy)
```

## 注意事项

- `memory_agent/models/` 是 Python 包目录 (数据模型)，与根目录 `models/` (嵌入模型文件) 是不同的目录
- 根目录 `models/` 包含 `bge-small-zh-v1.5` 嵌入模型，其内部 `.gitignore` 已忽略所有文件
- `MemoryType` 使用 `(str, Enum)` 而非 `StrEnum` 以保证 Python 3.10 兼容性
- 现有的 `.env` 文件包含旧版环境变量名 (如 `MEMORY_AGENT_*`, `LLM_*`)，通过 `extra="ignore"` 配置容忍这些变量
- 阶段1 未实现的功能目录 (core, storage, embedding, llm) 仅包含占位 `__init__.py`，将在后续阶段填充

## 测试建议

- **异常测试**：已覆盖异常实例化、error_code 访问、继承链、pickle 序列化
- **配置测试**：已覆盖默认值、环境变量覆盖 (通过 monkeypatch)、脱敏、非法值拒绝
- **模型测试**：已覆盖自生成字段 (UUID/时间戳)、序列化往返、字段校验
- 后续阶段可通过 `sample_config` fixture 获取即用配置，通过 `tmp_dir` fixture 获取隔离的文件系统环境

## 已知限制

- 日志系统的 JSON 格式未包含完整的异常堆栈信息 (仅包含异常消息)，如需完整堆栈可在后续增强 `JsonFormatter`
- `MemoryConfig.model_dump()` 脱敏仅在序列化时生效，`config.deepseek_api_key` 直接访问仍返回完整值 (这是预期行为，仅在对外输出时脱敏)
- `setup_logging` 中使用 `# type: ignore` 注释避免循环导入，后续阶段提取配置到独立类型定义文件后可移除
