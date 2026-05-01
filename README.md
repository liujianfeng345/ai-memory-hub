# Memory Agent

智能记忆代理 —— 为 AI Agent 提供可插拔的记忆管理系统。

Memory Agent 模拟人类记忆的多层次结构，提供三种记忆类型：
- **工作记忆（Working Memory）**：基于关键词匹配的会话内临时记忆，支持 TTL 自动过期。
- **情节记忆（Episodic Memory）**：基于向量语义检索的对话/事件记忆，支持自动摘要生成。
- **语义记忆（Semantic Memory）**：从情节记忆中通过 LLM 提取的持久化知识图谱实体，支持关系管理。

Memory Agent 设计为可被外部 Agent 项目通过 `pip install` 直接集成的独立 Python 包。

## 安装

### 从 PyPI 安装（推荐）

```bash
pip install ai-memory-hub
```

### 从源码安装（开发模式）

```bash
git clone <your-repo-url>
cd ai-memory-hub
pip install -e .
```

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

## 快速开始

```python
import asyncio
from memory_agent import MemoryManager, MemoryConfig

async def main():
    # 1. 加载配置（自动从环境变量和 .env 文件加载）
    config = MemoryConfig()

    # 2. 初始化管理器（自动装配所有内部组件）
    manager = MemoryManager(config)

    # 3. 写入三种类型的记忆
    session = "demo-session"

    # 工作记忆 —— 会话内临时上下文
    await manager.remember(
        "用户正在学习 Python 异步编程",
        memory_type="working",
        session_id=session,
    )

    # 情节记忆 —— 对话/事件记录
    await manager.remember(
        "今天用户问了关于 asyncio 的问题，表现出对 Python 并发编程的浓厚兴趣",
        memory_type="episodic",
        session_id=session,
    )

    # 语义记忆 —— LLM 自动提取实体并持久化
    await manager.remember(
        "用户偏好：Python 编程语言，日常使用 VSCode 编辑器",
        memory_type="semantic",
    )

    # 4. 跨类型检索（自动聚合并排序）
    results = await manager.recall(
        query="编程相关",
        memory_type=None,  # 检索所有类型
        top_k=5,
        session_id=session,
    )
    for item in results:
        print(f"[{item.memory_type.value}] {item.content}")

    # 5. 记忆整合 —— 从情节记忆中提取知识更新语义记忆
    consolidate_result = await manager.consolidate(time_window_hours=24)
    print(f"新建实体: {consolidate_result.new_entities}")

    # 6. 清理会话工作记忆
    await manager.clear_session(session)

asyncio.run(main())
```

完整示例见 [`examples/basic_usage.py`](examples/basic_usage.py)。

## 配置

### 环境变量

Memory Agent 通过 `MemoryConfig`（基于 pydantic-settings）加载配置，支持 `.env` 文件和环境变量。

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | `""` | DeepSeek API 密钥（**必填**） |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 使用的 LLM 模型名称 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API 基础地址 |
| `DEEPSEEK_TIMEOUT` | `30.0` | API 请求超时秒数 |
| `DEEPSEEK_MAX_RETRIES` | `3` | 最大重试次数 |
| `EMBEDDING_MODEL_NAME` | `models/bge-small-zh-v1.5` | 嵌入模型路径 |
| `EMBEDDING_DEVICE` | `cpu` | 推理设备（cpu/cuda） |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB 持久化目录 |
| `DEFAULT_TTL_SECONDS` | `3600` | 工作记忆默认存活时间（秒） |
| `MAX_CONTENT_LENGTH` | `50000` | 内容最大字符数 |
| `SUMMARY_THRESHOLD` | `2000` | 触发自动摘要的字符数阈值 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### .env 文件示例

```bash
DEEPSEEK_API_KEY=sk-your-key-here
EMBEDDING_DEVICE=cpu
CHROMA_PERSIST_DIR=./data/chroma
LOG_LEVEL=INFO
```

## 模块架构

```
+--------------------------------------------------------------+
|                     ai-memory-hub 总体架构                      |
+--------------------------------------------------------------+
|                                                              |
|  用户代码 / 外部 Agent                                        |
|       |                                                      |
|       v                                                      |
|  +---------------------------+   memory_agent/__init__.py     |
|  |      MemoryManager        |   (公开 API 导出)              |
|  |      (总调度器)            |                                |
|  +-----+----------+----------+                               |
|        |          |          |                                |
|        v          v          v                                |
|  +----------+ +----------+ +----------+                      |
|  | Working  | | Episodic | | Semantic |   core/               |
|  | Memory   | | Memory   | | Memory   |   (核心模块)           |
|  +----+-----+ +----+-----+ +----+-----+                      |
|       |            |            |                             |
|       v            v            v                             |
|  +----------+ +----------+ +----------+                      |
|  |InMemory  | | ChromaDB | | ChromaDB |   storage/            |
|  |Store     | |(collection| |(collection|  (存储后端)           |
|  |          | | episodic) | | semantic)|                      |
|  +----------+ +----+-----+ +----+-----+                      |
|                    |            |                             |
|                    v            v                             |
|              +----------------------+                        |
|              |   LocalEmbedder      |  embedding/             |
|              | (BGE-small-zh-v1.5)  |  (嵌入模型)             |
|              +----------------------+                        |
|                                                              |
|              +----------------------+                        |
|              |   DeepSeekClient     |  llm/                  |
|              | (DeepSeek API/       |  (LLM 客户端)           |
|              |  OpenAI 兼容)        |                        |
|              +----------------------+                        |
|                                                              |
|  数据模型层 (models/):                                        |
|  MemoryItem, Episode, Entity, ConsolidateResult              |
|                                                              |
|  工具层 (utils/):                                             |
|  MemoryConfig, 异常体系, Logger                              |
+--------------------------------------------------------------+
```

## 数据流

```
  remember("content", type="episodic")
       |
       v
  MemoryManager.remember()
       |
       +--> type="working"  --> WorkingMemory.add()    --> InMemoryStore
       |
       +--> type="episodic" --> EpisodicMemory.add_episode()
       |                           |
       |                           +--> LocalEmbedder.embed()
       |                           +--> ChromaStore.add()
       |                           +--> [LLM 摘要] (可选)
       |
       +--> type="semantic" --> LLM Client.extract_entities()
                                    |
                                    +--> SemanticMemory.add_entity()
                                    +--> SemanticMemory.add_relation()

  recall("query", type=None)
       |
       v
  MemoryManager.recall()
       |
       +--> asyncio.gather() 并行检索
       |       |
       |       +--> WorkingMemory.search()
       |       +--> EpisodicMemory.search()
       |       +--> SemanticMemory.search_entities()
       |
       +--> 聚合排序（工作记忆优先） --> List[MemoryItem]

  consolidate(time_window_hours=24)
       |
       v
  EpisodicMemory.get_recent()
       |
       +--> LLM Client (few-shot 提示词提取)
       |
       +--> 实体合并/创建 --> SemanticMemory
       +--> 关系建立 --> SemanticMemory.add_relation()
       |
       +--> ConsolidateResult
```

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 仅运行集成测试
pytest tests/test_manager.py -v

# 生成覆盖率报告
pytest tests/ -v --cov=memory_agent --cov-report=term
```

## 限制与注意事项

- **ChromaDB 并发限制**：`PersistentClient` 不支持多进程并发写入同一持久化目录，请勿在多个进程中共享同一 ChromaDB 目录。
- **嵌入模型**：首次运行时会下载 BGE 嵌入模型（约 100MB），请确保网络畅通或提前将模型放置于 `models/bge-small-zh-v1.5` 目录。
- **LLM 依赖**：`remember("semantic")` 和 `consolidate()` 依赖 DeepSeek API，请在运行前配置有效的 `DEEPSEEK_API_KEY`。

## 许可证

MIT License
