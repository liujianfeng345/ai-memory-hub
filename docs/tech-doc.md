# 技术文档：memory-agent（记忆智能体）

## 1. 概述

### 项目目标
构建一个通用的、可复用的 Python 记忆模块（库/SDK），为上层智能体应用提供多层次记忆能力。该模块以独立 Python 库的形式发布，其他应用通过 `pip install` 或源码依赖的方式集成，无需额外部署独立服务。

### 用户故事

| 角色 | 核心目标 |
|------|---------|
| **智能体开发者** | 在自己的 Agent 应用中集成记忆能力，仅需几行代码即可让 Agent "记住"用户偏好、历史对话和当前上下文，无需自行实现存储和检索逻辑 |
| **最终用户** | 与集成了记忆模块的智能体交互时，智能体能够记住个人偏好、历史话题，提供连贯且个性化的对话体验 |
| **运维人员** | 零外部依赖部署，所有组件本地运行；通过环境变量配置 API 密钥即可启动 |

### 核心功能

1. **工作记忆（Working Memory）**：存储当前会话/任务的上下文，支持 TTL 过期机制，会话结束自动清理。
2. **情景记忆（Episodic Memory）**：存储历史对话和事件的摘要，支持基于语义相似度的检索召回。
3. **语义记忆（Semantic Memory）**：存储用户知识图谱，包括实体、偏好、关系等结构化知识，支持查询与更新。
4. **记忆整合（Memory Consolidation）**：利用大模型自动从情景记忆中提取结构化知识，更新语义记忆。

---

## 2. 技术方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        应用层（调用方 Agent）                          │
│                                                                     │
│    from memory_agent import MemoryManager                           │
│    manager = MemoryManager()                                        │
│    manager.remember(...)   # 写入记忆                               │
│    manager.recall(...)     # 检索记忆                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MemoryAgent 核心层                               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    MemoryManager（总调度器）                     │ │
│  │  - 路由写入请求到对应记忆类型                                    │ │
│  │  - 聚合跨类型检索结果                                           │ │
│  │  - 触发记忆整合（Consolidation）                                 │ │
│  └───────┬──────────────┬───────────────┬─────────────────────────┘ │
│          │              │               │                           │
│          ▼              ▼               ▼                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐           │
│  │ WorkingMemory│ │EpisodicMemory│ │ SemanticMemory   │           │
│  │              │ │              │ │                  │           │
│  │ - 内存存储   │ │ - 向量检索   │ │ - 向量检索       │           │
│  │ - TTL 过期   │ │ - 时间过滤   │ │ - 知识图谱存储   │           │
│  │ - 会话绑定   │ │ - 摘要存储   │ │ - 实体/关系管理  │           │
│  └──────┬───────┘ └──────┬───────┘ └───────┬──────────┘           │
│         │                │                  │                       │
│         ▼                ▼                  ▼                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     存储层（Storage Backend）                   │  │
│  │                                                               │  │
│  │  ┌──────────────────┐    ┌───────────────────────┐           │  │
│  │  │  InMemoryStore   │    │   ChromaStore          │           │  │
│  │  │  (dict + TTL)   │    │   (ChromaDB 封装)      │           │  │
│  │  └──────────────────┘    └───────────────────────┘           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     基础设施层                                  │  │
│  │                                                               │  │
│  │  ┌──────────────────┐  ┌──────────────────┐                  │  │
│  │  │  LocalEmbedder   │  │  DeepSeekClient  │                  │  │
│  │  │  (BGE 模型)      │  │  (大模型 API)    │                  │  │
│  │  └──────────────────┘  └──────────────────┘                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈选型

| 层面 | 技术选择 | 版本要求 | 选型理由 |
|------|---------|---------|---------|
| **编程语言** | Python | 3.10+ | Agent 生态的主流语言；丰富的 ML/NLP 库支持；用户已配置 Python 环境 |
| **向量数据库** | ChromaDB | 0.5.x+ | 本地轻量部署，零配置启动；Python 原生 SDK；嵌入式模式无需独立服务进程；支持元数据过滤（实现时间范围检索）；Apache 2.0 开源协议 |
| **嵌入模型** | sentence-transformers + BGE | 最新稳定版 | 完全离线运行，无需网络请求；BGE（BAAI/bge-small-zh-v1.5）为中文语义检索优化，512 维向量，模型仅约 100MB；模型已在 `models/bge-small-zh-v1.5/` 本地预置，无需联网下载；sentence-transformers 提供统一接口，切换模型方便 |
| **大模型 API** | DeepSeek API（deepseek-chat） | - | 用户已配置 DEEPSEEK_API_KEY；中文能力强；API 兼容 OpenAI 格式，便于后续切换；成本低、推理速度快 |
| **包管理** | pip + setuptools | pip 23+, setuptools 68+ | Python 生态标准工具；支持 `pyproject.toml` 现代化配置；便于发布到 PyPI |
| **异步支持** | asyncio | Python 3.10 内置 | 标准库，无需额外依赖；支持异步 LLM 调用和并发检索 |
| **配置管理** | python-dotenv + Pydantic Settings | 最新稳定版 | dotenv 加载环境变量；Pydantic 提供类型安全的配置校验 |
| **日志** | logging（标准库） | Python 3.10 内置 | 标准库，无需额外依赖；结构化日志输出 |
| **测试框架** | pytest + pytest-asyncio | pytest 8.x | Python 生态事实标准；支持异步测试 |

### 2.3 关键设计决策

#### 决策 1：为什么选择 ChromaDB 而非 FAISS 或 Milvus Lite

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **ChromaDB** | 零配置嵌入式模式；Python 原生 API；内置元数据过滤；自动持久化；活跃社区 | 大规模场景（千万级向量）性能不如 Milvus | **选中**：对记忆场景（万-十万级向量）完全够用，零运维成本 |
| FAISS | 极致性能；Facebook 维护 | 无内置持久化，需自行管理序列化；无元数据过滤；C++ 底层，调试困难 | 不选：运维复杂度高，功能不完整 |
| Milvus Lite | 完整 Milvus 功能；高性能 | 仍为 Beta 阶段；安装复杂（需要特定系统库）；资源占用高 | 不选：过于重量级，作为库的依赖太重 |

#### 决策 2：为什么选择本地嵌入模型而非在线 API

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **本地 BGE 模型** | 完全离线；零延迟（无网络往返）；零 API 成本；数据隐私安全 | 首次加载需下载模型（约 100MB）；向量质量可能略低于商业 API | **选中**：符合项目"完全离线"的约束条件；BGE 在中文语义检索评测中表现优异 |
| OpenAI Embeddings | 向量质量极高；无需本地 GPU/CPU 推理 | 需要网络；按量计费；数据经过第三方服务器 | 不选：用户明确要求离线方案 |

#### 决策 3：记忆模块作为独立 Python 库而非微服务

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **Python 库/SDK** | 零网络开销（进程内调用）；部署简单（pip install）；与 Agent 代码天然整合 | 不支持跨语言调用；多进程共享需额外设计 | **选中**：目标用户是 Python Agent 开发者，库形式最便捷 |
| 微服务 | 跨语言支持；独立扩缩容 | 增加网络延迟和故障点；部署运维复杂；对简单场景过度设计 | 不选：场景不需要独立服务 |

#### 决策 4：ChromDB 客户端模式选择

使用 ChromaDB 的 **PersistentClient（嵌入式模式）**，而非 HTTP Client 模式。理由：
- 无需启动独立 ChromaDB 服务进程，`pip install chromadb` 后即可使用
- 数据自动持久化到本地磁盘，库初始化时自动加载
- 对单机单进程场景（Python 库）完全足够

---

## 3. 核心流程

### 3.1 主流程：记忆写入（remember）

```
调用方                   MemoryManager              对应Memory模块           Embedder/LLM/Store
  │                          │                          │                        │
  │  remember(content,       │                          │                        │
  │    memory_type,          │                          │                        │
  │    metadata)             │                          │                        │
  │─────────────────────────►│                          │                        │
  │                          │                          │                        │
  │                          │  根据 memory_type 路由    │                        │
  │                          │─────┬────────────────────│                        │
  │                          │     │                    │                        │
  │                          │     │  memory_type=      │                        │
  │                          │     │  "working"         │                        │
  │                          │     │───────────────────►│                        │
  │                          │     │                    │  直接存入 dict + TTL   │
  │                          │     │                    │───────────────────────►│
  │                          │     │                    │◄───────────────────────│
  │                          │     │                    │                        │
  │                          │     │  memory_type=      │                        │
  │                          │     │  "episodic"        │                        │
  │                          │     │───────────────────►│                        │
  │                          │     │                    │  调用 embedder         │
  │                          │     │                    │───────────────────────►│
  │                          │     │                    │◄─────── 向量 ──────────│
  │                          │     │                    │  存入 ChromaDB         │
  │                          │     │                    │  (collection:          │
  │                          │     │                    │   episodic_memory)     │
  │                          │     │                    │───────────────────────►│
  │                          │     │                    │                        │
  │                          │     │  memory_type=      │                        │
  │                          │     │  "semantic"        │                        │
  │                          │     │───────────────────►│                        │
  │                          │     │                    │  调用 embedder         │
  │                          │     │                    │───────────────────────►│
  │                          │     │                    │◄─────── 向量 ──────────│
  │                          │     │                    │  LLM提取实体+关系       │
  │                          │     │                    │───────────────────────►│
  │                          │     │                    │  存入 ChromaDB         │
  │                          │     │                    │  (collection:          │
  │                          │     │                    │   semantic_memory)     │
  │                          │     │                    │───────────────────────►│
  │                          │     │                    │                        │
  │                          │◄────┴────────────────────│                        │
  │◄─────── MemoryItem ─────│                          │                        │
  │                          │                          │                        │
```

### 3.2 主流程：记忆检索（recall）

```
调用方                   MemoryManager              对应Memory模块           Embedder/Store
  │                          │                          │                        │
  │  recall(query,           │                          │                        │
  │    memory_type,          │                          │                        │
  │    top_k, filters)       │                          │                        │
  │─────────────────────────►│                          │                        │
  │                          │                          │                        │
  │                          │  memory_type="working"   │                        │
  │                          │─────────────────────────►│                        │
  │                          │                          │  关键词匹配 + TTL检查   │
  │                          │◄─────── List[Memory] ────│                        │
  │                          │                          │                        │
  │                          │  memory_type="episodic"/ │                        │
  │                          │  "semantic" / None(全部) │                        │
  │                          │─────────────────────────►│                        │
  │                          │                          │  调用 embedder 嵌入query│
  │                          │                          │───────────────────────►│
  │                          │                          │◄─────── query向量 ─────│
  │                          │                          │  ChromaDB相似度检索     │
  │                          │                          │  + 元数据过滤           │
  │                          │                          │───────────────────────►│
  │                          │                          │◄── 相似度排序结果 ──────│
  │                          │◄─────── List[Memory] ────│                        │
  │                          │                          │                        │
  │                          │  去重 + 按相似度排序      │                        │
  │                          │  截取 top_k 结果         │                        │
  │                          │                          │                        │
  │◄─── List[MemoryItem] ───│                          │                        │
```

### 3.3 主流程：记忆整合（consolidate）

```
调用方                   MemoryManager              EpisodicMemory        SemanticMemory     DeepSeekClient
  │                          │                          │                      │                   │
  │  consolidate()           │                          │                      │                   │
  │─────────────────────────►│                          │                      │                   │
  │                          │  获取近期情景记忆         │                      │                   │
  │                          │  (过去24小时内新增)       │                      │                   │
  │                          │─────────────────────────►│                      │                   │
  │                          │◄─── List[Episode] ───────│                      │                   │
  │                          │                          │                      │                   │
  │                          │  构建提取提示词           │                      │                   │
  │                          │  （包含所有近期情景摘要）  │                      │                   │
  │                          │                          │                      │                   │
  │                          │ 调用大模型提取实体+偏好+关系                     │                   │
  │                          │─────────────────────────────────────────────────────────────────►│
  │                          │◄─────── {entities, preferences, relations} ──────────────────────│
  │                          │                          │                      │                   │
  │                          │  逐条写入语义记忆                               │                   │
  │                          │────────────────────────────────────────────────►│                   │
  │                          │                          │                      │                   │
  │◄─── ConsolidateResult ──│                          │                      │                   │
  │   (新增/更新实体数)      │                          │                      │                   │
```

### 3.4 异常流程处理

| 场景 | 处理策略 |
|------|---------|
| **嵌入模型未加载** | 首次调用 `embed()` 时自动加载模型，若加载失败抛出 `ModelLoadError`（继承自 `MemoryAgentError`），附带原始异常信息 |
| **ChromaDB 持久化文件损坏** | 启动时检测，若损坏则自动重置并记录 Warning 日志；用户可通过 `reset=True` 参数手动重置 |
| **LLM API 调用超时** | 默认超时 30 秒，重试 3 次（指数退避：1s, 2s, 4s），全部失败抛出 `LLMServiceError`，原始输入不丢失 |
| **LLM API 返回格式异常** | 使用 Pydantic 校验解析结果，解析失败抛出 `LLMResponseParseError`，附带原始响应文本用于调试 |
| **向量维度不匹配** | 在初始化时校验嵌入模型输出维度与 ChromaDB Collection 预期维度，不匹配时自动重建 Collection 并记录 Warning |
| **并发写入 ChromaDB** | ChromaDB 嵌入式模式不支持多进程并发写入；库内部使用 `asyncio.Lock` 保证单进程内协程安全；在文档中明确说明 |

---

## 4. 模块设计

### 4.1 模块划分

```
memory_agent/
├── __init__.py              # 公开 API 导出
├── core/
│   ├── __init__.py
│   ├── manager.py           # MemoryManager —— 总调度器
│   ├── working_memory.py    # WorkingMemory —— 工作记忆实现
│   ├── episodic_memory.py   # EpisodicMemory —— 情景记忆实现
│   └── semantic_memory.py   # SemanticMemory —— 语义记忆实现
├── storage/
│   ├── __init__.py
│   ├── in_memory_store.py   # InMemoryStore —— 内存级键值存储（供工作记忆使用）
│   └── chroma_store.py      # ChromaStore —— ChromaDB 封装层
├── embedding/
│   ├── __init__.py
│   └── local_embedder.py    # LocalEmbedder —— 本地嵌入模型封装
├── llm/
│   ├── __init__.py
│   └── deepseek_client.py   # DeepSeekClient —— DeepSeek API 客户端
├── models/
│   ├── __init__.py
│   ├── memory_item.py       # MemoryItem —— 记忆项基类
│   ├── episode.py           # Episode —— 情景记忆条目
│   ├── entity.py            # Entity —— 语义记忆实体
│   └── consolidate_result.py # ConsolidateResult —— 整合结果
├── utils/
│   ├── __init__.py
│   ├── config.py            # MemoryConfig —— 配置管理（Pydantic Settings）
│   ├── logger.py            # 日志配置
│   └── errors.py            # 自定义异常体系
└── py.typed                 # PEP 561 类型标记文件
```

### 4.2 核心模块职责

| 模块 | 所属层 | 职责 |
|------|-------|------|
| **MemoryManager** | 核心调度层 | 对外统一接口；路由记忆读写请求；聚合多类型检索结果；触发记忆整合 |
| **WorkingMemory** | 核心记忆层 | 管理当前会话上下文；支持增删查；TTL 自动过期；会话级隔离（通过 session_id） |
| **EpisodicMemory** | 核心记忆层 | 管理历史事件/对话摘要；语义相似度检索；时间范围过滤；写入前自动生成摘要 |
| **SemanticMemory** | 核心记忆层 | 管理用户知识图谱；实体 CRUD；关系查询；偏好管理 |
| **ChromaStore** | 存储层 | 封装 ChromaDB PersistentClient；管理 Collection 生命周期；提供统一的向量增删查接口 |
| **InMemoryStore** | 存储层 | 进程内 dict 存储；支持 TTL；线程安全 |
| **LocalEmbedder** | 基础设施层 | 加载本地 BGE 模型；提供文本到向量的嵌入接口；支持批量嵌入 |
| **DeepSeekClient** | 基础设施层 | DeepSeek Chat API 封装；支持同步/异步调用；自动重试；响应格式校验 |

### 4.3 接口定义

#### 4.3.1 MemoryManager（总调度器）

```python
class MemoryManager:
    """记忆管理总调度器"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化 MemoryManager。

        Args:
            config: 配置对象，若为 None 则从环境变量加载默认配置。

        Raises:
            ConfigError: 配置校验失败时抛出。
            ModelLoadError: 嵌入模型加载失败时抛出。
        """

    async def remember(
        self,
        content: str,
        memory_type: Literal["working", "episodic", "semantic"] = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> MemoryItem:
        """
        写入一条记忆。

        Args:
            content: 记忆内容文本。
            memory_type: 记忆类型，默认为情景记忆。
            metadata: 附加元数据（如时间戳、来源、标签等）。
            session_id: 会话标识（工作记忆必需，其他类型可选）。

        Returns:
            创建的记忆项对象，包含唯一 ID 和时间戳。

        Raises:
            ValueError: memory_type 为 "working" 但 session_id 为空时抛出。
            StorageError: 底层存储写入失败时抛出。
        """

    async def recall(
        self,
        query: str,
        memory_type: Optional[Literal["working", "episodic", "semantic"]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[MemoryItem]:
        """
        检索记忆。

        Args:
            query: 查询文本。
            memory_type: 限定检索的记忆类型，None 表示检索所有类型。
            top_k: 最大返回条数。
            filters: 元数据过滤条件（如 {"start_time": "2026-01-01", "end_time": "2026-04-30"}）。
            session_id: 若指定，检索工作记忆时限定会话。
            min_similarity: 最低相似度阈值（0.0 ~ 1.0），低于此值的结果被过滤。

        Returns:
            按相似度降序排列的记忆项列表。

        Raises:
            StorageError: 底层存储查询失败时抛出。
        """

    async def consolidate(
        self,
        session_id: Optional[str] = None,
        time_window_hours: int = 24,
        dry_run: bool = False,
    ) -> ConsolidateResult:
        """
        触发记忆整合：从近期情景记忆中提取结构化知识，更新语义记忆。

        Args:
            session_id: 限定整合的会话，None 表示所有会话。
            time_window_hours: 整合的时间窗口（小时）。
            dry_run: 若为 True，只返回即将执行的操作预览，不实际写入。

        Returns:
            整合结果，包含新增/更新的实体数、偏好数、关系数。

        Raises:
            LLMServiceError: 大模型调用失败时抛出。
            LLMResponseParseError: 大模型返回格式异常时抛出。
        """

    async def forget(
        self,
        memory_id: str,
        memory_type: Literal["working", "episodic", "semantic"],
    ) -> bool:
        """
        删除指定记忆。

        Args:
            memory_id: 记忆唯一标识。
            memory_type: 记忆类型。

        Returns:
            True 表示删除成功，False 表示记忆不存在。

        Raises:
            StorageError: 底层存储操作失败时抛出。
        """

    async def clear_session(self, session_id: str) -> int:
        """
        清除指定会话的所有工作记忆。

        Args:
            session_id: 会话标识。

        Returns:
            清除的记忆条数。
        """
```

#### 4.3.2 WorkingMemory

```python
class WorkingMemory:
    """工作记忆 —— 当前会话上下文管理"""

    def __init__(self, store: InMemoryStore, default_ttl_seconds: int = 3600):
        """
        Args:
            store: 内存级存储后端。
            default_ttl_seconds: 默认过期时间（秒），默认 1 小时。
        """

    async def add(
        self,
        content: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> MemoryItem:
        """添加一条工作记忆。若未指定 ttl_seconds，使用默认值。"""

    async def get_by_session(
        self, session_id: str, include_expired: bool = False
    ) -> List[MemoryItem]:
        """获取指定会话的所有工作记忆。"""

    async def search(
        self, query: str, session_id: Optional[str] = None, top_k: int = 5
    ) -> List[MemoryItem]:
        """使用关键词匹配搜索工作记忆（不依赖向量）。"""

    async def remove(self, memory_id: str) -> bool:
        """删除指定记忆。"""

    async def expire_session(self, session_id: str) -> int:
        """手动过期指定会话的所有记忆。"""
```

#### 4.3.3 EpisodicMemory

```python
class EpisodicMemory:
    """情景记忆 —— 历史事件和对话摘要的存储与检索"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        embedder: LocalEmbedder,
        llm_client: DeepSeekClient,
    ):
        """
        Args:
            chroma_store: ChromaDB 存储后端。
            embedder: 嵌入模型。
            llm_client: 大模型客户端，用于生成情景记忆摘要。
        """

    async def add_episode(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Episode:
        """
        添加一条情景记忆。
        内部流程：1) 若内容过长（> 2000 字符），调用 LLM 生成摘要；
                  2) 使用嵌入模型生成向量；
                  3) 存入 ChromaDB episodic_memory collection。
        """

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5,
    ) -> List[Episode]:
        """
        语义检索情景记忆。
        filters 支持：
          - start_time / end_time: isoformat 时间范围过滤
          - session_id: 按会话过滤
          - tags: 按标签过滤（List[str]）
        """

    async def get_recent(
        self, hours: int = 24, session_id: Optional[str] = None
    ) -> List[Episode]:
        """获取近期情景记忆（按时间戳过滤）。"""

    async def get_by_id(self, episode_id: str) -> Optional[Episode]:
        """按 ID 获取单条情景记忆。"""

    async def remove(self, episode_id: str) -> bool:
        """删除指定情景记忆。"""
```

#### 4.3.4 SemanticMemory

```python
class SemanticMemory:
    """语义记忆 —— 用户知识图谱管理"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        embedder: LocalEmbedder,
        llm_client: DeepSeekClient,
    ):
        """
        Args:
            chroma_store: ChromaDB 存储后端。
            embedder: 嵌入模型。
            llm_client: 大模型客户端，用于实体提取和关系推理。
        """

    async def add_entity(
        self,
        name: str,
        entity_type: str,        # "person", "organization", "topic", "preference", "fact"
        description: str,
        attributes: Optional[Dict[str, Any]] = None,
        related_entities: Optional[List[str]] = None,  # 关联实体 ID 列表
    ) -> Entity:
        """添加或更新一个知识实体（同名实体将合并更新）。"""

    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_type: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Entity]:
        """语义检索知识实体。"""

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """按 ID 获取实体。"""

    async def get_preferences(self, user_id: Optional[str] = None) -> List[Entity]:
        """获取用户偏好列表（entity_type = "preference"）。"""

    async def update_entity(
        self, entity_id: str, updates: Dict[str, Any]
    ) -> Entity:
        """更新实体属性。"""

    async def add_relation(
        self, source_id: str, target_id: str, relation_type: str
    ) -> None:
        """添加实体间关系。"""

    async def get_related_entities(
        self, entity_id: str, relation_type: Optional[str] = None
    ) -> List[Entity]:
        """获取与指定实体关联的其他实体。"""

    async def remove_entity(self, entity_id: str) -> bool:
        """删除实体及其所有关系。"""
```

#### 4.3.5 ChromaStore

```python
class ChromaStore:
    """ChromaDB 封装层 —— 统一的向量存储接口"""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "default",
        embedding_dimension: int = 512,
    ):
        """
        初始化 ChromaStore，连接或创建 ChromaDB Collection。

        若 Collection 已存在但向量维度与 embedding_dimension 参数不匹配，
        将自动删除旧 Collection 并创建新 Collection，同时通过 logging 输出
        Warning 日志。

        Args:
            persist_directory: 持久化数据目录。
            collection_name: Collection 名称。
            embedding_dimension: 预期的向量维度。
        """

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """批量添加向量记录。"""

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        向量相似度检索。

        Returns:
            {"ids": List[str], "documents": List[str],
             "metadatas": List[Dict], "distances": List[float]}
        """

    def get(self, ids: List[str]) -> Dict[str, Any]:
        """按 ID 批量获取记录。"""

    def delete(self, ids: List[str]) -> None:
        """按 ID 批量删除记录。"""

    def count(self) -> int:
        """返回 Collection 中的记录总数。"""

    def reset(self) -> None:
        """删除并重建 Collection（危险操作，用于修复损坏数据）。"""
```

#### 4.3.6 LocalEmbedder

```python
class LocalEmbedder:
    """本地嵌入模型封装"""

    def __init__(
        self,
        model_name: str = "models/bge-small-zh-v1.5",
        device: str = "cpu",          # "cpu" | "cuda" | "cuda:0"
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径。
            device: 推理设备。
            normalize: 是否归一化向量（推荐开启，用于余弦相似度）。
            cache_dir: 模型缓存目录，默认 ~/.cache/huggingface。

        Raises:
            ModelLoadError: 模型加载失败时抛出。
        """

    @property
    def dimension(self) -> int:
        """返回嵌入向量维度。"""

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        将文本转换为向量。

        Args:
            texts: 单个文本或文本列表。

        Returns:
            向量列表，每个向量为 List[float]。
            输入 "hello" -> [[0.1, 0.2, ...]]
            输入 ["hello", "world"] -> [[0.1, ...], [0.3, ...]]
        """

    def embed_query(self, query: str) -> List[float]:
        """
        将查询文本转换为向量（单条，用于检索）。

        Args:
            query: 查询文本。

        Returns:
            向量 List[float]。
        """
```

#### 4.3.7 DeepSeekClient

```python
class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Args:
            api_key: API 密钥，默认从环境变量 DEEPSEEK_API_KEY 读取。
            model: 模型名称。
            base_url: API 基础 URL。
            timeout: 请求超时（秒）。
            max_retries: 最大重试次数。

        Raises:
            ConfigError: api_key 未配置时抛出。
        """

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, Any]] = None,  # JSON mode
    ) -> str:
        """
        发送对话请求。

        Args:
            messages: 消息列表，格式 [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            response_format: 指定输出格式，如 {"type": "json_object"}。

        Returns:
            模型响应文本。

        Raises:
            LLMServiceError: API 调用失败（含重试耗尽）时抛出。
            LLMResponseParseError: response_format 指定为 JSON 但解析失败时抛出。
        """

    async def extract_entities(
        self, text: str
    ) -> Dict[str, Any]:
        """
        从文本中提取结构化实体。

        Returns:
            {
                "entities": [
                    {"name": "...", "type": "...", "description": "...", "attributes": {...}},
                    ...
                ],
                "preferences": [
                    {"subject": "...", "value": "...", "confidence": 0.9},
                    ...
                ],
                "relations": [
                    {"source": "...", "target": "...", "type": "..."},
                    ...
                ]
            }
        """
```

### 4.4 数据模型

```python
# models/memory_item.py

class MemoryItem(BaseModel):
    """记忆项基类"""
    id: str                          # UUID，唯一标识
    content: str                     # 记忆内容
    memory_type: MemoryType          # 枚举：working / episodic / semantic
    created_at: datetime             # 创建时间（UTC）
    updated_at: datetime             # 最后更新时间（UTC）
    metadata: Dict[str, Any] = {}    # 附加元数据
    session_id: Optional[str] = None # 关联的会话 ID

# models/episode.py

class Episode(MemoryItem):
    """情景记忆条目"""
    summary: Optional[str] = None    # LLM 生成的摘要（原始内容过长时）
    embedding: Optional[List[float]] = None  # 向量（仅内部使用，不对外暴露）
    importance: float = 1.0          # 重要性评分（0.0 ~ 1.0），用于优先级排序
    tags: List[str] = []             # 标签

# models/entity.py

class Entity(BaseModel):
    """语义记忆实体"""
    id: str                          # UUID
    name: str                        # 实体名称
    entity_type: str                 # 实体类型：person / organization / topic / preference / fact
    description: str                 # 实体描述
    attributes: Dict[str, Any] = {}  # 属性键值对
    embedding: Optional[List[float]] = None  # 向量（仅内部使用）
    related_entities: List[str] = [] # 关联实体 ID 列表
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0          # 置信度（0.0 ~ 1.0）

# models/consolidate_result.py

class ConsolidateResult(BaseModel):
    """记忆整合结果"""
    new_entities: int = 0
    updated_entities: int = 0
    new_preferences: int = 0
    updated_preferences: int = 0
    new_relations: int = 0
    episodes_processed: int = 0
    errors: List[str] = []
    dry_run: bool = False
```

### 4.5 依赖关系

```
MemoryManager
  ├── WorkingMemory ─────── InMemoryStore
  ├── EpisodicMemory ────── ChromaStore ───── ChromaDB PersistentClient
  │   ├──────────────────── LocalEmbedder ─── sentence-transformers
  │   └──────────────────── DeepSeekClient ── httpx (HTTP 客户端)
  └── SemanticMemory ────── ChromaStore
      ├──────────────────── LocalEmbedder
      └──────────────────── DeepSeekClient ── httpx (HTTP 客户端)
```

各模块之间通过构造函数注入依赖（依赖注入模式），便于单元测试时 mock 替换。

---

## 5. 非功能性需求

### 5.1 性能指标

| 指标 | 目标值 | 说明 |
|------|-------|------|
| 单条记忆写入延迟（episodic/semantic） | < 500ms | 含嵌入计算 + ChromaDB 写入 |
| 单条记忆写入延迟（working） | < 5ms | 纯内存操作 |
| Top-10 检索延迟 | < 300ms | 含查询嵌入 + ChromaDB 检索 |
| 嵌入模型首次加载时间 | < 10s | 冷启动，含模型下载（取决于网络） |
| 嵌入模型推理吞吐 | > 50 条/秒（CPU） | BGE-small 在普通 CPU 上的预期性能 |
| 支持记忆总量 | 10 万条 | ChromaDB 单 Collection 在此量级下性能稳定 |

### 5.2 安全

| 关注点 | 策略 |
|--------|------|
| **API 密钥管理** | 仅从环境变量 `DEEPSEEK_API_KEY` 读取，绝不硬编码；配置对象 `.model_dump()` 时自动脱敏 |
| **数据隔离** | 通过 `session_id` 和元数据过滤实现多用户/多会话数据隔离；调用方负责传入正确的 `session_id` |
| **输入校验** | 所有公开 API 使用 Pydantic 校验输入参数；`content` 字段限制最大长度 50000 字符 |
| **持久化文件权限** | ChromaDB 数据目录权限为 `0o700`（仅所有者可读写） |
| **依赖安全** | 使用 `pip-audit` 定期扫描依赖漏洞；`requirements.txt` 锁定版本 |

### 5.3 可观测性

| 维度 | 方案 |
|------|------|
| **日志** | 使用 Python `logging` 标准库；日志级别通过 `LOG_LEVEL` 环境变量控制（默认 INFO）；关键路径记录 DEBUG 日志（写入、检索、LLM 调用） |
| **结构化日志** | 使用 `logging.Formatter` 输出 JSON 格式（开发环境可切换为可读格式），每条日志包含 `timestamp`, `level`, `module`, `message`, `extra` |
| **性能埋点** | 在关键路径记录耗时（使用 `time.perf_counter()`）：嵌入计算、ChromaDB 写入/查询、LLM API 调用 |
| **错误追踪** | 所有自定义异常包含 `error_code` 和 `details` 字段，便于调用方捕获和处理 |

### 5.4 错误处理体系

```python
# utils/errors.py

class MemoryAgentError(Exception):
    """所有自定义异常的基类"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ConfigError(MemoryAgentError):
    """配置错误：缺少必需配置项、配置值非法"""
    error_code = "CONFIG_ERROR"

class StorageError(MemoryAgentError):
    """存储层错误：ChromaDB 读写失败、InMemoryStore 异常"""
    error_code = "STORAGE_ERROR"

class ModelLoadError(MemoryAgentError):
    """模型加载错误：嵌入模型文件不存在、OOM、网络下载失败"""
    error_code = "MODEL_LOAD_ERROR"

class EmbeddingError(MemoryAgentError):
    """嵌入计算错误：输入格式异常、推理失败"""
    error_code = "EMBEDDING_ERROR"

class LLMServiceError(MemoryAgentError):
    """LLM 服务错误：API 调用超时、认证失败、服务端错误"""
    error_code = "LLM_SERVICE_ERROR"

class LLMResponseParseError(MemoryAgentError):
    """LLM 响应解析错误：JSON 格式异常、字段缺失"""
    error_code = "LLM_RESPONSE_PARSE_ERROR"

class DimensionMismatchError(MemoryAgentError):
    """向量维度不匹配"""
    error_code = "DIMENSION_MISMATCH_ERROR"
```

---

## 6. 关键数据流

### 6.1 场景一：Agent 与用户对话，记住用户偏好

```
1. 用户说："我叫张三，喜欢喝咖啡，尤其喜欢蓝山咖啡"
2. Agent 调用：manager.remember("用户叫张三，喜欢喝咖啡，尤其喜欢蓝山咖啡", memory_type="episodic")
3. EpisodicMemory 收到内容 -> LocalEmbedder.embed() 生成向量 -> ChromaStore.add() 存入 episodic_memory collection
4. Agent 稍后调用：manager.consolidate()
5. MemoryManager 获取近24小时情景记忆 -> 构建提示词 -> DeepSeekClient.extract_entities() ->
   返回实体列表：
     - Entity(name="张三", type="person", attributes={"称呼": "张三"})
     - Entity(name="咖啡", type="preference", description="喜欢喝咖啡", attributes={"具体偏好": "蓝山咖啡"})
6. SemanticMemory 逐条 upsert 实体 -> 存入 semantic_memory collection
```

### 6.2 场景二：Agent 检索用户偏好，个性化回复

```
1. 用户问："帮我推荐一款饮品"
2. Agent 调用：prefs = manager.recall("用户饮品偏好", memory_type="semantic", top_k=3)
3. SemanticMemory 调用 LocalEmbedder.embed_query("用户饮品偏好") -> ChromaStore.query() ->
   返回 top-3 匹配实体：
     - Entity(name="咖啡", description="喜欢喝咖啡", attributes={"具体偏好": "蓝山咖啡"}, similarity=0.92)
     - Entity(name="茶", description="偶尔喝茶", similarity=0.78)
     - ...
4. Agent 使用返回的偏好信息，生成个性化回复："根据您的喜好，我推荐您试试这款新的蓝山咖啡豆..."
```

### 6.3 场景三：多轮对话中的上下文保持

```
1. 用户开启新会话，session_id = "sess-001"
2. 第一轮：用户问"Python 中如何读取 CSV 文件？"
   Agent 调用：manager.remember("用户在学习 Python 数据处理", memory_type="working", session_id="sess-001")
3. 第二轮：用户问"那 JSON 呢？"
   Agent 调用：ctx = manager.recall("Python 数据处理", memory_type="working", session_id="sess-001", top_k=5)
   返回工作记忆中的上下文 -> Agent 理解"那"指的是"如何读取文件"，从而正确回答 JSON 读取方法。
4. 会话结束：manager.clear_session("sess-001") -> 清理所有工作记忆。
```

---

## 7. 嵌入模型本地部署方案详细说明

### 7.1 默认模型：BAAI/bge-small-zh-v1.5（已预置在 models/ 目录）

| 属性 | 值 |
|------|-----|
| 模型名称 | BAAI/bge-small-zh-v1.5 |
| 向量维度 | 512 |
| 最大输入长度 | 512 tokens |
| 模型大小 | ~100MB |
| 推理速度（CPU） | ~100 条/秒（Intel i5 第10代+） |
| 内存占用 | ~500MB（加载后） |
| 支持语言 | 中文为主，也支持英文 |
| 许可证 | MIT |

### 7.2 备选模型

若默认模型不满足需求（如需要更高精度或更长文本支持），可通过配置切换：

| 模型 | 维度 | 大小 | 特点 |
|------|------|------|------|
| `BAAI/bge-base-zh-v1.5` | 768 | ~400MB | 更高精度，适合精细语义区分 |
| `BAAI/bge-large-zh-v1.5` | 1024 | ~1.3GB | 最高精度，适合生产环境（需 GPU） |
| `BAAI/bge-small-en-v1.5` | 384 | ~130MB | 纯英文场景，更小更快 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90MB | 通用英文模型，速度快 |
| `intfloat/multilingual-e5-small` | 384 | ~470MB | 多语言支持（100+ 语言） |

### 7.3 首次加载流程

```
1. 用户首次实例化 LocalEmbedder()
2. 调用 sentence_transformers.SentenceTransformer("models/bge-small-zh-v1.5")
3. 模型已预置在项目 models/ 目录下，直接从本地加载（无需联网）
4. 加载模型到内存（CPU 或 CUDA）
5. 预热：执行一次空推理，触发 JIT 编译优化
6. 就绪，后续调用 embed() 无需重新加载
```

### 7.4 离线环境部署

模型已预置在 `models/bge-small-zh-v1.5/` 目录，无需额外下载。若需切换其他模型，可下载后放入 models/ 目录并更新 `EMBEDDING_MODEL_NAME` 配置：

```bash
# 示例：下载其他模型到 models/ 目录
python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5'); \
    model.save('models/bge-base-zh-v1.5')"

# 然后修改 .env：
# EMBEDDING_MODEL_NAME=models/bge-base-zh-v1.5
```

### 7.5 GPU 加速（可选）

若部署环境有 NVIDIA GPU 且已安装 CUDA，可启用 GPU 推理：

```python
embedder = LocalEmbedder(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cuda"  # 自动使用第一个可用 GPU
)
```

需要额外的 pip 依赖：`torch` with CUDA support、`sentence-transformers`。

---

## 8. 项目目录结构（完整）

```
memory-agent-2/
├── docs/
│   ├── tech-doc.md              # 本技术文档
│   └── env-checklist.md         # 环境准备清单
├── memory_agent/                # Python 包根目录
│   ├── __init__.py
│   ├── py.typed
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── working_memory.py
│   │   ├── episodic_memory.py
│   │   └── semantic_memory.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── in_memory_store.py
│   │   └── chroma_store.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── local_embedder.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── deepseek_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── memory_item.py
│   │   ├── episode.py
│   │   ├── entity.py
│   │   └── consolidate_result.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── errors.py
├── tests/                       # 单元测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_manager.py
│   ├── test_working_memory.py
│   ├── test_episodic_memory.py
│   ├── test_semantic_memory.py
│   ├── test_chroma_store.py
│   ├── test_embedder.py
│   └── test_deepseek_client.py
├── examples/                    # 使用示例
│   └── basic_usage.py
├── pyproject.toml               # 项目元数据和构建配置
├── requirements.txt             # 生产依赖
├── requirements-dev.txt         # 开发依赖
├── .env.example                 # 环境变量模板
├── .gitignore
└── README.md
```
