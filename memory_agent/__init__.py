"""
Memory Agent —— 智能记忆代理。

agent-memory 是一个面向 AI Agent 的可插拔记忆管理系统，提供三种记忆类型：
- 工作记忆（WorkingMemory）：基于关键词匹配的会话内临时记忆。
- 情节记忆（EpisodicMemory）：基于向量语义检索的对话/事件记忆，支持自动摘要。
- 语义记忆（SemanticMemory）：从情节记忆中提取的持久化知识图谱实体。

核心入口为 MemoryManager，用户只需实例化该类即可获得完整的记忆管理能力。

快速开始::

    import asyncio
    from memory_agent import MemoryManager, MemoryConfig

    async def main():
        config = MemoryConfig()
        manager = MemoryManager(config)
        await manager.remember("用户喜欢 Python", memory_type="episodic")
        results = await manager.recall("编程语言")
        print(results)

    asyncio.run(main())


说明：导入此包不会自动加载模型或连接数据库。
仅在实例化 MemoryManager 时才会触发各组件的初始化。
"""

# ---------------------------------------------------------------------------
# 公开 API 导出
# ---------------------------------------------------------------------------

from memory_agent.core.manager import MemoryManager
from memory_agent.models.consolidate_result import ConsolidateResult
from memory_agent.models.entity import Entity
from memory_agent.models.episode import Episode
from memory_agent.models.memory_item import MemoryItem, MemoryType
from memory_agent.utils.config import MemoryConfig
from memory_agent.utils.errors import (
    ConfigError,
    DimensionMismatchError,
    EmbeddingError,
    LLMResponseParseError,
    LLMServiceError,
    MemoryAgentError,
    ModelLoadError,
    StorageError,
)

__all__ = [
    # 核心入口
    "MemoryManager",
    # 配置
    "MemoryConfig",
    # 数据模型
    "MemoryItem",
    "MemoryType",
    "Episode",
    "Entity",
    "ConsolidateResult",
    # 异常类
    "MemoryAgentError",
    "ConfigError",
    "StorageError",
    "ModelLoadError",
    "EmbeddingError",
    "LLMServiceError",
    "LLMResponseParseError",
    "DimensionMismatchError",
]
