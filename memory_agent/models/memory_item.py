"""
记忆条目基础模型。

定义 MemoryType 枚举和 MemoryItem 基类，所有具体记忆类型均继承自 MemoryItem。
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型枚举。

    Attributes:
        working: 工作记忆 —— 当前会话中的临时信息。
        episodic: 情节记忆 —— 已完结的对话片段或事件。
        semantic: 语义记忆 —— 从情节记忆中提取的持久化知识。
    """

    working = "working"
    episodic = "episodic"
    semantic = "semantic"


def _utc_now() -> datetime:
    """返回当前 UTC 时间，用作字段默认工厂函数。"""
    return datetime.now(tz=timezone.utc)


class MemoryItem(BaseModel):
    """记忆条目基类。

    所有记忆类型（工作记忆、情节记忆、语义记忆）均继承此类。

    Attributes:
        id: 唯一标识符，默认为 UUID4。
        content: 记忆内容文本。
        memory_type: 记忆类型。
        created_at: 创建时间（UTC）。
        updated_at: 最后更新时间（UTC）。
        metadata: 附加元数据字典。
        session_id: 可选，关联的会话标识符。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
