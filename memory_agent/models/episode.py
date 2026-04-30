"""
情节记忆模型。

Episode 继承自 MemoryItem，代表已完结的对话片段或事件，
包含摘要、嵌入向量、重要度评分和标签等额外字段。
"""

from typing import Any

from pydantic import Field

from memory_agent.models.memory_item import MemoryItem, MemoryType


class Episode(MemoryItem):
    """情节记忆条目。

    继承自 MemoryItem，增加了摘要、嵌入向量、重要度评分和标签字段，
    用于长期记忆管理和检索。

    Attributes:
        summary: 情节的自动生成摘要，可能为 None。
        embedding: 情节内容的向量嵌入，用于语义检索。
        importance: 重要度评分（1.0 为最低，数值越高越重要）。
        tags: 情节标签列表，用于分类检索。
    """

    summary: str | None = None
    embedding: list[float] | None = None
    importance: float = Field(default=1.0, ge=0.0)
    tags: list[str] = Field(default_factory=list)

    def __init__(self, **data: Any) -> None:
        # 确保情节记忆的 memory_type 为 episodic
        data.setdefault("memory_type", MemoryType.episodic)
        super().__init__(**data)
