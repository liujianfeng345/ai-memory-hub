"""
实体模型。

Entity 代表从情节记忆中提取的持久化知识单元，
包括人物、组织、话题、偏好和事实等类型。
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

# 合法的实体类型集合
VALID_ENTITY_TYPES = {"person", "organization", "topic", "preference", "fact"}


def _utc_now() -> datetime:
    """返回当前 UTC 时间，用作字段默认工厂函数。"""
    return datetime.now(tz=timezone.utc)


class Entity(BaseModel):
    """语义实体。

    从情节记忆中提取的持久化知识单元，可以是人物、组织、话题、偏好或事实。

    Attributes:
        id: 唯一标识符。
        name: 实体名称。
        entity_type: 实体类型（person / organization / topic / preference / fact）。
        description: 实体描述。
        attributes: 附加属性字典。
        embedding: 实体名称+描述的向量嵌入，用于语义检索。
        related_entities: 关联实体 ID 列表。
        created_at: 创建时间（UTC）。
        updated_at: 最后更新时间（UTC）。
        confidence: 置信度，范围 [0.0, 1.0]，1.0 表示完全确信。
    """

    id: str
    name: str
    entity_type: str
    description: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    related_entities: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """校验 entity_type 是否为合法值。

        Args:
            v: 实体类型字符串。

        Returns:
            校验通过的实体类型字符串。

        Raises:
            ValueError: 当 entity_type 不在合法值集合中时抛出。
        """
        if v.lower() not in VALID_ENTITY_TYPES:
            raise ValueError(f"entity_type 必须是 {VALID_ENTITY_TYPES} 之一，当前值: {v}")
        return v.lower()
