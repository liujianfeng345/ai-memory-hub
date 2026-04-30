"""
整合结果模型。

ConsolidateResult 记录一次情节记忆整合操作的结果摘要，
包括新增/更新的实体数、偏好数、关系数，以及错误信息等。
"""

from pydantic import BaseModel, Field


class ConsolidateResult(BaseModel):
    """情节记忆整合结果。

    记录一次 consolidate 操作的处理结果，供上层调用者了解整合进度和结果。

    Attributes:
        new_entities: 新创建的实体数量。
        updated_entities: 更新的实体数量。
        new_preferences: 新发现的偏好数量。
        updated_preferences: 更新的偏好数量。
        new_relations: 新发现的关系数量。
        episodes_processed: 已处理的情节记忆数量。
        errors: 处理过程中遇到的错误信息列表。
        dry_run: 是否为干运行模式（仅预览，不实际写入）。
    """

    new_entities: int = Field(default=0, ge=0)
    updated_entities: int = Field(default=0, ge=0)
    new_preferences: int = Field(default=0, ge=0)
    updated_preferences: int = Field(default=0, ge=0)
    new_relations: int = Field(default=0, ge=0)
    episodes_processed: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False
