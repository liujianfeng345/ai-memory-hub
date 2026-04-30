"""
数据模型单元测试。

验证所有 Pydantic 模型可正常实例化、序列化/反序列化、
字段校验和继承关系。
"""

import json
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from memory_agent.models.consolidate_result import ConsolidateResult
from memory_agent.models.entity import Entity
from memory_agent.models.episode import Episode
from memory_agent.models.memory_item import MemoryItem, MemoryType


class TestMemoryType:
    """MemoryType 枚举测试。"""

    def test_enum_values(self) -> None:
        """测试 MemoryType 包含三个预期值。"""
        assert MemoryType.working == "working"
        assert MemoryType.episodic == "episodic"
        assert MemoryType.semantic == "semantic"

    def test_enum_is_string(self) -> None:
        """测试 MemoryType 是 StrEnum，值与字符串可直接比较。"""
        assert MemoryType.working == "working"
        assert isinstance(MemoryType.working, str)


class TestMemoryItem:
    """MemoryItem 模型测试。"""

    def test_basic_instantiation(self) -> None:
        """测试 MemoryItem 基本实例化。"""
        item = MemoryItem(content="测试内容", memory_type=MemoryType.working)
        assert item.content == "测试内容"
        assert item.memory_type == MemoryType.working

    def test_auto_generated_uuid(self) -> None:
        """测试 MemoryItem 自动生成 UUID 格式的 id。"""
        item = MemoryItem(content="测试", memory_type=MemoryType.episodic)
        # 验证 id 是有效的 UUID 格式字符串
        parsed = uuid.UUID(item.id)
        assert parsed.version == 4  # UUID4

    def test_auto_generated_timestamps(self) -> None:
        """测试 MemoryItem 自动生成时间戳。"""
        before = datetime.now(tz=timezone.utc)
        item = MemoryItem(content="测试", memory_type=MemoryType.working)
        after = datetime.now(tz=timezone.utc)

        assert item.created_at is not None
        assert item.updated_at is not None
        assert item.created_at.tzinfo == timezone.utc
        assert item.updated_at.tzinfo == timezone.utc
        # 时间戳应在创建前后之间
        assert before <= item.created_at <= after

    def test_default_metadata(self) -> None:
        """测试 metadata 默认为空字典。"""
        item = MemoryItem(content="测试", memory_type=MemoryType.working)
        assert item.metadata == {}

    def test_default_session_id(self) -> None:
        """测试 session_id 默认为 None。"""
        item = MemoryItem(content="测试", memory_type=MemoryType.working)
        assert item.session_id is None

    def test_with_session_id(self) -> None:
        """测试可设置 session_id。"""
        item = MemoryItem(content="测试", memory_type=MemoryType.working, session_id="session-123")
        assert item.session_id == "session-123"

    def test_with_custom_metadata(self) -> None:
        """测试可设置自定义 metadata。"""
        meta = {"source": "test", "priority": 5}
        item = MemoryItem(content="测试", memory_type=MemoryType.working, metadata=meta)
        assert item.metadata == meta

    def test_serialization_to_json(self) -> None:
        """测试 model_dump_json 和 model_validate_json 的往返转换。"""
        item = MemoryItem(content="测试序列化", memory_type=MemoryType.episodic)
        json_str = item.model_dump_json()
        restored = MemoryItem.model_validate_json(json_str)

        assert restored.content == "测试序列化"
        assert restored.memory_type == MemoryType.episodic
        assert restored.id == item.id
        assert restored.metadata == item.metadata

    def test_serialization_preserves_uuid(self) -> None:
        """测试序列化后的 id 为有效 UUID 格式。"""
        item = MemoryItem(content="test", memory_type="episodic")
        data = json.loads(item.model_dump_json())
        parsed = uuid.UUID(data["id"])
        assert parsed.version == 4

    def test_custom_id(self) -> None:
        """测试可手动指定 id。"""
        custom_id = str(uuid.uuid4())
        item = MemoryItem(id=custom_id, content="测试", memory_type=MemoryType.working)
        assert item.id == custom_id

    def test_memory_type_from_string(self) -> None:
        """测试 memory_type 可从字符串自动转换。"""
        item = MemoryItem(content="测试", memory_type="working")
        assert item.memory_type == MemoryType.working

    def test_memory_type_literal_in_json(self) -> None:
        """测试 JSON 序列化中 memory_type 为字符串字面量。"""
        item = MemoryItem(content="test", memory_type="episodic")
        data = json.loads(item.model_dump_json())
        assert data["memory_type"] == "episodic"


class TestEpisode:
    """Episode 模型测试。"""

    def test_inherits_from_memory_item(self) -> None:
        """测试 Episode 继承自 MemoryItem。"""
        episode = Episode(content="情节内容")
        assert isinstance(episode, MemoryItem)

    def test_default_memory_type(self) -> None:
        """测试 Episode 默认 memory_type 为 episodic。"""
        episode = Episode(content="情节内容")
        assert episode.memory_type == MemoryType.episodic

    def test_has_episode_specific_fields(self) -> None:
        """测试 Episode 包含子类特有字段。"""
        episode = Episode(content="情节内容")
        assert episode.summary is None
        assert episode.embedding is None
        assert episode.importance == 1.0
        assert episode.tags == []

    def test_with_all_fields(self) -> None:
        """测试 Episode 所有字段可设置。"""
        episode = Episode(
            content="重要对话",
            summary="关于项目进度的讨论",
            embedding=[0.1, 0.2, 0.3],
            importance=8.5,
            tags=["项目", "进度"],
            session_id="sess-001",
        )
        assert episode.summary == "关于项目进度的讨论"
        assert episode.embedding == [0.1, 0.2, 0.3]
        assert episode.importance == 8.5
        assert episode.tags == ["项目", "进度"]
        assert episode.session_id == "sess-001"
        assert episode.memory_type == MemoryType.episodic

    def test_serialization_includes_subclass_fields(self) -> None:
        """测试序列化包含子类字段。"""
        episode = Episode(
            content="测试情节",
            summary="摘要内容",
            embedding=[0.5, 0.6],
            importance=7.0,
            tags=["tag1"],
        )
        json_str = episode.model_dump_json()
        restored = Episode.model_validate_json(json_str)

        assert restored.summary == "摘要内容"
        assert restored.embedding == [0.5, 0.6]
        assert restored.importance == 7.0
        assert restored.tags == ["tag1"]

    def test_tags_default_empty_list(self) -> None:
        """测试 tags 默认为空列表。"""
        episode = Episode(content="情节")
        assert episode.tags == []
        assert isinstance(episode.tags, list)

    def test_overwrite_memory_type(self) -> None:
        """测试可覆盖 Episode 的 memory_type（虽然不推荐）。"""
        episode = Episode(content="测试", memory_type=MemoryType.semantic)
        assert episode.memory_type == MemoryType.semantic


class TestEntity:
    """Entity 模型测试。"""

    def test_basic_instantiation(self) -> None:
        """测试 Entity 基本实例化。"""
        entity = Entity(
            id="ent-001",
            name="张三",
            entity_type="person",
            description="一个软件工程师",
        )
        assert entity.id == "ent-001"
        assert entity.name == "张三"
        assert entity.entity_type == "person"
        assert entity.description == "一个软件工程师"

    def test_default_values(self) -> None:
        """测试 Entity 默认值。"""
        entity = Entity(
            id="ent-002",
            name="Python",
            entity_type="topic",
            description="编程语言",
        )
        assert entity.attributes == {}
        assert entity.embedding is None
        assert entity.related_entities == []
        assert entity.confidence == 1.0

    def test_auto_generated_timestamps(self) -> None:
        """测试 Entity 自动生成 UTC 时间戳。"""
        entity = Entity(
            id="ent-003",
            name="测试实体",
            entity_type="fact",
            description="测试描述",
        )
        assert entity.created_at.tzinfo == timezone.utc
        assert entity.updated_at.tzinfo == timezone.utc

    def test_with_embedding(self) -> None:
        """测试 Entity 可携带嵌入向量。"""
        embedding = [0.1, 0.2, 0.3, 0.4]
        entity = Entity(
            id="ent-004",
            name="向量实体",
            entity_type="topic",
            description="带嵌入的实体",
            embedding=embedding,
        )
        assert entity.embedding == embedding

    def test_with_related_entities(self) -> None:
        """测试 Entity 可关联其他实体。"""
        entity = Entity(
            id="ent-005",
            name="关联实体",
            entity_type="person",
            description="有关联的实体",
            related_entities=["ent-001", "ent-002"],
        )
        assert entity.related_entities == ["ent-001", "ent-002"]

    @pytest.mark.parametrize(
        "valid_type",
        ["person", "organization", "topic", "preference", "fact"],
    )
    def test_valid_entity_types(self, valid_type) -> None:
        """测试所有合法 entity_type 值均通过校验。"""
        entity = Entity(
            id="ent-valid",
            name="测试",
            entity_type=valid_type,
            description="测试实体",
        )
        assert entity.entity_type == valid_type

    @pytest.mark.parametrize(
        "valid_type_mixed_case",
        ["Person", "ORGANIZATION", "Topic", "Preference", "FACT"],
    )
    def test_entity_type_case_insensitive(self, valid_type_mixed_case) -> None:
        """测试 entity_type 大小写不敏感。"""
        entity = Entity(
            id="ent-case",
            name="测试",
            entity_type=valid_type_mixed_case,
            description="测试实体",
        )
        assert entity.entity_type == valid_type_mixed_case.lower()

    def test_invalid_entity_type_raises(self) -> None:
        """测试非法 entity_type 值时校验失败。"""
        with pytest.raises(ValidationError) as exc_info:
            Entity(
                id="ent-invalid",
                name="无效实体",
                entity_type="invalid_type",
                description="测试",
            )
        assert "entity_type" in str(exc_info.value)

    def test_invalid_entity_type_empty_string_raises(self) -> None:
        """测试 entity_type 为空字符串时校验失败。"""
        with pytest.raises(ValidationError):
            Entity(
                id="ent-empty",
                name="空类型实体",
                entity_type="",
                description="测试",
            )

    def test_confidence_range(self) -> None:
        """测试 confidence 必须在 [0.0, 1.0] 范围内。"""
        # 边界值应通过
        Entity(id="e1", name="n", entity_type="fact", description="d", confidence=0.0)
        Entity(id="e2", name="n", entity_type="fact", description="d", confidence=1.0)

        # 超出范围应失败
        with pytest.raises(ValidationError):
            Entity(id="e3", name="n", entity_type="fact", description="d", confidence=1.5)
        with pytest.raises(ValidationError):
            Entity(id="e4", name="n", entity_type="fact", description="d", confidence=-0.1)

    def test_serialization(self) -> None:
        """测试 Entity 序列化往返。"""
        entity = Entity(
            id="ent-serial",
            name="序列化实体",
            entity_type="topic",
            description="测试序列化",
            related_entities=["ent-001"],
            confidence=0.95,
        )
        json_str = entity.model_dump_json()
        restored = Entity.model_validate_json(json_str)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.related_entities == entity.related_entities
        assert restored.confidence == entity.confidence


class TestConsolidateResult:
    """ConsolidateResult 模型测试。"""

    def test_default_values(self) -> None:
        """测试 ConsolidateResult 默认值均为零。"""
        result = ConsolidateResult()
        assert result.new_entities == 0
        assert result.updated_entities == 0
        assert result.new_preferences == 0
        assert result.updated_preferences == 0
        assert result.new_relations == 0
        assert result.episodes_processed == 0
        assert result.errors == []
        assert result.dry_run is False

    def test_with_values(self) -> None:
        """测试 ConsolidateResult 可设置非零值。"""
        result = ConsolidateResult(
            new_entities=5,
            updated_entities=3,
            new_preferences=2,
            updated_preferences=1,
            new_relations=4,
            episodes_processed=10,
            errors=["解析失败: 第3条"],
            dry_run=True,
        )
        assert result.new_entities == 5
        assert result.updated_entities == 3
        assert result.new_preferences == 2
        assert result.updated_preferences == 1
        assert result.new_relations == 4
        assert result.episodes_processed == 10
        assert result.errors == ["解析失败: 第3条"]
        assert result.dry_run is True

    def test_serialization(self) -> None:
        """测试 ConsolidateResult 序列化往返。"""
        result = ConsolidateResult(
            new_entities=1,
            episodes_processed=3,
            errors=["err1", "err2"],
        )
        json_str = result.model_dump_json()
        restored = ConsolidateResult.model_validate_json(json_str)

        assert restored.new_entities == 1
        assert restored.episodes_processed == 3
        assert restored.errors == ["err1", "err2"]

    def test_dry_run_serialization(self) -> None:
        """测试 dry_run 字段正确序列化。"""
        result = ConsolidateResult(dry_run=True)
        data = json.loads(result.model_dump_json())
        assert data["dry_run"] is True

    def test_negative_values_raises(self) -> None:
        """测试计数类字段不接受负值。"""
        with pytest.raises(ValidationError):
            ConsolidateResult(new_entities=-1)
        with pytest.raises(ValidationError):
            ConsolidateResult(episodes_processed=-5)
