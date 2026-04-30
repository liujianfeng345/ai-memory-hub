"""
SemanticMemory 单元测试。

验证语义实体的 CRUD 操作、同名合并（upsert）、属性深度合并、
双向关系管理、级联删除和嵌入向量不泄露等功能。
"""

import pytest

from memory_agent.core.semantic_memory import SemanticMemory


@pytest.mark.asyncio
class TestAddEntity:
    """添加实体测试。"""

    async def test_add_and_search_entity(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """add_entity 写入后，search_entities 可检索到。"""
        entity = await semantic_memory.add_entity(
            name="张三",
            entity_type="person",
            description="一个喜欢咖啡的用户",
            attributes={"年龄": 30},
        )
        assert entity.id is not None
        assert entity.name == "张三"
        assert entity.entity_type == "person"
        assert entity.description == "一个喜欢咖啡的用户"
        assert entity.attributes == {"年龄": 30}

        # 搜索验证
        results = await semantic_memory.search_entities(
            query="喜欢咖啡的人",
            top_k=5,
        )
        assert len(results) >= 1
        found = any("张三" == ent.name for ent in results)
        assert found, f"搜索结果应包含'张三'，实际: {[ent.name for ent in results]}"

    async def test_entity_type_validation(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """entity_type 不合法时抛出 ValueError。"""
        with pytest.raises(ValueError, match="entity_type"):
            await semantic_memory.add_entity(
                name="测试",
                entity_type="invalid_type",
                description="测试描述",
            )

    async def test_add_entity_valid_types(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """所有合法 entity_type 均可正常创建。"""
        valid_types = ["person", "organization", "topic", "preference", "fact"]
        for i, etype in enumerate(valid_types):
            entity = await semantic_memory.add_entity(
                name=f"实体_{i}",
                entity_type=etype,
                description=f"类型为 {etype} 的实体",
            )
            assert entity.entity_type == etype


@pytest.mark.asyncio
class TestEntityMerge:
    """同名实体合并（upsert）测试。"""

    async def test_merge_same_name_entity(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """同名实体第二次 add_entity 应合并更新。"""
        # 第一次创建
        await semantic_memory.add_entity(
            name="张三",
            entity_type="person",
            description="一个喜欢咖啡的用户",
            attributes={"年龄": 30},
        )

        # 第二次同名创建（合并）
        entity = await semantic_memory.add_entity(
            name="张三",
            entity_type="person",
            description="一个喜欢咖啡和茶的用户",
            attributes={"城市": "北京"},
        )

        assert entity.name == "张三"
        # description 应使用新值覆盖
        assert "茶" in entity.description
        # attributes 应深度合并
        assert "年龄" in entity.attributes, f"attributes 应保留'年龄': {entity.attributes}"
        assert "城市" in entity.attributes, f"attributes 应新增'城市': {entity.attributes}"
        assert entity.attributes["年龄"] == 30
        assert entity.attributes["城市"] == "北京"

    async def test_merge_attributes_deep_merge(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """attributes 深度合并：嵌套字典递归合并。"""
        # 第一次创建，带嵌套属性
        await semantic_memory.add_entity(
            name="李四",
            entity_type="person",
            description="一个开发者",
            attributes={"地址": {"城市": "北京", "区域": "朝阳"}},
        )

        # 第二次 merge，添加新的嵌套属性
        entity = await semantic_memory.add_entity(
            name="李四",
            entity_type="person",
            description="一个资深开发者",
            attributes={"地址": {"区域": "海淀", "街道": "中关村"}, "技能": "Python"},
        )

        # 深度合并：城市应保留，区域应被覆盖
        assert entity.attributes["地址"]["城市"] == "北京"  # 保留
        assert entity.attributes["地址"]["区域"] == "海淀"  # 被覆盖
        assert entity.attributes["地址"]["街道"] == "中关村"  # 新增
        assert entity.attributes["技能"] == "Python"  # 新增

    async def test_merge_related_entities_deduplicate(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """合并时 related_entities 追加并去重。"""
        entity_a = await semantic_memory.add_entity(
            name="实体A",
            entity_type="topic",
            description="第一个实体",
        )
        entity_b = await semantic_memory.add_entity(
            name="实体B",
            entity_type="topic",
            description="第二个实体",
        )

        # 创建带 related_entities 的实体
        await semantic_memory.add_entity(
            name="实体C",
            entity_type="topic",
            description="第三个实体",
            related_entities=[entity_a.id],
        )

        # 合并时追加 entity_b，同时重复 entity_a
        result = await semantic_memory.add_entity(
            name="实体C",
            entity_type="topic",
            description="第三个实体（更新）",
            related_entities=[entity_a.id, entity_b.id],
        )

        # 应去重
        assert entity_a.id in result.related_entities
        assert entity_b.id in result.related_entities
        # 检查没有重复
        unique = set(result.related_entities)
        assert len(unique) == len(result.related_entities)


@pytest.mark.asyncio
class TestSearchEntities:
    """搜索实体测试。"""

    async def test_search_entities_entity_type_filter(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """search_entities entity_type 过滤。"""
        await semantic_memory.add_entity(
            name="用户王五",
            entity_type="person",
            description="一个喜欢登山的用户",
        )
        await semantic_memory.add_entity(
            name="Python编程",
            entity_type="topic",
            description="Python 编程语言相关知识",
        )

        # 只搜索 person 类型
        results = await semantic_memory.search_entities(
            query="用户偏好",
            entity_type="person",
            top_k=10,
        )
        assert len(results) >= 1
        for entity in results:
            assert entity.entity_type == "person"

    async def test_search_entities_empty_store(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """空存储中搜索返回空列表。"""
        results = await semantic_memory.search_entities(query="任意查询")
        assert results == []


@pytest.mark.asyncio
class TestGetEntity:
    """get_entity 测试。"""

    async def test_get_entity_returns_correct(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """get_entity 按 ID 返回正确的实体。"""
        entity = await semantic_memory.add_entity(
            name="赵六",
            entity_type="person",
            description="一个喜欢读书的用户",
            attributes={"年龄": 25},
        )

        found = await semantic_memory.get_entity(entity.id)
        assert found is not None
        assert found.id == entity.id
        assert found.name == "赵六"
        assert found.description == "一个喜欢读书的用户"

    async def test_get_entity_nonexistent_returns_none(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """get_entity 对不存在的 ID 返回 None。"""
        found = await semantic_memory.get_entity("nonexistent-id")
        assert found is None


@pytest.mark.asyncio
class TestGetPreferences:
    """get_preferences 测试。"""

    async def test_get_preferences_returns_preference_entities(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """get_preferences 返回类型为 'preference' 的实体。"""
        # 写入多种类型的实体
        await semantic_memory.add_entity(
            name="偏好1",
            entity_type="preference",
            description="喜欢喝咖啡",
        )
        await semantic_memory.add_entity(
            name="偏好2",
            entity_type="preference",
            description="喜欢爬山",
        )
        await semantic_memory.add_entity(
            name="用户A",
            entity_type="person",
            description="一个用户",
        )
        await semantic_memory.add_entity(
            name="爱好话题",
            entity_type="topic",
            description="兴趣爱好",
        )

        results = await semantic_memory.get_preferences()
        assert len(results) == 2, f"预期 2 条 preference，实际 {len(results)}"
        for entity in results:
            assert entity.entity_type == "preference"

    async def test_get_preferences_empty(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """没有 preference 类型实体时返回空列表。"""
        await semantic_memory.add_entity(
            name="用户A",
            entity_type="person",
            description="一个用户",
        )
        results = await semantic_memory.get_preferences()
        assert results == []


@pytest.mark.asyncio
class TestUpdateEntity:
    """update_entity 测试。"""

    async def test_update_entity_description(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """update_entity 更新 description 后重新计算嵌入向量。"""
        entity = await semantic_memory.add_entity(
            name="待更新实体",
            entity_type="topic",
            description="原始描述",
            attributes={"版本": 1},
        )

        updated = await semantic_memory.update_entity(
            entity.id,
            updates={"description": "更新后的新描述", "attributes": {"版本": 2, "状态": "已更新"}},
        )

        assert updated.description == "更新后的新描述"
        assert updated.attributes["版本"] == 2
        assert updated.attributes["状态"] == "已更新"

        # 通过 get_entity 确认持久化成功
        found = await semantic_memory.get_entity(entity.id)
        assert found is not None
        assert found.description == "更新后的新描述"

    async def test_update_entity_nonexistent_raises(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """update_entity 对不存在的 ID 抛出 ValueError。"""
        with pytest.raises(ValueError, match="实体不存在"):
            await semantic_memory.update_entity(
                "nonexistent-id",
                updates={"description": "新描述"},
            )


@pytest.mark.asyncio
class TestRelations:
    """关系管理测试。"""

    async def test_add_relation_bidirectional(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """add_relation 建立双向关系后 get_related_entities 返回关联实体。"""
        entity_a = await semantic_memory.add_entity(
            name="用户A",
            entity_type="person",
            description="一个用户",
        )
        entity_b = await semantic_memory.add_entity(
            name="偏好B",
            entity_type="preference",
            description="喜欢喝咖啡",
        )

        await semantic_memory.add_relation(
            source_id=entity_a.id,
            target_id=entity_b.id,
            relation_type="has_preference",
        )

        # 从 A 获取关联实体，应包含 B
        related_a = await semantic_memory.get_related_entities(entity_a.id)
        assert len(related_a) >= 1
        rel_ids = [ent.id for ent in related_a]
        assert entity_b.id in rel_ids, f"A 的关联实体应包含 B，实际: {rel_ids}"

        # 双向关系：从 B 获取关联实体，应包含 A
        related_b = await semantic_memory.get_related_entities(entity_b.id)
        rel_b_ids = [ent.id for ent in related_b]
        assert entity_a.id in rel_b_ids, f"B 的关联实体应包含 A，实际: {rel_b_ids}"

    async def test_add_relation_nonexistent_raises(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """add_relation 对不存在的源实体抛出 ValueError。"""
        with pytest.raises(ValueError, match="源实体不存在"):
            await semantic_memory.add_relation(
                source_id="nonexistent",
                target_id="also_nonexistent",
                relation_type="test",
            )

    async def test_add_relation_target_nonexistent_raises(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """add_relation 对不存在的目标实体抛出 ValueError。"""
        entity_a = await semantic_memory.add_entity(
            name="实体A",
            entity_type="topic",
            description="测试实体",
        )
        with pytest.raises(ValueError, match="目标实体不存在"):
            await semantic_memory.add_relation(
                source_id=entity_a.id,
                target_id="nonexistent",
                relation_type="test",
            )

    async def test_get_related_entities_empty(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """无关联实体时 get_related_entities 返回空列表。"""
        entity = await semantic_memory.add_entity(
            name="孤立实体",
            entity_type="topic",
            description="没有关联的实体",
        )
        results = await semantic_memory.get_related_entities(entity.id)
        assert results == []

    async def test_chain_relations(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """链式关系：A → B → C，A 只关联 B，B 同时关联 A 和 C。"""
        entity_a = await semantic_memory.add_entity(name="A", entity_type="topic", description="实体A")
        entity_b = await semantic_memory.add_entity(name="B", entity_type="topic", description="实体B")
        entity_c = await semantic_memory.add_entity(name="C", entity_type="topic", description="实体C")

        await semantic_memory.add_relation(entity_a.id, entity_b.id, "related")
        await semantic_memory.add_relation(entity_b.id, entity_c.id, "related")

        # A 只直接关联 B
        related_a = await semantic_memory.get_related_entities(entity_a.id)
        assert len(related_a) == 1
        assert related_a[0].id == entity_b.id

        # B 关联 A 和 C
        related_b = await semantic_memory.get_related_entities(entity_b.id)
        assert len(related_b) == 2


@pytest.mark.asyncio
class TestRemoveEntity:
    """remove_entity 级联清理测试。"""

    async def test_remove_entity_cascade_cleanup(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """remove_entity 级联清理关联关系。"""
        entity_a = await semantic_memory.add_entity(
            name="实体A",
            entity_type="person",
            description="用户A",
        )
        entity_b = await semantic_memory.add_entity(
            name="实体B",
            entity_type="preference",
            description="偏好B",
        )

        # 建立双向关系
        await semantic_memory.add_relation(
            source_id=entity_a.id,
            target_id=entity_b.id,
            relation_type="has_preference",
        )

        # 确认关系存在
        related_a = await semantic_memory.get_related_entities(entity_a.id)
        assert len(related_a) >= 1

        # 删除 A
        removed = await semantic_memory.remove_entity(entity_a.id)
        assert removed is True

        # A 应不存在
        assert await semantic_memory.get_entity(entity_a.id) is None

        # B 的 related_entities 中不应再包含 A
        b_after = await semantic_memory.get_entity(entity_b.id)
        assert b_after is not None
        assert entity_a.id not in b_after.related_entities, "删除 A 后，B 的 related_entities 不应包含 A"

    async def test_remove_entity_nonexistent_returns_false(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """remove_entity 对不存在的 ID 返回 False。"""
        result = await semantic_memory.remove_entity("nonexistent-id")
        assert result is False


@pytest.mark.asyncio
class TestNoEmbeddingLeak:
    """验证返回对象不含 embedding 字段。"""

    async def test_add_entity_returns_no_embedding(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """add_entity 返回的 Entity 对象 embedding 字段为 None。"""
        entity = await semantic_memory.add_entity(
            name="测试实体",
            entity_type="fact",
            description="测试无嵌入泄露",
        )
        assert entity.embedding is None

    async def test_search_entities_returns_no_embedding(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """search_entities 返回的 Entity 对象 embedding 字段为 None。"""
        await semantic_memory.add_entity(
            name="搜索实体",
            entity_type="topic",
            description="搜索测试内容",
        )

        results = await semantic_memory.search_entities(query="搜索", top_k=5)
        for ent in results:
            assert ent.embedding is None, f"Entity {ent.id} 不应包含 embedding"

    async def test_get_entity_returns_no_embedding(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """get_entity 返回的 Entity 对象 embedding 字段为 None。"""
        entity = await semantic_memory.add_entity(
            name="获取实体",
            entity_type="fact",
            description="获取测试内容",
        )

        found = await semantic_memory.get_entity(entity.id)
        assert found is not None
        assert found.embedding is None

    async def test_get_preferences_returns_no_embedding(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """get_preferences 返回的 Entity 对象 embedding 字段为 None。"""
        await semantic_memory.add_entity(
            name="测试偏好",
            entity_type="preference",
            description="偏好测试内容",
        )

        results = await semantic_memory.get_preferences()
        for ent in results:
            assert ent.embedding is None

    async def test_update_entity_returns_no_embedding(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """update_entity 返回的 Entity 对象 embedding 字段为 None。"""
        entity = await semantic_memory.add_entity(
            name="待更新",
            entity_type="topic",
            description="更新前描述",
        )

        updated = await semantic_memory.update_entity(
            entity.id,
            updates={"description": "更新后描述"},
        )
        assert updated.embedding is None

    async def test_get_related_entities_returns_no_embedding(
        self,
        semantic_memory: SemanticMemory,
    ) -> None:
        """get_related_entities 返回的 Entity 对象 embedding 字段为 None。"""
        entity_a = await semantic_memory.add_entity(name="A", entity_type="topic", description="实体A")
        entity_b = await semantic_memory.add_entity(name="B", entity_type="topic", description="实体B")
        await semantic_memory.add_relation(entity_a.id, entity_b.id, "related")

        results = await semantic_memory.get_related_entities(entity_a.id)
        for ent in results:
            assert ent.embedding is None
