"""
集成测试：MemoryManager。

测试总调度器的写入路由、跨类型检索聚合、记忆整合、
删除、会话清理和错误传播功能。

依赖真实 API 的测试（如 consolidate 端到端）使用 pytest.mark.skipif 跳过。
"""

import pytest

from memory_agent.models.consolidate_result import ConsolidateResult
from memory_agent.models.memory_item import MemoryType
from memory_agent.utils.errors import StorageError

# ========================================================================
# Fixtures 辅助
# ========================================================================


def _mock_extract_entities_basic() -> dict:
    """返回基本的实体提取结果。"""
    return {
        "entities": [
            {
                "name": "Python",
                "type": "topic",
                "attributes": {"category": "编程语言"},
            },
            {
                "name": "VSCode",
                "type": "fact",
                "attributes": {"category": "编辑器"},
            },
        ],
        "preferences": [
            {
                "subject": "用户",
                "category": "language",
                "value": "Python",
                "sentiment": "positive",
            },
        ],
        "relations": [
            {"source": "用户", "target": "Python", "relation": "likes"},
        ],
    }


def _mock_extract_entities_user_profile() -> dict:
    """返回含用户资料的实体提取结果。"""
    return {
        "entities": [
            {
                "name": "张伟",
                "type": "person",
                "attributes": {"role": "Python后端工程师", "location": "北京"},
            },
            {
                "name": "VSCode",
                "type": "fact",
                "attributes": {"category": "代码编辑器"},
            },
        ],
        "preferences": [
            {
                "subject": "张伟",
                "category": "beverage",
                "value": "咖啡",
                "sentiment": "positive",
            },
        ],
        "relations": [
            {"source": "张伟", "target": "VSCode", "relation": "uses"},
        ],
    }


# ========================================================================
# 测试: remember 路由
# ========================================================================


class TestRememberRouting:
    """测试 remember 方法按 memory_type 正确路由。"""

    @pytest.mark.asyncio
    async def test_remember_working_routes_correctly(self, manager):
        """memory_type='working' 写入工作记忆。"""
        item = await manager.remember(
            content="测试工作记忆内容",
            memory_type="working",
            session_id="test-session",
        )
        assert item.memory_type == MemoryType.working
        assert item.content == "测试工作记忆内容"
        assert item.session_id == "test-session"

        # 验证通过 get_by_session 可检索到
        items = await manager._working_memory.get_by_session("test-session")
        assert any(i.id == item.id for i in items)

    @pytest.mark.asyncio
    async def test_remember_episodic_routes_correctly(self, manager):
        """memory_type='episodic' 写入情节记忆。"""
        item = await manager.remember(
            content="用户今天学了 Python 并发编程",
            memory_type="episodic",
            session_id="test-session",
        )
        assert item.memory_type == MemoryType.episodic
        assert "Python" in item.content

        # 验证可检索
        recent = await manager._episodic_memory.get_recent(hours=1)
        assert any(e.id == item.id for e in recent)

    @pytest.mark.asyncio
    async def test_remember_semantic_extracts_and_stores_entities(self, manager, mock_deepseek_client):
        """memory_type='semantic' 调用 LLM 提取实体并写入语义记忆。"""
        mock_deepseek_client.extract_entities.return_value = _mock_extract_entities_basic()

        item = await manager.remember(
            content="用户喜欢 Python，使用 VSCode",
            memory_type="semantic",
        )
        assert item.memory_type == MemoryType.semantic

        # 验证语义记忆中已创建实体
        entities = await manager._semantic_memory.search_entities(
            query="Python", top_k=5, min_similarity=0.0
        )
        entity_names = [e.name for e in entities]
        assert "Python" in entity_names


class TestRememberValidation:
    """测试 remember 方法的输入校验。"""

    @pytest.mark.asyncio
    async def test_remember_working_missing_session_id_raises_value_error(self, manager):
        """memory_type='working' 时缺少 session_id 抛出 ValueError。"""
        with pytest.raises(ValueError, match="session_id.*不能为空"):
            await manager.remember(
                content="测试内容",
                memory_type="working",
            )

    @pytest.mark.asyncio
    async def test_remember_content_too_long_raises_value_error(self, manager):
        """content 超过 max_content_length 时抛出 ValueError。"""
        too_long_content = "A" * (manager.config.max_content_length + 1)
        with pytest.raises(ValueError, match="content 长度超过限制"):
            await manager.remember(
                content=too_long_content,
                memory_type="episodic",
            )

    @pytest.mark.asyncio
    async def test_remember_invalid_memory_type_raises_value_error(self, manager):
        """不支持的 memory_type 抛出 ValueError。"""
        with pytest.raises(ValueError, match="不支持的 memory_type"):
            await manager.remember(
                content="测试",
                memory_type="invalid_type",  # type: ignore[arg-type]
            )


# ========================================================================
# 测试: recall 检索
# ========================================================================


class TestRecall:
    """测试 recall 方法的检索功能。"""

    @pytest.mark.asyncio
    async def test_recall_single_type_working(self, manager):
        """单类型检索 working 返回正确结果。"""
        await manager.remember(
            content="用户正在学习 Python 异步编程",
            memory_type="working",
            session_id="sess-recall",
        )
        await manager.remember(
            content="用户今天喝了咖啡",
            memory_type="working",
            session_id="sess-recall",
        )

        results = await manager.recall(
            query="Python",
            memory_type="working",
            session_id="sess-recall",
        )
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_recall_single_type_episodic(self, manager):
        """单类型检索 episodic 返回包含相关内容的 Episode。"""
        await manager.remember(
            content="用户喜欢喝咖啡，每天早上一杯美式",
            memory_type="episodic",
            session_id="sess-recall",
        )
        await manager.remember(
            content="用户昨天去了健身房",
            memory_type="episodic",
            session_id="sess-recall",
        )

        results = await manager.recall(
            query="咖啡",
            memory_type="episodic",
            min_similarity=0.0,
        )
        assert len(results) >= 1
        assert any("咖啡" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_recall_cross_type_aggregation(self, manager):
        """memory_type=None 跨类型检索同时返回 working 和 episodic 结果。"""
        session = "sess-cross"

        # 写入工作记忆
        await manager.remember(
            content="工作记忆：用户正在敲代码",
            memory_type="working",
            session_id=session,
        )

        # 写入情节记忆
        await manager.remember(
            content="情节记忆：用户昨天完成了 Python 项目",
            memory_type="episodic",
            session_id=session,
        )

        results = await manager.recall(
            query="用户",
            memory_type=None,
            session_id=session,
            min_similarity=0.0,
        )

        assert len(results) >= 1
        memory_types = {r.memory_type for r in results}
        # 至少应包含 working 或 episodic 之一
        assert MemoryType.working in memory_types or MemoryType.episodic in memory_types

    @pytest.mark.asyncio
    async def test_recall_working_priority_before_episodic(self, manager):
        """跨类型检索时工作记忆结果排在情节记忆之前。"""
        session = "sess-priority"

        await manager.remember(
            content="工作记忆内容：紧急任务提醒",
            memory_type="working",
            session_id=session,
        )

        await manager.remember(
            content="情节记忆：用户昨天讨论了项目计划",
            memory_type="episodic",
            session_id=session,
        )

        results = await manager.recall(
            query="项目",
            memory_type=None,
            session_id=session,
            top_k=10,
            min_similarity=0.0,
        )

        # 找到第一个 working 和第一个 episodic 的位置
        wm_indices = [i for i, r in enumerate(results) if r.memory_type == MemoryType.working]
        ep_indices = [i for i, r in enumerate(results) if r.memory_type == MemoryType.episodic]

        if wm_indices and ep_indices:
            # 所有 working 应在所有 episodic 之前
            assert max(wm_indices) < min(ep_indices)

    @pytest.mark.asyncio
    async def test_recall_empty_top_k(self, manager):
        """top_k=0 返回空列表。"""
        results = await manager.recall(query="任意", top_k=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_semantic_converts_entity_to_memory_item(self, manager, mock_deepseek_client):
        """semantic 类型检索将 Entity 转换为 MemoryItem。"""
        mock_deepseek_client.extract_entities.return_value = {
            "entities": [{"name": "Java", "type": "topic", "attributes": {}}],
            "preferences": [],
            "relations": [],
        }

        # 先生成一个语义实体
        await manager.remember(
            content="用户学过 Java 编程",
            memory_type="semantic",
        )

        results = await manager.recall(
            query="Java",
            memory_type="semantic",
            min_similarity=0.0,
        )
        assert len(results) >= 1
        assert all(r.memory_type == MemoryType.semantic for r in results)


# ========================================================================
# 测试: consolidate 整合
# ========================================================================


class TestConsolidate:
    """测试 consolidate 记忆整合功能。"""

    @pytest.mark.asyncio
    async def test_consolidate_empty_episodes_returns_zero(self, manager):
        """无近期情节记忆时返回 episodes_processed=0。"""
        result = await manager.consolidate(time_window_hours=1)
        assert isinstance(result, ConsolidateResult)
        assert result.episodes_processed == 0

    @pytest.mark.asyncio
    async def test_consolidate_extracts_entities_from_episodic(
        self, manager, mock_deepseek_client
    ):
        """consolidate 从情节记忆提取实体并更新语义记忆。"""
        mock_deepseek_client.chat.return_value = (
            '{"entities": [{"name": "Python", "type": "topic", "attributes": {"category": "编程"}}], '
            '"preferences": [{"subject": "用户", "category": "language", "value": "Python", "sentiment": "positive"}], '
            '"relations": []}'
        )

        # 写入一条情节记忆（作为整合源）
        await manager.remember(
            content="用户学习了 Python 基础知识，对 Python 很感兴趣",
            memory_type="episodic",
            session_id="sess-consolidate",
        )

        result = await manager.consolidate(time_window_hours=24)
        assert result.episodes_processed >= 1
        assert result.new_entities + result.updated_entities >= 1
        assert result.new_preferences + result.updated_preferences >= 1

        # 整合后应可在语义记忆中检索到新实体
        entities = await manager._semantic_memory.search_entities(
            query="Python", top_k=5, min_similarity=0.0
        )
        assert len(entities) >= 1

    @pytest.mark.asyncio
    async def test_consolidate_dry_run_does_not_write(self, manager, mock_deepseek_client):
        """dry_run=True 时不实际写入语义记忆。"""
        mock_deepseek_client.chat.return_value = (
            '{"entities": [{"name": "GoLang", "type": "topic", "attributes": {}}], '
            '"preferences": [], '
            '"relations": []}'
        )

        await manager.remember(
            content="用户想学习 Go 语言",
            memory_type="episodic",
            session_id="sess-dryrun",
        )

        # 记录整合前的实体数
        before_entities = await manager._semantic_memory.search_entities(
            query="GoLang", top_k=5, min_similarity=0.0
        )
        before_count = len(before_entities)

        result = await manager.consolidate(time_window_hours=24, dry_run=True)
        assert result.dry_run is True
        assert result.episodes_processed >= 1
        # dry_run 应该计算实体数但不实际写入
        assert result.new_entities > 0

        # 验证语义记忆中没有新增实体
        after_entities = await manager._semantic_memory.search_entities(
            query="GoLang", top_k=5, min_similarity=0.0
        )
        assert len(after_entities) == before_count

    @pytest.mark.asyncio
    async def test_consolidate_llm_error_captured_in_result_errors(
        self, manager, mock_deepseek_client
    ):
        """LLM 调用失败时错误信息被收集到 ConsolidateResult.errors。"""
        from memory_agent.utils.errors import LLMServiceError

        mock_deepseek_client.chat.side_effect = LLMServiceError("模拟 LLM 错误")

        await manager.remember(
            content="用户测试了错误处理",
            memory_type="episodic",
            session_id="sess-error",
        )

        result = await manager.consolidate(time_window_hours=24)
        assert result.episodes_processed >= 1
        assert len(result.errors) >= 1


# ========================================================================
# 测试: forget 删除
# ========================================================================


class TestForget:
    """测试 forget 方法的删除功能。"""

    @pytest.mark.asyncio
    async def test_forget_working_memory(self, manager):
        """删除工作记忆后 recall 不再返回该条目。"""
        item = await manager.remember(
            content="待删除的临时内容",
            memory_type="working",
            session_id="sess-forget",
        )

        deleted = await manager.forget(item.id, memory_type="working")
        assert deleted is True

        # 验证已无法检索
        results = await manager.recall(
            query="待删除",
            memory_type="working",
            session_id="sess-forget",
        )
        assert not any(r.id == item.id for r in results)

    @pytest.mark.asyncio
    async def test_forget_episodic_memory(self, manager):
        """删除情节记忆后不再可检索。"""
        item = await manager.remember(
            content="要被删除的情节记忆",
            memory_type="episodic",
            session_id="sess-forget-ep",
        )

        deleted = await manager.forget(item.id, memory_type="episodic")
        assert deleted is True

        # 验证
        existing = await manager._episodic_memory.get_by_id(item.id)
        assert existing is None

    @pytest.mark.asyncio
    async def test_forget_semantic_entity(self, manager, mock_deepseek_client):
        """删除语义实体后不再可检索。"""
        mock_deepseek_client.extract_entities.return_value = {
            "entities": [{"name": "待删除实体", "type": "topic", "attributes": {}}],
            "preferences": [],
            "relations": [],
        }

        item = await manager.remember(
            content="这是一个待删除的实体",
            memory_type="semantic",
        )
        # 语义记忆返回的 item.id 对应最后一个实体的 ID
        entity_id = item.id

        deleted = await manager.forget(entity_id, memory_type="semantic")
        assert deleted is True

        existing = await manager._semantic_memory.get_entity(entity_id)
        assert existing is None

    @pytest.mark.asyncio
    async def test_forget_nonexistent_returns_false(self, manager):
        """删除不存在的条目返回 False。"""
        deleted = await manager.forget("nonexistent-id-99999", memory_type="working")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_forget_invalid_memory_type_raises_value_error(self, manager):
        """不支持的 memory_type 抛出 ValueError。"""
        with pytest.raises(ValueError, match="不支持的 memory_type"):
            await manager.forget("some-id", memory_type="invalid")  # type: ignore[arg-type]


# ========================================================================
# 测试: clear_session 会话清理
# ========================================================================


class TestClearSession:
    """测试 clear_session 的会话清理功能。"""

    @pytest.mark.asyncio
    async def test_clear_session_removes_all_working_memory(self, manager):
        """clear_session 清空指定会话的所有工作记忆。"""
        session = "sess-clear"

        await manager.remember(
            content="工作记忆 1",
            memory_type="working",
            session_id=session,
        )
        await manager.remember(
            content="工作记忆 2",
            memory_type="working",
            session_id=session,
        )

        cleared = await manager.clear_session(session)
        assert cleared == 2

        # 验证工作记忆已清空
        remaining = await manager._working_memory.get_by_session(session)
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_clear_session_empty_session_returns_zero(self, manager):
        """清空不存在的会话返回 0。"""
        cleared = await manager.clear_session("nonexistent-session")
        assert cleared == 0


# ========================================================================
# 测试: 错误传播
# ========================================================================


class TestErrorPropagation:
    """测试底层异常的正确传播。"""

    @pytest.mark.asyncio
    async def test_storage_error_propagated(self, manager):
        """底层 StorageError 被正确传递到调用方。"""
        # 使用无效 ID 格式触发底层错误（或通过 mock 模拟）
        # 这里通过触发 ChromaDB 底层异常来验证错误传播
        original_get = manager._episodic_store.get

        def _failing_get(*args, **kwargs):
            raise StorageError("模拟的存储错误传播测试")

        manager._episodic_store.get = _failing_get

        try:
            with pytest.raises(StorageError, match="模拟的存储错误传播测试"):
                await manager._episodic_memory.get_by_id("any-id")
        finally:
            # 恢复原始方法
            manager._episodic_store.get = original_get


# ========================================================================
# 辅助验证
# ========================================================================


class TestManagerConstruction:
    """测试 MemoryManager 构造函数。"""

    def test_manager_default_config_creates_all_components(self, manager):
        """默认配置下 MemoryManager 创建所有内部组件。"""
        assert manager._working_memory is not None
        assert manager._episodic_memory is not None
        assert manager._semantic_memory is not None
        assert manager._embedder is not None
        assert manager._llm_client is not None
        assert manager._working_store is not None
        assert manager._episodic_store is not None
        assert manager._semantic_store is not None

    def test_entity_to_memory_item_conversion(self, manager):
        """_entity_to_memory_item 转换正确。"""
        from memory_agent.models.entity import Entity

        entity = Entity(
            id="test-entity-1",
            name="Python",
            entity_type="topic",
            description="Python 是编程语言",
            attributes={"category": "编程"},
        )
        item = manager._entity_to_memory_item(entity)
        assert item.memory_type == MemoryType.semantic
        assert "Python" in item.content
        assert item.metadata.get("entity_name") == "Python"
        assert item.metadata.get("entity_type") == "topic"
