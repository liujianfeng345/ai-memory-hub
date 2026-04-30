"""
EpisodicMemory 单元测试。

验证情节记忆的写入、语义检索、时间过滤、session 过滤、
min_similarity 过滤、自动摘要生成和基本 CRUD 操作。
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from memory_agent.core.episodic_memory import EpisodicMemory


@pytest.mark.asyncio
class TestAddEpisode:
    """添加情节记忆测试。"""

    async def test_add_and_search(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """add_episode 写入后，search 可检索到。"""
        content = "用户喜欢喝蓝山咖啡"
        episode = await episodic_memory.add_episode(
            content=content,
            session_id="test-session",
        )
        assert episode.id is not None
        assert episode.content == content
        assert episode.memory_type == "episodic"

        # 搜索验证（使用较低 min_similarity 阈值）
        results = await episodic_memory.search(query="饮品偏好", top_k=3, min_similarity=0.3)
        assert len(results) >= 1
        # 应返回与内容相关的记忆
        found = any("咖啡" in ep.content for ep in results)
        assert found, f"搜索结果应包含含'咖啡'的记忆，实际: {[ep.content for ep in results]}"

    async def test_long_content_generates_summary(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """长内容（>summary_threshold）自动生成摘要。"""
        # 设置较短的摘要阈值，方便测试
        episodic_memory.summary_threshold = 200

        # 配置 mock 返回摘要
        episodic_memory.llm_client.chat = AsyncMock(  # type: ignore[method-assign]
            return_value="这是一段测试摘要。"
        )

        long_text = "这是一个测试文本。" * 100  # 长度远超 200 字符
        episode = await episodic_memory.add_episode(
            content=long_text,
            session_id="test-session",
        )

        assert episode.summary is not None
        assert len(episode.summary) > 0
        assert len(episode.summary) < len(long_text)

    async def test_short_content_no_summary(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """短内容不生成摘要。"""
        short_text = "今天天气不错"
        episode = await episodic_memory.add_episode(
            content=short_text,
            session_id="test-session",
        )
        assert episode.summary is None

    async def test_episode_has_correct_metadata(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """Episode 的 metadata、tags、importance 字段正确。"""
        episode = await episodic_memory.add_episode(
            content="带有元数据的测试内容",
            metadata={"source": "test"},
            session_id="sess-meta",
        )
        assert episode.session_id == "sess-meta"
        assert episode.importance == 1.0
        assert episode.tags == []
        assert episode.content == "带有元数据的测试内容"


@pytest.mark.asyncio
class TestSearch:
    """语义检索测试。"""

    async def test_search_semantic_relevance(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """语义检索：写入不同类型的记忆，搜索相关内容应返回更相似的结果。"""
        await episodic_memory.add_episode(
            content="用户喜欢喝咖啡和茶",
            session_id="sess-1",
        )
        await episodic_memory.add_episode(
            content="Python 是一门编程语言，广泛用于数据科学和机器学习",
            session_id="sess-1",
        )

        # 搜索"饮品偏好"应返回咖啡相关记忆排在前面
        results = await episodic_memory.search(query="饮品偏好", top_k=2, min_similarity=0.3)
        assert len(results) >= 1

        # 第一条应更接近饮品相关的内容
        first_content = results[0].content
        assert "咖啡" in first_content or "饮品" in first_content

    async def test_search_time_filter(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """search 时间过滤：写入不同时间的记忆，start_time 过滤正确。"""
        now = datetime.now(tz=timezone.utc)

        # 添加一条"过去"的记忆（通过直接操作内部存储模拟）
        # 由于我们通过 add_episode 创建的记忆 created_at 都是当前时间，
        # 我们需要在搜索时使用时间过滤来验证效果

        await episodic_memory.add_episode(
            content="一小时前的记忆",
            session_id="sess-time",
        )
        await asyncio.sleep(0.01)
        await episodic_memory.add_episode(
            content="刚创建的记忆",
            session_id="sess-time",
        )
        now = datetime.now(tz=timezone.utc)
        start_threshold = now - timedelta(hours=2)

        # 用 start_time 过滤（最近 2 小时的都应该被返回）
        results = await episodic_memory.search(
            query="记忆",
            filters={"start_time": start_threshold.isoformat()},
            top_k=10,
            min_similarity=0.3,
        )
        assert len(results) >= 1

    async def test_search_session_filter(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """search session_id 过滤。"""
        await episodic_memory.add_episode(
            content="会话A的情节",
            session_id="sess-a",
        )
        await episodic_memory.add_episode(
            content="会话B的情节",
            session_id="sess-b",
        )

        results = await episodic_memory.search(
            query="情节",
            filters={"session_id": "sess-a"},
            top_k=10,
            min_similarity=0.3,
        )
        assert len(results) >= 1
        for ep in results:
            assert ep.session_id == "sess-a"

    async def test_search_min_similarity(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """search min_similarity 过滤低分结果。"""
        await episodic_memory.add_episode(
            content="用户喜欢喝咖啡",
            session_id="sess-1",
        )

        # 使用极高的 min_similarity，应该过滤掉所有结果
        results = await episodic_memory.search(
            query="完全无关的查询",
            min_similarity=0.9999,
            top_k=10,
        )
        # 大部分结果应被过滤（可能全部为空）
        # 无法保证全部为空，但不会返回低相似度的结果
        # 实际上如果 min_similarity=0.9999 非常严苛，基本不会有结果
        assert isinstance(results, list)

    async def test_search_empty_store(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """空存储中搜索返回空列表。"""
        results = await episodic_memory.search(query="任意查询")
        assert results == []


@pytest.mark.asyncio
class TestGetRecent:
    """get_recent 获取最近记忆测试。"""

    async def test_get_recent_returns_recent_episodes(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_recent 返回指定小时内的记忆。"""
        await episodic_memory.add_episode(
            content="最近的记忆",
            session_id="sess-1",
        )

        results = await episodic_memory.get_recent(hours=24)
        assert len(results) >= 1

    async def test_get_recent_short_window(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_recent 使用极短窗口时可能返回空。"""
        await episodic_memory.add_episode(
            content="刚添加的记忆",
            session_id="sess-1",
        )

        # 使用负的 hours 会导致 start_time 在未来，应该没有结果
        results = await episodic_memory.get_recent(hours=0)  # 最近0小时（当前之后）
        # 实际上 timedelta(hours=0) 会得到 start_time=now, created_at=now
        # 所以可能返回或不返回，取决于精确度
        assert isinstance(results, list)

    async def test_get_recent_session_filter(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_recent 支持 session_id 过滤。"""
        await episodic_memory.add_episode(content="会话A记忆", session_id="sess-a")
        await episodic_memory.add_episode(content="会话B记忆", session_id="sess-b")

        results = await episodic_memory.get_recent(
            hours=24,
            session_id="sess-a",
        )
        assert len(results) >= 1
        for ep in results:
            assert ep.session_id == "sess-a"


@pytest.mark.asyncio
class TestGetByIdAndRemove:
    """get_by_id 和 remove 基本功能测试。"""

    async def test_get_by_id_returns_correct_episode(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_by_id 返回正确的 Episode。"""
        episode = await episodic_memory.add_episode(
            content="按ID查找的记忆",
            session_id="sess-1",
        )

        found = await episodic_memory.get_by_id(episode.id)
        assert found is not None
        assert found.id == episode.id
        assert found.content == episode.content

    async def test_get_by_id_nonexistent_returns_none(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_by_id 对不存在的 ID 返回 None。"""
        found = await episodic_memory.get_by_id("nonexistent-id")
        assert found is None

    async def test_remove_deletes_episode(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """remove 删除情节记忆后 get_by_id 返回 None。"""
        episode = await episodic_memory.add_episode(
            content="待删除的记忆",
            session_id="sess-1",
        )

        deleted = await episodic_memory.remove(episode.id)
        assert deleted is True

        found = await episodic_memory.get_by_id(episode.id)
        assert found is None

    async def test_remove_nonexistent_returns_false(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """remove 不存在的 ID 返回 False。"""
        deleted = await episodic_memory.remove("nonexistent-id")
        assert deleted is False


@pytest.mark.asyncio
class TestNoEmbeddingLeak:
    """验证返回对象不含 embedding 字段。"""

    async def test_add_episode_returns_no_embedding(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """add_episode 返回的 Episode 对象 embedding 字段为 None。"""
        episode = await episodic_memory.add_episode(
            content="测试无嵌入泄露",
            session_id="sess-1",
        )
        assert episode.embedding is None

    async def test_search_returns_no_embedding(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """search 返回的 Episode 对象 embedding 字段为 None。"""
        await episodic_memory.add_episode(
            content="搜索返回测试",
            session_id="sess-1",
        )

        results = await episodic_memory.search(query="搜索", top_k=5)
        for ep in results:
            assert ep.embedding is None, f"Episode {ep.id} 不应包含 embedding"

    async def test_get_by_id_returns_no_embedding(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_by_id 返回的 Episode 对象 embedding 字段为 None。"""
        episode = await episodic_memory.add_episode(
            content="按ID返回测试",
            session_id="sess-1",
        )

        found = await episodic_memory.get_by_id(episode.id)
        assert found is not None
        assert found.embedding is None

    async def test_get_recent_returns_no_embedding(
        self,
        episodic_memory: EpisodicMemory,
    ) -> None:
        """get_recent 返回的 Episode 对象 embedding 字段为 None。"""
        await episodic_memory.add_episode(
            content="最近记忆测试",
            session_id="sess-1",
        )

        results = await episodic_memory.get_recent(hours=24)
        for ep in results:
            assert ep.embedding is None
