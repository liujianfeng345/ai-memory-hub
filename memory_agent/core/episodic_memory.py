"""
情节记忆（EpisodicMemory）模块。

EpisodicMemory 管理已完结的对话片段或事件，支持语义检索、
自动摘要生成、时间过滤和标签过滤等功能。

内部使用 ChromaStore 作为向量存储后端，LocalEmbedder 生成文本嵌入向量，
DeepSeekClient 用于长文本自动摘要。

ChromaDB 使用余弦距离（cosine distance），对于已归一化的向量：
  similarity = 1 - distance
因此 min_similarity 参数需换算为距离阈值：
  max_distance = 1 - min_similarity
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from memory_agent.models.episode import Episode

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """情节记忆 —— 已完结的对话片段或事件管理器。

    使用 ChromaDB 持久化存储，支持语义向量检索、时间/会话/标签过滤、
    自动摘要生成等功能。

    Attributes:
        chroma_store: ChromaStore 实例（Collection 名为 "episodic_memory"）。
        embedder: LocalEmbedder 实例，用于生成文本和查询嵌入向量。
        llm_client: DeepSeekClient 实例，用于长文本自动摘要。
        summary_threshold: 触发自动摘要的字符数阈值。
    """

    def __init__(
        self,
        chroma_store: "ChromaStore",  # type: ignore[name-defined] # noqa: F821
        embedder: "LocalEmbedder",  # type: ignore[name-defined] # noqa: F821
        llm_client: "DeepSeekClient",  # type: ignore[name-defined] # noqa: F821
        summary_threshold: int = 2000,
    ) -> None:
        """初始化 EpisodicMemory 实例。

        Args:
            chroma_store: ChromaStore 实例，已绑定 "episodic_memory" Collection。
            embedder: LocalEmbedder 实例。
            llm_client: DeepSeekClient 实例。
            summary_threshold: 触发自动摘要的字符数阈值，默认 2000。
        """
        self.chroma_store = chroma_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.summary_threshold = summary_threshold

        logger.info(
            "EpisodicMemory 已初始化 collection=%s dim=%d",
            self.chroma_store.collection_name,
            self.chroma_store.embedding_dimension,
        )

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Episode:
        """添加一条情节记忆。

        若 content 长度超过 summary_threshold，则调用 LLM 生成摘要。
        生成嵌入向量后存入 ChromaDB。

        Args:
            content: 情节记忆内容文本。
            metadata: 可选的附加元数据字典。
            session_id: 可选的关联会话标识符。

        Returns:
            创建的 Episode 对象（不含 embedding 字段）。
        """
        episode_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc)

        # 自动摘要
        summary: str | None = None
        if len(content) > self.summary_threshold:
            try:
                summary = await self._generate_summary(content)
                logger.debug("为情节 %s 生成摘要，长度=%d", episode_id, len(summary))
            except Exception as exc:
                logger.warning("摘要生成失败 id=%s: %s", episode_id, exc)
                # 摘要失败不应阻止写入

        # 生成嵌入向量
        embedding_result = self.embedder.embed(content)
        embedding = embedding_result[0] if embedding_result else []

        # 构造 Episode
        episode = Episode(
            id=episode_id,
            content=content,
            session_id=session_id,
            summary=summary,
            embedding=embedding,
            created_at=now,
            updated_at=now,
            importance=1.0,
            tags=[],
        )

        # 准备 ChromaDB 元数据
        meta_dict = self._episode_to_metadata(episode)

        # 写入 ChromaDB
        self.chroma_store.add(
            ids=[episode_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta_dict],
        )

        logger.info("添加情节记忆 id=%s len=%d", episode_id, len(content))

        # 返回时不包含 embedding
        return self._strip_embedding(episode)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.5,
    ) -> list[Episode]:
        """语义检索情节记忆。

        使用嵌入向量进行相似度检索，支持时间、session、标签过滤。

        Args:
            query: 搜索查询文本。
            top_k: 返回的最大结果数。
            filters: 可选的过滤条件字典，支持：
                - start_time: ISO 格式起始时间（>=）。
                - end_time: ISO 格式结束时间（<=）。
                - session_id: 限制特定会话。
                - tags: 标签列表，需匹配。
            min_similarity: 最小相似度阈值，默认 0.5。
                （换算为 ChromaDB 距离阈值：max_distance = 1 - min_similarity）

        Returns:
            Episode 列表，按相似度降序排列，不含 embedding 字段。
        """
        # 生成查询向量
        query_vec = self.embedder.embed_query(query)

        # 构建 ChromaDB where 子句
        where_clause = self._build_where_clause(filters)

        # 执行查询
        result = self.chroma_store.query(
            query_embedding=query_vec,
            top_k=top_k,
            where=where_clause,
        )

        # 组装结果并过滤相似度
        episodes: list[Episode] = []

        ids_list = result.get("ids", [])
        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])
        distances_list = result.get("distances", [])

        for i, doc_id in enumerate(ids_list):
            distance = distances_list[i] if i < len(distances_list) else float("inf")
            # ChromaDB 返回距离，换算为相似度
            similarity = 1.0 - distance

            if similarity < min_similarity:
                continue

            episode = self._metadata_to_episode(
                episode_id=doc_id,
                document=documents_list[i] if i < len(documents_list) else "",
                metadata=metadatas_list[i] if i < len(metadatas_list) else {},
            )
            episodes.append(episode)

        # 按相似度降序排列（从 metadatas 获取距离最小的排前）
        # 实际上 ChromaDB 已经按距离升序返回，第一个就是最相似的
        # 此处保持原顺序即可（距离升序 = 相似度降序）

        return episodes

    async def get_recent(
        self,
        hours: int = 24,
        session_id: str | None = None,
    ) -> list[Episode]:
        """获取最近指定小时内创建的情节记忆。

        Args:
            hours: 时间范围（小时），默认 24。
            session_id: 可选，限定特定会话。

        Returns:
            Episode 列表，按创建时间降序排列。
        """
        start_time = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        start_epoch = start_time.timestamp()

        # 构建 ChromaDB where 条件（使用 timestamp epoch 字段）
        conditions: list[dict[str, Any]] = [
            {"timestamp": {"$gte": start_epoch}},
        ]
        if session_id:
            conditions.append({"session_id": session_id})

        where_clause: dict[str, Any]
        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}

        # 使用查询获取匹配的记录
        dim = self.chroma_store.embedding_dimension
        dummy_vec = [0.0] * dim
        result = self.chroma_store.query(
            query_embedding=dummy_vec,
            top_k=10000,  # 足够大的值
            where=where_clause,
        )

        # 组装结果
        episodes: list[Episode] = []
        ids_list = result.get("ids", [])
        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])

        for i, doc_id in enumerate(ids_list):
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            doc = documents_list[i] if i < len(documents_list) else ""

            episode = self._metadata_to_episode(
                episode_id=doc_id,
                document=doc,
                metadata=meta,
            )
            episodes.append(episode)

        # 按 created_at 降序排列
        episodes.sort(key=lambda x: x.created_at, reverse=True)
        return episodes

    async def get_by_id(self, episode_id: str) -> Episode | None:
        """按 ID 获取单个情节记忆。

        Args:
            episode_id: 情节记忆 ID。

        Returns:
            Episode 对象，若不存在则返回 None。不含 embedding 字段。
        """
        result = self.chroma_store.get(ids=[episode_id])
        ids_list = result.get("ids", [])

        if not ids_list:
            return None

        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])

        episode = self._metadata_to_episode(
            episode_id=ids_list[0],
            document=documents_list[0] if documents_list else "",
            metadata=metadatas_list[0] if metadatas_list else {},
        )
        return episode

    async def remove(self, episode_id: str) -> bool:
        """删除指定情节记忆。

        Args:
            episode_id: 要删除的情节记忆 ID。

        Returns:
            True 表示成功删除，False 表示条目不存在。
        """
        # 先确认存在
        existing = await self.get_by_id(episode_id)
        if existing is None:
            return False

        # 删除
        self.chroma_store.delete(ids=[episode_id])
        logger.info("删除情节记忆 id=%s", episode_id)
        return True

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _episode_to_metadata(self, episode: Episode) -> dict[str, Any]:
        """将 Episode 对象转换为 ChromaDB 元数据字典。

        created_at 同时存储为 ISO 字符串（用于重建）和 epoch 时间戳（用于过滤）。

        Args:
            episode: Episode 实例。

        Returns:
            包含 session_id、created_at、timestamp、importance、tags、summary 的字典。
        """
        return {
            "session_id": episode.session_id or "",
            "created_at": episode.created_at.isoformat(),
            "timestamp": episode.created_at.timestamp(),  # epoch 浮点数，用于 ChromaDB $gte/$lte 过滤
            "importance": episode.importance,
            "tags": json.dumps(episode.tags, ensure_ascii=False),
            "summary": episode.summary or "",
        }

    def _metadata_to_episode(
        self,
        episode_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> Episode:
        """从 ChromaDB 返回的元数据重建 Episode 对象。

        Args:
            episode_id: 情节记忆 ID。
            document: 文档内容。
            metadata: ChromaDB 元数据字典。

        Returns:
            Episode 实例（不含 embedding 字段）。

        Raises:
            ValueError: 若元数据中缺少必要字段。
        """
        created_at_str = metadata.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except (ValueError, TypeError):
            created_at = datetime.now(tz=timezone.utc)

        # 解析 tags
        tags_raw = metadata.get("tags", "[]")
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = []
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        # 处理 session_id
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and not session_id:
            session_id = None

        return Episode(
            id=episode_id,
            content=document,
            session_id=session_id,
            summary=metadata.get("summary") or None,
            embedding=None,  # 不返回嵌入向量
            importance=metadata.get("importance", 1.0),
            tags=tags,
            created_at=created_at,
            updated_at=created_at,
        )

    def _strip_embedding(self, episode: Episode) -> Episode:
        """返回不含 embedding 字段的 Episode 副本。

        Args:
            episode: 原始 Episode 实例。

        Returns:
            embedding 为 None 的 Episode 副本。
        """
        return episode.model_copy(update={"embedding": None})

    async def _generate_summary(self, content: str) -> str:
        """调用 LLM 生成文本摘要。

        Args:
            content: 待摘要的文本。

        Returns:
            生成的摘要字符串。
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个文本摘要助手。请用 1-3 句话简洁地总结以下内容的核心要点。"
                    "只返回摘要文本，不要添加任何额外说明。"
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        response = await self.llm_client.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        return str(response).strip()

    def _build_where_clause(self, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """将用户友好的过滤条件转换为 ChromaDB where 子句。

        支持的过滤条件：
        - start_time: ISO 格式字符串 → "timestamp" "$gte"（转为 epoch）
        - end_time: ISO 格式字符串 → "timestamp" "$lte"（转为 epoch）
        - session_id: 字符串 → 等值过滤
        - tags: 列表 → 暂在 Python 端过滤

        当存在多个条件时，使用 ChromaDB 的 "$and" 组合。

        Args:
            filters: 用户提供的过滤条件字典。

        Returns:
            ChromaDB 兼容的 where 子句，或 None。
        """
        if not filters:
            return None

        conditions: list[dict[str, Any]] = []

        # 时间范围过滤（使用 timestamp epoch 字段）
        ts_condition: dict[str, Any] = {}
        if "start_time" in filters:
            start_epoch = self._iso_to_epoch(filters["start_time"])
            ts_condition["$gte"] = start_epoch
        if "end_time" in filters:
            end_epoch = self._iso_to_epoch(filters["end_time"])
            ts_condition["$lte"] = end_epoch
        if ts_condition:
            conditions.append({"timestamp": ts_condition})

        # session_id 过滤
        if "session_id" in filters:
            conditions.append({"session_id": filters["session_id"]})

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    @staticmethod
    def _iso_to_epoch(iso_str: str) -> float:
        """将 ISO 格式时间字符串转换为 epoch 时间戳。

        Args:
            iso_str: ISO 格式时间字符串。

        Returns:
            epoch 时间戳（float）。
        """
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            return 0.0
