"""
语义记忆（SemanticMemory）模块。

SemanticMemory 管理从情节记忆中提取的持久化知识，包括人物、组织、
话题、偏好和事实等实体类型。支持实体合并（upsert）、属性深度合并、
双向关系管理和级联删除等功能。

内部使用专用 ChromaDB Collection "semantic_memory" 存储实体向量和元数据。

深度合并说明：
- 实体的 attributes 字典采用递归深度合并策略：
  嵌套字典进行递归合并，标量值直接覆盖，列表值直接替换。
  例如：{"地址": {"城市": "北京"}} + {"地址": {"区域": "海淀"}}
  → {"地址": {"城市": "北京", "区域": "海淀"}}
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from memory_agent.models.entity import VALID_ENTITY_TYPES, Entity

logger = logging.getLogger(__name__)


def _deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """递归深度合并两个字典。

    合并策略：
    - 对于嵌套字典：递归合并。
    - 对于其他类型：overlay 的值覆盖 base 的值。
    - base 和 overlay 中原有的键均保留。

    Args:
        base: 基础字典。
        overlay: 覆盖字典，其值优先级更高。

    Returns:
        合并后的新字典（不修改输入字典）。
    """
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 深度合并嵌套字典
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # 覆盖或新增
            result[key] = value
    return result


class SemanticMemory:
    """语义记忆 —— 持久化知识管理器。

    使用 ChromaDB 作为向量存储后端，管理人物、组织、话题、
    偏好和事实等实体类型的增删改查。

    Attributes:
        chroma_store: ChromaStore 实例（Collection 名为 "semantic_memory"）。
        embedder: LocalEmbedder 实例。
        llm_client: DeepSeekClient 实例（预留，当前阶段未使用）。
    """

    def __init__(
        self,
        chroma_store: "ChromaStore",  # type: ignore[name-defined] # noqa: F821
        embedder: "LocalEmbedder",  # type: ignore[name-defined] # noqa: F821
        llm_client: "DeepSeekClient",  # type: ignore[name-defined] # noqa: F821
    ) -> None:
        """初始化 SemanticMemory 实例。

        Args:
            chroma_store: ChromaStore 实例，已绑定 "semantic_memory" Collection。
            embedder: LocalEmbedder 实例。
            llm_client: DeepSeekClient 实例（预留）。
        """
        self.chroma_store = chroma_store
        self.embedder = embedder
        self.llm_client = llm_client

        logger.info(
            "SemanticMemory 已初始化 collection=%s dim=%d",
            self.chroma_store.collection_name,
            self.chroma_store.embedding_dimension,
        )

    # ------------------------------------------------------------------
    # 公开方法: 写入与查询
    # ------------------------------------------------------------------

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        attributes: dict[str, Any] | None = None,
        related_entities: list[str] | None = None,
    ) -> Entity:
        """添加或更新语义实体（upsert）。

        若已存在同名字的实体，则进行合并更新：
        - description 使用最新值
        - attributes 深度合并（嵌套字典递归合并）
        - related_entities 追加并去重
        - 嵌入向量重新计算

        若不存在同名字实体，则新建。

        Args:
            name: 实体名称。
            entity_type: 实体类型，必须在 {"person", "organization", "topic", "preference", "fact"} 中。
            description: 实体描述文本。
            attributes: 可选的附加属性字典。
            related_entities: 可选的关联实体 ID 列表。

        Returns:
            创建或更新后的 Entity 对象（不含 embedding 字段）。

        Raises:
            ValueError: 当 entity_type 不合法时抛出。
        """
        # 校验 entity_type
        if entity_type.lower() not in VALID_ENTITY_TYPES:
            raise ValueError(f"entity_type 必须是 {VALID_ENTITY_TYPES} 之一，当前值: {entity_type}")
        entity_type = entity_type.lower()

        # 查找是否已存在同名字实体
        existing = await self._find_entity_by_name(name)

        if existing is not None:
            # 合并更新
            entity = await self._merge_entity(
                existing=existing,
                new_description=description,
                new_attributes=attributes or {},
                new_related_entities=related_entities or [],
            )
            logger.info(
                "合并实体 name=%s type=%s id=%s",
                name,
                entity_type,
                entity.id,
            )
        else:
            # 新建实体
            entity_id = str(uuid.uuid4())
            now = datetime.now(tz=timezone.utc)

            entity = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type,
                description=description,
                attributes=attributes or {},
                embedding=None,
                related_entities=related_entities or [],
                created_at=now,
                updated_at=now,
                confidence=1.0,
            )

            # 生成嵌入向量
            embedding = self._compute_embedding(entity)
            entity.embedding = embedding

            # 写入 ChromaDB
            self._write_entity(entity)

            logger.info(
                "创建实体 name=%s type=%s id=%s",
                name,
                entity_type,
                entity_id,
            )

        # 返回时不包含 embedding 字段
        return self._strip_embedding(entity)

    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_type: str | None = None,
        min_similarity: float = 0.5,
    ) -> list[Entity]:
        """语义检索实体。

        使用嵌入向量相似度查询，支持按实体类型过滤。

        Args:
            query: 搜索查询文本。
            top_k: 返回的最大结果数。
            entity_type: 可选的实体类型过滤。
            min_similarity: 最小相似度阈值，默认 0.5。

        Returns:
            Entity 列表，按相似度降序排列，不含 embedding 字段。
        """
        query_vec = self.embedder.embed_query(query)

        # 构建 where 子句
        where_clause: dict[str, Any] | None = None
        if entity_type:
            where_clause = {"entity_type": entity_type.lower()}

        result = self.chroma_store.query(
            query_embedding=query_vec,
            top_k=top_k,
            where=where_clause,
        )

        entities: list[Entity] = []

        ids_list = result.get("ids", [])
        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])
        distances_list = result.get("distances", [])

        for i, doc_id in enumerate(ids_list):
            distance = distances_list[i] if i < len(distances_list) else float("inf")
            similarity = 1.0 - distance

            if similarity < min_similarity:
                continue

            entity = self._metadata_to_entity(
                entity_id=doc_id,
                document=documents_list[i] if i < len(documents_list) else "",
                metadata=metadatas_list[i] if i < len(metadatas_list) else {},
            )
            entities.append(entity)

        return entities

    async def get_entity(self, entity_id: str) -> Entity | None:
        """按 ID 获取单个实体。

        Args:
            entity_id: 实体 ID。

        Returns:
            Entity 对象，若不存在则返回 None。不含 embedding 字段。
        """
        result = self.chroma_store.get(ids=[entity_id])
        ids_list = result.get("ids", [])

        if not ids_list:
            return None

        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])

        return self._metadata_to_entity(
            entity_id=ids_list[0],
            document=documents_list[0] if documents_list else "",
            metadata=metadatas_list[0] if metadatas_list else {},
        )

    async def get_preferences(
        self,
        user_id: str | None = None,
    ) -> list[Entity]:
        """获取所有偏好类型的实体。

        Args:
            user_id: 可选，过滤特定用户 ID 的偏好（在 attributes.user_id 中匹配）。

        Returns:
            Entity 列表，不含 embedding 字段。
        """
        # 使用 ChromaDB 查询获取所有 "preference" 类型的实体
        # 使用一个较大的 top_k 和零向量来获取全部
        dim = self.chroma_store.embedding_dimension
        dummy_vec = [0.0] * dim

        result = self.chroma_store.query(
            query_embedding=dummy_vec,
            top_k=10000,
            where={"entity_type": "preference"},
        )

        entities: list[Entity] = []
        ids_list = result.get("ids", [])
        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])

        for i, doc_id in enumerate(ids_list):
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            doc = documents_list[i] if i < len(documents_list) else ""

            # Python 端二次过滤 user_id
            if user_id is not None:
                attrs = self._parse_attributes(meta.get("attributes", "{}"))
                attr_user_id = attrs.get("user_id")
                if attr_user_id != user_id:
                    continue

            entity = self._metadata_to_entity(
                entity_id=doc_id,
                document=doc,
                metadata=meta,
            )
            entities.append(entity)

        return entities

    async def update_entity(
        self,
        entity_id: str,
        updates: dict[str, Any],
    ) -> Entity:
        """更新实体字段。

        采用 "先删除旧记录，再写入新记录" 的方式（ChromaDB 不支持原地更新）。
        若 updates 中包含 description 且与原值不同，则重新计算嵌入向量。

        Args:
            entity_id: 要更新的实体 ID。
            updates: 要更新的字段及新值字典。

        Returns:
            更新后的 Entity 对象。

        Raises:
            ValueError: 当实体不存在时抛出。
        """
        existing = await self.get_entity(entity_id)
        if existing is None:
            raise ValueError(f"实体不存在: {entity_id}")

        # 合并 updates 到现有实体
        # 对 attributes 做深度合并
        attrs_update = updates.pop("attributes", None)

        update_data = {
            "description": updates.get("description", existing.description),
            "entity_type": updates.get("entity_type", existing.entity_type),
            "name": updates.get("name", existing.name),
            "confidence": updates.get("confidence", existing.confidence),
            "related_entities": updates.get("related_entities", existing.related_entities),
        }

        # 处理 attributes 深度合并
        if attrs_update is not None and isinstance(attrs_update, dict):
            new_attributes = _deep_merge_dicts(existing.attributes, attrs_update)
        else:
            new_attributes = existing.attributes

        # 检查 description 是否变化
        description_changed = update_data["description"] != existing.description

        # 构造更新后的 Entity
        now = datetime.now(tz=timezone.utc)
        updated_entity = Entity(
            id=entity_id,
            name=update_data["name"],
            entity_type=update_data["entity_type"],
            description=update_data["description"],
            attributes=new_attributes,
            embedding=None,
            related_entities=update_data["related_entities"],
            created_at=existing.created_at,
            updated_at=now,
            confidence=update_data["confidence"],
        )

        # 若 description 变化，重新计算嵌入向量；否则复用旧向量
        if description_changed:
            updated_entity.embedding = self._compute_embedding(updated_entity)
            logger.debug("description 已变化，重新计算嵌入向量 id=%s", entity_id)
        else:
            # 需要从 ChromaDB 中重新获取 embedding
            # （实际上 get 不从 ChromaDB 返回 embedding，但我们仍需写入 embedding 向量）
            # 直接重新计算以避免依赖旧向量
            updated_entity.embedding = self._compute_embedding(updated_entity)

        # 删除旧记录，写入新记录
        self.chroma_store.delete(ids=[entity_id])
        self._write_entity(updated_entity)

        logger.info("更新实体 id=%s", entity_id)

        return self._strip_embedding(updated_entity)

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
    ) -> None:
        """建立两个实体之间的双向关系。

        将 target_id 添加到 source 实体的 related_entities 中，
        同时将 source_id 添加到 target 实体的 related_entities 中。

        Args:
            source_id: 关系源实体 ID。
            target_id: 关系目标实体 ID。
            relation_type: 关系类型字符串。

        Raises:
            ValueError: 当任一实体不存在时抛出。
        """
        source = await self.get_entity(source_id)
        if source is None:
            raise ValueError(f"源实体不存在: {source_id}")

        target = await self.get_entity(target_id)
        if target is None:
            raise ValueError(f"目标实体不存在: {target_id}")

        # 更新 source 的 related_entities
        source_related = list(set(source.related_entities + [target_id]))
        await self._update_entity_relations(source_id, source_related)

        # 同步更新 target 的 related_entities（双向关系）
        target_related = list(set(target.related_entities + [source_id]))
        await self._update_entity_relations(target_id, target_related)

        logger.info(
            "建立关系 source=%s target=%s type=%s",
            source_id,
            target_id,
            relation_type,
        )

    async def get_related_entities(
        self,
        entity_id: str,
        relation_type: str | None = None,
    ) -> list[Entity]:
        """获取与指定实体关联的所有实体。

        Args:
            entity_id: 实体 ID。
            relation_type: 可选的关系类型过滤。
                （当前版本 relations 不单独存储类型，此处预留参数）。

        Returns:
            关联的 Entity 列表。

        Raises:
            ValueError: 当实体不存在时抛出。
        """
        entity = await self.get_entity(entity_id)
        if entity is None:
            raise ValueError(f"实体不存在: {entity_id}")

        related_ids = entity.related_entities
        if not related_ids:
            return []

        # 批量获取关联实体
        result = self.chroma_store.get(ids=related_ids)

        entities: list[Entity] = []
        ids_list = result.get("ids", [])
        documents_list = result.get("documents", [])
        metadatas_list = result.get("metadatas", [])

        for i, doc_id in enumerate(ids_list):
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            doc = documents_list[i] if i < len(documents_list) else ""

            # 如果有关系类型过滤（当前版本：relations 不单独存储类型）
            if relation_type is not None:
                # 预留：未来可在 metadata 中存储关系类型再过滤
                pass

            related_entity = self._metadata_to_entity(
                entity_id=doc_id,
                document=doc,
                metadata=meta,
            )
            entities.append(related_entity)

        return entities

    async def remove_entity(self, entity_id: str) -> bool:
        """删除实体并级联清理所有关联关系。

        1. 获取待删除实体。
        2. 从所有关联实体的 related_entities 中移除该 entity_id。
        3. 删除 ChromaDB 中的记录。

        Args:
            entity_id: 要删除的实体 ID。

        Returns:
            True 表示成功删除，False 表示实体不存在。
        """
        entity = await self.get_entity(entity_id)
        if entity is None:
            return False

        # 从所有关联实体的 related_entities 中移除该 ID
        for related_id in entity.related_entities:
            try:
                related_entity = await self.get_entity(related_id)
                if related_entity is not None:
                    new_related = [rid for rid in related_entity.related_entities if rid != entity_id]
                    if new_related != related_entity.related_entities:
                        await self._update_entity_relations(related_id, new_related)
                        logger.debug(
                            "从实体 %s 的 related_entities 中移除 %s",
                            related_id,
                            entity_id,
                        )
            except Exception as exc:
                logger.warning(
                    "清理关联关系时出错 entity=%s related=%s: %s",
                    entity_id,
                    related_id,
                    exc,
                )

        # 删除 ChromaDB 中的记录
        self.chroma_store.delete(ids=[entity_id])
        logger.info("删除实体 id=%s name=%s", entity_id, entity.name)
        return True

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _find_entity_by_name(self, name: str) -> Entity | None:
        """在 ChromaDB 中按名称查找实体。

        Args:
            name: 实体名称。

        Returns:
            找到的 Entity 对象，若不存在则返回 None。
        """
        # 使用 ChromaDB metadata 过滤查询
        dim = self.chroma_store.embedding_dimension
        dummy_vec = [0.0] * dim

        try:
            result = self.chroma_store.query(
                query_embedding=dummy_vec,
                top_k=10000,
                where={"name": name},
            )
        except Exception:
            # 如果 name 字段的 where 查询失败，退回全量扫描
            all_result = self.chroma_store.get(ids=[])
            result = {
                "ids": all_result.get("ids", []),
                "documents": all_result.get("documents", []),
                "metadatas": all_result.get("metadatas", []),
                "distances": [0.0] * len(all_result.get("ids", [])),
            }

        ids_list = result.get("ids", [])
        metadatas_list = result.get("metadatas", [])
        documents_list = result.get("documents", [])

        for i, meta in enumerate(metadatas_list):
            if meta.get("name") == name:
                return self._metadata_to_entity(
                    entity_id=ids_list[i] if i < len(ids_list) else "",
                    document=documents_list[i] if i < len(documents_list) else "",
                    metadata=meta,
                )

        return None

    async def _merge_entity(
        self,
        existing: Entity,
        new_description: str,
        new_attributes: dict[str, Any],
        new_related_entities: list[str],
    ) -> Entity:
        """合并更新已有实体。

        策略：
        - description: 使用新值覆盖
        - attributes: 深度合并
        - related_entities: 追加并去重
        - 重新计算嵌入向量

        Args:
            existing: 现有实体。
            new_description: 新的描述文本。
            new_attributes: 新的属性字典（与现有属性深度合并）。
            new_related_entities: 新的关联实体 ID 列表。

        Returns:
            合并后的 Entity 对象。
        """
        now = datetime.now(tz=timezone.utc)

        # 深度合并 attributes
        merged_attributes = _deep_merge_dicts(existing.attributes, new_attributes)

        # 合并 related_entities 并去重
        merged_related = list(dict.fromkeys(existing.related_entities + new_related_entities))

        merged_entity = Entity(
            id=existing.id,
            name=existing.name,
            entity_type=existing.entity_type,
            description=new_description,
            attributes=merged_attributes,
            embedding=None,
            related_entities=merged_related,
            created_at=existing.created_at,
            updated_at=now,
            confidence=existing.confidence,
        )

        # description 变化，重新计算嵌入向量
        merged_entity.embedding = self._compute_embedding(merged_entity)

        # 删除旧记录，写入新记录
        self.chroma_store.delete(ids=[existing.id])
        self._write_entity(merged_entity)

        return merged_entity

    async def _update_entity_relations(
        self,
        entity_id: str,
        related_entities: list[str],
    ) -> None:
        """更新实体的 related_entities 列表（仅更新此字段，不改变其他数据）。

        采用删除+重新写入的方式（ChromaDB 的限制）。

        Args:
            entity_id: 实体 ID。
            related_entities: 新的关联实体 ID 列表。
        """
        entity = await self.get_entity(entity_id)
        if entity is None:
            return

        now = datetime.now(tz=timezone.utc)
        updated = Entity(
            id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            attributes=entity.attributes,
            embedding=None,
            related_entities=related_entities,
            created_at=entity.created_at,
            updated_at=now,
            confidence=entity.confidence,
        )

        # 重新计算嵌入向量以写入
        updated.embedding = self._compute_embedding(updated)

        # 删除旧记录，写入新记录
        self.chroma_store.delete(ids=[entity_id])
        self._write_entity(updated)

    def _compute_embedding(self, entity: Entity) -> list[float]:
        """计算实体的嵌入向量。

        使用 entity.description 作为嵌入文本。

        Args:
            entity: 实体对象。

        Returns:
            嵌入向量。
        """
        embedding_result = self.embedder.embed(entity.description)
        return embedding_result[0] if embedding_result else []

    def _write_entity(self, entity: Entity) -> None:
        """将实体写入 ChromaDB。

        Args:
            entity: 实体对象（必须已设置 embedding 字段）。
        """
        embedding = entity.embedding
        if embedding is None:
            embedding = self._compute_embedding(entity)

        meta_dict = self._entity_to_metadata(entity)
        self.chroma_store.add(
            ids=[entity.id],
            documents=[entity.description],
            embeddings=[embedding],
            metadatas=[meta_dict],
        )

    # ------------------------------------------------------------------
    # 序列化辅助方法
    # ------------------------------------------------------------------

    def _entity_to_metadata(self, entity: Entity) -> dict[str, Any]:
        """将 Entity 对象转换为 ChromaDB 元数据字典。

        Args:
            entity: Entity 实例。

        Returns:
            包含 name、entity_type、attributes、related_entities、
            created_at、updated_at、confidence 的字典。
        """
        return {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "attributes": json.dumps(entity.attributes, ensure_ascii=False),
            "related_entities": json.dumps(entity.related_entities, ensure_ascii=False),
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "confidence": entity.confidence,
        }

    def _metadata_to_entity(
        self,
        entity_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> Entity:
        """从 ChromaDB 返回的元数据重建 Entity 对象。

        不返回 embedding 字段。

        Args:
            entity_id: 实体 ID。
            document: 存储的文档内容（即 description）。
            metadata: ChromaDB 元数据字典。

        Returns:
            Entity 实例（embedding=None）。
        """
        created_at = self._parse_datetime(metadata.get("created_at", ""))
        updated_at = self._parse_datetime(metadata.get("updated_at", ""))

        return Entity(
            id=entity_id,
            name=metadata.get("name", ""),
            entity_type=metadata.get("entity_type", "fact"),
            description=document,
            attributes=self._parse_attributes(metadata.get("attributes", "{}")),
            embedding=None,  # 不返回嵌入向量
            related_entities=self._parse_related_entities(metadata.get("related_entities", "[]")),
            created_at=created_at,
            updated_at=updated_at,
            confidence=metadata.get("confidence", 1.0),
        )

    def _strip_embedding(self, entity: Entity) -> Entity:
        """返回不含 embedding 字段的 Entity 副本。

        Args:
            entity: 原始 Entity 实例。

        Returns:
            embedding 为 None 的 Entity 副本。
        """
        return entity.model_copy(update={"embedding": None})

    @staticmethod
    def _parse_datetime(dt_str: str) -> datetime:
        """解析 ISO 格式日期时间字符串。

        Args:
            dt_str: ISO 格式字符串。

        Returns:
            datetime 对象，解析失败时返回当前 UTC 时间。
        """
        try:
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            return datetime.now(tz=timezone.utc)

    @staticmethod
    def _parse_attributes(attrs_raw: Any) -> dict[str, Any]:
        """解析 attributes JSON 字符串。

        Args:
            attrs_raw: JSON 字符串或已解析的字典。

        Returns:
            解析后的属性字典。
        """
        if isinstance(attrs_raw, dict):
            return attrs_raw
        if isinstance(attrs_raw, str):
            try:
                parsed = json.loads(attrs_raw)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    @staticmethod
    def _parse_related_entities(raw: Any) -> list[str]:
        """解析 related_entities JSON 字符串。

        Args:
            raw: JSON 字符串或已解析的列表。

        Returns:
            解析后的实体 ID 列表。
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return []
