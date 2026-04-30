"""
MemoryManager —— 总调度器。

MemoryManager 是 memory-agent 的唯一用户入口，承担依赖装配器角色。
负责将工作记忆、情节记忆、语义记忆三个子系统协调为一个统一接口，
提供记忆写入路由、跨类型检索聚合、记忆整合（consolidate）等能力。

ChromaDB PersistentClient 不支持多进程并发写入，MemoryManager 的使用者
应注意不要在多个进程中共享同一持久化目录。
"""

import asyncio
import json
import logging
import time
from typing import Any, Literal

from memory_agent.core.episodic_memory import EpisodicMemory
from memory_agent.core.semantic_memory import SemanticMemory
from memory_agent.core.working_memory import WorkingMemory
from memory_agent.embedding.local_embedder import LocalEmbedder
from memory_agent.llm.deepseek_client import DeepSeekClient
from memory_agent.models.consolidate_result import ConsolidateResult
from memory_agent.models.entity import VALID_ENTITY_TYPES, Entity
from memory_agent.models.memory_item import MemoryItem, MemoryType
from memory_agent.storage.chroma_store import ChromaStore
from memory_agent.storage.in_memory_store import InMemoryStore
from memory_agent.utils.config import MemoryConfig
from memory_agent.utils.errors import (
    LLMResponseParseError,
    StorageError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# consolidate 使用的 few-shot 提示词
# 帮助 LLM 稳定输出符合预期的 JSON 结构
# ---------------------------------------------------------------------------

_CONSOLIDATE_SYSTEM_PROMPT = """你是一个信息整合助手。请从以下对话片段中提取三类结构化信息：

1. **entities**: 文本中提到的人、物、地点、组织等实体。每个实体应包含：
   - name: 实体名称
   - type: 实体类型（person, location, organization, object, event, other）
   - attributes: 实体的属性描述（可选）

2. **preferences**: 文本中表达的用户偏好或倾向。每条偏好应包含：
   - subject: 偏好主体（通常是用户或文本中的角色）
   - category: 偏好类别（如 food, activity, technology, sport 等）
   - value: 偏好的具体内容
   - sentiment: 情感倾向（positive, negative, neutral）

3. **relations**: 实体之间的关系。每条关系应包含：
   - source: 关系起点实体名称
   - target: 关系终点实体名称
   - relation: 关系类型（如 likes, works_at, lives_in, knows 等）

请以 JSON 格式返回，包含 entities、preferences、relations 三个键。
若某类信息不存在，对应键的值应为空列表。

--- 示例 1 ---
输入：
用户张伟是一名Python后端工程师，在北京工作。他喜欢喝咖啡，特别是拿铁。他常用VSCode编辑器，与李华在同一团队。

输出：
{
  "entities": [
    {"name": "张伟", "type": "person", "attributes": {"role": "Python后端工程师", "location": "北京"}},
    {"name": "李华", "type": "person", "attributes": {"team": "与张伟同团队"}},
    {"name": "VSCode", "type": "object", "attributes": {"category": "代码编辑器"}}
  ],
  "preferences": [
    {"subject": "张伟", "category": "beverage", "value": "拿铁咖啡", "sentiment": "positive"}
  ],
  "relations": [
    {"source": "张伟", "target": "李华", "relation": "colleague"},
    {"source": "张伟", "target": "VSCode", "relation": "uses"}
  ]
}

--- 示例 2 ---
输入：
用户今天学习了机器学习的基础知识，对深度学习特别感兴趣。她提到周末喜欢去公园跑步。

输出：
{
  "entities": [
    {"name": "机器学习", "type": "topic", "attributes": {"level": "基础"}},
    {"name": "深度学习", "type": "topic", "attributes": {}},
    {"name": "公园", "type": "location", "attributes": {}}
  ],
  "preferences": [
    {"subject": "用户", "category": "activity", "value": "跑步", "sentiment": "positive"},
    {"subject": "用户", "category": "topic", "value": "深度学习", "sentiment": "positive"}
  ],
  "relations": [
    {"source": "深度学习", "target": "机器学习", "relation": "subfield_of"}
  ]
}

请严格按照以上格式输出 JSON，不要添加任何解释文字。"""

# LLM 返回的实体类型到 Valid Entity 类型的映射
# LLM extract_entities 可能返回 object/location/event/other，
# 这些需要映射到 Entity 模型支持的合法类型
_LLM_TYPE_TO_VALID: dict[str, str] = {
    "person": "person",
    "organization": "organization",
    "topic": "topic",
    "preference": "preference",
    "fact": "fact",
    "object": "fact",
    "location": "fact",
    "event": "fact",
    "other": "fact",
}


def _normalize_entity_type(entity_type: str) -> str:
    """将 LLM 返回的实体类型映射为合法的 Entity 类型。

    若输入类型已在 VALID_ENTITY_TYPES 中，直接返回；
    否则映射到 "fact"。

    Args:
        entity_type: LLM 返回的实体类型字符串。

    Returns:
        合法的 Entity 实体类型字符串。
    """
    normalized = entity_type.lower().strip()
    if normalized in VALID_ENTITY_TYPES:
        return normalized
    return _LLM_TYPE_TO_VALID.get(normalized, "fact")


class MemoryManager:
    """总调度器 —— 统一记忆管理入口。

    负责装配所有内部组件（工作记忆、情节记忆、语义记忆、嵌入模型、LLM 客户端），
    并提供统一的记忆写入、检索、整合和清理接口。

    注意：
    - ChromaDB 的 PersistentClient 不支持多进程并发写入同一持久化目录。
    - 构造函数承担"依赖装配器"角色，创建所有内部组件。

    Attributes:
        config: 当前使用的 MemoryConfig 实例。
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """初始化 MemoryManager 并装配所有内部组件。

        若 config 为 None，则从环境变量 / .env 文件加载默认 MemoryConfig()。

        Args:
            config: 可选的 MemoryConfig 实例，为 None 时从环境变量自动加载。

        Raises:
            ConfigError: 配置无效时抛出（延迟校验，仅在需要 LLM 功能时触发）。
        """
        init_start = time.perf_counter()

        # 加载配置
        if config is None:
            config = MemoryConfig()  # 从环境变量 / .env 自动加载
        self.config = config

        # 配置日志
        logging.getLogger(__name__).setLevel(self.config.log_level_int)

        # ----- 装配内部组件 -----
        # 1. 存储后端
        self._working_store = InMemoryStore()
        self._episodic_store = ChromaStore(
            persist_directory=self.config.chroma_persist_dir,
            collection_name="episodic_memory",
            embedding_dimension=512,
        )
        self._semantic_store = ChromaStore(
            persist_directory=self.config.chroma_persist_dir,
            collection_name="semantic_memory",
            embedding_dimension=512,
        )

        # 2. 嵌入模型
        self._embedder = LocalEmbedder(
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
        )

        # 3. LLM 客户端（延迟校验 API key）
        self._llm_client = DeepSeekClient(
            api_key=self.config.deepseek_api_key,
            model=self.config.deepseek_model,
            base_url=self.config.deepseek_base_url,
            timeout=self.config.deepseek_timeout,
            max_retries=self.config.deepseek_max_retries,
        )

        # 4. 工作记忆
        self._working_memory = WorkingMemory(
            store=self._working_store,
            default_ttl_seconds=self.config.default_ttl_seconds,
        )

        # 5. 情节记忆
        self._episodic_memory = EpisodicMemory(
            chroma_store=self._episodic_store,
            embedder=self._embedder,
            llm_client=self._llm_client,
            summary_threshold=self.config.summary_threshold,
        )

        # 6. 语义记忆
        self._semantic_memory = SemanticMemory(
            chroma_store=self._semantic_store,
            embedder=self._embedder,
            llm_client=self._llm_client,
        )

        init_elapsed = time.perf_counter() - init_start
        logger.info(
            "MemoryManager 初始化完成，耗时=%.3fs, chroma_dir=%s",
            init_elapsed,
            self.config.chroma_persist_dir,
        )

    # ------------------------------------------------------------------
    # 公开方法: 写入
    # ------------------------------------------------------------------

    async def remember(
        self,
        content: str,
        memory_type: Literal["working", "episodic", "semantic"] = "episodic",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> MemoryItem:
        """写入一条记忆。

        根据 memory_type 将内容路由到对应的记忆子系统：

        - ``"working"`` → 写入工作记忆（需提供 session_id）。
        - ``"episodic"`` → 写入情节记忆。
        - ``"semantic"`` → 调用 LLM 提取实体后写入语义记忆。

        Args:
            content: 记忆内容文本。
            memory_type: 记忆类型，可选 "working" / "episodic" / "semantic"。
            metadata: 可选的附加元数据字典。
            session_id: 可选的会话标识符（working 类型时为必填）。

        Returns:
            创建的 MemoryItem 或包装为 MemoryItem 的 Entity。

        Raises:
            ValueError: content 超长或 memory_type="working" 时 session_id 为空。
            StorageError: 底层存储操作失败。
            LLMServiceError / LLMResponseParseError: LLM 调用失败。
        """
        op_start = time.perf_counter()

        # 校验 content 长度
        if len(content) > self.config.max_content_length:
            raise ValueError(
                f"content 长度超过限制: {len(content)} > {self.config.max_content_length}"
            )

        # 校验 memory_type
        if memory_type not in ("working", "episodic", "semantic"):
            raise ValueError(
                f"不支持的 memory_type: {memory_type}，"
                f"可选值: working, episodic, semantic"
            )

        # working 类型需 session_id
        if memory_type == "working" and not session_id:
            raise ValueError("memory_type='working' 时 session_id 不能为空")

        try:
            if memory_type == "working":
                result: MemoryItem = await self._working_memory.add(
                    content=content,
                    session_id=session_id,  # type: ignore[arg-type] # 已校验不为空
                    metadata=metadata,
                )
            elif memory_type == "episodic":
                episode = await self._episodic_memory.add_episode(
                    content=content,
                    metadata=metadata,
                    session_id=session_id,
                )
                result = episode  # Episode 继承自 MemoryItem
            else:  # semantic
                # 调用 LLM 提取实体
                extracted = await self._llm_client.extract_entities(content)
                entities_list = extracted.get("entities", [])
                preferences_list = extracted.get("preferences", [])
                relations_list = extracted.get("relations", [])

                last_entity: Entity | None = None

                # 逐条创建实体
                for ent in entities_list:
                    name = ent.get("name", "未知实体")
                    raw_type = ent.get("type", "fact")
                    entity_type = _normalize_entity_type(raw_type)
                    attributes = ent.get("attributes", {})
                    if not isinstance(attributes, dict):
                        attributes = {}
                    attrs_text = json.dumps(attributes, ensure_ascii=False) if attributes else "无详细属性"
                    description = f"{name}: {attrs_text}"

                    last_entity = await self._semantic_memory.add_entity(
                        name=name,
                        entity_type=entity_type,
                        description=description,
                        attributes=attributes,
                    )

                # 逐条创建偏好实体
                for pref in preferences_list:
                    category = pref.get("category", "unknown")
                    value = pref.get("value", "")
                    subject = pref.get("subject", "用户")
                    sentiment = pref.get("sentiment", "neutral")
                    name = f"{subject}偏好:{category}"
                    description = f"{subject} 偏好 {category}: {value}（情感: {sentiment}）"
                    pref_attributes: dict[str, Any] = {
                        "subject": subject,
                        "category": category,
                        "value": value,
                        "sentiment": sentiment,
                    }

                    last_entity = await self._semantic_memory.add_entity(
                        name=name,
                        entity_type="preference",
                        description=description,
                        attributes=pref_attributes,
                    )

                # 建立关系
                for rel in relations_list:
                    source_name = rel.get("source", "")
                    target_name = rel.get("target", "")
                    relation_type = rel.get("relation", "related_to")

                    # 按名称查找源和目标实体
                    source_entity = await self._semantic_memory._find_entity_by_name(source_name)
                    target_entity = await self._semantic_memory._find_entity_by_name(target_name)

                    if source_entity and target_entity:
                        await self._semantic_memory.add_relation(
                            source_id=source_entity.id,
                            target_id=target_entity.id,
                            relation_type=relation_type,
                        )
                    else:
                        logger.debug(
                            "关系建立跳过: source=%s found=%s, target=%s found=%s",
                            source_name,
                            source_entity is not None,
                            target_name,
                            target_entity is not None,
                        )

                # 返回最后一个创建的实体，若无则返回一个空的 MemoryItem
                if last_entity is not None:
                    result = self._entity_to_memory_item(last_entity)
                else:
                    result = MemoryItem(
                        content=content,
                        memory_type=MemoryType.semantic,
                        metadata=metadata or {},
                        session_id=session_id,
                    )

            op_elapsed = time.perf_counter() - op_start
            logger.debug(
                "remember type=%s 耗时=%.3fs",
                memory_type,
                op_elapsed,
            )
            return result

        except (ValueError, StorageError):
            # 已知异常直接透传
            raise
        except Exception as exc:
            logger.exception("remember 失败 type=%s: %s", memory_type, exc)
            raise StorageError(
                f"remember 操作失败: {exc}",
                details={
                    "memory_type": memory_type,
                    "content_length": len(content),
                    "original_error": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # 公开方法: 检索
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        memory_type: Literal["working", "episodic", "semantic"] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        min_similarity: float = 0.5,
    ) -> list[MemoryItem]:
        """跨类型记忆检索。

        若 ``memory_type`` 为 ``None``（检索所有类型），则并行检索 working、
        episodic 和 semantic 三种记忆，聚合后按以下规则排序：
        - 工作记忆关键词匹配结果排在向量检索结果之前（更高优先级）。
        - 向量检索结果（episodic + semantic）按相似度降序排列。
        - 最终截取 top_k 条。

        若指定 ``memory_type``，则仅检索对应类型。

        Args:
            query: 搜索查询文本。
            memory_type: 可选的记忆类型过滤，None 表示检索全部类型。
            top_k: 返回的最大结果数，默认 10。
            filters: 可选的过滤条件字典（仅对 episodic 有效）。
            session_id: 可选的会话标识符（working 类型时限定会话）。
            min_similarity: 最小相似度阈值，默认 0.5。

        Returns:
            MemoryItem 列表，按优先级降序排列。若指定了相似度阈值，
            低于该阈值的结果将被过滤。
        """
        if top_k <= 0:
            return []

        if memory_type is not None and memory_type not in ("working", "episodic", "semantic"):
            raise ValueError(
                f"不支持的 memory_type: {memory_type}，"
                f"可选值: working, episodic, semantic"
            )

        try:
            if memory_type is not None:
                # 单类型检索
                return await self._recall_single(
                    query=query,
                    memory_type=memory_type,
                    top_k=top_k,
                    filters=filters,
                    session_id=session_id,
                    min_similarity=min_similarity,
                )
            else:
                # 跨类型并行检索
                return await self._recall_all(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    session_id=session_id,
                    min_similarity=min_similarity,
                )
        except StorageError:
            raise
        except Exception as exc:
            logger.exception("recall 失败: %s", exc)
            raise StorageError(
                f"recall 操作失败: {exc}",
                details={
                    "query": query,
                    "memory_type": str(memory_type),
                    "original_error": str(exc),
                },
            ) from exc

    async def _recall_single(
        self,
        query: str,
        memory_type: Literal["working", "episodic", "semantic"],
        top_k: int,
        filters: dict[str, Any] | None,
        session_id: str | None,
        min_similarity: float,
    ) -> list[MemoryItem]:
        """单类型检索实现。"""
        if memory_type == "working":
            results = await self._working_memory.search(
                query=query,
                session_id=session_id,
                top_k=top_k,
            )
            # 工作记忆结果已经是 MemoryItem 列表
            return results

        elif memory_type == "episodic":
            episodes = await self._episodic_memory.search(
                query=query,
                top_k=top_k,
                filters=filters,
                min_similarity=min_similarity,
            )
            return list(episodes)  # Episode 即 MemoryItem

        else:  # semantic
            entities = await self._semantic_memory.search_entities(
                query=query,
                top_k=top_k,
                entity_type=None,
                min_similarity=min_similarity,
            )
            return [self._entity_to_memory_item(e) for e in entities]

    async def _recall_all(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        session_id: str | None,
        min_similarity: float,
    ) -> list[MemoryItem]:
        """跨类型并行检索实现。

        并行发起三种检索，使用 asyncio.gather(return_exceptions=True)
        防止单个模块失败导致整体检索中断。
        """
        # 并行检索任务
        tasks = []

        # 工作记忆：仅当 session_id 不为空时检索
        if session_id:
            tasks.append(
                asyncio.ensure_future(
                    self._working_memory.search(
                        query=query,
                        session_id=session_id,
                        top_k=top_k,
                    )
                )
            )
        else:
            # 无 session_id 时添加一个返回空列表的已完成 future
            tasks.append(asyncio.ensure_future(asyncio.sleep(0, result=[])))  # type: ignore[arg-type]

        # 情节记忆
        tasks.append(
            asyncio.ensure_future(
                self._episodic_memory.search(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    min_similarity=min_similarity,
                )
            )
        )

        # 语义记忆
        tasks.append(
            asyncio.ensure_future(
                self._semantic_memory.search_entities(
                    query=query,
                    top_k=top_k,
                    entity_type=None,
                    min_similarity=min_similarity,
                )
            )
        )

        # 并行执行，return_exceptions=True 防止单个失败导致全部中断
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 解析结果
        working_results: list[MemoryItem] = []
        episodic_results: list[MemoryItem] = []
        semantic_results: list[MemoryItem] = []

        if isinstance(results[0], Exception):
            logger.warning("工作记忆检索失败: %s", results[0])
        else:
            working_results = results[0]  # type: ignore[assignment]

        if isinstance(results[1], Exception):
            logger.warning("情节记忆检索失败: %s", results[1])
        else:
            episodic_results = list(results[1])  # type: ignore[union-attr]

        if isinstance(results[2], Exception):
            logger.warning("语义记忆检索失败: %s", results[2])
        else:
            semantic_results = [self._entity_to_memory_item(e) for e in results[2]]  # type: ignore[union-attr]

        # 聚合排序：
        # 1. 工作记忆排在前面（更高优先级）
        # 2. 情节记忆其次（有向量相似度）
        # 3. 语义记忆最后
        merged = working_results + episodic_results + semantic_results

        # 截取 top_k 条
        return merged[:top_k]

    # ------------------------------------------------------------------
    # 公开方法: 记忆整合
    # ------------------------------------------------------------------

    async def consolidate(
        self,
        session_id: str | None = None,
        time_window_hours: int = 24,
        dry_run: bool = False,
    ) -> ConsolidateResult:
        """从近期情节记忆中提取知识，整合到语义记忆。

        执行步骤：
        1. 获取近期（默认 24 小时内）的情节记忆。
        2. 将多条情节记忆的内容拼接为整合上下文。
        3. 调用 LLM 提取实体、偏好和关系（使用 few-shot 提示词）。
        4. 对于提取的每个实体/偏好，在语义记忆中按名称查找已有实体：
           - 存在则更新合并，计数 updated_entities / updated_preferences。
           - 不存在则新建，计数 new_entities / new_preferences。
        5. 对于每个关系，解析实体名称后在语义记忆中建立关联，
           计数 new_relations。
        6. 若 dry_run=True，只生成预览信息，不实际写入语义记忆。

        Args:
            session_id: 可选，限定处理特定会话的情节记忆。
            time_window_hours: 时间窗口（小时），默认 24。
            dry_run: 是否仅预览不写入，默认 False。

        Returns:
            ConsolidateResult 包含新增/更新的实体数、偏好数、关系数和错误信息。
        """
        logger.info(
            "开始记忆整合 time_window=%dh session=%s dry_run=%s",
            time_window_hours,
            session_id,
            dry_run,
        )

        result = ConsolidateResult(dry_run=dry_run)

        try:
            # 1. 获取近期情节记忆
            episodes = await self._episodic_memory.get_recent(
                hours=time_window_hours,
                session_id=session_id,
            )
            result.episodes_processed = len(episodes)

            if not episodes:
                logger.info("没有找到近期情节记忆，跳过整合")
                return result

            # 2. 构建整合上下文：使用 content 或 summary
            context_parts: list[str] = []
            for ep in episodes:
                text = ep.summary if ep.summary else ep.content
                context_parts.append(f"[{ep.created_at.isoformat()}] {text}")

            combined_text = "\n---\n".join(context_parts)
            logger.debug("整合上下文长度=%d，包含 %d 条情节", len(combined_text), len(episodes))

            # 3. 使用 few-shot 提示词调用 LLM 提取结构化信息
            extracted = await self._consolidate_extract(combined_text)

            entities_list: list[dict[str, Any]] = extracted.get("entities", [])
            preferences_list: list[dict[str, Any]] = extracted.get("preferences", [])
            relations_list: list[dict[str, Any]] = extracted.get("relations", [])

            # 4. 处理实体
            for ent in entities_list:
                try:
                    name = ent.get("name", "未知实体")
                    raw_type = ent.get("type", "fact")
                    entity_type = _normalize_entity_type(raw_type)
                    attributes = ent.get("attributes", {})
                    if not isinstance(attributes, dict):
                        attributes = {}
                    attrs_text = json.dumps(attributes, ensure_ascii=False) if attributes else "无详细属性"
                    description = f"{name}: {attrs_text}"

                    if dry_run:
                        # dry_run 模式：检查是否存在同名实体，仅计数
                        existing = await self._semantic_memory._find_entity_by_name(name)
                        if existing is not None:
                            result.updated_entities += 1
                        else:
                            result.new_entities += 1
                    else:
                        # 实际写入：add_entity 内部已做 upsert（按名称查找并合并）
                        # 然而 add_entity 返回的 Entity 是合并后的，不会告知是新建还是更新。
                        # 因此需要先手动查找来判断。
                        existing = await self._semantic_memory._find_entity_by_name(name)
                        await self._semantic_memory.add_entity(
                            name=name,
                            entity_type=entity_type,
                            description=description,
                            attributes=attributes,
                        )
                        if existing is not None:
                            result.updated_entities += 1
                        else:
                            result.new_entities += 1
                except Exception as entity_exc:
                    error_msg = f"处理实体 '{ent.get('name', '未知')}' 失败: {entity_exc}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

            # 5. 处理偏好
            for pref in preferences_list:
                try:
                    category = pref.get("category", "unknown")
                    value = pref.get("value", "")
                    subject = pref.get("subject", "用户")
                    sentiment = pref.get("sentiment", "neutral")
                    name = f"{subject}偏好:{category}"
                    description = f"{subject} 偏好 {category}: {value}（情感: {sentiment}）"
                    pref_attributes: dict[str, Any] = {
                        "subject": subject,
                        "category": category,
                        "value": value,
                        "sentiment": sentiment,
                    }

                    if dry_run:
                        existing = await self._semantic_memory._find_entity_by_name(name)
                        if existing is not None:
                            result.updated_preferences += 1
                        else:
                            result.new_preferences += 1
                    else:
                        existing = await self._semantic_memory._find_entity_by_name(name)
                        await self._semantic_memory.add_entity(
                            name=name,
                            entity_type="preference",
                            description=description,
                            attributes=pref_attributes,
                        )
                        if existing is not None:
                            result.updated_preferences += 1
                        else:
                            result.new_preferences += 1
                except Exception as pref_exc:
                    error_msg = f"处理偏好 '{pref.get('category', '未知')}' 失败: {pref_exc}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

            # 6. 处理关系
            for rel in relations_list:
                try:
                    source_name = rel.get("source", "")
                    target_name = rel.get("target", "")
                    relation_type = rel.get("relation", "related_to")

                    if not source_name or not target_name:
                        continue

                    if dry_run:
                        source_entity = await self._semantic_memory._find_entity_by_name(source_name)
                        target_entity = await self._semantic_memory._find_entity_by_name(target_name)
                        if source_entity and target_entity:
                            result.new_relations += 1
                    else:
                        source_entity = await self._semantic_memory._find_entity_by_name(source_name)
                        target_entity = await self._semantic_memory._find_entity_by_name(target_name)
                        if source_entity and target_entity:
                            await self._semantic_memory.add_relation(
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                relation_type=relation_type,
                            )
                            result.new_relations += 1
                        else:
                            logger.debug(
                                "关系跳过: source=%s found=%s, target=%s found=%s",
                                source_name,
                                source_entity is not None,
                                target_name,
                                target_entity is not None,
                            )
                except Exception as rel_exc:
                    error_msg = (
                        f"处理关系 '{rel.get('source', '?')}' -> "
                        f"'{rel.get('target', '?')}' 失败: {rel_exc}"
                    )
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

            logger.info(
                "记忆整合完成: new_entities=%d, updated_entities=%d, "
                "new_preferences=%d, updated_preferences=%d, "
                "new_relations=%d, episodes=%d, errors=%d",
                result.new_entities,
                result.updated_entities,
                result.new_preferences,
                result.updated_preferences,
                result.new_relations,
                result.episodes_processed,
                len(result.errors),
            )
            return result

        except Exception as exc:
            logger.exception("consolidate 失败: %s", exc)
            result.errors.append(f"整合过程异常: {exc}")
            return result

    async def _consolidate_extract(self, combined_text: str) -> dict[str, Any]:
        """使用 few-shot 提示词调用 LLM 提取结构化信息。

        Args:
            combined_text: 多条情节记忆拼接后的文本。

        Returns:
            包含 entities、preferences、relations 的字典。

        Raises:
            LLMServiceError: API 调用失败。
            LLMResponseParseError: 返回无法解析的 JSON。
        """
        user_message = f"请从以下对话片段中提取实体、偏好和关系：\n\n{combined_text}"

        raw_response = await self._llm_client.chat(
            messages=[
                {"role": "system", "content": _CONSOLIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=4096,
        )

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(
                "整合提取返回的不是合法 JSON",
                details={"original_response": raw_response, "parse_error": str(e)},
            ) from e

        # 确保返回必需的键
        return {
            "entities": parsed.get("entities", []),
            "preferences": parsed.get("preferences", []),
            "relations": parsed.get("relations", []),
        }

    # ------------------------------------------------------------------
    # 公开方法: 删除与会话清理
    # ------------------------------------------------------------------

    async def forget(
        self,
        memory_id: str,
        memory_type: Literal["working", "episodic", "semantic"],
    ) -> bool:
        """删除指定记忆条目。

        根据 memory_type 将删除请求路由到对应的记忆子系统。

        Args:
            memory_id: 要删除的记忆条目 ID。
            memory_type: 记忆类型。

        Returns:
            True 表示成功删除，False 表示条目不存在。

        Raises:
            ValueError: memory_type 不合法。
            StorageError: 底层存储操作失败。
        """
        if memory_type not in ("working", "episodic", "semantic"):
            raise ValueError(
                f"不支持的 memory_type: {memory_type}，"
                f"可选值: working, episodic, semantic"
            )

        try:
            if memory_type == "working":
                return await self._working_memory.remove(memory_id)
            elif memory_type == "episodic":
                return await self._episodic_memory.remove(memory_id)
            else:  # semantic
                return await self._semantic_memory.remove_entity(memory_id)
        except Exception as exc:
            logger.exception("forget 失败 id=%s type=%s: %s", memory_id, memory_type, exc)
            raise StorageError(
                f"forget 操作失败: {exc}",
                details={
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "original_error": str(exc),
                },
            ) from exc

    async def clear_session(self, session_id: str) -> int:
        """清空指定会话的所有工作记忆。

        调用 WorkingMemory.expire_session() 将所有与该会话关联的
        工作记忆条目标记为立即过期。

        Args:
            session_id: 要清空的会话标识符。

        Returns:
            清除的记忆条数。

        Raises:
            StorageError: 底层存储操作失败。
        """
        try:
            return await self._working_memory.expire_session(session_id)
        except Exception as exc:
            logger.exception("clear_session 失败 session=%s: %s", session_id, exc)
            raise StorageError(
                f"clear_session 操作失败: {exc}",
                details={
                    "session_id": session_id,
                    "original_error": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_to_memory_item(entity: Entity) -> MemoryItem:
        """将 Entity 转换为 MemoryItem（用于跨类型检索结果统一）。

        使用 entity.description 作为 content，
        将 entity.name、entity_type、attributes 等存入 metadata。

        Args:
            entity: 语义实体对象。

        Returns:
            包装后的 MemoryItem 实例。
        """
        return MemoryItem(
            id=entity.id,
            content=entity.description or f"{entity.name}: {entity.entity_type}",
            memory_type=MemoryType.semantic,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            metadata={
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "attributes": entity.attributes,
                "related_entities": entity.related_entities,
                "confidence": entity.confidence,
            },
        )
