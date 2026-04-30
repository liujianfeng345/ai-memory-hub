"""
工作记忆（WorkingMemory）模块。

WorkingMemory 管理当前会话中的临时信息，提供基于关键词匹配的检索、
TTL 自动过期和会话索引维护能力。内部使用 InMemoryStore 作为存储后端，
并通过 asyncio.Lock 保证协程并发安全。

注意事项：
- 虽然 InMemoryStore 已使用 threading.Lock 保证线程安全，
  但在 asyncio 协程并发场景下，多个协程可能交错执行导致会话索引
  出现竞态条件（先 get 再 set 的非原子操作）。因此本模块额外使用
  asyncio.Lock 保护涉及"读取-修改-写入"的复合操作。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from memory_agent.models.memory_item import MemoryItem, MemoryType

logger = logging.getLogger(__name__)

# 中文标点符号集合，用于分词
_CJK_PUNCTUATION = {
    "，",
    "。",
    "！",
    "？",
    "；",
    "：",
    "、",
    "“",
    "”",
    "‘",
    "’",
    "（",
    "）",
    "【",
    "】",
    "《",
    "》",
    "—",
    "…",
    "～",
}
# 英文标点符号
_EN_PUNCTUATION = {",", ".", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}"}


def _is_cjk(ch: str) -> bool:
    """判断字符是否为 CJK 统一表意文字。

    覆盖范围包括基本汉字（U+4E00-U+9FFF）和扩展区部分字符。

    Args:
        ch: 单个字符。

    Returns:
        True 表示该字符是 CJK 汉字。
    """
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF  # CJK 统一表意文字
        or 0x3400 <= cp <= 0x4DBF  # CJK 扩展 A
        or 0x20000 <= cp <= 0x2A6DF  # CJK 扩展 B
    )


def _tokenize(text: str) -> set[str]:
    """对文本进行简单分词，返回小写的词集合。

    分词策略：
    1. 将文本转为小写。
    2. 将中英文标点替换为空格。
    3. 按空白字符切分为片段。
    4. 对每个片段：
       - 若片段含 CJK 字符（纯中文段落），按字符切分为单字（作为一元词），
         同时生成相邻二元组（bigram），全部加入词集合。
       - 若片段为 ASCII 字母/数字/下划线，直接作为词加入。
    5. 过滤掉空字符串。

    这种混合策略使得：
    - 英文"Python programming"按空格分词为{"python", "programming"}。
    - 中文"用户喜欢喝咖啡"切分为单字 + bigram：
      {"用", "户", "喜", "欢", "喝", "咖", "啡",
       "用户", "户喜", "喜欢", "欢喝", "喝咖", "咖啡"}
    - 搜索"咖啡"时匹配到"咖啡"的 bigram，提升 Jaccard 相似度。

    Args:
        text: 待分词的文本。

    Returns:
        分词后的词集合（小写）。
    """
    text = text.lower()
    # 将标点替换为空格
    for punct in _CJK_PUNCTUATION | _EN_PUNCTUATION:
        text = text.replace(punct, " ")
    # 按空白切分为片段
    segments = text.split()

    tokens: set[str] = set()
    for seg in segments:
        if not seg:
            continue
        # 判断是否为纯 CJK 片段
        if any(_is_cjk(ch) for ch in seg):
            # CJK 片段：添加单字符 token
            for ch in seg:
                if ch.strip():
                    tokens.add(ch)
            # 添加相邻 bigram
            for i in range(len(seg) - 1):
                bigram = seg[i : i + 2]
                if not any(ch.isspace() for ch in bigram):
                    tokens.add(bigram)
        else:
            # ASCII 片段：直接作为词
            tokens.add(seg)

    # 过滤空字符串
    return {t for t in tokens if t}


def _jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """计算两个词集合的 Jaccard 相似度。

    Jaccard = |A ∩ B| / |A ∪ B|
    若两个集合均为空，返回 1.0。若仅一个为空，返回 0.0。

    Args:
        set1: 第一个词集合。
        set2: 第二个词集合。

    Returns:
        Jaccard 相似度，范围 [0.0, 1.0]。
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


class WorkingMemory:
    """工作记忆 —— 当前会话中的临时信息管理器。

    使用 InMemoryStore 作为存储后端，支持 TTL 自动过期、
    会话索引维护和关键词检索。

    Attributes:
        store: 注入的 InMemoryStore 实例。
        default_ttl_seconds: 默认的条目存活时间（秒）。
    """

    def __init__(
        self,
        store: "InMemoryStore",  # type: ignore[name-defined] # noqa: F821
        default_ttl_seconds: int = 3600,
    ) -> None:
        """初始化 WorkingMemory 实例。

        Args:
            store: InMemoryStore 实例（依赖注入）。
            default_ttl_seconds: 默认的条目存活时间（秒），必须 > 0。
        """
        self.store = store
        self.default_ttl_seconds = default_ttl_seconds

        # asyncio 锁，保护涉及"读取-修改-写入"的复合操作
        self._session_lock = asyncio.Lock()

        logger.info(
            "WorkingMemory 已初始化 default_ttl=%ds",
            self.default_ttl_seconds,
        )

    async def add(
        self,
        content: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> MemoryItem:
        """添加一条工作记忆条目。

        Args:
            content: 记忆内容文本。
            session_id: 关联的会话标识符，不能为空字符串。
            metadata: 可选的附加元数据字典。
            ttl_seconds: 该条目的 TTL（秒），为 None 时使用 default_ttl_seconds。

        Returns:
            创建的 MemoryItem 对象。

        Raises:
            ValueError: 当 session_id 为空字符串时抛出。
        """
        if not session_id:
            raise ValueError("session_id 不能为空字符串")

        # 确定 TTL
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        # 构造 MemoryItem
        now = datetime.now(tz=timezone.utc)
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.working,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        memory_key = f"wm:{item.id}"
        session_key = f"wm:session:{session_id}"

        # 将 MemoryItem 序列化为 dict 存入 store
        item_dict = item.model_dump()

        # 使用 asyncio.Lock 保护会话索引的"读取-修改-写入"操作
        async with self._session_lock:
            # 存入记忆条目
            self.store.set(memory_key, item_dict, ttl=effective_ttl)

            # 更新会话索引：追加新记忆 ID
            existing_ids: list[str] = self.store.get(session_key)
            if existing_ids is None:
                existing_ids = []
            # 防御：从 store 取回的值可能不是 list（数据损坏等情况）
            if not isinstance(existing_ids, list):
                existing_ids = []
            existing_ids.append(item.id)
            self.store.set(session_key, existing_ids)

        logger.debug(
            "添加工作记忆 id=%s session=%s ttl=%ds",
            item.id,
            session_id,
            effective_ttl,
        )
        return item

    async def get_by_session(
        self,
        session_id: str,
        include_expired: bool = False,
    ) -> list[MemoryItem]:
        """获取指定会话的所有工作记忆条目。

        Args:
            session_id: 会话标识符。
            include_expired: 是否包含已过期的条目，默认 False。

        Returns:
            MemoryItem 列表，按 created_at 降序排列。
        """
        session_key = f"wm:session:{session_id}"
        memory_ids: list[str] = self.store.get(session_key)
        if not memory_ids or not isinstance(memory_ids, list):
            return []

        items: list[MemoryItem] = []
        for mid in memory_ids:
            memory_key = f"wm:{mid}"
            raw = self.store.get(memory_key)
            if raw is None:
                # 条目可能已过期或不存在
                continue
            try:
                item = MemoryItem(**raw)
            except Exception:
                logger.warning("解析 MemoryItem 失败 id=%s", mid)
                continue

            # 若不需要包含过期条目，检查 TTL
            if not include_expired:
                if not self.store.exists(memory_key):
                    # 条目在 get 与 exists 之间过期（懒删除）
                    continue

            items.append(item)

        # 按 created_at 降序排列
        items.sort(key=lambda x: x.created_at, reverse=True)
        return items

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """基于关键词 Jaccard 相似度的搜索。

        对每条记忆的 content 与 query 进行分词，
        计算 Jaccard 相似度，按相似度降序返回 top_k 条。

        Args:
            query: 搜索查询文本。
            session_id: 可选，限定在指定会话范围内搜索。
            top_k: 返回的最大结果数，默认 5。

        Returns:
            按 Jaccard 相似度降序排列的 MemoryItem 列表。
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # 获取候选记忆
        if session_id:
            candidates = await self.get_by_session(session_id, include_expired=False)
        else:
            # 无 session 限制时，从所有 key 中获取记忆
            candidates = []
            all_keys = self.store.keys()
            for key in all_keys:
                if not key.startswith("wm:") or key.startswith("wm:session:"):
                    continue
                raw = self.store.get(key)
                if raw is None:
                    continue
                try:
                    item = MemoryItem(**raw)
                except Exception:
                    continue
                candidates.append(item)

        if not candidates:
            return []

        # 计算每条记忆与 query 的 Jaccard 相似度
        scored: list[tuple[float, MemoryItem]] = []
        for item in candidates:
            content_tokens = _tokenize(item.content)
            score = _jaccard_similarity(query_tokens, content_tokens)
            if score > 0:
                scored.append((score, item))

        # 按相似度降序排列，取 top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def remove(self, memory_id: str) -> bool:
        """删除指定记忆条目。

        同时从对应的会话索引中移除该 ID。

        Args:
            memory_id: 要删除的记忆 ID。

        Returns:
            True 表示成功删除，False 表示条目不存在。
        """
        memory_key = f"wm:{memory_id}"

        # 先获取条目以确定 session_id
        raw = self.store.get(memory_key)
        if raw is None:
            return False

        try:
            item = MemoryItem(**raw)
        except Exception:
            logger.warning("remove 时解析 MemoryItem 失败 id=%s", memory_id)
            return False

        session_id = item.session_id

        # 删除记忆条目
        deleted = self.store.delete(memory_key)
        if not deleted:
            return False

        # 从会话索引中移除该 ID
        if session_id:
            async with self._session_lock:
                session_key = f"wm:session:{session_id}"
                existing_ids = self.store.get(session_key)
                if existing_ids and isinstance(existing_ids, list):
                    new_ids = [mid for mid in existing_ids if mid != memory_id]
                    if new_ids != existing_ids:
                        self.store.set(session_key, new_ids)
                        logger.debug(
                            "从会话索引中移除 id=%s session=%s",
                            memory_id,
                            session_id,
                        )

        logger.debug("删除工作记忆 id=%s", memory_id)
        return True

    async def expire_session(self, session_id: str) -> int:
        """清空整个会话的所有记忆条目。

        将所有记忆条目标记为立即过期，并删除会话索引。

        Args:
            session_id: 要清空的会话标识符。

        Returns:
            过期的记忆条数。
        """
        session_key = f"wm:session:{session_id}"

        async with self._session_lock:
            memory_ids: list[str] = self.store.get(session_key)
            if not memory_ids or not isinstance(memory_ids, list):
                # 即使索引不存在，也尝试删除（可能之前已部分清理）
                self.store.delete(session_key)
                return 0

            count = 0
            for mid in memory_ids:
                memory_key = f"wm:{mid}"
                if self.store.expire_now(memory_key):
                    count += 1

            # 删除会话索引
            self.store.delete(session_key)

        logger.info("会话过期 session=%s 共 %d 条记忆", session_id, count)
        return count
