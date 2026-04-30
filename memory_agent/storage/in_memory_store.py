"""
线程安全的内存键值存储，支持 TTL 过期机制。

InMemoryStore 供工作记忆（WorkingMemory）使用，提供带自动过期的
键值对存储能力。过期采用懒删除策略（get 时检查），配合
cleanup_expired() 支持定时主动清理。
"""

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class EntryExpiredError(Exception):
    """内部使用的哨兵异常，标记条目已过期。"""

    pass


class InMemoryStore:
    """线程安全的内存键值存储。

    内部使用 Dict[str, Dict[str, Any]] 存储数据，每个值字典包含：
    - _value: 实际存储的数据
    - _created_at: 创建时间戳（time.time()）
    - _ttl: TTL 秒数（None 表示永不过期）

    线程安全：所有读/写操作均受 threading.Lock 保护。
    """

    def __init__(self) -> None:
        """初始化空的 InMemoryStore 实例。"""
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        logger.debug("InMemoryStore 已初始化")

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """存入键值对，可选设置 TTL（秒）。

        Args:
            key: 键名。
            value: 要存储的值，可以为任意类型。
            ttl: 生存时间（秒），None 表示永不过期。必须 >= 0。
        """
        with self._lock:
            entry = {
                "_value": value,
                "_created_at": time.time(),
                "_ttl": ttl,
            }
            self._store[key] = entry
            logger.debug("SET key=%s ttl=%s", key, ttl if ttl is not None else "forever")

    def get(self, key: str) -> Any | None:
        """获取键对应的值。

        若 key 不存在或已过期，返回 None。
        过期的条目会被自动删除（懒删除策略）。

        Args:
            key: 键名。

        Returns:
            存储的值，如果键不存在或已过期则返回 None。
        """
        with self._lock:
            try:
                value = self._get_internal(key)
                logger.debug("GET key=%s hit", key)
                return value
            except (KeyError, EntryExpiredError):
                logger.debug("GET key=%s miss", key)
                return None

    def _get_internal(self, key: str) -> Any:
        """内部获取方法，不加锁（调用方必须已持有锁）。

        检查过期逻辑：若已过期则删除条目并抛出 EntryExpiredError。
        若 key 不存在则抛出 KeyError。

        Args:
            key: 键名。

        Returns:
            存储的值。

        Raises:
            KeyError: 键不存在。
            EntryExpiredError: 条目已过期。
        """
        entry = self._store[key]
        ttl = entry["_ttl"]
        if ttl is not None:
            elapsed = time.time() - entry["_created_at"]
            if elapsed >= ttl:
                del self._store[key]
                raise EntryExpiredError(f"键 '{key}' 已过期")
        return entry["_value"]

    def delete(self, key: str) -> bool:
        """删除键。

        Args:
            key: 键名。

        Returns:
            True 表示成功删除，False 表示键不存在。
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug("DELETE key=%s success", key)
                return True
            logger.debug("DELETE key=%s not_found", key)
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在且未过期。

        Args:
            key: 键名。

        Returns:
            True 表示键存在且未过期，False 表示不存在或已过期。
        """
        with self._lock:
            if key not in self._store:
                return False
            entry = self._store[key]
            ttl = entry["_ttl"]
            if ttl is not None:
                elapsed = time.time() - entry["_created_at"]
                if elapsed >= ttl:
                    # 懒删除过期条目
                    del self._store[key]
                    return False
            return True

    def keys(self) -> list[str]:
        """返回所有有效键列表（自动排除已过期的键）。

        Returns:
            当前存储中所有未过期的键名列表。
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            for key, entry in self._store.items():
                ttl = entry["_ttl"]
                if ttl is not None:
                    elapsed = current_time - entry["_created_at"]
                    if elapsed >= ttl:
                        expired_keys.append(key)

            # 清理过期键
            for key in expired_keys:
                del self._store[key]

            if expired_keys:
                logger.debug("KEYS 清理了 %d 个过期条目", len(expired_keys))

            return list(self._store.keys())

    def clear(self) -> int:
        """清空全部数据。

        Returns:
            清除的条目数量。
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.debug("CLEAR 清除了 %d 个条目", count)
            return count

    def expire_now(self, key: str) -> bool:
        """立即过期指定键（设置其 TTL 为 0）。

        若键存在且未过期，则将其 TTL 设为 0 使其立即过期，
        下一次 get 或 exists 检查时将自动删除该条目。

        Args:
            key: 键名。

        Returns:
            True 表示操作成功（键存在且未过期），False 表示键不存在或已过期。
        """
        with self._lock:
            if key not in self._store:
                logger.debug("EXPIRE_NOW key=%s not_found", key)
                return False
            entry = self._store[key]
            ttl = entry["_ttl"]
            if ttl is not None:
                elapsed = time.time() - entry["_created_at"]
                if elapsed >= ttl:
                    # 已过期
                    del self._store[key]
                    logger.debug("EXPIRE_NOW key=%s already_expired", key)
                    return False
                # 将 TTL 设为 0，使条目立即过期
                entry["_ttl"] = 0
                # 同时更新 created_at 以配合新的 TTL=0
                entry["_created_at"] = time.time()
                logger.debug("EXPIRE_NOW key=%s expired_now", key)
                return True
            else:
                # TTL 为 None（永不过期），设为 0 使其立即过期
                entry["_ttl"] = 0
                entry["_created_at"] = time.time()
                logger.debug("EXPIRE_NOW key=%s expired_now (was forever)", key)
                return True

    def cleanup_expired(self) -> int:
        """主动扫描并清理所有过期条目。

        遍历全部存储条目，删除所有已过期的条目。
        供后续定时任务定期调用，防止内存泄漏。

        Returns:
            清理的条目数量。
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            for key, entry in self._store.items():
                ttl = entry["_ttl"]
                if ttl is not None:
                    elapsed = current_time - entry["_created_at"]
                    if elapsed >= ttl:
                        expired_keys.append(key)

            for key in expired_keys:
                del self._store[key]

            count = len(expired_keys)
            if count > 0:
                logger.debug("CLEANUP_EXPIRED 清理了 %d 个过期条目", count)
            return count

    def __len__(self) -> int:
        """返回当前有效条目数量（自动排除过期条目）。

        Returns:
            有效条目数量。
        """
        # 使用 keys() 会触发过期清理，直接返回清理后的数量
        return len(self.keys())
