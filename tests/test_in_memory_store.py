"""
InMemoryStore 单元测试。

验证基本 CRUD 操作、TTL 过期机制、线程安全和清理功能。
"""

import threading
import time

from memory_agent.storage.in_memory_store import InMemoryStore


class TestBasicCRUD:
    """基本 CRUD 操作测试：set / get / delete / exists。"""

    def test_set_and_get_simple_value(self, in_memory_store: InMemoryStore) -> None:
        """测试 set 后 get 返回正确的值。"""
        in_memory_store.set("k1", "v1")
        assert in_memory_store.get("k1") == "v1"

    def test_set_and_get_none_value(self, in_memory_store: InMemoryStore) -> None:
        """测试 value 为 None 也能正确存取。"""
        in_memory_store.set("k_none", None)
        assert in_memory_store.get("k_none") is None
        # 注意：get 返回 None 可能代表值就是 None 或 key 不存在
        # 需要通过 exists 区分
        assert in_memory_store.exists("k_none") is True

    def test_get_nonexistent_key_returns_none(self, in_memory_store: InMemoryStore) -> None:
        """测试获取不存在的 key 返回 None。"""
        assert in_memory_store.get("nonexistent") is None

    def test_set_overwrites_existing_key(self, in_memory_store: InMemoryStore) -> None:
        """测试对同一 key 重复 set 会覆盖旧值。"""
        in_memory_store.set("k1", "old_value")
        in_memory_store.set("k1", "new_value")
        assert in_memory_store.get("k1") == "new_value"

    def test_set_with_complex_value(self, in_memory_store: InMemoryStore) -> None:
        """测试存储复杂对象（dict、list）。"""
        complex_data = {"name": "test", "items": [1, 2, 3], "nested": {"a": True}}
        in_memory_store.set("complex", complex_data)
        assert in_memory_store.get("complex") == complex_data

    def test_delete_existing_key(self, in_memory_store: InMemoryStore) -> None:
        """测试删除存在的 key 返回 True。"""
        in_memory_store.set("k1", "v1")
        assert in_memory_store.delete("k1") is True
        assert in_memory_store.get("k1") is None

    def test_delete_nonexistent_key_returns_false(self, in_memory_store: InMemoryStore) -> None:
        """测试删除不存在的 key 返回 False。"""
        assert in_memory_store.delete("nonexistent") is False

    def test_exists_existing_key(self, in_memory_store: InMemoryStore) -> None:
        """测试已存在的 key exists 返回 True。"""
        in_memory_store.set("k1", "v1")
        assert in_memory_store.exists("k1") is True

    def test_exists_nonexistent_key(self, in_memory_store: InMemoryStore) -> None:
        """测试不存在的 key exists 返回 False。"""
        assert in_memory_store.exists("nonexistent") is False

    def test_keys_returns_all_keys(self, in_memory_store: InMemoryStore) -> None:
        """测试 keys() 返回所有有效键列表。"""
        in_memory_store.set("k1", 1)
        in_memory_store.set("k2", 2)
        in_memory_store.set("k3", 3)
        keys = in_memory_store.keys()
        assert sorted(keys) == ["k1", "k2", "k3"]

    def test_keys_empty_store_returns_empty_list(self, in_memory_store: InMemoryStore) -> None:
        """测试空存储的 keys() 返回空列表。"""
        assert in_memory_store.keys() == []

    def test_clear_returns_count(self, in_memory_store: InMemoryStore) -> None:
        """测试 clear() 返回清除的条目数。"""
        in_memory_store.set("k1", 1)
        in_memory_store.set("k2", 2)
        in_memory_store.set("k3", 3)
        count = in_memory_store.clear()
        assert count == 3
        assert in_memory_store.keys() == []

    def test_clear_empty_store_returns_zero(self, in_memory_store: InMemoryStore) -> None:
        """测试清空空存储返回 0。"""
        assert in_memory_store.clear() == 0


class TestTTL:
    """TTL 过期机制测试。"""

    def test_ttl_expired_returns_none(self, in_memory_store: InMemoryStore) -> None:
        """测试 TTL=1 秒的数据在 1.1 秒后 get 返回 None。"""
        in_memory_store.set("k1", "v1", ttl=1)
        # 刚设置后应该能获取
        assert in_memory_store.get("k1") == "v1"
        # 等待过期
        time.sleep(1.1)
        assert in_memory_store.get("k1") is None

    def test_ttl_expired_exists_returns_false(self, in_memory_store: InMemoryStore) -> None:
        """测试过期后 exists 返回 False。"""
        in_memory_store.set("k1", "v1", ttl=1)
        time.sleep(1.1)
        assert in_memory_store.exists("k1") is False

    def test_no_ttl_never_expires(self, in_memory_store: InMemoryStore) -> None:
        """测试不传 TTL（默认永不过期）的数据在长时间内仍可获取。"""
        in_memory_store.set("k_forever", "persistent")
        time.sleep(1.1)
        assert in_memory_store.get("k_forever") == "persistent"
        assert in_memory_store.exists("k_forever") is True

    def test_ttl_zero_immediately_expired(self, in_memory_store: InMemoryStore) -> None:
        """测试 TTL=0 的数据立即过期。"""
        in_memory_store.set("k0", "v0", ttl=0)
        assert in_memory_store.get("k0") is None
        assert in_memory_store.exists("k0") is False

    def test_get_after_expiry_removes_entry(self, in_memory_store: InMemoryStore) -> None:
        """测试过期条目在 get 时被自动删除（懒删除），不再出现在 keys() 中。"""
        in_memory_store.set("k1", "v1", ttl=0)
        # get 触发懒删除
        in_memory_store.get("k1")
        assert "k1" not in in_memory_store.keys()

    def test_expire_now_sets_ttl_to_zero(self, in_memory_store: InMemoryStore) -> None:
        """测试 expire_now 使未过期的 key 立即过期。"""
        in_memory_store.set("k1", "v1", ttl=9999)
        assert in_memory_store.exists("k1") is True
        result = in_memory_store.expire_now("k1")
        assert result is True
        # expire_now 将 TTL 设为 0，下次 get 应返回 None
        assert in_memory_store.get("k1") is None

    def test_expire_now_on_nonexistent_key(self, in_memory_store: InMemoryStore) -> None:
        """测试对不存在的 key 调用 expire_now 返回 False。"""
        assert in_memory_store.expire_now("nonexistent") is False

    def test_expire_now_on_already_expired_key(self, in_memory_store: InMemoryStore) -> None:
        """测试对已过期的 key 调用 expire_now 返回 False。"""
        in_memory_store.set("k1", "v1", ttl=1)
        time.sleep(1.1)
        assert in_memory_store.expire_now("k1") is False

    def test_expire_now_on_forever_key(self, in_memory_store: InMemoryStore) -> None:
        """测试对永不过期的 key 调用 expire_now 使其过期。"""
        in_memory_store.set("k_forever", "persistent")  # 无 TTL
        result = in_memory_store.expire_now("k_forever")
        assert result is True
        assert in_memory_store.get("k_forever") is None

    def test_keys_excludes_expired_entries(self, in_memory_store: InMemoryStore) -> None:
        """测试 keys() 自动排除已过期的键。"""
        in_memory_store.set("k1", "v1", ttl=1)
        in_memory_store.set("k2", "v2")  # 永不过期
        time.sleep(1.1)
        keys = in_memory_store.keys()
        assert "k1" not in keys
        assert "k2" in keys


class TestCleanup:
    """cleanup_expired 主动清理测试。"""

    def test_cleanup_expired_removes_expired(self, in_memory_store: InMemoryStore) -> None:
        """测试 cleanup_expired 清理过期条目并返回清理数量。"""
        in_memory_store.set("k1", "v1", ttl=0)  # 立即过期
        in_memory_store.set("k2", "v2", ttl=0)  # 立即过期
        in_memory_store.set("k3", "v3")  # 永不过期
        count = in_memory_store.cleanup_expired()
        assert count == 2
        assert in_memory_store.get("k3") == "v3"
        assert len(in_memory_store.keys()) == 1

    def test_cleanup_expired_no_expired_returns_zero(self, in_memory_store: InMemoryStore) -> None:
        """测试无过期条目时 cleanup_expired 返回 0。"""
        in_memory_store.set("k1", "v1")
        in_memory_store.set("k2", "v2")
        assert in_memory_store.cleanup_expired() == 0

    def test_cleanup_expired_empty_store_returns_zero(self, in_memory_store: InMemoryStore) -> None:
        """测试空存储下 cleanup_expired 返回 0。"""
        assert in_memory_store.cleanup_expired() == 0


class TestThreadSafety:
    """线程安全测试。"""

    def test_concurrent_writes_same_key(self, in_memory_store: InMemoryStore) -> None:
        """测试多个线程并发写入同一 key，验证最终一致性。

        10 个线程各执行 100 次对同一 key 的 set 操作，
        预期不会发生数据竞争导致的异常，且最终能获取到值。
        """
        errors = []
        key = "shared_key"
        num_threads = 10
        num_writes = 100

        def writer(thread_id: int) -> None:
            """每个线程写入 num_writes 次。"""
            for i in range(num_writes):
                try:
                    in_memory_store.set(key, f"t{thread_id}_w{i}")
                except Exception as e:
                    errors.append(f"t{thread_id}_w{i}: {e}")

        threads = []
        for t_id in range(num_threads):
            thread = threading.Thread(target=writer, args=(t_id,))
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 不应发生任何异常
        assert len(errors) == 0, f"并发写入发生了 {len(errors)} 个错误: {errors[:5]}"

        # 能正常获取到值
        value = in_memory_store.get(key)
        assert value is not None, "并发写入后应能获取到值"
        assert value.startswith("t"), f"值格式异常: {value}"

    def test_concurrent_set_and_get_no_crash(self, in_memory_store: InMemoryStore) -> None:
        """测试并发 set 和 get 操作不会导致崩溃。

        多个线程同时进行 set 和 get 操作，验证无异常抛出。
        """
        errors = []
        num_threads = 10
        num_ops = 50

        def worker(thread_id: int) -> None:
            for i in range(num_ops):
                try:
                    in_memory_store.set(f"k_t{thread_id}", f"v_{i}")
                    in_memory_store.get(f"k_t{thread_id}")
                    in_memory_store.exists(f"k_t{thread_id}")
                    in_memory_store.keys()
                except Exception as e:
                    errors.append(f"t{thread_id}_op{i}: {e}")

        threads = []
        for t_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(t_id,))
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"并发操作发生了 {len(errors)} 个错误: {errors[:5]}"

    def test_concurrent_delete_and_get(self, in_memory_store: InMemoryStore) -> None:
        """测试并发 delete 和 get 操作不会导致崩溃。"""
        in_memory_store.set("shared", "value")
        errors = []

        def deleter() -> None:
            for _ in range(200):
                try:
                    in_memory_store.delete("shared")
                    in_memory_store.set("shared", "value")
                except Exception as e:
                    errors.append(f"deleter: {e}")

        def getter() -> None:
            for _ in range(200):
                try:
                    in_memory_store.get("shared")
                except Exception as e:
                    errors.append(f"getter: {e}")

        t1 = threading.Thread(target=deleter)
        t2 = threading.Thread(target=getter)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"并发删除/读取发生了 {len(errors)} 个错误"


class TestEdgeCases:
    """边界情况和特殊输入测试。"""

    def test_set_with_empty_string_key(self, in_memory_store: InMemoryStore) -> None:
        """测试空字符串作为 key。"""
        in_memory_store.set("", "empty_key_value")
        assert in_memory_store.get("") == "empty_key_value"

    def test_stores_multiple_types(self, in_memory_store: InMemoryStore) -> None:
        """测试存储多种数据类型。"""
        test_cases = [
            ("int", 42),
            ("float", 3.14),
            ("bool_true", True),
            ("bool_false", False),
            ("str", "hello"),
            ("list", [1, 2, 3]),
            ("dict", {"a": 1}),
            ("tuple", (1, 2)),
        ]
        for key, value in test_cases:
            in_memory_store.set(key, value)

        for key, expected in test_cases:
            assert in_memory_store.get(key) == expected

    def test_set_large_number_of_keys(self, in_memory_store: InMemoryStore) -> None:
        """测试存储大量 key 不会出错。"""
        n = 1000
        for i in range(n):
            in_memory_store.set(f"key_{i}", i)

        assert len(in_memory_store.keys()) == n
        assert in_memory_store.get("key_500") == 500
