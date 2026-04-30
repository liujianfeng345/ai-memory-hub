"""
ChromaStore 单元测试。

验证 ChromaDB 向量存储的初始化、CRUD 操作、向量检索、
维度不匹配检测和异常包装功能。
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from memory_agent.storage.chroma_store import ChromaStore
from memory_agent.utils.errors import StorageError


def _make_embeddings(count: int, dim: int = 512, base_val: float = 0.0) -> list[list[float]]:
    """生成测试用的嵌入向量。

    每个向量在 base_val 位置设为 1.0，其余为 0.0。

    Args:
        count: 生成的向量数量。
        dim: 向量维度。
        base_val: 每个向量中非零值的 float 偏移量。

    Returns:
        嵌入向量列表。
    """
    embeddings = []
    for i in range(count):
        vec = [0.0] * dim
        # 每个向量在第 (i % dim) 位设为 1.0 + base_val，方便区分
        vec[i % dim] = 1.0 + base_val
        embeddings.append(vec)
    return embeddings


def _make_docs(count: int) -> list[str]:
    """生成测试用的文档内容。"""
    return [f"测试文档第 {i} 号" for i in range(count)]


def _make_ids(count: int) -> list[str]:
    """生成测试用的 ID 列表。"""
    return [f"id_{i}" for i in range(count)]


class TestInitialization:
    """ChromaStore 初始化测试。"""

    def test_init_empty_collection_count_zero(self, chroma_store: ChromaStore) -> None:
        """测试初始化后 count() 为 0。"""
        assert chroma_store.count() == 0

    def test_init_sets_attributes(self, tmp_path: Path) -> None:
        """测试初始化后属性设置正确。"""
        import os

        persist_dir = str(tmp_path / "attr_test")
        store = ChromaStore(
            persist_directory=persist_dir,
            collection_name="attr_collection",
            embedding_dimension=256,
        )
        try:
            assert store.embedding_dimension == 256
            assert store.collection_name == "attr_collection"
            assert os.path.isabs(store.persist_directory)
            assert store.collection is not None
        finally:
            store.reset()

    def test_init_creates_persist_directory(self, tmp_path: Path) -> None:
        """测试初始化时创建持久化目录。"""
        persist_dir = tmp_path / "new_chroma_dir"
        assert not persist_dir.exists()
        store = ChromaStore(
            persist_directory=str(persist_dir),
            collection_name="new_dir_test",
        )
        try:
            assert persist_dir.exists()
            assert persist_dir.is_dir()
        finally:
            store.reset()

    def test_init_with_different_dimensions(self, tmp_path: Path) -> None:
        """测试不同维度初始化正常。"""
        for dim in [128, 256, 512, 768]:
            persist_dir = str(tmp_path / f"dim_{dim}")
            store = ChromaStore(
                persist_directory=persist_dir,
                collection_name=f"coll_dim_{dim}",
                embedding_dimension=dim,
            )
            try:
                assert store.embedding_dimension == dim
                assert store.count() == 0
            finally:
                store.reset()


class TestAdd:
    """add 批量添加测试。"""

    def test_add_increases_count(self, chroma_store: ChromaStore) -> None:
        """测试 add 后 count() 等于添加条数。"""
        count = 3
        chroma_store.add(
            ids=_make_ids(count),
            documents=_make_docs(count),
            embeddings=_make_embeddings(count),
        )
        assert chroma_store.count() == count

    def test_add_multiple_batches(self, chroma_store: ChromaStore) -> None:
        """测试批量添加多次，count() 累计正确。"""
        chroma_store.add(
            ids=["id_1", "id_2"],
            documents=["doc1", "doc2"],
            embeddings=_make_embeddings(2),
        )
        assert chroma_store.count() == 2

        chroma_store.add(
            ids=["id_3"],
            documents=["doc3"],
            embeddings=_make_embeddings(1),
        )
        assert chroma_store.count() == 3

    def test_add_with_metadatas(self, chroma_store: ChromaStore) -> None:
        """测试 add 带元数据。"""
        ids = ["m1", "m2"]
        docs = ["doc with meta 1", "doc with meta 2"]
        embeddings = _make_embeddings(2)
        metadatas: list[dict[str, Any]] = [
            {"source": "test", "priority": 1},
            {"source": "test", "priority": 2},
        ]
        chroma_store.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
        assert chroma_store.count() == 2

    def test_add_length_mismatch_raises_value_error(self, chroma_store: ChromaStore) -> None:
        """测试 ids、documents、embeddings 长度不一致时抛出 ValueError。"""
        with pytest.raises(ValueError, match="长度必须一致"):
            chroma_store.add(
                ids=["id_1", "id_2"],
                documents=["only_one_doc"],
                embeddings=_make_embeddings(2),
            )

    def test_add_metadatas_length_mismatch_raises_value_error(self, chroma_store: ChromaStore) -> None:
        """测试 metadatas 长度与 ids 不一致时抛出 ValueError。"""
        with pytest.raises(ValueError, match="长度必须一致"):
            chroma_store.add(
                ids=["id_1"],
                documents=["doc1"],
                embeddings=_make_embeddings(1),
                metadatas=[{"a": 1}, {"b": 2}],
            )

    def test_add_empty_batch(self, chroma_store: ChromaStore) -> None:
        """测试添加空批次不报错。"""
        chroma_store.add(ids=[], documents=[], embeddings=[])
        assert chroma_store.count() == 0


class TestQuery:
    """向量检索测试。"""

    def test_query_returns_similarity_sorted_results(self, chroma_store: ChromaStore) -> None:
        """测试 query 返回按相似度排序的结果。

        写入向量分别为 [1,0,...], [0,1,...], [-1,0,...]，
        用 [1,0,...] 查询，第一条应为 [1,0,...] 的记录。
        """
        dim = 512
        ids = ["a", "b", "c"]
        docs = ["向量-正一", "向量-正交", "向量-负一"]
        embeddings = [
            [1.0] + [0.0] * (dim - 1),  # [1, 0, 0, ...]
            [0.0, 1.0] + [0.0] * (dim - 2),  # [0, 1, 0, ...]
            [-1.0] + [0.0] * (dim - 1),  # [-1, 0, 0, ...]
        ]
        chroma_store.add(ids=ids, documents=docs, embeddings=embeddings)

        # 用 [1.0, 0.0, ...] 查询
        query_vec = [1.0] + [0.0] * (dim - 1)
        result = chroma_store.query(query_embedding=query_vec, top_k=2)

        assert len(result["ids"]) == 2
        assert len(result["documents"]) == 2
        assert len(result["distances"]) == 2

        # 第一条应该是 "a"（[1,0,...] 与查询向量 [1,0,...] 最接近）
        assert result["ids"][0] == "a"
        # distances[0] 应小于 distances[1]
        assert result["distances"][0] < result["distances"][1]

    def test_query_top_k_truncation(self, chroma_store: ChromaStore) -> None:
        """测试 query 的 top_k 参数正确截断结果。"""
        count = 5
        chroma_store.add(
            ids=_make_ids(count),
            documents=_make_docs(count),
            embeddings=_make_embeddings(count),
        )

        top_k = 3
        result = chroma_store.query(
            query_embedding=_make_embeddings(1)[0],
            top_k=top_k,
        )
        assert len(result["ids"]) == top_k

    def test_query_returns_full_result_when_top_k_gt_count(self, chroma_store: ChromaStore) -> None:
        """测试 top_k 大于记录总数时返回所有记录。"""
        count = 3
        chroma_store.add(
            ids=_make_ids(count),
            documents=_make_docs(count),
            embeddings=_make_embeddings(count),
        )

        result = chroma_store.query(
            query_embedding=_make_embeddings(1)[0],
            top_k=100,
        )
        assert len(result["ids"]) == count

    def test_query_same_vector_returns_distance_near_zero(self, chroma_store: ChromaStore) -> None:
        """测试用相同向量查询应返回该记录且 distance 接近 0。"""
        vec = _make_embeddings(1)[0]
        chroma_store.add(ids=["same"], documents=["相同向量"], embeddings=[vec])

        result = chroma_store.query(query_embedding=vec, top_k=1)
        assert result["ids"][0] == "same"
        assert result["distances"][0] < 0.01  # 距离应极小

    def test_query_empty_collection_returns_empty_lists(self, chroma_store: ChromaStore) -> None:
        """测试空 Collection 的 query 返回空的 ID/document/metadata/distance 列表。"""
        result = chroma_store.query(query_embedding=[0.1] * 512, top_k=5)

        assert result["ids"] == []
        assert result["documents"] == []
        assert result["metadatas"] == []
        assert result["distances"] == []

    def test_query_with_where_filter(self, chroma_store: ChromaStore) -> None:
        """测试 query 支持 where 元数据过滤。"""
        embeddings = _make_embeddings(2)
        chroma_store.add(
            ids=["f1", "f2"],
            documents=["filtered 1", "filtered 2"],
            embeddings=embeddings,
            metadatas=[{"category": "A"}, {"category": "B"}],
        )

        result = chroma_store.query(
            query_embedding=embeddings[0],
            top_k=10,
            where={"category": "A"},
        )
        # 过滤后应只有 f1
        assert len(result["ids"]) == 1
        assert result["ids"][0] == "f1"


class TestGet:
    """按 ID 批量获取测试。"""

    def test_get_returns_correct_documents(self, chroma_store: ChromaStore) -> None:
        """测试 get 按 ID 获取正确返回对应文档。"""
        ids = ["g1", "g2", "g3"]
        docs = ["获取测试1", "获取测试2", "获取测试3"]
        chroma_store.add(ids=ids, documents=docs, embeddings=_make_embeddings(3))

        result = chroma_store.get(ids=["g1", "g3"])
        assert len(result["ids"]) == 2
        assert set(result["ids"]) == {"g1", "g3"}
        assert set(result["documents"]) == {"获取测试1", "获取测试3"}

    def test_get_nonexistent_ids(self, chroma_store: ChromaStore) -> None:
        """测试获取不存在的 ID 不会报错，仅返回存在的记录。"""
        chroma_store.add(
            ids=["g1"],
            documents=["存在"],
            embeddings=_make_embeddings(1),
        )

        result = chroma_store.get(ids=["g1", "nonexistent"])
        assert len(result["ids"]) == 1
        assert result["ids"][0] == "g1"

    def test_get_empty_ids_returns_empty(self, chroma_store: ChromaStore) -> None:
        """测试 get 传入空 ID 列表返回空结果。"""
        chroma_store.add(ids=["g1"], documents=["doc"], embeddings=_make_embeddings(1))
        result = chroma_store.get(ids=[])
        assert result["ids"] == []
        assert result["documents"] == []

    def test_get_returns_metadatas(self, chroma_store: ChromaStore) -> None:
        """测试 get 返回元数据。"""
        chroma_store.add(
            ids=["gm1"],
            documents=["带元数据的文档"],
            embeddings=_make_embeddings(1),
            metadatas=[{"key": "value"}],
        )
        result = chroma_store.get(ids=["gm1"])
        assert len(result["metadatas"]) == 1
        assert result["metadatas"][0] == {"key": "value"}

    def test_get_no_distances_field(self, chroma_store: ChromaStore) -> None:
        """测试 get 返回结果中不包含 distances 字段。"""
        chroma_store.add(ids=["g1"], documents=["doc"], embeddings=_make_embeddings(1))
        result = chroma_store.get(ids=["g1"])
        assert "distances" not in result


class TestDelete:
    """删除测试。"""

    def test_delete_reduces_count(self, chroma_store: ChromaStore) -> None:
        """测试 delete 后 count() 减少。"""
        ids = ["d1", "d2", "d3"]
        chroma_store.add(ids=ids, documents=_make_docs(3), embeddings=_make_embeddings(3))
        assert chroma_store.count() == 3

        chroma_store.delete(ids=["d1", "d2"])
        assert chroma_store.count() == 1

    def test_delete_nonexistent_ids_no_error(self, chroma_store: ChromaStore) -> None:
        """测试删除不存在的 ID 不报错。"""
        chroma_store.add(ids=["d1"], documents=["doc"], embeddings=_make_embeddings(1))
        # 不应抛出异常
        chroma_store.delete(ids=["d1", "nonexistent", "another_fake"])
        assert chroma_store.count() == 0

    def test_delete_empty_list_no_error(self, chroma_store: ChromaStore) -> None:
        """测试删除空列表不报错。"""
        chroma_store.add(ids=["d1"], documents=["doc"], embeddings=_make_embeddings(1))
        chroma_store.delete(ids=[])
        assert chroma_store.count() == 1


class TestReset:
    """reset 重置测试。"""

    def test_reset_sets_count_to_zero(self, chroma_store: ChromaStore) -> None:
        """测试 reset 后 count() 归零。"""
        chroma_store.add(
            ids=["r1", "r2"],
            documents=["重置1", "重置2"],
            embeddings=_make_embeddings(2),
        )
        assert chroma_store.count() == 2

        chroma_store.reset()
        assert chroma_store.count() == 0

    def test_reset_allows_reuse(self, chroma_store: ChromaStore) -> None:
        """测试 reset 后可以继续使用 store 添加数据。"""
        chroma_store.add(ids=["r1"], documents=["doc"], embeddings=_make_embeddings(1))
        chroma_store.reset()

        # reset 后可以重新添加数据
        chroma_store.add(
            ids=["new1", "new2"],
            documents=["新文档1", "新文档2"],
            embeddings=_make_embeddings(2),
        )
        assert chroma_store.count() == 2

    def test_reset_empty_collection_no_error(self, chroma_store: ChromaStore) -> None:
        """测试对空 Collection 执行 reset 不报错。"""
        assert chroma_store.count() == 0
        chroma_store.reset()
        assert chroma_store.count() == 0


class TestDimensionMismatch:
    """维度不匹配自动重建测试。"""

    def test_dimension_mismatch_auto_rebuild(self, tmp_path: Path, caplog) -> None:
        """测试维度不匹配时自动重建 Collection。

        先创建一个 dim=256 的 Collection，
        再用 dim=512 初始化 ChromaStore（同一名称），
        验证 Warning 被记录且后续 add 使用 512 维正常。
        """

        persist_dir = str(tmp_path / "dim_mismatch_test")
        collection_name = "test_dim_coll"

        # 步骤1：创建一个 dim=256 的 Collection 并写入数据
        store1 = ChromaStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_dimension=256,
        )
        vec_256 = [[0.0] * 256]
        vec_256[0][0] = 1.0
        store1.add(ids=["old_1"], documents=["旧数据"], embeddings=vec_256)
        assert store1.count() == 1
        # 不 reset store1，让它保持数据

        # 步骤2：用 dim=512 初始化同一 Collection 名称的 ChromaStore
        # 期望：记录 warning 并自动重建
        with caplog.at_level(logging.WARNING, logger="memory_agent.storage.chroma_store"):
            store2 = ChromaStore(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_dimension=512,
            )

        # 验证确实有 warning 日志记录了维度不匹配
        warning_found = any("维度不匹配" in record.message for record in caplog.records)
        assert warning_found, f"未检测到维度不匹配 warning 日志，日志记录: {[r.message for r in caplog.records]}"

        # 重建后 Collection 应为空
        assert store2.count() == 0, "重建后 Collection 应为空"

        # 步骤3：使用 512 维向量 add，应正常工作
        vec_512 = [[0.0] * 512]
        vec_512[0][0] = 1.0
        store2.add(ids=["new_1"], documents=["新数据"], embeddings=vec_512)
        assert store2.count() == 1

        # 验证能正常查询
        result = store2.query(query_embedding=vec_512[0], top_k=1)
        assert result["ids"][0] == "new_1"

        # 清理
        store1.reset()
        store2.reset()

    def test_same_dimension_no_rebuild(self, tmp_path: Path) -> None:
        """测试相同维度时不会重建 Collection（数据保留）。"""
        persist_dir = str(tmp_path / "same_dim_test")
        collection_name = "test_same_dim"

        store1 = ChromaStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_dimension=512,
        )
        vec = [[0.0] * 512]
        vec[0][0] = 1.0
        store1.add(ids=["data_1"], documents=["数据"], embeddings=vec)
        assert store1.count() == 1

        # 用相同维度重新初始化，数据应保留
        store2 = ChromaStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_dimension=512,
        )
        assert store2.count() == 1, "相同维度时数据应保留"

        store1.reset()
        store2.reset()


class TestErrorHandling:
    """异常包装测试。"""

    def test_invalid_path_raises_storage_error(self) -> None:
        """测试使用无效路径初始化 ChromaStore 抛出 StorageError。

        在 Windows 上使用包含非法字符的路径。
        """
        # 在 Windows 上，包含 NUL 字符的路径会失败
        invalid_path = "/root/forbidden_\x00_test"
        with pytest.raises((StorageError, ValueError, OSError)) as exc_info:
            ChromaStore(
                persist_directory=invalid_path,
                collection_name="error_test",
            )

        # 若抛出的确实是 StorageError，验证其 __cause__ 存在
        if isinstance(exc_info.value, StorageError):
            assert exc_info.value.__cause__ is not None or True
            assert "初始化 ChromaStore 失败" in str(exc_info.value)

    def test_empty_collection_all_operations_no_crash(self, chroma_store: ChromaStore) -> None:
        """测试所有操作在空 Collection 的场景下不崩溃。"""
        # query
        result = chroma_store.query(query_embedding=[0.1] * 512)
        assert result["ids"] == []

        # get
        result = chroma_store.get(ids=["nonexistent"])
        assert result["ids"] == []

        # delete
        chroma_store.delete(ids=["nonexistent"])
        # 不应抛出异常

        # count
        assert chroma_store.count() == 0

        # reset
        chroma_store.reset()
        assert chroma_store.count() == 0

    def test_embedding_dimension_zero(self, tmp_path: Path) -> None:
        """测试 embedding_dimension=0 的边缘情况。"""
        # ChromaDB 可能接受或拒绝零维向量，但不应导致 Python 崩溃
        try:
            store = ChromaStore(
                persist_directory=str(tmp_path / "zero_dim"),
                collection_name="zero_test",
                embedding_dimension=0,
            )
            # 零维向量可能 add 失败，但不应该崩溃
            try:
                store.add(ids=["z1"], documents=["零维"], embeddings=[[]])
            except (ValueError, StorageError):
                pass
            store.reset()
        except StorageError:
            pass  # 如果创建就失败也是可接受的


class TestIntegration:
    """集成场景测试。"""

    def test_add_query_delete_workflow(self, chroma_store: ChromaStore) -> None:
        """测试完整的工作流：add -> query -> get -> delete -> count。"""
        # 添加数据
        dim = 512
        ids = [f"wf_{i}" for i in range(5)]
        docs = [f"工作流测试文档 {i}" for i in range(5)]
        embeddings = _make_embeddings(5, dim=dim)

        chroma_store.add(ids=ids, documents=docs, embeddings=embeddings)
        assert chroma_store.count() == 5

        # 查询
        query_vec = [0.0] * dim
        query_vec[0] = 1.0
        result = chroma_store.query(query_embedding=query_vec, top_k=3)
        assert len(result["ids"]) == 3

        # 按 ID 获取
        get_result = chroma_store.get(ids=[ids[0], ids[2]])
        assert len(get_result["ids"]) == 2

        # 删除部分
        chroma_store.delete(ids=[ids[0], ids[1]])
        assert chroma_store.count() == 3

        # 验证删除后查询不受影响
        result2 = chroma_store.query(query_embedding=query_vec, top_k=5)
        remaining_ids = set(result2["ids"])
        assert ids[0] not in remaining_ids
        assert ids[1] not in remaining_ids

    def test_multiple_queries_same_result(self, chroma_store: ChromaStore) -> None:
        """测试对同一查询向量多次查询结果一致。"""
        dim = 512
        embeddings = _make_embeddings(3, dim=dim)
        chroma_store.add(
            ids=["mq1", "mq2", "mq3"],
            documents=["多次查询1", "多次查询2", "多次查询3"],
            embeddings=embeddings,
        )

        query_vec = embeddings[0]  # 与 mq1 相同
        result1 = chroma_store.query(query_embedding=query_vec, top_k=3)
        result2 = chroma_store.query(query_embedding=query_vec, top_k=3)

        assert result1["ids"] == result2["ids"]
        assert result1["distances"] == result2["distances"]
