"""
ChromaDB 向量存储封装。

ChromaStore 封装 chromadb.PersistentClient，提供向量存储、检索、
维度检测和异常包装功能，供情景记忆和语义记忆模块使用。
"""

import logging
import os
from typing import Any

import chromadb
from chromadb import PersistentClient

from memory_agent.utils.errors import StorageError

logger = logging.getLogger(__name__)


class ChromaStore:
    """ChromaDB 向量存储封装。

    使用 ChromaDB PersistentClient 进行本地持久化存储，
    提供 Collection 的创建、向量增删改查等操作。
    所有 ChromaDB 原生异常均被捕获并包装为 StorageError。

    Attributes:
        persist_directory: 持久化目录的绝对路径。
        collection_name: Collection 名称。
        embedding_dimension: 嵌入向量的维度。
        client: ChromaDB PersistentClient 实例。
        collection: ChromaDB Collection 实例。
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_dimension: int = 512,
    ) -> None:
        """初始化 ChromaStore。

        创建 PersistentClient 并获取或创建指定的 Collection。
        若已存在的 Collection 的 embedding_dimension 与参数不一致，
        则记录 warning 日志，删除旧 Collection 并重新创建。

        Args:
            persist_directory: 持久化存储目录路径。
            collection_name: Collection 名称。
            embedding_dimension: 嵌入向量维度，默认 512。

        Raises:
            StorageError: 当 ChromaDB 初始化或操作失败时抛出。
        """
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        try:
            # 在 Windows 下确保持久化路径已规范化
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client: PersistentClient = chromadb.PersistentClient(path=self.persist_directory)

            # 检测 Collection 是否已存在并验证维度
            self.collection = self._get_or_create_collection()

            logger.info(
                "ChromaStore 初始化完成 persist_dir=%s collection=%s dim=%d count=%d",
                self.persist_directory,
                self.collection_name,
                self.embedding_dimension,
                self.collection.count() if self.collection else 0,
            )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"初始化 ChromaStore 失败: {e}",
                details={
                    "persist_directory": self.persist_directory,
                    "collection_name": self.collection_name,
                    "original_error": str(e),
                },
            ) from e

    def _get_or_create_collection(self) -> Any:
        """获取或创建 Collection，处理维度/距离度量不匹配的情况。

        若 Collection 已存在且 metadata 中的 embedding_dimension 或 hnsw:space
        与预期不一致，则记录 warning，删除旧 Collection 并重建。

        Returns:
            ChromaDB Collection 实例。

        Raises:
            StorageError: 当 ChromaDB 操作失败时抛出。
        """
        try:
            # 尝试获取已存在的 Collection
            existing = self.client.get_collection(name=self.collection_name)

            needs_rebuild = False

            # 检查维度是否匹配
            stored_dim = None
            if existing.metadata:
                stored_dim_str = existing.metadata.get("embedding_dimension")
                if stored_dim_str is not None:
                    stored_dim = int(stored_dim_str)

            if stored_dim is not None and stored_dim != self.embedding_dimension:
                logger.warning(
                    "Collection '%s' 维度不匹配: 已存储 dim=%d, 请求 dim=%d。",
                    self.collection_name,
                    stored_dim,
                    self.embedding_dimension,
                )
                needs_rebuild = True

            # 检查距离度量是否匹配（旧 Collection 可能没有此 metadata）
            stored_space = None
            if existing.metadata:
                stored_space = existing.metadata.get("hnsw:space")

            if stored_space is not None and stored_space != "cosine":
                logger.warning(
                    "Collection '%s' 距离度量不匹配: 已存储 '%s', 请求 'cosine'。",
                    self.collection_name,
                    stored_space,
                )
                needs_rebuild = True

            if needs_rebuild:
                self.client.delete_collection(name=self.collection_name)
                return self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "embedding_dimension": str(self.embedding_dimension),
                        "hnsw:space": "cosine",
                    },
                )

            return existing

        except ValueError:
            # Collection 不存在（get_collection 抛出 ValueError）
            pass
        except Exception:
            # 其他异常也可能是 Collection 不存在
            pass

        # 创建新的 Collection（使用余弦距离以匹配 L2 归一化向量）
        return self.client.create_collection(
            name=self.collection_name,
            metadata={
                "embedding_dimension": str(self.embedding_dimension),
                "hnsw:space": "cosine",
            },
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """批量添加向量记录。

        校验 ids、documents、embeddings 三个列表长度一致，
        不一致时抛出 ValueError。

        Args:
            ids: 记录 ID 列表。
            documents: 文档内容列表。
            embeddings: 嵌入向量列表，每个向量为 List[float]。
            metadatas: 可选的元数据列表，长度需与 ids 一致。

        Raises:
            ValueError: 当输入列表长度不一致时抛出。
            StorageError: 当 ChromaDB 操作失败时抛出。
        """
        # 空列表直接返回，避免触发 ChromaDB 对空列表的校验
        if len(ids) == 0:
            logger.debug("ADD 空批次，跳过")
            return

        if len(ids) != len(documents) or len(ids) != len(embeddings):
            raise ValueError(
                f"ids、documents、embeddings 长度必须一致: "
                f"ids={len(ids)}, documents={len(documents)}, embeddings={len(embeddings)}"
            )

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(
                f"ids、documents、embeddings、metadatas 长度必须一致: "
                f"ids={len(ids)}, documents={len(documents)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.debug("ADD 添加了 %d 条记录到 collection='%s'", len(ids), self.collection_name)
        except ValueError:
            raise
        except Exception as e:
            raise StorageError(
                f"添加向量记录失败: {e}",
                details={
                    "collection_name": self.collection_name,
                    "count": len(ids),
                    "original_error": str(e),
                },
            ) from e

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """向量相似度检索。

        使用余弦相似度在 Collection 中查询与 query_embedding 最相似的记录。

        Args:
            query_embedding: 查询向量。
            top_k: 返回的最大结果数，默认 10。
            where: 可选的元数据过滤条件。
            where_document: 可选的文档内容过滤条件。

        Returns:
            字典格式：{"ids": List[str], "documents": List[str],
            "metadatas": List[Dict], "distances": List[float]}。
            若 Collection 为空，返回所有列表为空列表。

        Raises:
            StorageError: 当 ChromaDB 查询失败时抛出。
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                where_document=where_document,
            )

            # ChromaDB 返回嵌套列表格式，需要展平
            # 处理空结果的情况
            ids_list = results.get("ids", [[]])
            documents_list = results.get("documents", [[]])
            metadatas_list = results.get("metadatas", [[]])
            distances_list = results.get("distances", [[]])

            return {
                "ids": ids_list[0] if ids_list and ids_list[0] else [],
                "documents": documents_list[0] if documents_list and documents_list[0] else [],
                "metadatas": metadatas_list[0] if metadatas_list and metadatas_list[0] else [],
                "distances": distances_list[0] if distances_list and distances_list[0] else [],
            }

        except Exception as e:
            raise StorageError(
                f"向量查询失败: {e}",
                details={
                    "collection_name": self.collection_name,
                    "top_k": top_k,
                    "original_error": str(e),
                },
            ) from e

    def get(self, ids: list[str]) -> dict[str, Any]:
        """按 ID 批量获取记录。

        Args:
            ids: 要获取的记录 ID 列表。

        Returns:
            字典格式：{"ids": List[str], "documents": List[str],
            "metadatas": List[Dict]}（无 distances 字段）。
            若某些 ID 不存在，返回的结果中不包含这些 ID 的记录。

        Raises:
            StorageError: 当 ChromaDB 操作失败时抛出。
        """
        # 空列表直接返回，避免触发 ChromaDB 对空列表的校验
        if len(ids) == 0:
            return {"ids": [], "documents": [], "metadatas": []}

        try:
            results = self.collection.get(ids=ids)

            ids_list = results.get("ids", [])
            documents_list = results.get("documents", [])
            metadatas_list = results.get("metadatas", [])

            return {
                "ids": ids_list if ids_list else [],
                "documents": documents_list if documents_list else [],
                "metadatas": metadatas_list if metadatas_list else [],
            }
        except Exception as e:
            raise StorageError(
                f"获取记录失败: {e}",
                details={
                    "collection_name": self.collection_name,
                    "ids": ids,
                    "original_error": str(e),
                },
            ) from e

    def delete(self, ids: list[str]) -> None:
        """按 ID 批量删除记录。

        若某些 ID 不存在，ChromaDB 不会报错，仅删除存在的记录。

        Args:
            ids: 要删除的记录 ID 列表。

        Raises:
            StorageError: 当 ChromaDB 操作失败时抛出。
        """
        # 空列表直接返回，避免触发 ChromaDB 对空列表的校验
        if len(ids) == 0:
            logger.debug("DELETE 空批次，跳过")
            return

        try:
            self.collection.delete(ids=ids)
            logger.debug("DELETE 删除了 %d 条记录从 collection='%s'", len(ids), self.collection_name)
        except Exception as e:
            raise StorageError(
                f"删除记录失败: {e}",
                details={
                    "collection_name": self.collection_name,
                    "ids": ids,
                    "original_error": str(e),
                },
            ) from e

    def count(self) -> int:
        """返回 Collection 中的记录总数。

        Returns:
            记录总数。

        Raises:
            StorageError: 当 ChromaDB 操作失败时抛出。
        """
        try:
            return self.collection.count()
        except Exception as e:
            raise StorageError(
                f"获取记录数失败: {e}",
                details={
                    "collection_name": self.collection_name,
                    "original_error": str(e),
                },
            ) from e

    def reset(self) -> None:
        """删除整个 Collection 并重建同名的空 Collection。

        用于修复损坏数据或测试清理。此操作不可逆。

        Raises:
            StorageError: 当 ChromaDB 操作失败时抛出。
        """
        collection_name = self.collection_name
        try:
            self.client.delete_collection(name=collection_name)
            logger.info("已删除 Collection '%s'，正在重建...", collection_name)
        except Exception as e:
            raise StorageError(
                f"删除 Collection 失败: {e}",
                details={
                    "collection_name": collection_name,
                    "original_error": str(e),
                },
            ) from e

        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "embedding_dimension": str(self.embedding_dimension),
                    "hnsw:space": "cosine",
                },
            )
            logger.info("Collection '%s' 已重建 dimension=%d", collection_name, self.embedding_dimension)
        except Exception as e:
            raise StorageError(
                f"重建 Collection 失败: {e}",
                details={
                    "collection_name": collection_name,
                    "original_error": str(e),
                },
            ) from e
