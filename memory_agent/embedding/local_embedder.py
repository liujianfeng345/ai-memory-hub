"""
本地嵌入模型封装（LocalEmbedder）。

基于 sentence-transformers 的 BGE 中文嵌入模型，提供懒加载、归一化、
BGE 查询前缀自动添加等功能。所有异常均包装为项目自定义异常，
方便上层模块统一处理。
"""

import logging
import time

from memory_agent.utils.errors import EmbeddingError, ModelLoadError

logger = logging.getLogger(__name__)

# BGE 模型推荐的查询前缀，用于提升检索效果
_BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class LocalEmbedder:
    """本地嵌入模型封装。

    使用懒加载模式初始化 SentenceTransformer 模型，在首次调用 embed/embed_query
    或显式调用 warmup 时才加载模型文件。支持批量嵌入和单条查询嵌入。

    Attributes:
        model_name: 模型路径或 HuggingFace 模型名称。
        device: 推理设备（cpu / cuda / mps）。
        normalize: 是否对输出向量做 L2 归一化。
        cache_dir: HuggingFace 模型缓存目录。
    """

    def __init__(
        self,
        model_name: str = "models/bge-small-zh-v1.5",
        device: str = "cpu",
        normalize: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        """初始化 LocalEmbedder 实例。

        构造函数仅保存配置参数，不加载模型。

        Args:
            model_name: 模型路径或 HuggingFace 模型名称。
            device: 推理设备，可选 "cpu"、"cuda"、"mps"。
            normalize: 是否对输出向量做 L2 归一化，默认 True。
            cache_dir: 可选的 HuggingFace 缓存目录。
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.cache_dir = cache_dir

        # 懒加载：构造函数中不加载模型，仅设为 None
        self._model: SentenceTransformer | None = None  # type: ignore[name-defined] # noqa: F821

    def _ensure_model_loaded(self) -> None:
        """确保模型已加载（懒加载入口）。

        若 self._model 为 None，则从磁盘加载模型并执行一次预热推理。
        加载失败时抛出 ModelLoadError，成功时记录加载耗时日志。

        Raises:
            ModelLoadError: 模型文件不存在、内存不足或加载过程中发生其他异常。
        """
        if self._model is not None:
            return

        logger.info(
            "开始加载嵌入模型: model_name=%s, device=%s, normalize=%s",
            self.model_name,
            self.device,
            self.normalize,
        )
        start_time = time.time()

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
        except FileNotFoundError as e:
            raise ModelLoadError(
                f"模型文件未找到: {self.model_name}",
                details={"original_error": str(e)},
            ) from e
        except MemoryError as e:
            raise ModelLoadError(
                f"内存不足，无法加载模型: {self.model_name}",
                details={"original_error": str(e)},
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"加载模型失败: {self.model_name}",
                details={"original_error": str(e), "error_type": type(e).__name__},
            ) from e

        # 执行空文本预热推理，触发 JIT 优化
        try:
            self._model.encode([""])
        except Exception as e:
            raise ModelLoadError(
                f"模型预热失败: {self.model_name}",
                details={"original_error": str(e)},
            ) from e

        elapsed = time.time() - start_time
        logger.info(
            "模型加载完成: model_name=%s, 耗时=%.2fs",
            self.model_name,
            elapsed,
        )

    @property
    def dimension(self) -> int:
        """获取嵌入向量的维度。

        Returns:
            嵌入向量的维度数（bge-small-zh-v1.5 为 512）。

        Raises:
            ModelLoadError: 模型尚未加载且加载失败时抛出。
        """
        self._ensure_model_loaded()
        assert self._model is not None  # 帮助 mypy 推断类型
        return self._model.get_sentence_embedding_dimension()  # type: ignore[no-any-return]

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """将文本转换为嵌入向量。

        支持单条文本和批量文本输入。单条文本返回长度为 1 的二维列表。

        Args:
            texts: 单个字符串或字符串列表。

        Returns:
            嵌入向量列表，每个向量的维度由 dim.properties 决定。
            若输入为空列表则返回空列表。

        Raises:
            EmbeddingError: 嵌入计算过程中发生异常。
            ModelLoadError: 模型尚未加载且加载失败时抛出。
        """
        # 空列表快速返回
        if isinstance(texts, list) and len(texts) == 0:
            return []

        # str → [str] 统一处理
        if isinstance(texts, str):
            texts = [texts]

        self._ensure_model_loaded()
        assert self._model is not None

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        except Exception as e:
            raise EmbeddingError(
                f"嵌入计算失败（文本数={len(texts)}）",
                details={"original_error": str(e), "text_count": len(texts)},
            ) from e

        # 转换为 Python float 列表格式
        return embeddings.tolist()  # type: ignore[no-any-return]

    def embed_query(self, query: str) -> list[float]:
        """将查询文本转换为嵌入向量（一维）。

        自动为查询添加 BGE 模型推荐的检索前缀，以获得更好的检索效果。
        返回的向量与 embed() 结果中的单条向量形状一致。

        Args:
            query: 查询文本，不能为空字符串。

        Returns:
            一维嵌入向量 List[float]。

        Raises:
            ValueError: query 为空字符串时抛出。
            EmbeddingError: 嵌入计算过程中发生异常。
            ModelLoadError: 模型尚未加载且加载失败时抛出。
        """
        if not query:
            raise ValueError("embed_query 的 query 参数不能为空字符串")

        self._ensure_model_loaded()
        assert self._model is not None

        # BGE 模型推荐在查询文本前添加前缀以提升检索效果
        query_with_prefix = _BGE_QUERY_PREFIX + query

        try:
            embedding = self._model.encode(
                query_with_prefix,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        except Exception as e:
            raise EmbeddingError(
                "查询嵌入计算失败",
                details={"original_error": str(e), "query_length": len(query)},
            ) from e

        return embedding.tolist()  # type: ignore[no-any-return]

    def warmup(self) -> None:
        """公开预热方法。

        调用此方法可提前加载模型，避免首次 embed/embed_query 时等待。
        等价于调用 _ensure_model_loaded()。

        Raises:
            ModelLoadError: 模型加载失败。
        """
        self._ensure_model_loaded()
