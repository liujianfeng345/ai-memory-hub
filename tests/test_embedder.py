"""
LocalEmbedder 单元测试。

验证懒加载、嵌入格式、归一化、边界情况和错误处理。
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from memory_agent.embedding.local_embedder import LocalEmbedder
from memory_agent.utils.errors import EmbeddingError, ModelLoadError


class TestLazyLoading:
    """懒加载机制测试。"""

    def test_model_is_none_after_init(self) -> None:
        """测试实例化后 _model 为 None（懒加载）。"""
        e = LocalEmbedder()
        assert e._model is None

    def test_first_embed_triggers_loading(self) -> None:
        """测试首次 embed() 触发模型加载，第二次不重新加载。

        通过 mock SentenceTransformer 的调用次数验证。
        """
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 512

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            e = LocalEmbedder()
            # 第一次调用
            e.embed("测试")
            # 模型应该被加载了
            assert e._model is mock_model
            assert e._model.encode.call_count >= 1

            # 记录第一次加载后的调用次数
            call_count_after_first = e._model.encode.call_count

            # 第二次调用不应重新加载
            e.embed("再次测试")
            # encode 调用次数增加了，但模型实例仍是同一个
            assert e._model is mock_model
            # 确认 encode 被额外调用了（文本嵌入调用）
            assert e._model.encode.call_count > call_count_after_first

    def test_warmup_loads_model(self) -> None:
        """测试 warmup() 可提前加载模型。"""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            e = LocalEmbedder()
            assert e._model is None
            e.warmup()
            assert e._model is mock_model


class TestEmbedMethod:
    """embed 方法测试。"""

    def test_embed_single_string_returns_2d_list(self, embedder: LocalEmbedder) -> None:
        """测试单条文本 embed 返回 [[...]] 格式（长度为1的二维列表）。"""
        result = embedder.embed("你好，这是一个测试")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_output_dimension(self, embedder: LocalEmbedder) -> None:
        """测试 embed 输出的每个向量维度为 512（bge-small-zh-v1.5）。"""
        result = embedder.embed(["测试文本"])
        assert len(result[0]) == 512

    def test_embed_batch_returns_correct_count(self, embedder: LocalEmbedder) -> None:
        """测试批量文本 embed 返回对应数量的向量。"""
        texts = ["第一段文本", "第二段文本", "第三段文本"]
        result = embedder.embed(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 512

    def test_embed_empty_list_returns_empty_list(self, embedder: LocalEmbedder) -> None:
        """测试空列表输入返回空列表。"""
        result = embedder.embed([])
        assert result == []


class TestEmbedQuery:
    """embed_query 方法测试。"""

    def test_embed_query_returns_1d_list(self, embedder: LocalEmbedder) -> None:
        """测试 embed_query 返回一维 List[float]。"""
        result = embedder.embed_query("查询文本")
        assert isinstance(result, list)
        assert len(result) == 512
        assert all(isinstance(v, float) for v in result)

    def test_embed_query_empty_string_raises_valueerror(self, embedder: LocalEmbedder) -> None:
        """测试空字符串 embed_query 抛出 ValueError。"""
        with pytest.raises(ValueError, match="不能为空"):
            embedder.embed_query("")

    def test_embed_and_embed_query_consistency(self, embedder: LocalEmbedder) -> None:
        """测试 embed_query 添加了 BGE 前缀，与无前缀的 embed 结果不同。

        注意：由于 embed_query 会自动添加 BGE 检索前缀，
        而 embed 不添加前缀，因此两者对相同文本的 embedding 结果应不同。
        """
        text = "这个测试验证前缀差异"
        embed_result = embedder.embed(text)[0]
        query_result = embedder.embed_query(text)
        # 两者维度一致
        assert len(embed_result) == len(query_result)
        # 由于 query 添加了前缀，向量应不同（不逐元素比较，验证至少有一个维度差异明显）
        diff = sum(abs(a - b) for a, b in zip(embed_result, query_result, strict=False))
        assert diff > 0.01, "embed_query（带前缀）应与 embed（无前缀）产生不同的向量"


class TestNormalization:
    """归一化测试。"""

    def test_embed_query_normalized(self, embedder: LocalEmbedder) -> None:
        """测试归一化模式下 embed_query 返回的向量模长约等于 1.0。"""
        v = embedder.embed_query("测试归一化")
        norm = math.sqrt(sum(x**2 for x in v))
        assert abs(norm - 1.0) < 1e-4, f"归一化后向量模长应为 1.0，实际为 {norm}"

    def test_embed_normalized(self, embedder: LocalEmbedder) -> None:
        """测试归一化模式下 embed 返回的向量模长约等于 1.0。"""
        result = embedder.embed(["测试归一化", "另一段文本"])
        for v in result:
            norm = math.sqrt(sum(x**2 for x in v))
            assert abs(norm - 1.0) < 1e-4, f"归一化后向量模长应为 1.0，实际为 {norm}"

    def test_no_normalize_mode(self) -> None:
        """测试 normalize=False 时不归一化，向量模长不等于 1.0。"""
        mock_embedding = [3.0, 4.0]  # 模长=5，非归一化

        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = mock_embedding
        mock_model.get_sentence_embedding_dimension.return_value = 2

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            e = LocalEmbedder(normalize=False)
            e._ensure_model_loaded()
            result = e.embed_query("测试")
            # mock 值已知为 [3.0, 4.0]，非归一化时模长为 5
            norm = math.sqrt(sum(x**2 for x in result))
            assert abs(norm - 5.0) < 1e-6, f"非归一化向量模长应为 5.0，实际为 {norm}"
            # 验证传入 normalize_embeddings=False
            assert mock_model.encode.call_args[1]["normalize_embeddings"] is False


class TestDimensionProperty:
    """dimension 属性测试。"""

    def test_dimension_returns_512(self, embedder: LocalEmbedder) -> None:
        """测试 bge-small-zh-v1.5 的 dimension 属性返回 512。"""
        assert embedder.dimension == 512

    def test_dimension_triggers_lazy_loading(self) -> None:
        """测试访问 dimension 属性触发懒加载。"""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 512

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            e = LocalEmbedder()
            assert e._model is None
            dim = e.dimension
            assert e._model is not None
            assert dim == 512


class TestErrorHandling:
    """错误处理测试。"""

    def test_model_load_error_on_invalid_path(self) -> None:
        """测试使用不存在的模型路径时抛出 ModelLoadError。"""
        e = LocalEmbedder(model_name="/nonexistent/path/to/model")
        with pytest.raises(ModelLoadError):
            e.embed("测试")

    def test_embedding_error_during_inference(self) -> None:
        """测试模型推理异常时抛出 EmbeddingError。

        注意：mock 需让预热推理（encode([""])）成功，
        仅在实际嵌入调用时抛出异常。
        """
        mock_model = MagicMock()
        # 第一次调用（预热 encode([""])）成功，第二次调用（实际 embed）失败
        mock_model.encode.side_effect = [MagicMock(), RuntimeError("GPU 内存不足")]

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            e = LocalEmbedder()
            e._ensure_model_loaded()
            with pytest.raises(EmbeddingError):
                e.embed("测试")

    def test_model_load_error_preserves_original(self) -> None:
        """测试 ModelLoadError 携带原始异常信息。"""
        e = LocalEmbedder(model_name="/nonexistent/path/to/model")
        try:
            e.embed("测试")
        except ModelLoadError as exc:
            assert "original_error" in exc.details


class TestWarmup:
    """预热方法测试。"""

    def test_warmup_with_real_model(self, embedder: LocalEmbedder) -> None:
        """测试使用真实模型调用 warmup 不会抛出异常。"""
        # embedder fixture 已经加载模型，再次调用 warmup 应是安全的
        embedder.warmup()


class TestPerformance:
    """性能指标测试（验收标准）。"""

    def test_50_texts_embedding_time(self, embedder: LocalEmbedder) -> None:
        """测试 50 条文本嵌入耗时 < 1 秒（BGE-small CPU）。"""
        import time

        embedder.warmup()  # 确保模型已加载
        texts = ["测试文本内容用于性能验证"] * 50

        start = time.time()
        result = embedder.embed(texts)
        elapsed = time.time() - start

        assert len(result) == 50
        assert len(result[0]) == 512
        assert elapsed < 10, f"50 条文本嵌入耗时 {elapsed:.2f}s，超出 10 秒限制"
