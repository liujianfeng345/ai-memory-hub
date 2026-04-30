"""
测试全局 fixtures。

为后续阶段提供可复用的测试夹具，包括临时目录、示例配置、
嵌入模型实例以及 DeepSeek 客户端（真实/模拟）。
"""

import os
from collections.abc import Generator
from pathlib import Path

import pytest

# 确保测试环境有默认的测试 API key，避免因缺少环境变量导致非 API 测试失败
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test000000000000")


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """创建临时目录 fixture。

    使用 pytest 内置的 tmp_path，提供隔离的文件系统环境，
    供后续阶段的存储和文件操作测试使用。

    Args:
        tmp_path: pytest 内置 fixture，自动创建和清理临时目录。

    Returns:
        临时目录的 Path 对象。
    """
    return tmp_path


@pytest.fixture
def sample_config(tmp_path: Path) -> "MemoryConfig":  # type: ignore  # noqa: F821
    """创建示例配置 fixture。

    使用临时路径和假 API key 构造 MemoryConfig 实例，
    避免测试依赖真实的 API 密钥或文件系统路径。

    Args:
        tmp_path: 临时目录路径 fixture。

    Returns:
        配置了临时路径和假 API key 的 MemoryConfig 实例。
    """
    from memory_agent.utils.config import MemoryConfig

    return MemoryConfig(
        deepseek_api_key="sk-test000000000000",
        chroma_persist_dir=str(tmp_path / "chroma"),
        log_level="DEBUG",
    )


@pytest.fixture
def in_memory_store() -> Generator["InMemoryStore", None, None]:  # type: ignore  # noqa: F821
    """创建全新的 InMemoryStore 实例 fixture。

    每次测试获取一个干净的空 InMemoryStore，测试完成后自动清空。

    Yields:
        全新的 InMemoryStore 实例。
    """
    from memory_agent.storage.in_memory_store import InMemoryStore

    store = InMemoryStore()
    yield store
    store.clear()


@pytest.fixture
def chroma_store(tmp_path: Path) -> Generator["ChromaStore", None, None]:  # type: ignore  # noqa: F821
    """创建 ChromaStore 实例 fixture。

    使用 tmp_path 创建临时持久化目录，每次测试获取独立的 ChromaStore，
    teardown 时调用 reset() 清理数据。

    Args:
        tmp_path: pytest 内置 fixture，提供临时目录。

    Yields:
        配置了临时目录的 ChromaStore 实例。
    """
    from memory_agent.storage.chroma_store import ChromaStore

    persist_dir = str(tmp_path / "test_chroma")
    store = ChromaStore(
        persist_directory=persist_dir,
        collection_name="test_collection",
        embedding_dimension=512,
    )
    yield store
    # teardown: 清理数据
    store.reset()


# ---------------------------------------------------------------------------
# 阶段 3: 嵌入层与大模型客户端 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> "LocalEmbedder":  # type: ignore  # noqa: F821
    """创建 LocalEmbedder 实例 fixture（模块级别，复用模型实例）。

    使用 scope="module" 避免每次测试都重新加载嵌入模型，大幅缩短测试时间。

    Returns:
        LocalEmbedder 实例（默认 cpu 模式，开启归一化）。
    """
    from memory_agent.embedding.local_embedder import LocalEmbedder

    return LocalEmbedder()


def _has_real_api_key() -> bool:
    """检查是否配置了真实的 DeepSeek API 密钥。

    返回 False 当 API key 为占位测试值或空值时，
    此时依赖真实 API 的测试应跳过。

    Returns:
        True 如果看起来是真实 API key，False 否则。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    # 排除占位测试 key 和空值
    return bool(api_key) and api_key != "sk-test000000000000"


@pytest.fixture
def deepseek_client() -> "DeepSeekClient":  # type: ignore  # noqa: F821
    """创建 DeepSeekClient 实例 fixture。

    当 DEEPSEEK_API_KEY 未配置真实密钥时自动跳过。
    由于需要真实 API 调用，仅在有有效 API key 时使用。

    Returns:
        DeepSeekClient 实例。

    Raises:
        pytest.skip: 当没有配置真实 API key 时。
    """
    if not _has_real_api_key():
        pytest.skip("DEEPSEEK_API_KEY not configured with a real key")
    from memory_agent.llm.deepseek_client import DeepSeekClient

    return DeepSeekClient()


@pytest.fixture
def mock_deepseek_client() -> "DeepSeekClient":  # type: ignore  # noqa: F821
    """创建模拟的 DeepSeekClient fixture。

    使用 unittest.mock.AsyncMock 创建假 DeepSeekClient，
    用于不需要真实 API 调用的测试场景（如重试逻辑、JSON 解析等）。

    Returns:
        经过 AsyncMock 包装的 DeepSeekClient 实例。
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from memory_agent.llm.deepseek_client import DeepSeekClient

    # 使用 MagicMock 避免 ConfigError，直接构造一个假客户端
    with patch.object(DeepSeekClient, "__init__", lambda self, **kw: None):
        client = DeepSeekClient.__new__(DeepSeekClient)
        client.api_key = "sk-mock"
        client.model = "deepseek-chat"
        client.base_url = "https://api.deepseek.com/v1"
        client.timeout = 30.0
        client.max_retries = 3
        client._client = MagicMock()
        client.chat = AsyncMock()
        client.extract_entities = AsyncMock()
        client._retry_with_backoff = AsyncMock()
        return client
