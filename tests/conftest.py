"""
测试全局 fixtures。

为后续阶段提供可复用的测试夹具，包括临时目录和示例配置。
"""

import os
from collections.abc import Generator
from pathlib import Path

import pytest

# 确保测试运行时不加载真实的 .env 文件，避免影响测试结果
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
