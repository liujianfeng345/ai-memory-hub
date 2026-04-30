"""
测试全局 fixtures。

为后续阶段提供可复用的测试夹具，包括临时目录和示例配置。
"""

import os
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
