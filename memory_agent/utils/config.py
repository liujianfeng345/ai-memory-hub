"""
配置管理模块。

使用 pydantic-settings 的 BaseSettings 实现类型安全的配置加载，
支持从 .env 文件和环境变量中读取配置项，自动校验和脱敏。
"""

import logging
from typing import Any

from pydantic import field_serializer, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryConfig(BaseSettings):
    """Memory Agent 全局配置。

    所有配置项均可通过环境变量或 .env 文件设置，优先级：环境变量 > .env 文件 > 默认值。

    Attributes:
        deepseek_api_key: DeepSeek API 密钥，脱敏输出时仅显示前4位和后4位。
        deepseek_model: 使用的模型名称。
        deepseek_base_url: API 基础地址。
        deepseek_timeout: API 请求超时秒数，必须 > 0。
        deepseek_max_retries: 最大重试次数，必须 >= 0。
        embedding_model_name: 嵌入模型本地路径或 HuggingFace 模型名称。
        embedding_device: 推理设备（cpu / cuda）。
        chroma_persist_dir: ChromaDB 持久化目录。
        default_ttl_seconds: 工作记忆默认 TTL（秒），必须 > 0。
        log_level: 日志级别（DEBUG / INFO / WARNING / ERROR）。
        max_content_length: 内容最大字符数，必须 > 0。
        summary_threshold: 触发自动摘要的字符数阈值，必须 > 0。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- DeepSeek API 配置 ---
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_timeout: float = 30.0
    deepseek_max_retries: int = 3

    # --- 嵌入模型配置 ---
    embedding_model_name: str = "models/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # --- 向量数据库配置 ---
    chroma_persist_dir: str = "./data/chroma"

    # --- 记忆管理配置 ---
    default_ttl_seconds: int = 3600
    max_content_length: int = 50000
    summary_threshold: int = 2000

    # --- 日志配置 ---
    log_level: str = "INFO"

    @field_serializer("deepseek_api_key")
    def serialize_api_key(self, value: str, _info: Any) -> str:
        """对 API 密钥进行脱敏处理：仅保留前4位和后4位，中间用 **** 替代。

        空字符串或长度不足8的密钥返回 "****"。
        """
        if not value:
            return ""
        if len(value) <= 8:
            return value[:4] + "****"
        return value[:4] + "****" + value[-4:]

    @field_validator("deepseek_timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """校验超时值必须为正数。"""
        if v <= 0:
            raise ValueError(f"deepseek_timeout 必须大于 0，当前值: {v}")
        return v

    @field_validator("deepseek_max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        """校验重试次数必须为非负整数。"""
        if v < 0:
            raise ValueError(f"deepseek_max_retries 必须 >= 0，当前值: {v}")
        return v

    @field_validator("default_ttl_seconds")
    @classmethod
    def validate_ttl(cls, v: int) -> int:
        """校验 TTL 必须为正数。"""
        if v <= 0:
            raise ValueError(f"default_ttl_seconds 必须大于 0，当前值: {v}")
        return v

    @field_validator("max_content_length")
    @classmethod
    def validate_max_content_length(cls, v: int) -> int:
        """校验最大内容长度必须为正数。"""
        if v <= 0:
            raise ValueError(f"max_content_length 必须大于 0，当前值: {v}")
        return v

    @field_validator("summary_threshold")
    @classmethod
    def validate_summary_threshold(cls, v: int) -> int:
        """校验摘要阈值必须为正数。"""
        if v <= 0:
            raise ValueError(f"summary_threshold 必须大于 0，当前值: {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """校验日志级别必须是 Python logging 支持的有效级别。"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level 必须是 {valid_levels} 之一，当前值: {v}")
        return upper

    @field_validator("embedding_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """校验推理设备为合法值。"""
        valid_devices = {"cpu", "cuda", "mps"}
        lower = v.lower()
        if lower not in valid_devices:
            raise ValueError(f"embedding_device 必须是 {valid_devices} 之一，当前值: {v}")
        return lower

    @property
    def log_level_int(self) -> int:
        """返回 log_level 对应的 Python logging 级别整数。"""
        return getattr(logging, self.log_level, logging.INFO)
