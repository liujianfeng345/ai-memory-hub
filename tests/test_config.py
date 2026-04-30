"""
配置管理模块测试。

验证 MemoryConfig 的默认值、环境变量覆盖、脱敏和校验功能。
"""

import pytest
from pydantic import ValidationError

from memory_agent.utils.config import MemoryConfig


class TestDefaultValues:
    """默认值测试。"""

    def test_default_deepseek_api_key(self) -> None:
        """测试 deepseek_api_key 默认为空字符串。"""
        config = MemoryConfig(deepseek_api_key="")  # 显式覆盖 env
        assert config.deepseek_api_key == ""

    def test_default_deepseek_model(self) -> None:
        """测试 deepseek_model 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.deepseek_model == "deepseek-chat"

    def test_default_deepseek_base_url(self) -> None:
        """测试 deepseek_base_url 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.deepseek_base_url == "https://api.deepseek.com/v1"

    def test_default_deepseek_timeout(self) -> None:
        """测试 deepseek_timeout 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.deepseek_timeout == 30.0

    def test_default_deepseek_max_retries(self) -> None:
        """测试 deepseek_max_retries 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.deepseek_max_retries == 3

    def test_default_embedding_model_name(self) -> None:
        """测试 embedding_model_name 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.embedding_model_name == "models/bge-small-zh-v1.5"

    def test_default_embedding_device(self) -> None:
        """测试 embedding_device 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.embedding_device == "cpu"

    def test_default_chroma_persist_dir(self) -> None:
        """测试 chroma_persist_dir 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.chroma_persist_dir == "./data/chroma"

    def test_default_ttl_seconds(self) -> None:
        """测试 default_ttl_seconds 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.default_ttl_seconds == 3600

    def test_default_log_level(self) -> None:
        """测试 log_level 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.log_level == "INFO"

    def test_default_max_content_length(self) -> None:
        """测试 max_content_length 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.max_content_length == 50000

    def test_default_summary_threshold(self) -> None:
        """测试 summary_threshold 默认值。"""
        config = MemoryConfig(deepseek_api_key="test")
        assert config.summary_threshold == 2000


class TestEnvironmentVariables:
    """环境变量覆盖测试。"""

    def test_env_overrides_api_key(self, monkeypatch) -> None:
        """测试 DEEPSEEK_API_KEY 环境变量覆盖默认值。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-test-key-1234")
        config = MemoryConfig()
        assert config.deepseek_api_key == "sk-env-test-key-1234"

    def test_env_overrides_log_level(self, monkeypatch) -> None:
        """测试 LOG_LEVEL 环境变量覆盖默认值。"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = MemoryConfig(deepseek_api_key="test")
        assert config.log_level == "DEBUG"

    def test_env_overrides_timeout(self, monkeypatch) -> None:
        """测试 DEEPSEEK_TIMEOUT 环境变量覆盖默认值。"""
        monkeypatch.setenv("DEEPSEEK_TIMEOUT", "60.0")
        config = MemoryConfig(deepseek_api_key="test")
        assert config.deepseek_timeout == 60.0

    def test_env_overrides_ttl(self, monkeypatch) -> None:
        """测试 DEFAULT_TTL_SECONDS 环境变量覆盖默认值。"""
        monkeypatch.setenv("DEFAULT_TTL_SECONDS", "7200")
        config = MemoryConfig(deepseek_api_key="test")
        assert config.default_ttl_seconds == 7200

    def test_env_does_not_affect_other_fields(self, monkeypatch) -> None:
        """测试设置单个环境变量不影响其他字段的默认值。"""
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        config = MemoryConfig(deepseek_api_key="test")
        # log_level 被覆盖
        assert config.log_level == "ERROR"
        # 其他字段保持默认
        assert config.deepseek_model == "deepseek-chat"
        assert config.embedding_device == "cpu"


class TestApiKeySerialization:
    """API Key 脱敏测试。"""

    def test_normal_key_desensitized(self) -> None:
        """测试正常长度的 API key 脱敏：仅保留前4后4。"""
        config = MemoryConfig(deepseek_api_key="sk-abcdefgh12345678")
        dumped = config.model_dump()
        assert dumped["deepseek_api_key"] == "sk-a****5678"

    def test_short_key_desensitized(self) -> None:
        """测试短 API key（<=8字符）脱敏。"""
        config = MemoryConfig(deepseek_api_key="sk-test")
        dumped = config.model_dump()
        assert dumped["deepseek_api_key"] == "sk-t****"

    def test_empty_key(self) -> None:
        """测试空 API key 脱敏。"""
        config = MemoryConfig(deepseek_api_key="")
        dumped = config.model_dump()
        assert dumped["deepseek_api_key"] == ""

    def test_very_long_key_desensitized(self) -> None:
        """测试超长 API key 脱敏。"""
        long_key = "sk-" + "a" * 100
        config = MemoryConfig(deepseek_api_key=long_key)
        dumped = config.model_dump()
        assert dumped["deepseek_api_key"] == "sk-a****" + long_key[-4:]

    def test_desensitized_key_not_expose_full(self) -> None:
        """测试脱敏后的 key 不包含原始完整值。"""
        original = "sk-secret-api-key-1234567890"
        config = MemoryConfig(deepseek_api_key=original)
        dumped = config.model_dump()
        assert original not in dumped["deepseek_api_key"]


class TestValidation:
    """字段校验测试。"""

    def test_deepseek_timeout_zero_raises(self) -> None:
        """测试 deepseek_timeout = 0 时校验失败。"""
        with pytest.raises(ValidationError) as exc_info:
            MemoryConfig(deepseek_api_key="test", deepseek_timeout=0)
        assert "deepseek_timeout" in str(exc_info.value)

    def test_deepseek_timeout_negative_raises(self) -> None:
        """测试 deepseek_timeout 为负数时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", deepseek_timeout=-1.0)

    def test_deepseek_max_retries_negative_raises(self) -> None:
        """测试 deepseek_max_retries 为负数时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", deepseek_max_retries=-1)

    def test_default_ttl_zero_raises(self) -> None:
        """测试 default_ttl_seconds = 0 时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", default_ttl_seconds=0)

    def test_default_ttl_negative_raises(self) -> None:
        """测试 default_ttl_seconds 为负数时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", default_ttl_seconds=-100)

    def test_max_content_length_zero_raises(self) -> None:
        """测试 max_content_length = 0 时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", max_content_length=0)

    def test_summary_threshold_zero_raises(self) -> None:
        """测试 summary_threshold = 0 时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", summary_threshold=0)

    def test_invalid_log_level_raises(self) -> None:
        """测试非法 log_level 时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", log_level="INVALID")

    def test_valid_log_levels_accepted(self) -> None:
        """测试所有合法 log_level 值均通过校验。"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = MemoryConfig(deepseek_api_key="test", log_level=level)
            assert config.log_level == level

    def test_log_level_case_insensitive(self) -> None:
        """测试 log_level 大小写不敏感。"""
        config = MemoryConfig(deepseek_api_key="test", log_level="debug")
        assert config.log_level == "DEBUG"

    def test_invalid_embedding_device_raises(self) -> None:
        """测试非法 embedding_device 时校验失败。"""
        with pytest.raises(ValidationError):
            MemoryConfig(deepseek_api_key="test", embedding_device="tpu")

    def test_valid_embedding_devices_accepted(self) -> None:
        """测试所有合法 embedding_device 值均通过校验。"""
        for device in ["cpu", "cuda", "mps"]:
            config = MemoryConfig(deepseek_api_key="test", embedding_device=device)
            assert config.embedding_device == device


class TestLogLevelInt:
    """log_level_int 属性测试。"""

    def test_log_level_int_debug(self) -> None:
        """测试 DEBUG 级别映射。"""
        config = MemoryConfig(deepseek_api_key="test", log_level="DEBUG")
        import logging

        assert config.log_level_int == logging.DEBUG

    def test_log_level_int_info(self) -> None:
        """测试 INFO 级别映射。"""
        config = MemoryConfig(deepseek_api_key="test", log_level="INFO")
        import logging

        assert config.log_level_int == logging.INFO


class TestSampleConfig:
    """sample_config fixture 测试。"""

    def test_sample_config_uses_temp_paths(self, sample_config) -> None:
        """测试 sample_config 使用临时路径。"""
        assert "chroma" in sample_config.chroma_persist_dir

    def test_sample_config_uses_fake_key(self, sample_config) -> None:
        """测试 sample_config 使用假 API key。"""
        assert sample_config.deepseek_api_key == "sk-test000000000000"

    def test_sample_config_log_level(self, sample_config) -> None:
        """测试 sample_config 默认 log_level 为 DEBUG。"""
        assert sample_config.log_level == "DEBUG"
