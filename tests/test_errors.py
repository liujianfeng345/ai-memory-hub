"""
自定义异常体系测试。

验证所有异常类型可被正确实例化、__str__ 输出格式正确、
继承链正确，以及支持 pickle 序列化。
"""

import pickle

import pytest

from memory_agent.utils.errors import (
    ConfigError,
    DimensionMismatchError,
    EmbeddingError,
    LLMResponseParseError,
    LLMServiceError,
    MemoryAgentError,
    ModelLoadError,
    StorageError,
)

# 所有具体异常子类列表
ALL_EXCEPTIONS = [
    ConfigError,
    StorageError,
    ModelLoadError,
    EmbeddingError,
    LLMServiceError,
    LLMResponseParseError,
    DimensionMismatchError,
]


class TestMemoryAgentError:
    """MemoryAgentError 基类测试。"""

    def test_basic_instantiation(self) -> None:
        """测试基类可正确实例化。"""
        exc = MemoryAgentError("测试错误")
        assert exc.message == "测试错误"
        assert exc.details == {}
        assert exc.error_code == "MEMORY_AGENT_ERROR"

    def test_with_details(self) -> None:
        """测试基类可携带结构化详情。"""
        details = {"key": "DEEPSEEK_API_KEY", "value": "missing"}
        exc = MemoryAgentError("缺少配置项", details=details)
        assert exc.details == details

    def test_str_format(self) -> None:
        """测试 __str__ 输出包含 error_code 和 message。"""
        exc = MemoryAgentError("测试错误")
        output = str(exc)
        assert "MEMORY_AGENT_ERROR" in output
        assert "测试错误" in output
        # 验证格式：[ERROR_CODE] message
        assert output.startswith("[MEMORY_AGENT_ERROR]")

    def test_is_exception(self) -> None:
        """测试基类是 Exception 的子类。"""
        assert issubclass(MemoryAgentError, Exception)


class TestConcreteExceptions:
    """具体异常子类测试。"""

    @pytest.mark.parametrize(
        "exc_class,expected_code",
        [
            (ConfigError, "CONFIG_ERROR"),
            (StorageError, "STORAGE_ERROR"),
            (ModelLoadError, "MODEL_LOAD_ERROR"),
            (EmbeddingError, "EMBEDDING_ERROR"),
            (LLMServiceError, "LLM_SERVICE_ERROR"),
            (LLMResponseParseError, "LLM_RESPONSE_PARSE_ERROR"),
            (DimensionMismatchError, "DIMENSION_MISMATCH_ERROR"),
        ],
    )
    def test_error_code(self, exc_class, expected_code) -> None:
        """测试每个异常子类设置了正确的 error_code 类属性。"""
        assert exc_class.error_code == expected_code

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_inherits_from_memory_agent_error(self, exc_class) -> None:
        """测试所有具体异常子类 is-a MemoryAgentError。"""
        exc = exc_class("测试")
        assert isinstance(exc, MemoryAgentError)
        assert isinstance(exc, Exception)

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_str_includes_error_code(self, exc_class) -> None:
        """测试每个异常的 __str__ 输出包含其 error_code。"""
        exc = exc_class("测试消息")
        output = str(exc)
        assert exc_class.error_code in output
        assert "测试消息" in output

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_instantiation_with_details(self, exc_class) -> None:
        """测试所有异常子类可携带 details。"""
        details = {"field": "test", "value": 42}
        exc = exc_class("测试", details=details)
        assert exc.details == details

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_message_property(self, exc_class) -> None:
        """测试所有异常子类正确设置 message 属性。"""
        msg = "这是一条测试错误消息"
        exc = exc_class(msg)
        assert exc.message == msg


class TestExceptionPickle:
    """异常 pickle 序列化测试。"""

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_pickle_roundtrip(self, exc_class) -> None:
        """测试所有异常子类可被 pickle 序列化和反序列化。"""
        original = exc_class("序列化测试", details={"nested": {"key": "val"}})
        dumped = pickle.dumps(original)
        restored = pickle.loads(dumped)

        assert restored.message == original.message
        assert restored.details == original.details
        assert restored.error_code == original.error_code
        assert type(restored) is type(original)

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_pickle_preserves_custom_attributes(self, exc_class) -> None:
        """测试 pickle 后 error_code 和详情保持一致。"""
        exc = exc_class("pickle 测试", details={"a": 1})
        restored = pickle.loads(pickle.dumps(exc))
        assert restored.error_code == exc.error_code


class TestExceptionUsability:
    """异常可用性测试（验收标准场景）。"""

    def test_config_error_as_per_acceptance(self) -> None:
        """测试验收标准3：ConfigError 的正确使用。
        场景：from memory_agent.utils.errors import ConfigError;
              raise ConfigError("missing key", {"key": "DEEPSEEK_API_KEY"})
        预期：抛出 ConfigError，str(e) 包含 "CONFIG_ERROR" 和 "missing key"，
              e.details 为 {"key": "DEEPSEEK_API_KEY"}
        """
        details = {"key": "DEEPSEEK_API_KEY"}
        exc = ConfigError("missing key", details)
        assert exc.error_code == "CONFIG_ERROR"
        assert "CONFIG_ERROR" in str(exc)
        assert "missing key" in str(exc)
        assert exc.details == details

    def test_can_catch_by_base_class(self) -> None:
        """测试可通过 MemoryAgentError 捕获所有子类异常。"""
        for exc_class in ALL_EXCEPTIONS:
            try:
                raise exc_class("测试")
            except MemoryAgentError as e:
                assert isinstance(e, exc_class)

    def test_exception_chain_inheritance(self) -> None:
        """测试子类 → 基类 → Exception 的完整继承链。"""
        exc = StorageError("存储失败")
        assert isinstance(exc, StorageError)
        assert isinstance(exc, MemoryAgentError)
        assert isinstance(exc, Exception)
