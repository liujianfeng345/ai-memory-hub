"""
自定义异常体系。

所有异常均继承自 MemoryAgentError 基类，每个子类通过 error_code 类属性标识错误类型，
方便调用方通过 `type(exc).error_code` 做程序化判断。
"""

from typing import Any


class MemoryAgentError(Exception):
    """Memory Agent 异常基类。

    所有自定义异常均应继承此类，以提供统一的 error_code 和 details 机制。

    Attributes:
        error_code: 错误码，子类应覆盖此类属性。
        message: 错误描述信息。
        details: 可选的错误详情字典，用于传递结构化的上下文信息。
    """

    error_code: str = "MEMORY_AGENT_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """初始化异常实例。

        Args:
            message: 错误描述信息。
            details: 可选的错误详情字典。
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """返回包含 error_code 和 message 的格式化字符串。"""
        return f"[{self.error_code}] {self.message}"

    def __reduce__(self) -> tuple[Any, ...]:
        """支持 pickle 序列化。"""
        return (self.__class__, (self.message, self.details))


class ConfigError(MemoryAgentError):
    """配置相关错误。

    当环境变量缺失、配置值非法或配置文件解析失败时抛出。
    """

    error_code = "CONFIG_ERROR"


class StorageError(MemoryAgentError):
    """存储相关错误。

    当 ChromaDB 读写操作失败、持久化目录不可访问时抛出。
    """

    error_code = "STORAGE_ERROR"


class ModelLoadError(MemoryAgentError):
    """模型加载错误。

    当嵌入模型或 LLM 模型文件加载失败时抛出。
    """

    error_code = "MODEL_LOAD_ERROR"


class EmbeddingError(MemoryAgentError):
    """嵌入计算错误。

    当文本嵌入计算过程中发生异常时抛出。
    """

    error_code = "EMBEDDING_ERROR"


class LLMServiceError(MemoryAgentError):
    """LLM 服务调用错误。

    当 DeepSeek API 请求失败、网络超时或服务不可用时抛出。
    """

    error_code = "LLM_SERVICE_ERROR"


class LLMResponseParseError(MemoryAgentError):
    """LLM 响应解析错误。

    当 API 返回的 JSON 格式异常或无法提取结构化数据时抛出。
    """

    error_code = "LLM_RESPONSE_PARSE_ERROR"


class DimensionMismatchError(MemoryAgentError):
    """向量维度不匹配错误。

    当嵌入模型输出的向量维度与 ChromaDB 集合预期的维度不一致时抛出。
    """

    error_code = "DIMENSION_MISMATCH_ERROR"
