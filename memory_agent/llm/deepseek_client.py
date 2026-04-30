"""
DeepSeek API 客户端（DeepSeekClient）。

基于 OpenAI 兼容接口的异步客户端，支持指数退避重试、结构化 JSON 输出
以及实体/偏好/关系提取。所有异常均包装为项目自定义异常。
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import BaseModel, ValidationError

from memory_agent.utils.errors import (
    ConfigError,
    LLMResponseParseError,
    LLMServiceError,
)

logger = logging.getLogger(__name__)

# extract_entities 使用的 system prompt
_EXTRACT_SYSTEM_PROMPT = """你是一个信息提取助手。请从给定文本中提取以下三类结构化信息：

1. **entities**: 文本中提到的人、物、地点、组织等实体。每个实体应包含：
   - name: 实体名称
   - type: 实体类型（person, location, organization, object, event, other）
   - attributes: 实体的属性描述（可选）

2. **preferences**: 文本中表达的用户偏好或倾向。每条偏好应包含：
   - subject: 偏好主体（通常是用户或文本中的角色）
   - category: 偏好类别（如 food, activity, technology, sport 等）
   - value: 偏好的具体内容
   - sentiment: 情感倾向（positive, negative, neutral）

3. **relations**: 实体之间的关系。每条关系应包含：
   - source: 关系起点实体名称
   - target: 关系终点实体名称
   - relation: 关系类型（如 likes, works_at, lives_in, knows 等）

请以 JSON 格式返回，包含 entities、preferences、relations 三个键。
若某类信息不存在，对应键的值应为空列表。"""


class _EntityExtractionResult(BaseModel):
    """extract_entities 返回结果的 Pydantic 校验模型。

    Attributes:
        entities: 提取的实体列表。
        preferences: 提取的偏好列表。
        relations: 提取的关系列表。
    """

    entities: list[dict[str, Any]]
    preferences: list[dict[str, Any]]
    relations: list[dict[str, Any]]


def _is_retryable_error(error: Exception) -> bool:
    """判断异常是否可重试。

    仅对网络层面的错误（连接失败、超时、服务端 5xx）进行重试。
    4xx 错误（认证失败、参数错误等）立即抛出，不重试。

    Args:
        error: 捕获到的异常。

    Returns:
        True 如果可以重试，False 否则。
    """
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True

    if isinstance(error, APIStatusError):
        # 5xx 服务端错误可重试，4xx 客户端错误不重试
        return error.status_code >= 500

    return False


class DeepSeekClient:
    """DeepSeek API 异步客户端。

    使用 OpenAI 官方 SDK 的 AsyncOpenAI 与 DeepSeek API 通信。
    自行实现指数退避重试逻辑，仅对网络/服务端错误重试。

    Attributes:
        api_key: DeepSeek API 密钥。
        model: 模型名称。
        base_url: API 基础地址。
        timeout: 请求超时秒数。
        max_retries: 最大重试次数（不包含首次请求）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """初始化 DeepSeekClient。

        Args:
            api_key: DeepSeek API 密钥，若为 None 则从环境变量 DEEPSEEK_API_KEY 读取。
            model: 模型名称，默认 "deepseek-chat"。
            base_url: API 基础地址。
            timeout: 请求超时秒数。
            max_retries: 最大重试次数（不包含首次请求），默认 3。

        Raises:
            ConfigError: api_key 未配置或为空字符串。
        """
        # 解析 API key
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")

        if not api_key:
            raise ConfigError(
                "DEEPSEEK_API_KEY 未配置",
                details={"hint": "请设置环境变量 DEEPSEEK_API_KEY 或传入 api_key 参数"},
            )

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # 使用 max_retries=0，自行管理重试逻辑
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def _retry_with_backoff(self, func, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        """通用指数退避重试逻辑。

        重试次数上限为 self.max_retries，每次重试等待 2^(i-1) 秒。
        仅对可重试的网络/服务端错误重试，4xx 等客户端错误直接抛出。

        Args:
            func: 要执行的异步可调用对象。
            *args: 位置参数，传递给 func。
            **kwargs: 关键字参数，传递给 func。

        Returns:
            func 的返回值。

        Raises:
            LLMServiceError: 所有重试耗尽后仍失败时抛出。
            openai 异常: 不可重试的错误（4xx 等）直接向上抛出。
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e

                # 判断是否可重试
                if not _is_retryable_error(e):
                    # 不可重试的错误直接抛出（如 4xx 认证/参数错误）
                    raise

                # 已达到最大重试次数
                if attempt >= self.max_retries:
                    break

                # 指数退避：首次重试等待 1s，第二次 2s，第三次 4s
                delay = 2**attempt
                logger.warning(
                    "API 调用失败，将在 %.1fs 后进行第 %d/%d 次重试: %s",
                    delay,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                await asyncio.sleep(delay)

        # 所有重试耗尽
        raise LLMServiceError(
            f"DeepSeek API 调用失败，已重试 {self.max_retries} 次",
            details={
                "original_error": str(last_error) if last_error else "未知错误",
                "retries": self.max_retries,
            },
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """发送对话请求并返回 LLM 响应文本。

        通过重试机制包装 OpenAI API 调用，支持 JSON mode 下的响应解析校验。

        Args:
            messages: 对话消息列表，每条消息包含 "role" 和 "content" 字段。
            temperature: 生成温度，值越高随机性越大，默认 0.7。
            max_tokens: 最大生成 token 数，默认 2048。
            response_format: 可选的响应格式约束，如 {"type": "json_object"}。

        Returns:
            LLM 生成的响应文本。

        Raises:
            LLMServiceError: API 调用重试耗尽后失败。
            LLMResponseParseError: JSON mode 下响应不是合法 JSON。
        """

        async def _call() -> str:
            start_time = time.time()

            # DeepSeek API 目前的 response_format 支持情况，
            # 仅当用户明确传入时才添加此参数
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                create_kwargs["response_format"] = response_format

            response = await self._client.chat.completions.create(**create_kwargs)

            elapsed = time.time() - start_time
            content = response.choices[0].message.content or ""

            # 记录请求日志
            usage = response.usage
            token_info = ""
            if usage:
                token_info = f", prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}"

            logger.info(
                "API 调用成功: model=%s, 耗时=%.2fs%s",
                self.model,
                elapsed,
                token_info,
            )

            # JSON mode 校验
            if response_format and response_format.get("type") == "json_object":
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    raise LLMResponseParseError(
                        "LLM 返回的内容不是合法 JSON",
                        details={"original_response": content, "parse_error": str(e)},
                    ) from e

            return content

        return await self._retry_with_backoff(_call)  # type: ignore[no-any-return]

    async def extract_entities(self, text: str) -> dict[str, Any]:
        """从文本中提取实体、偏好和关系。

        使用结构化的 system prompt 引导 LLM 进行信息提取，
        并通过 Pydantic 模型校验返回结果的字段完整性。

        Args:
            text: 待提取信息的文本。

        Returns:
            包含 entities、preferences、relations 三个键的字典。

        Raises:
            LLMResponseParseError: LLM 返回非法 JSON 或字段不满足 Pydantic 模型。
            LLMServiceError: API 调用重试耗尽后失败。
        """
        raw_response = await self.chat(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(
                "extract_entities 返回的内容不是合法 JSON",
                details={"original_response": raw_response, "parse_error": str(e)},
            ) from e

        try:
            result = _EntityExtractionResult(**parsed)
        except ValidationError as e:
            raise LLMResponseParseError(
                "extract_entities 返回的 JSON 结构不符合预期",
                details={
                    "original_response": raw_response,
                    "validation_errors": e.errors(),
                },
            ) from e

        return result.model_dump()
