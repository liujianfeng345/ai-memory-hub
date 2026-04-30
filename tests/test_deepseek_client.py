"""
DeepSeekClient 单元测试。

验证 API key 校验、指数退避重试、JSON mode 解析、
extract_entities 结构化提取以及错误处理。
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from memory_agent.llm.deepseek_client import DeepSeekClient, _is_retryable_error
from memory_agent.utils.errors import ConfigError, LLMResponseParseError, LLMServiceError


def _make_retryable_error() -> APIConnectionError:
    """构造可重试的 APIConnectionError（用于重试测试）。"""
    return APIConnectionError(message="连接失败", request=MagicMock())


def _make_timeout_error() -> APITimeoutError:
    """构造可重试的 APITimeoutError（用于重试测试）。"""
    return APITimeoutError(request=MagicMock())


def _make_4xx_error() -> APIStatusError:
    """构造 4xx 认证错误（不可重试）。"""
    response = MagicMock()
    response.status_code = 401
    return APIStatusError("认证失败", response=response, body=None)


def _set_chat_completions(client: DeepSeekClient, mock_completions: MagicMock) -> None:
    """绕过 read-only property 设置 chat.completions mock。"""
    object.__setattr__(client._client.chat, "completions", mock_completions)


class TestIsRetryableError:
    """_is_retryable_error 辅助函数测试。"""

    def test_connection_error_is_retryable(self) -> None:
        """测试 APIConnectionError 可重试。"""
        assert _is_retryable_error(_make_retryable_error())

    def test_timeout_error_is_retryable(self) -> None:
        """测试 APITimeoutError 可重试。"""
        assert _is_retryable_error(_make_timeout_error())

    def test_5xx_error_is_retryable(self) -> None:
        """测试 5xx 服务端错误可重试。"""
        response = MagicMock()
        response.status_code = 500
        error = APIStatusError("server error", response=response, body=None)
        assert _is_retryable_error(error) is True

    def test_4xx_error_is_not_retryable(self) -> None:
        """测试 4xx 客户端错误不可重试。"""
        response = MagicMock()
        response.status_code = 401
        error = APIStatusError("unauthorized", response=response, body=None)
        assert _is_retryable_error(error) is False

    def test_429_error_is_not_retryable(self) -> None:
        """测试 429 限流错误不可重试（由 openai 内置限流逻辑处理或直接失败）。"""
        response = MagicMock()
        response.status_code = 429
        error = APIStatusError("rate limited", response=response, body=None)
        assert _is_retryable_error(error) is False


class TestConfig:
    """配置与构造测试。"""

    def test_constructor_uses_env_api_key(self) -> None:
        """测试构造函数从环境变量读取 API key。"""
        client = DeepSeekClient(api_key="sk-env-test-key")
        assert client.api_key == "sk-env-test-key"

    def test_constructor_raises_config_error_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """测试未配置 API key 时抛出 ConfigError。"""
        # 临时移除环境变量中的 API key
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
            DeepSeekClient(api_key=None)

    def test_constructor_accepts_explicit_key(self) -> None:
        """测试显式传入 api_key 参数有效。"""
        client = DeepSeekClient(api_key="sk-explicit-key")
        assert client.api_key == "sk-explicit-key"

    def test_default_values(self) -> None:
        """测试默认参数值。"""
        client = DeepSeekClient(api_key="sk-test")
        assert client.model == "deepseek-chat"
        assert client.base_url == "https://api.deepseek.com/v1"
        assert client.timeout == 30.0
        assert client.max_retries == 3


class TestRetryWithBackoff:
    """指数退避重试逻辑测试。"""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_two_failures(self) -> None:
        """测试前 2 次抛出 APIConnectionError，第 3 次成功。

        验证总共调用了 3 次。
        """
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(
            side_effect=[
                _make_retryable_error(),
                _make_retryable_error(),
                _make_mock_response("成功响应"),
            ]
        )

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        result = await client.chat([{"role": "user", "content": "你好"}])
        assert result == "成功响应"
        assert mock_completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_llm_service_error(self) -> None:
        """测试所有重试耗尽后抛出 LLMServiceError。"""
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(side_effect=_make_retryable_error())

        client = DeepSeekClient(api_key="sk-test", max_retries=3)
        _set_chat_completions(client, mock_completions)

        with pytest.raises(LLMServiceError) as exc_info:
            await client.chat([{"role": "user", "content": "你好"}])

        assert "retries" in exc_info.value.details
        # 总调用次数 = 1 次初始 +  max_retries 次重试 = 4
        assert mock_completions.create.call_count == 4

    @pytest.mark.asyncio
    async def test_4xx_error_does_not_retry(self) -> None:
        """测试 4xx 错误立即抛出，不重试。"""
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(side_effect=_make_4xx_error())

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        with pytest.raises(APIStatusError):
            await client.chat([{"role": "user", "content": "你好"}])

        # 401 应立即抛出，仅调用 1 次
        assert mock_completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_timeout_errors(self) -> None:
        """测试 APITimeoutError 触发重试。"""
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(
            side_effect=[
                _make_timeout_error(),
                _make_mock_response("最终成功"),
            ]
        )

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        result = await client.chat([{"role": "user", "content": "你好"}])
        assert result == "最终成功"
        assert mock_completions.create.call_count == 2


class TestChatBasic:
    """chat 基本对话测试。"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("DEEPSEEK_API_KEY") or os.environ["DEEPSEEK_API_KEY"] == "sk-test000000000000",
        reason="DEEPSEEK_API_KEY not configured with a real key",
    )
    async def test_chat_returns_nonempty_response(self, deepseek_client: DeepSeekClient) -> None:
        """测试 chat 发送简单消息并获得非空响应（需真实 API key）。"""
        response = await deepseek_client.chat(
            [{"role": "user", "content": "请回复'你好'两个字"}],
            temperature=0.3,
            max_tokens=50,
        )
        assert isinstance(response, str)
        assert len(response) > 0


class TestJsonMode:
    """JSON mode 解析测试。"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("DEEPSEEK_API_KEY") or os.environ["DEEPSEEK_API_KEY"] == "sk-test000000000000",
        reason="DEEPSEEK_API_KEY not configured with a real key",
    )
    async def test_chat_json_mode_returns_valid_json(self, deepseek_client: DeepSeekClient) -> None:
        """测试 JSON mode 下 chat 返回合法 JSON 字符串（需真实 API key）。"""
        response = await deepseek_client.chat(
            [
                {
                    "role": "user",
                    "content": '请返回一个 JSON 对象：{"name": "张三", "age": 30}',
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(response)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_chat_json_mode_parse_failure_raises_error(self) -> None:
        """测试 JSON mode 下返回非 JSON 内容时抛出 LLMResponseParseError。"""
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=_make_mock_response("这不是合法的 JSON"))

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        with pytest.raises(LLMResponseParseError) as exc_info:
            await client.chat(
                [{"role": "user", "content": "返回JSON"}],
                response_format={"type": "json_object"},
            )

        assert "original_response" in exc_info.value.details
        assert exc_info.value.details["original_response"] == "这不是合法的 JSON"

    @pytest.mark.asyncio
    async def test_chat_json_mode_valid_response_no_error(self) -> None:
        """测试 JSON mode 下返回合法 JSON 时正常返回字符串。"""
        valid_json = '{"key": "value", "items": [1, 2, 3]}'
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=_make_mock_response(valid_json))

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        result = await client.chat(
            [{"role": "user", "content": "返回JSON"}],
            response_format={"type": "json_object"},
        )
        assert result == valid_json


class TestExtractEntities:
    """extract_entities 结构化提取测试。"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("DEEPSEEK_API_KEY") or os.environ["DEEPSEEK_API_KEY"] == "sk-test000000000000",
        reason="DEEPSEEK_API_KEY not configured with a real key",
    )
    async def test_extract_entities_returns_required_keys(self, deepseek_client: DeepSeekClient) -> None:
        """测试 extract_entities 返回的结构包含 entities、preferences、relations。

        输入文本包含人物、偏好信息，验证三个键都存在且类型正确。
        """
        text = "张三喜欢喝蓝山咖啡，他今年30岁，住在北京。他每周都去健身房。"
        result = await deepseek_client.extract_entities(text)

        assert "entities" in result
        assert "preferences" in result
        assert "relations" in result
        assert isinstance(result["entities"], list)
        assert isinstance(result["preferences"], list)
        assert isinstance(result["relations"], list)

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_json_raises_error(self) -> None:
        """测试 LLM 返回非法 JSON 时 extract_entities 抛出 LLMResponseParseError。"""
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=_make_mock_response("不是JSON的纯文本回复"))

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        with pytest.raises(LLMResponseParseError):
            await client.extract_entities("测试文本")

    @pytest.mark.asyncio
    async def test_extract_entities_missing_keys_raises_error(self) -> None:
        """测试 JSON 缺少必须字段时抛出 LLMResponseParseError。"""
        # 返回缺少 preferences 和 relations 的 JSON
        incomplete_json = json.dumps({"entities": [{"name": "张三"}]})
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=_make_mock_response(incomplete_json))

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        with pytest.raises(LLMResponseParseError):
            await client.extract_entities("测试文本")

    @pytest.mark.asyncio
    async def test_extract_entities_valid_structure(self) -> None:
        """测试返回合法 JSON 结构时正确返回字典。"""
        valid_result = {
            "entities": [{"name": "张三", "type": "person", "attributes": {"age": 30}}],
            "preferences": [
                {
                    "subject": "张三",
                    "category": "drink",
                    "value": "蓝山咖啡",
                    "sentiment": "positive",
                }
            ],
            "relations": [],
        }
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=_make_mock_response(json.dumps(valid_result)))

        client = DeepSeekClient(api_key="sk-test")
        _set_chat_completions(client, mock_completions)

        result = await client.extract_entities("张三喜欢喝蓝山咖啡")
        assert result == valid_result
        assert isinstance(result["entities"], list)
        assert isinstance(result["preferences"], list)
        assert isinstance(result["relations"], list)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _make_mock_response(content: str) -> MagicMock:
    """构造模拟的 OpenAI chat completion 响应。

    Args:
        content: 响应文本内容。

    Returns:
        模拟的 chat completion 对象。
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    return mock_response
