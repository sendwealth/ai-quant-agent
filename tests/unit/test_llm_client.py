"""Tests for LLMClient — LangChain ChatOpenAI wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from quant_agent.llm.client import LLMClient, LLMError, get_llm_client


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_settings(**overrides):
    """Create a mock Settings object with sensible defaults."""
    defaults = {
        "openai_api_key": "sk-test-openai-key",
        "openai_model": "gpt-4o",
        "openai_base_url": "https://api.openai.com/v1",
        "zhipu_api_key": None,
        "zhipu_model": "glm-4",
        "llm_timeout": 30,
        "llm_max_retries": 2,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


# ── Init tests ──────────────────────────────────────────────────────────────


class TestLLMClientInit:
    """LLMClient.__init__ — provider selection and validation."""

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_init_with_openai_key(self, mock_chat_cls):
        """When openai_api_key is set, ChatOpenAI uses OpenAI defaults."""
        settings = _make_settings(openai_api_key="sk-test-123")
        client = LLMClient(settings=settings)

        mock_chat_cls.assert_called_once_with(
            model="gpt-4o",
            api_key="sk-test-123",
            base_url="https://api.openai.com/v1",
            timeout=30,
            max_retries=2,
            temperature=0.3,
        )
        assert client.model == "gpt-4o"

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_init_with_zhipu_key(self, mock_chat_cls):
        """When only zhipu_api_key is set, ChatOpenAI uses ZhiPu base URL."""
        settings = _make_settings(
            openai_api_key=None,
            zhipu_api_key="zhipu-test-key",
            zhipu_model="glm-4-flash",
        )
        client = LLMClient(settings=settings)

        mock_chat_cls.assert_called_once_with(
            model="glm-4-flash",
            api_key="zhipu-test-key",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            timeout=30,
            max_retries=2,
            temperature=0.3,
        )
        assert client.model == "glm-4-flash"

    def test_init_raises_when_no_api_key(self):
        """LLMError raised when neither openai_api_key nor zhipu_api_key is set."""
        settings = _make_settings(openai_api_key=None, zhipu_api_key=None)
        with pytest.raises(LLMError, match="No API key configured"):
            LLMClient(settings=settings)

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_init_openai_takes_priority_over_zhipu(self, mock_chat_cls):
        """When both keys are present, OpenAI is preferred."""
        settings = _make_settings(
            openai_api_key="sk-openai",
            zhipu_api_key="zhipu-key",
        )
        client = LLMClient(settings=settings)

        call_kwargs = mock_chat_cls.call_args
        assert call_kwargs[1]["api_key"] == "sk-openai"
        assert call_kwargs[1]["model"] == "gpt-4o"


# ── invoke() tests ──────────────────────────────────────────────────────────


class TestLLMClientInvoke:
    """LLMClient.invoke — basic chat completion."""

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_invoke_sends_system_and_user_messages(self, mock_chat_cls):
        """invoke() sends a SystemMessage + HumanMessage and returns content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Analysis complete."
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        client = LLMClient(settings=_make_settings())
        result = client.invoke("You are an analyst.", "Analyze 300750.")

        assert result == "Analysis complete."
        # Verify messages passed to llm.invoke
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].content == "You are an analyst."
        assert call_args[1].content == "Analyze 300750."

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_invoke_wraps_exceptions_as_llm_error(self, mock_chat_cls):
        """Any exception from the underlying LLM is wrapped in LLMError."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Network unreachable")
        mock_chat_cls.return_value = mock_llm

        client = LLMClient(settings=_make_settings())
        with pytest.raises(LLMError, match="LLM invoke failed"):
            client.invoke("system", "user")

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_invoke_passes_through_rate_limit_error(self, mock_chat_cls):
        """Rate-limit errors are wrapped in LLMError (not swallowed)."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Rate limit exceeded")
        mock_chat_cls.return_value = mock_llm

        client = LLMClient(settings=_make_settings())
        with pytest.raises(LLMError, match="Rate limit"):
            client.invoke("s", "u")


# ── structured_output() tests ──────────────────────────────────────────────


class TestLLMClientStructuredOutput:
    """LLMClient.structured_output — Pydantic schema enforcement."""

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_structured_output_returns_schema_instance(self, mock_chat_cls):
        """structured_output() returns an instance of the provided Pydantic schema."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str
            score: float

        expected = TestSchema(answer="yes", score=0.95)

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = expected

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_chat_cls.return_value = mock_llm

        client = LLMClient(settings=_make_settings())
        result = client.structured_output("sys", "user", TestSchema)

        assert isinstance(result, TestSchema)
        assert result.answer == "yes"
        assert result.score == 0.95

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_structured_output_passes_schema_to_with_structured_output(self, mock_chat_cls):
        """structured_output() calls with_structured_output with the schema class."""
        from pydantic import BaseModel

        class Dummy(BaseModel):
            x: int

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = Dummy(x=1)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_chat_cls.return_value = mock_llm

        client = LLMClient(settings=_make_settings())
        client.structured_output("s", "u", Dummy)

        mock_llm.with_structured_output.assert_called_once_with(Dummy)

    @patch("quant_agent.llm.client.ChatOpenAI")
    def test_structured_output_wraps_parse_failure(self, mock_chat_cls):
        """If the LLM returns unparseable data, LLMError is raised."""
        from pydantic import BaseModel

        class StrictSchema(BaseModel):
            value: int

        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = ValueError("bad schema")
        mock_chat_cls.return_value = mock_llm

        client = LLMClient(settings=_make_settings())
        with pytest.raises(LLMError, match="LLM structured_output failed"):
            client.structured_output("s", "u", StrictSchema)


# ── get_llm_client() singleton tests ────────────────────────────────────────


class TestGetLLMClient:
    """get_llm_client — singleton factory."""

    def test_get_llm_client_is_cached(self):
        """get_llm_client() returns the same instance on repeated calls (lru_cache)."""
        # Clear the cache first so we get a clean test
        get_llm_client.cache_clear()

        with patch("quant_agent.llm.client.ChatOpenAI"):
            with patch("quant_agent.llm.client.get_settings") as mock_gs:
                mock_gs.return_value = _make_settings()
                client1 = get_llm_client()
                client2 = get_llm_client()

        assert client1 is client2
        # Clean up for other tests
        get_llm_client.cache_clear()

    def test_get_llm_client_returns_llm_client_instance(self):
        """get_llm_client() returns an actual LLMClient."""
        get_llm_client.cache_clear()

        with patch("quant_agent.llm.client.ChatOpenAI"):
            with patch("quant_agent.llm.client.get_settings") as mock_gs:
                mock_gs.return_value = _make_settings()
                client = get_llm_client()

        assert isinstance(client, LLMClient)
        get_llm_client.cache_clear()
