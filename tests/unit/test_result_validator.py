"""Unit tests for the LLM-based ResultValidator.

These tests exercise the result-validation service with a mocked OpenAI client,
covering the confidence parsing, error translation and disabled-validation paths.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.services.result_validator import ResultValidator


def _make_response(content: str) -> MagicMock:
    """Build a fake ChatCompletion with the given message content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.model_dump.return_value = {"content": content}
    return response


@pytest.fixture
def validator() -> ResultValidator:
    """Create a ResultValidator backed by a mocked OpenAI client."""
    with patch("pg_mcp.services.result_validator.AsyncOpenAI") as mock_cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        mock_cls.return_value = client
        instance = ResultValidator(
            openai_config=OpenAIConfig(api_key="sk-test"),
            validation_config=ValidationConfig(),
        )
    instance.client = client
    return instance


def _set_response(validator: ResultValidator, content: str) -> None:
    """Configure the mocked client to return a fixed JSON payload."""
    validator.client.chat.completions.create = AsyncMock(return_value=_make_response(content))


class TestResultValidator:
    """Test suite for ResultValidator."""

    @pytest.mark.asyncio
    async def test_validation_disabled_skips_api(self, validator: ResultValidator) -> None:
        """When disabled, a high-confidence result is returned without an API call."""
        validator.validation_config = ValidationConfig(enabled=False)
        validator.client.chat.completions.create = AsyncMock()

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"c": 1}], row_count=1
        )

        assert result.confidence == 100
        assert result.is_acceptable is True
        validator.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_validation(self, validator: ResultValidator) -> None:
        """A well-formed JSON response is parsed into the result model."""
        _set_response(
            validator,
            '{"confidence": 88, "explanation": "matches well", "suggestion": null}',
        )

        result = await validator.validate(
            question="How many users?",
            sql="SELECT COUNT(*) FROM users",
            results=[{"c": 10}],
            row_count=1,
        )

        assert result.confidence == 88
        assert result.explanation == "matches well"
        assert result.is_acceptable is True

    @pytest.mark.asyncio
    async def test_low_confidence_is_not_acceptable(self, validator: ResultValidator) -> None:
        """Results below the threshold are marked unacceptable."""
        _set_response(
            validator,
            '{"confidence": 20, "explanation": "wrong table", "suggestion": "use orders"}',
        )

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 20
        assert result.is_acceptable is False
        assert result.suggestion == "use orders"

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_is_clamped(self, validator: ResultValidator) -> None:
        """Confidence values outside 0-100 are clamped."""
        _set_response(validator, '{"confidence": 150, "explanation": "x"}')

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 100

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_defaults(self, validator: ResultValidator) -> None:
        """A non-numeric confidence falls back to 50."""
        _set_response(validator, '{"confidence": "high", "explanation": "x"}')

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 50

    @pytest.mark.asyncio
    async def test_invalid_json_returns_moderate_confidence(
        self, validator: ResultValidator
    ) -> None:
        """Unparseable JSON yields a moderate, unacceptable result."""
        _set_response(validator, "not-json")

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 60
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_defaults_for_missing_fields(self, validator: ResultValidator) -> None:
        """Missing fields fall back to documented defaults."""
        _set_response(validator, "{}")

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 50
        assert result.explanation == "No explanation provided"

    @pytest.mark.asyncio
    async def test_empty_choices_raises(self, validator: ResultValidator) -> None:
        """An empty choices list raises LLMError."""
        response = MagicMock()
        response.choices = []
        response.model_dump.return_value = {}
        validator.client.chat.completions.create = AsyncMock(return_value=response)

        with pytest.raises(LLMError, match="empty response"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_empty_content_raises(self, validator: ResultValidator) -> None:
        """An empty message body raises LLMError."""
        _set_response(validator, "")

        with pytest.raises(LLMError, match="empty message content"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_timeout_maps_to_llm_timeout(self, validator: ResultValidator) -> None:
        """A timeout is translated to LLMTimeoutError."""
        validator.client.chat.completions.create = AsyncMock(side_effect=TimeoutError())

        with pytest.raises(LLMTimeoutError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_rate_limit_maps_to_unavailable(self, validator: ResultValidator) -> None:
        """Rate limiting is translated to LLMUnavailableError."""
        validator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("rate_limit exceeded")
        )

        with pytest.raises(LLMUnavailableError, match="rate limit"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_auth_error_maps_to_unavailable(self, validator: ResultValidator) -> None:
        """Authentication failures are translated to LLMUnavailableError."""
        validator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("authentication failed")
        )

        with pytest.raises(LLMUnavailableError, match="authentication"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_unexpected_error_maps_to_llm_error(self, validator: ResultValidator) -> None:
        """Other failures are translated to LLMError."""
        validator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("something else broke")
        )

        with pytest.raises(LLMError, match="Result validation failed"):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_results_are_sampled(self, validator: ResultValidator) -> None:
        """Only sample_rows rows are sent to the LLM."""
        _set_response(validator, '{"confidence": 90, "explanation": "ok"}')
        validator.validation_config = ValidationConfig(sample_rows=2)
        rows = [{"i": i} for i in range(10)]

        await validator.validate(question="q", sql="SELECT 1", results=rows, row_count=10)

        create_call = validator.client.chat.completions.create.await_args
        prompt = create_call.kwargs["messages"][1]["content"]
        assert '"i": 2' not in prompt
        assert '"i": 1' in prompt
