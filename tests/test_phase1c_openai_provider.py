"""Unit tests for NeuralPlane Phase 1C OpenAI Provider.

Covers:
  - OpenAIProvider: API key resolution (env / explicit / missing)
  - OpenAIProvider: model resolution (env / explicit / default)
  - OpenAIProvider: temperature clamping (0.0–2.0)
  - OpenAIProvider: max_tokens from env / explicit / default
  - OpenAIProvider.complete(): happy path (mock HTTP 200 + valid response)
  - OpenAIProvider.complete(): HTTP error (mock 401 → RuntimeError subclass HTTPError)
  - OpenAIProvider.complete(): HTTP server error (mock 500 → RuntimeError subclass HTTPError)
  - OpenAIProvider.complete(): malformed JSON → ParseError after retries exhausted
  - OpenAIProvider.complete(): retry then succeed (fail 2 times, 3rd OK)
  - OpenAIProvider.complete(): retry exhausted (fail 3 times → ParseError)
  - OpenAIProvider.complete(): empty response → ParseError
  - CLI integration: openai provider with mock, full pipeline
  - CLI: missing API key → FATAL exit

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure src/neuralplane/ is on the path for direct imports.
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from neuralplane.ai_parser import ParseError
from neuralplane.providers.openai_provider import (
    ConfigurationError,
    HTTPError,
    OpenAIProvider,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_mock_response(
    status: int,
    body: dict[str, Any] | None = None,
    reason: str = "OK",
) -> MagicMock:
    """Return a mock context-manager that behaves like urllib.response addinfourl."""
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.reason = reason
    mock_response.read.return_value = (
        json.dumps(body).encode("utf-8") if body is not None else b""
    )
    # Must explicitly set __enter__ to return self, otherwise with-statement
    # creates a new MagicMock instead of using our configured mock_response.
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


def _valid_completion_response(content: str = '{"title": "Test", "dod": ["Done"]}') -> dict[str, Any]:
    """Return a minimal valid chat completions response body."""
    return {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


# =====================================================================
# Configuration tests — API key
# =====================================================================
class TestOpenAIProviderAPIKey(unittest.TestCase):
    """🔴 負面測試 / 📏 範圍測試: API key resolution."""

    def test_missing_api_key_raises_configuration_error(self) -> None:
        """No API key argument and no LLM_API_KEY env var → ConfigurationError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                OpenAIProvider()
        self.assertIn("LLM_API_KEY", str(ctx.exception))

    def test_api_key_from_env(self) -> None:
        """LLM_API_KEY env var is used when no explicit key is passed."""
        with patch.dict(os.environ, {"LLM_API_KEY": "env-key-123"}, clear=False):
            provider = OpenAIProvider()
        self.assertEqual(provider._api_key, "env-key-123")

    def test_api_key_explicit_overrides_env(self) -> None:
        """Explicit api_key argument takes precedence over env var."""
        with patch.dict(os.environ, {"LLM_API_KEY": "env-key-456"}, clear=False):
            provider = OpenAIProvider(api_key="explicit-key-789")
        self.assertEqual(provider._api_key, "explicit-key-789")

    def test_empty_string_api_key_raises(self) -> None:
        """Empty string API key (even if in env) → ConfigurationError."""
        with patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
            with self.assertRaises(ConfigurationError):
                OpenAIProvider()


# =====================================================================
# Configuration tests — model
# =====================================================================
class TestOpenAIProviderModel(unittest.TestCase):
    """📏 範圍測試: model resolution."""

    def test_default_model(self) -> None:
        """No model argument / env var → default gpt-4o."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        self.assertEqual(provider._model, "gpt-4o")

    def test_model_from_env(self) -> None:
        """LLM_MODEL env var is used when no explicit model is passed."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_MODEL": "gpt-4o-mini"}, clear=False
        ):
            provider = OpenAIProvider()
        self.assertEqual(provider._model, "gpt-4o-mini")

    def test_model_explicit_overrides_env(self) -> None:
        """Explicit model argument takes precedence over env var."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_MODEL": "gpt-4o-mini"}, clear=False
        ):
            provider = OpenAIProvider(model="explicit-model")
        self.assertEqual(provider._model, "explicit-model")


# =====================================================================
# Configuration tests — temperature
# =====================================================================
class TestOpenAIProviderTemperature(unittest.TestCase):
    """📏 範圍測試 / 🎯 正確性測試: temperature clamping."""

    def test_default_temperature(self) -> None:
        """No temperature argument / env var → default 0.3."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        self.assertEqual(provider._temperature, 0.3)

    def test_temperature_from_env(self) -> None:
        """LLM_TEMPERATURE env var is used when no explicit temperature is passed."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_TEMPERATURE": "0.7"}, clear=False
        ):
            provider = OpenAIProvider()
        self.assertEqual(provider._temperature, 0.7)

    def test_temperature_explicit_overrides_env(self) -> None:
        """Explicit temperature argument takes precedence over env var."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_TEMPERATURE": "0.7"}, clear=False
        ):
            provider = OpenAIProvider(temperature=0.9)
        self.assertEqual(provider._temperature, 0.9)

    def test_temperature_clamp_below_zero(self) -> None:
        """Temperature below 0.0 is clamped to 0.0."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider(temperature=-0.5)
        self.assertEqual(provider._temperature, 0.0)

    def test_temperature_clamp_above_two(self) -> None:
        """Temperature above 2.0 is clamped to 2.0."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider(temperature=3.0)
        self.assertEqual(provider._temperature, 2.0)

    def test_temperature_clamp_at_boundary(self) -> None:
        """Boundary values 0.0 and 2.0 are accepted unchanged."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            low = OpenAIProvider(temperature=0.0)
            high = OpenAIProvider(temperature=2.0)
        self.assertEqual(low._temperature, 0.0)
        self.assertEqual(high._temperature, 2.0)

    def test_temperature_invalid_env_value_falls_back_to_default(self) -> None:
        """Non-numeric LLM_TEMPERATURE env var → default 0.3."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_TEMPERATURE": "not-a-number"}, clear=False
        ):
            provider = OpenAIProvider()
        self.assertEqual(provider._temperature, 0.3)


# =====================================================================
# Configuration tests — max_tokens
# =====================================================================
class TestOpenAIProviderMaxTokens(unittest.TestCase):
    """📏 範圍測試: max_tokens resolution."""

    def test_default_max_tokens(self) -> None:
        """No max_tokens argument / env var → default 4096."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        self.assertEqual(provider._max_tokens, 4096)

    def test_max_tokens_from_env(self) -> None:
        """LLM_MAX_TOKENS env var is used when no explicit max_tokens is passed."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_MAX_TOKENS": "2048"}, clear=False
        ):
            provider = OpenAIProvider()
        self.assertEqual(provider._max_tokens, 2048)

    def test_max_tokens_explicit_overrides_env(self) -> None:
        """Explicit max_tokens argument takes precedence over env var."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_MAX_TOKENS": "2048"}, clear=False
        ):
            provider = OpenAIProvider(max_tokens=1024)
        self.assertEqual(provider._max_tokens, 1024)

    def test_max_tokens_invalid_env_value_falls_back_to_default(self) -> None:
        """Non-numeric LLM_MAX_TOKENS env var → default 4096."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_MAX_TOKENS": "not-a-number"}, clear=False
        ):
            provider = OpenAIProvider()
        self.assertEqual(provider._max_tokens, 4096)


# =====================================================================
# complete() — happy path
# =====================================================================
class TestOpenAIProviderCompleteHappyPath(unittest.TestCase):
    """🟢 正面測試 / 🎯 正確性測試: valid HTTP 200 response."""

    @patch("urllib.request.urlopen")
    def test_happy_path_returns_content(self, mock_urlopen: MagicMock) -> None:
        """HTTP 200 with valid JSON body → returns content string."""
        response_body = _valid_completion_response(
            content='{"title": "Login API", "dod": ["Done"]}'
        )
        mock_urlopen.return_value = _make_mock_response(200, response_body)

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        result = provider.complete("build login API")

        self.assertEqual(result, '{"title": "Login API", "dod": ["Done"]}')
        mock_urlopen.assert_called_once()
        call_kwargs = mock_urlopen.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 60.0)

    @patch("urllib.request.urlopen")
    def test_request_body_is_correct(self, mock_urlopen: MagicMock) -> None:
        """Request body contains model, messages, temperature, max_tokens."""
        response_body = _valid_completion_response()
        mock_urlopen.return_value = _make_mock_response(200, response_body)

        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_MODEL": "gpt-4o-mini", "LLM_TEMPERATURE": "0.5"},
            clear=False,
        ):
            provider = OpenAIProvider(model="my-model", temperature=0.8, max_tokens=1000)
        provider.complete("hello")

        call_args = mock_urlopen.call_args
        request: Any = call_args[0][0]  # positional arg = request object
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "my-model")
        self.assertEqual(body["temperature"], 0.8)
        self.assertEqual(body["max_tokens"], 1000)
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])

    @patch("urllib.request.urlopen")
    def test_authorization_header_set(self, mock_urlopen: MagicMock) -> None:
        """Authorization header uses Bearer token with the API key."""
        response_body = _valid_completion_response()
        mock_urlopen.return_value = _make_mock_response(200, response_body)

        with patch.dict(os.environ, {"LLM_API_KEY": "secret-abc123"}, clear=False):
            provider = OpenAIProvider()
        provider.complete("test")

        call_args = mock_urlopen.call_args
        request: Any = call_args[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-abc123")


# =====================================================================
# complete() — HTTP errors
# =====================================================================
class TestOpenAIProviderHTTPError(unittest.TestCase):
    """🔴 負面測試 / 🎯 正確性測試: HTTP error handling."""

    @patch("urllib.request.urlopen")
    def test_401_unauthorized_raises_http_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 401 → HTTPError (subclass of RuntimeError); no retry."""
        mock_urlopen.return_value = _make_mock_response(401, reason="Unauthorized")

        with patch.dict(os.environ, {"LLM_API_KEY": "bad-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(HTTPError) as ctx:
            provider.complete("test")
        self.assertIn("401", str(ctx.exception))
        # Only one attempt (no retry on 4xx)
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("urllib.request.urlopen")
    def test_500_server_error_raises_http_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 500 → HTTPError (subclass of RuntimeError); no retry."""
        mock_urlopen.return_value = _make_mock_response(500, reason="Internal Server Error")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(HTTPError) as ctx:
            provider.complete("test")
        self.assertIn("500", str(ctx.exception))
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("urllib.request.urlopen")
    def test_429_rate_limit_raises_http_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 429 → HTTPError; no retry."""
        mock_urlopen.return_value = _make_mock_response(429, reason="Too Many Requests")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(HTTPError):
            provider.complete("test")


# =====================================================================
# complete() — retry and parse error
# =====================================================================
class TestOpenAIProviderRetryAndParseError(unittest.TestCase):
    """🔴 負面測試 / 📏 範圍測試 / 🎯 正確性測試: retry logic and ParseError."""

    @patch("urllib.request.urlopen")
    def test_malformed_json_on_first_attempt_succeeds_on_second(
        self, mock_urlopen: MagicMock
    ) -> None:
        """First call returns invalid JSON; second call succeeds → returns content."""
        fail_response = _make_mock_response(200, {"choices": []})  # missing message content
        success_body = _valid_completion_response(content='{"title": "Fixed"}')
        success_response = _make_mock_response(200, success_body)

        # Return fail once, then success
        mock_urlopen.side_effect = [fail_response, success_response]

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider(max_retries=3)
        result = provider.complete("test")

        self.assertEqual(result, '{"title": "Fixed"}')
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_malformed_json_retry_exhausted_raises_parse_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """All 3 attempts return malformed JSON → ParseError after 4th try (3+1)."""
        # Return malformed response every time (choices missing content)
        malformed = _make_mock_response(200, {"choices": [{"message": {}}]})
        mock_urlopen.return_value = malformed

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider(max_retries=3)
        with self.assertRaises(ParseError) as ctx:
            provider.complete("test")

        self.assertIn("choices", str(ctx.exception).lower())
        # 4 total calls: initial + 3 retries
        self.assertEqual(mock_urlopen.call_count, 4)

    @patch("urllib.request.urlopen")
    def test_empty_response_body_raises_parse_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 200 but empty body → ParseError."""
        mock_urlopen.return_value = _make_mock_response(200, None)

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(ParseError):
            provider.complete("test")

    @patch("urllib.request.urlopen")
    def test_non_json_response_raises_parse_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 200 but body is not JSON → ParseError."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.reason = "OK"
        mock_response.read.return_value = b"not json at all"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(ParseError):
            provider.complete("test")

    @patch("urllib.request.urlopen")
    def test_choices_not_a_list_raises_parse_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 200 but choices is not a list → ParseError."""
        mock_urlopen.return_value = _make_mock_response(200, {"choices": "not a list"})

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(ParseError):
            provider.complete("test")

    @patch("urllib.request.urlopen")
    def test_empty_choices_list_raises_parse_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 200 but choices is an empty list → ParseError."""
        mock_urlopen.return_value = _make_mock_response(200, {"choices": []})

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(ParseError):
            provider.complete("test")

    @patch("urllib.request.urlopen")
    def test_message_missing_content_raises_parse_error(self, mock_urlopen: MagicMock) -> None:
        """HTTP 200 but message.content is missing → ParseError."""
        body = {
            "choices": [
                {"message": {"role": "assistant"}}  # missing content
            ]
        }
        mock_urlopen.return_value = _make_mock_response(200, body)

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
        with self.assertRaises(ParseError):
            provider.complete("test")


# =====================================================================
# CLI integration tests
# =====================================================================
class TestCLIOpenAIIntegration(unittest.TestCase):
    """🟢 正面測試 / 🔴 負面測試 / 🎯 正確性測試: CLI with openai provider."""

    @patch("urllib.request.urlopen")
    def test_cli_e2e_openai_happy_path(self, mock_urlopen: MagicMock) -> None:
        """CLI with --provider openai and mocked valid response → valid JSON report."""
        response_body = _valid_completion_response(
            content=json.dumps(
                {
                    "title": "登入 API",
                    "description": "JWT auth",
                    "dod": ["done"],
                    "priority": "high",
                    "labels": [],
                    "assignee_suggestions": [],
                },
                ensure_ascii=False,
            )
        )
        mock_urlopen.return_value = _make_mock_response(200, response_body)

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key-openai"}, clear=False):
            from neuralplane.cli_e2e_dry_run import main as e2e_main

            # Capture stdout
            with patch("sys.stdout", new_callable=MagicMock) as mock_stdout:
                with patch("sys.stderr", new_callable=MagicMock) as mock_stderr:
                    with patch("sys.argv", [
                        "neuralplane-e2e-dry-run",
                        "--input", "examples/task_request.sample.txt",
                        "--provider", "openai",
                    ]):
                        returncode = e2e_main()

        self.assertEqual(returncode, 0)

    def test_cli_parse_task_openai_missing_api_key_fatal(self) -> None:
        """--provider openai without LLM_API_KEY → _fatal called (exits with code 1)."""
        with patch.dict(os.environ, {}, clear=True):
            from neuralplane.cli_parse_task_draft import main as parse_main

            with patch("neuralplane.cli_parse_task_draft.load_dotenv") as mock_dotenv:
                with patch("sys.stderr", new_callable=MagicMock) as mock_stderr:
                    with patch("sys.argv", [
                        "neuralplane-parse-task-draft",
                        "--input", "examples/task_request.sample.txt",
                        "--provider", "openai",
                    ]):
                        with self.assertRaises(SystemExit) as ctx:
                            parse_main()
        self.assertEqual(ctx.exception.code, 1)


# =====================================================================
# E2E pipeline with OpenAI provider
# =====================================================================
class TestE2EPipelineWithOpenAIProvider(unittest.TestCase):
    """🟢 正面測試: full pipeline (parse_task_request + E2E report) with mocked provider."""

    @patch("urllib.request.urlopen")
    def test_e2e_pipeline_with_mocked_openai(self, mock_urlopen: MagicMock) -> None:
        """Mocked OpenAI provider + E2E pipeline → valid E2EDryRunReport."""
        response_body = _valid_completion_response(
            content=json.dumps(
                {
                    "title": "建立會員登入 API",
                    "description": "bcrypt + JWT",
                    "dod": ["bcrypt驗證", "JWT 24h"],
                    "priority": "high",
                    "labels": ["auth"],
                    "assignee_suggestions": [],
                },
                ensure_ascii=False,
            )
        )
        mock_urlopen.return_value = _make_mock_response(200, response_body)

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            from neuralplane.e2e_dry_run import build_dry_run_plane_config, run_e2e_dry_run
            from neuralplane.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider()
            config = build_dry_run_plane_config()
            user_request = "建立會員登入 API"
            report = run_e2e_dry_run(user_request, provider, config)

        # task_draft is a dict (see E2EDryRunReport.task_draft type annotation)
        self.assertEqual(report.task_draft["title"], "建立會員登入 API")
        self.assertIn("auth", report.task_draft["labels"])
        self.assertEqual(report.dry_run, True)
        self.assertEqual(report.external_api_calls, 0)


if __name__ == "__main__":
    unittest.main()
