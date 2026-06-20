"""NeuralPlane Phase 1C — OpenAI-compatible LLM Provider.

Uses ONLY stdlib ``urllib.request`` to call an OpenAI-compatible chat completions
endpoint.  No external packages (e.g. openai PyPI) are required.

Environment variables
---------------------
``LLM_API_KEY``        Required. Your API key for the LLM provider.
``LLM_BASE_URL``       Base URL for the API. Defaults to
                       "https://api.openai.com/v1".
``LLM_MODEL``          Model name. Defaults to "gpt-4o".
``LLM_TEMPERATURE``    Sampling temperature (0.0–2.0). Defaults to 0.3.
``LLM_MAX_TOKENS``     Max tokens to generate. Defaults to 4096.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import urllib.request
except ImportError:  # pragma: no cover
    raise ImportError("OpenAIProvider requires the stdlib urllib.request module.")

from neuralplane.ai_parser import ParseError


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------


class ConfigurationError(ValueError):
    """Raised when the provider is misconfigured (e.g. missing API key)."""

    pass


class HTTPError(RuntimeError):
    """Raised when the LLM API returns a non-2xx HTTP response."""

    pass


# ----------------------------------------------------------------------
# OpenAIProvider
# ----------------------------------------------------------------------


class OpenAIProvider:
    """OpenAI-compatible LLM provider using stdlib ``urllib.request``.

    Parameters
    ----------
    api_key:
        API key for the provider. If omitted, read from the ``LLM_API_KEY``
        environment variable. Raises ``ConfigurationError`` if neither is
        available.
    model:
        Model name to use. Defaults to "gpt-4o". Can also be configured via
        the ``LLM_MODEL`` environment variable.
    base_url:
        Base URL for the chat completions endpoint (without the trailing
        "/chat/completions" path). Defaults to "https://api.openai.com/v1".
        Can also be configured via the ``LLM_BASE_URL`` environment variable.
    temperature:
        Sampling temperature in the range 0.0–2.0. Values outside this range
        are clamped. Defaults to 0.3. Can also be configured via the
        ``LLM_TEMPERATURE`` environment variable.
    max_tokens:
        Maximum number of tokens to generate. Defaults to 4096.
        Can also be configured via the ``LLM_MAX_TOKENS`` environment variable.
    max_retries:
        Number of times to retry on malformed JSON or missing ``choices`` in
        the response. Defaults to 3.
    timeout:
        HTTP request timeout in seconds. Defaults to 60.

    Raises
    ------
    ConfigurationError
        If no API key is available (neither argument nor environment variable).
    ParseError
        If the response body cannot be parsed as JSON or is missing the
        ``choices`` field after all retries are exhausted.
    HTTPError
        If the LLM API returns a non-2xx status code.
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _DEFAULT_MODEL = "gpt-4o"
    _DEFAULT_TEMPERATURE = 0.3
    _DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        # API key — argument takes precedence over env var
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if not self._api_key:
            raise ConfigurationError(
                "No LLM API key provided. Set the LLM_API_KEY environment variable "
                "or pass api_key='your-key' to OpenAIProvider."
            )

        self._model = model or os.environ.get("LLM_MODEL", self._DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get(
            "LLM_BASE_URL", self._DEFAULT_BASE_URL
        )

        # Temperature — read from env or argument, with clamping
        temp_arg = temperature if temperature is not None else None
        temp_env = os.environ.get("LLM_TEMPERATURE")
        if temp_arg is not None:
            raw_temp = temp_arg
        elif temp_env is not None:
            try:
                raw_temp = float(temp_env)
            except ValueError:
                raw_temp = self._DEFAULT_TEMPERATURE
        else:
            raw_temp = self._DEFAULT_TEMPERATURE
        self._temperature = max(0.0, min(2.0, raw_temp))

        # Max tokens — read from env or argument
        tokens_arg = max_tokens if max_tokens is not None else None
        tokens_env = os.environ.get("LLM_MAX_TOKENS")
        if tokens_arg is not None:
            self._max_tokens = tokens_arg
        elif tokens_env is not None:
            try:
                self._max_tokens = int(tokens_env)
            except ValueError:
                self._max_tokens = self._DEFAULT_MAX_TOKENS
        else:
            self._max_tokens = self._DEFAULT_MAX_TOKENS

        self._max_retries = max_retries
        self._timeout = timeout

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """Call the LLM and return its raw text response.

        Parameters
        ----------
        prompt:
            The input prompt string.

        Returns
        -------
        str
            The raw content string from the LLM response ``choices[0].message.content``.

        Raises
        ------
        ConfigurationError
            If no API key was configured.
        HTTPError
            If the LLM API returns a non-2xx status code.
        ParseError
            If the response is not valid JSON or is missing the ``choices`` field
            after all retries are exhausted.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        body_bytes = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self._base_url.rstrip('/')}/chat/completions",
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        last_exc: Exception = RuntimeError("unknown error")

        for attempt in range(self._max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    status = response.status
                    if not (200 <= status < 300):
                        raise HTTPError(
                            f"LLM API returned HTTP {status} ({response.reason}). "
                            f"URL: {request.full_url}"
                        )
                    response_body = response.read().decode("utf-8")
                    return self._extract_content(response_body)

            except HTTPError:
                # Non-2xx — do not retry, propagate immediately
                raise
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, ParseError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    continue
                # retries exhausted
                raise ParseError(
                    f"LLM response is not valid JSON or is missing 'choices' "
                    f"after {self._max_retries + 1} attempts: {exc}"
                ) from exc

        # Should not reach here, but defensive
        raise ParseError(f"LLM response parse failed after {self._max_retries + 1} attempts") from last_exc  # pragma: no cover

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_content(self, response_body: str) -> str:
        """Extract ``content`` from the first ``choices[0].message`` in a
        chat completions response body.

        Parameters
        ----------
        response_body:
            Raw JSON string from the API response.

        Returns
        -------
        str
            The content string from the first choice.

        Raises
        ------
        ParseError
            If the body is not valid JSON or is missing the required structure.
        """
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Response body is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ParseError(
                f"Response root must be a JSON object, got {type(data).__name__}"
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ParseError(
                "Response 'choices' is missing or empty. "
                "Expected a non-empty list at data.choices."
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ParseError(
                f"First choice must be a JSON object, got {type(first_choice).__name__}"
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ParseError(
                f"Choice 'message' must be a JSON object, got {type(message).__name__}"
            )

        content = message.get("content")
        if not isinstance(content, str):
            raise ParseError(
                f"Message 'content' must be a string, got {type(content).__name__}"
            )

        return content