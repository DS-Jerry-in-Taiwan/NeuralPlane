"""NeuralPlane Phase 1C — Pluggable LLM Providers.

This package contains LLM provider implementations that satisfy the
``LLMProvider`` protocol defined in ``neuralplane.ai_parser``.

Currently shipped
-----------------
``OpenAIProvider`` — OpenAI-compatible API provider using stdlib ``urllib``.
"""

from __future__ import annotations

from neuralplane.providers.openai_provider import OpenAIProvider

__all__ = ["OpenAIProvider"]