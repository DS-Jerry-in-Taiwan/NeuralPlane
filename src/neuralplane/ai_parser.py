"""NeuralPlane Phase 1A — AI Parser Pipeline.

Turns a natural-language request into a validated ``TaskDraft`` using an
injectable LLM provider (Phase 1A uses only a static/deterministic provider).

Pipeline::

    user_request
      → build_task_draft_prompt()
      → LLMProvider.complete()
      → extract_json_object()
      → normalize_task_draft_dict()
      → TaskDraft.from_dict()
      → TaskDraft.validate()
      → TaskDraft

No external API calls are made in this module; the LLMProvider is fully
injectable for testing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from neuralplane.models import TaskDraft, ValidationError
from neuralplane.prompt_templates import build_task_draft_prompt


# ----------------------------------------------------------------------
# LLMProvider interface
# ----------------------------------------------------------------------


class LLMProvider(Protocol):
    """Minimal protocol for an LLM that returns text completions.

    Implementations can be real (OpenAI, Anthropic, …) or fake/static for
    testing.  Phase 1A ships with ``StaticLLMProvider``.
    """

    def complete(self, prompt: str) -> str:
        """Return the LLM's raw text response to ``prompt``."""
        ...


# ----------------------------------------------------------------------
# StaticLLMProvider (testing-only)
# ----------------------------------------------------------------------


class StaticLLMProvider:
    """A deterministic, no-network LLMProvider that always returns the same
    pre-configured text.

    Parameters
    ----------
    response_text:
        The exact string this provider will return on every call to
        ``complete()``.  Intended to be a JSON fixture in tests.
    """

    def __init__(self, response_text: str) -> None:
        self._response = response_text

    def complete(self, prompt: str) -> str:
        """Return the pre-configured response text (prompt is ignored)."""
        return self._response


# ----------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------


class ParseError(Exception):
    """Raised when the provider response cannot be parsed as a TaskDraft JSON."""

    pass


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from provider output.

    Parameters
    ----------
    text:
        Raw provider output. May be:
        - A bare JSON object: ``{...}``
        - A JSON object wrapped in a markdown code fence: `````json\n{...}\n``` ``` ``

    Returns
    -------
    dict[str, Any]
        The parsed JSON object.

    Raises
    ------
    ParseError
        If ``text`` does not contain a parseable JSON object (e.g. it is a
        JSON array, a primitive, or not valid JSON at all).
    """
    # Try bare JSON object first.
    text = text.strip()

    # Strip markdown code fence if present (e.g. ```json\n{...}\n```).
    # Match from opening fence to closing fence.
    fence_match = re.match(
        r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Provider response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ParseError(
            f"Provider response must be a JSON object, "
            f"got {type(parsed).__name__} instead"
        )

    return parsed


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------


def normalize_task_draft_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative defaults to a raw provider dict before constructing
    a ``TaskDraft``.

    This function only fills in missing keys with safe empty defaults; it
    does NOT rewrite or relax any Phase 1B validation rules.

    Parameters
    ----------
    data:
        Raw dict from ``extract_json_object()``.

    Returns
    -------
    dict[str, Any]
        A dict with all required TaskDraft keys present (using safe defaults
        for missing optional fields).
    """
    # Normalise empty-string priority to None so the existing validator
    # (which treats "" as invalid) does not reject it.
    priority = data.get("priority")
    if priority == "":
        priority = None

    return {
        "title": data.get("title", ""),
        "description": data.get("description") or "",
        "dod": data.get("dod") or [],
        "priority": priority,
        "assignee_suggestions": data.get("assignee_suggestions") or [],
        "labels": data.get("labels") or [],
    }


# ----------------------------------------------------------------------
# Main parser entry-point
# ----------------------------------------------------------------------


class ParserInputError(Exception):
    """Raised when the user-supplied natural-language input is invalid."""

    pass


def parse_task_request(user_request: str, provider: LLMProvider) -> TaskDraft:
    """Parse a natural-language request into a validated ``TaskDraft``.

    Parameters
    ----------
    user_request:
        Natural-language description of the task to be performed.
    provider:
        An ``LLMProvider`` instance used to obtain the raw completion text.

    Returns
    -------
    TaskDraft
        A validated TaskDraft derived from the provider's response.

    Raises
    ------
    ParserInputError
        If ``user_request`` is empty/whitespace after trimming.
    ParseError
        If the provider's output cannot be parsed as a JSON object.
    ValidationError
        If the parsed TaskDraft fails ``TaskDraft.validate()``.
    """
    # Step 1: build prompt (may raise ValueError / ParserInputError)
    try:
        prompt = build_task_draft_prompt(user_request)
    except ValueError as exc:
        raise ParserInputError(str(exc)) from exc

    # Step 2: call the provider
    raw_response = provider.complete(prompt)

    # Step 3: extract JSON object
    try:
        json_dict = extract_json_object(raw_response)
    except ParseError:
        raise

    # Step 4: normalise
    normalised = normalize_task_draft_dict(json_dict)

    # Step 5: build TaskDraft
    task = TaskDraft.from_dict(normalised)

    # Step 6: validate (Phase 1B rules applied here)
    task.validate()

    return task
