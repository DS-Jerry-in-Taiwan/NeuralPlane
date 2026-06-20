"""Prompt templates for the NeuralPlane Phase 1A AI Parser.

Provides prompt builders that instruct the LLM to output strict JSON matching
the TaskDraft schema. No provider-specific secrets are included.
"""

from __future__ import annotations

ALLOWED_PRIORITIES_STR = "low, medium, high, urgent"


def build_task_draft_prompt(user_request: str) -> str:
    """Build a strict JSON-only prompt for TaskDraft generation.

    Parameters
    ----------
    user_request:
        Natural-language task description from the user.

    Returns
    -------
    str
        A prompt string instructing the LLM to produce a JSON object.

    Raises
    ------
    ValueError
        If ``user_request`` is empty or whitespace-only after trimming.
    """
    trimmed = user_request.strip()
    if not trimmed:
        raise ValueError("user_request cannot be empty or whitespace-only")

    lines = [
        "You are a task-planning assistant. Given the following user request, "
        "output a single JSON object — no markdown fences, no explanation, "
        "no additional text.",
        "",
        "Output format:",
        "{",
        f'  "title": "<short task title, max {255} characters>",',
        '  "description": "<detailed description of the task background and scope>",',
        "  \"dod\": [",
        '    "<definition-of-done criterion 1 — must be testable/observable>",',
        '    "<definition-of-done criterion 2 — must be testable/observable>",',
        "    ...",
        "  ],",
        f'  "priority": "<one of {ALLOWED_PRIORITIES_STR} or null>",',
        '  "labels": ["<label1>", "<label2>", ...],',
        '  "assignee_suggestions": ["<username or ID>", ...]',
        "}",
        "",
        "Rules:",
        f"  - priority must be one of: {ALLOWED_PRIORITIES_STR}, or null if unspecified.",
        "  - dod must contain at least 1 item; each item must be a concrete, "
        "testable/observable criterion.",
        "  - Do NOT use vague criteria like 'function works correctly' — instead "
        "describe a concrete pass/fail condition.",
        "  - labels and assignee_suggestions may be empty arrays.",
        "  - description may be an empty string.",
        "",
        "User request:",
        trimmed,
    ]

    return "\n".join(lines)
