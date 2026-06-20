"""HTML renderer for NeuralPlane Phase 1B.

Turns a validated TaskDraft into a Plane-compatible ``description_html`` string
that contains:
  1. The original description (HTML-escaped).
  2. A human-readable [DoD] section rendered as an unordered list.

All user-supplied text (title, description, DoD items) is HTML-escaped before
being embedded so that no injection is possible.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralplane.models import TaskDraft


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def render_description_html(task: TaskDraft) -> str:
    """Render a Plane-compatible ``description_html`` for a TaskDraft.

    The output always contains:
      - An ``<h2>[DoD]</h2>`` anchor (for Phase 3 AI parsing).
      - A ``<ul>`` list with one ``<li>`` per DoD criterion.
      - The description text.

    Parameters
    ----------
    task:
        A validated TaskDraft instance.

    Returns
    -------
    str
        An HTML fragment suitable for use as Plane's ``description_html`` field.
    """
    escaped_description = _escape_html(task.description)
    escaped_title = _escape_html(task.title)

    dod_items_html = "".join(
        f"<li>{_escape_html(item.strip())}</li>"
        for item in task.dod
    )

    # Build the HTML in sections so the [DoD] anchor is always present.
    # If description is empty we still render the DoD section.
    sections: list[str] = []

    if escaped_description:
        sections.append(f"<p>{escaped_description}</p>")

    sections.append("<h2>[DoD]</h2>")
    sections.append(f"<ul>{dod_items_html}</ul>")

    return "\n".join(sections)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _escape_html(text: str) -> str:
    """Escape special HTML characters in ``text``.

    Uses stdlib ``html.escape`` which handles &, <, >, ", and '.
    """
    return html.escape(text, quote=True)
