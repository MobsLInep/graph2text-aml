"""The browser-side half of the timer: the only place that can see a tab lose visibility.

Two clocks run during an item, and this is the one that is right. The server-side
:class:`~g2t_aml.human.study_ui.BlurAwareTimer` counts wall-clock time between the render
and the submission, and it cannot tell a rater reading carefully from a rater who switched
to their inbox — Streamlit does not re-run when a tab is hidden, so nothing reaches Python
at all. ``visibilitychange`` fires in the browser and nowhere else.

That matters more here than it would in most instrumentation, because time-to-usable-draft
is one of the two measurements the whole study is built around. It is compared across
systems as a paired difference against the Bronze template baseline, and a single item
carrying a fifteen-minute interruption can outweigh the effect being measured across a
whole arm.

The component is one static HTML file talking the Streamlit postMessage protocol directly.
No build step, no node, no bundled JavaScript in the lockfile: ``index.html`` is the whole
implementation and is readable in a text editor, which for a released artifact is worth
more than the ergonomics of a framework.

**When it fails, the study does not.** If Streamlit is absent, or the component fails to
register, :func:`visibility_timer` returns ``None`` and the caller falls back to the server
clock — recording ``timing_source="server"`` on every affected response. That flag is the
point: a set of times collected without visibility tracking is still usable, but it means
something different, and the analysis reports the two sources separately rather than
pooling them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["COMPONENT_NAME", "component_path", "visibility_timer"]

#: The registered component name. Namespaced because Streamlit's component registry is
#: global to the process and a collision would silently render the wrong iframe.
COMPONENT_NAME = "g2t_aml_visibility_timer"


def component_path() -> Path:
    """Return the directory holding the component's ``index.html``.

    Returns:
        The path. Exists in the installed package as well as in the source tree, which is
        why the HTML lives beside this module rather than under a top-level ``assets/``.
    """
    return Path(__file__).parent


@lru_cache(maxsize=1)
def _declare() -> Any:
    """Register the component with Streamlit once per process.

    Cached rather than assigned to a module global because Streamlit's component registry
    rejects a second declaration of the same name, and the Streamlit script re-executes
    top to bottom on every interaction.

    Returns:
        The component function, or None when Streamlit is not installed or registration
        fails. Never raises: a broken timer must degrade to the server clock rather than
        end a rater's session.
    """
    try:
        import streamlit.components.v1 as components

        return components.declare_component(COMPONENT_NAME, path=str(component_path()))
    except Exception:  # any failure here means "fall back to the server clock"
        return None


def visibility_timer(key: str) -> dict[str, Any] | None:
    """Render the timer and return its latest reading.

    Args:
        key: Streamlit widget key. Must be unique per item, so that advancing to the next
            item resets the accumulator rather than carrying the previous item's elapsed
            time into it.

    Returns:
        A mapping with ``active_ms``, ``hidden_ms`` and ``n_blurs``, or None when the
        component is unavailable or has not yet reported. None is the caller's signal to
        record the server clock and mark the response ``timing_source="server"``.
    """
    component = _declare()
    if component is None:
        return None
    try:
        value = component(key=key, default=None)
    except Exception:  # see _declare
        return None
    if not isinstance(value, dict):
        return None
    return value
