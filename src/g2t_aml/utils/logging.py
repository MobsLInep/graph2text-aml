"""Rich-backed structured logging for pipeline scripts.

Two things every pipeline entrypoint owes the log: the configuration it started from and
a summary with elapsed time when it finishes. `stage` gives you both for free and makes
sure a crash is logged with its traceback before it propagates.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console(stderr=True)
_CONFIGURED = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Install the rich handler on the root logger, once per process.

    Args:
        level: Threshold for the console handler.
        log_file: If given, additionally tee plain-text records to this file, which is
            what run directories keep.
        force: Reconfigure even if logging was already set up in this process.

    Returns:
        The ``g2t_aml`` package logger.

    Raises:
        OSError: If ``log_file`` cannot be opened.
    """
    global _CONFIGURED  # noqa: PLW0603
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return logging.getLogger("g2t_aml")

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(logging.DEBUG)
    console_handler = RichHandler(
        console=_CONSOLE,
        rich_tracebacks=True,
        markup=False,
        show_path=False,
        omit_repeated_times=False,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
        )
        root.addHandler(file_handler)

    _CONFIGURED = True
    return logging.getLogger("g2t_aml")


def get_logger(name: str) -> logging.Logger:
    """Return a package-namespaced logger.

    Args:
        name: Usually ``__name__``. Names outside the package are re-rooted under
            ``g2t_aml`` so a single level change controls all pipeline output.

    Returns:
        The logger.
    """
    if name == "__main__" or not name.startswith("g2t_aml"):
        name = f"g2t_aml.{name.rsplit('.', 1)[-1]}"
    return logging.getLogger(name)


def log_mapping(logger: logging.Logger, title: str, mapping: Mapping[str, Any]) -> None:
    """Log a flat key/value block at INFO level.

    Args:
        logger: Destination logger.
        title: Heading printed above the block.
        mapping: Values to print. Rendered with ``repr`` so strings stay quoted.
    """
    logger.info("%s", title)
    width = max((len(str(k)) for k in mapping), default=0)
    for key, value in mapping.items():
        logger.info("  %-*s = %r", width, key, value)


@contextmanager
def stage(
    name: str, logger: logging.Logger | None = None, **context: Any
) -> Iterator[dict[str, Any]]:
    """Bracket a pipeline stage with start/end logging and elapsed time.

    Args:
        name: Stage name, e.g. ``"extract-facts"``.
        logger: Logger to use; defaults to the package logger.
        **context: Key/values logged at stage start, typically resolved config values.

    Yields:
        A mutable dict. Anything the body puts into it is logged as the end-of-stage
        summary, so a stage can report counts it only learns while running.

    Raises:
        Exception: Re-raises whatever the body raised, after logging the failure and
            the elapsed time.
    """
    log = logger or logging.getLogger("g2t_aml")
    summary: dict[str, Any] = {}
    log.info("=== START %s ===", name)
    if context:
        log_mapping(log, "config:", context)
    started = time.perf_counter()
    try:
        yield summary
    except Exception:
        log.exception("=== FAILED %s after %.2fs ===", name, time.perf_counter() - started)
        raise
    else:
        elapsed = time.perf_counter() - started
        summary.setdefault("elapsed_seconds", round(elapsed, 3))
        log_mapping(log, "summary:", summary)
        log.info("=== END %s (%.2fs) ===", name, elapsed)
