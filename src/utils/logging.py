"""Logging setup: readable console output plus JSON-lines rotating file logs."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Attribute names every LogRecord has; anything else on a record came from ``extra={...}``.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_CONFIGURED_MARKER = "_imsdb_handler"


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra ``logger.info(..., extra={...})`` keys are kept."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record as a single-line JSON object.

        Always includes ``ts`` (ISO-8601 UTC, milliseconds), ``level``, ``logger``, ``msg`` and ``thread``;
        adds every non-reserved, non-underscore attribute passed via ``extra``, and ``exc`` with the
        formatted traceback when the record carries exception info. Non-JSON values are ``str()``-ed.

        Args:
            record: The record emitted by the logging framework.

        Returns:
            The JSON text (no trailing newline; the handler adds it).
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "thread": record.threadName,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable one-line console format: ``HH:MM:SS LEVEL   logger: message``."""

    def __init__(self) -> None:
        """Configure the fixed console format string and time-only date format."""
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")


def setup_logging(log_dir: Path, level: str = "INFO", *, console: bool = True) -> None:
    """Configure the ``imsdb`` logger hierarchy. Safe to call more than once.

    Handlers previously added by this function (tagged with ``_CONFIGURED_MARKER``) are removed and
    closed first, so repeated calls do not duplicate output. The file handler always logs at DEBUG to
    ``<log_dir>/imsdb.log`` as JSON lines, rotating at 10 MiB with 10 backups. The logger does not
    propagate to the root logger.

    Args:
        log_dir: Directory for the log files; created if missing.
        level: Console log level name (case-insensitive); unknown names fall back to INFO.
        console: Whether to also log to stderr in the readable console format.
    """
    root = logging.getLogger("imsdb")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        if getattr(handler, _CONFIGURED_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "imsdb.log", maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    setattr(file_handler, _CONFIGURED_MARKER, True)
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(getattr(logging, level.upper(), logging.INFO))
        stream.setFormatter(ConsoleFormatter())
        setattr(stream, _CONFIGURED_MARKER, True)
        root.addHandler(stream)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger of the ``imsdb`` hierarchy.

    Args:
        name: Component name, e.g. ``"crawler"``.

    Returns:
        The logger named ``imsdb.<name>``, which uses the handlers set up by :func:`setup_logging`.
    """
    return logging.getLogger(f"imsdb.{name}")
