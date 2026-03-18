from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional


def load_config(config_path: str | Path) -> dict:
    """Load and return the config dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid clobbering progress bars."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level=level)
        # Import lazily so tqdm stays optional in non-CLI contexts.
        from tqdm import tqdm  # type: ignore

        self._tqdm = tqdm

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._tqdm.write(msg)
        except Exception:
            self.handleError(record)


def configure_logging(
    *,
    level: int = logging.INFO,
    use_tqdm: bool = True,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure root logging once for CLI scripts.

    - If use_tqdm=True and tqdm is available, routes log lines through tqdm.write().
      This keeps progress bars readable.
    - Otherwise falls back to a normal StreamHandler.
    """
    root = logging.getLogger()

    # Avoid stacking duplicate handlers when configure_logging is called multiple times
    # (common in notebooks / re-runs).
    handler_types = {type(h) for h in root.handlers}
    handler: logging.Handler

    if use_tqdm:
        try:
            if _TqdmLoggingHandler in handler_types:
                handler = next(
                    h for h in root.handlers if isinstance(h, _TqdmLoggingHandler)
                )
            else:
                handler = _TqdmLoggingHandler()
        except Exception:
            # tqdm not available or failed to import; use standard stream logging.
            handler = logging.StreamHandler()
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    if type(handler) not in handler_types:
        root.addHandler(handler)

    root.setLevel(level)

    return logging.getLogger(logger_name) if logger_name else logging.getLogger()

