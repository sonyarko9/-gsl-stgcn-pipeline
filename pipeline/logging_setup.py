"""Simple logging initializer for the pipeline.

Call `init_logging()` early (e.g. in `pipeline/main.py`) to configure
the root logger with a console handler and a consistent format.
"""
from __future__ import annotations

import logging


def init_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    root.setLevel(level)
    root.addHandler(handler)
