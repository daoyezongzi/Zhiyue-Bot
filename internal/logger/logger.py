from __future__ import annotations

import logging
from typing import Final


def init_logger(level: str = "INFO", debug_mode: bool = False) -> None:
    fmt: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level.upper(), format=fmt)
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
