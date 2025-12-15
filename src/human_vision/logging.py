from __future__ import annotations

import logging
from typing import Any


LOGGER_NAME = "human_vision"


def get_logger(**kwargs: Any) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    if kwargs:
        adapter = logging.LoggerAdapter(logger, extra=kwargs)
        return adapter  # type: ignore[return-value]
    return logger
