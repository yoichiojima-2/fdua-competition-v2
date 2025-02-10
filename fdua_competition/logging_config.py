import logging
from logging import Logger

from fdua_competition.get_version import get_version


def get_logger() -> Logger:
    logger = logging.getLogger(f"v{get_version()}")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
