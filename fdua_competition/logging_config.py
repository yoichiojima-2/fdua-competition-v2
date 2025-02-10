import logging
import os
from logging import Logger
from pathlib import Path

from fdua_competition.get_version import get_version

LOG_DIR = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/logs"


def get_logger() -> Logger:
    logger = logging.getLogger(f"v{get_version()}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(LOG_DIR / f"{get_version()}.log")
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger()
