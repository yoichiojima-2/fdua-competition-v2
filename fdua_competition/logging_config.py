import logging
import os
from pathlib import Path

from fdua_competition.get_version import get_version
from fdua_competition.enums import LogLevel


def get_logger() -> logging.Logger:
    log_dir = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"v{get_version()}")

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = logging.FileHandler(log_dir / f"{get_version()}.log")
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def set_log_level(level: LogLevel) -> None:
    logger.setLevel(level.value)


logger = get_logger()
