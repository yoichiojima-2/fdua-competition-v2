import logging
import os
from logging import Logger
from pathlib import Path
from fdua_competition.enums import Mode

from fdua_competition.get_version import get_version



def get_logger(mode: Mode) -> Logger:
    log_dir = Path(os.environ["FDUA_DIR"]) / f".fdua-competition/logs/{mode.value}"
    logger = logging.getLogger(f"{mode.value}-v{get_version()}")

    log_dir.mkdir(parents=True, exist_ok=True)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_dir / f"{get_version()}.log")
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger(Mode.TEST)