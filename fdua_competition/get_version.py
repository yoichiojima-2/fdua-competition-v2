import os
import tomllib
from pathlib import Path


def get_version():
    with open(Path(os.environ["FDUA_DIR"]) / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]
