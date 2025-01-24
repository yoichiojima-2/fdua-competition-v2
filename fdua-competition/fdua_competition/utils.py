from pathlib import Path


def get_interim_dir() -> Path:
    dir = Path().home() / ".fdua-competition/interim"
    dir.mkdir(exist_ok=True)
    return dir