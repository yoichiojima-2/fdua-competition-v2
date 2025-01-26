from pathlib import Path


def get_documents_dir() -> Path:
    return Path().home() / ".fdua-competition/downloads/documents"


def get_interim_dir() -> Path:
    dir = Path().home() / ".fdua-competition/interim"
    dir.mkdir(exist_ok=True, parents=True)
    return dir
