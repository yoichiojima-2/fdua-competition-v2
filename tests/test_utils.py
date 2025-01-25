from pathlib import Path
from fdua_competition.utils import get_interim_dir


def test_get_interim_dir():
    dir = get_interim_dir()
    assert dir.exists()
    assert dir.is_dir()
    assert dir == Path().home() / ".fdua-competition/interim"