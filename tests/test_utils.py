import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))
from fdua_competition.enums import Mode
from fdua_competition.utils import get_queries, get_root, write_result


def test_get_root():
    assert get_root().name == ".fdua-competition"


def test_get_queries():
    queries = get_queries(mode=Mode.TEST)
    assert isinstance(queries, list)
    assert queries


class ChatResponse(BaseModel):
    response: str = Field(...)


def test_write_result(tmp_path):
    responses = [ChatResponse(response=f"test_{i}") for i in range(3)]
    write_result("test", responses)
    assert Path(get_root() / "results/test.csv").exists()
    df = pd.read_csv(get_root() / "results/test.csv", header=None)
    assert len(df) == 3
