import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from fdua_competition.utils import get_root, write_result


def test_get_root():
    assert get_root().name == ".fdua-competition"


def test_write_result(tmp_path):
    responses = [
        {"response": "response1"},
        {"response": "response2"},
        {"response": "response3"},
    ]
    write_result("test", responses)
    assert Path(get_root() / "results/test.csv").exists()
    df = pd.read_csv(get_root() / "results/test.csv", header=None)
    assert len(df) == 3
