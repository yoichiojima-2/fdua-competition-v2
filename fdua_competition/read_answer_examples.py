import os
import textwrap
from pathlib import Path

import pandas as pd

from fdua_competition.enums import Mode
from fdua_competition.utils import dict_to_yaml


def read_answer_examples(mode: Mode) -> str:
    match mode:
        case mode.SUBMIT:
            df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/validation/ans_txt.csv")
            return dict_to_yaml([{"query": row[1].problem, "example answer": row[1].ground_truth} for row in df.iterrows()])
        case mode.TEST:
            return textwrap.dedent(
                """
                - xxxは何年か -> good: xxxx年  /  bad: xxxはxxxx年です
                - xxxはaとbどちらか -> good: a  /  bad: xxxはaです
                - aとbのどちらがxxか -> good: a  /  bad : xxxなのはaです
                - 何%か -> response: good: 10%  /  bad: 10  # 単位をつける
                """
            )
        case _:
            raise ValueError(f"Invalid mode: {mode}")
