import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from fdua_competition.enums import Mode


def get_root() -> Path:
    return Path(os.getenv("FDUA_DIR")) / ".fdua-competition"


def get_queries(mode: Mode) -> list[str]:
    match mode:
        case Mode.TEST:
            df = pd.read_csv(get_root() / "validation/ans_txt.csv")
            return df["problem"].tolist()
        case Mode.SUBMIT:
            df = pd.read_csv(get_root() / "query.csv")
            return df["problem"].tolist()
        case _:
            raise ValueError(f"): unknown mode: {mode}")


def print_before_retry(retry_state):
    print(
        f":( retrying attempt {retry_state.attempt_number} after exception: {retry_state.outcome.exception()}", file=sys.stderr
    )


def write_result(output_name: str, responses: list[BaseModel]) -> None:
    assert responses[0].response, "response field is missing"
    output_path = get_root() / f"results/{output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"response": res.response} for res in responses])
    df.to_csv(output_path, header=False)
    print(f"[write_result] done: {output_path}")
