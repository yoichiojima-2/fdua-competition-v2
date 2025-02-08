import os
from pathlib import Path
from pprint import pprint

import pandas as pd
from pydantic import BaseModel
from tenacity import RetryCallState

from fdua_competition.enums import Mode


def read_queries(mode: Mode) -> list[str]:
    match mode:
        case Mode.TEST:
            df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/validation/ans_txt.csv")
            return df["problem"].tolist()
        case Mode.SUBMIT:
            df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/query.csv")
            return df["problem"].tolist()
        case _:
            raise ValueError(f"): unknown mode: {mode}")


def write_result(output_name: str, responses: list[BaseModel]) -> None:
    assert responses[0].response, "response field is missing"
    output_path = Path(os.environ["FDUA_DIR"]) / f".fdua-competition/results/{output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{"response": res.response} for res in responses])
    df.to_csv(output_path, header=False)

    print(f"[write_result] done: {output_path}")


def log_retry(state: RetryCallState) -> None:
    pprint(f":( retrying attempt {state.attempt_number} after exception: {state.outcome.exception()}")
