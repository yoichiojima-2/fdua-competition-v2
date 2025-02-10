import os
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel
from tenacity import RetryCallState

from fdua_competition.enums import Mode
from fdua_competition.get_version import get_version
from fdua_competition.logging_config import logger


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


def write_result(responses: list[BaseModel]) -> None:
    assert responses[0].response, "response field is missing"
    output_path = Path(os.environ["FDUA_DIR"]) / f".fdua-competition/results/v{get_version()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"response": res.response} for res in responses])
    df.to_csv(output_path, header=False)
    logger.info(f"[write_result] done: {output_path}")


def before_sleep_hook(state: RetryCallState) -> None:
    logger.warning(f":( retrying attempt {state.attempt_number} after exception: {state.outcome.exception()}")


def dict_to_yaml(model: BaseModel) -> str:
    return yaml.dump(model, allow_unicode=True, default_flow_style=False, sort_keys=False)
