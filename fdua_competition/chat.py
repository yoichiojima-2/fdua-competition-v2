import sys
import typing as t
from pathlib import Path

import pandas as pd
from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

sys.path.append(str(Path(__file__).parent.parent))
from fdua_competition.enums import ChatModelOption, Mode
from fdua_competition.utils import get_root, print_before_retry


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


def get_chat_model(opt: ChatModelOption) -> ChatOpenAI:
    match opt:
        case ChatModelOption.AZURE:
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("system", "context:\n{context}"),
            ("user", "query:\n{query}"),
        ]
    )


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def invoke_chain_with_retry(chain: Runnable, payload: dict[str, t.Any]) -> t.Any:
    return chain.invoke(payload)
