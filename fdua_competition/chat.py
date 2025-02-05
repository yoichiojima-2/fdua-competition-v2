import sys
import typing as t
from pathlib import Path

from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

sys.path.append(str(Path(__file__).parent.parent))
from fdua_competition.enums import ChatModelOption


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


def invoke_chain_with_retry(chain: Runnable, payload: dict[str, t.Any]) -> t.Any:
    return chain.invoke(payload)
