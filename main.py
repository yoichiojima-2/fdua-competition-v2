import argparse
import os
import sys
import warnings
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

load_dotenv("secrets/.env")


class Mode(Enum):
    SUBMIT = "submit"
    TEST = "test"


class ChatModelOption(Enum):
    AZURE = "azure"


class EmbeddingModelOption(Enum):
    AZURE = "azure"


class VectorStoreOption(Enum):
    CHROMA = "chroma"
    IN_MEMORY = "in-memory"


def print_before_retry(retry_state):
    print(f"retrying attempt {retry_state.attempt_number} after exception: {retry_state.outcome.exception()}", file=sys.stderr)


def get_root() -> Path:
    return Path(os.getenv("FDUA_DIR")) / ".fdua-competition"


def get_documents_dir(mode: Mode) -> Path:
    match mode:
        case Mode.TEST:
            return get_root() / "validation/documents"
        case Mode.SUBMIT:
            return get_root() / "documents"
        case _:
            raise ValueError(f"): unknown mode: {mode}")


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


def get_pages(path: Path) -> Iterable[Document]:
    for doc in PyPDFium2Loader(path).lazy_load():
        yield doc


def get_embedding_model(opt: EmbeddingModelOption) -> OpenAIEmbeddings:
    match opt:
        case EmbeddingModelOption.AZURE:
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"): unknown model: {opt}")


def get_vectorstore(output_name: str, opt: VectorStoreOption, embeddings: OpenAIEmbeddings) -> VectorStore:
    match opt:
        case VectorStoreOption.IN_MEMORY:
            return InMemoryVectorStore(embeddings)
        case VectorStoreOption.CHROMA:
            persist_path = get_root() / f"vectorstore/chroma/fdua-competition_{output_name}"
            print(f"[get_vectorstore] chroma: {persist_path}")
            return Chroma(
                collection_name=persist_path.name,
                embedding_function=embeddings,
                persist_directory=str(persist_path.parent),
            )
        case _:
            raise ValueError(f"): unknown vectorstore: {opt}")


def get_existing_sources(vectorstore: VectorStore) -> set[str]:
    return {metadata.get("source") for metadata in vectorstore.get().get("metadatas")}


@retry(stop=stop_after_attempt(24), wait=wait_fixed(10), before_sleep=print_before_retry)
def add_documents_with_retry(vectorstore: VectorStore, batch: list[Document]) -> None:
    vectorstore.add_documents(batch)


def add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: Iterable[Document], batch_size: int = 36) -> None:
    batch = []
    for page in tqdm(pages, desc="adding pages.."):
        batch.append(page)
        if len(batch) == batch_size:
            add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []
    if batch:
        add_documents_with_retry(vectorstore=vectorstore, batch=batch)


def get_documents(document_dir: Path) -> list[Path]:
    return [path for path in document_dir.glob("*.pdf")]


def add_document_to_vectorstore(documents: list[Path], vectorstore: VectorStore) -> None:
    existing_sources = get_existing_sources(vectorstore)
    for path in documents:
        if str(path) in existing_sources:
            print(f"[add_document_to_vectorstore] skipping existing document: {path}")
            continue
        print(f"[add_document_to_vectorstore] adding document to vectorstore: {path}")
        pages = get_pages(path=path)
        add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages)


def get_chat_model(opt: ChatModelOption) -> ChatOpenAI:
    match opt:
        case ChatModelOption.AZURE:
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


class Response(BaseModel):
    query: str = Field(description="the query that was asked.")
    response: str = Field(description="the answer for the given query")
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    contexts: list[str] = Field(description="the context that the response was based on with its file path and page number.")


def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "# {system_prompt}"),
            ("system", "## context\n{context}"),
            ("user", "## query\n{query}"),
        ]
    )


@retry(stop=stop_after_attempt(24), wait=wait_fixed(10), before_sleep=print_before_retry)
def build_context(vectorstore: VectorStore, query: str) -> str:
    pages = vectorstore.as_retriever().invoke(query)
    contexts = ["\n".join([f"page_content: {page.page_content}", f"metadata: {page.metadata}"]) for page in pages]
    return "\n---\n".join(contexts)


def write_result(output_name: str, responses: list[Response]) -> None:
    output_path = get_root() / f"results/{output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"response": res["response"]} for res in responses])
    df.to_csv(output_path, header=False)
    print(f"[write_result] done: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    opt = parser.add_argument
    opt("--output-name", "-o", type=str)
    opt("--mode", "-m", type=str, choices=[choice.value for choice in Mode], default=Mode.TEST.value)
    opt(
        "--vectorstore",
        "-v",
        type=str,
        choices=[choice.value for choice in VectorStoreOption],
        default=VectorStoreOption.CHROMA.value,
    )
    return parser.parse_args()


@retry(stop=stop_after_attempt(24), wait=wait_fixed(10), before_sleep=print_before_retry)
def invoke_chain_with_retry(chain, payload):
    return chain.invoke(payload)


@traceable
def main(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> None:
    embeddings = get_embedding_model(EmbeddingModelOption.AZURE)
    vectorstore = get_vectorstore(output_name=output_name, opt=vectorstore_option, embeddings=embeddings)
    docs = get_documents(document_dir=get_documents_dir(mode=mode))
    add_document_to_vectorstore(docs, vectorstore)

    prompt_template = get_prompt_template()
    chat_model = get_chat_model(ChatModelOption.AZURE)
    parser = JsonOutputParser(pydantic_object=Response)
    chain = prompt_template | chat_model | parser

    system_prompt = " ".join(
        [
            "answer the question using only the provided context in {language}.",
            "return a json object that contains the following keys: query, response, reason, organization_name, contexts.",
            "make sure that the response field is under 54 tokens and contains no commas. other fields do not have token limits.",
            "do not include honorifics or polite expressions; use plain, assertive language in the response field.",
            "the response field must be based only from given context.",
            "do not include any special characters that can cause json parsing errors across all fields. this must be satisfied regardless of the language.",
        ]
    )

    responses = []
    for query in tqdm(get_queries(mode=mode), desc="querying.."):
        res = invoke_chain_with_retry(
            chain,
            {
                "system_prompt": system_prompt,
                "query": query,
                "context": build_context(vectorstore=vectorstore, query=query),
                "language": "japanese",
            },
        )
        pprint(res)
        print()
        responses.append(res)

    pprint(responses)
    write_result(output_name=output_name, responses=responses)
    print("[main] :)  done")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    main(output_name=args.output_name, mode=Mode(args.mode), vectorstore_option=VectorStoreOption(args.vectorstore))
