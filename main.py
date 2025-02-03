import argparse
import os
import re
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
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv("secrets/.env")


class Mode(Enum):
    submit = "submit"
    test = "test"


class VectorStoreOption(Enum):
    chroma = "chroma"
    in_memory = "in-memory"


def get_root() -> Path:
    return Path(os.getenv("FDUA_DIR")) / ".fdua-competition"


def get_documents_dir(mode: Mode) -> Path:
    match mode:
        case "test":
            return get_root() / "validation/documents"
        case "submit":
            return get_root() / "documents"
        case _:
            raise ValueError(f"): unknown mode: {mode}")


def get_queries(mode: Mode) -> list[str]:
    match mode:
        case "test":
            df = pd.read_csv(get_root() / "validation/ans_txt.csv")
            return df["problem"].tolist()
        case "submit":
            df = pd.read_csv(get_root() / "query.csv")
            return df["problem"].tolist()
        case _:
            raise ValueError(f"): unknown mode: {mode}")


def get_pages(path: Path) -> Iterable[Document]:
    for doc in PyPDFium2Loader(path).lazy_load():
        yield doc


def get_embedding_model(opt: str) -> OpenAIEmbeddings:
    match opt:
        case "azure":
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"): unknown model: {opt}")


def get_vectorstore(output_name: str, opt: str, embeddings: OpenAIEmbeddings) -> VectorStore:
    match opt:
        case "in-memory":
            return InMemoryVectorStore(embeddings)
        case "chroma":
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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def add_documents_with_retry(vectorstore: VectorStore, batch: list[Document]) -> None:
    vectorstore.add_documents(batch)


def add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: Iterable[Document], batch_size: int = 8) -> None:
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


def get_prompt(system_prompt: str, query: str, retriever: VectorStoreRetriever, language: str = "Japanese") -> ChatPromptValue:
    context = "\n---\n".join(
        ["\n".join([f"page_content: {page.page_content}", f"metadata: {page.metadata}"]) for page in retriever.invoke(query)]
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "**context**\n{context}"),
            ("user", "query: {query}"),
        ]
    ).invoke(
        {
            "language": language,
            "context": context,
            "query": query,
        }
    )


def get_chat_model(opt: str) -> ChatOpenAI:
    match opt:
        case "azure":
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


class Response(BaseModel):
    query: str = Field(description="the query that was asked.")
    response: str = Field(
        description=(
            "the answer for the given query\n"
            "this field must:\n"
            "- be strictly within 54 tokens regardless of language\n"
            "- provide a single, concise response\n"
            "- not provide extra details, explanations, or redundant words"
            "- not include honorifics or polite expressions; use plain, assertive language\n"
            "- be based only from given context'\n"
        )
    )
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    sources: list[str] = Field(description="the sources of the response.")
    context: str = Field(description="the cleansed context in given prompt.")


def cleanse_response(response: str) -> str:
    return re.sub(r"[\x00-\x1F]+", "", response.replace(",", ""))


def write_result(output_name: str, responses: list[Response]) -> None:
    output_path = get_root() / f"results/{output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"response": cleanse_response(res.response)} for res in responses])
    df.to_csv(output_path, header=False)
    print(f"[write_result] done: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    opt = parser.add_argument
    opt("--output-name", "-o", type=str)
    opt("--mode", "-m", type=str, choices=[choice.value for choice in Mode], default="test")
    opt("--vectorstore", "-v", type=str, choices=[choice.value for choice in VectorStoreOption], default="chroma")
    return parser.parse_args()


@traceable
def main(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> None:
    embedding_model = get_embedding_model("azure")
    vectorstore = get_vectorstore(output_name=output_name, opt=vectorstore_option, embeddings=embedding_model)
    docs = get_documents(document_dir=get_documents_dir(mode=mode))
    add_document_to_vectorstore(docs, vectorstore)
    retriever = vectorstore.as_retriever()

    chat_model = get_chat_model("azure").with_structured_output(Response)
    system_prompt = (
        "answer the question using only the provided context in {language}.\n"
        "return your answer as valid json on a single line with no control characters.\n"
        "ensure the 'response' field is under 54 tokens and contains no commas."
    )

    responses = []
    for query in tqdm(get_queries(mode=mode), desc="querying.."):
        prompt = get_prompt(system_prompt, query, retriever)
        res = chat_model.invoke(prompt)
        pprint(res)
        responses.append(res)

    write_result(output_name=output_name, responses=responses)
    print("[main] :)  done")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    main(output_name=args.output_name, mode=args.mode, vectorstore_option=args.vectorstore)
