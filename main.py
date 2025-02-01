import argparse
import warnings
from pathlib import Path
from pprint import pprint
from typing import Iterable, Literal

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

TAG = "simple"
Mode = Literal["submit", "test"]
VectorStoreOption = Literal["chroma", "in-memory"]

load_dotenv("secrets/.env")


# [START: paths]


def get_root() -> Path:
    return Path(__file__).parent / ".fdua-competition"


def get_documents_dir(mode: Mode) -> Path:
    match mode:
        case "test":
            return get_root() / "validation/documents"
        case "submit":
            return get_root() / "documents"
        case _:
            raise ValueError(f"unknown mode: {mode}")


# [END: paths]


# [START: queries]
def get_queries(mode: Mode) -> list[str]:
    match mode:
        case "test":
            df = pd.read_csv(get_root() / "validation/ans_txt.csv")
            return df["problem"].tolist()

        case "submit":
            df = pd.read_csv(get_root() / "query.csv")
            return df["problem"].tolist()

        case _:
            raise ValueError(f"unknown mode: {mode}")


# [END: queries]


# [START: vectorstores]
def get_pages(path: Path) -> Iterable[Document]:
    for doc in PyPDFium2Loader(path).lazy_load():
        yield doc


def get_embedding_model(opt: str) -> OpenAIEmbeddings:
    match opt:
        case "azure":
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"unknown model: {opt}")


def get_vectorstore(mode: Mode, opt: str, embeddings: OpenAIEmbeddings) -> VectorStore:
    match opt:
        case "in-memory":
            return InMemoryVectorStore(embeddings)
        case "chroma":
            return Chroma(
                collection_name=f"fdua-competition_{TAG}_{mode}",
                embedding_function=embeddings,
                persist_directory=str(get_root() / "vectorstore/chroma"),
            )
        case _:
            raise ValueError(f"unknown vectorstore: {opt}")


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
            batch = []  # clear the batch

    # add any remaining documents in the last batch
    if batch:
        add_documents_with_retry(vectorstore=vectorstore, batch=batch)


def get_documents(document_dir: Path) -> list[Path]:
    return [path for path in document_dir.glob("*.pdf")]


def add_document_to_vectorstore(documents: list[Path], vectorstore: VectorStore) -> None:
    existing_sources = get_existing_sources(vectorstore)
    for path in documents:
        if str(path) in existing_sources:
            continue

        print(f"adding document to vectorstore: {path}")
        pages = get_pages(path=path)
        add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages)


# [END: vectorstores]


# [START: chat]


def get_prompt(system_prompt: str, query: str, retriever: VectorStoreRetriever, language: str = "Japanese") -> ChatPromptValue:
    relevant_pages = retriever.invoke(query)
    context = "\n".join(
        ["\n".join([f"metadata: {page.metadata}", f"page_content: {page.page_content}"]) for page in relevant_pages]
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "context: {context}"),
            ("user", "query: {query}"),
        ]
    ).invoke(
        {
            "language": language,
            "context": "\n".join(context),
            "query": query,
        }
    )


def get_chat_model(opt: str) -> ChatOpenAI:
    match opt:
        case "azure":
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


# [END: chat]


def parse_metadata(metadata: list[dict]) -> str:
    return ",  ".join([f"p{data['page']} - {Path(data['source']).name}" for data in metadata])


class Response(BaseModel):
    query: str = Field(description="the query that was asked")
    response: str = Field(description="the response that was given")
    reason: str = Field(description="the reason for the response")
    organization_name: str = Field(description="the organization name that query is about")
    sources: list[str] = Field(description="the sources of the response")


def main(mode: Mode, vectorstore_option: VectorStoreOption) -> None:
    embedding_model = get_embedding_model("azure")
    vectorstore = get_vectorstore(mode=mode, opt=vectorstore_option, embeddings=embedding_model)
    docs = get_documents(document_dir=get_documents_dir(mode=mode))
    add_document_to_vectorstore(docs, vectorstore)
    retriever = vectorstore.as_retriever()

    chat_model = get_chat_model("azure").with_structured_output(Response)
    system_prompt = "Answer the following question based only on the provided context in {language}"

    for query in tqdm(get_queries(mode=mode), desc="querying.."):
        prompt = get_prompt(system_prompt, query, retriever)
        res = chat_model.invoke(prompt)
        pprint(res)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, choices=["test", "submit"], default="test")
    parser.add_argument("--vectorstore", "-v", type=str, choices=["chroma", "in-memory"], default="chroma")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    main(mode=args.mode, vectorstore_option=args.vectorstore)
