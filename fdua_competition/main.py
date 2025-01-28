import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def get_root() -> Path:
    return Path().home() / ".fdua-competition"


def get_documents_dir() -> Path:
    return get_root() / "downloads/documents"


def get_output_path() -> Path:
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M')}.md"


@traceable
def get_queries() -> list[str]:
    df = pd.read_csv(get_root() / "downloads/query.csv")
    return df["problem"].tolist()


@traceable
def get_pages(filename: str) -> Iterable[Document]:
    pdf_path = get_documents_dir() / filename
    for doc in PyPDFium2Loader(pdf_path).lazy_load():
        yield doc


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
@traceable
def add_documents_with_retry(vectorstore, batch):
    vectorstore.add_documents(batch)


@traceable
def add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: Iterable[Document], batch_size: int = 5) -> None:
    batch = []
    for page in tqdm(pages, desc="adding pages.."):
        batch.append(page)
        if len(batch) == batch_size:
            add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []  # clear the batch

    # add any remaining documents in the last batch
    if batch:
        vectorstore.add_documents(batch)


@traceable
def build_vectorstore(model: str, embedding_class: OpenAIEmbeddings, vectorstore_class: VectorStore) -> VectorStore:
    print("[build_vectorstore] building vectorstore..")
    embeddings = embedding_class(model=model)
    vectorstore = vectorstore_class(embedding=embeddings)

    for path in get_documents_dir().glob("*.pdf"):
        print(f"adding document to vectorstore: {path}")
        add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=get_pages(path))

    print("[build_vectorstore] done building vectorstore")

    return vectorstore


@traceable
def get_prompt(
    system_prompt: str,
    query: str,
    vectorstore: VectorStore,
    language: str = "Japanese",
) -> ChatPromptValue:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "context: {context}"),
            ("user", "query: {query}"),
        ]
    ).invoke(
        {
            "language": language,
            "context": "\n".join([page.page_content for page in vectorstore.as_retriever().invoke(query)]),
            "query": query,
        }
    )


@traceable
def get_chat_model(model: str) -> ChatOpenAI:
    match model:
        case "azure":
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {model}")


@traceable
def main() -> None:
    system_prompt = "Answer the following question based only on the provided context in {language}"

    vectorstore = build_vectorstore(
        model="embedding",
        embedding_class=AzureOpenAIEmbeddings,
        vectorstore_class=InMemoryVectorStore,
    )
    chat_model = get_chat_model("azure")

    with get_output_path().open(mode="a") as f:
        f.write("# Results\n")

        for query in tqdm(get_queries(), desc="querying.."):
            prompt = get_prompt(system_prompt, query, vectorstore)
            res = chat_model.invoke(prompt)

            f.write(f"## {query}\n")
            f.write(f"{res.content}\n\n")

            print(f"done\nquery: {query}\noutput: {res.content}\n\n")


if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
