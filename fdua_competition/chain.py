import warnings
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def get_documents_dir() -> Path:
    return Path().home() / ".fdua-competition/downloads/documents"


def get_pages(filename: str) -> Iterable[Document]:
    pdf_path = get_documents_dir() / filename
    for doc in PyPDFium2Loader(pdf_path).lazy_load():
        yield doc


def get_vectorstore(
    model: str = "embedding",
    embedding_class: OpenAIEmbeddings = AzureOpenAIEmbeddings,
    vectorstore_class: VectorStore = InMemoryVectorStore,
) -> VectorStore:
    embeddings = embedding_class(model=model)
    return vectorstore_class(embedding=embeddings)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def add_documents_with_retry(vectorstore, batch):
    vectorstore.add_documents(batch)


def add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: Iterable[Document], batch_size: int = 5) -> None:
    batch = []
    for page in tqdm(pages, desc="adding pages to vectorstore"):
        batch.append(page)
        if len(batch) == batch_size:
            add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []  # clear the batch

    # add any remaining documents in the last batch
    if batch:
        vectorstore.add_documents(batch)


def get_prompt(
    system_prompt: str,
    query: str,
    vectorstore: VectorStore,
    language: str = "Japanese",
) -> ChatPromptValue:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "context: {context}"),
            ("user", "query: {query}"),
        ]
    )
    return prompt_template.invoke(
        {
            "language": language,
            "context": "\n".join([page.page_content for page in vectorstore.as_retriever().invoke(query)]),
            "query": query,
        }
    )


def get_chat_model() -> ChatOpenAI:
    return AzureChatOpenAI(azure_deployment="4omini")


def main() -> None:
    system_prompt = "Answer the following question based only on the provided context in {language}"
    query = "4℃ホールディングスの2024年2月29日現在の連結での従業員数は何名か"

    vectorstore = get_vectorstore()

    for path in get_documents_dir().glob("*.pdf"):
        print(f"processing: {path}")
        add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=get_pages(path))

    chat_model = get_chat_model()
    prompt = get_prompt(system_prompt, query, vectorstore)
    res = chat_model.invoke(prompt)
    print(res.content)


if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
