from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever


def get_documents_dir() -> Path:
    return Path().home() / ".fdua-competition/downloads/documents"


def get_pdf_paths() -> Iterable[Path]:
    documents_dir = get_documents_dir()
    for pdf_path in documents_dir.glob("1.pdf"):
        yield pdf_path


def get_pages(filename: str) -> Iterable[Document]:
    pdf_path = get_documents_dir() / filename
    for doc in PyPDFium2Loader(pdf_path).lazy_load():
        yield doc


def get_all_pages() -> Iterable[Document]:
    for pdf in get_pdf_paths():
        for page in get_pages(pdf.name):
            yield page


def make_retriever() -> VectorStoreRetriever:
    pages = list(get_all_pages())

    embeddings = AzureOpenAIEmbeddings(model="embedding")
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_documents(pages)

    return vectorstore.as_retriever()


def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the following question based only on the provided context in {language}",
            ),
            ("system", "context: {context}"),
            ("user", "query: {query}"),
        ]
    )


def main() -> None:
    load_dotenv()

    query = "4℃ホールディングスの2024年2月29日現在の連結での従業員数は何名か"
    prompt_template = get_prompt_template()
    retriever = make_retriever()

    prompt = prompt_template.invoke(
        {
            "language": "Japanese",
            "context": retriever.invoke(query)[0].page_content,
            "query": query,
        }
    )

    llm = AzureChatOpenAI(azure_deployment="4omini")
    res = llm.invoke(prompt)
    print(res.content)


if __name__ == "__main__":
    main()
