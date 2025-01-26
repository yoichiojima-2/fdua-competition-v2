from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


def get_documents_dir() -> Path:
    return Path().home() / ".fdua-competition/downloads/documents"


def get_pages(filename: str) -> Iterable[Document]:
    pdf_path = get_documents_dir() / filename
    for doc in PyPDFium2Loader(pdf_path).lazy_load():
        yield doc


def make_retriever(pages: list[Document]) -> VectorStoreRetriever:
    embeddings = AzureOpenAIEmbeddings(model="embedding")
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    return vectorstore.add_documents(pages).as_retriever()


def get_prompt_template() -> ChatPromptTemplate:
    system_prompt = "Answer the following question based only on the provided context in {language}"
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "context: {context}"),
            ("user", "query: {query}"),
        ]
    )


def main() -> None:
    query = "4℃ホールディングスの2024年2月29日現在の連結での従業員数は何名か"

    pages = list(get_pages("1.pdf"))
    retriever = make_retriever(pages)

    prompt = get_prompt_template().invoke(
        {
            "language": "Japanese",
            "context": "\n".join([page.page_content for page in retriever.invoke(query)]),
            "query": query,
        }
    )

    llm = AzureChatOpenAI(azure_deployment="4omini")
    res = llm.invoke(prompt)
    print(res.content)


if __name__ == "__main__":
    load_dotenv()
    main()
