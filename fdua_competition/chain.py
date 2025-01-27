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

    pages = list(get_pages("1.pdf"))
    vectorstore = get_vectorstore()
    vectorstore.add_documents(pages)

    prompt = get_prompt(system_prompt, query, vectorstore)

    chat_model = get_chat_model()
    res = chat_model.invoke(prompt)
    print(res.content)


if __name__ == "__main__":
    load_dotenv()
    main()
