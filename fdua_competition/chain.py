from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFium2Loader

from fdua_competition.utils import get_documents_dir


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


load_dotenv()

pages = get_all_pages()
embeddings = AzureOpenAIEmbeddings(model="embedding")
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(pages)

print(vectorstore.search(query="test", search_type="similarity"))

# retriever = vectorstore.as_retriever()

# retrieved_documents = retriever.invoke("what is written in the document?")
# print(retrieved_documents[0].page_content)
