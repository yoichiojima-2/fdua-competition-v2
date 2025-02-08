import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fdua_competition.enums import Mode


def get_document_dir(mode: Mode = Mode.TEST) -> Path:
    match mode:
        case Mode.SUBMIT:
            return Path(os.environ["FDUA_DIR"]) / ".fdua-competition/documents"
        case Mode.TEST:
            return Path(os.environ["FDUA_DIR"]) / ".fdua-competition/validation/documents"
        case _:
            raise ValueError("Invalid mode")


def load_documents(mode: Mode = Mode.TEST) -> list[Document]:
    doc_dir = get_document_dir(mode)
    docs = []
    for path in doc_dir.rglob("*.pdf"):
        for doc in PyPDFium2Loader(path).lazy_load():
            docs.append(doc)
    return docs


def split_document(doc: Document) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "．", "？", "！", "「", "」", "【", "】"],
    )
    split_doc = splitter.split_text(doc.page_content)
    return [Document(page_content=d, metadata=doc.metadata) for d in split_doc]
