import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from fdua_competition.enums import Mode


def get_document_dir(mode: Mode = Mode.TEST) -> Path:
    match mode:
        case Mode.SUBMIT:
            return Path(os.environ["FDUA_DIR"]) / ".fdua-competition/documents"
        case Mode.TEST:
            return Path(os.environ["FDUA_DIR"]) / ".fdua-competition/validation/documents"
        case _:
            raise ValueError("Invalid mode")


def load_documents_concurrently(mode: Mode = Mode.TEST) -> list[Document]:
    doc_dir = get_document_dir(mode)
    docs = []
    paths = list(doc_dir.rglob("*.pdf"))
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(PyPDFium2Loader(path).lazy_load): path for path in paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="loading documents.."):
            for doc in future.result():
                docs.append(doc)
    return docs


def load_documents(mode: Mode = Mode.TEST) -> list[Document]:
    return load_documents_concurrently(mode)


def split_document(doc: Document) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "．", "？", "！", "「", "」", "【", "】"],
    )
    split_doc = splitter.split_text(doc.page_content)
    return [Document(page_content=d, metadata=doc.metadata) for d in split_doc]
