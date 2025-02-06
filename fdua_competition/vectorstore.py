from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from fdua_competition.enums import EmbeddingModelOption, Mode, VectorStoreOption
from fdua_competition.utils import get_root, print_before_retry


def get_documents_dir(mode: Mode) -> Path:
    match mode:
        case Mode.TEST:
            return get_root() / "validation/documents"
        case Mode.SUBMIT:
            return get_root() / "documents"
        case _:
            raise ValueError(f"): unknown mode: {mode}")


def get_document_list(document_dir: Path) -> list[Path]:
    return [path for path in document_dir.glob("*.pdf")]


def load_pages(path: Path) -> Iterable[Document]:
    for doc in PyPDFium2Loader(path).lazy_load():
        yield doc


def get_embedding_model(opt: EmbeddingModelOption) -> OpenAIEmbeddings:
    match opt:
        case EmbeddingModelOption.AZURE:
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"): unknown model: {opt}")


def prepare_vectorstore(output_name: str, opt: VectorStoreOption, embeddings: OpenAIEmbeddings) -> VectorStore:
    match opt:
        case VectorStoreOption.IN_MEMORY:
            return InMemoryVectorStore(embeddings)
        case VectorStoreOption.CHROMA:
            persist_directory = get_root() / "vectorstores/chroma"
            print(f"[prepare_vectorstore] chroma: {persist_directory}")
            persist_directory.mkdir(parents=True, exist_ok=True)
            return Chroma(
                collection_name=output_name,
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
            )
        case _:
            raise ValueError(f"): unknown vectorstore: {opt}")


def _get_existing_sources_in_vectorstore(vectorstore: VectorStore) -> set[str]:
    return {metadata.get("source") for metadata in vectorstore.get().get("metadatas")}


def _add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: Iterable[Document], batch_size: int = 8) -> None:
    batch = []
    for page in tqdm(pages, desc="adding pages.."):
        batch.append(page)
        if len(batch) == batch_size:
            _add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []
    if batch:
        _add_documents_with_retry(vectorstore=vectorstore, batch=batch)


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def _add_documents_with_retry(vectorstore: VectorStore, batch: list[Document]) -> None:
    vectorstore.add_documents(batch)


def add_documents_to_vectorstore(documents: list[Path], vectorstore: VectorStore) -> None:
    existing_sources = _get_existing_sources_in_vectorstore(vectorstore)
    for path in documents:
        if str(path) in existing_sources:
            print(f"[add_document_to_vectorstore] skipping existing document: {path}")
            continue
        print(f"[add_document_to_vectorstore] adding document to vectorstore: {path}")
        pages = load_pages(path=path)
        _add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages)


def build_vectorstore(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> VectorStore:
    embeddings = get_embedding_model(EmbeddingModelOption.AZURE)
    vectorstore = prepare_vectorstore(output_name=output_name, opt=vectorstore_option, embeddings=embeddings)
    docs = get_document_list(document_dir=get_documents_dir(mode=mode))
    add_documents_to_vectorstore(docs, vectorstore)
    return vectorstore


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def build_context(vectorstore: VectorStore, query: str) -> str:
    pages = vectorstore.as_retriever().invoke(query)
    contexts = ["\n".join([f"page_content: {page.page_content}", f"metadata: {page.metadata}"]) for page in pages]
    return "\n---\n".join(contexts)
