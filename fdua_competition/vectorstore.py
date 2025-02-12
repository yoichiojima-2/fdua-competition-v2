import os
import typing as t
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

from fdua_competition.cleanse import cleanse_pdf
from fdua_competition.enums import EmbeddingOpt, LogLevel, Mode
from fdua_competition.get_version import get_version
from fdua_competition.logging_config import logger, set_log_level
from fdua_competition.models import create_embeddings
from fdua_competition.pdf_handler import load_documents
from fdua_competition.utils import before_sleep_hook

BATCH_SIZE = 1


class FduaVectorStore:
    def __init__(self, mode: Mode, embeddings: Embeddings):
        self.embeddings = embeddings
        self.persist_directory = (
            Path(os.environ["FDUA_DIR"]) / f".fdua-competition/vectorstores/chroma/v{get_version()}/{mode.value}"
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        logger.info(f"[FduaVectorStore] {self.persist_directory}")
        self.vectorstore = Chroma(
            collection_name="fdua-competition",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    def get(self, **kwargs) -> dict[str, t.Any]:
        return self.vectorstore.get(**kwargs)

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever(**kwargs)

    def cleanse_concurrently(self, docs: list[Document]) -> list[Document]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(cleanse_pdf, docs))

    @retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
    def add(self, docs: list[Document]) -> None:
        cleansed_docs = self.cleanse_concurrently(docs)
        self.vectorstore.add_documents(cleansed_docs)
        for doc in cleansed_docs:
            logger.info(f"[FduaVectorStore] added {doc.metadata}\n{doc.page_content}\n")

    def add_documents_concurrently(self, docs: list[Document]) -> None:
        batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.add, batch) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc="populating vectorstore.."):
                future.result()

    def is_existing(self, doc: Document) -> bool:
        return doc.metadata in self.get()["metadatas"]

    def is_empty(self) -> bool:
        return len(self.get()["metadatas"]) == 0

    def populate(self) -> None:
        docs = load_documents(self.mode)
        docs_to_add = docs if self.is_empty() else [doc for doc in docs if not self.is_existing(doc)]
        logger.info(f"[FduaVectorStore]\n- adding: {len(docs_to_add)} docs\n- skippting: {len(docs) - len(docs_to_add)} docs\n")
        self.add_documents_concurrently(docs_to_add)
        logger.info("[FduaVectorStore] done populating vectorstore")

    def reset_collection(self) -> None:
        self.vectorstore.reset_collection()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    opt("--embeddings", "-e", type=str, default=EmbeddingOpt.AZURE.value, choices=[e.value for e in EmbeddingOpt])
    opt("--reset", "-r", type=bool, default=False)
    opt("--log-level", "-l", type=str, default=LogLevel.INFO.value)
    return parser.parse_args()


def prepare_vectorstore() -> None:
    args = parse_args()
    set_log_level(LogLevel(args.log_level))
    embeddings = create_embeddings(EmbeddingOpt(args.embeddings))
    vs = FduaVectorStore(mode=Mode(args.mode), embeddings=embeddings)
    if args.reset:
        logger.warning("resetting vectorstore")
        vs.reset_collection()
    vs.populate()


if __name__ == "__main__":
    prepare_vectorstore()
