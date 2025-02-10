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

from fdua_competition.enums import EmbeddingOpt
from fdua_competition.get_version import get_version
from fdua_competition.logging_config import logger
from fdua_competition.models import create_embeddings
from fdua_competition.pdf_handler import load_documents
from fdua_competition.utils import before_sleep_hook
from fdua_competition.cleanse import CleansePDF, cleanse_pdf

BATCH_SIZE = 1


class FduaVectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.persist_directory = Path(os.environ["FDUA_DIR"]) / f".fdua-competition/vectorstores/chroma/v{get_version()}"
        self.persist_directory.mkdir(parents=True, exist_ok=True)

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

    @retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
    def add(self, docs: list[Document]) -> None:
        cleansed_docs = [Document(page_content=cleanse_pdf(doc.page_content).output, metadata=doc.metadata) for doc in docs]
        self.vectorstore.add_documents(cleansed_docs)

    def add_documents_concurrently(self, docs: list[Document]) -> None:
        batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.add, batch) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc="populating vectorstore.."):
                future.result()

    def populate(self) -> None:
        self.vectorstore.reset_collection()
        docs = load_documents()
        self.add_documents_concurrently(docs)
        logger.info("[FduaVectorStore] done populating vectorstore")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings",
        "-e",
        type=EmbeddingOpt,
        default=EmbeddingOpt.AZURE,
        choices=list(EmbeddingOpt),
        help="the type of embeddings to use",
    )
    return parser.parse_args()


def prepare_vectorstore() -> None:
    args = parse_args()
    embeddings = create_embeddings(args.embeddings)
    vs = FduaVectorStore(embeddings=embeddings)
    vs.populate()


if __name__ == "__main__":
    prepare_vectorstore()
