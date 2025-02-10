import os
import typing as t
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from fdua_competition.enums import EmbeddingOpt
from fdua_competition.get_version import get_version
from fdua_competition.logger import get_logger
from fdua_competition.models import create_embeddings
from fdua_competition.pdf_handler import load_documents
from fdua_competition.utils import before_sleep_hook

BATCH_SIZE = 4
logger = get_logger()


def add_documents_concurrently(vectorstore: VectorStore, docs: list[Document]) -> None:
    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(vectorstore.add, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="populating vectorstore.."):
            future.result()


class FduaVectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.persist_directory = Path(os.getenv("FDUA_DIR")) / f".fdua-competition/vectorstores/chroma/v{get_version()}"
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

    @retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=before_sleep_hook)
    def add(self, docs: list[Document]) -> None:
        self.vectorstore.add_documents(docs)

    def populate(self) -> None:
        self.vectorstore.reset_collection()
        docs = load_documents()
        add_documents_concurrently(self, docs)


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
