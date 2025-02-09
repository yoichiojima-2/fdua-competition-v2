import os
import typing as t
from argparse import ArgumentParser, Namespace
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from fdua_competition.enums import EmbeddingOpt
from fdua_competition.models import create_embeddings
from fdua_competition.pdf_handler import load_documents
from fdua_competition.utils import get_version, log_retry


class FduaVectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.persist_directory = Path(os.getenv("FDUA_DIR")) / f".fdua-competition/vectorstores/chroma/v{get_version()}"
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        print(f"[FduaVectorStore] {self.persist_directory}")
        self.vectorstore = Chroma(
            collection_name="fdua-competition",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    def get(self, **kwargs) -> dict[str, t.Any]:
        return self.vectorstore.get(**kwargs)

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever(**kwargs)

    @retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=log_retry)
    def add(self, docs: list[Document]) -> None:
        self.vectorstore.add_documents(docs)

    def populate(self) -> None:
        self.vectorstore.reset_collection()
        docs = load_documents()

        # [start: adding documents]

        # # v2.3: recursive split
        # for doc in tqdm(docs, desc="populating vectorstore.."):
        #     split_doc = split_document(doc)
        #     self.add(split_doc)

        # # v2.4: page by page (premitive but better)
        # for doc in tqdm(docs, desc="populating vectorstore.."):
        #     self.add([doc])

        # v2.5: page by page in batches
        size = 8
        batches = [docs[i : i + size] for i in range(0, len(docs), size)]
        for doc in tqdm(batches, desc="populating vectorstore.."):
            self.add(doc)

        # [end: adding documents]


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
