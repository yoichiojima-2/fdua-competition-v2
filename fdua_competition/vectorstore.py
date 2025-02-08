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
from fdua_competition.pdf_handler import load_documents, split_document
from fdua_competition.utils import log_retry


class FduaVectorStore:
    def __init__(self, output_name: str, embeddings: Embeddings):
        self.embeddings = embeddings
        self.persist_directory = Path(os.getenv("FDUA_DIR")) / f".fdua-competition/vectorstores/chroma/{output_name}"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name="fdua-competition",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    def get(self) -> dict[str, t.Any]:
        return self.vectorstore.get()

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever(**kwargs)

    @retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=log_retry)
    def add(self, docs: list[Document]) -> None:
        self.vectorstore.add_documents(docs)

    def populate(self) -> None:
        self.vectorstore.reset_collection()
        docs = load_documents()
        for doc in tqdm(docs, desc="populating vectorstore.."):
            split_doc = split_document(doc)
            self.add(split_doc)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-name",
        "-o",
        type=str,
        help="the name of the vectorstore to persist",
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=EmbeddingOpt,
        default=EmbeddingOpt.AZURE,
        choices=list(EmbeddingOpt),
        help="the type of embeddings to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_embeddings(args.embeddings)
    vs = FduaVectorStore(output_name=args.output_name, embeddings=embeddings)
    vs.populate()


if __name__ == "__main__":
    main()
