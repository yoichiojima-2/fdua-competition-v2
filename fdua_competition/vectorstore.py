import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from fdua_competition.embeddings import create_embeddings
from fdua_competition.enums import EmbeddingOpt, VectorStoreOpt
from fdua_competition.pdf_handler import load_documents, split_document


class FduaVectorStore:
    def __init__(self, embeddings: Embeddings, opt: VectorStoreOpt = VectorStoreOpt.CHROMA):
        match opt:
            case VectorStoreOpt.CHROMA:
                persist_directory = Path(os.getenv("FDUA_DIR")) / ".fdua-competition/vectorstores/chroma"
                persist_directory.mkdir(parents=True, exist_ok=True)
                self.vectorstore = Chroma(
                    collection_name=os.environ["OUTPUT_NAME"],
                    embedding_function=embeddings,
                    persist_directory=str(persist_directory),
                )
            case _:
                raise ValueError("Invalid vector store option")

    def add(self, docs: list[Document]) -> None:
        self.vectorstore.add_documents(docs)


def build_vectorstore(
    embeddings_opt: EmbeddingOpt = EmbeddingOpt.AZURE,
    vectorstore_opt: VectorStoreOpt = VectorStoreOpt.CHROMA,
) -> None:
    embeddings = create_embeddings(opt=embeddings_opt)
    vs = FduaVectorStore(embeddings, opt=vectorstore_opt)

    docs = load_documents()
    for doc in tqdm(docs, desc="adding documents to vectorstore"):
        split_doc = split_document(doc)
        vs.add(split_doc)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=EmbeddingOpt,
        default=EmbeddingOpt.AZURE,
        choices=list(EmbeddingOpt),
        help="the type of embeddings to use",
    )
    parser.add_argument(
        "--vectorstore",
        type=VectorStoreOpt,
        default=VectorStoreOpt.CHROMA,
        choices=list(VectorStoreOpt),
        help="the type of vectorstore to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_vectorstore(embeddings_opt=args.embeddings, vectorstore_opt=args.vectorstore)


if __name__ == "__main__":
    main()
