"""
vectorstoreの構築
"""

from pathlib import Path
import typing as t
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
    """
    指定されたモードに基づいてPDFが入っているパスを取得する
    args:
        mode (Mode): 動作モード(TEST または SUBMIT)
    returns:
        Path: PDFが入っているパス
    Raises:
        ValueError: 未知のモードが指定された場合
    """
    match mode:
        case Mode.TEST:
            return get_root() / "validation/documents"

        case Mode.SUBMIT:
            return get_root() / "documents"

        case _:
            raise ValueError(f"): unknown mode: {mode}")


def get_document_list(document_dir: Path) -> list[Path]:
    """
    指定されたディレクトリ内の PDF ファイルのパスのリストを取得する
    args:
        document_dir (Path): Documentディレクトリのパス
    returns:
        list[Path]: PDF ファイルのパスのリスト
    """
    return [path for path in document_dir.glob("*.pdf")]


def load_pages(path: Path) -> t.Iterable[Document]:
    """
    PDF ファイルからページごとのDocument(Document)を読み込むジェネレーター
    args:
        path (Path): PDF ファイルのパス
    yields:
        Document: 読み込まれたDocumentページ
    """
    for doc in PyPDFium2Loader(path).lazy_load():
        yield doc


def get_embedding_model(opt: EmbeddingModelOption) -> OpenAIEmbeddings:
    """
    指定されたembeddingモデルオプションに基づいてembeddingモデルを取得する
    args:
        opt (EmbeddingModelOption): embeddingモデルのオプション
    returns:
        OpenAIEmbeddings: embeddingモデルのインスタンス
    Raises:
        ValueError: 未知のモデルオプションが指定された場合
    """
    match opt:
        case EmbeddingModelOption.AZURE:
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"): unknown model: {opt}")


def prepare_vectorstore(output_name: str, opt: VectorStoreOption, embeddings: OpenAIEmbeddings) -> VectorStore:
    """
    指定されたパラメータに基づいてvectorstoreを準備する
    args:
        output_name (str): vectorstoreのコレクション名
        opt (VectorStoreOption): vectorstoreのオプション
        embeddings (OpenAIEmbeddings): embeddingモデルのインスタンス
    returns:
        VectorStore: 構築されたvectorstore
    Raises:
        ValueError: 未知のvectorstoreオプションが指定された場合
    """
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
    """
    vectorstore内に既に登録されているDocumentのソース一覧を取得する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
    returns:
        set[str]: 登録済みDocumentのソースの集合
    """
    return {metadata.get("source") for metadata in vectorstore.get().get("metadatas")}


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def _add_documents_with_retry(vectorstore: VectorStore, batch: list[Document]) -> None:
    """
    リトライ機能付きでDocumentのバッチをvectorstoreに追加する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
        batch (list[Document]): 追加するDocumentのバッチ
    """
    vectorstore.add_documents(batch)


def _add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: t.Iterable[Document], batch_size: int = 8) -> None:
    """
    Documentページをバッチごとにvectorstoreへ追加する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
        pages (Iterable[Document]): 追加するDocumentページのIterable
        batch_size (int, optional): バッチサイズ (デフォルトは8)
    """
    batch = []

    for page in tqdm(pages, desc="adding pages.."):
        batch.append(page)

        if len(batch) == batch_size:
            _add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []

    if batch:
        _add_documents_with_retry(vectorstore=vectorstore, batch=batch)


def add_documents_to_vectorstore(documents: list[Path], vectorstore: VectorStore) -> None:
    """
    指定された PDF ファイル群をvectorstoreに追加する
    既に登録されているDocumentはスキップされる
    args:
        documents (list[Path]): PDF ファイルのパスのリスト
        vectorstore (VectorStore): vectorstoreのインスタンス
    """
    existing_sources = _get_existing_sources_in_vectorstore(vectorstore)

    for path in documents:
        if str(path) in existing_sources:
            print(f"[add_document_to_vectorstore] skipping existing document: {path}")
            continue

        print(f"[add_document_to_vectorstore] adding document to vectorstore: {path}")
        pages = load_pages(path=path)
        _add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages)


def build_vectorstore(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> VectorStore:
    """
    指定されたパラメータに基づいてvectorstoreを構築する
    PDF ファイルを読み込み, Documentをvectorstoreに追加する
    args:
        output_name (str): vectorstoreのコレクション名
        mode (Mode): 動作モード(TEST または SUBMIT)
        vectorstore_option (VectorStoreOption): 使用するvectorstoreのオプション
    returns:
        VectorStore: 構築されたvectorstore
    """
    embeddings = get_embedding_model(EmbeddingModelOption.AZURE)
    vectorstore = prepare_vectorstore(output_name=output_name, opt=vectorstore_option, embeddings=embeddings)
    docs = get_document_list(document_dir=get_documents_dir(mode=mode))
    add_documents_to_vectorstore(docs, vectorstore)

    return vectorstore


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def retrieve_context(vectorstore: VectorStore, query: str) -> str:
    """
    指定されたクエリに対して, 関連Documentから文脈を構築する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
        query (str): クエリ
    returns:
        str: 構築された文脈情報
    """
    pages = vectorstore.as_retriever().invoke(query)

    # 各ページの内容とメタデータを整形して文脈として連結
    contexts = ["\n".join([f"page_content: {page.page_content}", f"metadata: {page.metadata}"]) for page in pages]

    return "\n---\n".join(contexts)
