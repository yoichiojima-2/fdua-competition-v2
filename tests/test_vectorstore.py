import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

sys.path.append(str(Path(__file__).parent.parent))

from fdua_competition.enums import EmbeddingModelOption, Mode, VectorStoreOption
from fdua_competition.utils import get_root
from fdua_competition.vectorstore import (
    _add_documents_with_retry,
    _add_pages_to_vectorstore_in_batches,
    _get_existing_sources_in_vectorstore,
    add_documents_to_vectorstore,
    get_document_list,
    get_documents_dir,
    get_embedding_model,
    load_pages,
    prepare_vectorstore,
)

load_dotenv(Path(__file__).parent.parent / "secrets/.env")


def test_get_document_dir():
    assert get_documents_dir(Mode.TEST).name == "documents"


def test_get_document_list():
    doc_dir = get_documents_dir(Mode.TEST)
    docs = get_document_list(doc_dir)
    assert len(docs)


def test_load_pages():
    pages = list(load_pages(get_documents_dir(Mode.TEST) / "1.pdf"))
    assert len(pages)


def test_get_embedding_model():
    model = get_embedding_model(EmbeddingModelOption.AZURE)
    assert isinstance(model, OpenAIEmbeddings)
    documents = ["this", "is", "a", "test"]
    embeddings = model.embed_documents(documents)
    assert len(embeddings) == len(documents)
    try:
        get_embedding_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_prepare_vectorstore():
    embeddings = AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE)
    in_memory_vectorstore = prepare_vectorstore(
        output_name="test_in_memory", opt=VectorStoreOption.IN_MEMORY, embeddings=embeddings
    )
    assert isinstance(in_memory_vectorstore, InMemoryVectorStore)
    chroma = prepare_vectorstore(output_name="test_chroma", opt=VectorStoreOption.CHROMA, embeddings=embeddings)
    assert isinstance(chroma, Chroma)
    assert Path(get_root() / "vectorstores/chroma").exists()
    try:
        prepare_vectorstore(output_name="unittest", opt="should_raise_err", embeddings=embeddings)
        assert False
    except ValueError:
        assert True


def test_get_existing_source_in_vectorstore():
    db = Chroma(
        collection_name="test-get-existing-source-in-vectorstore",
        embedding_function=AzureOpenAIEmbeddings(azure_deployment="embedding"),
        persist_directory=str(get_root() / "vectorstores/chroma"),
    )
    db.add_documents([Document(page_content="this is a test", metadata={"source": "test"})])
    sources = _get_existing_sources_in_vectorstore(db)
    assert len(sources)


def test_chroma_minimal():
    embeddings = AzureOpenAIEmbeddings(azure_deployment="embedding")
    chroma = Chroma(
        collection_name="test-chroma-minimal",
        embedding_function=embeddings,
        persist_directory=str(get_root() / "vectorstores/chroma"),
    )
    assert isinstance(chroma, Chroma)
    doc = Document(page_content="this is a test", metadata={"title": "test"})
    chroma.add_documents(documents=[doc], ids=["1"])
    assert chroma.get(ids=["1"])


def test_add_pages_to_vectorstore_in_batches():
    collection_name = "test-add-pages-to-vectorstore-in-batches"
    vectorstore = prepare_vectorstore(
        output_name=collection_name,
        opt=VectorStoreOption.CHROMA,
        embeddings=AzureOpenAIEmbeddings(azure_deployment="embedding"),
    )
    pages = list(load_pages(get_documents_dir(Mode.TEST) / "2.pdf"))
    _add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages[:3])
    assert True


def test_add_documents_minimally():
    db = prepare_vectorstore(
        output_name="test-add-documents-minimally",
        opt=VectorStoreOption.CHROMA,
        embeddings=AzureOpenAIEmbeddings(azure_deployment="embedding"),
    )
    documents = [Document(page_content="this is a test", metadata={"title": "test"})]
    db.add_documents(documents)
    assert db.get(ids=["1"])


def test_add_documents_with_retry():
    vectorstore = prepare_vectorstore(
        output_name="test-add-documents-with-retry",
        opt=VectorStoreOption.CHROMA,
        embeddings=AzureOpenAIEmbeddings(azure_deployment="embedding"),
    )
    pages = [next(load_pages(get_documents_dir(Mode.TEST) / "1.pdf"))]
    _add_documents_with_retry(vectorstore, pages)
    assert True


def test_add_documents_to_vectorstore():
    db = prepare_vectorstore(
        output_name="test-add-documents-to-vectorstore",
        opt=VectorStoreOption.CHROMA,
        embeddings=AzureOpenAIEmbeddings(azure_deployment="embedding"),
    )
    docs = get_document_list(get_documents_dir(Mode.TEST))
    add_documents_to_vectorstore(documents=docs[:3], vectorstore=db)
    assert db.get(ids=["1"])
