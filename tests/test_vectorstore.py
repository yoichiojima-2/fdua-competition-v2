import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).parent.parent))

from fdua_competition.enums import EmbeddingModelOption, Mode, VectorStoreOption
from fdua_competition.utils import get_root
from fdua_competition.vectorstore import (
    _add_documents_with_retry,
    _get_existing_sources_in_vectorstore,
    add_documents_to_vectorstore,
    get_document_list,
    get_documents_dir,
    get_embedding_model,
    load_pages,
    prepare_vectorstore,
    _add_pages_to_vectorstore_in_batches,
)

load_dotenv(Path(__file__).parent.parent / "secrets/.env")


def test_get_document_dir():
    assert get_documents_dir(Mode.TEST).name == "documents"


def test_get_document_list():
    documentlist = get_document_list(get_documents_dir(Mode.TEST))
    assert isinstance(documentlist, list)
    assert documentlist


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
    in_memory_vectorstore = prepare_vectorstore(output_name=Mode.TEST, opt=VectorStoreOption.IN_MEMORY, embeddings=embeddings)
    assert isinstance(in_memory_vectorstore, InMemoryVectorStore)

    chroma = prepare_vectorstore(output_name=VectorStoreOption.CHROMA, opt=VectorStoreOption.CHROMA, embeddings=embeddings)
    assert isinstance(chroma, Chroma)
    assert Path(get_root() / "vectorstore/chroma").exists()

    try:
        prepare_vectorstore(output_name="unittest", opt="should_raise_err", embeddings=embeddings)
        assert False
    except ValueError:
        assert True


def test_get_existing_source_in_vectorstore():
    ...


def test_add_pages_to_vectorstore_in_batches():
    db_path = get_root() / "vectorstore/chroma/fdua-competition-test-addrpages-to-vectorstore-in-batches"
    db.unlink(missing_ok=True)
    vectorstore = prepare_vectorstore(
        output_name=db_path.name,
        opt=VectorStoreOption.CHROMA,
        embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE),
    )
    pages = list(load_pages(get_documents_dir(Mode.TEST) / "1.pdf"))
    _add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages[:3])
    assert True


def test_chroma_minimal():
    embeddings = AzureOpenAIEmbeddings(azure_deployment="embedding")
    chroma = Chroma(
        collection_name="test-chroma-minimal",
        embedding_function=embeddings,
        persist_directory="test_chroma_db"
    )
    assert isinstance(chroma, Chroma)
    doc = Document(page_content="this is a test", metadata={"title": "test"})
    chroma.add_documents(documents=[doc], ids=["1"])
    assert True



def test_add_documents_minimally():
    db_path = get_root() / "vectorstore/chroma/fdua-competition-test-add-documents-minimally"
    db_path.unlink(missing_ok=True)
    vectorstore = prepare_vectorstore(
        output_name=db_path.name,
        opt=VectorStoreOption.CHROMA,
        embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE),
    )
    documents = [Document(page_content="this is a test", metadata={"title": "test"})]
    vectorstore.add_documents(documents)



# def test_add_documents_to_vectorstore():
#     db_path = get_root() / "vectorstore/chroma/fdua-competition-test-add-documents-to-vectorstore"
#     db_path.unlink(missing_ok=True)
#     vectorstore = prepare_vectorstore(
#         output_name=db_path.name,
#         opt=VectorStoreOption.CHROMA,
#         embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE),
#     )
#     docs = get_document_list(get_documents_dir(Mode.TEST))
#     add_documents_to_vectorstore(documents=docs[:3], vectorstore=vectorstore)
#     assert True


# def test_add_documents_with_retry():
#     vectorstore = prepare_vectorstore(
#         output_name="unittest",
#         opt=VectorStoreOption.CHROMA,
#         embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE),
#     )
#     pages = [next(load_pages(get_documents_dir(Mode.TEST) / "1.pdf"))]
#     _add_documents_with_retry(vectorstore, pages)
#     assert True
