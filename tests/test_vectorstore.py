import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

sys.path.append(str(Path(__file__).parent.parent))

from fdua_competition.enums import (EmbeddingModelOption, Mode,
                                    VectorStoreOption)
from fdua_competition.utils import get_root
from fdua_competition.vectorstore import (_add_documents_with_retry,
                                          _get_existing_sources_in_vectorstore,
                                          add_documents_to_vectorstore,
                                          get_document_list, get_documents_dir,
                                          get_embedding_model, load_pages,
                                          prepare_vectorstore)

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
        prepare_vectorstore(output_name="unit-test", opt="should_raise_err", embeddings=embeddings)
        assert False
    except ValueError:
        assert True


def test_get_existing_source_in_vectorstore():
    vectorstore = prepare_vectorstore(output_name="unit-test", opt=VectorStoreOption.CHROMA, embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE))
    assert _get_existing_sources_in_vectorstore(vectorstore) == set()

def test_add_documents_to_vectorstore():
    vectorstore = prepare_vectorstore(output_name="unit-test", opt=VectorStoreOption.CHROMA, embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE))
    docs = get_document_list(get_documents_dir(Mode.TEST))
    add_documents_to_vectorstore(documents=docs[5], vectorstore=vectorstore)
    assert True

def test_add_documents_with_retry():
    vectorstore = prepare_vectorstore(output_name="unit-test", opt=VectorStoreOption.CHROMA, embeddings=AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE))
    pages = [next(load_pages(get_documents_dir(Mode.TEST) / "1.pdf"))]
    _add_documents_with_retry(vectorstore, pages)
    assert True
