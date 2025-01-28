from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings

from fdua_competition.main import (
    add_documents_with_retry,
    get_chat_model,
    get_documents_dir,
    get_embedding_model,
    get_pages,
    get_prompt,
    get_queries,
    get_root,
    get_vectorstore,
)


def test_get_document_dir():
    assert get_documents_dir().name == "documents"


def test_get_pages():
    pages = list(get_pages("1.pdf"))
    assert len(pages)


def test_get_prompt():
    system_prompt = "test system prompt: {language}"
    query = "test query"
    embeddings = AzureOpenAIEmbeddings(model="embedding")
    vectorstore = InMemoryVectorStore(embeddings)
    prompt = get_prompt(system_prompt, query, vectorstore)
    assert prompt


def test_get_chat_model():
    chat_model = get_chat_model("azure")
    assert isinstance(chat_model, AzureChatOpenAI)

    try:
        get_chat_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_get_queries():
    queries = get_queries()
    assert isinstance(queries, list)
    assert queries


def test_get_embedding_model():
    model = get_embedding_model("azure")
    assert isinstance(model, OpenAIEmbeddings)

    try:
        get_embedding_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_get_vectorstore():
    embeddings = get_embedding_model("azure")

    in_memory_vectorstore = get_vectorstore("in-memory", embeddings)
    assert isinstance(in_memory_vectorstore, InMemoryVectorStore)

    chroma = get_vectorstore("chroma", embeddings)
    assert isinstance(chroma, Chroma)
    assert Path(get_root() / "vectorstore/chroma").exists()

    try:
        get_vectorstore("should_raise_err", embeddings)
        assert False
    except ValueError:
        assert True


def test_add_documents_with_retry():
    embeddings = AzureOpenAIEmbeddings(model="embedding")
    vectorstore = InMemoryVectorStore(embeddings)
    pages = get_pages("1.pdf")
    add_documents_with_retry(vectorstore, pages)
    assert True
