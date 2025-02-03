from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings

from main import (
    ChatModelOption,
    EmbeddingModelOption,
    Mode,
    VectorStoreOption,
    add_documents_with_retry,
    get_chat_model,
    get_documents_dir,
    get_embedding_model,
    get_pages,
    get_prompt_template,
    get_queries,
    get_root,
    get_vectorstore,
)


def test_get_document_dir():
    assert get_documents_dir(Mode.TEST).name == "documents"


def test_get_pages():
    pages = list(get_pages(get_documents_dir(Mode.TEST) / "1.pdf"))
    assert len(pages)


def test_get_prompt_template():
    prompt = get_prompt_template()
    assert isinstance(prompt, ChatPromptTemplate)


def test_get_chat_model():
    chat_model = get_chat_model(ChatModelOption.AZURE)
    assert isinstance(chat_model, AzureChatOpenAI)

    try:
        get_chat_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_get_queries():
    queries = get_queries(mode=Mode.TEST)
    assert isinstance(queries, list)
    assert queries


def test_get_embedding_model():
    model = get_embedding_model(EmbeddingModelOption.AZURE)
    assert isinstance(model, OpenAIEmbeddings)

    try:
        get_embedding_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_get_vectorstore():
    embeddings = AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE)
    in_memory_vectorstore = get_vectorstore(output_name=Mode.TEST, opt=VectorStoreOption.IN_MEMORY, embeddings=embeddings)
    assert isinstance(in_memory_vectorstore, InMemoryVectorStore)

    chroma = get_vectorstore(output_name=VectorStoreOption.CHROMA, opt=VectorStoreOption.CHROMA, embeddings=embeddings)
    assert isinstance(chroma, Chroma)
    assert Path(get_root() / "vectorstore/chroma").exists()

    try:
        get_vectorstore(output_name="test", opt="should_raise_err", embeddings=embeddings)
        assert False
    except ValueError:
        assert True


def test_add_documents_with_retry():
    embeddings = AzureOpenAIEmbeddings(model=EmbeddingModelOption.AZURE)
    vectorstore = InMemoryVectorStore(embeddings)
    pages = get_pages("1.pdf")
    add_documents_with_retry(vectorstore, pages)
    assert True
