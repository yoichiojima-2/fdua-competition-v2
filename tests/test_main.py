from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings

from fdua_competition.main import get_chat_model, get_documents_dir, get_pages, get_prompt, get_queries


def test_document_dir():
    assert get_documents_dir().name == "documents"


def test_get_pages():
    pages = list(get_pages("1.pdf"))
    assert len(pages) > 0


def test_prompt():
    system_prompt = "test system prompt: {language}"
    query = "test query"
    embeddings = AzureOpenAIEmbeddings(model="embedding")
    vectorstore = InMemoryVectorStore(embeddings)
    prompt = get_prompt(system_prompt, query, vectorstore)
    assert prompt


def test_get_chat_model():
    chat_model = get_chat_model("azure")
    assert chat_model
    try:
        get_chat_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_get_queries():
    queries = get_queries()
    assert isinstance(queries, list)
    assert queries
