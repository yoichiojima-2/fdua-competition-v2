from fdua_competition import chain


def test_document_dir():
    assert chain.get_documents_dir().name == "documents"


def test_get_pages():
    pages = list(chain.get_pages("1.pdf"))
    assert len(pages) > 0


def test_get_vectorstore():
    vectorstore = chain.get_vectorstore()
    assert vectorstore


def test_prompt():
    system_prompt = "test system prompt: {language}"
    query = "test query"
    vectorstore = chain.get_vectorstore()
    prompt = chain.get_prompt(system_prompt, query, vectorstore)
    assert prompt


def test_get_chat_model():
    chat_model = chain.get_chat_model()
    assert chat_model


def test_get_queries():
    queries = chain.get_queries()
    assert isinstance(queries, list)
    assert queries
