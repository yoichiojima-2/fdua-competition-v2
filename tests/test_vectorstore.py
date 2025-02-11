# import Document
from langchain_core.documents import Document

from fdua_competition.enums import Mode
from fdua_competition.models import create_embeddings
from fdua_competition.vectorstore import FduaVectorStore


def test_vectorstore():
    embeddings = create_embeddings()
    vs = FduaVectorStore(mode=Mode("test"), embeddings=embeddings)
    docs = [
        Document(
            page_content="test page 1",
            metadata={"source": "dummy-source", "page": 1},
        ),
        Document(
            page_content="test page 1",
            metadata={"source": "dummy-source", "page": 2},
        ),
    ]
    vs.add(docs)
    assert vs.get()
