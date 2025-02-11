from langchain_core.documents import Document

from fdua_competition.pdf_handler import load_documents


def test_load_documents():
    docs = load_documents()
    assert docs
    assert isinstance(docs[0], Document)
