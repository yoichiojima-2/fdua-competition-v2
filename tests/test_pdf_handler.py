from langchain_core.documents import Document

from fdua_competition.pdf_handler import load_documents, split_document


def test_load_documents():
    docs = load_documents()
    assert docs
    assert isinstance(docs[0], Document)


def test_split_document():
    doc = Document(page_content="this is a test", metadata={"source": "test-source", "page": 1})
    split_doc = split_document(doc)
    first_element = split_doc[0]
    first_element.metadata == doc.metadata
    assert isinstance(first_element, Document)
