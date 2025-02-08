from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from fdua_competition.models import create_chat_model, create_embeddings


def test_create_embeddings():
    assert isinstance(create_embeddings(), Embeddings)


def test_create_chat_model():
    assert isinstance(create_chat_model(), BaseChatModel)
