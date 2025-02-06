from langchain_openai import AzureChatOpenAI

from fdua_competition.enums import ChatModelOption
from fdua_competition.rag import get_chat_model, read_prompt


def test_get_chat_model():
    chat_model = get_chat_model(ChatModelOption.AZURE)
    assert isinstance(chat_model, AzureChatOpenAI)
    try:
        get_chat_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_read_prompt():
    assert read_prompt("test") == "this is a test"
