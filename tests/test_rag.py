import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

sys.path.append(str(Path(__file__).parent.parent))
from fdua_competition.enums import ChatModelOption
from fdua_competition.rag import get_chat_model, read_prompt

load_dotenv(Path(__file__).parent.parent / "secrets/.env")


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
