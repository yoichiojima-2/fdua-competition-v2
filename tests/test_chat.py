import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

sys.path.append(str(Path(__file__).parent.parent))
from fdua_competition.chat import get_chat_model, get_prompt_template, invoke_chain_with_retry
from fdua_competition.enums import ChatModelOption, Mode

load_dotenv(Path(__file__).parent.parent / "secrets/.env")



def test_get_chat_model():
    chat_model = get_chat_model(ChatModelOption.AZURE)
    assert isinstance(chat_model, AzureChatOpenAI)
    try:
        get_chat_model("should_raise_err")
        assert False
    except ValueError:
        assert True


def test_get_prompt_template():
    prompt = get_prompt_template()
    assert isinstance(prompt, ChatPromptTemplate)


class SimpleChain(Runnable):
    def invoke(self, input_data):
        return "invoked"


def test_invoke_chain_with_retry():
    chain = SimpleChain()
    payload = {"test": "this is atest"}
    print(invoke_chain_with_retry(chain, payload))
