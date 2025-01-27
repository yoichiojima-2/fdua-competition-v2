from dotenv import load_dotenv
from openai import AzureOpenAI


def test_azure_openai():
    load_dotenv()

    llm = AzureOpenAI()
    messages = [{"role": "user", "content": "test"}]
    res = llm.chat.completions.create(model="4omini", messages=messages)

    assert len(res.choices) > 0
