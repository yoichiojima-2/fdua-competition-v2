from dotenv import load_dotenv
from openai import AzureOpenAI


def test_azure_openai():
    load_dotenv()
    res = AzureOpenAI().chat.completions.create(model="4omini", messages=[{"role": "user", "content": "test"}])
    assert len(res.choices) > 0
