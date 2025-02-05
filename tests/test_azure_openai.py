from dotenv import load_dotenv
from openai import AzureOpenAI


def test_azure_openai():
    load_dotenv("secrets/.env")

    llm = AzureOpenAI()
    messages = [{"role": "user", "content": "test"}]
    chat_res = llm.chat.completions.create(model="4omini", messages=messages)
    assert len(chat_res.choices) > 0

    embedding_res = llm.embeddings.create(input="this is a test", model="embedding")
    assert embedding_res.data[0].embedding
