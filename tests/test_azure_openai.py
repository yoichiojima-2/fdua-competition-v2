from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv("secrets/.env")


def test_chat():
    llm = AzureOpenAI()
    messages = [{"role": "user", "content": "test"}]
    chat_res = llm.chat.completions.create(model="4omini", messages=messages)
    assert len(chat_res.choices) > 0


def test_embedding():
    llm = AzureOpenAI()
    embedding_res = llm.embeddings.create(input="this is a test", model="embedding")
    assert embedding_res.data[0].embedding
