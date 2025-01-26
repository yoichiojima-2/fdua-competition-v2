from openai import AzureOpenAI
from dotenv import load_dotenv


def vectorize_text(text: str) -> list[str]:
    load_dotenv()
    client = AzureOpenAI()
    res = client.embeddings.create(input=text, model="embedding")
    return res.data[0].embedding
