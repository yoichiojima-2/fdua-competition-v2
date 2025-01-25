import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT_URL = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")


def vectorize_text(text: str) -> list[str]:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_URL,
        api_version=AZURE_OPENAI_VERSION,
    )

    res = client.embeddings.create(input=text, model="embedding")

    return res.data[0].embedding
