import os
from openai import AzureOpenAI
from dotenv import load_dotenv


def vectorize_text(text: str) -> list[str]:
    load_dotenv()

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
    )

    res = client.embeddings.create(input=text, model="embedding")

    return res.data[0].embedding
