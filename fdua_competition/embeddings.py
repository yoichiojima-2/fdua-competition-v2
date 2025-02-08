from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings

from fdua_competition.enums import EmbeddingOpt


def create_embeddings(opt: EmbeddingOpt = EmbeddingOpt.AZURE) -> Embeddings:
    match opt:
        case EmbeddingOpt.AZURE:
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError("Invalid embedding option")
