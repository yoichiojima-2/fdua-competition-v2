from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from fdua_competition.enums import ChatOpt, EmbeddingOpt


TEMPRATURE = 0

def create_embeddings(opt: EmbeddingOpt = EmbeddingOpt.AZURE) -> Embeddings:
    match opt:
        case EmbeddingOpt.AZURE:
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError("Invalid embedding option")


def create_chat_model(opt: ChatOpt = ChatOpt.AZURE) -> BaseChatModel:
    match opt:
        case ChatOpt.AZURE:
            return AzureChatOpenAI(azure_deployment="4omini", temperature=TEMPRATURE)
        case _:
            raise ValueError("Invalid chat option")
