from enum import Enum


class Mode(Enum):
    SUBMIT = "submit"
    TEST = "test"


class ChatModelOption(Enum):
    AZURE = "azure"


class EmbeddingModelOption(Enum):
    AZURE = "azure"


class VectorStoreOption(Enum):
    CHROMA = "chroma"
    IN_MEMORY = "in-memory"
