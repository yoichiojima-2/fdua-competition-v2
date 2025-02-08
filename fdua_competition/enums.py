from enum import Enum


class Mode(Enum):
    SUBMIT = "submit"
    TEST = "test"


class EmbeddingOpt(Enum):
    AZURE = "azure"


class ChatOpt(Enum):
    AZURE = "azure"
