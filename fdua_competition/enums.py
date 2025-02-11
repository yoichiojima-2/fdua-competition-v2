from enum import Enum


class Mode(Enum):
    SUBMIT = "submit"
    TEST = "test"


class EmbeddingOpt(Enum):
    AZURE = "azure"


class ChatOpt(Enum):
    AZURE = "azure"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
