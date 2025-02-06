"""
各種オプションを定義するenum
"""

from enum import Enum


class Mode(Enum):
    """
    動作モードを表すenum

    Attributes:
        SUBMIT: 提出用モード
        TEST: テスト用モード
    """

    SUBMIT = "submit"  # 提出用モード
    TEST = "test"  # テスト用モード


class ChatModelOption(Enum):
    """
    chatモデルのオプションを表すenum

    Attributes:
        AZURE: Azure のchatモデルを使用
    """

    AZURE = "azure"


class EmbeddingModelOption(Enum):
    """
    embeddingモデルのオプションを表すenum.

    Attributes:
        AZURE: Azure のembeddingモデルを使用
    """

    AZURE = "azure"


class VectorStoreOption(Enum):
    """
    vectorstoreのオプションを表すenum.

    Attributes:
        CHROMA: Chroma vectorstore
        IN_MEMORY: インメモリのvectorstore
    """

    CHROMA = "chroma"
    IN_MEMORY = "in-memory"
