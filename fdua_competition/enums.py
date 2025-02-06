"""
各種オプションを定義するenum
"""

from enum import Enum


class Mode(Enum):
    """
    動作モードを表すenum
    attributes:
        SUBMIT: 提出用モード
        TEST: テスト用モード
    """
    SUBMIT = "submit"  # 提出用モード
    TEST = "test"  # テスト用モード


class ChatModelOption(Enum):
    """
    chatモデルのオプションを表すenum
    attributes:
        AZURE: Azure のchatモデルを使用
    """
    AZURE = "azure"


class EmbeddingModelOption(Enum):
    """
    embeddingモデルのオプションを表すenum.
    attributes:
        AZURE: Azure のembeddingモデルを使用
    """
    AZURE = "azure"


class VectorStoreOption(Enum):
    """
    vectorstoreのオプションを表すenum.
    attributes:
        CHROMA: Chroma vectorstore
        IN_MEMORY: インメモリのvectorstore
    """
    CHROMA = "chroma"
    IN_MEMORY = "in-memory"
