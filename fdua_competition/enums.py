"""
各種オプションを定義する列挙型
"""

from enum import Enum


class Mode(Enum):
    """
    動作モードを表す列挙型

    Attributes:
        SUBMIT: 提出用モード
        TEST: テスト用モード
    """

    SUBMIT = "submit"  # 提出用モード
    TEST = "test"  # テスト用モード


class ChatModelOption(Enum):
    """
    チャットモデルのオプションを表す列挙型

    Attributes:
        AZURE: Azure のチャットモデルを使用
    """

    AZURE = "azure"


class EmbeddingModelOption(Enum):
    """
    埋め込みモデルのオプションを表す列挙型。

    Attributes:
        AZURE: Azure の埋め込みモデルを使用
    """

    AZURE = "azure"


class VectorStoreOption(Enum):
    """
    ベクトルストアのオプションを表す列挙型。

    Attributes:
        CHROMA: Chroma ベクトルストア
        IN_MEMORY: インメモリのベクトルストア
    """

    CHROMA = "chroma"
    IN_MEMORY = "in-memory"
