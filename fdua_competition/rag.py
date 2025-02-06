"""
RAGに関連するクラスと関数を定義
"""

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from fdua_competition.enums import ChatModelOption
from fdua_competition.utils import print_before_retry
from fdua_competition.vectorstore import retrieve_context


def get_chat_model(opt: ChatModelOption) -> ChatOpenAI:
    """
    指定されたチャットモデルオプションに基づいてチャットモデルを取得する
    Args:
        opt (ChatModelOption): 使用するチャットモデルのオプション
    Returns:
        ChatOpenAI: 選択されたチャットモデルのインスタンス
    Raises:
        ValueError: 未知のモデルオプションが指定された場合
    """
    match opt:
        case ChatModelOption.AZURE:
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


def read_prompt(target: str) -> str:
    """
    プロンプトファイルからテキストを読み込む
    Args:
        target (str): プロンプトファイル名（拡張子なし）
    Returns:
        str: プロンプトの内容
    """
    return (Path(__file__).parent / f"prompts/{target}.txt").read_text()


# =============================================================================
# [start: base rag class]
# =============================================================================
@dataclass
class RAG(ABC):
    """
    RAGの基底クラス
    Attributes:
        vectorstore (VectorStore): 使用するベクトルストア
        chat_model_option (ChatModelOption): 使用するチャットモデルのオプション(デフォルトはAZURE)
    """

    vectorstore: VectorStore
    chat_model_option: ChatModelOption = ChatModelOption.AZURE

    @property
    @abstractmethod
    def prompt_template(self) -> ChatPromptTemplate:
        """
        プロンプトテンプレートを返すプロパティ. サブクラスで実装する
        """
        ...

    @abstractmethod
    def build_payload(self, query: str) -> dict[str, t.Any]:
        """
        クエリに応じたペイロードを構築するメソッド. サブクラスで実装する
        Args:
            query (str): ユーザーからの質問
        Returns:
            dict[str, Any]: ペイロードの辞書
        """
        ...

    @property
    @abstractmethod
    def output_structure(self) -> BaseModel:
        """
        出力構造 (Pydanticモデル)を返すプロパティ. サブクラスで実装する
        """
        ...

    @property
    def chat_model(self) -> ChatOpenAI:
        """
        構造化出力を設定したチャットモデルを返すプロパティ
        Returns:
            ChatOpenAI: 構造化出力を持つチャットモデルのインスタンス
        """
        return get_chat_model(self.chat_model_option).with_structured_output(self.output_structure)

    @property
    def chain(self) -> Runnable:
        """
        プロンプトテンプレートとチャットモデルを連結したチェーンを返すプロパティ
        Returns:
            Runnable: チェーンの実行可能なオブジェクト
        """
        return self.prompt_template | self.chat_model

    @retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
    def invoke(self, query: str) -> BaseModel:
        """
        クエリに対して RAG チェーンを実行し、回答を取得する
        Args:
            query (str): ユーザーからの質問
        Returns:
            BaseModel: 構造化された回答を含む Pydantic モデル
        """
        payload = self.build_payload(query)

        # ベクトルストアから文脈を構築してペイロードに追加
        payload["context"] = retrieve_context(vectorstore=self.vectorstore, query=query)

        return self.chain.invoke(payload)


# =============================================================================
# [end: base rag class]
# =============================================================================


# =============================================================================
# [start: research_assistant]
# =============================================================================
class ResearchAssistantResponse(BaseModel):
    """
    ResearchAssistant の応答を表す Pydantic モデル。
    Attributes:
        query (str): 質問内容
        response (str): 回答内容
        reason (str): 回答の理由
        organization_name (str): 関連する組織名
        contexts (list[str]): 回答に基づく文脈情報のリスト
    """

    query: str = Field(description="the query that was asked.")
    response: str = Field(description="the answer for the given query")
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    contexts: list[str] = Field(description="the context that the response was based on with its file path and page number.")


class ResearchAssistant(RAG):
    """
    質問に対する文脈に基づいた回答を生成するRAGの実装
    Attributes:
        vectorstore (VectorStore): 使用するベクトルストア
        chat_model_option (ChatModelOption): 使用するチャットモデルのオプション
        language (str): 使用する言語（デフォルトは "japanese"）
    """

    def __init__(
        self, vectorstore: VectorStore, chat_model_option: ChatModelOption = ChatModelOption.AZURE, language: str = "japanese"
    ):
        """
        ResearchAssistant のインスタンスを初期化する
        Args:
            vectorstore (VectorStore): 使用するベクトルストア
            chat_model_option (ChatModelOption, optional): チャットモデルのオプション。デフォルトは ChatModelOption.AZURE
            language (str, optional): 使用する言語。デフォルトは "japanese"
        """
        super().__init__(vectorstore=vectorstore, chat_model_option=chat_model_option)
        self.language = language

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """
        ResearchAssistant 用のプロンプトテンプレートを返すプロパティ
        Returns:
            ChatPromptTemplate: プロンプトテンプレートのインスタンス
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("system", "context:\n{context}"),
                ("user", "query:\n{query}"),
            ]
        )

    def build_payload(self, query: str) -> dict[str, t.Any]:
        """
        クエリに基づいてペイロードを構築する
        Args:
            query (str): ユーザーからの質問
        Returns:
            dict[str, Any]: ペイロードの辞書
        """
        return {"system_prompt": read_prompt("research_assistant"), "query": query, "language": self.language}

    @property
    def output_structure(self) -> BaseModel:
        """
        ResearchAssistant の出力構造を返すプロパティ
        Returns:
            BaseModel: ResearchAssistantResponseモデル
        """
        return ResearchAssistantResponse


# =============================================================================
# [end: research_assistant]
# =============================================================================
