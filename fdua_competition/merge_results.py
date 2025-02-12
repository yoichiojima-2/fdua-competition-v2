import textwrap

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.models import create_chat_model
from fdua_competition.logging_config import logger
from fdua_competition.utils import before_sleep_hook, dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore


class MergeResultsOutput(BaseModel):
    query: str = Field(description="The original user question.")
    res_index: str = Field(description="The answer generated using context retrieved via index-based search.")
    certainty_index: float = Field(description="The certainty score for the index-based answer.")
    res_simple: str = Field(description="The answer generated using context retrieved via simple retrieval.")
    certainty_simple: float = Field(description="The certainty score for the simple retrieval answer.")
    output: str = Field(description="The final merged answer after evaluating both results.")
    reason: str = Field(description="A brief explanation of how the merged answer was derived.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def merge_results(res_index: AnswerQueryOutput, res_simple: AnswerQueryOutput, vectorstore: FduaVectorStore, query: str) -> MergeResultsOutput:
    logger.info("[merge_results] merging results..")
    role = textwrap.dedent(
        """
        あなたはリサーチアシスタントであり、同じクエリに対して異なる検索方法で生成された2つの回答を統合する任務を持っています。
        以下に2つの結果セットがあります：

        - **結果1 (res_index):** インデックスベースの検索を使用して取得したコンテキストに基づく回答。
        - **結果2 (res_simple):** シンプルな検索方法を使用して取得したコンテキストに基づく回答。

        *結果2は結果1よりも信頼性が低い可能性があります。必要に応じて参照できるように、取得したコンテキストを添付します。*

        あなたの任務は、以下の要件を満たす統合された回答を作成することです：

        1. **クエリ:** "query" フィールドには元のユーザーの質問を使用してください。
        2. **元の回答:** "res_index", "certainty_index", "res_simple", "certainty_simple" フィールドに元の回答と対応する確信度スコアを含めてください。
        3. **統合された回答 ("output"):**
           - 1つの結果が明らかに信頼性が高い場合、その回答を使用してください。
           - 両方の結果が確信度と内容で類似している場合、簡潔な回答を合成してください。
           - 両方が情報が不明であることを示している場合、統合された回答を "unknown" に設定してください。
        4. **理由:** "reason" フィールドに統合された回答をどのように決定したかの簡単な説明を記載してください。
        5. JSON解析を壊す可能性のあるカンマや特殊文字を使用しないでください。
        6. 必要な回答テキストのみを簡潔に出力し、明確で最小限にしてください。

        出力は以下のJSON構造に厳密に従い、追加のフィールドやテキストを含めないでください：
        {
            "query": "元のユーザーの質問",
            "res_index": "インデックスベースの検索からの回答",
            "certainty_index": <float>,
            "res_simple": "シンプルな検索からの回答",
            "certainty_simple": <float>,
            "output": "統合された回答",
            "reason": "統合された回答を決定した理由"
        }

        出力全体を日本語で記述し、余分なテキストを含めないでください。
        """
    )
    prompt = textwrap.dedent(
        f"""
        ### result 1 (the answer based on context retrieved with index): {dict_to_yaml(res_index.model_dump())}
        ### result 2 (the answer based on context): {dict_to_yaml(res_simple.model_dump())}

        ### context for reference: {vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)}
        """
    )
    chat_model = create_chat_model().with_structured_output(AnswerQueryOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{role}"),
            ("user", "context: {context}"),
        ]
    )
    payload = {"role": role, "context": prompt}
    logger.info(f"[merge_results] {dict_to_yaml(prompt_template.invoke(payload).model_dump())}")
    chain = prompt_template | chat_model
    return chain.invoke(payload)
