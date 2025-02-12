import textwrap
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.enums import Mode
from fdua_competition.index_documents import read_document_index
from fdua_competition.index_pages import read_page_index
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.utils import before_sleep_hook, dict_to_yaml


class ReferenceDocOutput(BaseModel):
    query: str = Field(..., title="The user query.")
    source: str = Field(..., title="The file path of the document.")
    reason: str = Field(..., title="The reason for selecting this source.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def search_reference_doc(query: str, mode: Mode) -> ReferenceDocOutput:
    role = textwrap.dedent(
        """
        あなたは情報検索に特化した賢いアシスタントです。ユーザークエリを分析し、提供されたデータセットからどのソースを参照すべきかを判断することが任務です。

        ## 指示:
        - クエリの文脈と関連性に基づいて各ソースの関連性を考慮してください。
        - 複数のソースが関連している場合は、すべて含めてください。
        - クエリに関連しないソースは選択しないでください。
        - 関連するソースが見つからない場合は、空のリスト (`[]`) を返してください。

        ## 入力:
        - **user_query**: ユーザーが行った質問やリクエスト。
        - **organization_index**: ドキュメントに含まれる組織名のリストとソースのファイルパスを含む辞書のリスト。

        ## 出力:
        クエリに関連するソースのパスのリストを返してください。

        ユーザーのリクエストに直接関連するソースを選択してください。
        """
    )

    chat_model = create_chat_model().with_structured_output(ReferenceDocOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", f"index: {read_document_index(mode)}"), ("user", f"query: {query}")]
    )
    chain = prompt_template | chat_model
    res = chain.invoke({"query": query})

    logger.info(f"[search_reference_doc]\n{dict_to_yaml(res.model_dump())}\n")
    return res


class ReferencePageOutput(BaseModel):
    query: str = Field(description="The user query.")
    source: str = Field(description="The file path of the document.")
    pages: list[int] = Field(description="The list of page numbers to refer.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def search_reference_pages(query: str, source: Path, mode: Mode) -> ReferencePageOutput:
    role = textwrap.dedent(
        """
        あなたはドキュメント検索に特化した賢いアシスタントです。ユーザークエリを分析し、関連するドキュメントのページを特定することが任務です。

        ## 指示:
        - クエリの文脈を慎重に考慮し、インデックスされたドキュメントの内容と一致させてください。
        - クエリに関連すると思われるページをすべて含めてください。ただし、20ページ以内に収めてください。
        - ページが有用なコンテキストや部分的な情報を提供する場合、それを選択してください。
        - クエリに関連するページが見つからない場合は、空のリスト (`[]`) を返してください。

        ## 入力:
        - **user_query**: ユーザーが行った質問やリクエスト。
        - **page_index**: ドキュメントのファイルパス、ページ番号、ページのテキスト内容を含む辞書のリスト。

        ## 出力:
        クエリに関連するページ番号のリストとソースのファイルパスを返してください。

        クエリに関連するページを選択し、回答に役立つコンテキストを提供してください。
        """
    )

    chat_model = create_chat_model().with_structured_output(ReferencePageOutput)
    page_index = read_page_index(source, mode)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", f"index: {page_index}"), ("user", "query: {query}")]
    )
    chain = prompt_template | chat_model
    payload = {"query": query}
    logger.debug(prompt_template.invoke(payload))

    res = chain.invoke(payload)

    logger.info(f"[search_reference_pages]\n{dict_to_yaml(res.model_dump())}\n")
    return res
