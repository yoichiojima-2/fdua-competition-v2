import re
import textwrap

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.tools import divide_number, round_number
from fdua_competition.utils import before_sleep_hook, dict_to_yaml


# todo: move these cleansers
class CleansePDF(BaseModel):
    input: str = Field(description="The raw context data extracted from a PDF.")
    output: str = Field(description="The cleansed 'response' string that satisfies the requirements.")


def split_document(doc: Document) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "．", "？", "！", "「", "」", "【", "】"],
    )
    split_doc = splitter.split_text(doc.page_content)
    return [Document(page_content=d, metadata=doc.metadata) for d in split_doc]


def remove_special_characters(doc: Document) -> Document:
    # remove control characters
    pattern = r"[\x00-\x08\x0B-\x0C\x0E-\x1F]"
    return Document(page_content=re.sub(pattern, "", doc.page_content), metadata=doc.metadata)


def cleanse_pdf(doc: Document) -> Document:
    role = textwrap.dedent(
        """
        あなたはテキストの精緻化に特化したアシスタントです。
        提供された入力はPDFから解析された生データであり、乱雑で不要なアーティファクトを含んでいる可能性があります。
        あなたの任務は、この生のコンテキストを最小限の修正でクリーンアップし、重要な情報を失わないようにすることです。

        ## 指示:
        - 余分な空白、句読点の誤り、不要なアーティファクトなどの小さなフォーマットの問題を修正してください。ただし、重要な内容を削除しないでください。
        - 言い換えや新しい情報の追加はしないでください。
        - テキストをクリーンアップしながら、すべての重要な詳細を保持してください。
        - 最終出力は簡潔でなければなりません。
        - JSON解析を壊す可能性のあるカンマや特殊文字を使用しないでください。

        ## 入力:
        - **context**: PDFから抽出された生のコンテキストデータ。

        ## 出力:
        最小限の修正でクリーンアップされたコンテキストテキストを返し、元の情報をすべて保持してください。
        """
    )
    chat_model = create_chat_model().bind_tools([divide_number, round_number]).with_structured_output(CleansePDF)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("user", "input: {input}")])
    chain = prompt_template | chat_model

    docs = split_document(doc)
    cleansed_text = "".join([chain.invoke({"input": remove_special_characters(doc)}).output for doc in docs])
    res = CleansePDF(input=doc.page_content, output=cleansed_text)
    logger.info(f"[cleanse_pdf] done\n{dict_to_yaml(res.model_dump())}\n")

    # build Document object
    return Document(page_content=res.output, metadata=doc.metadata)


class CleanseContext(BaseModel):
    query: str = Field(description="The query string that was used to generate the answer.")
    input: str = Field(description="The raw answer output provided in the 'response' field.")
    output: str = Field(description="The cleansed 'response' string that satisfies the requirements.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def cleanse_context(context: AnswerQueryOutput) -> CleanseContext:
    role = textwrap.dedent(
        """
        あなたはテキストの精緻化に特化したアシスタントです。
        提供された入力はPDFから解析された生データであり、乱雑で不要なアーティファクトを含んでいる可能性があります。
        あなたの任務は、この生のコンテキストを最小限の修正でクリーンアップし、重要な情報を失わないようにすることです。

        ## 指示:
        - 余分な空白、句読点の誤り、不要なアーティファクトなどの小さなフォーマットの問題を修正してください。ただし、重要な内容を削除しないでください。
        - 言い換えや新しい情報の追加はしないでください。
        - テキストをクリーンアップしながら、すべての重要な詳細を保持してください。
        - 最終出力は簡潔でなければなりません。
        - JSON解析を壊す可能性のあるカンマや特殊文字を使用しないでください。

        ## 入力:
        - **context**: PDFから抽出された生のコンテキストデータ。

        ## 出力:
        最小限の修正でクリーンアップされたコンテキストテキストを返し、元の情報をすべて保持してください。
        """
    )
    chat_model = create_chat_model().bind_tools([divide_number, round_number]).with_structured_output(CleanseContext)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("user", "input: {input}")])
    chain = prompt_template | chat_model
    res = chain.invoke({"input": dict_to_yaml(context.model_dump())})

    logger.info(f"[cleanse_context]\n{dict_to_yaml(res.model_dump())}\n")
    return res


class CleanseResponseOutput(BaseModel):
    query: str = Field(description="The query string that was used to generate the answer.")
    input: str = Field(description="The raw answer output provided in the 'response' field.")
    output: str = Field(description="The cleansed 'response' string that satisfies the requirements.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def cleanse_response(answer: AnswerQueryOutput) -> CleanseResponseOutput:
    role = textwrap.dedent(
        """
        あなたはテキストの精緻化に特化したアシスタントです。
        提供された入力はPDFから解析された生データであり、乱雑で不要なアーティファクトを含んでいる可能性があります。
        あなたの任務は、この生のコンテキストを最小限の修正でクリーンアップし、重要な情報を失わないようにすることです。
        また、特定の回答を求める直接的な質問があった場合、その質問のカテゴリに対応するデフォルトの回答を返してください。

        ## 指示:
        - 数量で答える問題の回答には、質問に記載の単位を使うこと.
        - 参照元に答えの手がかりが見つからないと判断される場合はその旨を「分かりません」と答えること.
        - queryに過不足なく回答すること.
        - outputの文字数は日本語だと17~25文字が上限です.
        - 小数点第2位を四捨五入はround(n, 1)と同義です. 同様に、小数点第3位を四捨五入はround(n, 2) ex: 1.2345 -> 1.23
        - 敬語は不要. 端的に回答すること.
        - 余分な空白、句読点の誤り、不要なアーティファクトなどの小さなフォーマットの問題を修正してください。ただし、重要な内容を削除しないでください。
        - 言い換えや新しい情報の追加はしないでください。
        - テキストをクリーンアップしながら、すべての重要な詳細を保持してください。
        - 最終出力は簡潔でなければなりません。
        - JSON解析を壊す可能性のあるカンマや特殊文字を使用しないでください。

        ## 例:
        - xxxは何年か -> good: xxxx年  /  bad: xxxはxxxx年です
        - xxxはaとbどちらか -> good: a  /  bad: xxxはaです
        - aとbのどちらがxxか -> good: a  /  bad : xxxなのはaです
        - 何%か -> response: good: 10%  /  bad: 10  # 単位をつける

        ## 入力:
        - **context**: PDFから抽出された生のコンテキストデータ。

        ## 出力:
        最小限の修正でクリーンアップされたコンテキストテキストを返し、元の情報をすべて保持してください。
        """
    )

    chat_model = create_chat_model().bind_tools([divide_number, round_number]).with_structured_output(CleanseResponseOutput)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("user", "answer: {answer}")])
    payload = {"answer": dict_to_yaml(answer.model_dump())}

    logger.info(f"[cleanse_response]\n{dict_to_yaml(prompt_template.invoke(payload).model_dump())}\n")
    chain = prompt_template | chat_model
    res = chain.invoke(payload)
    logger.info(f"[cleanse_answer_query]\n{dict_to_yaml(res.model_dump())}\n")

    return res
