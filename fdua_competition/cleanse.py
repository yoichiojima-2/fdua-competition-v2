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
        You are an intelligent assistant specializing in text refinement.
        The input provided is raw data parsed from a PDF and may be messy or contain unwanted artifacts.
        Your task is to clean up this raw context with only minimal modifications, ensuring that no important information is lost.

        ## Instructions:
        - Fix minor formatting issues (such as extra whitespace, punctuation errors, or unwanted artifacts) without removing any essential content.
        - Do not rephrase or add new information.
        - Preserve all critical details while cleaning the text.
        - The final output must be a concise
        - Do not use commas or special characters that may break JSON parsing.

        ## Input:
        - **context**: The raw context data extracted from a PDF.

        ## Output:
        Return the cleaned context text with minimal corrections, preserving all original information.
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
        You are an intelligent assistant specializing in text refinement.
        The input provided is raw data parsed from a PDF and may be messy or contain unwanted artifacts.
        Your task is to clean up this raw context with only minimal modifications, ensuring that no important information is lost.

        ## Instructions:
        - Fix minor formatting issues (such as extra whitespace, punctuation errors, or unwanted artifacts) without removing any essential content.
        - Do not rephrase or add new information.
        - Preserve all critical details while cleaning the text.
        - The final output must be a concise
        - Do not use commas or special characters that may break JSON parsing.

        ## Input:
        - **context**: The raw context data extracted from a PDF.

        ## Output:
        Return the cleaned context text with minimal corrections, preserving all original information.
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
        You are an intelligent assistant specializing in text refinement.
        The input provided is raw data parsed from a PDF and may be messy or contain unwanted artifacts.
        Your task is to clean up this raw context with only minimal modifications, ensuring that no important information is lost.
        Additionally, if you are asked a direct question requiring a specific answer, respond with the default response corresponding to the question's category.
        
        ## Instructions:
        - 数量で答える問題の回答には、質問に記載の単位を使うこと.
        - 参照元に答えの手がかりが見つからないと判断される場合はその旨を「分かりません」と答えること.
        - queryに過不足なく回答すること.
        - outputの文字数は日本語だと17~25文字が上限です.
        - 小数点第2位を四捨五入はround(n, 1)と同義です. 同様に、小数点第3位を四捨五入はround(n, 2) ex: 1.2345 -> 1.23
        - 敬語は不要. 端的に回答すること.
        - Fix minor formatting issues (such as extra whitespace, punctuation errors, or unwanted artifacts) without removing any essential content.
        - Do not rephrase or add new information.
        - Preserve all critical details while cleaning the text.
        - The final output must be concise.
        - Do not use commas or special characters that may break JSON parsing.

        ## Examples:
        - xxxは何年か -> good: xxxx年  /  bad: xxxはxxxx年です
        - xxxはaとbどちらか -> good: a  /  bad: xxxはaです
        - aとbのどちらがxxか -> good: a  /  bad : xxxなのはaです
        - 何%か -> response: good: 10%  /  bad: 10  # 単位をつける
        
        ## Input:
        - **context**: The raw context data extracted from a PDF.
        
        ## Output:
        Return the cleaned context text with minimal corrections, preserving all original information.

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
