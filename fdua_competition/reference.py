import textwrap
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from fdua_competition.index_documents import read_document_index
from fdua_competition.index_pages import read_page_index
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.utils import dict_to_yaml


class ReferenceDocOutput(BaseModel):
    query: str = Field(..., title="The user query.")
    source: str = Field(..., title="The file path of the document.")
    reason: str = Field(..., title="The reason for selecting this source.")


def search_reference_doc(query: str) -> ReferenceDocOutput:
    role = textwrap.dedent(
        """
        You are an intelligent assistant specializing in information retrieval. Your task is to analyze a user query and determine which sources from the provided dataset should be referenced to answer it.

        ## Instructions:
        - Consider the relevance of each source based on the **organizations mentioned** and the **context of the query**.
        - If multiple sources are relevant, include all of them.
        - Do **not** select sources that are unrelated to the query.
        - If **no relevant sources** are found, respond with an empty list (`[]`).

        ## Input:
        - **user_query**: The question or request made by the user.
        - **organization_index**: A list of dictionaries containing:
            - `"organizations"`: The list of organization names found in the document.
            - `"source"`: The file path of the document.

        ## Output:
        Return a list of `"source"` paths that are relevant to the query.

        Ensure that the selected sources directly relate to the user's request.
        """
    )

    chat_model = create_chat_model().with_structured_output(ReferenceDocOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", f"index: {read_document_index()}"), ("user", f"query: {query}")]
    )
    chain = prompt_template | chat_model
    res = chain.invoke({"query": query})

    logger.info(f"[search_reference_doc]\n{dict_to_yaml(res.model_dump())}\n")
    return res


class ReferencePageOutput(BaseModel):
    query: str = Field(..., title="The user query.")
    source: str = Field(..., title="The file path of the document.")
    pages: list[int] = Field(..., title="The list of page numbers to refer.")


def search_reference_pages(query: str, source: Path) -> ReferencePageOutput:
    role = textwrap.dedent(
        """
        You are an intelligent assistant specializing in document retrieval. Your task is to analyze a user query and determine the pages from relevant documents that are most likely to contain the answer.

        ## Instructions:
        - Consider the **context of the query** and carefully match it with the **content of the indexed documents**.
        - Identify and select only those pages that are most likely to provide a direct answer to the query, rather than listing every relevant page.
        - If multiple pages from the same document are possible candidates, prioritize the ones that best address the query.
        - If **no pages** are likely to contain the answer, return an empty list (`[]`).

        ## Input:
        - **user_query**: The question or request made by the user.
        - **page_index**: A list of dictionaries containing:
            - `"source"`: The file path of the document.
            - `"page"`: The page number.
            - `"content"`: The text content of the page.

        ## Output:
        Return the `"source"` file path and a list of page numbers that are most likely to contain the answer to the user's query.

        Ensure that the selected pages best provide the answer or critical information relevant to the query.
        """
    )

    chat_model = create_chat_model().with_structured_output(ReferencePageOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", f"index: {read_page_index(source)}"), ("user", f"query: {query}")]
    )
    chain = prompt_template | chat_model
    res = chain.invoke({"query": query})

    logger.info(f"[search_reference_pages]\n{dict_to_yaml(res.model_dump())}\n")
    return res
