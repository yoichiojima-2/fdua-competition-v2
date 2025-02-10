import textwrap

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from fdua_competition.index_documents import read_document_index
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model


class ReferenceOutput(BaseModel):
    query: str = Field(..., title="The user query.")
    source: str = Field(..., title="The file path of the document.")
    reason: str = Field(..., title="The reason for selecting this source.")


def search_source_to_refer(query: str) -> ReferenceOutput:
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

    chat_model = create_chat_model().with_structured_output(ReferenceOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", f"index: {read_document_index()}"), ("user", f"query: {query}")]
    )
    chain = prompt_template | chat_model
    reference = chain.invoke({"query": query})

    logger.info(f"[search_source_to_refer]\nquery: {query}\nreference: {reference.source}\n\n")
    return reference
