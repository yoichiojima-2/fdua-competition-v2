import textwrap

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.utils import before_sleep_hook, dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore

MAX_RETRIEVES = 16


class RefineQueryOutput(BaseModel):
    input: str = Field(description="The original query text.")
    output: str = Field(description="The improved query text.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def refine_query(query: str, vectorstore: FduaVectorStore) -> RefineQueryOutput:
    logger.info(f"[refine_query] refining query..: {query}")
    role = textwrap.dedent(
        """
        You are an intelligent assistant specialized in query refinement.
        Your task is to analyze the provided query along with the context from the retrieved documents.
        Improve the query by making it more specific, clear, and relevant while preserving the original intent.
        
        ## Instructions:
        - Review the original query and the context provided.
        - Enhance the query by incorporating pertinent keywords and ensuring clarity.
        - Do not alter the core meaning of the query or introduce extraneous information.
        - Return only the improved query text.
        """
    )

    context = "\n---\n".join(
        [doc.page_content for doc in vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVES}).invoke(query)]
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("user", "original query: {query}\n\ncontext: {context}")]
    )

    chat_model = create_chat_model().with_structured_output(RefineQueryOutput)
    chain = prompt_template | chat_model

    res = chain.invoke({"query": query, "context": context})

    logger.info(f"[refine_query] done: {dict_to_yaml(res.model_dump())}")

    return res
